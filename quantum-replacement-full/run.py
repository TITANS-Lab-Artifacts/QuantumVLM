"""Train all 16 replacements, resume, evaluate saved models, or summarize."""
import bootstrap
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from config import write_json
from data import VSRDataset
from experiment_config import digest
from replacement_model import FAMILIES, POSITIONS, load_model
from replacement_metrics import evaluate
from resource_monitor import Monitor
from settings import load, run_dir, source_hashes
from trainer import train, dataset_signature
from utils import seed_everything, sha256, environment


def summarize(root):
    import csv
    rows = []
    for f in sorted(Path(root).rglob('completion.json')):
        completion = json.loads(f.read_text())
        registry = json.loads((f.parent / 'registry.json').read_text())
        e = registry['signature']['config']['experiment']
        architecture = json.loads((f.parent / 'replacement_architecture.json').read_text())
        row = {'run': str(f.parent), 'family': e['family'], 'position': e['placement'], 'seed': e['seed'],
               **completion['metrics'], **{k: architecture[k] for k in ['parameters_before','parameters_removed','replacement_parameters','parameters_after']},
               'training_seconds': completion.get('training_seconds'), 'reload_verified': completion.get('reload_verified')}
        resources = [json.loads(p.read_text()) for p in f.parent.glob('resources_*.json')]
        row['wall_seconds_all_sessions'] = sum(r['wall_seconds'] for r in resources)
        row['peak_ram_bytes'] = max((r['peak_ram_rss_process_tree_bytes'] or 0 for r in resources), default=None)
        row['peak_cpu_percent_sampled'] = max((r['peak_cpu_process_percent_sampled'] or 0 for r in resources), default=None)
        row['peak_gpu_allocated_bytes'] = max((g['allocated_bytes'] for r in resources for g in r['cuda_allocator_peaks']), default=None)
        rows.append(row)
    dest = Path(root) / 'summary'
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / 'results.json', rows)
    if rows:
        with (dest / 'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# Full-backbone replacement results', '',
             '| Circuit | Position | Seed | Accuracy | Balanced accuracy | Parameters removed | Parameters added | Reload |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        b = r['balanced_accuracy']
        lines.append('| %s | %s | %d | %.2f%% | %s | %d | %d | %s |' % (
            r['family'],r['position'],r['seed'],100*r['accuracy'],
            '%.2f%%' % (100*b) if b is not None else 'NA',r['parameters_removed'],r['replacement_parameters'],r['reload_verified']))
    (dest / 'results.md').write_text('\n'.join(lines) + '\n')
    print('Summary: ' + str(dest))


def evaluate_saved(c, checkpoint):
    checkpoint = Path(checkpoint) if checkpoint else run_dir(c) / 'best'
    ds = VSRDataset(c['data']['root'],c['data']['track'],c['evaluation']['split'],c['data']['eval_limit'])
    signature = {'checkpoint': sha256(checkpoint/'model.pt'), 'metadata':sha256(checkpoint/'replacement.json'),
                 'data': dataset_signature(ds),'config':c,'sources':source_hashes(),'environment':environment()}
    out = run_dir(c) / 'evaluation' / (c['evaluation']['split'] + '_' + digest(signature)[:12])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'clean_results.json').exists():
        print('Completed matching evaluation reused: '+str(out)); return
    write_json(out/'registry.json',signature)
    seed_everything(c['experiment']['seed'])
    try:
        with Monitor(c['monitor']['interval_seconds']) as monitor:
            m = load_model(checkpoint,c['model']['device'])
            result = evaluate(m,ds,draws=c['evaluation']['bootstrap_draws'],seed=c['experiment']['seed'])
            from trainer import atomic_json
            atomic_json(out/'clean_results.json',result)
    finally:
        if 'monitor' in locals() and hasattr(monitor,'result'):
            write_json(out/('resources_%d.json'%time.time_ns()),monitor.result)
    print(result['metrics'])


def main():
    # Relative paths are always relative to this subfolder, even when launched from its parent.
    os.chdir(bootstrap.HERE)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','summarize'])
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--family',choices=FAMILIES)
    p.add_argument('--position',choices=list(POSITIONS))
    p.add_argument('--seed',type=int)
    p.add_argument('--all',action='store_true',help='Run the 16 quantum replacements sequentially; never trains original baseline')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--set',action='append',default=[])
    a = p.parse_args()
    overrides = list(a.set)
    for key,value in [('family',a.family),('placement',a.position),('seed',a.seed)]:
        if value is not None: overrides.append('experiment.%s=%s'%(key,value))
    c = load(a.config,overrides)
    if a.action=='summarize': summarize(c['output']['root']); return
    if a.all:
        if a.checkpoint or c['output']['run_name'] or a.family or a.position:
            p.error('--all cannot be combined with a single checkpoint, run name, family, or position')
        for family in FAMILIES:
            for site in POSITIONS:
                command=[sys.executable,'-u',str(bootstrap.HERE/'run.py'),a.action,'--config',a.config,'--family',family,'--position',site]
                for value in overrides: command+=['--set',value]
                if a.resume: command+=['--resume']
                subprocess.run(command,check=True)
        summarize(c['output']['root']); return
    if a.action=='train':
        if a.checkpoint: p.error('Use --resume for training. --checkpoint is evaluation-only.')
        existing=(run_dir(c)/'registry.json').exists()
        train(c,resume=a.resume and existing)
    else: evaluate_saved(c,a.checkpoint)

if __name__=='__main__': main()
