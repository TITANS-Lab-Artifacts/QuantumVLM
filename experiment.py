"""Single entry point: train, evaluate, diagnose, attack, and summarize."""
import argparse
import json
from pathlib import Path
import sys
import torch
from config import write_json
from experiment_config import load, run_dir, digest
from experiment_model import load_model, provenance
from experiment_train import train, dataset_signature, atomic_json
from experiment_metrics import evaluate
from resource_monitor import Monitor
from utils import seed_everything, environment, source_hashes
from data import VSRDataset


def action(c,kind,checkpoint=None,resume=False):
    out=run_dir(c);checkpoint=Path(checkpoint) if checkpoint else out/'best'
    if not checkpoint.exists(): raise FileNotFoundError('Checkpoint missing: '+str(checkpoint))
    split=c['evaluation']['split'];limit=c['attacks']['limit'] if kind=='attack' else c['data']['eval_limit']
    if kind=='diagnose': split='dev'
    dataset=VSRDataset(c['data']['root'],c['data']['track'],split,limit)
    signature={'config':c,'checkpoint':provenance(checkpoint),'data':dataset_signature(dataset),
               'environment':environment(),'sources':source_hashes(),'action':kind}
    if kind=='attack' and 'transfer' in c['attacks']['names']:
        signature['surrogate']=provenance(c['attacks']['surrogate_checkpoint'])
    folder=out/kind/(split+'_'+digest(signature)[:12]);folder.mkdir(parents=True,exist_ok=True)
    print('Action output: '+str(folder),flush=True)
    if (folder/'completed.json').exists(): print('Completed compatible action reused');return
    import fcntl
    lock=(folder/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(folder/'action_registry.json',signature)
    seed_everything(c['experiment']['seed'])
    import time
    session=time.time_ns()
    try:
        with Monitor(c['monitor']['interval_seconds']) as monitor:
            if kind=='attack':
                attack(c,checkpoint,folder)
            else:
                m=load_model(checkpoint,c['model']['device'],'float32' if kind=='diagnose' else None)
                if kind=='evaluate':
                    result=evaluate(m,dataset,draws=c['evaluation']['bootstrap_draws'],seed=c['experiment']['seed'])
                    write_json(folder/'clean_results.json',result)
                else:
                    from gradient_check import check_gradients
                    reports=[]
                    for i in range(min(c['diagnostics']['examples'],len(dataset))):
                        row=dataset[i];ctx=m.context(row['pixels'],row['caption'])
                        report=check_gradients(lambda x:m(x,ctx),row['pixels'].unsqueeze(0).to(m.device_),
                             steps=c['diagnostics']['steps'],directions=c['diagnostics']['directions'],seed=c['experiment']['seed']+i)
                        reports.append({'id':row['id'],**report});print(report['status'],flush=True)
                    result={'status':'passed' if reports and all(r['status']=='passed' for r in reports) else 'needs_investigation',
                            'examples':reports,'precision':'float32','checkpoint':provenance(checkpoint),
                            'source_hashes':source_hashes(),'environment':environment()}
                    write_json(folder/'gradient_report.json',result)
                    if result['status']!='passed':
                        print('Gradient diagnostic is inconclusive/failed. Inspect report before robustness claims.',flush=True)
            atomic_json(folder/'completed.json',{'action':kind})
    finally:
        if 'monitor' in locals() and hasattr(monitor,'result'): write_json(folder/('resources_%d.json'%session),monitor.result)
        lock.close()


def attack(c,checkpoint,folder):
    import run_attacks
    a=c['attacks'];ev=c['evaluation']
    names=a['names']
    if 'transfer' in names:
        surrogate=Path(a['surrogate_checkpoint'])
        if not surrogate.exists(): raise FileNotFoundError('Train/set the surrogate first: '+str(surrogate))
        if surrogate.resolve()==checkpoint.resolve(): raise ValueError('Select a different surrogate for transfer')
    # Require a diagnostic for this exact checkpoint; do not silently bless old results.
    parent=run_dir(c)/'diagnose'
    reports=[]
    for path in parent.glob('*/gradient_report.json'):
        r=json.loads(path.read_text())
        if r.get('checkpoint')==provenance(checkpoint) and r.get('source_hashes')==source_hashes() and r.get('environment')==environment(): reports.append(r)
    if not reports: raise ValueError('Run experiment.py diagnose for this checkpoint before attacks')
    gradient_status='passed' if any(r['status']=='passed' for r in reports) else 'needs_investigation'
    write_json(folder/'gradient_status.json',{'status':gradient_status,'note':'Float32 diagnostic; attack precision is checkpoint precision. Inconclusive diagnostics prevent validated white-box robustness claims.'})
    # Preserve the existing pixel attack implementations, query accounting, and clean checks.
    argv=['run_attacks.py','--checkpoint',str(checkpoint),'--data-dir',c['data']['root'],
          '--track',c['data']['track'],'--device',c['model']['device'],'--split',ev['split'],
          '--attacks',*names,'--epsilon',str(a['epsilon']),'--steps',str(a['steps']),
          '--restarts',str(a['restarts']),'--query-budget',str(a['query_budget']),
          '--spsa-samples',str(a['spsa_samples']),'--spsa-delta',str(a['spsa_delta']),
          '--hsj-steps',str(a['hsj_steps']),'--seed',str(c['experiment']['seed']),
          '--objective',a['objective'],'--output',str(folder),'--resume']
    reference=checkpoint.parent/'clean_results.json'
    if not reference.exists(): reference=checkpoint.parent/'best_development.json'
    if ev['split']=='dev' and reference.exists(): argv+=['--reference-clean',str(reference)]
    if a['limit'] is not None: argv+=['--limit',str(a['limit'])]
    if a['step_size'] is not None: argv+=['--step-size',str(a['step_size'])]
    if a['save_images']: argv+=['--save-images']
    if ev['allow_test']: argv+=['--allow-test']
    if 'transfer' in names: argv+=['--surrogate-checkpoint',a['surrogate_checkpoint']]
    if a['objective']!='untargeted':
        # Binary targeted results are the corresponding source-class subset of an
        # identical untargeted search. Reuse candidates only after exact registry match.
        import copy
        import shutil
        current=json.loads((folder/'action_registry.json').read_text())
        expected=copy.deepcopy(current);expected['config']['attacks']['objective']='untargeted'
        for registry in (run_dir(c)/'attack').glob('*/action_registry.json'):
            source=registry.parent
            if source==folder or not (source/'completed.json').exists(): continue
            if json.loads(registry.read_text())!=expected: continue
            source_label=0 if a['objective']=='false_acceptance' else 1
            for file in source.glob('example_*.json'):
                row=json.loads(file.read_text())
                if row['label']!=source_label: continue
                destination=folder/file.name
                if not destination.exists(): shutil.copy2(file,destination)
                index=file.stem.split('_')[-1]
                if a['save_images']:
                    for image in (source/'adversarial_tensors').glob(index+'_*.pt'):
                        dest=folder/'adversarial_tensors'/image.name;dest.parent.mkdir(exist_ok=True)
                        if not dest.exists(): shutil.copy2(image,dest)
            write_json(folder/'reused_untargeted.json',{'source':str(source),'reason':'Same binary objective on selected source class; exact action registry match except objective.'})
            break
    old=sys.argv
    try: sys.argv=argv;run_attacks.main()
    finally: sys.argv=old
    summary=json.loads((folder/'summary.json').read_text());summary['gradient_validation']=gradient_status
    write_json(folder/'summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','diagnose','attack','summarize'])
    p.add_argument('--config',default='configs/shared.yaml');p.add_argument('--set',action='append',default=[],metavar='section.key=value')
    p.add_argument('--checkpoint');p.add_argument('--resume',action='store_true')
    args=p.parse_args();c=load(args.config,args.set)
    if args.action=='train':
        if args.checkpoint: p.error('train resumes via --resume; --checkpoint is for evaluation/attacks')
        train(c,args.resume)
    elif args.action=='summarize':
        from summarize_experiments import summarize
        summarize(c['output']['root'])
    else: action(c,args.action,args.checkpoint,args.resume)

if __name__=='__main__': main()
