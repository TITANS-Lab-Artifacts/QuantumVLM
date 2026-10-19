"""Collect clean, attack and resource results without model execution."""
import csv
import json
from pathlib import Path
from config import write_json


def summarize(root):
    root=Path(root);clean=[];attack=[]
    for file in sorted(root.rglob('completion.json')):
        run=file.parent;reg=run/'registry.json'
        if not reg.exists(): continue
        c=json.loads(reg.read_text())['signature']['config'];r=json.loads(file.read_text());e=c['experiment']
        resources=[json.loads(f.read_text()) for f in run.glob('resources_*.json')]
        row={**e,**r.get('metrics',{}),'run':str(run),'training_seconds':r.get('training_seconds'),
             'validation_seconds':r.get('validation_seconds'),'cache_prepare_seconds':r.get('cache_prepare_seconds'),
             'epochs_completed':r.get('epochs_completed'),'reload_verified':r.get('reload_verified'),
             'wall_seconds_all_sessions':sum(s['wall_seconds'] for s in resources),
             'peak_ram_bytes':max((s['peak_ram_rss_process_tree_bytes'] or 0 for s in resources),default=None),
             'peak_cpu_percent_sampled':max((s['peak_cpu_process_percent_sampled'] or 0 for s in resources),default=None),
             'peak_gpu_allocated_bytes':max((d['allocated_bytes'] for s in resources for d in s['cuda_allocator_peaks']),default=None)}
        params=run/'parameters.json'
        if params.exists(): row.update(json.loads(params.read_text())['counts'])
        clean.append(row)
    for file in sorted(root.rglob('attack/*/summary.json')):
        r=json.loads(file.read_text());attack.append({'run':str(file.parent),'objective':r.get('objective'),
            'gradient_validation':r.get('gradient_validation'),**r['metrics'],'targeted':r.get('targeted')})
    dest=root/'summaries';dest.mkdir(exist_ok=True)
    for name,rows in [('clean',clean),('attacks',attack)]:
        write_json(dest/(name+'.json'),rows)
        if rows:
            fields=sorted(set().union(*(r.keys() for r in rows)))
            with (dest/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['# Experiment summary','','Only completed runs are listed. Accuracy values are proportions in CSV/JSON.','',
           '| Family | Placement | Mode | Seed | Accuracy | Balanced accuracy | Reload verified |',
           '|---|---|---|---:|---:|---:|---|']
    for r in clean:
        b=r.get('balanced_accuracy');lines.append('| %s | %s | %s | %s | %.2f%% | %s | %s |'%(r['family'],r['placement'],r['mode'],r['seed'],100*r['accuracy'],'%.2f%%'%(100*b) if b is not None else 'NA',r.get('reload_verified')))
    (dest/'summary.md').write_text('\n'.join(lines)+'\n');print('Summaries: '+str(dest))
