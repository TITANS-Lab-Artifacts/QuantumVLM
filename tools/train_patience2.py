"""Resume-safe patience=2 launcher. Install in PROJECT/tools, never project root."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def migrate(registry, wanted, expected_sources, expected_environment, apply=False):
    """Only lower an unfinished run's patience; preserve checkpoints and history."""
    if not registry.exists(): return 2
    record=json.loads(registry.read_text());signature=record['signature']
    if signature['source_hashes']!=expected_sources:
        raise ValueError('Source signature differs. Finish the separate code-version migration first: '+str(registry))
    if signature['environment']!=expected_environment:
        raise ValueError('Environment changed: '+str(registry))
    saved=signature['config'];comparison=copy.deepcopy(saved)
    comparison['training']['patience']=2
    if comparison!=wanted:
        raise ValueError('A setting other than patience changed: '+str(registry))
    old=saved['training']['patience']
    if (registry.parent/'completion.json').exists():
        print('Completed run preserved with its original patience=%s: %s'%(old,registry.parent),flush=True)
        return old
    if old==2: return 2
    if old<2: raise ValueError('This tool only makes stopping more aggressive')
    print('Unfinished run: patience %s -> 2; saved stale counter retained: %s'%(old,registry.parent),flush=True)
    if apply:
        import fcntl
        with (registry.parent/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if json.loads(registry.read_text())!=record: raise ValueError('Registry changed during inspection')
            stamp=str(time.time_ns());backup=registry.with_name('registry.before_patience2.'+stamp+'.json')
            backup.write_bytes(registry.read_bytes())
            signature['config']['training']['patience']=2
            record.setdefault('early_stopping_changes',[]).append({'time_unix':time.time(),
                'old_patience':old,'new_patience':2,'monitor':'dev.loss','backup':backup.name,
                'checkpoint_state_changed':False,'completed_results_changed':False})
            tmp=registry.with_name('registry.patience2.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,registry)
    return 2


def main():
    root=Path(__file__).resolve().parents[1]
    if not (root/'experiment.py').exists(): raise SystemExit('Put this script in PROJECT/tools/train_patience2.py')
    sys.path.insert(0,str(root))
    from experiment_config import FAMILIES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replacement',action='store_true')
    p.add_argument('--config');p.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES[1:])
    p.add_argument('--placements',nargs='+');p.add_argument('--modes',nargs='+',default=['quantum_only'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--set',action='append',default=[])
    p.add_argument('--run',action='store_true',help='Apply audited patience changes and run; otherwise preview only')
    a=p.parse_args()
    if a.replacement:
        folder=root/'quantum-replacement-full';sys.path.insert(0,str(folder))
        from settings import load,run_dir,source_hashes
        script='run.py';sites=a.placements or ['r1','r2','r3','r4'];config=a.config or 'config.yaml'
        if 'classical' in a.families: raise ValueError('No classical replacement configuration')
    else:
        folder=root
        from experiment_config import load,run_dir
        from utils import source_hashes
        script='experiment.py';sites=a.placements or ['q1','q2','q3','q4'];config=a.config or 'configs/shared.yaml'
    os.chdir(folder)
    from utils import environment
    sources=source_hashes();env=environment();plans=[]
    for family in a.families:
        modes=['full_unfrozen'] if a.replacement else a.modes
        for site in (['q4'] if family=='classical' else sites):
            for mode in modes:
                for seed in a.seeds:
                    overrides=['experiment.family='+family,'experiment.placement='+site,'experiment.mode='+mode,'experiment.seed='+str(seed)]+a.set+['training.patience=2']
                    c=load(config,overrides);reg=run_dir(c)/'registry.json'
                    effective=migrate(reg,c,sources,env,False)
                    plans.append((c,reg,overrides,effective))
    if not a.run:
        print('Preview only: %d configurations checked. Add --run to apply and resume.'%len(plans));return
    for c,reg,overrides,effective in plans:
        effective=migrate(reg,c,sources,env,True)
        command=[sys.executable,'-u',script,'train','--config',config]
        for setting in overrides+['training.patience='+str(effective)]:command+=['--set',setting]
        if reg.exists():command+=['--resume']
        print('Running',c['experiment'],'patience='+str(effective),flush=True)
        subprocess.run(command,check=True,cwd=folder)

if __name__=='__main__': main()
