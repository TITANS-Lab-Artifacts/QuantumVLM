"""Sequential real experiments; complete compatible runs are reused."""
import argparse
import subprocess
import sys
from experiment_config import FAMILIES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','diagnose','attack'])
    p.add_argument('--config',default='configs/shared.yaml')
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=['classical'])
    p.add_argument('--placements',nargs='+',choices=['q1','q2','q3','q4'],default=['q1','q2','q3','q4'])
    p.add_argument('--modes',nargs='+');p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    for family in a.families:
        modes=a.modes or (['zero_shot','full_frozen','frozen_head','projector','lora','full_unfrozen'] if family=='classical' else ['quantum_only','quantum_head','quantum_projector','quantum_lora','quantum_full'])
        for site in (['q4'] if family=='classical' else a.placements):
            for mode in modes:
                for seed in a.seeds:
                    command=[sys.executable,'-u','experiment.py',a.action,'--config',a.config]
                    overrides=['experiment.family='+family,'experiment.mode='+mode,'experiment.placement='+site,'experiment.seed='+str(seed)]
                    # Avoid using a model as its own transfer surrogate.
                    if family=='classical' and mode=='frozen_head' and a.action=='attack':
                        overrides.append('attacks.surrogate_checkpoint=runs/classical/lora/seed0/best')
                    for value in overrides+a.set: command+=['--set',value]
                    if a.resume:
                        from experiment_config import load,run_dir
                        cfg=load(a.config,overrides+a.set)
                        if (run_dir(cfg)/'registry.json').exists(): command+=['--resume']
                    print('\nRunning '+family+' '+site+' '+mode+' seed '+str(seed),flush=True)
                    subprocess.run(command,check=True)

if __name__=='__main__': main()
