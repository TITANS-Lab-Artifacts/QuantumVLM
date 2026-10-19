"""Explicit, audited source migration; dry-run by default. No checkpoint edits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from utils import source_hashes, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='runs');p.add_argument('--apply',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parent
    old=json.loads((root/'legacy_source_hashes.json').read_text());new=source_hashes()
    plans=[];locks=[]
    try:
        for path in sorted(Path(a.runs).rglob('registry.json')):
            record=json.loads(path.read_text());sig=record.get('signature',{})
            if sig.get('source_hashes')==new: print('Already current:',path);continue
            if sig.get('source_hashes')!=old:
                raise ValueError('Unknown source version; no migration performed. Preserve and inspect '+str(path))
            e=sig['config']['experiment'];complete=(path.parent/'completion.json').exists()
            if not complete and not (e['family']!='classical' and e['mode']=='quantum_only'):
                raise ValueError('Incomplete non-quantum_only run requires separate review: '+str(path))
            lock=(path.parent/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
            if path.with_name('registry.before_prefix_cache.json').exists():
                raise ValueError('Existing migration backup; inspect before continuing: '+str(path))
            state=path.parent/'last/training_state.pt'
            if not state.exists() and not (path.parent/'last').exists(): state=path.parent/'last.previous/training_state.pt'
            if state.exists():
                check=state.with_name('training_state_hash.json')
                if not check.exists() or json.loads(check.read_text())['sha256']!=sha256(state):
                    raise ValueError('Invalid training checkpoint hash: '+str(state))
            print(('Preserve completed' if complete else 'Resume saved checkpoint' if state.exists() else 'No saved optimizer progress'),path.parent)
            plans.append((path,record))
        if not a.apply:
            print('Dry run: %d registries eligible. Use --apply after backup and stopping all writers.'%len(plans));return
        for path,record in plans:
            backup=path.with_name('registry.before_prefix_cache.json')
            if backup.exists(): raise ValueError('Existing migration backup: '+str(backup))
            backup.write_bytes(path.read_bytes())
            record['signature']['source_hashes']=new
            record['prefix_cache_migration']={'time_unix':time.time(),'previous_registry_sha256':sha256(backup),
                'reason':'Frozen-prefix cache implementation; configs/checkpoints untouched; runtime parity checks required',
                'previous_sources':old,'new_sources':new}
            tmp=path.with_suffix('.migration.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,path)
        print('Migrated %d registries. Existing checkpoints and completed results were not modified.'%len(plans))
    finally:
        for lock in locks: lock.close()

if __name__=='__main__': main()
