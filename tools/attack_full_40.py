"""Run 40-example attacks, preserving compatible per-attack progress (Python 3.9+)."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from contextlib import ExitStack

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()


def compatible(saved, current):
    """Only the cohort limit and the known transfer-cache overlay may differ."""
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    try:
        old_limit = old['config']['attacks']['limit']
        new_limit = new['config']['attacks']['limit']
        if old_limit is not None and old_limit < new_limit:
            return False
        old['config']['attacks']['limit'] = new_limit
        old_data, new_data = old.pop('data'), new.pop('data')
        if old_data['manifest'] != new_data['manifest']:
            return False
        if old_data['ids'][:len(new_data['ids'])] != new_data['ids']:
            return False
        if any(old_data['images'].get(k) != v for k, v in new_data['images'].items()):
            return False
        key = 'local_full_attack_optimization'
        if key not in old['sources']:
            new['sources'].pop(key, None)
        return old == new
    except (KeyError, TypeError):
        return False


def migrate(c, checkpoint, out, experiment):
    from data import VSRDataset
    from experiment_train import dataset_signature
    from experiment_config import digest
    from utils import environment
    import fcntl
    dataset = VSRDataset(c['data']['root'], c['data']['track'],
                         c['evaluation']['split'], c['attacks']['limit'])
    signature = dict(config=c, checkpoint=experiment.provenance(checkpoint),
                     data=dataset_signature(dataset), environment=environment(),
                     sources=experiment.source_hashes(), action='attack')
    if 'transfer' in c['attacks']['names']:
        signature['surrogate'] = experiment.provenance(c['attacks']['surrogate_checkpoint'])
    dest = out / 'attack' / (c['evaluation']['split'] + '_' + digest(signature)[:12])
    if dest.exists():
        return  # Original runner locks and resumes this exact action.
    candidates = []
    for registry in sorted((out / 'attack').glob('*/action_registry.json')):
        if compatible(json.loads(registry.read_text()), signature):
            candidates.append(registry.parent)
    if not candidates:
        if any((out / 'attack').glob('*/example_*.json')):
            raise RuntimeError('Saved attacks exist but no compatible source was found. '
                               'No attacks restarted. Check configuration, checkpoint, data, environment and source versions.')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=dest.name+'.migrating-', dir=str(dest.parent)))
    try:
        merged = {}
        with ExitStack() as stack:
            for source in candidates:
                lock = stack.enter_context((source / '.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for path in sorted(source.glob('example_*.json')):
                    i = int(path.stem.split('_')[1])
                    if i >= len(dataset.rows):
                        continue
                    row = json.loads(path.read_text())
                    if row['id'] != dataset.rows[i]['id']:
                        raise ValueError('Example identity mismatch: '+str(path))
                    if not set(row['attacks']).issubset(c['attacks']['names']):
                        raise ValueError('Unexpected attack method: '+str(path))
                    if i not in merged:
                        merged[i] = copy.deepcopy(row)
                        merged[i]['attacks'] = {}
                    target = merged[i]
                    for field in ['id','label','pixels_sha256','prompt_sha256','clean_prediction','clean_correct']:
                        if target[field] != row[field]:
                            raise ValueError('Conflicting saved example: '+str(path))
                    for method, result in row['attacks'].items():
                        if method in target['attacks'] and target['attacks'][method] != result:
                            raise ValueError('Conflicting saved attack results: '+str(path)+' '+method)
                        target['attacks'][method] = result
                        if c['attacks']['save_images'] and not result.get('status','').startswith('skipped_'):
                            tensor = source/'adversarial_tensors'/('%06d_%s.pt' % (i,method))
                            if not tensor.is_file():
                                raise FileNotFoundError('Missing saved tensor: '+str(tensor))
                            dest_tensor = stage/'adversarial_tensors'/tensor.name
                            dest_tensor.parent.mkdir(exist_ok=True)
                            if dest_tensor.exists() and fast.digest_file(dest_tensor) != fast.digest_file(tensor):
                                raise ValueError('Conflicting saved tensors: '+str(tensor))
                            shutil.copy2(tensor,dest_tensor)
            for i,row in merged.items():
                row['robust_correct'] = row['clean_correct'] and not any(a.get('success',False) for a in row['attacks'].values())
                fast.atomic_json(stage/('example_%06d.json'%i),row)
            fast.atomic_json(stage/'action_registry.json',signature)
            fast.atomic_json(stage/'migration.json',dict(
                sources=[str(p) for p in candidates], records=len(merged),
                attack_records=sum(len(r['attacks']) for r in merged.values()),
                helper_sha256=fast.digest_file(SCRIPT),
                reason='Verified first 40 examples; identical per-example protocol. Original outputs retained.'))
            os.rename(stage,dest)
        print('RESUME: preserved %d examples / %d saved attack entries for the 40-example cohort' %
              (len(merged),sum(len(r['attacks']) for r in merged.values())),flush=True)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def main():
    global fast
    spec = importlib.util.spec_from_file_location('attack_fast',HERE/'attack_full_fast.py')
    fast = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast)
    fast.migrate = migrate
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        fast.optimized_worker(json.load(sys.stdin))
        return
    if any(a=='--limit' or a.startswith('--limit=') for a in sys.argv):
        raise SystemExit('This launcher fixes the limit at 40. Omit --limit.')
    sys.argv += ['--limit','40']
    import subprocess
    original = subprocess.run
    def launch(command,**kwargs):
        if len(command)>3 and command[2]==str(Path(fast.base.__file__).resolve()) and command[3]=='_worker':
            command=list(command);command[2]=str(SCRIPT)
        return original(command,**kwargs)
    fast.base.subprocess.run=launch
    fast.base.main()


if __name__=='__main__':
    main()
