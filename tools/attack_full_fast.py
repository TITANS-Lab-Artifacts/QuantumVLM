"""Exact transfer-candidate reuse; audited migration from attack_full.py results."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_attack_base', HERE / 'attack_full.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SCRIPT = Path(__file__).resolve()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)


def protocol_sources(sources):
    return {k: v for k, v in sources.items() if k != 'local_full_attack_optimization'}


def migrate(c, checkpoint, out, experiment):
    """Copy per-attack records only after all base protocol signatures match."""
    from data import VSRDataset
    from experiment_train import dataset_signature
    from experiment_config import digest
    from utils import environment
    dataset = VSRDataset(c['data']['root'], c['data']['track'],
                         c['evaluation']['split'], c['attacks']['limit'])
    signature = {'config': c, 'checkpoint': experiment.provenance(checkpoint),
                 'data': dataset_signature(dataset), 'environment': environment(),
                 'sources': experiment.source_hashes(), 'action': 'attack'}
    if 'transfer' in c['attacks']['names']:
        signature['surrogate'] = experiment.provenance(c['attacks']['surrogate_checkpoint'])
    dest = out / 'attack' / (c['evaluation']['split'] + '_' + digest(signature)[:12])
    # Existing new-format results are already handled by the original resume code.
    if dest.exists(): return
    expected = copy.deepcopy(signature)
    expected['sources'] = protocol_sources(expected['sources'])
    matches = []
    for registry in (out / 'attack').glob('*/action_registry.json'):
        saved = json.loads(registry.read_text())
        # Only migrate original-launcher results, never arbitrary optimization versions.
        if 'local_full_attack_optimization' in saved.get('sources', {}): continue
        if saved == expected: matches.append(registry.parent)
    if not matches:
        if any((out / 'attack').glob('*/example_*.json')):
            raise RuntimeError('Existing attack records have incompatible configuration, environment, checkpoint, or source signatures. Nothing was migrated; inspect action_registry.json instead of restarting silently.')
        return
    if len(matches) != 1: raise RuntimeError('Ambiguous legacy attack folders; no records copied.')
    source = matches[0]
    import fcntl
    with (source / '.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Old attack still running. Stop it before restarting.')
        staging = dest.with_name(dest.name + '.migrating-%d' % os.getpid())
        staging.mkdir(parents=True, exist_ok=False)
        try:
            count = 0
            hashes = {}
            for record in sorted(source.glob('example_*.json')):
                row = json.loads(record.read_text())
                i = int(record.stem.split('_')[1])
                if i >= len(dataset.rows) or row['id'] != dataset.rows[i]['id']:
                    raise RuntimeError('Legacy example identity mismatch: ' + str(record))
                if not set(row['attacks']).issubset(c['attacks']['names']):
                    raise RuntimeError('Legacy attack methods differ.')
                if c['attacks']['save_images']:
                    for name, result in row['attacks'].items():
                        if result.get('status', '').startswith('skipped_'): continue
                        tensor = source / 'adversarial_tensors' / ('%06d_%s.pt' % (i, name))
                        if not tensor.is_file(): raise RuntimeError('Saved adversarial tensor missing: ' + str(tensor))
                        target = staging / 'adversarial_tensors' / tensor.name
                        target.parent.mkdir(exist_ok=True)
                        shutil.copy2(tensor, target)
                shutil.copy2(record, staging / record.name)
                hashes[record.name] = digest_file(record); count += 1
            atomic_json(staging / 'migration.json', {'source': str(source),
                'source_action_registry_sha256': digest_file(source / 'action_registry.json'),
                'record_hashes': hashes, 'records': count,
                'reason': 'Only exact transfer-cache execution overlay changed; base protocol signatures match.',
                'new_execution_sources': signature['sources']})
            os.rename(staging, dest)
            print('RESUME: preserved %d example records from %s' % (count, source), flush=True)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def install_cache(job, run_attacks):
    import torch
    from attacks.common import AttackResult, validate_candidate
    from utils import environment, source_hashes
    from experiment_model import provenance
    original = run_attacks.transfer_pgd
    cache_root = base.ROOT / 'attack_transfer_cache'
    cache_root.mkdir(exist_ok=True)
    surrogate_info = provenance(job['config']['attacks']['surrogate_checkpoint'])
    env = environment()
    gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    surrogate_sources = source_hashes()
    def tensor_hash(t):
        t = t.detach().cpu().contiguous()
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()
    def cached(predict, original_image, label, **kwargs):
        # The existing transfer lambda binds its fully constructed prompt context.
        defaults = predict.__defaults__ or ()
        ctx = defaults[0] if defaults else None
        if ctx is None or not hasattr(ctx, 'input_ids'):
            print('TRANSFER: context unavailable; performing original search.', flush=True)
            return original(predict, original_image, label, **kwargs)
        signature = {'version': 1, 'surrogate': surrogate_info, 'environment': env,
            'gpu': gpu, 'base_sources': surrogate_sources,
            'optimization_sha256': digest_file(SCRIPT),
            'pixels': tensor_hash(original_image), 'shape': list(original_image.shape),
            'pixel_dtype': str(original_image.dtype), 'label': int(label), 'options': kwargs,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'prompt': ctx.prompt, 'input_ids': tensor_hash(ctx.input_ids),
            'cpu_rng': tensor_hash(torch.get_rng_state()),
            'cuda_rng': [tensor_hash(x) for x in torch.cuda.get_rng_state_all()]}
        # Quantum family/target checkpoint is intentionally absent: transfer construction
        # uses only the shared surrogate. Source files and runtime are still recorded.
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        folder = cache_root / key; folder.mkdir(exist_ok=True)
        import fcntl
        with (folder / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meta = folder / 'metadata.json'; tensor = folder / 'candidate.pt'
            if meta.exists():
                saved = json.loads(meta.read_text())
                if saved['signature'] != signature or digest_file(tensor) != saved['tensor_sha256']:
                    raise RuntimeError('Transfer cache integrity mismatch: ' + str(folder))
                payload = torch.load(tensor, map_location='cpu', weights_only=True)
                image = payload['image'].to(original_image.device)
                torch.set_rng_state(payload['cpu_rng'])
                if payload['cuda_rng']: torch.cuda.set_rng_state_all(payload['cuda_rng'])
                validate_candidate(image, original_image, kwargs['epsilon'])
                details = dict(saved['details'])
                details.update(transfer_cache_hit=True, transfer_cache_key=key,
                               original_surrogate_calls=saved['calls'], surrogate_calls=0,
                               original_search_seconds=saved['search_seconds'])
                print('TRANSFER CACHE HIT: skipped %d surrogate forwards.' % saved['calls'], flush=True)
                return AttackResult(image, 0, saved['status'], details)
            print('TRANSFER CACHE MISS: generating candidate for later targets.', flush=True)
            started = time.monotonic()
            result = original(predict, original_image, label, **kwargs)
            validate_candidate(result.image, original_image, kwargs['epsilon'])
            tmp = folder / ('candidate.%d.tmp' % os.getpid())
            torch.save({'image': result.image.detach().cpu(), 'cpu_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all()}, tmp); os.replace(tmp, tensor)
            atomic_json(meta, {'signature': signature, 'tensor_sha256': digest_file(tensor),
                'calls': result.calls, 'status': result.status, 'details': result.details,
                'search_seconds': time.monotonic() - started})
            result.details.update(transfer_cache_hit=False, transfer_cache_key=key)
            return result
    run_attacks.transfer_pgd = cached
    # Report progress without altering the search, seed, or candidate selection.
    for name in ['fgsm', 'pgd', 'spsa', 'hop_skip_jump']:
        fn = getattr(run_attacks, name)
        def timed(*args, _fn=fn, _name=name, **kwargs):
            print('START', _name, kwargs.get('loss', ''), flush=True)
            started = time.monotonic(); result = _fn(*args, **kwargs)
            print('DONE %s: %.1fs; calls=%d' % (_name, time.monotonic()-started, result.calls), flush=True)
            return result
        setattr(run_attacks, name, timed)


def optimized_worker(job):
    os.chdir(base.ROOT); sys.path.insert(0, str(base.ROOT))
    import experiment, run_attacks
    action = experiment.action
    def optimized_action(c, kind, checkpoint=None, resume=False):
        old_sources = experiment.source_hashes
        def sources():
            result = old_sources()
            result['local_full_attack_optimization'] = digest_file(SCRIPT)
            return result
        experiment.source_hashes = sources; run_attacks.source_hashes = sources
        if kind == 'attack': migrate(c, Path(checkpoint), Path(job['out']), experiment)
        return action(c, kind, checkpoint, resume)
    experiment.action = optimized_action
    install_cache(job, run_attacks)
    base.worker(job)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        optimized_worker(json.load(sys.stdin))
    else:
        # Original main creates subprocesses using its module __file__. Redirect only
        # subprocess entrypoint; base.worker's provenance uses the original file.
        original_file = base.__file__
        import subprocess
        original_run = subprocess.run
        def launch(command, **kwargs):
            if len(command)>3 and command[2]==str(Path(original_file).resolve()) and command[3]=='_worker':
                command = list(command); command[2] = str(SCRIPT)
            return original_run(command, **kwargs)
        base.subprocess.run = launch
        base.main()
