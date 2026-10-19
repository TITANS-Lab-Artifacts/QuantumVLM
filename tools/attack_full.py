"""Preflight and local attacks for fully trained quantum models (Python 3.9+)."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['basic-ring-quantum-4qubit', 'basic-ring-quantum-8qubit',
            'strongly_entangling_4qubits', 'strongly_entangling_8qubits']


def read(path):
    return json.loads(Path(path).read_text())


def absolute(path, base):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def configuration(args):
    sys.path.insert(0, str(ROOT))
    from experiment_config import load
    return load(str(absolute(args.config, ROOT)))


def preflight(args, groups):
    errors, jobs = [], []
    required = ['experiment.py', 'experiment_model.py', 'experiment_train.py',
                'experiment_config.py', 'experiment_metrics.py', 'model.py', 'config.py',
                'utils.py', 'data.py', 'preprocessing.py', 'gradient_check.py',
                'run_attacks.py', 'resource_monitor.py', 'compare_results.py',
                'feature_cache.py', 'quantum_shared.py', 'attacks/__init__.py',
                'attacks/common.py', 'attacks/oracles.py', 'attacks/white_box.py',
                'attacks/score_based.py', 'attacks/decision_based.py', 'attacks/transfer.py']
    if 'replacement' in groups:
        required += ['quantum-replacement-full/' + n for n in
                     ['replacement_model.py', 'bootstrap.py', 'settings.py', 'config.yaml']]
    required += [f + '/circuit.py' for f in FAMILIES]
    for name in required:
        if not (ROOT / name).is_file(): errors.append('Missing source: ' + str(ROOT / name))
    for module in ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'PIL', 'yaml', 'psutil']:
        if importlib.util.find_spec(module) is None: errors.append('Missing Python package: ' + module)
    if errors: return jobs, errors
    shared = configuration(args)
    if 'hsj' in shared['attacks']['names'] and importlib.util.find_spec('foolbox') is None:
        errors.append('Missing Python package: foolbox')
    if shared['evaluation']['split'] == 'test' and not shared['evaluation']['allow_test']:
        errors.append('Test evaluation is not enabled in the configuration.')
    checked_data = set()
    checked_surrogates = set()
    def require(path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append('Missing/empty file: ' + str(path))
    def checkpoint_files(best):
        if (best / 'replacement.json').is_file():
            require(best / 'model.pt')
        elif (best / 'experiment.json').is_file():
            info = read(best / 'experiment.json')
            for n in ['base/checkpoint.json', 'base/weights.pt']: require(best / n)
            if info['config']['experiment']['family'] != 'classical': require(best / 'adapter.pt')
        else:
            for n in ['checkpoint.json', 'weights.pt']: require(best / n)
    for group in groups:
        base = ROOT if group == 'insertion' else ROOT / 'quantum-replacement-full'
        if group == 'insertion':
            runroot = absolute(args.insertion_root or shared['output']['root'], ROOT)
        else:
            import yaml
            rc = yaml.safe_load((base / 'config.yaml').read_text()) or {}
            runroot = absolute(args.replacement_root or rc.get('output', {}).get('root', 'runs'), base)
        for family in FAMILIES:
            for site in (['q1', 'q2', 'q3', 'q4'] if group == 'insertion' else ['r1', 'r2', 'r3', 'r4']):
                for seed in args.seeds:
                    mode = 'quantum_full' if group == 'insertion' else 'full_unfrozen'
                    out = runroot / family / site
                    if group == 'insertion': out = out / mode
                    out = out / ('seed%d' % seed)
                    before = len(errors)
                    for n in ['registry.json', 'completion.json', 'best_development.json']: require(out / n)
                    if len(errors) != before: continue
                    c = copy.deepcopy(read(out / 'registry.json')['signature']['config'])
                    e = c['experiment']
                    expected = (family, site, mode, seed)
                    actual = (e['family'], e['placement'], e['mode'], e['seed'])
                    if actual != expected:
                        errors.append('Registry configuration mismatch: ' + str(out)); continue
                    best = out / 'best'
                    meta = best / ('experiment.json' if group == 'insertion' else 'replacement.json')
                    require(meta)
                    if not meta.is_file(): continue
                    saved = read(meta)['config']['experiment']
                    if tuple(saved[k] for k in ['family', 'placement', 'mode', 'seed']) != expected:
                        errors.append('Best-checkpoint configuration mismatch: ' + str(best)); continue
                    checkpoint_files(best)
                    if read(out / 'completion.json').get('reload_verified') is not True:
                        errors.append('Checkpoint reload not verified: ' + str(out))
                    c['attacks'] = copy.deepcopy(shared['attacks'])
                    c['diagnostics'] = copy.deepcopy(shared['diagnostics'])
                    c['evaluation'] = copy.deepcopy(shared['evaluation'])
                    if args.limit is not None: c['attacks']['limit'] = args.limit
                    if args.device: c['model']['device'] = args.device
                    c['data']['root'] = str(absolute(c['data']['root'], base))
                    if args.data_dir: c['data']['root'] = str(absolute(args.data_dir, ROOT))
                    if args.surrogate:
                        c['attacks']['surrogate_checkpoint'] = str(absolute(args.surrogate, ROOT))
                    else:
                        c['attacks']['surrogate_checkpoint'] = str(absolute(c['attacks']['surrogate_checkpoint'], ROOT))
                    if 'transfer' in c['attacks']['names']:
                        surrogate = Path(c['attacks']['surrogate_checkpoint'])
                        if surrogate == best: errors.append('Surrogate equals target: ' + str(best))
                        if surrogate not in checked_surrogates:
                            checkpoint_files(surrogate); checked_surrogates.add(surrogate)
                    for split in set(['dev', c['evaluation']['split']]):
                        data = Path(c['data']['root'])
                        manifest = data / c['data']['track'] / (split + '.jsonl')
                        if manifest in checked_data: continue
                        checked_data.add(manifest); require(manifest)
                        if manifest.is_file():
                            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
                            if not rows: errors.append('Empty dataset: ' + str(manifest))
                            for name in sorted({row['image'] for row in rows}): require(data / 'images' / name)
                    if len(errors) == before:
                        jobs.append({'group': group, 'out': str(out), 'config': c})
                        print('READY', group, family, site, mode, 'seed', seed, flush=True)
    return jobs, errors


def worker(job):
    # Isolated process: reuse the existing attack logic with replacement-aware loading.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import experiment, experiment_model, run_attacks
    from utils import environment, sha256
    standard_load, standard_provenance = experiment_model.load_model, experiment_model.provenance
    original_sources = experiment.source_hashes
    if job['group'] == 'replacement':
        sys.path.insert(0, str(ROOT / 'quantum-replacement-full'))
        import replacement_model, settings
        original_sources = settings.source_hashes
    def load_any(path, device='auto', dtype=None):
        if (Path(path) / 'replacement.json').is_file():
            return replacement_model.load_model(path, device, dtype)
        return standard_load(path, device, dtype)
    def provenance(path):
        if path is None or not (Path(path) / 'replacement.json').is_file():
            return standard_provenance(path)
        path = Path(path); info = read(path / 'replacement.json'); current = environment()
        return {'weights_sha256': sha256(path / 'model.pt'),
                'metadata_sha256': sha256(path / 'replacement.json'),
                'environment_differences': {k: {'saved': v, 'current': current.get(k)}
                    for k, v in info['environment'].items() if v != current.get(k)}}
    def sources():
        result = original_sources()
        result['local_full_attack_launcher'] = sha256(Path(__file__))
        return result
    out = Path(job['out'])
    experiment.load_model = load_any; experiment.provenance = provenance
    experiment.source_hashes = sources; experiment.run_dir = lambda _: out
    run_attacks.load_checkpoint = load_any; run_attacks.provenance = provenance
    run_attacks.source_hashes = sources
    c = job['config']
    experiment.action(c, 'diagnose', out / 'best')
    experiment.action(c, 'attack', out / 'best', resume=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'insertion', 'replacement', '_worker'])
    p.add_argument('--config', default='configs/shared.yaml')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--device', help='For example cuda:0 or cuda:1')
    p.add_argument('--data-dir'); p.add_argument('--insertion-root'); p.add_argument('--replacement-root')
    p.add_argument('--surrogate'); p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.action == '_worker': worker(json.load(sys.stdin)); return
    if a.limit is not None and a.limit < 1: p.error('--limit must be positive')
    groups = ['insertion', 'replacement'] if a.action == 'check' else [a.action]
    try:
        jobs, errors = preflight(a, groups)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise SystemExit('Preflight failed: ' + str(exc))
    if errors:
        print('\nPREFLIGHT FAILED. No diagnostics or attacks started.', file=sys.stderr)
        for e in dict.fromkeys(errors): print(' - ' + e, file=sys.stderr)
        raise SystemExit(1)
    expected = 16 * len(a.seeds) * len(groups)
    if len(jobs) != expected: raise SystemExit('Incomplete job matrix; nothing started.')
    print('\nPREFLIGHT PASSED: %d full-training targets. No training will run.' % len(jobs), flush=True)
    if a.action == 'check': return
    for job in jobs:
        print('\nATTACK TARGET:', job['out'], flush=True)
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '_worker'],
                       input=json.dumps(job), text=True, check=True)


if __name__ == '__main__':
    main()
