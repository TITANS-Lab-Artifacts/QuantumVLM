"""Configuration for full-unfrozen replacement experiments only."""
import bootstrap
import copy
from pathlib import Path
import yaml
from experiment_config import DEFAULTS, merge
from replacement_model import FAMILIES, POSITIONS
from utils import sha256, source_hashes as parent_hashes


def load(path, overrides=()):
    c = copy.deepcopy(DEFAULTS)
    c['experiment'].update(family=FAMILIES[0], mode='full_unfrozen', placement='r1')
    c['data']['root'] = '../data/vsr'
    c['cache']['enabled'] = False
    c['output']['root'] = 'runs'
    if path:
        merge(c, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        key, value = item.split('=', 1)
        keys = key.split('.')
        target = c
        for k in keys[:-1]:
            target = target[k]
        if keys[-1] not in target:
            raise ValueError('Unknown option: ' + key)
        target[keys[-1]] = yaml.safe_load(value)
    e, t = c['experiment'], c['training']
    if e['family'] not in FAMILIES or e['placement'] not in POSITIONS:
        raise ValueError('Choose a supported circuit family and r1/r2/r3/r4')
    if e['mode'] != 'full_unfrozen' or c['cache']['enabled']:
        raise ValueError('This package requires full_unfrozen training with feature caching disabled')
    if e['blocks'] < 1 or e['chunk_size'] < 1:
        raise ValueError('Circuit blocks and chunk size must be positive')
    if e['block_index'] is not None:
        raise ValueError('R2/R3/R4 have registered indices 0/11/23; do not override block_index')
    for k in ['epochs', 'patience', 'accumulation', 'save_every_updates', 'log_every_updates']:
        if t[k] < 1:
            raise ValueError(k + ' must be positive')
    if not 0 <= t['warmup_fraction'] < 1 or t['min_delta'] < 0 or t['weight_decay'] < 0 or t['clip_norm'] <= 0:
        raise ValueError('Invalid optimization settings')
    if c['monitor']['interval_seconds'] <= 0 or c['evaluation']['bootstrap_draws'] < 1:
        raise ValueError('Invalid monitoring or bootstrap setting')
    if c['evaluation']['split'] not in ['dev', 'test']:
        raise ValueError('Evaluation split must be dev or test')
    if c['evaluation']['split'] == 'test' and not c['evaluation']['allow_test']:
        raise ValueError('Explicit evaluation.allow_test=true is required for final test evaluation')
    for key in ['train_limit','eval_limit']:
        val = c['data'][key]
        if val is not None and (not isinstance(val, int) or val < 1):
            raise ValueError('Data limits must be positive or null')
    return c


def run_dir(c):
    e = c['experiment']
    if c['output']['run_name']:
        return Path(c['output']['root']) / c['output']['run_name']
    return Path(c['output']['root']) / e['family'] / e['placement'] / ('seed%d' % e['seed'])


def source_hashes():
    result = {'parent/' + k: v for k, v in parent_hashes().items()}
    result.update({'replacement/' + p.name: sha256(p) for p in sorted(bootstrap.HERE.glob('*.py'))})
    return result
