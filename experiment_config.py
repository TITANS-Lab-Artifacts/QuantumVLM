"""Strict YAML configuration and dotted command-line overrides (Python 3.9+)."""
import copy
import hashlib
import json
from pathlib import Path
import yaml

FAMILIES = ['classical', 'basic-ring-quantum-4qubit', 'basic-ring-quantum-8qubit',
            'strongly_entangling_4qubits', 'strongly_entangling_8qubits']
DEFAULTS = {
 'model': {'model_id': 'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
           'revision': '24efa9ce1ebc62c778946f8c6b119922ea6169bd',
           'device': 'auto', 'dtype': 'auto', 'lora_rank': 8, 'lora_alpha': 16,
           'lora_dropout': 0.05, 'gradient_checkpointing': False},
 'experiment': {'family': 'classical', 'mode': 'lora', 'placement': 'q4',
                'blocks': 2, 'block_index': None, 'chunk_size': 256, 'seed': 0},
 'data': {'root': 'data/vsr', 'track': 'random', 'train_limit': None, 'eval_limit': None},
 'training': {'epochs': 20, 'patience': 5, 'min_delta': 0.0, 'accumulation': 16,
              'lr': None, 'weight_decay': 0.01, 'clip_norm': 1.0,
              'warmup_fraction': 0.05, 'save_every_updates': 100, 'log_every_updates': 10},
 'cache': {'enabled': True, 'directory': 'feature_cache', 'verify_examples': 3,
           'atol': 1e-5, 'rtol': 1e-4},
 'evaluation': {'split': 'dev', 'allow_test': False, 'bootstrap_draws': 1000},
 'monitor': {'interval_seconds': 1.0},
 'attacks': {'names': ['fgsm', 'pgd_ce', 'pgd_margin', 'spsa', 'hsj', 'transfer'],
             'epsilon': 4/255, 'steps': 100, 'restarts': 5, 'step_size': None,
             'query_budget': 1000, 'spsa_samples': 16, 'spsa_delta': 0.01,
             'hsj_steps': 32, 'limit': 200, 'save_images': True,
             'objective': 'untargeted', 'surrogate_checkpoint': 'runs/classical/frozen_head/seed0/best'},
 'diagnostics': {'examples': 3, 'directions': 3, 'steps': [0.0003, 0.0001, 0.00003, 0.00001]},
 'output': {'root': 'runs', 'run_name': None},
}

def merge(dst, src):
    for k,v in src.items():
        if k not in dst: raise ValueError('Unknown config option: '+k)
        if isinstance(dst[k], dict):
            if not isinstance(v,dict): raise ValueError('Expected mapping: '+k)
            merge(dst[k],v)
        else: dst[k]=v

def load(path, overrides=()):
    c=copy.deepcopy(DEFAULTS)
    if path: merge(c,yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        key,value=item.split('=',1); keys=key.split('.'); target=c
        for k in keys[:-1]: target=target[k]
        if keys[-1] not in target: raise ValueError('Unknown override: '+key)
        target[keys[-1]]=yaml.safe_load(value)
    e,t=c['experiment'],c['training']
    if e['family'] not in FAMILIES: raise ValueError('Unknown family')
    allowed=['zero_shot','full_frozen','frozen_head','projector','lora','full_unfrozen'] if e['family']=='classical' else ['full_frozen','quantum_only','quantum_head','quantum_projector','quantum_lora','quantum_full']
    if e['mode'] not in allowed: raise ValueError('Mode must be one of '+str(allowed))
    if e['placement'] not in ['q1','q2','q3','q4']: raise ValueError('Unknown placement')
    for k in ['epochs','patience','accumulation','save_every_updates','log_every_updates']:
        if t[k]<1: raise ValueError(k+' must be positive')
    if not 0<=t['warmup_fraction']<1 or t['min_delta']<0: raise ValueError('Invalid stopping/scheduler settings')
    if c['monitor']['interval_seconds']<=0: raise ValueError('Invalid monitor interval')
    if c['evaluation']['split']=='test' and not c['evaluation']['allow_test']: raise ValueError('Set evaluation.allow_test=true only after locking experiments')
    if c['attacks']['objective'] not in ['untargeted','false_acceptance','false_rejection']: raise ValueError('Unknown attack objective')
    if t['clip_norm']<=0 or t['weight_decay']<0: raise ValueError('Invalid optimizer settings')
    if c['cache']['verify_examples']<1 or c['cache']['atol']<0 or c['cache']['rtol']<0: raise ValueError('Invalid cache verification settings')
    if c['diagnostics']['examples']<1 or c['diagnostics']['directions']<1: raise ValueError('Invalid diagnostic counts')
    if c['evaluation']['bootstrap_draws']<1: raise ValueError('bootstrap_draws must be positive')
    for limit in (c['data']['train_limit'],c['data']['eval_limit'],c['attacks']['limit']):
        if limit is not None and (not isinstance(limit,int) or limit<1): raise ValueError('Limits must be positive integers or null')
    return c

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def run_dir(c):
    e=c['experiment']; name=c['output']['run_name']
    if name: return Path(c['output']['root'])/name
    parts=[c['output']['root'],e['family']]
    if e['family']!='classical': parts.append(e['placement'])
    return Path(*parts)/e['mode']/('seed%d'%e['seed'])
