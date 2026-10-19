"""Unified model construction and verified, portable learned-weight checkpoints."""
import importlib.util
import json
from pathlib import Path
import torch
from config import ModelConfig, write_json
from model import VLMClassifier, load_checkpoint as load_classical
from quantum_shared import QuantumVLM
from utils import sha256, environment

BASE_MODES={'quantum_only':'full_frozen','quantum_head':'frozen_head',
            'quantum_projector':'projector','quantum_lora':'lora','quantum_full':'full_unfrozen'}

def attach(base,c):
    e=c['experiment']
    if e['family']=='classical': return base
    path=Path(__file__).parent/e['family']/'circuit.py'
    spec=importlib.util.spec_from_file_location(e['family'].replace('-','_')+'_circuit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return QuantumVLM(base,e['placement'],module.QuantumResidual,e['blocks'],e['block_index'],e['chunk_size'])

def build(c):
    e=c['experiment']
    if e['family']!='classical' and e['mode']=='full_frozen':
        raise ValueError('Frozen quantum evaluation requires --checkpoint from a trained run')
    mode=BASE_MODES.get(e['mode'],e['mode'])
    base=VLMClassifier.load(ModelConfig(**c['model'],mode=mode))
    return attach(base,c)

def base_of(m): return m.base if isinstance(m,QuantumVLM) else m

def save_model(m,path,c):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    base=base_of(m);base.save_checkpoint(path/'base')
    info={'config':c,'environment':environment(),'base_hash':sha256(path/'base/weights.pt')}
    if isinstance(m,QuantumVLM):
        torch.save(m.adapter.state_dict(),path/'adapter.pt')
        info.update(adapter_hash=sha256(path/'adapter.pt'),architecture=m.describe())
    write_json(path/'experiment.json',info)

def load_model(path,device='auto',dtype=None):
    path=Path(path)
    if not (path/'experiment.json').exists(): return load_classical(path,device,dtype)
    info=json.loads((path/'experiment.json').read_text())
    if sha256(path/'base/weights.pt')!=info['base_hash']: raise ValueError('Base hash mismatch')
    base=load_classical(path/'base',device,dtype)
    m=attach(base,info['config'])
    if isinstance(m,QuantumVLM):
        if sha256(path/'adapter.pt')!=info['adapter_hash']: raise ValueError('Adapter hash mismatch')
        state=torch.load(path/'adapter.pt',map_location='cpu',weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('Nonfinite adapter')
        m.adapter.load_state_dict(state,strict=True)
    m.eval();return m

def provenance(path):
    from utils import checkpoint_provenance
    path=Path(path)
    if not (path/'experiment.json').exists(): return checkpoint_provenance(path)
    info=json.loads((path/'experiment.json').read_text());current=environment()
    return {'experiment_sha256':sha256(path/'experiment.json'),
            'base':checkpoint_provenance(path/'base'),
            'adapter_sha256':sha256(path/'adapter.pt') if (path/'adapter.pt').exists() else None,
            'environment_differences':{k:{'saved':v,'current':current.get(k)} for k,v in info['environment'].items() if v!=current.get(k)}}
