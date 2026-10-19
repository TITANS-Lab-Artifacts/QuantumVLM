"""Actual component removal, full-backbone training and complete state checkpoints."""
import bootstrap
import importlib.util
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from config import ModelConfig, write_json
from model import VLMClassifier
from utils import sha256, environment

FAMILIES = ['basic-ring-quantum-4qubit', 'basic-ring-quantum-8qubit',
            'strongly_entangling_4qubits', 'strongly_entangling_8qubits']
POSITIONS = {'r1': None, 'r2': 0, 'r3': 11, 'r4': 23}

class Replacement(nn.Module):
    def __init__(self, din, dout, family, blocks=2, chunk_size=256, normalize=False):
        super().__init__()
        path = bootstrap.PARENT / family / 'circuit.py'
        spec = importlib.util.spec_from_file_location(family.replace('-', '_') + '_replacement_circuit', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Reuse the circuit only, not the residual wrapper or its projections.
        temporary = module.QuantumResidual(din, blocks, chunk_size)
        self.circuit = temporary.circuit if hasattr(temporary, 'circuit') else temporary.ring
        q = self.circuit.specification()['qubits']
        self.in_proj = nn.Linear(din, q)
        self.out_proj = nn.Linear(q, dout)
        self.normalize, self.chunk_size = normalize, chunk_size
        self.din, self.dout = din, dout
        self.token_evaluations = 0
        self.collect_stats = False
        # Standard Linear initialization: no old residual scale or tiny-delta initialization.

    def forward(self, hidden):
        shape, dtype = hidden.shape, hidden.dtype
        flat = hidden.reshape(-1, self.din).to(self.in_proj.weight.dtype)
        outputs = []
        for part in flat.split(self.chunk_size):
            if self.normalize:
                part = F.layer_norm(part, (self.din,))
            angles = math.pi * torch.tanh(self.in_proj(part))
            outputs.append(self.out_proj(self.circuit(angles)))
        if self.collect_stats:
            self.token_evaluations += len(flat)
        return torch.cat(outputs).reshape(*shape[:-1], self.dout).to(dtype)


def install(base, c):
    e = c['experiment']
    site = e['placement']
    if e['family'] not in FAMILIES or site not in POSITIONS:
        raise ValueError('Unsupported replacement family or position')
    count_before = sum(p.numel() for p in base.parameters())
    language = base.backbone.model.language_model
    if len(language.layers) != 24:
        raise ValueError('This registered R1–R4 plan requires the pinned 24-layer backbone')
    if site == 'r1':
        owner, name = base.backbone.model, 'multi_modal_projector'
        original = getattr(owner, name)
        din, dout = original.linear_1.in_features, original.linear_2.out_features
    else:
        owner, name = language.layers[POSITIONS[site]], 'mlp'
        original = getattr(owner, name)
        din, dout = original.gate_proj.in_features, original.down_proj.out_features
    removed = sum(p.numel() for p in original.parameters())
    new = Replacement(din, dout, e['family'], e['blocks'], e['chunk_size'], site == 'r1').to(base.device_)
    setattr(owner, name, new)
    del original
    base.requires_grad_(True)
    base.replacement_metadata = {
        'position': site, 'decoder_index': POSITIONS[site], 'family': e['family'],
        'input_dimension': din, 'output_dimension': dout,
        'parameters_before': count_before, 'parameters_removed': removed,
        'replacement_parameters': sum(p.numel() for p in new.parameters()),
        'parameters_after': sum(p.numel() for p in base.parameters()),
        'circuit': new.circuit.specification(),
        'normalization': 'non-affine LayerNorm' if site == 'r1' else 'existing upstream RMSNorm',
        'internal_residual': False, 'all_retained_weights_trainable': True,
        'initialization': 'PyTorch Linear defaults; existing circuit angle initialization',
    }
    base.eval()
    return base


def build(c):
    if c['experiment']['mode'] != 'full_unfrozen':
        raise ValueError('Only full_unfrozen mode is supported in this package')
    return install(VLMClassifier.load(ModelConfig(**c['model'], mode='full_unfrozen')), c)


def base_of(m):
    return m


def save_model(m, path, c):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # All retained model parameters and persistent buffers, not just quantum weights.
    state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(state, path / 'model.pt')
    write_json(path / 'replacement.json', {
        'config': c, 'resolved_model': vars(m.settings),
        'architecture': m.replacement_metadata, 'environment': environment(),
        'model_sha256': sha256(path / 'model.pt'),
    })


def load_model(path, device='auto', dtype=None):
    path = Path(path)
    info = json.loads((path / 'replacement.json').read_text())
    if sha256(path / 'model.pt') != info['model_sha256']:
        raise ValueError('Model checkpoint checksum mismatch')
    cfg = ModelConfig(**info['resolved_model'])
    cfg.device = device
    cfg.gradient_checkpointing = False
    if dtype and dtype != 'auto':
        cfg.dtype = dtype
    m = install(VLMClassifier.load(cfg), info['config'])
    state = torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint values')
    m.load_state_dict(state, strict=True)
    m.eval()
    return m
