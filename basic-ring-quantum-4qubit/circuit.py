"""Exact differentiable four-qubit RY/CNOT statevector simulation in PyTorch.

Basis |q0 q1 q2 q3>, q0 is the most significant bit. Real amplitudes are
sufficient because every gate in this specific circuit is real. No shots,
noise, hardware calls, detached tensors, or external quantum library.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class BasicRing4(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.blocks = blocks
        self.theta = nn.Parameter(torch.empty(blocks, 4).uniform_(-0.1, 0.1))
        indices = torch.arange(16)
        for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
            perm = indices ^ (((indices >> (3 - control)) & 1) << (3 - target))
            self.register_buffer("cnot_%d_%d" % (control, target), perm)
        signs = torch.stack(
            [1 - 2 * ((indices >> (3 - q)) & 1) for q in range(4)], dim=1
        )
        self.register_buffer("z_signs", signs.float())

    @staticmethod
    def ry(state, angle, qubit):
        shape = state.shape
        x = state.reshape(-1, 2, 2, 2, 2).movedim(qubit + 1, -1)
        c, s = torch.cos(angle / 2), torch.sin(angle / 2)
        c, s = c.reshape(-1, 1, 1, 1), s.reshape(-1, 1, 1, 1)
        a, b = x[..., 0], x[..., 1]
        return (
            torch.stack((c * a - s * b, s * a + c * b), dim=-1)
            .movedim(-1, qubit + 1)
            .reshape(shape)
        )

    def statevector(self, angles):
        if angles.ndim != 2 or angles.shape[1] != 4:
            raise ValueError("Expected [number_of_tokens, 4] angles")
        state = torch.zeros(
            angles.shape[0], 16, dtype=angles.dtype, device=angles.device
        )
        state[:, 0] = 1
        for q in range(4):
            state = self.ry(state, angles[:, q], q)
        for block in range(self.blocks):
            for q in range(4):
                state = self.ry(state, self.theta[block, q].expand(angles.shape[0]), q)
            for c, t in ((0, 1), (1, 2), (2, 3), (3, 0)):
                state = state.index_select(-1, getattr(self, "cnot_%d_%d" % (c, t)))
        return state

    def forward(self, angles):
        state = self.statevector(angles)
        return state.square() @ self.z_signs.to(state.dtype)

    def specification(self):
        sequence = [{"gate": "RY", "wire": q, "angle": "z[%d]" % q} for q in range(4)]
        for b in range(self.blocks):
            sequence += [
                {"gate": "RY", "wire": q, "angle": "theta[%d,%d]" % (b, q)}
                for q in range(4)
            ]
            sequence += [
                {"gate": "CNOT", "control": c, "target": t}
                for c, t in ((0, 1), (1, 2), (2, 3), (3, 0))
            ]
        return {
            "qubits": 4,
            "blocks": self.blocks,
            "encoding": "RY once, no re-uploading",
            "measurement": ["Z0", "Z1", "Z2", "Z3"],
            "shots": None,
            "simulation": "exact real statevector, float32, torch autograd",
            "basis": "|q0 q1 q2 q3>, q0 most significant",
            "ordered_gates": sequence,
            "one_qubit_gates": 4 + 4 * self.blocks,
            "two_qubit_gates": 4 * self.blocks,
        }


class QuantumResidual(nn.Module):
    def __init__(self, width, blocks=2, chunk_size=256):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.width, self.chunk_size = width, chunk_size
        self.ring = BasicRing4(blocks)
        self.in_proj = nn.Linear(width, 4)
        self.out_proj = nn.Linear(4, width)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.last_tokens = 0
        self.collect_stats = False
        self.circuit_evaluations = 0
        self.relative_update_sum = 0.0
        self.forward_count = 0

    def forward(self, hidden):
        dtype, shape = hidden.dtype, hidden.shape
        flat = hidden.reshape(-1, self.width).to(self.in_proj.weight.dtype)
        updates = []
        for part in flat.split(self.chunk_size):
            angles = math.pi * torch.tanh(
                self.in_proj(F.layer_norm(part, (self.width,)))
            )
            updates.append(self.out_proj(self.ring(angles)))
        self.last_tokens = len(flat)
        delta = (self.scale * torch.cat(updates, dim=0)).reshape(shape)
        effective_delta = delta.to(dtype)
        if self.collect_stats:
            self.circuit_evaluations += self.last_tokens
            ratio = (
                effective_delta.detach().float().norm()
                / hidden.detach().float().norm().clamp_min(1e-12)
            )
            self.relative_update_sum += ratio.item()
            self.forward_count += 1
        return hidden + effective_delta
