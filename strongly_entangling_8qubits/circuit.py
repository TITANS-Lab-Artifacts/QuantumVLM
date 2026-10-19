"""Exact complex statevector for RY encoding and strongly entangling Rot blocks.
Rot(phi,theta,omega) acts chronologically RZ(phi), RY(theta), RZ(omega).
Its matrix is RZ(omega) @ RY(theta) @ RZ(phi). Qubit zero is MSB.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

QUBITS = 8


class StronglyEntangling(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.blocks = blocks
        self.ranges = [1 + b % (QUBITS - 1) for b in range(blocks)]
        self.theta = nn.Parameter(torch.empty(blocks, QUBITS, 3).uniform_(-0.1, 0.1))
        indices = torch.arange(2**QUBITS)
        for r in sorted(set(self.ranges)):
            for c in range(QUBITS):
                t = (c + r) % QUBITS
                perm = indices ^ (
                    ((indices >> (QUBITS - 1 - c)) & 1) << (QUBITS - 1 - t)
                )
                self.register_buffer("cnot_%d_%d" % (c, t), perm)
        self.register_buffer(
            "z_signs",
            torch.stack(
                [1 - 2 * ((indices >> (QUBITS - 1 - q)) & 1) for q in range(QUBITS)],
                dim=1,
            ).float(),
        )

    @staticmethod
    def rotation(state, angle, qubit, axis):
        x = state.reshape((-1,) + (2,) * QUBITS).movedim(qubit + 1, -1)
        a, b = x[..., 0], x[..., 1]
        angle = angle.reshape((-1,) + (1,) * (QUBITS - 1))
        if axis == "Y":
            c, s = torch.cos(angle / 2), torch.sin(angle / 2)
            u, v = c * a - s * b, s * a + c * b
        elif axis == "Z":
            phase = torch.complex(torch.cos(angle / 2), torch.sin(angle / 2))
            u, v = phase.conj() * a, phase * b
        else:
            raise ValueError("Unsupported rotation")
        return torch.stack((u, v), dim=-1).movedim(-1, qubit + 1).reshape_as(state)

    def statevector(self, angles):
        if (
            angles.ndim != 2
            or angles.shape[1] != QUBITS
            or angles.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Expected [tokens,qubits] real float32/float64 angles")
        dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        state = torch.zeros(
            angles.shape[0], 2**QUBITS, dtype=dtype, device=angles.device
        )
        state[:, 0] = 1
        for q in range(QUBITS):
            state = self.rotation(state, angles[:, q], q, "Y")
        for b, r in enumerate(self.ranges):
            for q in range(QUBITS):
                for k, axis in enumerate(("Z", "Y", "Z")):
                    state = self.rotation(
                        state, self.theta[b, q, k].expand(len(angles)), q, axis
                    )
            for c in range(QUBITS):
                state = state.index_select(
                    -1, getattr(self, "cnot_%d_%d" % (c, (c + r) % QUBITS))
                )
        return state

    def forward(self, angles):
        state = self.statevector(angles)
        probabilities = state.real.square() + state.imag.square()
        return probabilities @ self.z_signs.to(probabilities.dtype)

    def specification(self):
        gates = [{"gate": "RY", "wire": q, "angle": "z[%d]" % q} for q in range(QUBITS)]
        for b, r in enumerate(self.ranges):
            for q in range(QUBITS):
                for k, gate in enumerate(("RZ", "RY", "RZ")):
                    gates.append(
                        {
                            "gate": gate,
                            "wire": q,
                            "angle": "theta[%d,%d,%d]" % (b, q, k),
                        }
                    )
            gates += [
                {"gate": "CNOT", "control": c, "target": (c + r) % QUBITS}
                for c in range(QUBITS)
            ]
        return {
            "family": "strongly_entangling",
            "qubits": QUBITS,
            "blocks": self.blocks,
            "ranges": self.ranges,
            "encoding": "RY once, no re-uploading",
            "measurement": ["Z%d" % q for q in range(QUBITS)],
            "shots": None,
            "simulation": "exact complex statevector; complex64 with float32 parameters; torch autograd",
            "basis": "qubit 0 most significant",
            "rot_convention": "RZ(omega) @ RY(theta) @ RZ(phi)",
            "ordered_gates": gates,
            "one_qubit_gates": QUBITS + 3 * QUBITS * self.blocks,
            "two_qubit_gates": QUBITS * self.blocks,
            "nominal_circuit_parameters": 3 * QUBITS * self.blocks,
            "known_readout_inactive_angles": [
                "theta[%d,%d,2]" % (self.blocks - 1, q) for q in range(QUBITS)
            ],
            "note": "Last-block final RZ angles do not change Z expectations through subsequent CNOTs; retained to match the full Rot template. Additional degeneracies are not excluded.",
        }


class QuantumResidual(nn.Module):
    def __init__(self, width, blocks=2, chunk_size=256):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.width, self.chunk_size = width, chunk_size
        self.circuit = StronglyEntangling(blocks)
        self.in_proj = nn.Linear(width, 8)
        self.out_proj = nn.Linear(8, width)
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
            updates.append(self.out_proj(self.circuit(angles)))
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
