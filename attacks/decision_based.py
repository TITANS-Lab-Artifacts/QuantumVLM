"""Foolbox HopSkipJump with only hard labels exposed to the attack."""

import torch
from torch import nn
from attacks.common import AttackResult
from attacks.oracles import BudgetExceeded


class HardLabelModel(nn.Module):
    def __init__(self, oracle):
        super().__init__()
        self.oracle = oracle

    def forward(self, x):
        labels = self.oracle.labels(x)
        # Foolbox expects a class-score array; fixed one-hot values convey labels only.
        return torch.nn.functional.one_hot(labels, 2).float()


def hop_skip_jump(
    oracle,
    original,
    label,
    epsilon=4 / 255,
    steps=32,
    initial_samples=16,
    max_samples=128,
):
    import foolbox as fb

    label = torch.as_tensor(label, device=original.device).long().reshape(1)
    wrapped = HardLabelModel(oracle).eval()
    fmodel = fb.PyTorchModel(wrapped, bounds=(0, 1), device=original.device)
    attack = fb.attacks.HopSkipJumpAttack(
        constraint="linf",
        steps=steps,
        initial_gradient_eval_steps=initial_samples,
        max_gradient_eval_steps=max_samples,
        init_attack=fb.attacks.LinearSearchBlendedUniformNoiseAttack(
            directions=100, steps=50
        ),
    )
    status = "completed"
    try:
        _, clipped, _ = attack(fmodel, original, label, epsilons=epsilon)
        oracle.labels(clipped)
    except BudgetExceeded:
        status = "query_budget_exhausted"
    except ValueError as exc:
        if "init_attack failed" not in str(exc):
            raise
        status = "initialization_failed"
    return AttackResult(
        oracle.tracker.best,
        oracle.calls,
        status,
        {"library": "foolbox", "method": "HopSkipJump", "constraint": "linf"},
    )
