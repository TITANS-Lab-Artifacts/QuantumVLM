"""SPSA: a two-sided, score-query finite-difference attack with projected Adam updates."""

import torch
from attacks.common import AttackResult, objective, project
from attacks.oracles import BudgetExceeded


def spsa(
    oracle,
    original,
    label,
    epsilon=4 / 255,
    steps=100,
    samples=16,
    delta=0.01,
    step_size=None,
    restarts=1,
):
    if samples < 1 or steps < 1 or delta <= 0 or restarts < 1:
        raise ValueError("Invalid SPSA settings")
    label = torch.as_tensor(label, device=original.device).long().reshape(1)
    alpha = step_size if step_size is not None else epsilon / 10
    status = "completed"
    try:
        oracle.scores(original)
        for _ in range(restarts):
            x = project(
                original + torch.empty_like(original).uniform_(-epsilon, epsilon),
                original,
                epsilon,
            )
            first = torch.zeros_like(x)
            second = torch.zeros_like(x)
            for t in range(1, steps + 1):
                grad = torch.zeros_like(x)
                for _ in range(samples):
                    u = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
                    # Queries can leave the epsilon ball, but always stay in the valid RGB domain.
                    plus = (x + delta * u).clamp(0, 1)
                    minus = (x - delta * u).clamp(0, 1)
                    lp = objective(oracle.scores(plus), label, "margin")
                    lm = objective(oracle.scores(minus), label, "margin")
                    grad.add_((lp - lm).view(1, 1, 1, 1) * u / (2 * delta * samples))
                first = 0.9 * first + 0.1 * grad
                second = 0.999 * second + 0.001 * grad.square()
                direction = (first / (1 - 0.9**t)) / (
                    (second / (1 - 0.999**t)).sqrt() + 1e-8
                )
                x = project(x + alpha * direction, original, epsilon).detach()
                oracle.scores(x)
    except BudgetExceeded:
        status = "query_budget_exhausted"
    return AttackResult(
        oracle.tracker.best,
        oracle.calls,
        status,
        {
            "method": "SPSA with Adam",
            "samples_per_step": samples,
            "delta": delta,
            "restarts": restarts,
        },
    )
