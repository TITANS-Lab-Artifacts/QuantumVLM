"""Pixel constraints, success-preserving candidate selection and attack outputs."""

from dataclasses import dataclass, field
import torch
from torch.nn import functional as F


@dataclass
class AttackResult:
    image: torch.Tensor
    calls: int
    status: str = "completed"
    details: dict = field(default_factory=dict)


def project(x, original, epsilon):
    return torch.maximum(
        torch.minimum(x, original + epsilon), original - epsilon
    ).clamp(0, 1)


def validate_candidate(x, original, epsilon):
    if x.shape != original.shape or not torch.isfinite(x).all():
        raise ValueError("Invalid attack image")
    distance = (x - original).abs().max().item()
    if x.min() < -1e-6 or x.max() > 1 + 1e-6 or distance > epsilon + 1e-6:
        raise ValueError(f"Attack violated the pixel threat model: L_inf={distance}")
    return distance


def objective(scores, label, loss="ce"):
    if loss == "ce":
        return F.cross_entropy(scores, label, reduction="none")
    if loss == "margin":
        return scores.gather(1, (1 - label)[:, None]).squeeze(1) - scores.gather(
            1, label[:, None]
        ).squeeze(1)
    raise ValueError(loss)


class CandidateTracker:
    def __init__(self, original, label, epsilon):
        self.original = original.detach()
        self.label = int(label)
        self.epsilon = epsilon
        self.best = original.detach().clone()
        self.best_key = (-1, float("-inf"))
        self.success = False

    def observe(self, x, scores=None, prediction=None):
        if (x - self.original).abs().max().item() > self.epsilon + 1e-6:
            return
        pred = int(scores.argmax(-1).item()) if scores is not None else int(prediction)
        success = pred != self.label
        loss = (
            float(
                objective(
                    scores, torch.tensor([self.label], device=scores.device), "margin"
                ).item()
            )
            if scores is not None
            else 0.0
        )
        key = (int(success), loss)
        if key > self.best_key:
            self.best_key = key
            self.best = x.detach().clone()
        self.success |= success
