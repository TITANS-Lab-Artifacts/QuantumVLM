"""Simulated score-only and label-only access; every candidate forward is charged."""

import torch
from attacks.common import CandidateTracker


class BudgetExceeded(RuntimeError):
    pass


class _Oracle:
    def __init__(self, predict, original, label, epsilon, budget):
        if budget < 1:
            raise ValueError("Query budget must be positive")
        self._predict = predict
        self.budget = budget
        self.calls = 0
        self.tracker = CandidateTracker(original, label, epsilon)

    def _one(self, x):
        if self.calls >= self.budget:
            raise BudgetExceeded("Target query budget exhausted")
        self.calls += 1
        with torch.no_grad():
            return self._predict(x).detach()


class ScoreOracle(_Oracle):
    def scores(self, images):
        out = []
        for image in images:
            x = image.unsqueeze(0)
            scores = self._one(x)
            self.tracker.observe(x, scores=scores)
            out.append(scores)
        return torch.cat(out, 0)


class DecisionOracle(_Oracle):
    def labels(self, images):
        out = []
        for image in images:
            x = image.unsqueeze(0)
            label = self._one(x).argmax(-1)
            # The decision-only tracker never ranks candidates using confidence scores.
            self.tracker.observe(x, prediction=label.item())
            out.append(label)
        return torch.cat(out, 0)
