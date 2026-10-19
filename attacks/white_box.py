"""FGSM and multi-restart projected gradient descent in native pixel space."""

import torch
from attacks.common import AttackResult, CandidateTracker, objective, project


def pgd(
    predict,
    original,
    label,
    epsilon=4 / 255,
    steps=100,
    restarts=5,
    step_size=None,
    random_start=True,
    loss="ce",
):
    if steps < 1 or restarts < 1 or epsilon < 0:
        raise ValueError("Invalid PGD settings")
    if original.shape[0] != 1:
        raise ValueError("Attacks use one image/prompt at a time")
    label = torch.as_tensor(label, device=original.device).long().reshape(1)
    alpha = step_size if step_size is not None else 2.5 * epsilon / steps
    tracker = CandidateTracker(original, label.item(), epsilon)
    calls = 0
    with torch.no_grad():
        tracker.observe(original, scores=predict(original))
        calls += 1
    for _ in range(restarts):
        x = (
            project(
                original + torch.empty_like(original).uniform_(-epsilon, epsilon),
                original,
                epsilon,
            )
            if random_start
            else original.clone()
        )
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                scores = predict(x)
                calls += 1
                tracker.observe(x, scores=scores.detach())
                (grad,) = torch.autograd.grad(objective(scores, label, loss).sum(), x)
            if not torch.isfinite(grad).all():
                raise FloatingPointError("Nonfinite pixel gradient")
            x = project(x.detach() + alpha * grad.sign(), original, epsilon)
        with torch.no_grad():
            tracker.observe(x, scores=predict(x))
            calls += 1
    return AttackResult(
        tracker.best,
        calls,
        details={
            "steps": steps,
            "restarts": restarts,
            "step_size": alpha,
            "loss": loss,
            "zero_gradient_warning": bool(grad.abs().max() == 0),
        },
    )


def fgsm(predict, original, label, epsilon=4 / 255):
    return pgd(
        predict,
        original,
        label,
        epsilon,
        steps=1,
        restarts=1,
        step_size=epsilon,
        random_start=False,
    )
