"""Multi-example pixel-gradient diagnostic with explicit precision and repeatability."""

import argparse
import torch
from config import model_arguments, data_arguments, model_config, write_json
from data import VSRDataset
from model import VLMClassifier, load_checkpoint
from utils import seed_everything, environment, checkpoint_provenance, source_hashes


def check_gradients(
    predict,
    image,
    steps=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
    directions=3,
    rtol=0.1,
    atol=1e-3,
    seed=0,
):
    if directions < 1 or len(steps) < 2 or any(not 0 < s < 0.02 for s in steps):
        raise ValueError(
            "Use positive directions and at least two distinct steps in (0, 0.02)"
        )
    if len(set(steps)) != len(steps) or rtol < 0 or atol < 0:
        raise ValueError("Steps must be distinct; tolerances must be nonnegative")
    # Avoid clipping kinks; this modified diagnostic image is not an attack example.
    x = image.detach().float().clamp(0.02, 0.98).requires_grad_(True)

    def margin(v):
        scores = predict(v).double()
        return scores[0, 1] - scores[0, 0]

    value = margin(x)
    if not value.requires_grad:
        return {"status": "failed", "reason": "Detached output"}
    (grad,) = torch.autograd.grad(value, x, allow_unused=True)
    if grad is None or not torch.isfinite(grad).all() or not torch.isfinite(value):
        return {"status": "failed", "reason": "Missing or nonfinite gradient/output"}
    with torch.no_grad():
        repeats = [margin(x.detach()).item() for _ in range(3)]
    if not all(torch.isfinite(torch.tensor(v)) for v in repeats):
        return {"status": "failed", "reason": "Nonfinite repeated output"}
    noise = max(repeats + [value.item()]) - min(repeats + [value.item()])
    generator = torch.Generator(device=x.device).manual_seed(seed)
    probes = []
    passed_directions = 0
    for i in range(directions):
        direction = (
            torch.empty_like(x).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        )
        expected = (grad.double() * direction.double()).sum().item()
        agreements = []
        for step in sorted(steps, reverse=True):
            with torch.no_grad():
                plus = margin(x.detach() + step * direction).item()
                minus = margin(x.detach() - step * direction).item()
            measured = (plus - minus) / (2 * step)
            error = abs(expected - measured)
            # Two consecutive agreeing step sizes are required per direction.
            # Near-zero signals or unstable forward evaluations cannot pass.
            finite = bool(torch.isfinite(torch.tensor(measured)))
            resolved = abs(plus - minus) > max(10 * noise, 1e-7)
            agrees = (
                finite
                and resolved
                and error <= atol + rtol * max(abs(expected), abs(measured))
            )
            agreements.append(agrees)
            probes.append(
                {
                    "direction": i,
                    "step": step,
                    "autograd": expected,
                    "finite_difference": measured if finite else None,
                    "absolute_error": error if finite else None,
                    "relative_error": error / max(1e-12, abs(expected) + abs(measured))
                    if finite
                    else None,
                    "signal_resolved": resolved,
                    "agrees": agrees,
                }
            )
        passed_directions += int(any(a and b for a, b in zip(agreements, agreements[1:])))
    passed = (
        passed_directions == directions
        and noise <= 1e-6
        and grad.abs().max().item() > 0
    )
    return {
        "status": "passed" if passed else "inconclusive",
        "gradient_l2": grad.norm().item(),
        "gradient_max_abs": grad.abs().max().item(),
        "repeat_margin_range": noise,
        "directions_passed": passed_directions,
        "directions": directions,
        "rtol": rtol,
        "atol": atol,
        "finite_difference_probes": probes,
        "note": "Diagnostic agreement is not proof of gradient correctness. Inconclusive results require investigation; inspect precision, step sizes and preprocessing.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_arguments(p)
    data_arguments(p)
    p.set_defaults(dtype="float32")
    p.add_argument("--checkpoint")
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--directions", type=int, default=3)
    p.add_argument(
        "--steps", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="gradient_check.json")
    a = p.parse_args()
    if a.examples < 1 or a.start_index < 0:
        p.error("--examples must be positive and --start-index nonnegative")
    seed_everything(a.seed)
    dataset = VSRDataset(a.data_dir, a.track, "dev", a.limit)
    if a.start_index + a.examples > len(dataset):
        p.error("Requested diagnostic range exceeds development data")
    model = (
        load_checkpoint(a.checkpoint, a.device, a.dtype)
        if a.checkpoint
        else VLMClassifier.load(model_config(a, "zero_shot"))
    )
    model.eval()
    reports = []
    for i in range(a.start_index, a.start_index + a.examples):
        row = dataset[i]
        context = model.context(row["pixels"], row["caption"])
        report = check_gradients(
            lambda x, context=context: model(x, context),
            row["pixels"].unsqueeze(0).to(model.device_),
            steps=a.steps,
            directions=a.directions,
            seed=a.seed + i,
        )
        reports.append({"id": row["id"], "image": row["image"], **report})
        print(f"Example {i}: {report['status']}", flush=True)
    result = {
        "status": "passed"
        if all(r["status"] == "passed" for r in reports)
        else "needs_investigation",
        "examples": reports,
        "environment": environment(),
        "model": vars(model.settings),
        "checkpoint_provenance": checkpoint_provenance(a.checkpoint),
        "source_hashes": source_hashes(),
        "manifest_sha256": dataset.manifest_hash,
        "track": a.track,
        "split": "dev",
        "seed": a.seed,
        "diagnostic_input": "Native RGB float32, clamped to [0.02, 0.98] before probing",
    }
    write_json(a.output, result)
    if result["status"] != "passed":
        raise SystemExit(
            "Gradient diagnostic needs investigation; inspect the saved report."
        )


if __name__ == "__main__":
    main()
