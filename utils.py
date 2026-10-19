"""Reproducibility, metrics and resource accounting."""

import hashlib
import importlib.metadata
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(value):
    # CPU is the conservative fallback. MPS can be requested explicitly.
    return torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )


def resolve_dtype(value, device):
    if value == "auto":
        return (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    if (
        value == "bfloat16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This GPU does not support bfloat16; use float32.")
    return getattr(torch, value)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def environment():
    out = {}
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "foolbox",
        "numpy",
        "Pillow",
        "accelerate",
        "safetensors",
    ]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    out["cuda_version"] = torch.version.cuda
    out["cuda_device"] = (
        torch.cuda.get_device_name() if torch.cuda.is_available() else None
    )
    return out


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def accuracy_summary(rows, pred_key="prediction"):
    if not rows:
        raise ValueError("No examples were evaluated.")
    n = len(rows)
    correct = sum(r[pred_key] == r["label"] for r in rows)
    class_acc = [
        sum(r[pred_key] == y for r in rows if r["label"] == y)
        / max(1, sum(r["label"] == y for r in rows))
        for y in (0, 1)
    ]
    present = all(any(r["label"] == y for r in rows) for y in (0, 1))
    return {
        "n": n,
        "accuracy": correct / n,
        "balanced_accuracy": sum(class_acc) / 2 if present else None,
    }


def cluster_accuracy_interval(rows, key="prediction", seed=0, draws=1000):
    """Image-cluster bootstrap; this does not include training-seed uncertainty."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r["image"], [0, 0])
        g[0] += int(r[key] == r["label"])
        g[1] += 1
    values = np.array(list(groups.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        picked = values[rng.integers(len(values), size=len(values))].sum(0)
        estimates.append(picked[0] / picked[1])
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def prepare_output(path):
    """Reserve a fresh run directory, adding _001, _002, ... when occupied.

    Empty existing directories are reusable. Existing files and symlinks also
    trigger a suffix. An exclusive marker prevents simultaneous invocations
    from claiming the same empty directory. Old results are never appended to.
    """
    requested = Path(path).expanduser()
    index = 0
    while True:
        candidate = (
            requested
            if index == 0
            else requested.with_name(f"{requested.name}_{index:03d}")
        )
        index += 1
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if not candidate.exists() and not candidate.is_symlink():
                # A parent-path problem is not a collision at this run name.
                raise
            continue
        if candidate.is_symlink() or any(candidate.iterdir()):
            continue
        try:
            with (candidate / ".run_claim").open("x") as marker:
                marker.write(f"Reserved for a new run by process {os.getpid()}\n")
        except FileExistsError:
            continue
        if candidate != requested:
            print(f"Requested output is occupied: {requested}", flush=True)
        print(f"Output directory: {candidate}", flush=True)
        return candidate


def source_hashes():
    root = Path(__file__).resolve().parent
    return {
        str(f.relative_to(root)): sha256(f)
        for f in sorted(root.glob("*.py")) + sorted((root / "attacks").glob("*.py")) + sorted(root.glob("*/circuit.py"))
    }


def checkpoint_provenance(directory):
    if directory is None:
        return None
    import json

    folder = Path(directory)
    info = json.loads((folder / "checkpoint.json").read_text())
    saved_env = info.get("environment")
    if saved_env is None and (folder.parent / "registry.json").exists():
        saved_env = json.loads((folder.parent / "registry.json").read_text()).get(
            "environment"
        )
    current = environment()
    differences = {
        k: {"saved": saved_env.get(k), "current": current.get(k)}
        for k in ("torch", "torchvision", "transformers", "peft")
        if saved_env is not None and saved_env.get(k) != current.get(k)
    }
    return {
        "weights_sha256": sha256(folder / "weights.pt"),
        "metadata_sha256": sha256(folder / "checkpoint.json"),
        "saved_environment": saved_env,
        "environment_differences": differences,
    }


def input_fingerprint(pixels, prompt):
    """Hash decoded pixels and rendered prompt, not just manifest filenames."""
    tensor = pixels.detach().cpu().float().contiguous()
    payload = str(tuple(tensor.shape)).encode() + tensor.numpy().tobytes()
    return {
        "pixels_sha256": hashlib.sha256(payload).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
