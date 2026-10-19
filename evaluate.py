"""Clean VSR evaluation; test scoring is opt-in."""

import argparse
import time
import torch
from config import (
    model_arguments,
    data_arguments,
    model_config,
    validate_test_access,
    write_json,
)
from compare_results import compare_rows, read_rows
from data import VSRDataset
from model import VLMClassifier, load_checkpoint
from utils import (
    seed_everything,
    accuracy_summary,
    cluster_accuracy_interval,
    parameter_counts,
    environment,
    prepare_output,
    checkpoint_provenance,
    source_hashes,
    input_fingerprint,
)


def evaluate(model, dataset, log_every=10):
    if log_every < 1:
        raise ValueError("log_every must be positive")
    model.eval()
    rows = []
    total_loss = 0.0
    started = time.monotonic()
    print(
        f"Evaluating {len(dataset)} examples on {model.device_} "
        f"({model.dtype_}); preparing the first image...",
        flush=True,
    )
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            ctx = model.context(sample["pixels"], sample["caption"])
            scores = model(sample["pixels"].unsqueeze(0).to(model.device_), ctx)
            label = torch.tensor([sample["label"]], device=scores.device)
            total_loss += torch.nn.functional.cross_entropy(scores, label).item()
            rows.append(
                {
                    "id": sample["id"],
                    **input_fingerprint(sample["pixels"], ctx.prompt),
                    "image": sample["image"],
                    "label": sample["label"],
                    "prediction": scores.argmax(-1).item(),
                    "scores": scores[0].cpu().tolist(),
                    "relation": sample.get("relation"),
                }
            )
            done = i + 1
            if done == 1 or done % log_every == 0 or done == len(dataset):
                elapsed = time.monotonic() - started
                seconds_per_example = elapsed / done
                remaining = seconds_per_example * (len(dataset) - done)
                print(
                    f"Evaluated {done}/{len(dataset)} | "
                    f"{seconds_per_example:.2f} s/example | "
                    f"elapsed {elapsed / 60:.1f} min | estimated remaining {remaining / 60:.1f} min",
                    flush=True,
                )
    return {
        "metrics": dict(accuracy_summary(rows), loss=total_loss / len(rows)),
        "examples": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_arguments(p)
    data_arguments(p)
    p.add_argument("--checkpoint")
    p.add_argument(
        "--reference-clean", help="Compare every evaluated ID against this JSON/JSONL"
    )
    p.add_argument("--mode", choices=["zero_shot", "full_frozen"], default="zero_shot")
    p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.log_every < 1:
        p.error("--log-every must be positive")
    validate_test_access(a.split, a.allow_test)
    seed_everything(a.seed)
    out = prepare_output(a.output)
    a.requested_output = a.output
    a.output = str(out)
    dataset = VSRDataset(a.data_dir, a.track, a.split, a.limit)
    print("Loading model and processor...", flush=True)
    model = (
        load_checkpoint(a.checkpoint, a.device, a.dtype)
        if a.checkpoint
        else VLMClassifier.load(model_config(a))
    )
    start = time.time()
    result = evaluate(model, dataset, log_every=a.log_every)
    result.update(
        elapsed_seconds=time.time() - start,
        environment=environment(),
        parameter_counts=parameter_counts(model),
        settings=vars(model.settings),
        track=a.track,
        split=a.split,
        limit=a.limit,
        seed=a.seed,
        manifest_sha256=dataset.manifest_hash,
        checkpoint=a.checkpoint,
        checkpoint_provenance=checkpoint_provenance(a.checkpoint),
        source_hashes=source_hashes(),
        output=a.output,
        requested_output=a.requested_output,
    )
    result["metrics"]["accuracy_ci95_image_clusters"] = cluster_accuracy_interval(
        result["examples"], seed=a.seed
    )
    write_json(out / "clean_results.json", result)
    print(result["metrics"])
    if a.reference_clean:
        comparison = compare_rows(read_rows(a.reference_clean), result["examples"])
        write_json(out / "reference_comparison.json", comparison)
        if not comparison["passed"]:
            raise SystemExit(
                "Clean reference mismatch; inspect reference_comparison.json (results were saved)."
            )


if __name__ == "__main__":
    main()
