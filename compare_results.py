"""Compare clean predictions by ID, including attack JSONL logs; no GPU required."""

import argparse
import json
import math
from pathlib import Path
from config import write_json


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    return json.loads(path.read_text())["examples"]


def compare_rows(reference, current, score_atol=1e-4):
    if score_atol < 0 or not math.isfinite(score_atol):
        raise ValueError("score_atol must be finite and nonnegative")

    def index(rows):
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate example IDs in comparison")
        return result

    ref, cur = index(reference), index(current)
    if not cur:
        raise ValueError("Current results are empty")
    missing = sorted(set(cur) - set(ref))
    differences = []
    scores_compared = 0
    max_score_error = 0.0
    for key in sorted(set(cur) & set(ref)):
        a, b = ref[key], cur[key]
        reasons = []
        for field in ("label", "image"):
            if a[field] != b[field]:
                reasons.append(field)
        for field in ("pixels_sha256", "prompt_sha256"):
            if field in a and field in b and a[field] != b[field]:
                reasons.append(field)
        pa = a.get("clean_prediction", a.get("prediction"))
        pb = b.get("clean_prediction", b.get("prediction"))
        if pa is None or pb is None:
            raise ValueError("Missing prediction")
        if pa != pb:
            reasons.append("prediction")
        sa = a.get("clean_scores", a.get("scores"))
        sb = b.get("clean_scores", b.get("scores"))
        if sa is not None and sb is not None:
            scores_compared += 1
            if len(sa) != len(sb) or not all(math.isfinite(x) for x in sa + sb):
                reasons.append("invalid_scores")
            else:
                error = max(abs(x - y) for x, y in zip(sa, sb))
                max_score_error = max(error, max_score_error)
                if error > score_atol:
                    reasons.append("scores")
        if reasons:
            differences.append(
                {
                    "id": key,
                    "reasons": reasons,
                    "reference_prediction": pa,
                    "current_prediction": pb,
                }
            )
    return {
        "passed": not missing and not differences,
        "n_current": len(cur),
        "n_reference": len(ref),
        "missing_ids": missing,
        "differences": differences,
        "scores_compared": scores_compared,
        "max_absolute_score_error": max_score_error,
        "score_atol": score_atol,
        "scope": "Every current ID must match; unused reference IDs are allowed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_rows(
        read_rows(args.reference), read_rows(args.current), args.score_atol
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
