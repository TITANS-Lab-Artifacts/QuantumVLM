"""Evaluate known pixel attacks under explicit model-knowledge assumptions."""

import argparse
import json
from pathlib import Path
import time
import torch
from config import (
    model_arguments,
    data_arguments,
    model_config,
    validate_test_access,
    write_json,
)
from data import VSRDataset
from model import VLMClassifier
from experiment_model import load_model as load_checkpoint, provenance
from utils import (
    seed_everything,
    prepare_output,
    environment,
    source_hashes,
    checkpoint_provenance,
    input_fingerprint,
)
from evaluate import evaluate
from compare_results import compare_rows, read_rows
from attacks.common import validate_candidate
from attacks.oracles import ScoreOracle, DecisionOracle
from attacks.white_box import fgsm, pgd
from attacks.score_based import spsa
from attacks.decision_based import hop_skip_jump
from attacks.transfer import transfer_pgd

KNOWLEDGE = {
    "fgsm": "full / white-box",
    "pgd_ce": "full / white-box",
    "pgd_margin": "full / white-box",
    "spsa": "partial / score-query",
    "hsj": "no internals / decision-query",
    "transfer": "partial / surrogate",
}


def summarize(rows, names):
    n = len(rows)
    if n == 0:
        raise ValueError("No attack results")
    clean = sum(r["clean_correct"] for r in rows)
    robust = sum(r["robust_correct"] for r in rows)
    result = {
        "n": n,
        "clean_accuracy": clean / n,
        "empirical_robust_accuracy_union": robust / n,
        "attack_success_rate_on_clean_correct": (clean - robust) / clean
        if clean
        else None,
        "clean_correct_denominator": clean,
        "per_attack": {},
    }
    for name in names:
        failures = sum(
            r["clean_correct"] and r["attacks"][name]["success"] for r in rows
        )
        result["per_attack"][name] = {
            "empirical_robust_accuracy": (clean - failures) / n,
            "conditional_attack_success_rate": failures / clean if clean else None,
            "attack_calls": sum(r["attacks"][name]["calls"] for r in rows),
            "attack_call_type": "surrogate" if name == "transfer" else "target",
            "successful_attacks": failures,
            "clean_correct_denominator": clean,
            "statuses": {
                s: sum(r["attacks"][name]["status"] == s for r in rows)
                for s in sorted({r["attacks"][name]["status"] for r in rows})
            },
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_arguments(p)
    data_arguments(p)
    p.add_argument("--checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--objective", choices=["untargeted","false_acceptance","false_rejection"], default="untargeted")
    p.add_argument(
        "--reference-clean",
        help="Clean JSON reference; defaults to training development results for dev checkpoints",
    )
    p.add_argument("--surrogate-checkpoint")
    p.add_argument(
        "--attacks",
        nargs="+",
        choices=list(KNOWLEDGE),
        default=["fgsm", "pgd_ce", "pgd_margin", "spsa", "hsj"],
    )
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument(
        "--epsilon",
        type=float,
        default=4 / 255,
        help="L_inf radius in native [0,1] pixels",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--step-size", type=float)
    p.add_argument("--query-budget", type=int, default=1000)
    p.add_argument("--spsa-samples", type=int, default=16)
    p.add_argument("--spsa-delta", type=float, default=0.01)
    p.add_argument("--hsj-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    validate_test_access(a.split, a.allow_test)
    if not 0 <= a.epsilon <= 1 or a.query_budget < 1 or a.steps < 1 or a.restarts < 1:
        p.error("Invalid attack budget")
    if len(set(a.attacks)) != len(a.attacks):
        p.error("List each attack once")
    if "transfer" in a.attacks:
        if not a.surrogate_checkpoint:
            p.error("--surrogate-checkpoint is required for transfer")
        if (
            a.checkpoint
            and Path(a.checkpoint).resolve() == Path(a.surrogate_checkpoint).resolve()
        ):
            p.error("Transfer requires a different surrogate checkpoint")
    seed_everything(a.seed)
    out = Path(a.output) if a.resume else prepare_output(a.output)
    out.mkdir(parents=True,exist_ok=True)
    a.requested_output = a.output
    a.output = str(out)
    dataset = VSRDataset(a.data_dir, a.track, a.split, a.limit)
    model = (
        load_checkpoint(a.checkpoint, a.device, a.dtype)
        if a.checkpoint
        else VLMClassifier.load(model_config(a, "zero_shot"))
    )
    model.eval()
    surrogate = (
        load_checkpoint(a.surrogate_checkpoint, a.device, a.dtype)
        if "transfer" in a.attacks
        else None
    )
    if surrogate:
        surrogate.eval()
    registry = {
        **vars(a),
        "data_dir": str(a.data_dir),
        "model": vars(model.settings),
        "source_hashes": source_hashes(),
        "checkpoint_provenance": provenance(a.checkpoint),
        "surrogate_provenance": provenance(a.surrogate_checkpoint)
        if surrogate
        else None,
        "environment": environment(),
        "knowledge": {name: KNOWLEDGE[name] for name in a.attacks},
        "manifest_sha256": dataset.manifest_hash,
        "query_budget_scope": "per example per query attack; clean and final verification calls are separately reported",
        "domain": "native decoded RGB [0,1], before all deterministic crops/resizing/normalization",
        "query_domain": "HSJ and SPSA may query outside the epsilon ball; returned candidates must be inside it",
        "surrogate_model": vars(surrogate.settings) if surrogate else None,
    }
    reference = a.reference_clean
    if reference is None and a.checkpoint and a.split == "dev":
        for filename in ("best_development.json", "development.json"):
            candidate = Path(a.checkpoint).parent / filename
            if candidate.exists():
                reference = str(candidate)
                break
    registry["resolved_reference_clean"] = reference
    registry["attack_seed_policy"] = "seed + example_index*1009 + fixed_attack_id*9176"
    write_json(out / "registry.json", registry)
    provenance_info = registry["checkpoint_provenance"]
    if provenance_info and provenance_info["environment_differences"]:
        print(
            "Checkpoint environment differs: "
            + json.dumps(provenance_info["environment_differences"]),
            flush=True,
        )
    # Validate the whole attack cohort before spending attack queries.
    preflight_calls = 0
    if reference:
        clean = evaluate(model, dataset)
        preflight_calls = len(clean["examples"])
        write_json(out / "preflight_clean.json", clean)
        comparison = compare_rows(read_rows(reference), clean["examples"])
        write_json(out / "reference_comparison.json", comparison)
        if not comparison["passed"]:
            raise SystemExit(
                "Clean reference mismatch: attacks were not started. Inspect reference_comparison.json; re-evaluate and resolve the environment/checkpoint difference."
            )

    rows = []
    start = time.time()
    verification_calls = 0
    new_attack_target_calls = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["pixels"].unsqueeze(0).to(model.device_)
        label = sample["label"]
        ctx = model.context(sample["pixels"], sample["caption"])
        predict = lambda x, ctx=ctx: model(x, ctx)
        with torch.no_grad():
            clean_scores = predict(image)
            verification_calls += 1
        clean_pred = clean_scores.argmax(-1).item()
        record = {
            "id": sample["id"],
            **input_fingerprint(sample["pixels"], ctx.prompt),
            "image": sample["image"],
            "label": label,
            "clean_prediction": clean_pred,
            "clean_scores": clean_scores[0].float().cpu().tolist(),
            "clean_correct": clean_pred == label,
            "robust_correct": clean_pred == label,
            "attacks": {},
        }
        if reference:
            repeated = compare_rows([clean["examples"][i]], [record])
            if not repeated["passed"]:
                write_json(out / "repeatability_failure.json", repeated)
                raise SystemExit(
                    "Clean prediction changed after preflight; inspect repeatability_failure.json"
                )
        partial = out / ("example_%06d.json" % i)
        if a.resume and partial.exists():
            previous=json.loads(partial.read_text())
            for field in ('id','pixels_sha256','prompt_sha256','clean_prediction'):
                if previous[field]!=record[field]: raise ValueError('Attack resume input/prediction mismatch')
            record=previous
        for name in a.attacks:
            if name in record['attacks']: continue
            j = list(KNOWLEDGE).index(name)
            # Reproducible independent attack randomness, unaffected by preceding success.
            seed_everything(a.seed + i * 1009 + j * 9176)
            eligible = a.objective == "untargeted" or label == (0 if a.objective == "false_acceptance" else 1)
            if clean_pred != label or not eligible:
                record["attacks"][name] = {
                    "success": False,
                    "calls": 0,
                    "status": "skipped_clean_incorrect" if clean_pred != label else "skipped_other_target_class",
                    "prediction": clean_pred,
                }
                continue
            t = time.time()
            if name == "fgsm":
                result = fgsm(predict, image, label, a.epsilon)
            elif name.startswith("pgd_"):
                result = pgd(
                    predict,
                    image,
                    label,
                    a.epsilon,
                    a.steps,
                    a.restarts,
                    a.step_size,
                    loss=name.split("_")[1],
                )
            elif name == "spsa":
                oracle = ScoreOracle(predict, image, label, a.epsilon, a.query_budget)
                result = spsa(
                    oracle,
                    image,
                    label,
                    a.epsilon,
                    a.steps,
                    a.spsa_samples,
                    a.spsa_delta,
                    a.step_size,
                )
            elif name == "hsj":
                oracle = DecisionOracle(
                    predict, image, label, a.epsilon, a.query_budget
                )
                result = hop_skip_jump(oracle, image, label, a.epsilon, a.hsj_steps)
            elif name == "transfer":
                sctx = surrogate.context(sample["pixels"], sample["caption"])
                result = transfer_pgd(
                    lambda x, sctx=sctx: surrogate(x, sctx),
                    image.to(surrogate.device_),
                    label,
                    epsilon=a.epsilon,
                    steps=a.steps,
                    restarts=a.restarts,
                    step_size=a.step_size,
                )
                result.image = result.image.to(model.device_)
            if name != "transfer": new_attack_target_calls += result.calls
            distance = validate_candidate(result.image, image, a.epsilon)
            with torch.no_grad():
                adv_scores = predict(result.image)
                verification_calls += 1
            prediction = adv_scores.argmax(-1).item()
            success = prediction != label
            record["robust_correct"] &= not success
            record["attacks"][name] = {
                "success": success,
                "prediction": prediction,
                "scores": adv_scores[0].float().cpu().tolist(),
                "linf": distance,
                "calls": result.calls,
                "call_type": "surrogate" if name == "transfer" else "target",
                "status": result.status,
                "seconds": time.time() - t,
                "details": result.details,
            }
            if a.save_images:
                folder = out / "adversarial_tensors"
                folder.mkdir(exist_ok=True)
                # Float tensors preserve the actual bound; PNG re-quantization can change it.
                torch.save(result.image.detach().cpu(), folder / f"{i:06d}_{name}.pt")
            # Persist each finished attack so interruptions do not repeat expensive queries.
            from experiment_train import atomic_json
            atomic_json(partial,record)
        rows.append(record)
        from experiment_train import atomic_json
        atomic_json(partial,record)
        print(
            f"{i + 1}/{len(dataset)}; clean={record['clean_correct']}; union robust={record['robust_correct']}",
            flush=True,
        )
    with open(out / "examples.jsonl", "w") as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")
    result = {
        "metrics": summarize(rows, a.attacks),
        "seconds": time.time() - start,
        "clean_and_final_verification_target_calls": verification_calls,
        "preflight_target_calls": preflight_calls,
        "target_calls_this_invocation_completed_attacks": preflight_calls + verification_calls + new_attack_target_calls,
        "target_attack_calls_unique_completed_attacks": sum(
            r["attacks"][name]["calls"]
            for r in rows
            for name in a.attacks
            if name != "transfer"
        ),
        "interpretation": "Empirical robustness against this searched candidate union; not certified robustness.",
    }
    from utils import cluster_accuracy_interval
    groups={'white_box':['fgsm','pgd_ce','pgd_margin'],'score_query':['spsa'],'decision_query':['hsj'],'transfer':['transfer']}
    result['access_specific_unions']={}
    for group,methods in groups.items():
        methods=[n for n in methods if n in a.attacks]
        if not methods: continue
        selected=[{**r,'survived':r['clean_correct'] and not any(r['attacks'][n]['success'] for n in methods)} for r in rows]
        result['access_specific_unions'][group]={'attacks':methods,'robust_accuracy':sum(r['survived'] for r in selected)/len(selected)}
    # Bootstrap correctness indicators, clustered by source image.
    indicators=[{'image':r['image'],'label':1,'prediction':int(r['robust_correct'])} for r in rows]
    result['metrics']['robust_accuracy_ci95_image_clusters']=cluster_accuracy_interval(indicators,seed=a.seed)
    recalls=[sum(r['robust_correct'] for r in rows if r['label']==y)/sum(r['label']==y for r in rows) if any(r['label']==y for r in rows) else None for y in (0,1)]
    result['metrics']['robust_balanced_accuracy_union']=sum(recalls)/2 if all(x is not None for x in recalls) else None
    result['constraint_violations']=0
    result['constraint_policy']='Invalid candidates abort rather than enter the summary.'
    result['objective']=a.objective
    if a.objective != 'untargeted':
        source_label=0 if a.objective=='false_acceptance' else 1
        eligible_rows=[r for r in rows if r['label']==source_label and r['clean_correct']]
        result['targeted']={'source_label':source_label,'target_label':1-source_label,
          'clean_correct_source_denominator':len(eligible_rows),
          'per_attack_success_rate':{name:sum(r['attacks'][name]['success'] for r in eligible_rows)/len(eligible_rows) if eligible_rows else None for name in a.attacks},
          'note':'For this binary task, attacking the opposite label is targeted; algorithms are shared with untargeted evaluation.'}
    result['seconds_scope']='This invocation; per-attack seconds in examples include reused attacks.'
    result['verification_calls_scope']='This invocation only; use resources session files for cumulative costs.'
    write_json(out / "summary.json", result)
    print(result["metrics"])


if __name__ == "__main__":
    main()
