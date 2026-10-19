"""Experiment settings shared by training, evaluation and checkpoints."""

from dataclasses import dataclass
import json
from pathlib import Path

MODES = (
    "zero_shot",
    "full_frozen",
    "frozen_head",
    "projector",
    "lora",
    "full_unfrozen",
)


@dataclass
class ModelConfig:
    model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    revision: str = "main"
    mode: str = "zero_shot"
    device: str = "auto"
    dtype: str = "auto"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False


def model_arguments(parser):
    parser.add_argument("--model-id", default=ModelConfig.model_id)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--device", default="auto", help="auto, cuda, cuda:0, cpu or mps"
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "bfloat16"], default="auto"
    )


def data_arguments(parser):
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--track", choices=["random", "zeroshot"], default="random")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot-only first N examples; logged in results",
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def model_config(args, mode=None):
    return ModelConfig(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        mode=mode or args.mode,
    )


def validate_test_access(split, allow_test):
    if split == "test" and not allow_test:
        raise ValueError(
            "Test access requires --allow-test after the experiment and attacks are locked."
        )
