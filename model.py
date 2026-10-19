"""PyTorch LLaVA-OneVision binary scorer and explicit freezing policies."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from config import ModelConfig, MODES, write_json
from preprocessing import image_inputs, prepare_context
from utils import (
    resolve_device,
    resolve_dtype,
    parameter_counts,
    environment,
    source_hashes,
    sha256,
)


class VLMClassifier(nn.Module):
    def __init__(self, backbone, processor, config):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.settings = config
        self.device_ = next(backbone.parameters()).device
        self.dtype_ = next(backbone.parameters()).dtype
        self.head = None
        self.head_labels = None
        self.configure_training()

    @classmethod
    def load(cls, config):
        from transformers import (
            AutoConfig,
            AutoProcessor,
            LlavaOnevisionForConditionalGeneration,
        )

        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        hf_config = AutoConfig.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False
        )
        if hf_config.model_type != "llava_onevision":
            raise ValueError(
                "This implementation audits LLaVA-OneVision only; other VLMs need a separate wrapper."
            )
        revision = getattr(hf_config, "_commit_hash", None) or config.revision
        config = replace(
            config,
            revision=revision,
            device=str(device),
            dtype=str(dtype).split(".")[-1],
        )
        processor = AutoProcessor.from_pretrained(
            config.model_id, revision=revision, use_fast=True, trust_remote_code=False
        )
        if not processor.image_processor.__class__.__name__.endswith("Fast"):
            raise RuntimeError(
                "A fast tensor image processor is required for differentiable pixel attacks."
            )
        backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.model_id,
            revision=revision,
            config=hf_config,
            dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=False,
        ).to(device)
        backbone.config.use_cache = False
        return cls(backbone, processor, config)

    def configure_training(self):
        mode = self.settings.mode
        if mode not in MODES:
            raise ValueError(mode)
        self.backbone.requires_grad_(False)
        if mode == "full_unfrozen":
            self.backbone.requires_grad_(True)
        elif mode == "projector":
            self.backbone.model.multi_modal_projector.requires_grad_(True)
        elif mode == "frozen_head":
            self.head = nn.Linear(
                self.backbone.config.text_config.hidden_size, 2, dtype=torch.float32
            ).to(self.device_)
        elif mode == "lora":
            from peft import LoraConfig, inject_adapter_in_model

            # Restrict adapters to the language decoder, excluding the vision tower.
            targets = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
            cfg = LoraConfig(
                r=self.settings.lora_rank,
                lora_alpha=self.settings.lora_alpha,
                lora_dropout=self.settings.lora_dropout,
                target_modules=targets,
                bias="none",
            )
            self.backbone = inject_adapter_in_model(cfg, self.backbone)
        if self.settings.gradient_checkpointing and mode not in (
            "zero_shot",
            "full_frozen",
            "frozen_head",
        ):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.settings.mode in ("zero_shot", "full_frozen", "frozen_head"):
            self.backbone.eval()
        return self

    def context(self, image, caption):
        ctx = prepare_context(self.processor, image, caption)
        if self.head is not None:
            if any(len(ids) != 1 for ids in ctx.label_ids):
                raise ValueError(
                    "Frozen-head initialization requires single-token labels; use another mode for sequence scoring."
                )
            ids = tuple(x[0] for x in ctx.label_ids)
            if self.head_labels is None:
                with torch.no_grad():
                    self.head.weight.copy_(
                        self.backbone.lm_head.weight[list(ids)].float()
                    )
                    self.head.bias.zero_()
                self.head_labels = ids
            elif ids != self.head_labels:
                raise ValueError("Label tokenization varies across prompts.")
        return ctx

    def _hidden(self, images, context, suffix=()):
        if images.shape[0] != 1:
            raise ValueError(
                "Use microbatch size 1; gradient accumulation supplies the effective batch size."
            )
        if tuple(images.shape[-2:]) != (context.height, context.width):
            raise ValueError("Image shape changed after prompt preparation.")
        visual = image_inputs(self.processor, images, self.device_, self.dtype_)
        ids = context.input_ids.to(self.device_)
        mask = context.attention_mask.to(self.device_)
        if suffix:
            ids = torch.cat(
                [ids, torch.tensor([suffix], device=self.device_, dtype=torch.long)],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.ones((1, len(suffix)), device=self.device_, dtype=mask.dtype),
                ],
                dim=1,
            )
        return self.backbone.model(
            input_ids=ids,
            attention_mask=mask,
            **visual,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def forward(self, images, context):
        """Returns [False, True] scores. No detach/no_grad on the image computation."""
        if self.head is not None:
            if self.head_labels is None:
                raise RuntimeError("Call model.context before scoring the frozen head.")
            return self.head(self._hidden(images, context)[:, -1].float())
        if all(len(ids) == 1 for ids in context.label_ids):
            hidden = self._hidden(images, context)[:, -1]
            ids = [x[0] for x in context.label_ids]
            return F.linear(hidden, self.backbone.lm_head.weight[ids]).float()
        # Sequence log likelihood: every answer token contributes, including its prefix.
        scores = []
        n = context.input_ids.shape[1]
        for answer in context.label_ids:
            hidden = self._hidden(images, context, answer[:-1])[
                :, n - 1 : n - 1 + len(answer)
            ]
            logp = self.backbone.lm_head(hidden).float().log_softmax(-1)
            indices = torch.tensor(answer, device=self.device_).view(1, -1, 1)
            scores.append(logp.gather(-1, indices).squeeze(-1).sum(-1))
        return torch.stack(scores, dim=1)

    def save_checkpoint(self, directory, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        params = {
            n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad
        }
        # Includes all parameters in full_unfrozen, and only the learned part otherwise.
        temporary = directory / "weights.tmp"
        torch.save(params, temporary)
        temporary.replace(directory / "weights.pt")
        write_json(
            directory / "checkpoint.json",
            {
                "model": asdict(self.settings),
                "environment": environment(),
                "source_hashes": source_hashes(),
                "weights_sha256": sha256(directory / "weights.pt"),
                "head_labels": self.head_labels,
                "parameter_counts": parameter_counts(self),
                "trainable_names": list(params),
                "metadata": metadata or {},
            },
        )

    def restore_weights(self, directory):
        directory = Path(directory)
        info = json.loads((directory / "checkpoint.json").read_text())
        if (
            info.get("weights_sha256")
            and sha256(directory / "weights.pt") != info["weights_sha256"]
        ):
            raise ValueError(
                "Checkpoint weights hash mismatch; files may be mixed or corrupted"
            )
        state = torch.load(
            directory / "weights.pt", map_location="cpu", weights_only=True
        )
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected or set(info["trainable_names"]) != expected:
            raise ValueError(
                "Checkpoint parameter set does not match the configured training mode."
            )
        named = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != named[name].shape
                ):
                    raise ValueError(f"Checkpoint tensor shape mismatch: {name}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite checkpoint tensor: {name}")
                named[name].copy_(value)
        self.head_labels = tuple(info["head_labels"]) if info["head_labels"] else None
        self.eval()
        return info


def load_checkpoint(directory, device="auto", dtype=None):
    info = json.loads((Path(directory) / "checkpoint.json").read_text())
    cfg = ModelConfig(**info["model"])
    cfg.device = device
    if dtype and dtype != "auto":
        cfg.dtype = dtype
    cfg.gradient_checkpointing = False
    model = VLMClassifier.load(cfg)
    model.restore_weights(directory)
    return model
