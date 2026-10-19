"""Shared placements; circuit implementations remain unchanged."""
import torch
from torch import nn

PLACEMENTS = {
    "q1": "Every retained visual token, before the existing visual projector",
    "q2": "Every projected visual token, before packing and sequence insertion",
    "q3": "Final prompt position after the complete early decoder block",
    "q4": "Final prompt position after final decoder normalization, before LM head",
}


class QuantumVLM(nn.Module):
    def __init__(self, base, placement, residual_class, blocks=2, block_index=None, chunk_size=256):
        super().__init__()
        if placement not in PLACEMENTS:
            raise ValueError("Unknown placement")
        self.base = base
        self.base.eval()
        self.settings = base.settings
        self.bypass = False
        self.device_, self.dtype_ = base.device_, base.dtype_
        self.placement = placement
        decoder = base.backbone.model.language_model
        self.block_index = (
            max(0, len(decoder.layers) // 4 - 1) if block_index is None else block_index
        )
        if not 0 <= self.block_index < len(decoder.layers):
            raise ValueError("Decoder block index out of range")
        projector = base.backbone.model.multi_modal_projector
        width = (
            projector.linear_1.in_features
            if placement == "q1"
            else base.backbone.config.text_config.hidden_size
        )
        self.adapter = residual_class(width, blocks, chunk_size).to(self.device_)
        self.position = None
        self.hook_calls = 0
        if placement == "q1":
            self.handle = projector.register_forward_pre_hook(self._pre)
        elif placement == "q2":
            self.handle = projector.register_forward_hook(self._post)
        elif placement == "q3":
            self.handle = decoder.layers[self.block_index].register_forward_hook(
                self._post
            )
        else:
            self.handle = decoder.norm.register_forward_hook(self._post)

    def _adapt(self, value):
        self.hook_calls += 1
        if self.bypass:
            return value
        if self.placement in ("q1", "q2"):
            return self.adapter(value)
        if self.position is None:
            raise RuntimeError(
                "Use QuantumVLM.forward; prompt position has not been set"
            )
        index = self.position
        updated = self.adapter(value[:, index : index + 1, :])
        return torch.cat((value[:, :index], updated, value[:, index + 1 :]), dim=1)

    def _pre(self, module, args):
        return (self._adapt(args[0]),) + args[1:]

    def _post(self, module, args, output):
        if isinstance(output, tuple):
            return (self._adapt(output[0]),) + output[1:]
        return self._adapt(output)

    def train(self, mode=True):
        super().train(mode)
        self.base.train(mode)
        return self

    def context(self, image, caption):
        ctx = self.base.context(image, caption)
        if any(len(label) != 1 for label in ctx.label_ids):
            raise ValueError(
                "This placement experiment requires single-token False/True continuations"
            )
        return ctx

    def forward(self, image, context):
        self.position = context.input_ids.shape[1] - 1
        self.hook_calls = 0
        result = self.base(image, context)
        if self.hook_calls != 1:
            raise RuntimeError(
                "Expected exactly one adapter insertion per scoring forward"
            )
        return result

    def describe(self):
        return {
            "placement": self.placement,
            "site": PLACEMENTS[self.placement],
            "decoder_block_index": self.block_index if self.placement == "q3" else None,
            "width": self.adapter.width,
            "chunk_size": self.adapter.chunk_size,
            "normalization": "per-token LayerNorm, eps=1e-5, no affine parameters",
            "angles": "pi*tanh(W_in*LN(h)+b_in)",
            "residual": "h + scale*(W_out*<Z>+b_out), cast update to backbone dtype",
            "initialization": "scale=0.1, output weights Normal(0,0.01), output bias=0",
            "trainable": sum(p.numel() for p in self.adapter.parameters()),
            "circuit": (self.adapter.circuit if hasattr(self.adapter, "circuit") else self.adapter.ring).specification(),
        }

