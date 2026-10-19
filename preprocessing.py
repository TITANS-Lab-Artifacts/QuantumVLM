"""Official tensor image processing, with cached text metadata and live pixel gradients."""

from dataclasses import dataclass
import inspect
import torch

PROMPT = (
    "Determine whether the following statement correctly describes the image. "
    "Answer with exactly True or False.\nStatement: {caption}"
)


@dataclass
class PromptContext:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_ids: tuple
    height: int
    width: int
    prompt: str


def ensure_image_processor_compatibility(processor):
    """Repair OneVision's missing validation key without altering preprocessing.

    Transformers 4.57.1's fast processor injects batch_num_images itself, and
    _preprocess consumes it, but its validation names omit the key. Extend only
    this instance's list when that exact mismatch is present. Do not drop the
    metadata or silence unrelated warnings. Already-correct versions are no-ops.
    """
    image_processor = processor.image_processor
    if image_processor.__class__.__name__ != "LlavaOnevisionImageProcessorFast":
        return
    names = getattr(image_processor, "_valid_kwargs_names", None)
    if names is None or "batch_num_images" in names:
        return
    parameters = inspect.signature(image_processor._preprocess).parameters
    if "batch_num_images" in parameters:
        image_processor._valid_kwargs_names = [*names, "batch_num_images"]


def prepare_context(processor, pixels, caption):
    """Metadata preparation only. Live pixel values are rebuilt for every forward."""
    ensure_image_processor_compatibility(processor)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT.format(caption=caption)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Run the joint processor once on CPU to expand image placeholders correctly.
    inputs = processor(
        text=[prompt],
        images=[pixels.detach().cpu()],
        do_rescale=False,
        return_tensors="pt",
        padding=False,
    )
    tokenizer = processor.tokenizer
    prefix = tokenizer(prompt, add_special_tokens=False).input_ids
    labels = []
    for answer in ("False", "True"):
        full = tokenizer(prompt + answer, add_special_tokens=False).input_ids
        if full[: len(prefix)] != prefix or len(full) == len(prefix):
            raise ValueError(
                "Answer tokenization changes the prompt prefix; register an unambiguous answer boundary."
            )
        labels.append(tuple(full[len(prefix) :]))
    return PromptContext(
        inputs["input_ids"],
        inputs["attention_mask"],
        tuple(labels),
        pixels.shape[-2],
        pixels.shape[-1],
        prompt,
    )


def image_inputs(processor, images, device, dtype):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected a BCHW RGB tensor in [0,1].")
    ensure_image_processor_compatibility(processor)
    processed = processor.image_processor(
        images=list(images.unbind(0)),
        do_rescale=False,
        return_tensors="pt",
        device=device,
    )
    if images.requires_grad and not processed["pixel_values"].requires_grad:
        raise RuntimeError(
            "Image processor detached gradients. Use the pinned fast tensor processor."
        )
    return {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in processed.items()
    }
