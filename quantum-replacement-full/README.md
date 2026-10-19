# Quantum component replacement — full-backbone training

This folder contains the quantum replacement experiments (positions r1–r4) of the paper "Can Quantum Layers Offer an Advantage in Vision-Language Models?". It runs **16 independent replacement experiments**: four circuit families at four positions. All retained VLM parameters and replacement parameters are trainable. **It does not train the classical baseline**; the classical full fine-tuning baseline is trained from the repository root (see the root `commands.md`).

## Location in the repository

This folder must stay inside the repository root, because it imports the shared code from its parent folder:

```text
Quantum-Layers-in-VLMs/
  model.py
  experiment_config.py
  experiment_train.py
  data/vsr/
  basic-ring-quantum-4qubit/circuit.py
  basic-ring-quantum-8qubit/circuit.py
  strongly_entangling_4qubits/circuit.py
  strongly_entangling_8qubits/circuit.py
  quantum-replacement-full/
    run.py
    replacement_model.py
    replacement_metrics.py
    trainer.py
    settings.py
    bootstrap.py
    config.yaml
    README.md
    commands.md
```

The parent folder name does not matter. This folder imports parent code without modifying it. Use the same Python environment as the root: CUDA-enabled Torch 2.8.0, torchvision 0.23.0, Transformers 4.57.6 and PEFT 0.17.1, plus the packages in the root `requirements.txt`. No additional package is required.

**Follow [commands.md](commands.md) in order.**

## What the 16 experiments do

| Position | Actual component removed | Zero-based decoder index | Replacement dimensions in the pinned model |
|---|---|---:|---|
| `r1` | Entire visual-to-language projector | Not applicable | 1152 → 896 |
| `r2` | Entire gated MLP in decoder block 1 | 0 | 896 → 896 |
| `r3` | Entire gated MLP in decoder block 12 | 11 | 896 → 896 |
| `r4` | Entire gated MLP in decoder block 24 | 23 | 896 → 896 |

Each position runs separately with `basic-ring-quantum-4qubit`, `basic-ring-quantum-8qubit`, `strongly_entangling_4qubits`, and `strongly_entangling_8qubits`. A model contains only one replacement. The remaining 23 decoder MLPs stay classical in each decoder replacement experiment.

The component is replaced through module assignment, so its original weights are absent from the optimizer and saved deployed model. The replacement consists of classical compression, `pi*tanh` angle encoding, the existing quantum circuit, exact local-Z readout, and classical expansion. Classical projections are included in its parameter count. R1 applies a fixed non-affine LayerNorm. Decoder replacements receive the existing upstream RMSNorm output and do not normalize a second time. They preserve the original attention and outer residual additions; there is no extra residual inside the replacement.

Replacement projections use ordinary PyTorch Linear initialization, not the small residual-update initialization of the old insertion adapters. Existing circuit angle initialization and gate order are preserved. Two circuit blocks are the default. Each module applies shared weights token by token, including all sequence positions for decoder replacements. With binary final-token scoring, the last block's MLP contributes directly to the loss only through the final prompt position; earlier blocks can also affect it through later attention.

## Starting weights and training

Each independent run starts from the same pinned **original pretrained checkpoint**. It does not start from the fine-tuned classical baseline or from another replacement model. The classical full fine-tuning run from the repository root is the reference for comparison; it is not retrained here.

Defaults are seed 0, full training/development manifests, 20 maximum epochs, patience 5 on development loss (the paper's 16 replacement runs used patience 2: add `--set training.patience=2`), microbatch 1, accumulation 16, AdamW with learning rate 1e-5 for all trainable parameters, weight decay 0.01, gradient norm clipping at 1 and a 5% warmup followed by linear decay. `training.lr: null` chooses the full-unfrozen default of 1e-5. Set a numeric value to override it.

The best checkpoint has the lowest development loss under the configured `min_delta`. The independent final reload must reproduce the saved development scores/predictions within the parent's comparison tolerance before completion is recorded. Original backbone precision, input resolution and eager attention policy are retained. Quantum computation remains FP32/complex64, and its output is cast to the backbone dtype.

**Frozen-feature caching is disabled and rejected if enabled.** The entire backbone changes during training, so cached hidden features would be stale. Optional gradient checkpointing is a memory tradeoff and may increase runtime.

## Configuration

Edit `config.yaml` before starting. `--set section.key=value` overrides one option. Examples: `--set training.epochs=30`, `--set training.lr=0.00001`, or `--set model.gradient_checkpointing=true`.

The only supported mode is `full_unfrozen`. Positions are registered as r1–r4; arbitrary block-index overrides are rejected. Paths always resolve relative to this folder, even if `run.py` is invoked from the parent. The default dataset is `../data/vsr`, and results are saved in this folder's own `runs/` folder. They do not overwrite parent runs.

You may change the family, position or seed via command flags. For a changed training budget or hyperparameter setting, choose a new `output.run_name` or `output.root`; existing run configurations cannot be silently changed while resuming.

## Saved model and files

For example:

```text
quantum-replacement-full/runs/basic-ring-quantum-4qubit/r1/seed0/
  best/
    model.pt
    replacement.json
  last/
    model.pt
    replacement.json
    training_state.pt
    training_state_hash.json
  registry.json
  replacement_architecture.json
  parameters.json
  optimizer.json
  history.json
  best_development.json
  clean_results.json
  reload_comparison.json
  resources_<session>.json
  completion.json
```

`best/model.pt` is the **complete trained state dictionary**, including all retained backbone weights, replacement weights and persistent buffers. It is not an adapter-only checkpoint. `replacement.json` records actual model settings, architecture, configuration, environment and a checksum. Load it with this folder's `replacement_model.load_model`; the parent's original model loader does not understand this different architecture.

Reconstruction uses the pinned original model/processor assets and then strictly loads the complete saved state. Keep the Hugging Face model cache for offline loading. The new checkpoint is not a Hugging Face `save_pretrained` directory and does not bundle tokenizer files.

`last` additionally preserves optimizer, scheduler, random states, data order, cursor and early-stopping state. Saves occur every 100 optimizer updates and at epoch boundaries. After interruption, at most the unsaved updates since the last checkpoint repeat. Pending/previous directories protect save replacement. Trust only your own training-state files, since RNG restoration uses Python objects.

Completed matching runs are reused. Resume requires matching configuration, code, environment and data hashes. A changed parent source file also changes the registered code fingerprint. Source code and dataset files should stay stable during a multi-run job.

Full-backbone checkpoints and optimizer states are large. Best, last and temporary save copies need ample disk space. More frequent saves reduce lost work but increase I/O. Do not delete `last` if you still need exact training resume.

## Measurements

Every run records accuracy, balanced accuracy, loss, per-class recall, predicted-True rate, confusion matrix, per-example scores and final image-cluster confidence intervals. It also records parameters removed/added and actual total size, training/validation time, final evaluation latency/throughput, quantum token evaluations, GPU allocated/reserved peaks, sampled RAM, CPU and whole-device GPU utilization when available.

Resource samples follow the parent definitions: CPU 100% means one logical core; GPU NVML measurements include other processes; sampled peaks may miss brief spikes. Overall session wall time includes loading/saving overhead. Training-step time is reported separately. Final confidence intervals do not include seed-to-seed uncertainty.

`run.py summarize` creates JSON, CSV and Markdown tables from completed runs. It does not rerun models or train a baseline. Compare its results with the classical full-unfrozen baseline trained from the repository root. Note that the paper's classical baseline used patience 5 and the replacement runs used patience 2, so their training times are not directly comparable.

## Scope and verification

This folder implements the 16 quantum replacement configurations only. It does not run classical replacement controls, distillation, gate ablations or additional training modes. The root `run_attacks.py`/`experiment.py` loader cannot load these replacement checkpoints directly. Diagnostics and attacks on the replacement models are run with `tools/attack_full_40.py replacement` from the repository root (see the root `commands.md`), which loads them with `replacement_model.load_model`.

The implementation was checked with small real 24-block LLaVA-OneVision models: component removal, trainable parameters, quantum and image gradients, complete checkpoint reload, training and interrupted-run resume. No synthetic examples, test fixtures or test outputs are included. Source syntax targets Python 3.9; these checks used Python 3.12.
