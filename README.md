# Can Quantum Layers offer Performance and Security Advantages in Vision-Language Models?

This repository contains the code for the paper:

> Md Khalid Syfullah and Alvi Ataur Khalil, "Can Quantum Layers offer Performance and Security 
> Advantages in Vision-Language Models?" Computer Science, Southern Illinois
> University Carbondale, USA.

The study uses classically simulated quantum layers inside a vision-language
model (VLM) and measures their effect on (i) predictive accuracy,
(ii) computational cost, and (iii) adversarial robustness. The backbone is
LLaVA-OneVision with its Qwen2-0.5B single-image (SI) variant
(`llava-hf/llava-onevision-qwen2-0.5b-si-hf`, pinned revision
`24efa9ce1ebc62c778946f8c6b119922ea6169bd`). The task is the binary
True/False Visual Spatial Reasoning (VSR) task.

This repository contains code and documentation only. Trained checkpoints,
training logs, attack outputs, and other result files are not included.

## Research questions

- **RQ1:** How does inserting quantum layers at different locations within a
  VLM affect predictive performance and computational cost?
- **RQ2:** Can quantum layers effectively replace classical components while
  maintaining predictive performance and reducing computational cost?
- **RQ3:** How do VLMs with quantum-layer insertion or replacement compare with
  classical baselines in adversarial robustness under different levels of
  attacker access?

## Quantum circuits

Four circuits are implemented, each in its own folder. Every circuit is
simulated exactly with a differentiable PyTorch statevector simulator (no shot
sampling, no hardware noise, no external quantum library).

| Paper name | Folder | Qubits | Trainable layer |
|---|---|---:|---|
| BR4 | `basic-ring-quantum-4qubit/` | 4 | RY on every qubit, then a closed CNOT ring |
| BR8 | `basic-ring-quantum-8qubit/` | 8 | RY on every qubit, then a closed CNOT ring |
| SE4 | `strongly_entangling_4qubits/` | 4 | Rot (RZ, RY, RZ) on every qubit, then CNOTs with range `1 + (layer mod (n − 1))` |
| SE8 | `strongly_entangling_8qubits/` | 8 | Rot (RZ, RY, RZ) on every qubit, then CNOTs with range `1 + (layer mod (n − 1))` |

All circuits use `L = 2` trainable layers by default (`experiment.blocks: 2`).
For a selected token feature `h`, the module computes angles
`z = π·tanh(W_in N(h) + b_in)`, encodes them once with RY rotations, applies the
trainable layers, measures the Pauli-Z expectation of every qubit, and maps the
measurements back with `W_out m + b_out`.

- **Insertion:** `h' = h + γ·g(h)`, with trainable `γ` initialized to 0.1 and
  output weights initialized from Normal(0, 0.01) with zero bias.
- **Replacement:** the module output replaces the component's output. The
  module adds no residual of its own; the decoder's own residual connections
  remain.
- Insertion and projector replacement (r1) use LayerNorm without learnable
  scale and shift. Decoder replacements (r2–r4) reuse the decoder's
  RMSNorm-normalized input.
- Quantum computation runs in FP32 (complex64 for the strongly entangling
  statevector). Its output is cast back to the backbone dtype (BF16 on
  supported CUDA devices).

## Insertion and replacement locations

| Paper | Code | Location | Tokens processed |
|---|---|---|---|
| q1 | `q1` | Before the visual projector | All retained visual tokens |
| q2 | `q2` | After the visual projector, before image/text packing | All visual tokens |
| q3 | `q3` | After decoder block 5 (default `layers // 4 − 1` for 24 blocks) | Final prompt token |
| q4 | `q4` | After final decoder normalization, before the output head | Final prompt token |
| r1 | `r1` | Replaces the whole visual projector (1152 → 896) | All visual tokens |
| r2 | `r2` | Replaces the MLP of decoder block 0 (896 → 896) | All tokens entering that MLP |
| r3 | `r3` | Replaces the MLP of decoder block 11 (896 → 896) | All tokens entering that MLP |
| r4 | `r4` | Replaces the MLP of decoder block 23 (896 → 896) | All tokens entering that MLP |

Insertion is implemented in `quantum_shared.py`, and replacement in
`quantum-replacement-full/`. Each model contains exactly one quantum module.

## Training modes

| Paper | Code mode | Trainable components | Trainable params (M) |
|---|---|---|---:|
| Zero-shot / Full-frozen | `zero_shot`, `full_frozen` | None | 0.000 |
| Head only | `frozen_head` | Two-class linear head initialized from the False/True LM readout rows | 0.002 |
| Projector | `projector` | Visual-to-language projector | 1.837 |
| LoRA | `lora` | LoRA on decoder query and value projections (rank 8, `lora_alpha` 16, i.e. scaling factor alpha/rank = 2; dropout 0.05) | 0.541 |
| Full fine-tuning | `full_unfrozen` | All backbone weights | 893.676 |
| Q only | `quantum_only` | Quantum module (input projection, circuit, output projection, γ) | — |
| Q + head | `quantum_head` | Quantum module and two-class head | — |
| Q + proj. | `quantum_projector` | Quantum module and visual projector | — |
| Q + LoRA | `quantum_lora` | Quantum module and LoRA adapters | — |
| Q + full | `quantum_full` | Quantum module and full backbone | — |
| Replacement | `full_unfrozen` in `quantum-replacement-full/` | Replacement module and all retained backbone weights | — |

Parameter counts for every run are written to `parameters.json`. Replacing a
decoder MLP removes about 13.07 M parameters (893.68 M → about 880.61 M
trainable).

## Training setup

Training settings are in `configs/shared.yaml` (insertion and classical) and
`quantum-replacement-full/config.yaml` (replacement). The default learning-rate
rule is in `experiment_train.py`.

- VSR random split: 7,680 training and 1,097 development examples.
- Seed 0; at most 20 epochs; checkpoint with the lowest development loss.
- AdamW, weight decay 0.01, gradient-norm clipping 1.0, 5% warm-up followed by
  linear decay.
- Microbatch 1 with 16-step gradient accumulation (effective batch size 16).
- Learning rate (`training.lr: null`): 1e-5 for projector and full-backbone
  training (including `quantum_projector`, `quantum_full`, and replacement),
  and 1e-4 for the other trained modes.
- Each run used one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  GPU (96 GB), as recorded in every run's `registry.json`.

**Early-stopping patience.** The configuration default is
`training.patience: 5`. In the paper's runs:

| Runs | Patience |
|---|---:|
| Classical baselines | 5 |
| Insertion `quantum_head`, `quantum_projector`, `quantum_lora`, `quantum_full` (all circuits and positions) | 2 |
| All 16 replacement runs | 2 |
| Insertion `quantum_only`: BR4 q1–q4, BR8 q1–q4, SE4 q1 | 5 |
| Insertion `quantum_only`: SE4 q3–q4, SE8 q1–q4 | 2 |
| Insertion `quantum_only`: SE4 q2 | started at 5, lowered to 2 during training with `tools/train_patience2.py` |

To reproduce a patience-2 run from scratch, add `--set training.patience=2`.

## Attack setup

Attacks are run only on full-backbone models: the 16 `quantum_full` insertion
models, the 16 replacement models, and the classical `full_unfrozen` model
(33 models). Perturbations are applied to the decoded RGB image in [0, 1],
before preprocessing. The text input and model weights stay fixed.

| Attack | Access | Settings |
|---|---|---|
| FGSM | Gradient | One step of size ε |
| PGD-CE | Gradient | 5 random starts × 100 steps, step size α = 2.5ε/100, cross-entropy objective |
| PGD-margin | Gradient | Same as PGD-CE, margin objective |
| SPSA | Scores | 16 pairs of opposite perturbations (radius 0.01) per update, Adam step ε/10, at most 100 updates, 1 restart, 1,000-query budget |
| HSJ | Labels | Foolbox HopSkipJump, up to 32 iterations, 1,000-query budget |
| T-PGD | None (surrogate) | PGD-CE with the same settings on the classical `frozen_head` model, then tested on the target |

- ε = 4/255 (L∞).
- Paper cohort: the first 40 development examples.
- The default in `configs/shared.yaml` is `attacks.limit: 200`. The paper's
  40-example cohort is produced by `tools/attack_full_40.py` for the quantum
  models and by `--set attacks.limit=40` for the classical model.
- Each SPSA update uses 33 queries (16 pairs plus one evaluation of the updated
  image), so the 1,000-query budget allows about 30 updates. In practice the budget, not the 100-update limit, ends each
  SPSA run.

Robust accuracy counts initially incorrect examples as already non-robust.
Gradient diagnostics (`diagnose`) are run before each attack. An inconclusive
diagnostic is recorded as `needs_investigation` in the attack output.

## Evaluation metrics

- Clean accuracy, balanced accuracy (BA), True recall (TR), and False recall
  (FR).
- Robust accuracy (RA) for each attack and for the union of attacks.
- 95% confidence intervals from 1,000 bootstrap resamples grouped by image
  (`evaluation.bootstrap_draws: 1000`).
- Trainable parameters, training time, and peak memory (`resources_*.json`,
  `completion.json`).

## Repository structure

```text
Quantum-Layers-in-VLMs/
  README.md                          # This file
  commands.md                        # Commands to reproduce the paper's experiments
  FROZEN_PREFIX_CACHE.md             # Frozen-prefix cache design for quantum_only q1–q3
  VALIDATION.md                      # Implementation checks performed on the code
  SHA256SUMS.txt                     # Checksums of the delivered files
  requirements.txt                   # Python dependencies
  configs/shared.yaml                # Settings for classical and insertion runs
  experiment.py                      # Main entry point: train, evaluate, diagnose, attack, summarize
  run_matrix.py                      # Runs a matrix of families/positions/modes sequentially
  experiment_config.py               # YAML loading, overrides and run directories
  experiment_model.py                # Classical and quantum model construction, checkpoints
  experiment_train.py                # Training, early stopping and exact resume
  experiment_metrics.py              # Clean evaluation and class-specific metrics
  evaluate.py / compare_results.py   # Evaluation helpers
  model.py / preprocessing.py        # VLM scorer and differentiable pixel preprocessing
  data.py / prepare_data.py          # VSR loading and download
  quantum_shared.py                  # q1–q4 insertion hooks
  feature_cache.py / prefix_cache.py # Verified frozen-feature and frozen-prefix caches
  resource_monitor.py                # CPU, RAM and GPU measurements
  summarize_experiments.py           # CSV, JSON and Markdown summaries
  gradient_check.py                  # Finite-difference pixel-gradient diagnostics
  run_attacks.py                     # Attack execution and saved progress
  attacks/                           # FGSM, PGD, SPSA, HSJ and transfer attacks
  basic-ring-quantum-4qubit/circuit.py
  basic-ring-quantum-8qubit/circuit.py
  strongly_entangling_4qubits/circuit.py
  strongly_entangling_8qubits/circuit.py
  quantum-replacement-full/          # r1–r4 replacement experiments (full-backbone training)
  tools/
    attack_full.py                   # Diagnose and attack the 32 full-backbone quantum models
    attack_full_fast.py              # Same, with exact reuse of transfer candidates
    attack_full_40.py                # Same, fixed to the first 40 examples (paper setting)
    train_patience2.py               # Lowers patience to 2 for unfinished runs, with an audit record
  migrate_prefix_resume.py           # Maintenance: migrates runs started before the prefix cache
  legacy_source_hashes.json          # Source hashes used by migrate_prefix_resume.py
  utils.py / config.py               # Shared utilities
```

Generated folders (`data/`, `runs/`, `feature_cache/`,
`attack_transfer_cache/`, and `quantum-replacement-full/runs/`) are not
included.

## Installation

The paper's runs used PyTorch 2.8.0, torchvision 0.23.0, Transformers 4.57.6,
PEFT 0.17.1, Foolbox 3.3.4, NumPy 2.0.2, Pillow 11.3.0, Accelerate 1.10.1,
safetensors 0.7.0, and CUDA 12.8. These versions are recorded in each run's
`registry.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch 2.8.0 / torchvision 0.23.0 build for your system first.
python -m pip install -r requirements.txt
```

Keep Transformers at 4.57.6. A change from 4.57.1 to 4.57.6 changed checkpoint
behaviour in this project, and runs cannot be resumed across versions.

## Reproducing the experiments

See [commands.md](commands.md) for the complete command sequence. In short:

1. Download VSR with `prepare_data.py`.
2. Train the classical baselines with `run_matrix.py`.
3. Train the 80 insertion models (4 circuits × 4 positions × 5 modes) with
   `run_matrix.py`.
4. Train the 16 replacement models with `quantum-replacement-full/run.py`.
5. Summarize results with `experiment.py summarize` and `run.py summarize`.
6. Diagnose and attack the 33 full-backbone models on the first 40
   development examples.

Run commands from the repository root unless stated otherwise. Paths in
`configs/shared.yaml` are relative to the current working directory. Paths in
`quantum-replacement-full/config.yaml` are relative to that folder.

## Run outputs

Each training run is saved under `runs/<family>/<position>/<mode>/seed0/`
(classical runs under `runs/classical/<mode>/seed0/`; replacement runs under
`quantum-replacement-full/runs/<family>/<position>/seed0/`).

| File or directory | Contents |
|---|---|
| `registry.json` | Resolved configuration, package versions, source hashes, data hashes and split-overlap audit |
| `parameters.json` | Total and trainable parameter counts and trainable parameter names |
| `optimizer.json` | Learning rate, effective batch size and model settings |
| `history.json` | Per-epoch losses, accuracies, learning rate and cumulative time |
| `best/`, `last/` | Best development-loss checkpoint and latest resumable checkpoint |
| `best_development.json` | Per-example development predictions of the best checkpoint |
| `clean_results.json` | Full-path evaluation after reloading the best checkpoint |
| `reload_comparison.json` | Per-example check that the reloaded model reproduces saved predictions |
| `resources_<session>.json` | CPU, RAM and GPU samples for each invocation |
| `completion.json` | Final metrics, timing and reload status |
| `diagnose/`, `attack/` | Gradient diagnostics and per-example attack results |

Training saves a checkpoint every 100 optimizer updates and at the end of every
epoch. An interrupted run resumes with `--resume`. A run is resumed only if its
code, configuration, environment, and data hashes match. Load only your own
`training_state.pt` files, because restoring random-number state uses Python
pickle objects.

## Notes and limitations

- All results come from one seed (seed 0). The development split is used for
  both checkpoint selection and reporting.
- Classical and quantum runs used different early-stopping patience (see
  above), so training-time totals do not establish a speedup.
- The attack cohort is the first 40 development examples (deterministic, not
  random or stratified).
- An unsuccessful attack, an HSJ initialization failure, or an exhausted query
  budget does not by itself establish robustness.
- Circuits are simulated classically; the study does not evaluate quantum
  hardware or privacy attacks.
- Each run's signature includes the SHA-256 hashes of the Python source files
  and the resolved configuration values. Editing a source file or a
  configuration value prevents resuming or reusing runs made before the edit.

## AI Usage Statement

AI tools were used to assist with this repository. Specifically, AI assistance
was used to:

- validate the consistency of the code, configuration, and documentation
  against the paper;
- organize the codebase and repository structure, including separating code
  from result files; and
- edit and format the README and supporting documentation.

All AI-assisted outputs were reviewed and verified by the authors.
