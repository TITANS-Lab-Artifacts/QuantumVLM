# Commands to reproduce the paper's experiments

Run every command from the repository root unless a step says otherwise. All
runs use seed 0 and the settings in `configs/shared.yaml` (classical and
insertion) or `quantum-replacement-full/config.yaml` (replacement). See
`README.md` for what each setting means.

Runs are sequential on one GPU. Full-backbone runs need substantial disk space
for `best/`, `last/`, and optimizer state. Check free space with `df -h .`
before starting.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch 2.8.0 / torchvision 0.23.0 build first.
python -m pip install -r requirements.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

Expected versions: Torch 2.8.0 and Transformers 4.57.6.

## 2. Download the VSR dataset

```bash
python prepare_data.py --data-dir data/vsr --track random --download-images
```

This downloads the VSR random-split annotations (train, dev and test) from
Hugging Face (`cambridgeltl/vsr_random`) and the referenced COCO images for the
train and dev splits into `data/vsr/`. It writes `data/vsr/random/audit.json`.

## 3. Classical baselines (Table II)

```bash
python -u run_matrix.py train --families classical --seeds 0 --resume
```

This runs `zero_shot`, `full_frozen`, `frozen_head`, `projector`, `lora`, and
`full_unfrozen` with the default patience 5. The `frozen_head` checkpoint is
also the surrogate model for the transfer attack.

## 4. Quantum insertion (Figs. 4–5, Table III)

Joint-training modes and full-backbone insertion used patience 2:

```bash
python -u run_matrix.py train \
  --families basic-ring-quantum-4qubit basic-ring-quantum-8qubit strongly_entangling_4qubits strongly_entangling_8qubits \
  --placements q1 q2 q3 q4 \
  --modes quantum_head quantum_projector quantum_lora quantum_full \
  --seeds 0 --set training.patience=2 --resume
```

Quantum-only runs used patience 5 for BR4 q1–q4, BR8 q1–q4 and SE4 q1, and
patience 2 for SE4 q3–q4 and SE8 q1–q4:

```bash
python -u run_matrix.py train \
  --families basic-ring-quantum-4qubit basic-ring-quantum-8qubit \
  --placements q1 q2 q3 q4 --modes quantum_only --seeds 0 --resume

python -u run_matrix.py train \
  --families strongly_entangling_4qubits \
  --placements q1 --modes quantum_only --seeds 0 --resume

python -u run_matrix.py train \
  --families strongly_entangling_4qubits \
  --placements q2 q3 q4 --modes quantum_only --seeds 0 \
  --set training.patience=2 --resume

python -u run_matrix.py train \
  --families strongly_entangling_8qubits \
  --placements q1 q2 q3 q4 --modes quantum_only --seeds 0 \
  --set training.patience=2 --resume
```

In the paper's run, SE4 q2 `quantum_only` was started with patience 5 and
lowered to 2 during training with `tools/train_patience2.py`. Starting it
fresh with patience 2, as above, is not an exact replay of that run.

With `cache.enabled: true`, `quantum_only` at q1–q3 uses the verified
frozen-prefix cache (see `FROZEN_PREFIX_CACHE.md`), and q4 `quantum_only`,
q4 `quantum_head` and classical `frozen_head` use the frozen final-feature
cache. In the paper's runs, the frozen-prefix cache was added after some
basic-ring `quantum_only` runs had started, so those runs were trained fully
or partly without it. This affects only their training time. Before cached training starts, the cached and
original paths are compared on the first three examples of each split. A
mismatch stops the run.

## 5. Quantum replacement (Fig. 5(b), Table III)

```bash
cd quantum-replacement-full
python -u run.py train --all --seed 0 --set training.patience=2 --resume
python -u run.py summarize
cd ..
```

This trains the 16 replacement models (4 circuits × r1–r4) with full-backbone
training. Each starts from the pinned pretrained checkpoint. The summary is
written to `quantum-replacement-full/runs/summary/` (`results.md`, `.csv`,
`.json`).

To train one configuration:

```bash
cd quantum-replacement-full
python -u run.py train --family basic-ring-quantum-4qubit --position r4 --seed 0 \
  --set training.patience=2 --resume
cd ..
```

## 6. Summaries of classical and insertion runs

```bash
python -u experiment.py summarize
```

This writes `runs/summaries/summary.md`, `clean.csv`, `clean.json`,
`attacks.csv`, and `attacks.json`.
Training time and peak memory (Fig. 6) are in each run's `completion.json`,
`history.json`, and `resources_*.json`.

## 7. Adversarial attacks on the 33 full-backbone models (Table IV)

All attacks use ε = 4/255 on the first 40 development examples. The surrogate
for the transfer attack is `runs/classical/frozen_head/seed0/best` from step 3.

### Classical full fine-tuning

A gradient diagnostic must exist for a checkpoint before it can be attacked:

```bash
python -u run_matrix.py diagnose --families classical --modes full_unfrozen --seeds 0 \
  --set attacks.limit=40
python -u run_matrix.py attack --families classical --modes full_unfrozen --seeds 0 \
  --set attacks.limit=40 --resume
```

### Quantum insertion (`quantum_full`) and replacement models

`tools/attack_full_40.py` checks all required files, then runs the diagnostic
and all six attacks for each of the 16 models in a group, with the cohort fixed
at 40 examples:

```bash
python -u tools/attack_full_40.py check
python -u tools/attack_full_40.py insertion
python -u tools/attack_full_40.py replacement
```

Optional arguments: `--device cuda:0`, `--surrogate PATH`, `--data-dir PATH`,
`--insertion-root PATH`, and `--replacement-root PATH`. Do not pass `--limit`;
the launcher fixes it at 40.

Attack results for each model are written under its run folder in
`attack/dev_<hash>/` (`example_*.json`, `summary.json`, and adversarial
tensors).

## 8. Test split (not used in the paper)

The paper reports development results only. Test evaluation is disabled by
default. It needs the test images
(`python prepare_data.py --data-dir data/vsr --track random --download-images --image-splits test`)
and both settings below:

```bash
python -u experiment.py evaluate --set evaluation.split=test --set evaluation.allow_test=true \
  --set experiment.family=classical --set experiment.mode=lora --set experiment.placement=q4
```

## Resuming

Repeat the same command with `--resume` after an interruption. Training resumes
from the last saved checkpoint (every 100 optimizer updates and at each epoch
end). Completed runs with matching code, configuration, environment, and data
are reused. Do not change settings while resuming. Use a new `output.root` or
`output.run_name` for a changed experiment.

`migrate_prefix_resume.py` is a maintenance tool for runs that were started
with the code version before the frozen-prefix cache. It is not needed for a
fresh reproduction.
