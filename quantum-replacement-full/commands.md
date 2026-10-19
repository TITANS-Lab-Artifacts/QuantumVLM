# Commands — full-unfrozen quantum replacement

These commands train and evaluate the 16 replacement models. They use the VSR data prepared from the repository root (`../data/vsr`) and do not train the classical baseline.

## 1. Enter this folder

```bash
cd quantum-replacement-full
```

## 2. Activate the environment

Use the environment created for the repository root.

```bash
python -c "import torch, transformers; print('Torch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available())"
```

Use Transformers 4.57.6, the same as the root runs. Do not upgrade dependencies midway through this experiment.

## 3. Check paths and configuration

```bash
ls ../model.py ../experiment_train.py ../basic-ring-quantum-4qubit/circuit.py
ls ../data/vsr/random/train.jsonl ../data/vsr/random/dev.jsonl
cat config.yaml
df -h .
```

Default training uses all examples, seed 0, up to 20 epochs and patience 5. The paper's replacement runs used patience 2, which the commands below set with `--set training.patience=2`. The 16 full-backbone runs are sequential and can take substantial time and disk space.

## 4. Run all 16 configurations

```bash
python -u run.py train --all --set training.patience=2 --resume
```

This executes basic-ring 4-qubit r1/r2/r3/r4, basic-ring 8-qubit r1/r2/r3/r4, strongly-entangling 4-qubit r1/r2/r3/r4, then strongly-entangling 8-qubit r1/r2/r3/r4. Each model trains the full retained backbone and replacement module. There is no original-baseline training entry.

`--resume` is safe on the first launch. Repeating this exact command skips completed matching runs, resumes an interrupted run from its latest saved update, and then continues the remaining configurations. The launcher stops if any run fails; inspect that error before relaunching.

The script automatically evaluates and reload-verifies the best checkpoint for each run. At the end it generates summary tables.

## 5. Read or regenerate the summary

```bash
python -u run.py summarize
cat runs/summary/results.md
```

Detailed results: `runs/summary/results.csv` and `runs/summary/results.json`. Compare these with the classical full-unfrozen results trained from the repository root.

## 6. Train or resume one configuration instead

Use this instead of step 4 if you want to run a single model first. It is a full experiment, not a smoke test.

```bash
python -u run.py train \
  --family basic-ring-quantum-4qubit \
  --position r1 \
  --seed 0 \
  --set training.patience=2 \
  --resume
```

A second example, last decoder MLP with strongly entangling 8-qubit circuit:

```bash
python -u run.py train \
  --family strongly_entangling_8qubits \
  --position r4 \
  --seed 0 \
  --set training.patience=2 \
  --resume
```

Running step 4 later reuses these completed runs if their settings match.

## 7. Evaluate the saved best model without training

Training already produces a full development evaluation. This command is for a separate saved-model evaluation when needed:

```bash
python -u run.py evaluate \
  --family basic-ring-quantum-4qubit \
  --position r1 \
  --seed 0
```

The default checkpoint is `runs/basic-ring-quantum-4qubit/r1/seed0/best/`. To explicitly select it:

```bash
python -u run.py evaluate \
  --family basic-ring-quantum-4qubit \
  --position r1 \
  --checkpoint runs/basic-ring-quantum-4qubit/r1/seed0/best
```

The complete trained weights are in `best/model.pt`. Keep `best/replacement.json` with them. Standalone evaluation results are under the selected run's `evaluation/` directory.

## 8. Add another seed later

```bash
python -u run.py train --all --seed 1 --set training.patience=2 --resume
python -u run.py train --all --seed 2 --set training.patience=2 --resume
python -u run.py summarize
```

Keep `--set training.patience=2` to match the paper's replacement protocol.

These create separate seed directories and retain seed 0. Only add them when ready for another 16 training runs per seed.

## 9. Change the budget only for a new experiment

For example, a fresh 30-epoch matrix saved in a separate root:

```bash
python -u run.py train --all --resume \
  --set training.epochs=30 \
  --set output.root=runs_30epochs
```

Repeat the same overrides to resume or summarize that matrix:

```bash
python -u run.py summarize --set output.root=runs_30epochs
```

Do not change epochs while attempting exact resume of an existing 20-epoch run; its scheduler depends on the original budget.

## 10. Final test evaluation after locking the protocol

Ensure test images are available in `../data/vsr/images` (download them from the repository root with `prepare_data.py --image-splits test`). Then, for one selected model:

```bash
python -u run.py evaluate \
  --family basic-ring-quantum-4qubit \
  --position r1 \
  --set evaluation.split=test \
  --set evaluation.allow_test=true
```

To evaluate all 16 already-trained best checkpoints on test:

```bash
python -u run.py evaluate --all \
  --set evaluation.split=test \
  --set evaluation.allow_test=true
```

These commands do no training. Do not tune settings on test results. The primary summary table reports the completed training runs' development results; final test JSON files remain in each run's `evaluation/test_<hash>/clean_results.json`.
