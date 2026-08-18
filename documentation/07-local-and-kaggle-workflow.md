# Local and Kaggle Workflow

## Responsibility Split

### RTX 3050 6 GB

- Environment and CUDA verification.
- Data audit and split generation.
- Unit and integration tests.
- Eight-sample overfit gate.
- Short training and loss-ablation runs.
- Inference correctness and benchmark development.

### Kaggle GPU

- Main full-dataset training.
- Longer schedules and larger physical batches.
- Final checkpoint evaluation.
- Secondary seed or conservative fine-tuning when time permits.

## Shared Execution

Both environments invoke the same scripts and configuration schema. Kaggle notebooks contain no model classes, dataset implementations, losses, metrics, or checkpoint logic.

```text
notebook/bootstrap
        |
        v
install/import repository
        |
        v
python scripts/train.py --config ...
```

For the VS Code Kaggle extension, sync the repository into the Kaggle working directory and attach the training dataset through the notebook Input panel. Run `notebooks/kaggle_bootstrap.py`; it searches `SEMICON_DATA_ROOT`, the synced repository, `/kaggle/input/semicon2026`, and `/kaggle/input` for the canonical directories. Set custom paths when needed:

```python
import os
os.environ["SEMICON_REPO"] = "/kaggle/working/semicon2026"
os.environ["SEMICON_DATA_ROOT"] = "/kaggle/input/<attached-dataset-name>"
%run /kaggle/working/semicon2026/notebooks/kaggle_bootstrap.py
```

The launcher generates the grouped split only when absent and resumes from `checkpoints/latest.pt` when available. `kagglehub` is optional and is not required when the VS Code extension handles repository synchronization and dataset attachment.

Current Kaggle runtimes may mount private datasets below a versioned path such
as `/kaggle/input/datasets/<owner>/<slug>/versions/<n>`. If the direct paths do
not match, the launcher recursively locates `train/train/NoisyLR` and verifies
the sibling `train/train/GT` directory before deriving the data root.

## Automatic Hardware Policy

At startup, the runtime records GPU model, available memory, bf16 support, and Kaggle presence.

Initial policy:

| Hardware | Precision | Physical batch | Accumulation |
|---|---|---:|---:|
| RTX 3050 6 GB | fp16 | 1-2 | 8-16 |
| Kaggle T4/P100 | fp16 | 4-8 | 2-4 |
| Kaggle L4 | fp16/bf16 after parity check | 8-16 | 1-2 |
| H100 | bf16 preferred | Auto-probed | Usually 1 |
| CPU | fp32 | 1 | Not used for main training |

The effective batch target is approximately 16. OOM recovery halves physical batch size and increases accumulation while preserving the effective batch where practical.

## RTX 3050 Defaults

```yaml
crop_size_lr: 64
batch_size: 2
gradient_accumulation: 8
precision: fp16
validation_batch_size: 1
num_workers: 2
channels_last: true
```

Activation checkpointing is enabled only if the selected model fails the memory probe, because it increases training time.

## Kaggle Session Recovery

- Write `latest.pt` at regular step intervals and before normal shutdown.
- Write best checkpoints after validation improvements.
- Store metrics as append-safe JSON Lines or CSV.
- Resume automatically when the configured checkpoint exists.
- Copy critical artifacts to persistent Kaggle output before session termination.
- Validate split hash and configuration compatibility before resume.

## Data Paths

Paths are configuration values or CLI arguments, never source constants. Local and Kaggle configurations differ only in paths and hardware tuning.

## Failure Recovery

- CUDA OOM: reduce physical batch and retry once after clearing cached allocations.
- Corrupt input: fail data audit before training.
- Interrupted run: resume optimizer step and random states from `latest.pt`.
- Missing Kaggle accelerator: stop clearly instead of silently running a full job on CPU.
- Unsupported precision: fall back to the next validated precision and record the change.
