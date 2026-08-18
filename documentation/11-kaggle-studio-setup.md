# Kaggle Studio Setup

## Purpose

The VS Code Kaggle Studio extension uploads and runs `notebooks/kaggle_train.ipynb` on Kaggle GPU infrastructure. The notebook is only a launcher; model, data, training, validation, and checkpoint logic remain in repository Python modules.

## Authentication

Use either:

1. Command Palette: `Kaggle: Sign In`, then enter your Kaggle username and API key.
2. Store the legacy API file at `%USERPROFILE%\.kaggle\kaggle.json` on Windows.

Never commit API keys, `kaggle.json`, or access tokens. Run `Kaggle: Check API Status` before uploading data or launching a notebook.

## Project Metadata

The project metadata is configured for Kaggle user `krishnar15`:

```text
krishnar15/semiconductor-image-restoration-training
```

The notebook is private by default and GPU/internet access are enabled.

## Private Dataset

The semiconductor dataset should be uploaded as a private Kaggle dataset unless KLA explicitly authorizes public redistribution.

The local guarded uploader creates KaggleHub datasets as private resources and includes only canonical arrays:

```powershell
.\.venv\Scripts\python.exe scripts\upload_kaggle_dataset.py
.\.venv\Scripts\python.exe scripts\upload_kaggle_dataset.py --execute
```

The first command is a dry run. The second stages hard links where supported and uploads `krishnar15/semicon2026-private-data`. Authentication must be completed separately and tokens must never be passed as command-line arguments.

The uploaded dataset must preserve this structure:

```text
train/train/NoisyLR/*.npy
train/train/GT/*.npy
Test_NoisyLR/NoisyLR/*.npy
```

The private dataset is configured as:

```text
krishnar15/semicon2026-private-data
```

The upload has completed and Kaggle exposes the canonical `.npy` files. The
dataset remains private.

The notebook code is distributed through a second private dataset because a
Kaggle kernel push uploads the configured notebook, not the complete local
Python package:

```text
krishnar15/semicon2026-source
```

Create or update it after source changes:

```powershell
.\.venv\Scripts\python.exe scripts\upload_kaggle_source.py
.\.venv\Scripts\python.exe scripts\upload_kaggle_source.py --execute
```

The source uploader includes `src/`, `scripts/`, `configs/`, `splits/`, and
`notebooks/`. Therefore the degradation generator, curriculum configuration,
robustness evaluator, and Kaggle launcher are all shipped in this artifact.

Kaggle normally expands the uploaded source archive into a read-only
`semicon2026/` tree under `/kaggle/input`. The notebook copies that tree into
`/kaggle/working/semicon2026` before installing and launching training. Direct
ZIP extraction remains a fallback for resource layouts that preserve the
archive.

After upload:

1. Run `Kaggle: Attach Dataset`.
2. Select the private dataset.
3. Confirm its handle is added under `datasets` in `kaggle.yml` and `dataset_sources` in `kernel-metadata.json` if the extension does not update both automatically.
4. Keep the dataset private.

## Run

Before pushing, validate the synchronized metadata:

```powershell
.\.venv\Scripts\python.exe scripts\verify_kaggle_project.py
```

1. Open `notebooks/kaggle_train.ipynb`.
2. Run `Kaggle: Run Current Notebook` or click the rocket button.
3. Monitor the run under the extension's Runs view.
4. Download outputs when complete.

If the Kaggle run page reports `Accelerator: None`, stop that run. Open the
notebook on Kaggle, choose **Settings** or **Session options**, select a GPU
accelerator, and save/run a new version. If no GPU option is available, check
Kaggle phone verification and GPU quota. Do not allow full training to run on
CPU.

For this project, use the official CLI rather than the VS Code extension for
the GPU push. The extension's accelerator selector can rewrite the concrete
machine shape back to a generic CPU/GPU value. From the project root run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_kaggle_project.py
.\.venv\Scripts\kaggle.exe kernels push -p . --accelerator NvidiaTeslaT4
```

The second command uploads the launcher notebook and immediately starts a new
private Kaggle version. Check it and download its artifacts with:

```powershell
.\.venv\Scripts\kaggle.exe kernels status krishnar15/semiconductor-image-restoration-training
.\.venv\Scripts\kaggle.exe kernels output krishnar15/semiconductor-image-restoration-training -p kaggle-output -o
```

The current launcher trains experiment
`context-naf-noise-curriculum-v1`. It composes, in order:

1. `configs/context-naf-from-scratch.yaml` for the complete model and optimizer.
2. `configs/kaggle.yaml` for T4 batch size, workers, epochs, and crop curriculum.
3. `configs/context-naf-noise-curriculum.yaml` for progressive synthetic noise.

The final synthetic share is 10% for the first 20% of training, 30% through
70% of training, and 10% for the final refinement phase. Official validation
continues to use only the real grouped LR/GT pairs.

The T4 runtime also enables native PyTorch CUDA acceleration: fp16 autocast,
cuDNN algorithm autotuning, `channels_last` convolution layout, fused AdamW,
four loader workers, and four-batch worker prefetching. Training logs report
`samples_per_second`; compare this only at the same crop size because 128x128
crops require roughly four times the pixel work of 64x64 crops.

The official CLI supports concrete accelerator IDs. `NvidiaTeslaT4` is the
preferred default here; do not use `NvidiaTeslaP100` with the default Kaggle
CUDA image.

Expected artifacts for the current experiment:

```text
checkpoints/context-naf-noise-curriculum-v1/latest.pt
checkpoints/context-naf-noise-curriculum-v1/best.pt
reports/context-naf-noise-curriculum-v1/training.jsonl
configs/kaggle-context-runtime.yaml
```

The runtime launcher discovers the attached dataset automatically. If discovery fails, set `SEMICON_DATA_ROOT` in the notebook to the exact `/kaggle/input/<dataset-slug>` path.

## Windows Extension Compatibility

The installed Kaggle Studio extension version constructs Windows commands with
literal single quotes around arguments. For example, it can pass `"'kernels'"`
to the CLI instead of `kernels`, which causes an `invalid choice` error. The
workspace setting `kaggle.cliPath` therefore points to
`scripts/kaggle_cli_wrapper.py`, which strips only the extension's outer quote
pair and forwards the cleaned arguments to `.venv/Scripts/kaggle.exe`.

After changing the setting, run `Developer: Reload Window` from the VS Code
Command Palette and then `Kaggle: Check CLI Status`.

## Resume

Kaggle inputs are read-only and `/kaggle/working` is session-local. To resume
across separate runs, publish or upload `latest.pt` as a private Kaggle model
or dataset, attach it to the notebook, and copy it to
`checkpoints/context-naf-noise-curriculum-v1/latest.pt` in the writable
repository before executing the bootstrap. The source uploader intentionally
does not package checkpoints.

The launcher automatically adds `--resume` when the experiment-specific
`latest.pt` exists in the writable repository copy.
