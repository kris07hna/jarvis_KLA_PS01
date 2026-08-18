# Chat Handoff Memory

## Purpose

This document is a secure, self-contained handoff for a new chat or contributor. Read it before making changes. It records confirmed facts and current work state as of 2026-08-16. It deliberately contains no Kaggle API keys, OAuth tokens, passwords, or `kaggle.json` content.

## Immediate Status

The project is implemented and the primary full-quality training run is active on Kaggle. Do not restart or interrupt it solely to change architecture or print frequency.

Current Kaggle kernel:

```text
krishnar15/semiconductor-image-restoration-training
https://www.kaggle.com/code/krishnar15/semiconductor-image-restoration-training
```

Current live run facts:

- Kaggle accelerator: `GPU T4 x2`.
- The training process intentionally uses one T4 (`cuda:0`); the second GPU is idle. Do not add multi-GPU/DDP mid-run.
- PyTorch: `2.10.0+cu128`.
- CUDA available: `True`.
- GPU confirmed: `Tesla T4`.
- Precision: fp16 AMP.
- Training had reached approximately step `7,475 / 17,000`, epoch `44 / 100`, at last reported chat state.
- The notebook runs from `/kaggle/working/semicon2026`.
- Training data root auto-discovered successfully under `/kaggle/input/datasets/krishnar15/semicon2026-private-data`.

## Problem

Restore degraded, grayscale semiconductor inspection arrays at 2x spatial scale.

```text
128x128 LR -> 256x256 GT
256x256 LR -> 512x512 GT
```

Required degradation handling:

- Signal-dependent speckle noise; LR values may be negative or above 1.
- Gaussian-like noise/haze.
- Spatial downsampling / super-resolution.
- Out-of-distribution semiconductor structures.
- Fast, standalone NPY inference.

Official data format is NPY input/output with matching filenames.

## Dataset

Canonical local directories:

```text
train/train/NoisyLR/*.npy
train/train/GT/*.npy
Test_NoisyLR/NoisyLR/*.npy
```

Audited facts:

- 3,200 paired training arrays.
- 400 unpaired test LR arrays.
- LR training shape: `(128, 128)`, `float32`.
- GT training shape: `(256, 256)`, `float32`.
- GT range: `[0, 1]`.
- LR range: approximately `[-0.278563, 2.158005]`.
- LR values outside `[0,1]`: approximately `3.3933%`.
- All audited arrays are finite and all filename pairs match.
- Downsampling appears bicubic-like; bilinear is the strongest simple upsampling reference on noisy data.
- Ignore `__MACOSX`, `._*`, and `.DS_Store` artifacts.

Audit output:

```text
reports/data-audit.json
```

Validation split:

```text
splits/grouped-v1.json
```

- Grouped structurally similar GT images to reduce train/validation leakage.
- Train: 2,720 images.
- Validation: 480 images.
- Do not replace this with a random split for model selection.

## Architecture

Implementation:

```text
src/semicon_restore/models/naf.py
```

Model name: `MultiscaleNAF`.

Primary configuration:

```yaml
name: multiscale_naf
in_channels: 4
width: 64
blocks: [4, 8, 12]
conditioning: true
scale: 2
```

- Parameter count: approximately `8,180,292`.
- Fully convolutional: supports both official spatial sizes.
- Three-level NAF-style encoder-decoder operating mostly in LR feature space.
- Widths: `64 -> 128 -> 256 -> 128 -> 64`.
- NAF blocks use LayerNorm2d, depthwise convolution, SimpleGate, channel attention, and residual scaling.
- No BatchNorm, no transformer, no GAN, no FFT branch, no unsafe standard log transform.

### Input Features

For raw LR input `x`:

```text
raw       = x
bounded   = clamp(x, 0, 1)
overflow  = x - bounded
magnitude = abs(overflow)
```

The model receives `[raw, bounded, overflow, magnitude]`. Raw values are always retained; do not change this to destructive clipping or per-image normalization.

### Output

```text
base       = bilinear_2x(raw)
correction = PixelShuffle2x(model_features)
prediction = base + correction
```

- Loss uses unclamped prediction.
- Metrics and NPY serialization clamp output to `[0,1]`.
- Output must be 2D `float32`, finite, in `[0,1]`, at exact 2x dimensions, with same filename as input.

### Lightweight Conditioning

The model derives mean, std, negative fraction, above-one fraction, overflow magnitude, and gradient energy from the input. An MLP produces scale/shift values for the bottleneck and decoder. This is intentionally lightweight and is not a separate degradation classifier.

## Training

Primary config:

```text
configs/train.yaml
configs/kaggle.yaml
```

Core settings:

```text
Epochs: 100
LR crop: 64x64 -> GT crop 128x128
Optimizer: AdamW
Base LR: 2e-4
Warm-up: 500 optimizer steps
Schedule: cosine decay after warm-up
Weight decay: 1e-4
EMA decay: 0.999
Validation cadence: every 1,000 optimizer steps
Seed: 2026
```

Local RTX 3050:

```text
batch_size=2
gradient_accumulation=8
validation_batch_size=1
```

Kaggle T4:

```text
batch_size=8
gradient_accumulation=2
effective batch=16
validation_batch_size=8
```

The source code computes around 170 optimizer updates per epoch, for 17,000 total updates. The training data loader applies aligned crop, rotation, horizontal flip, and vertical flip. No synthetic degradation is enabled in the active base run (`synthetic_probability=0.0`).

### Loss

```text
0.70 * Charbonnier
+ 0.20 * (1 - SSIM)
+ 0.10 * gradient Charbonnier
```

Do not add GAN or LPIPS training loss mid-run. LPIPS is evaluation-only and still needs final evaluation integration/measurement.

### Anti-overfitting Controls

- Grouped validation split rather than random split.
- EMA validation checkpoint.
- Full-image validation.
- Fixed train/validation manifest.
- Controlled model capacity.
- Paired geometric augmentation.
- AdamW regularization.
- Checkpoints: `latest.pt` and `best.pt` after validation.

## Confirmed Training Results

These are grouped-validation EMA results from the active run:

| Step | Approx. Epoch | PSNR dB | SSIM |
|---:|---:|---:|---:|
| 1,000 | 6 | 24.905 | 0.64733 |
| 2,000 | 12 | 27.093 | 0.75205 |
| 3,000 | 18 | 28.143 | 0.79925 |
| 4,000 | 24 | 28.516 | 0.81505 |
| 5,000 | 30 | 28.677 | 0.82071 |
| 6,000 | 36 | 28.799 | 0.82392 |
| 7,000 | 42 | 28.890 | 0.82606 |

Current interpretation:

- Training is healthy.
- No NaN, Inf, CUDA OOM, or traceback has been reported.
- Validation PSNR and SSIM improved at every recorded evaluation.
- Gains are naturally diminishing, but model is still improving.
- Approximate bilinear training-pair baseline: `24.59 dB` PSNR and `0.595` SSIM.
- At step 7,000 the model improves PSNR by approximately `+4.30 dB` over bilinear under the grouped validation protocol.
- Current score is good, but hidden-test quality cannot be guaranteed because local test GT is unavailable.

Suggested rule: let the active run finish unless it errors or validation regresses persistently. Use final `best.pt`, not automatically `latest.pt`.

## Quality Score Context

Project-specific rough interpretation on this grouped split:

| Level | PSNR | SSIM |
|---|---:|---:|
| Basic | 25-28 dB | 0.70-0.80 |
| Good | 28-30 dB | 0.80-0.85 |
| Strong | 30+ dB | 0.85+ |

These are contextual targets, not official KLA thresholds. Inspect outputs for ringing, halos, oversmoothing, invented periodic structures, and intensity drift before selecting the final checkpoint.

## Local Environment

Workspace:

```text
C:\Users\krish\semicon2026
```

Virtual environment:

```text
C:\Users\krish\semicon2026\.venv
```

Confirmed local stack:

```text
Python: 3.11.15
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CUDA available: True
torchvision: 0.26.0+cu128
kagglehub: 1.0.2
Kaggle CLI: 2.2.4
```

Environment validation passed a real fp16 AMP convolution forward/backward test. The full 8.18M model training graph at batch 2 / LR crop 64 used approximately 541 MB allocated VRAM in the direct local probe.

Activate PowerShell environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
$env:PYTHONPATH = "src"
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Last confirmed local test state: `7 passed`.

## Kaggle Setup

New Kaggle account used by project:

```text
krishnar15
```

Private Kaggle resources:

```text
Kernel:          krishnar15/semiconductor-image-restoration-training
Image dataset:   krishnar15/semicon2026-private-data
Source dataset:  krishnar15/semicon2026-source
Accelerator:     NvidiaTeslaT4
```

Kaggle resources are private. Do not make semiconductor data public without organizer authorization.

### Why Two Datasets

- `semicon2026-private-data`: canonical 6,800 NPY arrays (3,200 LR train + 3,200 GT train + 400 test).
- `semicon2026-source`: zipped/unpacked source code, configs, scripts, splits, and notebook.

`kaggle kernels push` pushes notebook/metadata but does not make the complete local Python package available at runtime. The notebook copies unpacked `semicon2026/` source from the source dataset under `/kaggle/input` into `/kaggle/working/semicon2026` before installation.

### GPU Requirement

Use concrete CLI accelerator value:

```powershell
.\.venv\Scripts\kaggle.exe kernels push -p . --accelerator NvidiaTeslaT4
```

Do not use the installed VS Code Kaggle Studio extension for final GPU pushes. It has a Windows quoting issue (`'kernels'` literal arguments) and can rewrite generic accelerator settings. The project contains a wrapper for extension compatibility, but direct official CLI use is preferred.

The Kaggle notebook preflight prints `nvidia-smi`, CUDA visibility, PyTorch CUDA build, device count, and GPU name. It deliberately fails before training when CUDA is absent.

### Source Update Workflow

Whenever files under `src/`, `scripts/`, `configs/`, `splits/`, or `notebooks/` change, create a new private source-dataset version before pushing the kernel:

```powershell
.\.venv\Scripts\python.exe scripts\upload_kaggle_source.py --execute
.\.venv\Scripts\kaggle.exe kernels push -p . --accelerator NvidiaTeslaT4
```

The upload scripts can derive a KaggleHub token privately from authenticated CLI OAuth when `KAGGLE_API_TOKEN` is absent. They must never print, persist, or commit a token.

### Authentication Security

- Multiple old API/OAuth tokens were accidentally exposed in the original conversation and must be revoked.
- Never paste any token into a new chat.
- Preferred CLI authentication: `kaggle auth login --force`.
- For a current shell only, use `Read-Host` instead of literal token text:

```powershell
$env:KAGGLE_API_TOKEN = Read-Host "Enter Kaggle API token"
```

- Do not commit tokens, `kaggle.json`, `access_token`, or `.env`.

## Important Commands

### Audit and split

```powershell
.\.venv\Scripts\python.exe scripts\audit_data.py --lr-dir train\train\NoisyLR --gt-dir train\train\GT --report reports\data-audit.json
.\.venv\Scripts\python.exe scripts\create_splits.py --lr-dir train\train\NoisyLR --gt-dir train\train\GT --output splits\grouped-v1.json
```

### Train locally

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\train.py --config configs\train.yaml
```

### Resume locally/Kaggle

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\train.yaml --resume checkpoints\latest.pt
```

Kaggle launcher detects `checkpoints/latest.pt` and adds `--resume` automatically. It resumes model, EMA, optimizer, scheduler, AMP scaler, and random states at epoch boundaries.

### Evaluate final checkpoint

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint checkpoints\best.pt --lr-dir train\train\NoisyLR --gt-dir train\train\GT --manifest splits\grouped-v1.json
```

### Generate final test outputs

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\inference.py --input-dir Test_NoisyLR\NoisyLR --output-dir outputs\final --checkpoint checkpoints\best.pt --batch-size 16 --overwrite
.\.venv\Scripts\python.exe scripts\validate_outputs.py --input-dir Test_NoisyLR\NoisyLR --output-dir outputs\final
```

Expected validator result:

```text
Validated 400 output arrays.
```

### Benchmark final checkpoint

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\benchmark.py --input-dir Test_NoisyLR\NoisyLR --checkpoint checkpoints\best.pt --batch-size 16 --warmup 10 --iterations 50 --report reports\final-benchmark.json
```

### Kaggle status/logs

```powershell
.\.venv\Scripts\kaggle.exe kernels status krishnar15/semiconductor-image-restoration-training
.\.venv\Scripts\kaggle.exe kernels logs krishnar15/semiconductor-image-restoration-training
.\.venv\Scripts\kaggle.exe quota
```

## Important Files

```text
src/semicon_restore/models/naf.py       Model implementation
src/semicon_restore/data.py             NPY pairing, aligned crop, augmentation
src/semicon_restore/io.py               Strict safe NPY I/O and atomic outputs
src/semicon_restore/engine.py           Train/validate loop and progress logs
src/semicon_restore/losses.py           Charbonnier/SSIM/gradient objective
src/semicon_restore/checkpoint.py       EMA/checkpoint format
src/semicon_restore/inference.py        Batched standalone NPY restoration
scripts/train.py                        Training CLI
scripts/evaluate.py                     Paired validation CLI
scripts/inference.py                    KLA-style inference CLI
scripts/benchmark.py                    Timing/VRAM benchmark
scripts/validate_outputs.py             Output contract validator
scripts/overfit_smoke.py                Eight-sample overfit gate
scripts/upload_kaggle_dataset.py        Private data uploader
scripts/upload_kaggle_source.py         Private source uploader
scripts/verify_kaggle_project.py        Metadata preflight
notebooks/kaggle_train.ipynb            Thin Kaggle launcher
notebooks/kaggle_bootstrap.py            Kaggle data/source discovery and launcher
configs/train.yaml                      Base training configuration
configs/kaggle.yaml                     Kaggle overrides
kaggle.yml                              Kaggle project settings
kernel-metadata.json                    Kaggle kernel metadata
```

## Completed Verification Gates

- Dataset audit passed 3,200 aligned pairs.
- Grouped split created: 2,720 train / 480 validation.
- Unit/integration tests: 7 passed.
- CUDA fp16 AMP forward/backward test passed locally.
- Full model local training graph tested successfully.
- Eight-image deliberate overfit gate passed:

```text
Initial PSNR: 23.44 dB
Final PSNR:   38.23 dB
Initial Charbonnier loss: 0.05441
Final Charbonnier loss:   0.00955
```

- Local small smoke training passed.
- Checkpoint epoch-boundary resume passed.
- Smoke inference restored all 400 test images and output validator passed.
- Kaggle runtime now confirmed with Tesla T4 x2 and CUDA-enabled PyTorch.

## Known Caveats

- `README.md` calls commands “intended” even though most are now implemented; revise it during final submission polish.
- Current `evaluate.py` reports PSNR/SSIM only; LPIPS evaluation needs to be run/integrated before the final presentation if it is required in reported results.
- The model uses one of two assigned T4 GPUs. This is deliberate; do not add DDP in the middle of a good run.
- Kaggle outputs are session/version artifacts. Download `best.pt`, `latest.pt`, `training.jsonl`, and final reports when the run completes.
- The Kaggle source dataset must be versioned after any code change used by the notebook.
- Training run logs now include `epoch_start`, `train`, `validation_start`, `validation_progress`, and `validation_result`. The active run has this behavior.
- Kaggle `Output 0 B` while a notebook is running is normal; final outputs are published after run completion.

## Next Actions After Current Training Completes

1. Check kernel status and logs; do not use an earlier failed kernel version.
2. Download final Kaggle artifacts, especially `checkpoints/best.pt` and `reports/training.jsonl`.
3. Place/retain final `best.pt` at local `checkpoints/best.pt`.
4. Evaluate the grouped validation checkpoint locally and record PSNR/SSIM.
5. Add/run LPIPS evaluation if required for the report.
6. Run local final inference on all 400 test NPY inputs.
7. Run output validation; require 400 valid output arrays.
8. Run the final-model benchmark. Do not report smoke benchmark numbers as final.
9. Create before/restored/GT visual comparisons with fixed display range.
10. Update README with measured results, exact environment, weights location, final command, and known constraints.
11. Generate `requirements-freeze.txt` from the validated final environment.
12. Complete `documentation/09-submission-checklist.md` and create the KLA PDF/PPT.

## Do Not Do

- Do not expose or record API tokens.
- Do not train with CPU Kaggle runtimes; CUDA preflight must pass.
- Do not modify the active architecture/training hyperparameters mid-run.
- Do not replace grouped validation with random validation.
- Do not use `latest.pt` if `best.pt` has a better grouped validation score.
- Do not use test ground truth, because none is available locally.
- Do not submit smoke model outputs or smoke benchmark results.
- Do not rely on VS Code Kaggle extension for final T4 push; use the official CLI with `--accelerator NvidiaTeslaT4`.
