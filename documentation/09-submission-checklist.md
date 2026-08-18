# Submission Checklist & Verification Status

## Repository & Setup
- [x] **Public Repository**: Accessible at `https://github.com/kris07hna/jarvis_KLA_PS01`.
- [x] **Comprehensive README**: Includes Quick Start, virtual environment setup, CUDA PyTorch installation, evaluation commands, loss math, code comments, and directory layout.
- [x] **Standalone Submission Entrypoint**: `run.py <input-dir> <output-dir>` handles input/output directories autonomously.
- [x] **Model Weights Included**: Pre-trained ContextNAFNet model weights present at `models/best.pt` (29.15 dB PSNR).
- [x] **Dependencies Frozen**: Requirements pinned with CUDA 12.1 index-url in `requirements.txt`.
- [x] **Clean Credentials**: Zero API keys, local credentials, or private machine paths committed.

## Inference Contract & Output Verification
- [x] **File Discovery**: Discovers and processes all `.npy` files in specified `<input-dir>`.
- [x] **Filename Preservation**: Outputs saved with 1-to-1 matching filenames into `<output-dir>`.
- [x] **Data Type & Shape**: Exports 2D `float32` arrays scaled to exactly 2x spatial resolution (`128x128 -> 256x256`, `256x256 -> 512x512`).
- [x] **Value Bounds & Clamping**: Values strictly clamped within `[0.0, 1.0]` range.
- [x] **NaN/Inf Sanitization**: Hard-asserted zero `NaN` or `Inf` values in output arrays.
- [x] **100% Offline Inference**: Evaluates completely offline with zero internet access or API dependency.
- [x] **Hardware Adaptive**: Auto-detects NVIDIA CUDA GPU acceleration with graceful CPU fallback.

## Results & Quality Benchmarks
- [x] **Average PSNR Gain**: **+4.27 dB PSNR** improvement on 480 validation wafers (24.88 dB -> 29.154 dB).
- [x] **Average SSIM Gain**: **+0.028 SSIM** improvement (0.7968 -> 0.82486).
- [x] **Clean Wafer Peak**: **35.76 dB PSNR** (+5.23 dB gain on `000095.npy`).
- [x] **Patterned Wafer Peak**: **30.36 dB PSNR** (+5.05 dB gain on `000048.npy`).
- [x] **Visual Evidence & Figures**: Side-by-side animated demonstration GIF (`reports/wafer_restoration_demo.gif`), architecture blueprint (`reports/architexture.png`), hero panel (`reports/github_readme_hero.jpg`), and residual loss heatmaps (`reports/context-naf-comparison-10-heatmap.jpg`).

## Presentation & Paper Materials
- [x] IEEE Technical Paper Architecture writeup available in `documentation/`.
- [x] Architectural Decision Records (`ADR-001` through `ADR-003`) documented in `documentation/decisions/`.
- [x] Master documentation index available in `documentation/README.md`.
