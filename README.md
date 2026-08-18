# KLA Problem Statement – AI-Based Restoration of Degraded Images
**SEMICON Hackathon 2026 Submission | Team Jarvis (`jarvis_KLA_PS01`)**

State-of-the-art deep learning solution for high-accuracy restoration of low-resolution, high-noise semiconductor inspection images. Our solution utilizes **ContextNAFNet** architecture combined with **8-Fold Self-Ensemble Test-Time Augmentation (TTA)** to perform simultaneous speckle noise removal, Gaussian denoising, and 2x spatial super-resolution.

---

## 🌟 Live Wafer Restoration Demonstration (1 Frame / Sec)

Below is an animated visual demonstration cycling through 10 real semiconductor wafer samples at 1 second per frame (1000ms duration). The animation displays the **Original Low-Res Noisy Input** and the **ContextNAFNet Restored Output** side-by-side:

![SEMICON 2026 Wafer Restoration Animation](reports/wafer_restoration_demo.gif)

---

## 🏗️ Model Architecture Blueprint (ContextNAFNet)

The core architecture, **ContextNAFNet**, is specifically tailored for nanoscale semiconductor inspection imagery. It integrates multi-scale contextual feature extraction with Non-Linear Activation Free (NAF) blocks and pixel-shuffle 2x upscaling:

![ContextNAFNet Wafer Restoration Architecture](reports/architexture.png)

### Key Architectural Highlights
- **Context Feature Fusion**: Global spatial context blocks preserve fine semiconductor line edge structures and contact hole geometry.
- **NAF Blocks (Non-Linear Activation Free)**: Eliminates non-linear activation functions (like ReLU or GELU) in favor of Simple Gate mechanisms, significantly boosting restoration quality and speed.
- **2x PixelShuffle Super-Resolution**: Upscales `128x128 -> 256x256` and `256x256 -> 512x512` without spatial checkerboard artifacts.
- **8-Fold Self-Ensemble TTA**: Rotates ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and flips (horizontal, vertical) test inputs to average out noise residual variance.

---

## 📊 Visual Quality & Residual Loss Benchmark

Visual inspection across diverse wafer patterns showing input degraded images, restored outputs from `models/best.pt`, ground truth targets, and residual error loss heatmaps:

![Real Wafer Sample Input Flow & Restoration Quality Benchmark](reports/github_readme_hero.jpg)

### Detailed Residual Error Heatmaps
Below is the element-wise absolute difference heatmap highlighting the precision of edge and background restoration:

![ContextNAFNet Residual Heatmap](reports/context-naf-comparison-10-heatmap.jpg)

---

## 📈 Quantitative Benchmark Performance

Evaluated across a frozen validation set of 480 semiconductor wafer samples:

| Metric / Benchmark Target | Noisy Input | Restored Output (`models/best.pt`) | Net Improvement |
|:---|:---:|:---:|:---:|
| **Validation Dataset Average PSNR** | 24.88 dB | **29.154 dB** | **+4.27 dB PSNR** |
| **Validation Dataset Average SSIM** | 0.7968 | **0.82486** | **+0.028 SSIM** |
| **Clean Wafer Peak (`000095.npy`)** | 30.53 dB | **35.76 dB** | **+5.23 dB PSNR** |
| **Patterned Wafer (`000048.npy`)** | 25.31 dB | **30.36 dB** | **+5.05 dB PSNR** |
| **Standard Wafer (`001500.npy`)** | 23.56 dB / 16.97 dB SNR | **28.81 dB / 22.22 dB SNR** | **+5.26 dB SNR Gain** |
| **Heavy Noise Wafer (`002060.npy`)** | 21.22 dB / 14.60 dB SNR | **24.17 dB / 17.55 dB SNR** | **+2.95 dB SNR Gain** |

---

## 🚀 Quick Start Execution Command

To restore a directory of low-resolution noisy wafer `.npy` files:

```bash
python run.py <input-dir> <output-dir>
```

### Official Submission Example:
```bash
python run.py Test_NoisyLR/NoisyLR outputs
```

`run.py` autonomously discovers input `.npy` files, runs ContextNAFNet + 8-Fold TTA inference, clamps intensity bounds to `[0.0, 1.0]`, sanitizes `NaN`/`Inf` values, and saves 2x upscaled restored arrays into `<output-dir>` with matching filenames.

---

## 🛠️ Environment Setup & Installation

To guarantee PyTorch installs with **NVIDIA CUDA GPU support**:

```bash
# 1. Install NVIDIA CUDA 12.1 PyTorch wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install project dependencies
pip install -r requirements.txt
```

---

## 📂 Repository Directory Layout

```text
jarvis_KLA_PS01/
├── run.py                 # Primary KLA submission entrypoint
├── requirements.txt       # Dependencies list (PyTorch, NumPy, SciPy, PyYAML, tqdm, Matplotlib)
├── README.md              # Project documentation homepage
├── LICENSE                # MIT License
├── models/
│   └── best.pt            # Pre-trained ContextNAFNet model weights (29.15 dB PSNR)
├── src/
│   └── semicon_restore/   # Core deep learning package
│       ├── models/        # ContextNAFNet architecture definitions
│       ├── inference.py   # Self-ensemble TTA wrapper & batching engine
│       ├── checkpoint.py  # Safe model loading & device allocation
│       └── utils.py       # Metrics (PSNR/SSIM) and array helper utilities
├── documentation/         # Comprehensive design documentation
│   ├── 01-problem-and-requirements.md
│   ├── 02-system-architecture.md
│   ├── 03-data-contract.md
│   ├── 04-training-and-generalization.md
│   ├── 05-evaluation-and-benchmarking.md
│   ├── 06-environment-and-reproducibility.md
│   ├── 07-local-and-kaggle-workflow.md
│   ├── 08-inference-contract.md
│   └── decisions/         # Architectural Decision Records (ADRs)
└── reports/               # Visual evaluation figures, heatmaps, and animation GIFs
    ├── wafer_restoration_demo.gif
    ├── architexture.png
    ├── github_readme_hero.png
    └── context-naf-comparison-10-heatmap.png
```

---

## 📚 Technical Documentation Index

Detailed engineering documentation is available in the [`documentation/`](documentation/README.md) directory:

1. [Problem and Requirements](documentation/01-problem-and-requirements.md) – Problem scope, noise characteristics, resolution downsampling constraints.
2. [System Architecture](documentation/02-system-architecture.md) – End-to-end ContextNAFNet design and feature fusion.
3. [Data Contract](documentation/03-data-contract.md) – Data array schemas, shape upscaling rules, intensity ranges.
4. [Training and Generalization](documentation/04-training-and-generalization.md) – Multi-loss objective (L1 + Charbonnier + FFT High-Frequency Loss), learning rate schedules, data splits.
5. [Evaluation and Benchmarking](documentation/05-evaluation-and-benchmarking.md) – Quantitative benchmark protocols, timing analysis, and metrics.
6. [Inference Contract](documentation/08-inference-contract.md) – Autonomous standalone evaluation contract.
7. [Architectural Decision Records (ADRs)](documentation/decisions/) – Detailed rationale for key design choices.

---

## ✅ Technical Checklist & Requirement Compliance

- [x] **Single-Command Execution**: `run.py <input-dir> <output-dir>` processes all inputs in a single command.
- [x] **Automatic Output Directory Creation**: Missing destination directories are generated automatically.
- [x] **1-to-1 File Mapping**: Produces a corresponding restored `.npy` file for every input with matching filename.
- [x] **Shape Scaling & Format**: Outputs float32 2D arrays with exact 2x spatial super-resolution (`128x128 -> 256x256`, `256x256 -> 512x512`).
- [x] **Value Bounds & Sanitization**: Values strictly clamped within `[0.0, 1.0]`; full NaN/Inf sanitization.
- [x] **100% Offline Capability**: Runs standard PyTorch inference with zero network calls or external API dependencies.
- [x] **Hardware Adaptive**: Automatically leverages CUDA GPU if available, falling back gracefully to CPU.

---

## 📜 License & Citation

Distributed under the MIT License. See [`LICENSE`](LICENSE) for full details.