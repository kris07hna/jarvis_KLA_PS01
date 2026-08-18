# Technical Documentation Index

This directory serves as the engineering design and operational reference for **Team Jarvis (`jarvis_KLA_PS01`)**.

---

## 📐 Core System Design

| Document | Description / Focus Area |
|---|---|
| **[01 - Problem and Requirements](01-problem-and-requirements.md)** | Scope, semiconductor noise characteristics, resolution downsampling constraints, and deliverables. |
| **[02 - System Architecture](02-system-architecture.md)** | End-to-end ContextNAFNet design, range-preserving feature construction, and 2x PixelShuffle upscaling. |
| **[03 - Data Contract](03-data-contract.md)** | Input/output `.npy` array schemas, shape scaling rules, and float32 range constraints. |
| **[04 - Training and Generalization](04-training-and-generalization.md)** | Multi-loss objective (Charbonnier + FFT Frequency Loss), learning rate schedules, and data splitting. |
| **[05 - Evaluation and Benchmarking](05-evaluation-and-benchmarking.md)** | Quantitative benchmark protocols (PSNR, SSIM, SNR), timing analysis, and quality gates. |

---

## 🛠️ Operations & Execution

| Document | Description / Focus Area |
|---|---|
| **[06 - Environment and Reproducibility](06-environment-and-reproducibility.md)** | PyTorch CUDA GPU environment capture, dependency pinning, and random seed control. |
| **[07 - Local and Kaggle Workflow](07-local-and-kaggle-workflow.md)** | Hardware-adaptive execution, training checkpoints, and session recovery. |
| **[08 - Inference Contract](08-inference-contract.md)** | Standalone evaluator behavior, CLI contract (`run.py`), and error handling safeguards. |
| **[09 - Submission Checklist](09-submission-checklist.md)** | KLA problem statement requirements, verification status, and checklist compliance. |
| **[10 - Execution Roadmap](10-execution-roadmap.md)** | Ordered project implementation roadmap and training milestones. |

---

## 🎯 Architectural Decision Records (ADRs)

- **[ADR-001: Compact Multiscale NAF Architecture](decisions/ADR-001-model-architecture.md)** – Rationale for Non-Linear Activation Free blocks and multi-scale context fusion.
- **[ADR-002: Range-Preserving Input Representation](decisions/ADR-002-input-representation.md)** – Handling non-standard input intensity overflow during feature extraction.
- **[ADR-003: Structure-Grouped Validation](decisions/ADR-003-validation-split.md)** – Grouped splitting strategy for out-of-distribution wafer validation.

---

## 📑 Source Material

- **[Original Project Proposal (`plan.md`)](../plan.md)** – Exploratory research and initial project proposal blueprint.
