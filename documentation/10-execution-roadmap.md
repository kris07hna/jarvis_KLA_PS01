# Execution Roadmap

## Principle

The deadline reduces speculative experiments, not correctness, generalization controls, or submission reliability. Work proceeds through gates so expensive training starts only after the pipeline is proven.

## Phase 0: Foundation

- [x] Audit repository and dataset.
- [x] Establish problem, architecture, data, training, benchmark, and submission documentation.
- [x] Create Python 3.11 virtual environment.
- [x] Install and verify CUDA-enabled PyTorch on RTX 3050.
- [x] Add package metadata, dependency files, and configurations.

Exit gate: a real CUDA AMP forward/backward test passes locally.

## Phase 1: Data and Validation

- [x] Implement strict NPY discovery and loading.
- [x] Implement pair and range audit.
- [x] Implement structure descriptors and grouped split creation.
- [x] Persist audit report and split manifest.
- [x] Add data-contract tests.

Exit gate: all 3,200 pairs pass validation and no similarity group crosses the frozen split.

## Phase 2: Model and Training Core

- [x] Implement range-preserving input features.
- [x] Implement NAF blocks and multiscale architecture.
- [x] Implement losses and quality metrics.
- [x] Implement EMA and versioned resumable checkpoints.
- [x] Implement hardware-adaptive AMP and gradient accumulation.
- [x] Add shape, gradient, determinism, and resume tests.

Exit gate: the model overfits eight samples and an epoch-boundary checkpoint resumes with model, optimizer, scheduler, scaler, and random states intact.

## Phase 3: Main Training

- [x] Run short RTX 3050 smoke training.
- [ ] Launch main Kaggle training on original pairs.
- [ ] Monitor full-image grouped validation.
- [ ] Retain best PSNR, SSIM, combined, and latest checkpoints.
- [ ] Inspect severe-noise and high-gradient examples.

Exit gate: grouped validation materially beats bilinear without inspection-risk artifacts.

## Phase 4: Focused Improvement

Only one high-value change is evaluated at a time:

1. Range-aware channels versus raw-only input.
2. Lightweight degradation conditioning.
3. Conservative synthetic degradation fine-tuning.

FFT branches, large teachers, ensembles, and test-time augmentation are deferred unless the primary model completes early and a measured failure justifies them.

Exit gate: an addition is retained only when grouped validation improves, severe cases do not regress, and latency remains acceptable.

## Phase 5: Inference and Benchmark

- [x] Implement standalone inference CLI.
- [x] Implement output validator.
- [x] Implement layered benchmark with correct CUDA synchronization.
- [x] Validate both official input sizes.
- [x] Validate local CUDA path.
- [x] Generate all supplied test outputs from the smoke checkpoint.

Exit gate: clean-process read-to-write inference succeeds for the complete test directory.

## Phase 6: Release

- [ ] Freeze exact dependencies.
- [ ] Publish or link final weights.
- [ ] Complete README commands and measured results.
- [ ] Produce figures and presentation metrics.
- [ ] Run the complete submission checklist.
- [ ] Perform independent clean-checkout review.

## Deferred Backlog

- Explicit wavelet or FFT residual branch.
- Teacher/student distillation.
- ONNX or TensorRT export.
- Arbitrary-resolution tiled inference.
- Multi-fold grouped validation.
- Larger-scale pretraining.
