# ADR-001: Compact Multiscale NAF Architecture

## Status

Accepted for initial implementation.

## Context

The task combines denoising and fixed 2x super-resolution, with only 3,200 paired grayscale samples. Accuracy, out-of-distribution behavior, H100 throughput, and reproducibility are all evaluated. Development hardware is an RTX 3050 with 6 GB VRAM, while full training can use Kaggle GPUs.

The original proposal combined a degradation encoder, log transform, NAF backbone, frequency branch, FiLM at every stage, and distillation before establishing a baseline.

## Decision

Implement a three-level, approximately 8-12M parameter NAF-style encoder-decoder that operates in LR feature space and predicts a PixelShuffle 2x residual over bilinear interpolation. Add lightweight degradation conditioning at the bottleneck and decoder only.

## Rationale

- NAF-style blocks are proven restoration primitives with favorable compute cost.
- Multiscale features provide context for haze and structure without global FFT cost.
- Residual reconstruction improves optimization and preserves stable low frequencies.
- LR-domain processing supports RTX 3050 training and fast H100 inference.
- Capacity is meaningful but controlled for the effective dataset size.

## Consequences

- One fully convolutional checkpoint supports both official resolutions.
- FFT and distillation remain ablations rather than dependencies.
- Training must use grouped validation to detect structural memorization.
- PixelShuffle output channels and scale factor remain fixed at 2x.

## Alternatives Considered

- Full DAF-Net: rejected initially due to combined optimization, latency, and negative-input log-transform risks.
- Lightweight SwinIR: deferred due to higher implementation and latency complexity.
- Shallow CNN only: retained as a diagnostic fallback but considered insufficient as the primary quality model.
