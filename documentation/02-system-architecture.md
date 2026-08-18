# System Architecture

## System Flow

```text
.npy input directory
        |
        v
strict discovery and validation
        |
        v
float32 grayscale tensor [B,1,H,W]
        |
        v
range-preserving feature construction
        |
        +----> bilinear 2x reconstruction base
        |
        v
multiscale degradation-aware NAF restorer
        |
        v
PixelShuffle 2x correction
        |
        v
base + learned correction
        |
        v
clamp for serialization -> float32 .npy [2H,2W]
```

## Model Input

Given raw input `x`, construct four channels:

```text
raw       = x
bounded   = clamp(x, 0, 1)
overflow  = x - bounded
magnitude = abs(overflow)
```

The raw channel remains authoritative. Derived channels expose where and by how much the observation violates the clean-image range, which is useful evidence of speckle severity.

## Restoration Backbone

The primary model is a three-level NAF-style encoder-decoder operating mostly in the low-resolution domain.

Initial capacity target:

| Stage | Width | Blocks | Purpose |
|---|---:|---:|---|
| Input | 64 | 1 convolution | Map four input channels into features |
| Encoder 1 | 64 | 4 | Local denoising and structure extraction |
| Encoder 2 | 128 | 8 | Mid-scale context |
| Bottleneck | 256 | 12 | Global degradation correction context |
| Decoder 2 | 128 | 8 | Recover mid-scale structure |
| Decoder 1 | 64 | 4 | Recover local detail |
| Output | 4 | 1 convolution | PixelShuffle 2x residual correction |

The default configuration contains approximately 8.18 million parameters. A smaller width/depth configuration remains available for smoke tests if the RTX 3050 memory probe requires it.

## NAF-Style Block

Each block uses normalized residual processing, depthwise convolution, SimpleGate feature interaction, channel attention, and learnable residual scaling. BatchNorm is excluded because restoration depends on preserving absolute intensity statistics.

## Degradation Conditioning

A lightweight descriptor summarizes each input using differentiable statistics such as mean, standard deviation, out-of-range prevalence, overflow magnitude, and gradient energy. A small multilayer perceptron produces conditioning values for the bottleneck and decoder.

Conditioning is intentionally limited to a few stages. This preserves adaptive restoration without the cost and overfitting risk of injecting a large degradation encoder into every block.

## Reconstruction Head

The network predicts a high-resolution correction rather than the full output:

```text
base = bilinear(x, scale_factor=2)
correction = pixel_shuffle(head(features), 2)
prediction = base + correction
```

Training losses operate on the unclamped prediction. Metrics and serialized outputs use `clamp(prediction, 0, 1)`.

## Dynamic Resolution

The architecture contains no fixed-size linear layers over spatial dimensions. Inputs divisible by the encoder reduction factor are processed directly, allowing one checkpoint to restore both official resolutions.

## Excluded Initial Components

- Standard log transform: invalid for negative LR values.
- Global FFT branch: deferred until a matched ablation proves quality and latency value.
- GAN discriminator: unacceptable hallucination risk for inspection images.
- Output sigmoid: can saturate corrections near clean-range boundaries.
- Test-time augmentation or ensembles: direct throughput penalty.

## Scalability

- Training scales through crops, AMP, gradient accumulation, and checkpoint resume.
- Inference scales through batching, pinned memory, and asynchronous transfers.
- The model remains at LR resolution until its final head, avoiding expensive HR feature processing.
- Optional tiling may be added for larger-than-specified images, but is not part of the official path initially.
