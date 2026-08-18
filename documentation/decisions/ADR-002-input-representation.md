# ADR-002: Range-Preserving Input Representation

## Status

Accepted for initial implementation and matched ablation.

## Context

Degraded values extend below zero and above one because of the noise process. A standard logarithm is undefined for negative values, while clipping destroys potentially useful evidence about degradation severity.

## Decision

Retain the raw input and derive three additional channels:

```text
bounded   = clamp(x, 0, 1)
overflow  = x - bounded
magnitude = abs(overflow)
```

The model receives `[raw, bounded, overflow, magnitude]`.

## Rationale

- Raw values preserve all measured information.
- Bounded values provide a stable clean-range view.
- Signed overflow distinguishes dark and bright excursions.
- Magnitude exposes severity without requiring the model to derive an absolute value indirectly.
- The representation is cheap and valid for all finite real inputs.

## Consequences

- The first convolution accepts four channels.
- Checkpoints encode the feature representation version.
- A raw-only parameter-matched model is required to quantify benefit.
- Per-image normalization and destructive input clipping remain prohibited.

## Alternatives Considered

- Standard log: rejected because inputs may be negative.
- Shifted log: rejected because the shift is arbitrary and changes the effective physical model.
- Asinh auxiliary channel: deferred as an ablation if current channels underperform severe cases.
