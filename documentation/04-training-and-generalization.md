# Training and Generalization

## Objectives

Train a high-capacity restoration model without memorizing repeated semiconductor structures or over-sharpening uncertain detail.

## Loss Function

Initial candidate:

```text
L = 0.70 * Charbonnier(pred, target)
  + 0.20 * (1 - MS-SSIM(pred, target))
  + 0.10 * GradientCharbonnier(pred, target)
```

Losses operate on unclamped predictions. The initial smoke run also trains with Charbonnier alone so structural-loss bugs or ringing can be isolated quickly.

LPIPS and adversarial losses are excluded from initial optimization. LPIPS remains an evaluation metric; GAN loss is excluded because fabricated inspection structure is unacceptable.

## Sampling

- Use aligned LR/GT random crops with exact 2x coordinates.
- Default LR crop is `64x64`; corresponding GT crop is `128x128`.
- Mix uniform sampling with edge-aware sampling so flat areas and fine structures are both represented.
- Apply paired horizontal flips, vertical flips, and 90-degree rotations.
- Keep validation deterministic and full-image.

## Optimization

Initial defaults:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Schedule | Warm-up followed by cosine decay |
| Effective batch | Approximately 16 |
| Precision | Hardware-adaptive AMP |
| EMA | `0.999` initial decay |
| Gradient clipping | Disabled unless instability is measured |

Training is step-based so changing physical batch size between RTX 3050 and Kaggle does not change the intended schedule.

## Anti-Overfitting Controls

- Structure-grouped validation split.
- Full-image checkpoint selection.
- Weight decay and EMA.
- Controlled model capacity.
- Conservative geometric augmentation.
- Training/validation metric divergence monitoring.
- Severity-stratified validation.
- No repeated tuning against the final untouched validation subset.
- Two-seed confirmation when compute permits.

Patch count is never reported as independent dataset size. The effective structural sample count remains bounded by the original images and their similarity groups.

## Training Gates

1. Bilinear baseline metrics are recorded.
2. One batch completes forward and backward passes in fp32 and AMP.
3. The model deliberately overfits eight samples, proving alignment and capacity.
4. A short grouped-validation run improves over bilinear.
5. The main run proceeds only after checkpoint resume is verified.
6. Fine-tuning or architecture additions are accepted only when grouped validation improves without visible artifacts.

## Checkpoint Policy

Every resumable checkpoint stores:

- Model and EMA state dictionaries.
- Optimizer and scheduler states.
- AMP scaler state.
- Epoch, optimizer step, and samples seen.
- Python, NumPy, CPU torch, and CUDA random states. Epoch-boundary resumes are deterministic; a mid-epoch resume restarts the current epoch because DataLoader permutation state is not serialized.
- Complete resolved configuration.
- Split-manifest path and hash.
- Best metrics and checkpoint-selection rule.
- Package, Python, CUDA, GPU, and source-version metadata.

Retain `latest`, `best_psnr`, `best_ssim`, and `best_combined` checkpoints.

## Conservative Degradation Augmentation

Synthetic degradation is a second-stage experiment, not a default replacement for real pairs. At most 20% of batches initially use:

- Bicubic-like 2x downsampling.
- Positively skewed multiplicative speckle centered near observed severity.
- A smaller additive Gaussian component.
- Random pre/post-downsampling order.
- No LR clipping.

Synthetic fine-tuning is rejected if aggregate grouped validation, worst-severity validation, or visual fidelity regresses.

## Model Selection

Checkpoint selection considers:

- PSNR and SSIM on grouped validation.
- LPIPS as a secondary perceptual diagnostic.
- Worst-severity and high-gradient subsets.
- Pre-clamp range behavior.
- Ringing, halos, oversmoothing, repeated texture, and intensity drift.
- Full-pipeline inference cost.

No single training-loss minimum determines the final checkpoint.
