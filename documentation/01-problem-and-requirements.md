# Problem and Requirements

## Objective

Restore degraded, low-resolution, single-channel semiconductor inspection images to clean 2x spatial resolution. A single model must jointly remove signal-dependent speckle, reduce Gaussian-like haze, and reconstruct details lost during downsampling.

## Manufacturing Context

Inspection images support defect detection and dimensional verification. Restoration must improve visibility without inventing periodic patterns, ringing around edges, or false structures that could be interpreted as defects.

## Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Accept a directory containing two-dimensional grayscale `.npy` arrays. |
| FR-02 | Preserve signed and above-one input values during preprocessing. |
| FR-03 | Restore every image to exactly twice its input height and width. |
| FR-04 | Support `128x128 -> 256x256` and `256x256 -> 512x512` through one fully convolutional model. |
| FR-05 | Write one `float32` `.npy` output per input using the same filename. |
| FR-06 | Ensure serialized outputs are finite and constrained to `[0,1]`. |
| FR-07 | Provide a standalone inference script accepting input and output directory arguments without source edits. |
| FR-08 | Provide a reproducible training entry point and resumable checkpoints. |

## Quality Requirements

- Optimize PSNR, SSIM, and LPIPS without sacrificing defect-relevant structural fidelity.
- Validate on structure groups excluded from training to estimate out-of-distribution behavior.
- Inspect severe-noise and high-gradient subsets separately from aggregate metrics.
- Reject hallucinated textures, halos, ringing, and intensity drift even when one aggregate metric improves.

## Performance Requirements

- Use batched GPU inference when input shapes match.
- Adapt precision to hardware: fp16 on RTX/T4-class GPUs and bf16 where supported and validated.
- Measure complete read-to-write wall time as well as model-only GPU latency.
- Keep model loading outside per-image timing.
- Retain CPU inference as a correctness fallback, not the primary performance target.

## Reproducibility Requirements

- Freeze dependency versions for the final training environment.
- Persist split manifests, configuration, seeds, and environment metadata.
- Store model, EMA, optimizer, scheduler, scaler, and random states in checkpoints.
- Run inference from a clean environment before release.
- Avoid hidden notebook state and hardcoded machine-specific paths.

## Confirmed Dataset Facts

- 3,200 aligned training pairs are present.
- Training LR arrays are `128x128`; GT arrays are `256x256`.
- 400 test LR arrays are present.
- Arrays are grayscale `float32` and finite.
- GT values lie in `[0,1]`.
- LR values extend approximately from `-0.279` to `2.158`.
- About 3.4% of training LR values fall outside `[0,1]`.
- The observed reduction is most consistent with bicubic-like 2x downsampling.
- Noise is predominantly signal-dependent with a smaller additive component.

## Official Deliverables

- Public GitHub repository.
- Complete README with setup and inference instructions.
- Standalone Python evaluation script.
- Reproducible training script or notebook.
- Trained model weights.
- Restored test outputs.
- Complete environment freeze.
- Eight-to-nine-slide PDF following the provided idea-submission template.

## Out of Scope for the Initial Release

- Color restoration.
- Arbitrary scale factors other than 2x.
- Interactive UI or web service.
- GAN-based generation.
- Manual per-image correction.
- Mandatory FFT processing without measured benefit.
