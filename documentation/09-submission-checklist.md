# Submission Checklist

## Repository

- [ ] Public GitHub repository is accessible without authentication.
- [ ] README contains exact setup and inference commands.
- [ ] Standalone evaluation script accepts input and output directories.
- [ ] Training process is reproducible from scripts or a thin Kaggle launcher.
- [ ] Final model weights are available through Git LFS or a stable public link.
- [ ] Restored test outputs are included or linked.
- [ ] Minimal runtime requirements and complete environment freeze are present.
- [ ] License and dataset redistribution boundaries are documented.
- [ ] No credentials, local absolute paths, caches, or private data are committed.

## Inference Release

- [ ] All 400 supplied test inputs produce outputs.
- [ ] Output basenames exactly match inputs.
- [ ] Outputs are two-dimensional `float32` `.npy` arrays.
- [ ] Output dimensions are exactly 2x.
- [ ] Outputs are finite and within `[0,1]`.
- [ ] Model and checkpoint load without network access.
- [ ] CPU fallback succeeds.
- [ ] CUDA accelerated path succeeds.
- [ ] Fresh-environment invocation succeeds without manual edits.
- [ ] End-to-end benchmark report is archived.

## Results Evidence

- [ ] Frozen split manifest is included.
- [ ] Bilinear and other interpolation baselines are reported.
- [ ] PSNR, SSIM, and LPIPS contracts are stated.
- [ ] Aggregate and grouped-OOD metrics are reported.
- [ ] Before/restored/GT comparisons use identical intensity display settings.
- [ ] Difficult and severe-noise examples are included.
- [ ] Parameter count, checkpoint size, training hardware/time, and inference latency are measured.
- [ ] No target or unmeasured metric is presented as an achieved result.

## Presentation

- [ ] Slide 1: team details.
- [ ] Slide 2: semiconductor restoration problem and manufacturing impact.
- [ ] Slide 3: range-preserving degradation-aware concept.
- [ ] Slide 4: architecture, loss, data, and training pipeline.
- [ ] Slide 5: input decomposition, grouped validation, and efficient LR-domain reconstruction.
- [ ] Slide 6: metrics and visual comparisons.
- [ ] Slide 7: PyTorch/Kaggle/RTX stack, training time, model size, and measured inference.
- [ ] Slide 8: GitHub and optional video link.
- [ ] Slide 9: references.
- [ ] Final PDF follows `TeamName_KLA_PS01.pdf` naming.
- [ ] Instruction slide is removed.

## Final Review

- [ ] A teammate follows the README from a clean checkout.
- [ ] All documentation links work.
- [ ] Repository contains no oversized accidental dataset files.
- [ ] Model output is inspected for ringing, halos, invented patterns, and intensity drift.
- [ ] Final checkpoint selection rationale is documented.
