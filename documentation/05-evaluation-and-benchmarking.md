# Evaluation and Benchmarking

## Metric Contract

All quality metrics use the exact clamped output that would be serialized.

| Metric | Contract |
|---|---|
| PSNR | Per image, `data_range=1.0`, then aggregate |
| SSIM | Per image, grayscale, `data_range=1.0`, explicit implementation settings |
| LPIPS | Repeat grayscale into three channels and map `[0,1]` to `[-1,1]` |
| MAE/RMSE | Diagnostic fidelity metrics on `[0,1]` output |

Reports include mean, median, standard deviation, and relevant percentiles. Per-image records are retained to support severity and morphology analysis.

## Validation Views

- Overall structure-grouped validation.
- High and low degradation-severity bins.
- High-gradient versus low-gradient regions/images.
- Images containing negative LR values.
- Images containing above-one LR values.
- Largest morphology groups held out from training.

## Baselines

Required deterministic baselines:

- Nearest-neighbor 2x.
- Bilinear 2x.
- Bicubic 2x.
- Lanczos 2x where implementation is consistent.

Model improvements must be compared to bilinear, the strongest observed interpolation baseline on the supplied noisy pairs.

Smoke-model benchmark reports are engineering checks only. They must be labeled separately from final-model quality and throughput results because a smoke configuration intentionally uses reduced width and depth.

## Benchmark Layers

The benchmark separates:

1. Cold process start and model load.
2. File discovery and `.npy` reads.
3. CPU preprocessing.
4. Host-to-device transfer.
5. Model-only GPU inference.
6. Device-to-host transfer and postprocessing.
7. `.npy` writes.
8. Complete invocation wall time.

## Correct GPU Timing

- Use warm-up iterations for steady-state latency.
- Use `torch.inference_mode()`.
- Use CUDA events for model-only timing.
- Synchronize CUDA at timing boundaries.
- Keep model loading outside per-batch steady-state timing.
- Report batch size, precision, input shape, GPU, driver, CUDA runtime, and PyTorch version.
- Reset and report peak allocated and reserved GPU memory.

## Hardware-Adaptive Benchmarking

- RTX 3050/T4-class: fp16 after numerical parity validation.
- H100/A100-class: bf16 preferred if quality parity holds; fp16 remains available.
- CPU: fp32.
- `torch.compile` is benchmarked separately because compile startup can hurt end-to-end evaluation.
- Batch size is selected by shape and available memory, with a conservative fallback on OOM.

## Benchmark Output

The script writes JSON and console summaries containing:

- Image count and failures.
- Input-shape groups.
- Total wall time and images per second.
- Model latency mean, median, p90, p95, and p99.
- Read and write timing.
- Peak GPU memory.
- Parameter count and checkpoint size.
- Precision, batch size, and environment metadata.

## Acceptance Gates

- No missing, duplicate, malformed, or incorrectly named outputs.
- Exact 2x output dimensions.
- All outputs finite, `float32`, and within `[0,1]`.
- Material quality improvement over bilinear; target is at least 2 dB PSNR before complexity is increased.
- No visible inspection-risk artifacts.
- CPU and accelerated outputs agree within a documented numerical tolerance.
- Clean-process end-to-end benchmark completes successfully.
