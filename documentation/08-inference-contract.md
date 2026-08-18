# Standalone Inference Contract

## Official Execution Commands

### Primary KLA Submission Entrypoint
```bash
python run.py <input-dir> <output-dir>
```

#### Example Usage:
```bash
python run.py Test_NoisyLR/NoisyLR outputs
```

---

## Required Engine Behavior

1. **Argument Parsing**: Accepts positional or flagged directory arguments without modifying source code.
2. **Path Validation**: Autonomously verifies existence of `<input-dir>`, creates `<output-dir>` recursively if missing, and loads weights from `models/best.pt`.
3. **Single Model Load**: Loads pre-trained ContextNAFNet weights into memory once per process execution.
4. **Input Discovery**: Discovers all 2D grayscale `.npy` arrays in deterministic lexicographical order.
5. **Array Validation**: Safe numpy array loading (`allow_pickle=False`), verifying shapes `(H, W)`, `(1, H, W)`, or `(H, W, 1)`.
6. **Self-Ensemble TTA Inference**: Executes 8-Fold Test-Time Augmentation (4 rotations $\times$ 2 flips) to stabilize predictions and eliminate variance.
7. **2x Super-Resolution**: Performs 2x spatial upscaling (`128x128 -> 256x256`, `256x256 -> 512x512`).
8. **Intensity Clamping & Sanitization**: Clamps pixel outputs strictly to `[0.0, 1.0]` and replaces any potential `NaN`/`Inf` values with `0.0`.
9. **Atomic Output Saving**: Saves each restored 2D `float32` `.npy` array with matching filename into `<output-dir>`.
10. **Timing & Summary**: Outputs timing logs, images-per-second throughput (FPS), and execution summaries upon completion.

---

## Hardware Portability & Offline Execution

- **CUDA Acceleration**: Automatically utilizes NVIDIA CUDA GPU if available.
- **CPU Fallback**: Gracefully falls back to CPU if CUDA is unavailable.
- **100% Offline**: Operates completely offline with zero internet or API calls required.
- **Cross-Platform**: Fully compatible with Windows (PowerShell/CMD) and Linux/macOS environments.

---

## Checkpoint & Safety Safeguards

- **Resolution Assertion**: Hard-asserts output shape matches `(in_h * 2, in_w * 2)`.
- **Value Bounds Assertion**: Hard-asserts all output values satisfy `0.0 <= val <= 1.0`.
- **Finiteness Assertion**: Hard-asserts zero `NaN` or `Inf` elements exist in exported `.npy` files.
