# Standalone Inference Contract

## Command

```text
python scripts/inference.py --input-dir <directory> --output-dir <directory>
```

Optional flags include checkpoint path, device, precision, batch size, overwrite behavior, and strict supported-shape validation. The default checkpoint path is resolved relative to the repository, never the current shell directory.

## Required Behavior

1. Parse arguments without source edits.
2. Resolve and validate input, output, and checkpoint paths.
3. Load the checkpoint once.
4. Discover valid `.npy` inputs in deterministic filename order.
5. Reject unsafe or malformed arrays with clear file-specific errors.
6. Group equal shapes for batching.
7. Run hardware-adaptive inference.
8. Clamp predictions only for output serialization.
9. Save same-name `float32` arrays at exactly 2x dimensions.
10. Validate every saved output.
11. Print a concise execution and timing summary.
12. Return a nonzero process status on any unhandled failure.

## Portability

- CPU fallback is mandatory.
- CUDA is selected automatically when available.
- No notebook variables or environment-specific absolute paths are required.
- No network access is required during inference.
- Model architecture metadata required to load weights is contained in the checkpoint or stable code configuration.
- The script supports Windows and Linux path semantics.

## Performance

- Use `torch.inference_mode()`.
- Keep the model resident on the selected device.
- Use shape-grouped batches.
- Use pinned host buffers and nonblocking transfer when beneficial.
- Use channels-last only after correctness validation.
- Use fp16 or bf16 according to hardware policy and numerical parity checks.
- Avoid test-time augmentation and ensembles in the default evaluator.

## Safety and Validation

- Load arrays with `allow_pickle=False`.
- Reject object, complex, empty, nonfinite, and non-2D arrays.
- Do not follow unexpected recursive paths.
- Prevent output path traversal by deriving only the validated basename.
- Refuse accidental overwrite unless explicitly permitted.
- Write outputs atomically where practical.

## Checkpoint Compatibility

Checkpoint loading validates:

- Format version.
- Architecture name and parameters.
- Expected input feature count.
- Scale factor.
- State-dictionary completeness.
- EMA availability and selection.

Unknown or incompatible formats fail with an actionable message rather than partially loading weights.

## Exit Criteria

A release evaluator is accepted only after it runs successfully in a clean environment against representative inputs of both official sizes and the complete supplied test directory.
