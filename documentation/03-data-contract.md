# Data Contract

## Canonical Directories

```text
train/train/NoisyLR/*.npy
train/train/GT/*.npy
Test_NoisyLR/NoisyLR/*.npy
```

Only these canonical directories are scanned. Recursive loading from the workspace root is prohibited because extracted archives contain AppleDouble metadata.

## Training Pair Contract

- Pair files by identical filename, including extension.
- LR must be a two-dimensional numeric array.
- GT must be a two-dimensional numeric array.
- Both arrays must contain only finite values.
- GT height and width must each equal twice the LR dimensions.
- Arrays are converted to contiguous `float32` tensors.
- LR values are not clipped or normalized per image.
- GT is expected in `[0,1]`; violations fail the audit.

## Inference Input Contract

- Extension: `.npy` only.
- Rank: exactly two.
- Type: numeric, non-object.
- Values: finite.
- Supported official shapes: `128x128` and `256x256`.
- Loading must use `np.load(path, allow_pickle=False)`.
- Symbolic links and files outside the requested input directory are not followed by default.

## Output Contract

- Output basename must exactly match the input basename.
- Shape must be `(2 * input_height, 2 * input_width)`.
- Type must be `numpy.float32`.
- Values must be finite and constrained to `[0,1]`.
- Saving should be atomic where practical: write a temporary file, validate it, then replace the target.
- Existing outputs require an explicit overwrite option.

## Excluded Files

- Any path containing `__MACOSX`.
- Any filename beginning with `._`.
- `.DS_Store`.
- Directories and non-`.npy` files.

## Data Audit

The audit command will report:

- Number of LR, GT, paired, missing, and extra files.
- Shape and dtype distributions.
- Dataset extrema, mean, standard deviation, and nonfinite counts.
- Values outside `[0,1]` for LR and GT separately.
- Duplicate file hashes.
- Exact scale-factor violations.
- Unexpected metadata files.

Audit results must be persisted as machine-readable JSON and a concise console summary.

## Split Strategy

A random image-level split is not accepted as the primary validation because structurally near-identical GT samples exist.

The grouped split process is:

1. Build multiscale structural descriptors from GT only.
2. Compare identity, rotation, and reflection variants.
3. Connect sufficiently similar samples into components.
4. Verify large components to avoid grouping unrelated low-detail images.
5. Assign whole components to train or validation.
6. Balance intensity and observed degradation severity distributions.
7. Persist filenames, hashes, algorithm version, threshold, and seed.

Initial target is approximately 85% training and 15% grouped validation. The validation manifest remains untouched during model selection.

## Data Security

- Pickle loading is always disabled.
- Object arrays are rejected.
- Paths are resolved and validated before use.
- Raw data is not uploaded to telemetry or external services by project code.
- Dataset licensing and confidentiality remain an organizer/team responsibility and must be confirmed before public redistribution.
