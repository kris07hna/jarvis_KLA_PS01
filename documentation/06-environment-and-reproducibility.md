# Environment and Reproducibility

## Python Version

Python 3.11 is the project target. It provides broad compatibility across Windows CUDA PyTorch, Kaggle, restoration metrics, LPIPS, and optional export tooling.

The project uses a local `.venv`; system Python packages are not modified.

## Current Local State

Observed before setup:

```text
OS: Windows
GPU: NVIDIA GeForce RTX 3050, 6144 MiB
Driver: 610.74
System Python: 3.12.10 from Microsoft Store
System PyTorch: 2.13.0+cpu
CUDA available in system PyTorch: false
```

The NVIDIA driver is operational. CUDA-enabled PyTorch wheels include the required user-space runtime; installing the complete CUDA Toolkit is not an initial requirement.

## Dependency Layers

- `requirements.txt`: minimal inference dependencies.
- `requirements-train.txt`: training, visualization, and quality metrics.
- `requirements-freeze.txt`: exact final environment captured after validation.
- `kagglehub`: optional artifact-transfer utility; not required for the VS Code Kaggle workflow.
- `pyproject.toml`: package metadata, Python constraint, and development tooling.

PyTorch installation is platform-specific and must use an official CUDA wheel locally. Kaggle's preinstalled compatible PyTorch should be retained unless a documented incompatibility requires replacement.

## Environment Verification

Verification must execute real work:

1. Import PyTorch and print version/runtime metadata.
2. Confirm CUDA visibility and GPU name.
3. Allocate CUDA tensors.
4. Run a convolution forward and backward pass.
5. Run an AMP operation.
6. Confirm gradients and outputs are finite.
7. Report peak VRAM.

`torch.cuda.is_available()` alone is not sufficient proof.

## Determinism

- Seed Python, NumPy, CPU torch, and CUDA generators.
- Persist all random states in checkpoints.
- Use deterministic validation transforms and sample ordering.
- Record whether deterministic algorithms are enabled; strict determinism may be relaxed for training performance only when documented.
- Compare repeated validation from the same checkpoint to detect nondeterministic data or metric behavior.

## Environment Metadata

Each run records:

- Python and package versions.
- Operating system and architecture.
- PyTorch, CUDA runtime, cuDNN, and GPU details.
- Precision and TF32 settings.
- Resolved configuration.
- Split-manifest hash.
- Source revision when Git is available.
- Start/end timestamps and elapsed time.

## Fresh-Environment Test

Before release:

1. Create a new environment.
2. Install only documented dependencies.
3. Run unit tests.
4. Run inference on representative `128x128` and `256x256` arrays.
5. Validate output filenames, dimensions, dtype, finiteness, and range.
6. Run the benchmark command from a new process.
