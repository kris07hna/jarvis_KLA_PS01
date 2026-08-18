# Documentation Index

This directory is the authoritative design and operating record for the project. Documents are numbered in the order a contributor or reviewer should read them.

## Core Design

| Document | Purpose | Status |
|---|---|---|
| [01 - Problem and Requirements](01-problem-and-requirements.md) | Scope, constraints, success criteria, and official deliverables | Accepted |
| [02 - System Architecture](02-system-architecture.md) | End-to-end flow and model design | Accepted for implementation |
| [03 - Data Contract](03-data-contract.md) | Dataset schema, validation, splits, and output format | Accepted |
| [04 - Training and Generalization](04-training-and-generalization.md) | Optimization, anti-overfitting controls, and checkpointing | Accepted for baseline |
| [05 - Evaluation and Benchmarking](05-evaluation-and-benchmarking.md) | Metrics, timing protocol, and quality gates | Accepted |

## Operations

| Document | Purpose | Status |
|---|---|---|
| [06 - Environment and Reproducibility](06-environment-and-reproducibility.md) | Python, CUDA, dependencies, seeds, and environment capture | Accepted |
| [07 - Local and Kaggle Workflow](07-local-and-kaggle-workflow.md) | Hardware-adaptive execution and session recovery | Accepted |
| [08 - Inference Contract](08-inference-contract.md) | Standalone evaluator behavior and failure handling | Accepted |
| [09 - Submission Checklist](09-submission-checklist.md) | KLA repository, outputs, presentation, and release checks | Draft until results exist |
| [10 - Execution Roadmap](10-execution-roadmap.md) | Ordered implementation and training gates | Active |
| [11 - Kaggle Studio Setup](11-kaggle-studio-setup.md) | VS Code extension authentication, private data attachment, runs, and resume | Active |

## Decisions

- [ADR-001: Compact multiscale NAF architecture](decisions/ADR-001-model-architecture.md)
- [ADR-002: Range-preserving input representation](decisions/ADR-002-input-representation.md)
- [ADR-003: Structure-grouped validation](decisions/ADR-003-validation-split.md)

## Source Material

- The original DAF-Net proposal remains at [`../plan.md`](../plan.md).
- [`archive/README.md`](archive/README.md) explains how historical planning artifacts are handled.
- [Chat Handoff Memory](CHAT_HANDOFF.md) is the current secure implementation and training handoff for a new chat.

## Documentation Rules

- Update requirements and architecture before changing a public CLI or data contract.
- Record significant alternatives and tradeoffs as an ADR.
- Label proposed performance as a target; only measured results belong in reports.
- Keep Kaggle notebooks as launchers. Reusable logic belongs in `src/` or `scripts/`.
- Never document unverified benchmark numbers as achieved results.
