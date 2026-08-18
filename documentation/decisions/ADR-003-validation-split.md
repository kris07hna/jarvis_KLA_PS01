# ADR-003: Structure-Grouped Validation

## Status

Accepted.

## Context

The training set contains many highly correlated semiconductor structures and transformed variants. An image-random split can place near-duplicate structures in both training and validation, inflating measured generalization. The official test set contains different sources and structures.

## Decision

Build multiscale GT descriptors, account for rotations and reflections, connect highly similar samples into groups, and assign complete groups to train or validation. Persist the manifest and its generation metadata.

## Rationale

- GT descriptors measure morphology without contamination from random degradation.
- Group assignment reduces structural leakage.
- A frozen manifest supports comparable experiments and checkpoint decisions.
- Severity balancing prevents validation from becoming accidentally easier or harder only because of noise distribution.

## Consequences

- Split generation is a versioned preprocessing step.
- Random validation may be reported only as a secondary diagnostic.
- Large similarity components require verification to avoid over-grouping repetitive low-detail images.
- Model selection uses grouped full-image validation.

## Alternatives Considered

- Random 80/20 split: rejected as the primary split due to leakage.
- Filename-range split: rejected because related samples are not reliably contiguous.
- Source-category split: unavailable because source metadata is not supplied.
