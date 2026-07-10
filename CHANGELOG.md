# Changelog

This project follows semantic versioning while public APIs remain explicitly
marked alpha.

## 0.3.0 - 2026-07-10

### Added

- Strict YAML option and numeric-range validation.
- Batch HDF5 append, schema validation, count-only reads, and truncated-tail
  event recovery.
- Atomic surrogate checkpoints, posterior batches, status updates, and exports.
- Checkpoint training-row metadata and fallback to older valid checkpoints.
- Configurable failure backoff for training and active selection.
- Warmup acceptance reporting and bounded HMC step-size adaptation.
- CLI `validate` and `--version` support.
- Nonlinear, linear-Gaussian, bounded decay, and bimodal synthetic examples.
- Callback/gradient tests and end-to-end MPI tests for every synthetic example.
- Ruff, mypy, build metadata, and comprehensive documentation.

### Changed

- Coordinator caches completed-row count instead of loading full HDF5 arrays
  on every loop pass.
- Initial data import uses one durable HDF5 transaction.
- Surrogate warm starts preserve physical predictions when normalization
  changes.
- Active selection reuses completed/queued distances, updates incrementally,
  and chunks history distance calculations.
- Training metrics now distinguish total epochs from best epoch.
- Non-finite parameters, outputs, training data, and HMC initial states fail
  early with actionable errors.

### Compatibility

- Existing checkpoints without training-row metadata remain loadable and are
  conservatively retrained.
- Problem callback interfaces remain unchanged.
- New configuration validation may reject previously ignored misspelled keys or
  invalid numeric values.

## 0.2.0

- Separated OAS TPBT callbacks and configurations from the generic package.
- Made Bayes ASI explicitly problem-agnostic.
- Improved active-queue refresh behavior after surrogate updates.

## 0.1.0

- Initial adaptive surrogate inversion framework with MPI workers, HDF5
  storage, PyTorch surrogate training, HMC, and active selection.
