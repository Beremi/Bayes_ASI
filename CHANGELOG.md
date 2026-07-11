# Changelog

This project follows semantic versioning while public APIs remain explicitly
marked alpha.

## 0.5.0 - 2026-07-11

### Added

- Framework-owned randomized Latin-hypercube generation with exactly one point
  per marginal probability stratum.
- A required `Posterior.prior_transform(unit)` callback mapping open-unit-cube
  quantiles to the latent prior.
- Startup validation that rejects configurations unable to collect enough rows
  for the first surrogate instead of idling until the time budget expires.
- Prior-transform and stratification tests for Gaussian, bounded-uniform,
  Darcy-flow, synthetic, and OAS TPBT posterior implementations.

### Changed

- `Posterior.initial_z` now initializes HMC chains only; it is no longer used to
  generate the initial solver design.
- Fresh designs are generated in prior probability space, transformed to latent
  coordinates by the problem package, and then mapped to physical parameters.
- Initial-design journal records identify `latin_hypercube_prior` as the design
  method and record requested versus queued row counts.

### Compatibility

- Posterior callbacks must implement `prior_transform(unit)` before they can
  bootstrap a run that needs initial solver evaluations. Existing completed
  data and checkpoints remain compatible.

## 0.4.0 - 2026-07-11

### Added

- A finite-volume, three-material Darcy-flow inversion with 12 head sensors,
  outlet flux, optional field artifacts, and configurable expensive-job delay.
- Analytical Darcy consistency, material observability, exact inverse recovery,
  and full three-rank MPI tests with 1.5-second solver jobs.
- Exclusive POSIX run-directory locking and content-addressed configuration
  snapshots recorded in the event journal.
- GPL-3.0-or-later licensing, author and citation metadata, security guidance,
  typed-package metadata, pinned GitHub Actions CI, and dependency updates.

### Changed

- Surrogate training and HMC now honor coordinator deadlines internally.
- Restart replay preserves failed-job retry attempts and rejects active jobs
  from any surrogate version inconsistent with the loaded checkpoint.
- Runtime dependencies now declare supported minimum versions; SciPy remains an
  optional dependency used only by the Darcy example.

### Validation

- The ordinary suite covers configuration, storage, replay, queueing, active
  selection, surrogate training, HMC, callbacks, CLI, and recovery behavior.
- MPI gates cover all four synthetic problems and the deliberately expensive
  Darcy workflow, including persistence and checkpoint creation.

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
