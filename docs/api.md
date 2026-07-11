# Python and CLI API

## Supported Top-Level Imports

```python
from bayes_asi import (
    AppConfig,
    DifferentiableSurrogate,
    FactoryConfig,
    ForwardResult,
    ForwardSolver,
    Posterior,
    SolverContext,
    __version__,
    latin_hypercube,
    load_config,
)
```

The callback protocols use structural typing: a class does not need to inherit
from a Bayes ASI base class if it implements the required methods.

`latin_hypercube(rng, n_points, dimension)` returns a reproducible randomized
design in the strict open unit cube. Runtime bootstrapping passes this design to
`Posterior.prior_transform`; problem packages may also use it in callback tests.

## `SolverContext`

```python
SolverContext(
    run_dir: Path,
    work_dir: Path,
    job_id: str,
    worker_rank: int,
    metadata: dict[str, Any] = {},
)
```

Created once per worker attempt and passed to `ForwardSolver.evaluate`.

## `ForwardResult`

```python
ForwardResult(
    outputs: np.ndarray,
    metadata: dict[str, Any] = {},
    artifact_paths: list[str] = [],
)
```

`outputs` are flattened by the runtime to configured output width. Metadata and
artifact paths are written to the `job_completed` event.

## Configuration API

```python
config = load_config("problem.yaml")
print(config.problem.parameter_dim)
print(config.run.output_dir)
```

Configuration objects are slotted dataclasses. Constructing individual section
classes directly applies the same numeric validation as YAML loading.

`resolve_dotted("package.module:factory")` is available from
`bayes_asi.config` for advanced validation, but problem packages normally only
need public top-level imports.

## Lower-Level Modules

These modules are tested and useful for analysis, but they expose more
implementation detail than the top-level callback API:

- `bayes_asi.surrogate`: `SurrogateModel`, `SurrogateTrainer`, normalization,
  and checkpoint loading.
- `bayes_asi.hmc`: `HMCSampler` and `HMCResult`.
- `bayes_asi.active`: `ActiveSelector` and `SelectedBatch`.
- `bayes_asi.storage`: `HDF5Store`, `CompletedData`, and event replay helpers.
- `bayes_asi.queue`: in-memory `Job` and `QueueState`.

Their formats may evolve during alpha releases.

## CLI

### Version

```bash
bayes-asi --version
```

### Validate

```bash
bayes-asi validate CONFIG
```

Loads the schema and resolves factory callables. Prints a JSON summary. Does not
instantiate callbacks or check external solver files.

### Run

```bash
mpiexec -n N bayes-asi run CONFIG
```

Must execute under MPI with `N >= 2`.

### Status

```bash
bayes-asi status RUN_DIR
```

Parses and prints `status.json` as sorted, indented JSON.

### Export

```bash
bayes-asi export RUN_DIR [--format npz|csv] [--output PATH]
```

Default format is NPZ. Default output is `RUN_DIR/completed.FORMAT`. Parent
directories are created and replacement is atomic.

## Compatibility

The most stable integration surface is:

- YAML section names and documented options;
- factory invocation with `config=` and configured keyword arguments;
- `ForwardSolver`, `Posterior`, `SolverContext`, and `ForwardResult` contracts;
- Latin-hypercube generation and the `Posterior.prior_transform` contract;
- CLI commands documented above.

Checkpoint internals, journal event payload details, lower-level classes, and
run-loop implementation remain alpha-level interfaces.
