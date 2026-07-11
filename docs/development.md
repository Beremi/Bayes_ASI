# Development and Release

## Environment

Create a local environment with runtime and development dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[dev,examples]'
```

The project uses the standard library `unittest` runner, Ruff, mypy, and the
PEP 517 `build` frontend.

## Test Layers

### Unit and callback tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

This covers configuration, storage and replay, queue invariants, active
selection, surrogate normalization/checkpoints, HMC, runtime recovery, CLI,
Latin-hypercube prior transforms, all example callbacks, and finite-volume
Darcy-flow consistency and recovery.

### MPI integration

```bash
BAYES_ASI_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_mpi_smoke
```

The test launches four short three-rank runs, one for each synthetic example.
Each run must finish with completed data and a surrogate checkpoint. Keep this
gate opt-in because nested MPI execution is not available in every test runner.

The slower Darcy gate uses 1.5-second jobs and verifies full orchestration,
checkpoint creation, finite PDE outputs, timing, and configuration snapshots:

```bash
BAYES_ASI_DARCY_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_darcy_mpi
```

### External problem compatibility

When changing callback contracts or configuration, run the tests in every
known problem adapter with both source trees on `PYTHONPATH`. For the sibling
OAS TPBT repository in the original workspace:

```bash
cd ../Bayes_OAS_TPBT
PYTHONPATH=../Bayes_ASI/src:src python3 -m unittest discover -s tests
```

## Quality Gates

```bash
ruff check .
mypy src tests
python3 -m compileall -q src tests
git diff --check
```

Ruff checks import ordering, Python modernization, common bugs, and standard
errors. Mypy checks package and test annotations while ignoring only missing
type metadata for h5py, PyYAML, and SciPy.

## Building Artifacts

```bash
rm -rf build dist
python3 -m build
```

Inspect the artifacts:

```bash
python3 -m zipfile -l dist/bayes_asi-*.whl
tar -tf dist/bayes_asi-*.tar.gz
```

Test the wheel in a clean environment before release:

```bash
python3 -m venv /tmp/bayes-asi-wheel-test
/tmp/bayes-asi-wheel-test/bin/python -m pip install dist/bayes_asi-*.whl
/tmp/bayes-asi-wheel-test/bin/bayes-asi --version
```

## Performance Review

Changes should preserve these design constraints:

- rank 0 is the only HDF5 and journal writer;
- the coordinator hot loop does not load full completed arrays;
- initial data use batch HDF5 append;
- active selection does not allocate an unbounded candidate-by-history matrix;
- retries use backoff and do not generate event storms;
- worker callbacks do not share writable run-state files;
- checkpoint and status replacement remains atomic.

For large-output problems, measure surrogate training memory and checkpoint
size. For large sample histories, measure active-selection time with realistic
candidate and parameter dimensions.

## Numerical Review

When modifying surrogate or HMC code, test:

- finite gradients and outputs;
- exact prediction preservation across normalization adaptation;
- deterministic behavior for fixed seeds on CPU;
- checkpoint round-trip and legacy checkpoint loading;
- production acceptance separately from warmup acceptance;
- bounded adaptation under pathological acceptance;
- multiple dispersed chains for multimodal examples.

Do not interpret passing numerical smoke tests as posterior validation for a
new scientific problem. Problem owners must validate likelihood, prior,
identifiability, surrogate error, convergence, and posterior predictive fit.

## Commit Practice

Keep commits focused and independently testable. A useful sequence is:

1. behavior and tests;
2. performance or internal refactor and tests;
3. examples and integration tests;
4. documentation and packaging metadata.

Do not combine generated run data, model checkpoints, or problem-specific
scientific assets with framework changes.

## Release Checklist

Before publishing a version:

1. Confirm `LICENSE`, copyright, citation, author, and project URL metadata.
2. Update `__version__`, `project.version`, and `CHANGELOG.md` consistently.
3. Run unit, callback, MPI, external-adapter, Ruff, mypy, compile, and diff
   checks.
4. Build wheel and source distribution from a clean worktree.
5. Inspect artifact contents for run data, caches, secrets, or external model
   files.
6. Install the wheel in a clean environment and run at least one MPI example.
7. Confirm CI on every supported Python boundary and the MPI jobs.
8. Tag the exact tested commit and publish from that tag.
