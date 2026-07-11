# Contributing

Contributions should keep Bayes ASI independent of any concrete inverse
problem. Put model-specific solvers, observations, likelihoods, and datasets in
a separate package and use callback factories.

Before opening a change:

```bash
python3 -m pip install -e '.[dev,examples]'
PYTHONPATH=src python3 -m unittest discover -s tests
BAYES_ASI_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_mpi_smoke
BAYES_ASI_DARCY_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_darcy_mpi
ruff check .
mypy src tests
python3 -m build
```

Add focused tests for behavior changes. Update the configuration and integration
documentation when callback, YAML, persistence, or restart semantics change.

Keep generated `runs/`, HDF5 files, checkpoints, posterior batches, solver
artifacts, and Python caches out of commits.

See [Development and Release](docs/development.md) for architecture constraints,
numerical review guidance, compatibility testing, and release requirements.
