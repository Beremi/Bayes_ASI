# Bayes ASI

[![CI](https://github.com/Beremi/Bayes_ASI/actions/workflows/ci.yml/badge.svg)](https://github.com/Beremi/Bayes_ASI/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-4c1.svg)](LICENSE)

Bayes ASI is a problem-agnostic framework for adaptive surrogate Bayesian
inversion of expensive black-box models. It evaluates a user-supplied forward
solver with MPI workers, trains a differentiable PyTorch surrogate, samples the
surrogate posterior with Hamiltonian Monte Carlo (HMC), and adaptively chooses
new solver evaluations.

ASI means **Adaptive Surrogate Inversion**.

The package owns orchestration and numerical infrastructure. Scientific model
files, observations, priors, likelihoods, parameter transforms, and solver
wrappers remain in a separate problem package.

## Capabilities

- MPI coordinator/worker execution for arbitrary Python solver callbacks.
- Restartable HDF5 sample storage and an append-only JSONL event journal.
- Tanh multilayer-perceptron surrogates with input/output normalization.
- Normalization-correct warm starts when new training data arrive.
- Multi-chain HMC with dual-averaged step-size and diagonal-mass adaptation.
- Posterior-plus-distance active selection with duplicate and queue repulsion.
- Prior-distribution Latin-hypercube bootstrapping when no samples are available.
- Time budgets, solver retries, failure backoff, status reporting, and export.
- Strict YAML validation with actionable errors for unknown or invalid options.
- Reference problems spanning nonlinear, linear-Gaussian, bounded, multimodal,
  and finite-volume Darcy-flow inversion.

## Requirements

- Python 3.11 or newer.
- A POSIX operating system with advisory file locking.
- An MPI implementation and `mpi4py`.
- PyTorch, NumPy, h5py, and PyYAML.
- At least two MPI ranks for a run: one coordinator and one worker.

## Installation

For development from this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

The Darcy-flow example additionally requires SciPy:

```bash
python3 -m pip install -e '.[examples]'
```

Install example, quality, and packaging tools for development:

```bash
python3 -m pip install -e '.[dev,examples]'
```

## Quick Start

Validate and run the bounded synthetic example:

```bash
bayes-asi validate examples/bounded_decay.yaml
mpiexec -n 3 bayes-asi run examples/bounded_decay.yaml
bayes-asi status runs/bounded_decay
bayes-asi export runs/bounded_decay --format npz
```

Without an editable installation:

```bash
PYTHONPATH=src mpiexec -n 3 python3 -m bayes_asi.cli run examples/bounded_decay.yaml
```

Relative `run.output_dir` paths are resolved from the process working
directory. Run commands from the repository or problem-project root when using
the supplied configurations.

## Problem Boundary

A problem package provides three optional/required factories referenced from
YAML:

```yaml
solver:
  factory: my_problem.callbacks:make_solver
posterior:
  factory: my_problem.callbacks:make_posterior
initial_data:
  factory: my_problem.callbacks:load_initial_data
```

The solver factory returns an object with:

```python
evaluate(theta: np.ndarray, context: SolverContext) -> ForwardResult
```

The posterior factory returns an object implementing:

```python
initial_z(rng, n_chains) -> np.ndarray
prior_transform(unit) -> np.ndarray
z_to_theta(z: torch.Tensor) -> torch.Tensor
log_prior_z(z: torch.Tensor) -> torch.Tensor
log_likelihood(theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor
```

`prior_transform` maps Latin-hypercube coordinates from the open unit cube to
the latent prior. `initial_z` is used only to initialize HMC chains. `z` is the
unconstrained HMC state, while `theta` is sent to the forward solver. The tensor
transform and both log-density methods must be differentiable in PyTorch.

See [Problem Integration](docs/problem-integration.md) for complete contracts,
shape rules, validation advice, and a package template.

## Runtime Model

Rank 0 is the sole coordinator and writer. Ranks 1 through N-1 each evaluate
one solver job at a time.

```text
YAML + callbacks
       |
       v
  rank 0 coordinator ---- HDF5 / JSONL / checkpoints / status
       |
       +---- MPI work ----> rank 1 solver
       +---- MPI work ----> rank 2 solver
       +---- MPI work ----> ...
       |
       +---- train surrogate -> HMC -> active selection -> queue
```

The run directory contains:

- `data.h5`: completed parameters, outputs, timing, worker, and sequence data.
- `events.jsonl`: durable lifecycle, queue, training, and failure events.
- `status.json`: atomically replaced operational status.
- `configs/`: content-addressed snapshots of every configuration used.
- `checkpoints/surrogate_v*.pt`: atomic surrogate checkpoints.
- `posterior_batches/*.npz`: HMC samples used for active selection.
- `workers/rank_*/job_*`: per-job working directories available to callbacks.
- `run.lock`: advisory POSIX lock preventing concurrent coordinators.

See [Architecture](docs/architecture.md) and [Operations](docs/operations.md)
for lifecycle, restart, and failure semantics.

## Examples

| Configuration | Purpose | Dimensions `(z, theta, y)` |
| --- | --- | --- |
| `examples/synthetic.yaml` | Nonlinear unconstrained map | `(2, 2, 2)` |
| `examples/linear_gaussian.yaml` | Overdetermined linear-Gaussian inverse | `(2, 2, 3)` |
| `examples/bounded_decay.yaml` | Uniform bounded parameters with a logistic transform | `(3, 3, 16)` |
| `examples/bimodal.yaml` | Symmetric two-mode posterior | `(1, 1, 1)` |
| `examples/darcy_flow.yaml` | Three-material finite-volume Darcy flow | `(3, 3, 13)` |

All examples have callback and gradient tests. The Darcy example also verifies
material recovery and runs 1.5-second forward jobs through an end-to-end MPI
test. See [Examples](docs/examples.md) for equations and suggested experiments.

## Documentation

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started.md)
- [Configuration reference](docs/configuration.md)
- [Problem integration](docs/problem-integration.md)
- [Architecture and algorithms](docs/architecture.md)
- [Operations and troubleshooting](docs/operations.md)
- [Examples](docs/examples.md)
- [Python and CLI API](docs/api.md)
- [Development and release](docs/development.md)
- [Changelog](CHANGELOG.md)

## Verification

Run the regular suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the four synthetic examples through MPI:

```bash
BAYES_ASI_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_mpi_smoke
```

Run the deliberately slower Darcy MPI test:

```bash
BAYES_ASI_DARCY_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_darcy_mpi
```

Run quality and package checks:

```bash
ruff check .
mypy src tests
python3 -m build
```

## Project Status

Bayes ASI is alpha software. Run data and checkpoints are designed for robust
restart, but public APIs and checkpoint formats may still evolve.

## License and Citation

Copyright (C) 2026 Michal Béreš. Bayes ASI is distributed under the
[GNU General Public License v3.0 or later](LICENSE). Citation metadata is
provided in [CITATION.cff](CITATION.cff).
