# Bayes_ASI

Bayes_ASI is an adaptive Bayesian inversion framework for expensive black-box
forward models. It trains a differentiable PyTorch surrogate from completed
solver evaluations, samples the surrogate posterior with Hamiltonian Monte
Carlo, and selects new expensive evaluations from high-posterior regions while
keeping distance from completed and queued samples.

The first implementation uses:

- Python callbacks for problem-specific posterior and solver definitions.
- MPI worker ranks for expensive forward-model evaluations.
- HDF5 plus append-only JSONL logs for restartable data storage.
- A tanh MLP surrogate with input/output normalization and input gradients.
- Custom PyTorch HMC with dual-averaged step-size warmup.

## Package Boundary

`bayes-asi` contains only the reusable inversion engine, callback protocols,
storage, active selection, surrogate training, HMC sampling, CLI, and a small
synthetic example. Concrete scientific problems should live in separate
packages that expose solver and posterior factories referenced from YAML.

## Quick Start

From this repository:

```bash
python3 -m pip install -e .
mpiexec -n 3 bayes-asi run examples/synthetic.yaml
bayes-asi status runs/synthetic
bayes-asi export runs/synthetic --format npz
```

For local development without installing:

```bash
PYTHONPATH=src mpiexec -n 3 python3 -m bayes_asi.cli run examples/synthetic.yaml
PYTHONPATH=src python3 -m unittest discover -s tests
```

To connect a problem package without installing either package:

```bash
PYTHONPATH=/path/to/Bayes_ASI/src:/path/to/problem/src bayes-asi run problem.yaml
```

## Runtime Model

Rank 0 is the coordinator. It owns the HDF5 writer, queue state, surrogate
training, HMC sampling, active selection, and status files. Ranks 1..N-1 are
solver workers. Each worker receives one parameter vector at a time, evaluates
the configured black-box solver wrapper, and returns outputs or a structured
failure.

The coordinator writes:

- `data.h5`: completed parameter and output arrays.
- `events.jsonl`: append-only queue, worker, training, and failure events.
- `status.json`: current queue, solver, training, and budget status.
- `checkpoints/surrogate_v*.pt`: surrogate model checkpoints.

## Problem Callbacks

The YAML config points to two factories:

```yaml
solver:
  factory: bayes_asi.examples.synthetic:make_solver
posterior:
  factory: bayes_asi.examples.synthetic:make_posterior
```

The solver factory returns an object with:

```python
evaluate(theta: np.ndarray, context: SolverContext) -> ForwardResult
```

The posterior factory returns an object with:

```python
initial_z(rng, n_chains) -> np.ndarray
z_to_theta(z: torch.Tensor) -> torch.Tensor
log_prior_z(z: torch.Tensor) -> torch.Tensor
log_likelihood(theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor
```

`z` is the unconstrained latent variable used by HMC. `theta` is the physical
parameter vector sent to the black-box solver.
