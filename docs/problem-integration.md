# Problem Integration

Bayes ASI is intentionally unaware of any concrete scientific model. A problem
package connects physical parameters, external solver execution, observations,
priors, likelihoods, and optional existing data through Python factories.

## Recommended Package Layout

```text
my-problem/
  pyproject.toml
  src/
    my_problem/
      __init__.py
      callbacks.py
  configs/
    production.yaml
  tests/
    test_callbacks.py
```

Install both packages during local development:

```bash
python3 -m pip install -e /path/to/Bayes_ASI -e /path/to/my-problem
```

Or use both source trees directly:

```bash
PYTHONPATH=/path/to/Bayes_ASI/src:/path/to/my-problem/src \
  mpiexec -n 4 python3 -m bayes_asi.cli run configs/production.yaml
```

## Factory Invocation

For this YAML:

```yaml
solver:
  factory: my_problem.callbacks:make_solver
  kwargs:
    executable: /opt/model/bin/solve
```

Bayes ASI calls:

```python
make_solver(config=app_config, executable="/opt/model/bin/solve")
```

The solver factory is constructed separately on every worker. Avoid opening a
shared writable file or launching work in the factory itself. Rank 0 constructs
the posterior and optional initial-data object.

## Solver Contract

Implement `ForwardSolver`:

```python
from dataclasses import dataclass

import numpy as np

from bayes_asi import ForwardResult, SolverContext


@dataclass(slots=True)
class ModelSolver:
    executable: str

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        # Write inputs and run the model in context.work_dir.
        outputs = run_model(self.executable, theta, context.work_dir)
        return ForwardResult(
            outputs=np.asarray(outputs, dtype=float),
            metadata={"iterations": 12},
            artifact_paths=[str(context.work_dir / "model.log")],
        )
```

`theta` has shape `(problem.parameter_dim,)`. Returned outputs must contain
exactly `problem.output_dim` finite numeric values. A plain array is accepted as
a shorthand, but `ForwardResult` is preferred.

`SolverContext` provides:

| Field | Meaning |
| --- | --- |
| `run_dir` | Shared run output directory. Rank 0 owns shared database files. |
| `work_dir` | Unique directory `workers/rank_N/job_ID` for this attempt. |
| `job_id` | Durable job identifier. Retries retain the same identifier. |
| `worker_rank` | MPI worker rank. |
| `metadata` | Currently includes solver `attempt`. |

Do not write to `data.h5`, `events.jsonl`, or `status.json` from a worker.

### Solver Failures

Raise an exception for model failure, timeout, malformed output, or invalid
physics. The worker captures the message and traceback and returns a structured
failure to rank 0. The queue retries according to `queue.retry_limit`.

Bayes ASI rejects non-finite outputs even if the callback returns normally.
Metadata and artifact paths should be JSON-compatible. NumPy scalars/arrays and
`Path` values are normalized automatically in journal records.

## Posterior Contract

The posterior defines HMC in unconstrained coordinates and likelihood
evaluation against surrogate outputs.

```python
class ProblemPosterior:
    def initial_z(self, rng, n_chains):
        ...

    def prior_transform(self, unit):
        ...

    def z_to_theta(self, z):
        ...

    def log_prior_z(self, z):
        ...

    def log_likelihood(self, theta, y_pred):
        ...
```

Shape rules:

| Method | Input | Output |
| --- | --- | --- |
| `initial_z` | NumPy generator and HMC chain count N | finite NumPy array `(N, latent_dim)` |
| `prior_transform` | strict open-unit array `(N, latent_dim)` | finite latent-prior array `(N, latent_dim)` |
| `z_to_theta` | tensor `(N, latent_dim)` | tensor `(N, parameter_dim)` |
| `log_prior_z` | tensor `(N, latent_dim)` | tensor `(N,)` |
| `log_likelihood` | theta `(N, parameter_dim)`, prediction `(N, output_dim)` | tensor `(N,)` |

`prior_transform` is the inverse-CDF or more general generative transform for
the prior represented by `log_prior_z`. For independent priors it normally
applies one marginal inverse CDF per column. Correlated priors may apply a
Rosenblatt, copula, or matrix transform after consuming the independent unit
coordinates. Bayes ASI cannot infer this transform reliably from a log density,
so the problem package must provide it.

`z_to_theta`, `log_prior_z`, and `log_likelihood` must preserve PyTorch's
autograd graph. Do not convert their inputs to NumPy, call `.item()`, detach, or
wrap computed tensors with `torch.tensor(...)`.

Constants independent of `z` may be omitted from log densities. Jacobian terms
that depend on `z` must be included.

## Bounded Parameters

For independent bounds `(lower, upper)`, a stable logistic transform is:

```python
import numpy as np
import torch
import torch.nn.functional as F


def z_to_theta(z, lower, upper):
    return lower + (upper - lower) * torch.sigmoid(z)


def prior_transform(unit):
    return np.log(unit) - np.log1p(-unit)


def uniform_log_prior_z(z):
    # Uniform physical prior plus transform Jacobian, up to constants.
    return torch.sum(F.logsigmoid(z) + F.logsigmoid(-z), dim=-1)
```

For a Latin hypercube `unit`, the logit produces the transformed-uniform latent
prior; applying `z_to_theta` recovers a Latin hypercube over the bounded physical
prior. See `bayes_asi.examples.bounded_decay` for a complete implementation.

## Likelihood Design

The surrogate predicts the same flattened output vector as the real solver.
The posterior may:

- compare every output component to observations;
- reshape outputs into curves, fields, or repeated realizations;
- compute differentiable summary features;
- include correlated Gaussian covariance;
- select among differentiable likelihood variants.

Operations inside `log_likelihood` must remain differentiable with respect to
`y_pred` and, when used, `theta`. Hard indexing based on an argmax is allowed by
PyTorch but produces piecewise gradients and may degrade HMC. Prefer smooth
features where practical.

Validate observation dimensions, positive scales, covariance definiteness, and
external data paths in the posterior factory so failures happen before a long
run.

## Initial Data Contract

An optional initial-data factory returns either a mapping or a zero-argument
callable that returns a mapping:

```python
def load_initial_data(config, dataset_path):
    data = np.load(dataset_path)
    return {
        "theta": data["theta"],       # (N, parameter_dim)
        "outputs": data["outputs"],   # (N, output_dim)
        "metadata": {"source": dataset_path},
    }
```

Rows are rounded with `active_learning.duplicate_round_decimals` and duplicate
parameters are skipped. Accepted rows are appended in one HDF5 transaction.
Non-finite values and row-count mismatches are rejected.

## Callback Test Checklist

Every problem package should test:

1. Configured factory paths resolve and construct successfully.
2. `prior_transform` maps Latin-hypercube quantiles to the intended prior with
   exact configured shape.
3. `initial_z` and `z_to_theta` produce exact configured shapes.
4. Physical parameters satisfy bounds or other constraints.
5. One solver evaluation produces finite exact-width output.
6. Prior plus likelihood produces one finite value per row.
7. `torch.autograd.grad(log_probability.sum(), z)` is finite.
8. Missing solver assets produce an immediate actionable exception.
9. A very short MPI run creates data, status, and a surrogate checkpoint.

The bundled [synthetic tests](../tests/test_examples.py) are a compact model for
callback tests, and [the MPI test](../tests/test_mpi_smoke.py) demonstrates the
end-to-end gate.
