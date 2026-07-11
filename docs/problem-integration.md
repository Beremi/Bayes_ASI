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
            auxiliary={
                "experiments/source_a/full_curve": full_curve.astype(np.float32),
                "diagnostics/residual_history": residual_history.astype(np.float32),
            },
        )
```

`theta` has shape `(problem.parameter_dim,)`. Returned outputs must contain
exactly `problem.output_dim` finite numeric values. A plain array is accepted as
a shorthand, but `ForwardResult` is preferred.

`outputs` is the dense model vector trained by the surrogate and passed to
`log_likelihood`. `auxiliary` is a mapping of logical names to fixed-shape
numeric or boolean arrays. Auxiliary arrays are persisted row-by-row but never
enter surrogate training or likelihood evaluation. A field may be absent from
some rows; HDF5 stores a presence mask. Shape and dtype are fixed after a field
first appears. Use metadata for small JSON values and `artifact_paths` for
variable-size or externally managed files.

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

## Posterior And Prior Contracts

Bayes ASI owns the inference reference distribution: every internal coordinate
has prior `z ~ N(0, I)`. A posterior supplies a reversible physical-prior
transport and the likelihood:

```python
from dataclasses import dataclass

import torch

from bayes_asi.priors import IndependentPrior, beta, lognormal, normal, uniform


@dataclass(slots=True)
class ProblemPosterior:
    observations: torch.Tensor
    prior: IndependentPrior

    def log_likelihood(self, theta, y_pred):
        residual = y_pred - self.observations.to(y_pred)
        return -0.5 * torch.sum(residual**2, dim=-1)


prior = IndependentPrior(
    [
        uniform(0.1, 2.0),
        beta(2.0, 5.0, low=-1.0, high=3.0),
        normal(mean=0.0, sd=2.0),
        lognormal(log_mean=0.2, log_sd=0.5),
    ]
)
```

`IndependentPrior` also accepts serializable specifications such as
`{"distribution": "exponential", "rate": 2.0}`. Built-in marginals are
Uniform, Beta, Normal, Exponential, LogNormal, Gamma, Weibull, Student-t,
Laplace, and Logistic. `scipy_marginal(frozen_distribution)` supports other
continuous SciPy distributions with `cdf`, `sf`, `ppf`, `isf`, and `logpdf`.

The `PriorTransform` shape contract is:

| Member | Contract |
| --- | --- |
| `latent_dim` | width of the independent standard-normal reference |
| `parameter_dim` | width of physical solver input |
| `to_theta(z)` | differentiable tensor `(N, latent_dim)` to `(N, parameter_dim)` |
| `to_standard_normal(theta)` | NumPy array `(..., parameter_dim)` to `(..., latent_dim)` |

For independent marginals, Bayes ASI composes the standard-normal CDF with each
target inverse CDF. `GaussianCopulaPrior` additionally supplies correlated
marginals. A problem can implement `PriorTransform` directly for a Rosenblatt,
normalizing-flow, matrix, or other reversible generative transport.

The framework evaluates

```text
theta = posterior.prior.to_theta(z)
y_hat = surrogate(z)
log posterior = log N(z; 0, I) + posterior.log_likelihood(theta, y_hat)
```

No user Jacobian or latent log-prior callback is needed: the requested physical
prior is induced by pushing the Gaussian reference through the transport. The
built-in SciPy quantile transport supplies the exact implicit derivative
`d theta / d z = normal_pdf(z) / prior_pdf(theta)` to PyTorch autograd.

`log_likelihood` and custom `to_theta` implementations must preserve the
PyTorch graph. An optional `initial_z(rng, n_chains)` method may return finite
shape `(n_chains, latent_dim)` for dispersed or mode-aware starts; otherwise the
framework draws independent standard normals.

## Bounded Parameters

A bounded independent prior requires no hand-written logistic transform or
Jacobian:

```python
from bayes_asi.priors import IndependentPrior, uniform

prior = IndependentPrior(
    [
        uniform(0.5, 2.5),
        uniform(0.2, 2.0),
        uniform(-0.2, 0.5),
    ]
)
```

For an empty run, a Latin hypercube is generated in independent prior
probability coordinates, mapped to standard-normal `z`, and then transported to
physical parameters. Each bounded marginal therefore retains exactly one
initial point per prior-probability stratum.

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
        "auxiliary": {
            "experiments/source_a/full_curve": data["full_curve"],  # (N, ...)
        },
        "metadata": {"source": dataset_path},
    }
```

Physical rows are mapped back through `prior.to_standard_normal`, rounded with
`active_learning.duplicate_round_decimals`, and duplicate parameters are
skipped. Accepted rows and their selected auxiliary rows are appended in one
HDF5 transaction. Non-finite model values and row-count mismatches are rejected.

## Callback Test Checklist

Every problem package should test:

1. Configured factory paths resolve and construct successfully.
2. `prior.to_theta` and `prior.to_standard_normal` round-trip representative
   values with exact configured shapes.
3. A Latin hypercube retains one point in every prior-probability stratum.
4. Physical parameters satisfy bounds or other constraints.
5. One solver evaluation produces finite exact-width output.
6. Standard-normal prior plus likelihood produces one finite value per row.
7. `torch.autograd.grad(log_probability.sum(), z)` is finite.
8. Auxiliary data is persisted with its intended shape and presence mask but
   does not change the surrogate output width.
9. Missing solver assets produce an immediate actionable exception.
10. A very short MPI run creates data, status, and a surrogate checkpoint.

The bundled [synthetic tests](../tests/test_examples.py) are a compact model for
callback tests, and [the MPI test](../tests/test_mpi_smoke.py) demonstrates the
end-to-end gate.
