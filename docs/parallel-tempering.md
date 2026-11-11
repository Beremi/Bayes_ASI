# Parallel-tempered HMC

The standalone Python sampler runs one ladder of HMC replicas and exchanges
neighboring states. Replicas are batched in PyTorch on the selected device;
this API does not assign temperatures to MPI ranks. The YAML/MPI inversion
workflow continues to use `HMCSampler`.

## Target and callback contract

Import `ParallelTemperedHMCConfig`, `ParallelTemperedHMCSampler`, and
`ParallelTemperedHMCResult` from `bayes_asi` (or `bayes_asi.tempered_hmc`).
The callback receives a tensor of shape `(replicas, dimension)` and returns
`(log_reference, log_tempered)`, each with shape `(replicas,)`. Each row must
depend only on the corresponding input row, using the same density functions
for every row (no dependence on replica index). Use deterministic, differentiable
PyTorch operations on the supplied device and dtype; do not detach the target
from autograd. Both components and their gradients must be finite throughout
sampling; hard-support likelihoods represented by negative infinity are not
supported, including at beta zero. Use smooth targets in unconstrained
coordinates. These are log densities, not their negative energies.

Replica `r` targets

```text
log pi_r(z) = log_reference(z) + beta_r * log_tempered(z) + constant.
```

For Bayesian inversion, return the log prior in the sampling coordinates and
the log likelihood. When sampling in Bayes ASI standard-normal coordinates,
the reference is `-0.5 * z.square().sum(dim=-1)`, and the likelihood uses
`prior.to_theta(z)` and the surrogate prediction. Do not return the full log
posterior as the second component: that would count the prior twice at beta 1.
Every replica target must be normalizable. In particular, beta 0 requires a
proper reference density. Additive normalization constants can be omitted.

## Example

```python
import numpy as np
import torch
from bayes_asi import ParallelTemperedHMCConfig, ParallelTemperedHMCSampler


def log_density(z):
    log_prior = -0.5 * z.square().sum(dim=-1)
    log_likelihood = -0.5 * ((z - 2.0) / 0.5).square().sum(dim=-1)
    return log_prior, log_likelihood


config = ParallelTemperedHMCConfig(
    inverse_temperatures=(1.0, 0.5, 0.2, 0.0),
    warmup_steps=300,
    samples=500,
)
sampler = ParallelTemperedHMCSampler(
    config, log_density, seed=42, dtype=torch.float64,
)
result = sampler.sample(np.zeros((4, 1)))
posterior_samples = result.cold_z  # (500, 1), beta = 1 only
```

A two-mode example is included in
[`examples/parallel_tempering.py`](../examples/parallel_tempering.py):

```bash
PYTHONPATH=src python examples/parallel_tempering.py
```

## Configuration and adaptation

The default ladder has ten inverse temperatures from 1 to 0. A custom ladder
must contain at least two finite, strictly decreasing values in `[0, 1]`,
starting at 1; ending at 0 is optional. `initial_z` must contain one finite
row per replica.

Warmup defaults to 300 transitions, followed by 500 retained samples with
`thinning=1`. Neighbor swap attempts occur every `swap_interval` transitions
(default 1), alternating even and odd edges. With two replicas, every other
swap event has no eligible edge. Swaps exchange states and labels, while
step sizes and mass matrices remain attached to temperatures.

During warmup, the sampler adapts step sizes by dual averaging, estimates a
diagonal precision mass halfway through warmup (when warmup has at least 20
steps), and selects leapfrog counts using squared jumps per leapfrog step.
Production freezes adaptation and optionally jitters the selected counts
independently of the current state. `warmup_steps=0` disables adaptation.

Optional `sample` arguments `initial_step_sizes`, `initial_mass_diag`, and
`initial_leapfrog_steps` supply per-replica tuning values. Their shapes are
`(replicas,)`, `(replicas, dimension)`, and `(replicas,)`. They are initial
values, not a full restart checkpoint. `deadline_monotonic` is an absolute
`time.monotonic()` deadline; expiry raises `TimeoutError` without returning
partial samples. The callback itself cannot be interrupted by the deadline.

## Results and diagnostics

- `cold_z`: `(samples, dimension)` posterior draws at beta 1.
- `cold_log_prob`: `(samples,)` unnormalized cold log density.
- `replica_z`: `(samples, replicas, dimension)`, including hot replicas.
- `replica_reference_log_prob`, `replica_tempered_log_prob`, and
  `replica_labels`: `(samples, replicas)`. Labels track state movement.
- `accept_rate` and `warmup_accept_rate`: per-temperature HMC acceptance.
- `swap_accept_rate` and `warmup_swap_accept_rate`: per-neighbor-edge rates;
  an edge with no attempts has `NaN`, not zero acceptance.
- `step_size`, `mass_diag`, and `leapfrog_steps`: final tuning values.
- `leapfrog_efficiency`: warmup score for each selected count, or `NaN` if
  that count was not scored.
- `final_z`: last state at each temperature; timing and target-evaluation
  counts are also returned. A target evaluation is one batched callback call,
  not one replica evaluation.

Only cold draws represent the requested posterior. Hot draws must not be
pooled into posterior estimates. Passing the Gaussian and two-mode tests
checks bounded examples; it does not establish convergence for a new target.
Inspect mixing, swap rates, and repeated independently seeded runs.
