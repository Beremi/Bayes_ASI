# Synthetic Examples

The bundled examples are small enough for local MPI runs but exercise the same
callback, storage, surrogate, HMC, and selection paths as external models.

Validate all configurations:

```bash
for config in examples/*.yaml; do
  bayes-asi validate "$config"
done
```

## Nonlinear Unconstrained

Configuration: `examples/synthetic.yaml`

Parameters are unconstrained `(theta_0, theta_1)`. The forward map is:

```text
y_0 = sin(theta_0) + 0.2 theta_1^2
y_1 = theta_0 theta_1 + 0.1 theta_0
```

The prior is isotropic Gaussian and observations are generated at
`theta = (0.7, -1.2)` with independent Gaussian likelihood scales.

This example demonstrates nonlinear surrogate learning and direct identity
mapping from latent to physical coordinates.

```bash
mpiexec -n 3 bayes-asi run examples/synthetic.yaml
```

## Linear-Gaussian

Configuration: `examples/linear_gaussian.yaml`

Two parameters produce three observations:

```text
y = A theta + b
```

with a fixed full-column-rank design matrix. Gaussian priors and observation
errors make this a well-behaved reference problem whose exact posterior is also
Gaussian.

This example is useful for:

- checking that HMC means/covariances are plausible;
- testing an output dimension larger than parameter dimension;
- benchmarking surrogate overhead on an easy map.

```bash
mpiexec -n 3 bayes-asi run examples/linear_gaussian.yaml
```

## Bounded Exponential Decay

Configuration: `examples/bounded_decay.yaml`

The physical parameters are amplitude, decay rate, and offset:

```text
y(t) = amplitude * exp(-rate * t) + offset
```

at 16 fixed time points. Each parameter has finite lower and upper bounds. HMC
uses unconstrained `z`, mapped by a logistic transform. The transformed uniform
prior includes the required Jacobian term.

This is the most representative template for bounded physical calibration and
vector/curve outputs.

```bash
mpiexec -n 3 bayes-asi run examples/bounded_decay.yaml
```

## Bimodal Square Map

Configuration: `examples/bimodal.yaml`

The one-dimensional solver is:

```text
y = theta^2
```

An observation near `1.5^2` creates symmetric posterior modes near `-1.5` and
`+1.5`. Alternating chains start near both signs.

This example demonstrates an important limitation: fixed-trajectory HMC can
sample a local mode well but does not guarantee movement between separated
modes. Multiple dispersed chains are essential.

```bash
mpiexec -n 3 bayes-asi run examples/bimodal.yaml
```

## Tests

Callback tests exercise all examples without MPI:

```bash
PYTHONPATH=src python3 -m unittest tests.test_examples
```

They verify exact shapes, finite solver output, finite log probability, finite
autograd gradients, bounds, and bimodal symmetry.

The MPI integration test runs every example through the full system and
requires completed samples plus a trained checkpoint:

```bash
BAYES_ASI_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_mpi_smoke
```

Use these examples as starting points for a separate problem package, not as
base classes. The callback protocols are intentionally small enough that direct
problem-specific implementations remain clearer.
