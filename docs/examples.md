# Examples

The bundled examples exercise the same callback, storage, surrogate, HMC, and
selection paths as external models. Four are fast synthetic maps; the Darcy
example adds a sparse finite-volume PDE solve and deliberately slow jobs.

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

## Three-Material Darcy Flow

Configuration: `examples/darcy_flow.yaml`

The model solves steady saturated flow on the unit square:

```text
-div(k(x, theta) grad h(x)) = 0
```

It uses a cell-centered finite-volume discretization with a fixed head of one
on the left, zero on the right, and no-flow top and bottom boundaries. Three
log-conductivities control the background, a low-conductivity elliptical lens,
and a diagonal high-conductivity channel. The 13 outputs are 12 interpolated
head sensors and total outlet flux.

The reference configuration uses a 96 by 96 grid and enforces a 1.5-second
minimum callback duration. The delay makes scheduling, queue sizing, deadlines,
and MPI concurrency realistic while keeping the numerical solve reproducible
on ordinary development hardware.

```bash
mpiexec -n 3 bayes-asi run examples/darcy_flow.yaml
```

The numerical tests verify the exact linear solution for spatially constant
conductivity, sensitivity to each material, and recovery of all three planted
log-conductivities from a distant starting point.

## Tests

Callback tests exercise all examples without MPI:

```bash
PYTHONPATH=src python3 -m unittest tests.test_examples
```

They verify exact shapes, finite solver output, finite log probability, finite
autograd gradients, bounds, bimodal symmetry, and Darcy-flow consistency.

The MPI integration test runs all four synthetic examples through the full
system and requires completed samples plus a trained checkpoint:

```bash
BAYES_ASI_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_mpi_smoke
```

The opt-in Darcy MPI test runs three ranks with 1.5-second forward jobs. It
requires completed samples, finite PDE outputs, a trained surrogate checkpoint,
and a saved configuration snapshot:

```bash
BAYES_ASI_DARCY_MPI_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_darcy_mpi
```

Use these examples as starting points for a separate problem package, not as
base classes. The callback protocols are intentionally small enough that direct
problem-specific implementations remain clearer.
