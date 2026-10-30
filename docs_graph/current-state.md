# Current state

Verified 2026-10-30. Branch `feat/multisurrogate-ogs-inversion` extends upstream
`464e54c` for the Mont Terri workflow while preserving the default single-surrogate
configuration.

CI action updates from [PR 1](https://github.com/Beremi/Bayes_ASI/pull/1) and
[PR 2](https://github.com/Beremi/Bayes_ASI/pull/2) are integrated here:
SHA-pinned checkout 7.0.1 and setup-python 7.0.0. The removed setup-python
`pip-install` input was unused. Open MPI 4/5 runner mapping is explicit.

The optional hierarchy trains a strong teacher on completed forward evaluations,
distils a smaller student, uses student HMC to propose candidates, and screens a
bounded diverse subset with the teacher before filling the expensive solver queue.
Pseudo-targets stay outside solver storage. Matching checkpoints carry teacher and
student fingerprints; missing pairs regenerate within the run deadline.

Reusable KL fields, bounded mean-preserving porosity, rotated permeability tensors,
named/fixed parameters, truncated-normal priors and differentiable likelihood
components support external problem adapters. Scientific assets remain external.

The native 207-to-700 adapter imports no surrDAMH code. All 1,980 archived prior
round trips passed; 20 KL/likelihood reference cases agreed, and a new 730-day
native OGS job reproduced its archived target within 1.49e-14 log10 units.
Large model/data/run assets remain external.

[Verification](verification.json) records 106 unit/real-fixture tests (three MPI
gates skipped in that command), separately passing synthetic/Darcy/hierarchy MPI
tests, installed-wheel MPI, static checks and packaging. The local Python 3.14
environment reuses system scientific packages; unrelated inherited desktop
requirements prevent a globally clean `pip check`, while package requirements pass.

The built-in hierarchy uses MLP teacher/student models. A graph/PINN teacher
factory and other historical OGS schemas remain extensions. No new posterior
or scientific stream activation is claimed. The hierarchy is approximate
adaptive design, not exact delayed acceptance.
