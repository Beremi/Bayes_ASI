# External OGS inversion workspace

The optional `bayes_asi.integrations.ogs` adapter connects the existing OGS
support-response workflow to Bayes ASI. It imports the external model kernels
lazily; it does not import surrDAMH or bundle scientific data. Install the
additional model-kernel dependencies with `pip install -e '.[ogs]'`.

## Supported contract

The first adapter supports the existing **domain-KL, no-Archie input vector**:

| Ordered block | Columns |
| --- | ---: |
| Permeability mean, 100 KL coefficients, bedding angle, principal-value ratio | 103 |
| Mean porosity and 100 porosity KL coefficients | 101 |
| Wall-curve shift, suction scale and suction offset | 3 |
| Total physical parameters | 207 |

The output has 20 supports by 35 dates, flattened in support-major order. Each
value is the support-area average of **log10 cell water content**. It is not the
logarithm of averaged water content. Archie coefficients are profiled in the
likelihood; they are not hidden additional surrogate inputs.

The adapter checks complete feature names/order, support IDs, dimensions,
available time coverage and finite data. The 247-parameter sampled-Archie and
other split-direction/angle-field schemas require separate adapters; they are
rejected rather than interpreted positionally.

## External inputs

All factories take an explicit `workspace` and these paths, absolute or relative
to that workspace:

- `config_path`: the existing model/likelihood/prior JSON.
- `dataset_path`: NPZ with `X`, `feature_names`, `theta_log10`, support IDs,
  model days and observed resistivity.
- `source_result_dir` and `source_candidate_id`: the external reference campaign.

The workspace supplies the existing model, KL basis, observations, OGS launcher,
and `scripts/run_ogs_*` kernels. No scientific asset is copied to this repository.

| Factory | Behavior |
| --- | --- |
| `make_posterior` | Reconstruct named physical priors and the differentiable likelihood. |
| `make_solver` | Execute one native OGS evaluation in the worker's isolated directory. |
| `load_initial_data` | Import exact archived `X`/`theta_log10` pairs as completed forward evaluations. |

`make_solver` defaults to 730 days. `timeout_s` bounds its OGS subprocess and
`ogs_backend` selects the existing native/docker/auto launcher. Every attempt
gets a unique directory, including replay after interruption. Preparation,
timeout, skipped execution, nonzero exit or incomplete outputs cause failure;
none becomes a zero-valued successful training row.

## Priors and likelihoods

The adapter preserves configured prior means, scale multipliers and bounds.
Normal coefficients use normal transports, the flat principal-value ratio uses
a uniform transport, and bounded normals use truncated-normal transports. A
fixed physical coordinate is excluded from latent inference through
`NamedParameterSpace`. Stored/imported rows must agree with the configured prior.

The Torch box-constrained affine likelihood profiles `a+b*log10(theta)` with the
legacy coefficient bounds, per support or globally. Raw log-resistivity and the
existing fixed-period coefficient likelihood are supported. This is conditional
profiling, not integration over petrophysical uncertainty. The numerical
implementation is checked against independent SciPy fits and finite differences.

The output contract is explicitly named `legacy_support_logtheta`: the adapter
preserves the legacy secondary-saturation extraction and clipping behavior so
new and imported training targets mean the same thing. The later primary-pressure
retention reconstruction must be introduced as a **new output/data version**,
with archive regeneration or a documented compatibility study. Backend migration
does not silently resolve that scientific distinction.

## Local preflight and one-job verification

With the existing external workspace mounted:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/verify_ogs_adapter.py \
  --workspace /path/to/inversion_atempt --output-dir /path/to/local-verification
```

This creates a local `ogs_hierarchy.yaml`, checks all 1,980 archived parameter
round trips, and compares selected KL fields and likelihoods against the legacy
field function and independent SciPy fits. Add `--run-solver --timeout-s 1800`
to execute **one** native job. It writes a local verification record and model
artifacts; the comparison reports any difference from the corresponding archived
target and requires parity within `--parity-atol` (default `1e-8` in log10 water
content). It does not launch an inversion campaign.

Then validate or run the generated configuration:

```bash
bayes-asi validate /path/to/local-verification/ogs_hierarchy.yaml
mpiexec -n 3 bayes-asi run /path/to/local-verification/ogs_hierarchy.yaml
```

The generated one-hour budget is an editable example. Choose resources, prior
coverage, observation policies and evaluation budget before a production run.
It reproduces an existing **conditional ERT experiment**, not an approved joint
NMR/permeability/ERT objective. Bayes ASI does not automatically activate additional
measurement families or convert previous chains/checkpoints into a new posterior.

## Other inversion representations

`bayes_asi.fields.KLField` accepts already-scaled external bases, avoiding dense
covariance assembly on large meshes. It also provides small-matrix PSD
decomposition and bounded, mean-preserving logistic fields. The nonlinear
bounded field is not a Gaussian field. Rotated permeability tensors explicitly
use counterclockwise principal-axis angles.

`NamedParameterSpace` provides checked column order and fixed parameters.
`GaussianLikelihood` supports weighted Gaussian observation components for
additional problem callbacks. NMR centering, interval supports and their shared
errors belong in those scientific callbacks, with independently accepted data
policies. Generic callback support is not a completed scientific operator.
