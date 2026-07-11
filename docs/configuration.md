# Configuration Reference

Bayes ASI reads one YAML document. Unknown top-level keys and unknown keys in
known sections are rejected, which makes spelling mistakes fail before MPI
work begins.

Required sections are `problem`, `solver`, and `posterior`. All other sections
have defaults.

## Complete Template

```yaml
run:
  output_dir: runs/default
  time_budget_seconds: 3600.0
  drain_running_on_exit: true
  seed: 12345
  status_interval_seconds: 5.0

problem:
  latent_dim: 2
  parameter_dim: 2
  output_dim: 10

solver:
  factory: my_problem.callbacks:make_solver
  kwargs: {}

posterior:
  factory: my_problem.callbacks:make_posterior
  kwargs: {}

initial_data:
  factory: my_problem.callbacks:load_initial_data
  kwargs: {}

initial_design:
  n_points: 16

surrogate:
  hidden_layers: [64, 64]
  learning_rate: 0.001
  weight_decay: 0.0
  max_epochs: 500
  batch_size: 128
  validation_fraction: 0.2
  patience: 50
  scheduler_factor: 0.5
  scheduler_patience: 20
  min_train_samples: 16
  update_after_new_samples: 4
  retrain_from_scratch_every: 5
  scratch_improvement_rel: 0.02
  device: cpu
  gradient_epsilon: 0.000001
  failure_retry_seconds: 30.0

hmc:
  n_chains: 4
  warmup_steps: 100
  samples_per_chain: 200
  target_accept: 0.8
  initial_step_size: 0.05
  min_leapfrog_steps: 2
  max_leapfrog_steps: 32
  gamma: 0.05
  t0: 10.0
  kappa: 0.75

active_learning:
  candidate_pool_size: 512
  batch_max_size: 8
  posterior_weight: 1.0
  distance_weight: 0.6
  queued_repulsion_radius_multiplier: 2.0
  duplicate_round_decimals: 12
  failure_retry_seconds: 5.0

queue:
  min_pending_per_worker: 1
  max_pending_per_worker: 2
  refill_threshold_pending_per_worker: 1
  target_seconds_ahead: 60.0
  retry_limit: 1
```

Omit `initial_data` when no precomputed evaluations are available.

## `run`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `output_dir` | `runs/default` | path | Run database, journal, checkpoints, status, and worker directories. Relative paths use the process working directory. |
| `time_budget_seconds` | `3600.0` | `> 0` | Wall-clock budget measured by rank 0 after coordinator construction. |
| `drain_running_on_exit` | `true` | YAML boolean | If true, stop queueing at budget and wait for running jobs. If false, write status and call `MPI.Abort` at budget. |
| `seed` | `12345` | integer | Root seed for initial design, training splits, model initialization, HMC, and candidate subsampling. |
| `status_interval_seconds` | `5.0` | `> 0` | Minimum interval between atomic `status.json` updates. |

Quoted strings such as `"false"` are not accepted as booleans.

## `problem`

All options are required and positive integers.

| Option | Meaning |
| --- | --- |
| `latent_dim` | Width of the independent standard-normal coordinate `z` used by design, surrogate, HMC, and selection. |
| `parameter_dim` | Width of physical solver input `theta`. |
| `output_dim` | Width of the flattened forward-model output and surrogate prediction. |

The runtime reshapes callback values to these dimensions and fails on
incompatible sizes.

## Factory Sections

`solver`, `posterior`, and optional `initial_data` share this schema:

| Option | Required | Meaning |
| --- | --- | --- |
| `factory` | yes | Callable path as `package.module:callable` or `package.module.callable`. |
| `kwargs` | no | Mapping passed to the factory after `config=app_config`. |

Factories are called independently in each process that needs them. The
posterior and initial-data factories run on rank 0. The solver factory runs on
every worker rank.

## `initial_design`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `n_points` | `16` | `>= 0` | Target number of completed plus queued/running initial evaluations before surrogate-driven selection. |

Bayes ASI generates a randomized Latin hypercube in the open unit cube, applies
the standard-normal inverse CDF, and calls `posterior.prior.to_theta` to obtain
solver parameters. Imported physical parameters are mapped back with
`prior.to_standard_normal`. Set zero when existing or imported data already
supply enough training rows and no additional prior design is desired.

When no checkpoint or sufficient completed/imported data exist, the projected
initial rows must reach `surrogate.min_train_samples`. The runtime rejects an
undersized design immediately instead of waiting idly until the time budget.

## `surrogate`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `hidden_layers` | `[64, 64]` | every width `> 0`; empty list allowed | Tanh MLP hidden widths. An empty list creates a linear network. |
| `learning_rate` | `0.001` | `> 0` | Adam learning rate. |
| `weight_decay` | `0.0` | `>= 0` | Adam L2 weight decay. |
| `max_epochs` | `500` | `> 0` | Maximum epochs per training attempt. |
| `batch_size` | `128` | `> 0` | Training mini-batch size. Values above the training count produce full-batch training. |
| `validation_fraction` | `0.2` | `[0, 1)` | Deterministic held-out fraction. Runs with fewer than five rows use training loss. |
| `patience` | `50` | `> 0` | Epochs without metric improvement before early stopping. |
| `scheduler_factor` | `0.5` | `(0, 1)` | `ReduceLROnPlateau` learning-rate multiplier. |
| `scheduler_patience` | `20` | `>= 0` | Plateau epochs before reducing learning rate. |
| `min_train_samples` | `16` | `>= 2` | Completed rows required for the first surrogate. |
| `update_after_new_samples` | `4` | `> 0` | New completed rows required before updating an existing surrogate. |
| `retrain_from_scratch_every` | `5` | `>= 0` | Compare a scratch model every N training cycles. Zero disables scratch comparisons. |
| `scratch_improvement_rel` | `0.02` | `[0, 1)` | Relative validation-loss improvement required to prefer scratch over warm start. |
| `device` | `cpu` | nonempty PyTorch device | Training and HMC device, for example `cpu`, `cuda`, or `cuda:0`. CUDA requests fall back to CPU if CUDA is unavailable. |
| `gradient_epsilon` | `1e-6` | `> 0` | Standard deviations below this threshold are replaced by one during standard-normal input or model-output normalization. |
| `failure_retry_seconds` | `30.0` | `>= 0` | Backoff after a failed training attempt. A newly completed solver row permits an immediate retry. |

Warm starts adjust the first and last linear layers when normalization changes,
preserving physical predictions before optimization resumes.

## `hmc`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `n_chains` | `4` | `> 0` | Parallel HMC chains represented in one tensor batch. |
| `warmup_steps` | `100` | `>= 0` | Dual-averaging and diagonal-mass adaptation steps. |
| `samples_per_chain` | `200` | `> 0` | Post-warmup iterations retained for each chain. |
| `target_accept` | `0.8` | `(0, 1)` | Dual-averaging acceptance target. |
| `initial_step_size` | `0.05` | `> 0` | Initial leapfrog step size in latent coordinates. |
| `min_leapfrog_steps` | `2` | `> 0` | Minimum trajectory step count. |
| `max_leapfrog_steps` | `32` | `>= min` | Maximum trajectory step count. |
| `gamma` | `0.05` | `> 0` | Dual-averaging adaptation scale. |
| `t0` | `10.0` | `> 0` | Dual-averaging stabilization offset. |
| `kappa` | `0.75` | `(0, 1]` | Dual-averaging decay exponent. |

Adapted step sizes are bounded to `[1e-6, 10]`. Warmup uses randomized
trajectory lengths; sampling jitters around a tuned length.

## `active_learning`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `candidate_pool_size` | `512` | `> 0` | Maximum HMC rows considered in one selection. |
| `batch_max_size` | `8` | `> 0` | Maximum solver jobs selected in one refill. |
| `posterior_weight` | `1.0` | `>= 0` | Weight of standardized surrogate log posterior. |
| `distance_weight` | `0.6` | `>= 0` | Weight of standardized nearest-neighbor distance. |
| `queued_repulsion_radius_multiplier` | `2.0` | `> 0` | Makes queued/running/newly selected points repel candidates more strongly than completed points. |
| `duplicate_round_decimals` | `12` | `>= 0` | Decimal rounding used for durable duplicate keys. |
| `failure_retry_seconds` | `5.0` | `>= 0` | Backoff after HMC/selection failure or a zero-row selection. |

At least one of the two score weights must be positive. Candidates with a
non-finite log probability cannot be selected when posterior weight is active.

## `queue`

| Option | Default | Constraint | Meaning |
| --- | ---: | --- | --- |
| `min_pending_per_worker` | `1` | `>= 0` | Lower pending-work target used before solve-time throughput is known. |
| `max_pending_per_worker` | `2` | `>= min` | Upper pending-work target. |
| `refill_threshold_pending_per_worker` | `1` | `[0, max]` | Start a surrogate selection when pending jobs fall to this threshold times worker count. |
| `target_seconds_ahead` | `60.0` | `>= 0` | Desired queued solver time based on worker throughput and solve-time EWMA. |
| `retry_limit` | `1` | `>= 0` | Number of retries after the first failed solver attempt. |

The desired in-system count is the number of workers plus a bounded pending
count. Running jobs are never cleared when a new surrogate replaces pending
active-learning jobs.

## Validation

Run:

```bash
bayes-asi validate path/to/config.yaml
```

This confirms schema and callable resolution without constructing solver or
posterior objects. Callback-specific file checks belong in the callback
factories and occur when a run starts.
