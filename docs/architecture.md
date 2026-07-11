# Architecture and Algorithms

## Process Ownership

Bayes ASI uses a single-writer architecture.

- Rank 0 owns queue state, random generation, HDF5 writes, journal writes,
  status files, surrogate training, HMC, and active selection.
- Worker ranks own their solver instances and job working directories.
- Workers communicate only results and structured failures to rank 0.

This avoids HDF5 write contention and makes event replay deterministic.

## Coordinator Lifecycle

At startup rank 0:

1. Loads and validates YAML, then acquires an exclusive run-directory lock.
2. Creates or validates the run-directory HDF5 schema.
3. Migrates schema-one physical samples to standard-normal coordinates when
   needed, using the configured reversible prior.
4. Loads completed sample keys and count.
5. Reads valid journal events, ignoring only a truncated final line.
6. Tries checkpoints newest first until a compatible valid model loads.
7. Replays queued, nonterminal, nonduplicate jobs, preserving retry attempts.
8. Cancels active jobs whose source version differs from the loaded surrogate.
9. Saves a content-addressed configuration snapshot and journals its SHA-256.
10. Imports optional initial data in one transaction.
11. Queues missing prior-design points.

For a fresh design, Bayes ASI creates one randomized point in every equal-width
prior-probability stratum, maps it through the standard-normal inverse CDF, and
then applies `posterior.prior.to_theta`. Thus design, surrogate training, HMC,
and active-distance calculations share independent `N(0, I)` coordinates while
workers receive physical parameters. Optional `initial_z` is reserved for HMC
chain initialization and may use dispersed or mode-aware starts.

The run loop then:

1. Receives every currently available worker result.
2. Trains or updates the surrogate when enough new rows exist.
3. Stops queueing when the time budget expires.
4. Refills the active queue when it falls below the configured threshold.
5. Assigns pending jobs to idle workers.
6. Atomically updates status on schedule.

The completed-row count is cached in memory. Full HDF5 arrays are loaded only
when training or selection needs them, rather than on every 20 ms loop pass.

## Design Rationale

The prior layer follows the measure-transport formulation of coupling a simple
reference measure, commonly a standard Gaussian, to a target distribution; see
[Marzouk et al., *An introduction to sampling via measure transport*](https://arxiv.org/abs/1602.05023).
For scalar marginals, SciPy's documented
[`rv_continuous` interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html)
provides CDF, inverse-CDF, survival, and density operations across its standard
continuous distributions. Beta inversion uses SciPy's
[`betaincinv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betaincinv.html),
which wraps the established Boost implementation.

Rich result storage remains in HDF5 because the runtime already has one writer
and needs atomic row-aligned extension. h5py documents that
[chunked datasets are resizable](https://docs.h5py.org/en/stable/high/dataset.html)
and support transparent filters. LZF, byte shuffling, and Fletcher32 checksums
give fast lossless storage without another runtime dependency. An analysis
package may expose these groups as xarray objects later; xarray's
[`DataTree`](https://docs.xarray.dev/en/latest/user-guide/data-structures.html)
is suited to heterogeneous hierarchy, but it is not needed in the execution
path.

## Persistence

### HDF5

`data.h5` stores a `completed` group with equal-length datasets:

- `z`: `(N, latent_dim)` float64 standard-normal coordinates;
- `theta`: `(N, parameter_dim)` float64;
- `outputs`: `(N, output_dim)` float64 dense model values used by the surrogate;
- `elapsed_s`: `(N,)` float64;
- `worker_rank`: `(N,)` int64;
- `job_sequence`: `(N,)` int64;
- `started_at` and `finished_at`: `(N,)` float64 Unix timestamps.

File attributes record schema version and dimensions. Opening an existing run
with different dimensions or inconsistent dataset lengths fails immediately.
An `auxiliary` group stores each logical field as a chunked, losslessly
compressed `values` dataset and a row-aligned boolean `present` mask. Logical
names may be hierarchical, for example `experiments/source_a/full_curve`.
Fields can appear after a run starts; previous rows are marked absent. Shapes
and dtypes cannot change after first use. Batch append resizes completed and
auxiliary datasets together and rolls all dimensions back if any write fails.

### Event Journal

`events.jsonl` is append-only and fsynced per event. Important event types
include:

- run: `run_started`, `budget_reached`, `run_finished`;
- queue: `job_queued`, `job_started`, `job_completed`, `job_failed`,
  `job_cancelled`;
- surrogate: `surrogate_loaded`, `surrogate_trained`,
  `surrogate_train_failed`, `surrogate_checkpoint_load_failed`;
- active learning: `active_batch_selected`, `active_selection_failed`;
- import: `initial_sample_imported`, `initial_data_loaded`;
- design: `initial_design_generated` with method and requested/queued counts.

A crash can leave one incomplete final JSON line. Replay ignores that line but
rejects malformed earlier lines, since silently skipping history could revive
terminal jobs.

### Checkpoints

Surrogate checkpoints contain architecture, state dictionary, normalization,
standard-normal input-space identity, validation loss, model version, and
training-sample count. Saves use a temporary file followed by `os.replace`.

On restart, corrupt, dimension-incompatible, or legacy physical-input
checkpoints are logged and the next older checkpoint is attempted. Physical
checkpoints are retrained because silently applying them to `z` would represent
a different function.

## Surrogate Model

The surrogate is a configurable sequence of linear layers and tanh activations,
with a final linear output. Its input is the standard-normal coordinate `z`,
and its target is only `completed/outputs`; auxiliary fields are never included.
Training standardizes each coordinate and model-output component. Near-constant
dimensions use scale one to avoid division by zero.

Training uses Adam, optional weight decay, deterministic split/order seeds,
`ReduceLROnPlateau`, early stopping, and restoration of the best epoch.

When new data change normalization, a naive state-dictionary warm start changes
the represented physical function before training. Bayes ASI analytically
adjusts the first and final linear layers so predictions in physical units are
preserved under the new normalization. Periodic scratch training can replace a
warm model only when it exceeds the configured relative improvement.

## HMC

HMC operates in latent coordinates `z` and evaluates:

```text
theta = posterior.prior.to_theta(z)
y_hat = surrogate(z)
log p(z | data) = log N(z; 0, I)
                  + posterior.log_likelihood(theta, y_hat)
```

Warmup uses dual averaging for step size and empirical per-dimension variance
for a diagonal mass matrix. Leapfrog counts vary during warmup. Sampling
jitters around a trajectory count inferred from warmup radius. Adapted step
size is bounded to protect against numerical overflow.

Saved posterior batches include latent and physical samples, log probability,
production and warmup acceptance rates, step size, leapfrog count, mass
diagonal, and surrogate version.

This is a compact custom HMC implementation, not NUTS. Users should inspect
acceptance rates and posterior stability, especially for multimodal,
discontinuous, or strongly correlated problems.

Surrogate training checks the run deadline between batches, and HMC checks it
between leapfrog steps. A coordinator does not start another expensive phase
after the deadline. Running external solver callbacks remain governed by the
configured drain or abort policy because ordinary MPI messages cannot interrupt
a callback safely.

## Active Selection

The candidate pool is drawn from the surrogate posterior. For each candidate,
Bayes ASI computes:

```text
score = posterior_weight * zscore(log_posterior)
      + distance_weight  * zscore(nearest_scaled_distance)
```

Distances are scaled by candidate-pool standard deviation per standard-normal
coordinate.
Queued, running, and newly selected points have stronger repulsion through
`queued_repulsion_radius_multiplier`.

Completed and queued distances are computed once per batch. Each greedy
selection updates distance only against the new point. Large completed sets are
processed in chunks, bounding pairwise-memory use. Rounded parameter keys
prevent duplicate selection across completed, pending, running, and replayed
jobs.

## Queue Sizing

Before timing data exist, pending work uses `min_pending_per_worker`. Once solver
times are known, an exponentially weighted solve time estimates worker
throughput. Desired work ahead is:

```text
ceil(target_seconds_ahead * worker_count / solve_time_ewma)
```

bounded by configured minimum and maximum pending counts. The coordinator adds
the worker count to cover running capacity.

When a new surrogate is trained, pending active jobs from the previous model
are cancelled and replaced. Running jobs continue because external solvers
cannot be safely interrupted through ordinary MPI messages.

## Failure Containment

- Worker exceptions become journaled job failures and may retry.
- Non-finite solver parameters or outputs are rejected.
- Training and active-selection failures have configurable backoff.
- A newly completed sample permits immediate training retry.
- Unexpected fatal exceptions call `MPI.Abort(1)` so workers do not wait
  indefinitely.
- Status and posterior-batch files use atomic replacement.
- A POSIX advisory lock rejects a second coordinator for the same run directory.

See [Operations](operations.md) for operator-facing recovery procedures.
