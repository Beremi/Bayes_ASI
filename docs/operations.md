# Operations and Troubleshooting

## Launching MPI

Use at least two ranks:

```bash
mpiexec -n 8 bayes-asi run path/to/problem.yaml
```

This creates one coordinator and seven solver workers. Avoid requesting more
workers than the external solver license, memory, filesystem, or CPU allocation
can support. Solver callbacks should set their own thread counts when each MPI
rank launches a threaded native application.

Only one coordinator may own an output directory. A second launch fails with
the host and process recorded in `run.lock`; the file itself remains after a
clean exit, but the advisory lock is released.

## Watching a Run

Read the atomic status snapshot:

```bash
bayes-asi status runs/my_run
```

Watch durable events:

```bash
tail -f runs/my_run/events.jsonl
```

Typical healthy progression is:

```text
run_started
job_queued / job_started / job_completed
surrogate_trained
active_batch_selected
job_queued / ...
budget_reached
run_finished
```

`surrogate_train_failed` or `active_selection_failed` records include a
traceback and retry delay. Repeated failures after backoff need investigation.

## Time-Budget Semantics

The budget starts when `Coordinator.run` begins, after configuration, callback,
storage, and restart-state construction.

With `drain_running_on_exit: true`:

- no new jobs are assigned after the deadline;
- surrogate training and HMC stop at deadline checkpoints;
- running jobs finish and are stored;
- pending jobs remain journaled for a future resume;
- workers receive a normal stop message.

With `drain_running_on_exit: false`:

- final status and a stop event are written at the deadline;
- the MPI communicator is aborted immediately;
- external child processes may need scheduler or callback-level cleanup.

Use hard stop only when exceeding allocation time is worse than abrupt solver
termination.

## Restarting

To resume, use the same output directory and compatible problem dimensions:

```bash
mpiexec -n 8 bayes-asi run path/to/problem.yaml
```

Restart reconstructs pending jobs from journal history. Jobs are terminal after
completion, permanent failure, or cancellation. Completed parameter keys and
duplicate pending keys are excluded. Retry attempts survive a crash, so restart
cannot reset a job's retry limit. Active jobs tied to a surrogate version other
than the loaded checkpoint are cancelled and replaced.

Each invocation copies its YAML to `configs/<sha256>_<name>.yaml` and
records the full digest in `run_started`. Use that snapshot when auditing which
configuration produced a segment of a resumed run.

Safe changes for a resumed run generally include:

- extending `time_budget_seconds`;
- changing status interval;
- changing queue depth or target seconds ahead;
- changing training epoch/patience settings;
- changing active batch/candidate sizes.

Changes requiring scientific review include:

- prior, transform, likelihood, observations, or solver behavior;
- parameter/output meanings while retaining the same dimensions;
- duplicate rounding;
- output normalization assumptions.

Bayes ASI validates dimensions but cannot determine whether scientific meaning
changed. Prefer a new output directory for a different posterior or model.

## Checkpoint Recovery

Checkpoints are tried newest first. A bad checkpoint produces
`surrogate_checkpoint_load_failed`; an older valid checkpoint may then load.
The next successful training overwrites the failed version atomically.

Do not manually rename checkpoints to misleading version numbers. Model version
is stored inside the file and also drives queued-job provenance.

## Solver Failures

Inspect `job_failed` events for:

- `error` and `traceback`;
- worker rank and elapsed time;
- attempt number and `will_retry`;
- durable job sequence.

Each retry uses the same job ID and parameter vector. It gets a fresh callback
invocation but reuses the same job work directory. A callback should overwrite
or explicitly clean stale attempt output as appropriate.

Permanent failures remain known parameter keys during the run, preventing
active learning from immediately selecting the same point again.

## Storage Diagnostics

Use h5py to inspect dimensions without loading outputs:

```python
import h5py

with h5py.File("runs/my_run/data.h5", "r") as h5:
    print(dict(h5.attrs))
    print(h5["completed/theta"].shape)
    print(h5["completed/outputs"].shape)
```

All datasets in `completed` must have the same first dimension. Bayes ASI
refuses to open an inconsistent file. Preserve the original before attempting
manual HDF5 repair.

## Export

Default destinations are `completed.npz` or `completed.csv` in the run
directory:

```bash
bayes-asi export runs/my_run --format npz
bayes-asi export runs/my_run --format csv --output analysis/samples.csv
```

NPZ is compressed and preserves array dtypes/shapes. CSV is convenient for
small flat outputs but can become very wide for field or curve predictions.

## Common Problems

### `requires at least 2 MPI ranks`

The command was run directly or with `-n 1`. Use one coordinator plus workers.

### Factory module cannot be imported

Install the problem package or add its `src` directory to `PYTHONPATH` on every
rank. Validate with `bayes-asi validate` in the same environment.

### MPI ranks use different Python environments

Launch `python -m bayes_asi.cli` with an absolute interpreter path or activate
the environment in the scheduler script before `mpiexec`.

### CUDA requested but CPU is used

CUDA requests fall back to CPU when `torch.cuda.is_available()` is false. Check
the compute-node environment, CUDA build, and scheduler GPU allocation.

### Training fails repeatedly

Check for non-finite data, too few distinct parameter values, incompatible old
checkpoints, GPU memory pressure, or callback output changes. The event log
contains the traceback and completed row count.

### HMC acceptance is near zero

Reduce `initial_step_size`, inspect posterior gradients, increase warmup, or
reparameterize correlated/bounded variables. Verify surrogate accuracy near
posterior mass before trusting active selections.

### HMC misses a mode

Increase chains and initialize chains across plausible basins. HMC does not
normally cross strongly separated modes. The bimodal example intentionally
initializes alternating chains near both signs.

### Event journal has an incomplete final line

Restart tolerates one truncated final line, which can result from abrupt
termination. Malformed earlier lines are treated as corruption because replay
cannot safely infer missing state.
