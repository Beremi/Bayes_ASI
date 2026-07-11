# Getting Started

## 1. Prepare Python and MPI

Confirm that Python and MPI are available:

```bash
python3 --version
mpiexec --version
```

Create an environment and install Bayes ASI:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

`mpi4py` must be built or installed against the MPI implementation used by
`mpiexec`.

Install the optional SciPy dependency to run the Darcy-flow example:

```bash
python3 -m pip install -e '.[examples]'
```

## 2. Validate a Configuration

Validation checks YAML structure, numeric constraints, unknown option names,
and whether every configured factory path resolves to a callable:

```bash
bayes-asi validate examples/linear_gaussian.yaml
```

Validation deliberately does not instantiate callbacks. A real problem may
need external files or solver binaries that are only available on compute
nodes.

## 3. Run an Inversion

Use one coordinator and two workers:

```bash
mpiexec -n 3 bayes-asi run examples/linear_gaussian.yaml
```

MPI rank 0 coordinates the run. Other ranks construct the solver callback and
wait for jobs. A single-rank invocation is rejected.

The example writes to `runs/linear_gaussian` because relative output paths are
resolved from the current working directory.

## 4. Inspect Progress

From another terminal:

```bash
bayes-asi status runs/linear_gaussian
```

Useful fields include:

- `completed_count`, `pending_count`, and `running_count`;
- `idle_workers` and `worker_count`;
- `budget_remaining_s` and `stop_queueing`;
- `surrogate_version`, `surrogate_training_samples`, and
  `surrogate_val_loss`;
- `solve_time_ewma_s`.

For detailed history, inspect `events.jsonl`:

```bash
tail -f runs/linear_gaussian/events.jsonl
```

## 5. Export Completed Evaluations

Export a compressed NumPy archive:

```bash
bayes-asi export runs/linear_gaussian --format npz
```

Or a flat CSV:

```bash
bayes-asi export runs/linear_gaussian --format csv
```

Both formats contain parameter columns, forward outputs, elapsed solver time,
worker rank, and job sequence. Exports are written atomically so readers never
see a partially replaced destination.

## 6. Resume

Run the same configuration again with the same `run.output_dir`:

```bash
mpiexec -n 3 bayes-asi run examples/linear_gaussian.yaml
```

The coordinator acquires `run.lock`, loads completed data, replays nonterminal
queue events, and loads the newest valid compatible checkpoint. A corrupt
newest checkpoint is logged and the next older valid checkpoint is tried.
Duplicate parameters are not queued again. Each invocation stores a
content-addressed YAML copy under `configs/` and records its SHA-256 digest.

See [Operations](operations.md) before changing configuration or callbacks for
an existing run directory.
