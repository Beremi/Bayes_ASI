# Student proposals and teacher screening

The optional hierarchy spends most proposal work on a small distilled model,
then uses a stronger surrogate to decide which candidates deserve a full forward
solve. It is disabled by default, so existing configurations retain their behavior.

```text
completed solver evaluations -> strong teacher
                                    |
                           temporary predictions
                                    v
                              cheap student
                                    |
                            many HMC candidates
                                    v
                     diverse bounded screening set
                                    |
                         teacher likelihood scores
                                    v
                         expensive solver queue
```

## Configuration

```bash
bayes-asi validate examples/hierarchy.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpiexec -n 3 bayes-asi run examples/hierarchy.yaml
```

`surrogate` configures the teacher. `hierarchy.student` accepts the surrogate
training options and defaults to a smaller `[32, 32]` network. The initial
implementation trains both as normalized tanh MLPs; loading a graph/PINN teacher
through a public factory is a future extension. A larger MLP is not assumed to be
more accurate simply because it has more parameters.

| Option | Meaning |
| --- | --- |
| `hierarchy.enabled` | Enable student proposals and teacher screening. |
| `distillation_points` | Additional temporary latent inputs per teacher version. |
| `teacher_screen_size` | Maximum unique candidates evaluated by the teacher per refill. Must cover `active_learning.batch_max_size`. |
| `prediction_batch_size` | Bound each teacher/student prediction batch. |
| `exploration_fraction` | Random share of the screening set, drawn from the cheap candidate pool. |
| `student` | Student architecture, optimizer, epochs, validation split and device. |

Set HMC's chain/sample counts high enough to supply the desired cheap pool;
`active_learning.candidate_pool_size` caps that pool. The teacher cap applies to
candidate screening, not its evaluations for distillation.

## Training and selection

The teacher sees only completed solver rows. On each teacher update, the student
is trained from scratch on teacher predictions at completed latent coordinates
and a reproducible mixture of prior draws and perturbed completed coordinates.
Generated coordinates are clipped to `[-6,6]` for this design step only. Neither
the inference prior nor the HMC coordinates are truncated by that clipping.

Distillation targets are temporary arrays. They are never appended to `data.h5`,
counted as completed jobs, or reused as new ground truth. Student validation loss
measures agreement with the teacher, not accuracy against the physical solver.

HMC uses the matching student's differentiable output. Prescreening removes
known/duplicate physical points and mixes posterior/distance selection with a
random share. The teacher evaluates only those screened points. The runtime
recomputes their likelihood plus standard-normal latent prior and applies the
existing posterior-plus-distance selector to fill the solver queue. Nonfinite
teacher rows are excluded. Full-model failures retain normal retry semantics.

Random exploration is within the student's candidate pool; it does not guarantee
discovery of modes omitted by the student. Check prior coverage and use dispersed
starts, solver holdouts and additional designs for a scientific problem.

## Checkpoints, restart and budgets

The teacher retains `checkpoints/surrogate_vNNNN.pt`. The student has a separate
`student_vNNNN.pt` and pair metadata containing SHA-256 fingerprints of both
checkpoints. A matching version number alone is insufficient. Missing/corrupt
metadata or a replaced teacher causes deferred regeneration within the run
deadline, rather than reuse of a stale student.

When no valid student exists, the runtime records an explicit fallback to teacher
HMC. Regeneration uses retry backoff. Prediction batches and training honor the
run deadline. Individual library operations still finish before deadline checks;
the deadline does not preempt a single PyTorch kernel. Pair metadata replacement
is atomic; it is not an fsync-backed transaction. An interrupted publication
therefore causes safe regeneration.

`events.jsonl`, `status.json` and posterior NPZ batches record model roles and
versions. Selection events also record screened/finite counts, disagreement RMSE
and screening time. Existing running solver jobs remain valid when models update;
pending queue entries follow the existing cancellation/replay rules.

## Inference interpretation

These are **approximate surrogate-posterior HMC samples and adaptive evaluation
designs**. Teacher reranking is an acquisition decision. It is not a
Metropolis-Hastings correction, importance resampling, or exact delayed acceptance
against the full solver. Full-model evaluation of selected points alone does not
make the saved HMC chain an exact full-model posterior.

## Verification

Unit tests check model roles, screening caps, duplicate handling, invalid values,
checkpoint pairing, bounded regeneration and separation of pseudo-labels from
solver storage. The MPI test independently checks every stored output against
the analytic forward map across a restart:

```bash
BAYES_ASI_HIERARCHY_MPI_TEST=1 \
  python -m unittest discover -s tests -p test_hierarchy_mpi.py
```
