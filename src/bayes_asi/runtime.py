from __future__ import annotations

import math
import os
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from mpi4py import MPI

from bayes_asi.active import ActiveSelector
from bayes_asi.config import AppConfig, build_factory_object, load_config
from bayes_asi.design import latin_hypercube
from bayes_asi.hmc import HMCResult, HMCSampler
from bayes_asi.priors import standard_normal_icdf
from bayes_asi.queue import Job, QueueState
from bayes_asi.storage import HDF5Store, exclusive_run_lock, replay_queue_events, stable_theta_key
from bayes_asi.surrogate import SurrogateModel, SurrogateTrainer
from bayes_asi.types import ForwardResult, PriorTransform, SolverContext

TAG_WORK = 10
TAG_STOP = 11
TAG_RESULT = 12


def run_mpi(config_path: str | Path) -> None:
    config = load_config(config_path)
    comm = MPI.COMM_WORLD
    try:
        rank = comm.Get_rank()
        if rank == 0:
            with exclusive_run_lock(config.run.output_dir):
                Coordinator(config, comm).run()
        else:
            worker_loop(config, comm)
    except BaseException:
        if comm.Get_size() > 1:
            comm.Abort(1)
        raise


class Coordinator:
    def __init__(self, config: AppConfig, comm: Any) -> None:
        self.config = config
        self.comm = comm
        self.size = comm.Get_size()
        if self.size < 2:
            raise RuntimeError("Bayes_ASI run requires at least 2 MPI ranks: one coordinator and one worker")
        self.worker_ranks = list(range(1, self.size))
        self.worker_count = len(self.worker_ranks)
        self.available_workers = set(self.worker_ranks)
        self.rng = np.random.default_rng(config.run.seed)
        self.posterior = build_factory_object(config.posterior, config)
        candidate_prior = getattr(self.posterior, "prior", None)
        if candidate_prior is None or not callable(getattr(candidate_prior, "to_theta", None)) or not callable(
            getattr(candidate_prior, "to_standard_normal", None)
        ):
            raise TypeError("posterior must expose a reversible prior with to_theta and to_standard_normal")
        self.prior = cast(PriorTransform, candidate_prior)
        if int(self.prior.latent_dim) != config.problem.latent_dim:
            raise ValueError(
                f"posterior prior latent_dim={self.prior.latent_dim}, expected {config.problem.latent_dim}"
            )
        if int(self.prior.parameter_dim) != config.problem.parameter_dim:
            raise ValueError(
                f"posterior prior parameter_dim={self.prior.parameter_dim}, expected {config.problem.parameter_dim}"
            )
        self.store = HDF5Store(
            config.run.output_dir,
            config.problem.parameter_dim,
            config.problem.output_dim,
            latent_dim=config.problem.latent_dim,
        )
        self._migrate_latent_coordinates()
        self._validate_stored_coordinate_pairs()
        self.queue = QueueState()
        self.selector = ActiveSelector(config.active_learning)
        self.trainer = SurrogateTrainer(
            config.surrogate,
            config.problem.latent_dim,
            config.problem.output_dim,
            seed=config.run.seed,
        )
        self.surrogate: SurrogateModel | None = None
        self.surrogate_version = 0
        self.training_cycle = 0
        self.completed_at_last_train = 0
        self.completed_count = 0
        self.next_training_attempt_at = 0.0
        self.next_active_attempt_at = 0.0
        self.last_hmc_z: np.ndarray | None = None
        self.next_sequence = 0
        self.known_keys: set[tuple[float, ...]] = set()
        self._restore_state()

    def _migrate_latent_coordinates(self) -> None:
        if not self.store.latent_coordinates_pending():
            return
        completed = self.store.load_completed(allow_pending_latent=True)
        z = np.asarray(self.prior.to_standard_normal(completed.theta), dtype=np.float64)
        expected = (completed.n, self.config.problem.latent_dim)
        if z.shape != expected or not np.all(np.isfinite(z)):
            raise ValueError(
                f"prior.to_standard_normal returned invalid migration shape {z.shape}; expected {expected}"
            )
        self.store.complete_latent_migration(z)
        self.store.append_event(
            "storage_schema_migrated",
            {"schema_version": 2, "rows": completed.n, "coordinate_space": "standard_normal"},
        )

    def _validate_stored_coordinate_pairs(self) -> None:
        completed = self.store.load_completed()
        if completed.n == 0:
            return
        sample_count = min(completed.n, 512)
        indices = np.linspace(0, completed.n - 1, sample_count, dtype=int)
        self._validate_coordinate_pairs(
            completed.z[indices],
            completed.theta[indices],
            context="stored samples",
        )

    def _validate_coordinate_pairs(self, z: np.ndarray, theta: np.ndarray, *, context: str) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.config.problem.latent_dim)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1, self.config.problem.parameter_dim)
        if z.shape[0] != theta.shape[0] or not np.all(np.isfinite(z)) or not np.all(np.isfinite(theta)):
            raise ValueError(f"{context} has invalid standard-normal or physical coordinates")
        with torch.no_grad():
            expected = self.prior.to_theta(torch.as_tensor(z, dtype=torch.float32)).detach().cpu().numpy()
        if expected.shape != theta.shape or not np.all(np.isfinite(expected)):
            raise ValueError(f"configured prior returned invalid physical coordinates for {context}")
        if not np.allclose(expected, theta, rtol=2e-5, atol=2e-6):
            maximum_error = float(np.max(np.abs(expected - theta)))
            raise ValueError(
                f"{context} is inconsistent with the configured prior; maximum theta error={maximum_error:.6g}"
            )

    def run(self) -> None:
        start = time.monotonic()
        deadline = start + self.config.run.time_budget_seconds
        next_status = 0.0
        stop_queueing = False
        budget_event_written = False
        config_snapshot, config_sha256 = self.store.snapshot_config(self.config.config_path)
        self.store.append_event(
            "run_started",
            {
                "worker_count": self.worker_count,
                "time_budget_seconds": self.config.run.time_budget_seconds,
                "config_path": str(self.config.config_path),
                "config_snapshot": str(config_snapshot),
                "config_sha256": config_sha256,
            },
        )
        self._import_initial_data()
        self._ensure_initial_design()

        while True:
            self._receive_available_results()

            now = time.monotonic()
            if now >= deadline:
                stop_queueing = True
                if not budget_event_written:
                    self.store.append_event(
                        "budget_reached",
                        {
                            "completed_count": self.completed_count,
                            "pending_count": self.queue.pending_count,
                            "running_count": self.queue.running_count,
                            "drain_running_on_exit": self.config.run.drain_running_on_exit,
                        },
                    )
                    budget_event_written = True

            if not stop_queueing:
                self._maybe_train_surrogate(deadline)
                if time.monotonic() < deadline:
                    self._maybe_refill_active_queue(deadline)
                if time.monotonic() < deadline:
                    self._assign_available_workers()

            if not stop_queueing and time.monotonic() >= deadline:
                continue

            if now >= next_status:
                self._write_status(deadline, stop_queueing)
                next_status = now + self.config.run.status_interval_seconds

            if stop_queueing and not self.config.run.drain_running_on_exit:
                self.store.append_event(
                    "run_stopped_at_budget",
                    {
                        "completed_count": self.completed_count,
                        "pending_count": self.queue.pending_count,
                        "running_count": self.queue.running_count,
                    },
                )
                self._write_status(deadline, True, final=True)
                # Busy workers are inside black-box solvers and cannot receive TAG_STOP.
                # State is flushed above, then MPI aborts the active ranks to enforce budget.
                self.comm.Abort(0)

            if stop_queueing and self.queue.running_count == 0:
                break

            time.sleep(0.02)

        self._stop_workers()
        self._write_status(deadline, True, final=True)
        self.store.append_event("run_finished", {"completed_count": self.completed_count})

    def _import_initial_data(self) -> None:
        if self.config.initial_data is None:
            return
        loader = build_factory_object(self.config.initial_data, self.config)
        if callable(loader):
            initial = loader()
        else:
            initial = loader
        if not isinstance(initial, dict):
            raise TypeError("initial_data factory must return a mapping or a callable returning one")
        theta = np.asarray(initial["theta"], dtype=float).reshape(-1, self.config.problem.parameter_dim)
        outputs = np.asarray(initial["outputs"], dtype=float).reshape(-1, self.config.problem.output_dim)
        if theta.shape[0] != outputs.shape[0]:
            raise ValueError("initial_data theta and outputs row counts differ")
        z = np.asarray(self.prior.to_standard_normal(theta), dtype=float)
        expected_z_shape = (theta.shape[0], self.config.problem.latent_dim)
        if z.shape != expected_z_shape or not np.all(np.isfinite(z)):
            raise ValueError(
                f"prior.to_standard_normal must return finite shape {expected_z_shape}, got {z.shape}"
            )
        auxiliary_rows = self._initial_auxiliary_rows(initial.get("auxiliary"), theta.shape[0])
        z_rows: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        output_rows: list[np.ndarray] = []
        stored_rows: list[Mapping[str, np.ndarray]] = []
        sequences: list[int] = []
        skipped = 0
        for i, row in enumerate(theta):
            key = stable_theta_key(row, self.config.active_learning.duplicate_round_decimals)
            if key in self.known_keys:
                skipped += 1
                continue
            sequence = self.next_sequence
            self.next_sequence += 1
            self.known_keys.add(key)
            z_rows.append(z[i])
            rows.append(row)
            output_rows.append(outputs[i])
            stored_rows.append(auxiliary_rows[i])
            sequences.append(sequence)

        imported = len(rows)
        first_index = self.completed_count
        if imported:
            first_index = self.store.append_completed_batch(
                z=np.vstack(z_rows),
                theta=np.vstack(rows),
                outputs=np.vstack(output_rows),
                elapsed_s=np.zeros(imported),
                worker_rank=np.full(imported, -1, dtype=np.int64),
                job_sequence=np.asarray(sequences, dtype=np.int64),
                started_at=np.zeros(imported),
                finished_at=np.zeros(imported),
                auxiliary=stored_rows,
            )
            self.completed_count += imported
        metadata = initial.get("metadata", {})
        for offset, (z_row, row, sequence) in enumerate(zip(z_rows, rows, sequences, strict=True)):
            self.store.append_event(
                "initial_sample_imported",
                {
                    "job_id": f"initial_{sequence:08d}",
                    "job_sequence": sequence,
                    "hdf5_index": first_index + offset,
                    "z": z_row.tolist(),
                    "theta": row.tolist(),
                    "metadata": metadata,
                },
            )
        self.store.append_event(
            "initial_data_loaded",
            {
                "rows": int(theta.shape[0]),
                "imported": imported,
                "skipped_existing": skipped,
                "auxiliary_fields": sorted(str(name) for name in (initial.get("auxiliary") or {})),
                "metadata": metadata,
            },
        )

    @staticmethod
    def _initial_auxiliary_rows(raw: Any, rows: int) -> list[dict[str, np.ndarray]]:
        result: list[dict[str, np.ndarray]] = [{} for _ in range(rows)]
        if raw is None:
            return result
        if not isinstance(raw, Mapping):
            raise TypeError("initial_data auxiliary must be a mapping of names to row-aligned arrays")
        for raw_name, raw_values in raw.items():
            name = str(raw_name)
            values = np.asarray(raw_values)
            if values.ndim == 0 or values.shape[0] != rows:
                raise ValueError(f"initial_data auxiliary field {name!r} must have leading dimension {rows}")
            for index in range(rows):
                result[index][name] = values[index]
        return result

    def _restore_state(self) -> None:
        completed = self.store.load_completed()
        self.completed_count = completed.n
        self.known_keys = {
            stable_theta_key(row, self.config.active_learning.duplicate_round_decimals)
            for row in completed.theta
        }
        events = self.store.read_events()
        for checkpoint in self.store.checkpoint_paths(newest_first=True):
            try:
                candidate = SurrogateModel.load(checkpoint, device=self.trainer.device)
                if candidate.input_dim != self.config.problem.latent_dim:
                    raise ValueError("checkpoint input dimension does not match the latent configuration")
                if candidate.input_space != "standard_normal":
                    raise ValueError("checkpoint was trained in physical rather than standard-normal coordinates")
                if candidate.output_dim != self.config.problem.output_dim:
                    raise ValueError("checkpoint output dimension does not match the configuration")
                if candidate.training_samples is not None and candidate.training_samples > completed.n:
                    raise ValueError("checkpoint was trained with more samples than are stored")
            except Exception as exc:
                self.store.append_event(
                    "surrogate_checkpoint_load_failed",
                    {"path": str(checkpoint), "error": str(exc)},
                )
                continue
            self.surrogate = candidate
            self.surrogate_version = candidate.version
            self.training_cycle = candidate.version
            self.completed_at_last_train = candidate.training_samples or 0
            self.store.append_event(
                "surrogate_loaded",
                {
                    "path": str(checkpoint),
                    "version": self.surrogate_version,
                    "training_samples": candidate.training_samples,
                },
            )
            break

        pending = replay_queue_events(
            events,
            self.known_keys,
            decimals=self.config.active_learning.duplicate_round_decimals,
        )
        max_sequence = int(np.max(completed.job_sequence)) if completed.job_sequence.size else -1
        for event in events:
            if "job_sequence" in event:
                max_sequence = max(max_sequence, int(event["job_sequence"]))
        stale_jobs: list[dict[str, Any]] = []
        for job_data in pending:
            sequence = int(job_data.get("sequence", max_sequence + 1))
            max_sequence = max(max_sequence, sequence)
            source = str(job_data.get("source", "replayed"))
            source_version = self._source_surrogate_version(source)
            incompatible_source = source_version is not None and source_version != self.surrogate_version
            if incompatible_source:
                stale_jobs.append({"job_id": str(job_data["job_id"]), "sequence": sequence})
                self.store.append_event(
                    "job_cancelled",
                    {
                        "job_id": str(job_data["job_id"]),
                        "job_sequence": sequence,
                        "theta": np.asarray(job_data["theta"], dtype=float).tolist(),
                        "attempt": int(job_data.get("attempt", 0)),
                        "source": source,
                        "reason": "incompatible_surrogate_on_restore",
                        "surrogate_version": self.surrogate_version,
                    },
                )
                continue
            theta = np.asarray(job_data["theta"], dtype=float).reshape(self.config.problem.parameter_dim)
            raw_z = job_data.get("z")
            z = (
                np.asarray(self.prior.to_standard_normal(theta.reshape(1, -1)), dtype=float).reshape(
                    self.config.problem.latent_dim
                )
                if raw_z is None
                else np.asarray(raw_z, dtype=float).reshape(self.config.problem.latent_dim)
            )
            if not np.all(np.isfinite(z)):
                raise ValueError(f"replayed job {job_data['job_id']} has non-finite standard-normal coordinates")
            self._validate_coordinate_pairs(z.reshape(1, -1), theta.reshape(1, -1), context="replayed job")
            job = Job(
                job_id=str(job_data["job_id"]),
                sequence=sequence,
                theta=theta,
                z=z,
                attempt=int(job_data.get("attempt", 0)),
                source=source,
                queued_at=time.time(),
            )
            self.queue.add_pending(job)
            self.known_keys.add(stable_theta_key(job.theta, self.config.active_learning.duplicate_round_decimals))
        self.next_sequence = max_sequence + 1
        if stale_jobs:
            self.store.append_event(
                "queue_pending_cleared",
                {
                    "reason": "incompatible_surrogate_on_restore",
                    "surrogate_version": self.surrogate_version,
                    "cancelled_count": len(stale_jobs),
                    "job_ids": [str(job["job_id"]) for job in stale_jobs],
                    "job_sequences": [int(job["sequence"]) for job in stale_jobs],
                },
            )

    def _ensure_initial_design(self) -> None:
        target = self.config.initial_design.n_points
        projected_rows = max(target, self.completed_count + self.queue.in_system_count)
        if self.surrogate is None and projected_rows < self.config.surrogate.min_train_samples:
            raise ValueError(
                "cannot bootstrap surrogate: completed, replayed, and configured initial-design rows "
                f"provide at most {projected_rows} samples, but surrogate.min_train_samples="
                f"{self.config.surrogate.min_train_samples}; increase initial_design.n_points or provide initial_data"
            )
        missing = target - self.completed_count - self.queue.in_system_count
        if missing <= 0:
            return
        z, theta = self._draw_initial_points(missing)
        queued = 0
        for z_row, theta_row in zip(z, theta, strict=True):
            queued += self._queue_theta(theta_row, z=z_row, source="initial") is not None
        self.store.append_event(
            "initial_design_generated",
            {
                "method": "latin_hypercube_prior",
                "requested_count": missing,
                "queued_count": queued,
                "latent_dim": self.config.problem.latent_dim,
                "coordinate_space": "standard_normal",
            },
        )

    def _draw_initial_theta(self, n: int) -> np.ndarray:
        return self._draw_initial_points(n)[1]

    def _draw_initial_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        unit = latin_hypercube(self.rng, n, self.config.problem.latent_dim)
        z = np.asarray(standard_normal_icdf(unit), dtype=np.float32)
        if z.shape != (n, self.config.problem.latent_dim) or not np.all(np.isfinite(z)):
            raise ValueError(f"standard-normal design has invalid shape {z.shape}")
        with torch.no_grad():
            theta = self.prior.to_theta(torch.as_tensor(z, device=self.trainer.device))
        theta_np = theta.detach().cpu().numpy().reshape(n, self.config.problem.parameter_dim)
        if not np.all(np.isfinite(theta_np)):
            raise ValueError("posterior prior returned non-finite initial parameters")
        return z, theta_np

    @staticmethod
    def _source_surrogate_version(source: str) -> int | None:
        prefix = "surrogate_v"
        if not source.startswith(prefix):
            return None
        try:
            return int(source[len(prefix) :])
        except ValueError:
            return None

    def _queue_theta(
        self,
        theta: np.ndarray,
        *,
        source: str,
        z: np.ndarray | None = None,
        attempt: int = 0,
    ) -> Job | None:
        theta = np.asarray(theta, dtype=float).reshape(self.config.problem.parameter_dim)
        if not np.all(np.isfinite(theta)):
            raise ValueError("cannot queue non-finite parameters")
        if z is None:
            z = np.asarray(self.prior.to_standard_normal(theta.reshape(1, -1)), dtype=float).reshape(
                self.config.problem.latent_dim
            )
        else:
            z = np.asarray(z, dtype=float).reshape(self.config.problem.latent_dim)
        if not np.all(np.isfinite(z)):
            raise ValueError("cannot queue non-finite standard-normal coordinates")
        self._validate_coordinate_pairs(z.reshape(1, -1), theta.reshape(1, -1), context="queued job")
        key = stable_theta_key(theta, self.config.active_learning.duplicate_round_decimals)
        if key in self.known_keys:
            return None
        sequence = self.next_sequence
        self.next_sequence += 1
        job = Job(
            job_id=f"job_{sequence:08d}",
            sequence=sequence,
            theta=theta,
            z=z,
            attempt=attempt,
            source=source,
            queued_at=time.time(),
        )
        self.queue.add_pending(job)
        self.known_keys.add(key)
        self.store.append_event(
            "job_queued",
            {
                "job_id": job.job_id,
                "job_sequence": job.sequence,
                "z": z.tolist(),
                "theta": theta.tolist(),
                "attempt": attempt,
                "source": source,
            },
        )
        return job

    def _clear_pending_for_new_surrogate(
        self,
        *,
        previous_surrogate_version: int,
        completed_theta: np.ndarray,
        completed_count: int,
    ) -> None:
        cancelled = self.queue.clear_pending()
        if not cancelled:
            return

        decimals = self.config.active_learning.duplicate_round_decimals
        completed_keys = {stable_theta_key(row, decimals) for row in completed_theta}
        running_keys = {stable_theta_key(job.theta, decimals) for job in self.queue.running.values()}
        for job in cancelled:
            key = stable_theta_key(job.theta, decimals)
            if key not in completed_keys and key not in running_keys:
                self.known_keys.discard(key)
            self.store.append_event(
                "job_cancelled",
                {
                    "job_id": job.job_id,
                    "job_sequence": job.sequence,
                    "theta": job.theta.tolist(),
                    "attempt": job.attempt,
                    "source": job.source,
                    "reason": "new_surrogate_trained",
                    "previous_surrogate_version": previous_surrogate_version,
                    "surrogate_version": self.surrogate_version,
                    "completed_count": completed_count,
                },
            )
        self.store.append_event(
            "queue_pending_cleared",
            {
                "reason": "new_surrogate_trained",
                "previous_surrogate_version": previous_surrogate_version,
                "surrogate_version": self.surrogate_version,
                "completed_count": completed_count,
                "cancelled_count": len(cancelled),
                "job_ids": [job.job_id for job in cancelled],
                "job_sequences": [job.sequence for job in cancelled],
            },
        )

    def _assign_available_workers(self) -> None:
        for worker_rank in sorted(list(self.available_workers)):
            job = self.queue.pop_for_worker(worker_rank)
            if job is None:
                break
            self.available_workers.remove(worker_rank)
            self.store.append_event(
                "job_started",
                {
                    "job_id": job.job_id,
                    "job_sequence": job.sequence,
                    "worker_rank": worker_rank,
                    "attempt": job.attempt,
                },
            )
            payload = {
                "job_id": job.job_id,
                "job_sequence": job.sequence,
                "theta": job.theta,
                "attempt": job.attempt,
            }
            self.comm.send(payload, dest=worker_rank, tag=TAG_WORK)

    def _receive_available_results(self) -> None:
        status = MPI.Status()
        while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            worker_rank = status.Get_source()
            result = self.comm.recv(source=worker_rank, tag=TAG_RESULT)
            job = self.queue.finish_worker(worker_rank)
            self.available_workers.add(worker_rank)
            self._handle_result(worker_rank, job, result)

    def _handle_result(self, worker_rank: int, job: Job | None, result: dict[str, Any]) -> None:
        if job is None:
            self.store.append_event(
                "unexpected_result",
                {"worker_rank": worker_rank, "result_job_id": result.get("job_id")},
            )
            return
        elapsed = float(result.get("elapsed_s", 0.0))
        self.queue.update_solve_time(elapsed)
        if result.get("ok"):
            if job.z is None:
                raise RuntimeError(f"completed job {job.job_id} is missing standard-normal coordinates")
            outputs = np.asarray(result["outputs"], dtype=float).reshape(self.config.problem.output_dim)
            index = self.store.append_completed(
                z=job.z,
                theta=job.theta,
                outputs=outputs,
                elapsed_s=elapsed,
                worker_rank=worker_rank,
                job_sequence=job.sequence,
                started_at=float(result.get("started_at", time.time())),
                finished_at=float(result.get("finished_at", time.time())),
                auxiliary=result.get("auxiliary", {}),
            )
            self.completed_count += 1
            self.next_training_attempt_at = 0.0
            self.store.append_event(
                "job_completed",
                {
                    "job_id": job.job_id,
                    "job_sequence": job.sequence,
                    "worker_rank": worker_rank,
                    "hdf5_index": index,
                    "elapsed_s": elapsed,
                    "metadata": result.get("metadata", {}),
                    "artifact_paths": result.get("artifact_paths", []),
                    "auxiliary_fields": sorted(result.get("auxiliary", {})),
                },
            )
        else:
            will_retry = job.attempt < self.config.queue.retry_limit
            self.store.append_event(
                "job_failed",
                {
                    "job_id": job.job_id,
                    "job_sequence": job.sequence,
                    "worker_rank": worker_rank,
                    "attempt": job.attempt,
                    "will_retry": int(will_retry),
                    "error": result.get("error", "unknown error"),
                    "traceback": result.get("traceback", ""),
                    "elapsed_s": elapsed,
                },
            )
            if will_retry:
                retry = Job(
                    job_id=job.job_id,
                    sequence=job.sequence,
                    theta=job.theta,
                    z=job.z,
                    attempt=job.attempt + 1,
                    source=job.source,
                    queued_at=time.time(),
                )
                self.queue.add_pending(retry)

    def _maybe_train_surrogate(self, deadline: float | None = None) -> None:
        if self.completed_count < self.config.surrogate.min_train_samples:
            return
        new_since_train = self.completed_count - self.completed_at_last_train
        if self.surrogate is not None and new_since_train < self.config.surrogate.update_after_new_samples:
            return
        now = time.monotonic()
        if now < self.next_training_attempt_at:
            return

        completed = self.store.load_completed()

        previous_surrogate_version = self.surrogate_version
        previous_completed_at_last_train = self.completed_at_last_train
        version = self.surrogate_version + 1
        try:
            warm = self.trainer.train(
                completed.z,
                completed.outputs,
                version=version,
                previous=self.surrogate,
                force_scratch=False,
                deadline_monotonic=deadline,
            )
            chosen = warm
            run_scratch = (
                self.surrogate is not None
                and self.config.surrogate.retrain_from_scratch_every > 0
                and (self.training_cycle + 1) % self.config.surrogate.retrain_from_scratch_every == 0
                and (deadline is None or time.monotonic() < deadline)
            )
            if run_scratch:
                scratch = self.trainer.train(
                    completed.z,
                    completed.outputs,
                    version=version,
                    previous=None,
                    force_scratch=True,
                    deadline_monotonic=deadline,
                )
                threshold = warm.val_loss * (1.0 - self.config.surrogate.scratch_improvement_rel)
                if scratch.val_loss < threshold:
                    chosen = scratch
            checkpoint = self.store.checkpoints_dir / f"surrogate_v{version:04d}.pt"
            chosen.model.save(checkpoint)
            self.surrogate = chosen.model
            self.surrogate_version = version
            self.training_cycle += 1
            self.completed_at_last_train = completed.n
            self.next_training_attempt_at = 0.0
            self.store.append_event(
                "surrogate_trained",
                {
                    "version": version,
                    "completed_count": completed.n,
                    "checkpoint": str(checkpoint),
                    "train_loss": chosen.train_loss,
                    "val_loss": chosen.val_loss,
                    "epochs": chosen.epochs,
                    "best_epoch": chosen.best_epoch,
                    "trained_from_scratch": chosen.trained_from_scratch,
                    "stopped_by_deadline": chosen.stopped_by_deadline,
                },
            )
            if previous_surrogate_version > 0 and completed.n > previous_completed_at_last_train:
                self._clear_pending_for_new_surrogate(
                    previous_surrogate_version=previous_surrogate_version,
                    completed_theta=completed.theta,
                    completed_count=completed.n,
                )
        except Exception as exc:
            self.next_training_attempt_at = now + self.config.surrogate.failure_retry_seconds
            self.store.append_event(
                "surrogate_train_failed",
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_count": completed.n,
                    "retry_after_seconds": self.config.surrogate.failure_retry_seconds,
                },
            )

    def _maybe_refill_active_queue(self, deadline: float | None = None) -> None:
        if self.surrogate is None:
            self._ensure_initial_design()
            return
        threshold = self.config.queue.refill_threshold_pending_per_worker * self.worker_count
        if self.queue.pending_count > threshold:
            return
        desired = self._desired_in_system_count()
        count = min(self.config.active_learning.batch_max_size, max(0, desired - self.queue.in_system_count))
        if count <= 0:
            return
        now = time.monotonic()
        if now < self.next_active_attempt_at:
            return
        completed = self.store.load_completed()
        queued_parts = []
        pending = self.queue.pending_z()
        running = self.queue.running_z()
        if pending.size:
            queued_parts.append(pending.reshape(-1, self.config.problem.latent_dim))
        if running.size:
            queued_parts.append(running.reshape(-1, self.config.problem.latent_dim))
        queued = np.vstack(queued_parts) if queued_parts else np.empty((0, self.config.problem.latent_dim))
        try:
            hmc_result = self._draw_surrogate_posterior(deadline)
            candidate_count = min(self.config.active_learning.candidate_pool_size, hmc_result.theta.shape[0])
            if hmc_result.theta.shape[0] > candidate_count:
                indices = self.rng.choice(hmc_result.theta.shape[0], size=candidate_count, replace=False)
                candidate_z = hmc_result.z[indices]
                candidate_theta = hmc_result.theta[indices]
                candidate_logp = hmc_result.log_prob[indices]
            else:
                candidate_z = hmc_result.z
                candidate_theta = hmc_result.theta
                candidate_logp = hmc_result.log_prob
            posterior_path = self._save_hmc_batch(hmc_result)
            selected = self.selector.select(
                candidate_z=candidate_z,
                candidate_theta=candidate_theta,
                candidate_log_prob=candidate_logp,
                existing_z=completed.z,
                queued_z=queued,
                count=count,
                known_keys=self.known_keys,
            )
            for z_row, theta_row in zip(selected.z, selected.theta, strict=True):
                self._queue_theta(theta_row, z=z_row, source=f"surrogate_v{self.surrogate_version}")
            if selected.theta.shape[0] == 0:
                self.next_active_attempt_at = now + self.config.active_learning.failure_retry_seconds
            else:
                self.next_active_attempt_at = 0.0
            self.store.append_event(
                "active_batch_selected",
                {
                    "surrogate_version": self.surrogate_version,
                    "requested": count,
                    "selected": int(selected.theta.shape[0]),
                    "hmc_accept_rate": hmc_result.accept_rate,
                    "hmc_warmup_accept_rate": hmc_result.warmup_accept_rate,
                    "hmc_step_size": hmc_result.step_size,
                    "hmc_leapfrog_steps": hmc_result.leapfrog_steps,
                    "posterior_batch_path": str(posterior_path),
                    "scores": selected.scores.tolist(),
                    "min_distances": selected.min_distances.tolist(),
                },
            )
        except Exception as exc:
            self.next_active_attempt_at = now + self.config.active_learning.failure_retry_seconds
            self.store.append_event(
                "active_selection_failed",
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "surrogate_version": self.surrogate_version,
                    "retry_after_seconds": self.config.active_learning.failure_retry_seconds,
                },
            )

    def _draw_surrogate_posterior(self, deadline: float | None = None) -> HMCResult:
        if self.surrogate is None:
            raise RuntimeError("surrogate is not available")
        expected_shape = (self.config.hmc.n_chains, self.config.problem.latent_dim)
        if self.last_hmc_z is None or self.last_hmc_z.shape != expected_shape:
            initializer = getattr(self.posterior, "initial_z", None)
            if callable(initializer):
                initial_z = np.asarray(initializer(self.rng, self.config.hmc.n_chains), dtype=np.float32)
            else:
                initial_z = self.rng.normal(size=expected_shape).astype(np.float32)
            if initial_z.shape != expected_shape or not np.all(np.isfinite(initial_z)):
                raise ValueError(f"posterior initial_z must return finite shape {expected_shape}")
        else:
            initial_z = self.last_hmc_z
        sampler = HMCSampler(
            self.config.hmc,
            self.posterior,
            self.surrogate,
            seed=int(self.rng.integers(0, 2**31 - 1)),
            device=self.trainer.device,
        )
        result = sampler.sample(initial_z, deadline_monotonic=deadline)
        self.last_hmc_z = result.z[-self.config.hmc.n_chains :].copy()
        return result

    def _save_hmc_batch(self, hmc_result: HMCResult) -> Path:
        batch_dir = self.config.run.output_dir / "posterior_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        path = batch_dir / f"surrogate_v{self.surrogate_version:04d}_batch_{time.time_ns()}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            z=hmc_result.z,
            theta=hmc_result.theta,
            log_prob=hmc_result.log_prob,
            accept_rate=np.asarray([hmc_result.accept_rate], dtype=float),
            warmup_accept_rate=np.asarray([hmc_result.warmup_accept_rate], dtype=float),
            step_size=np.asarray([hmc_result.step_size], dtype=float),
            leapfrog_steps=np.asarray([hmc_result.leapfrog_steps], dtype=int),
            mass_diag=hmc_result.mass_diag,
            surrogate_version=np.asarray([self.surrogate_version], dtype=int),
        )
        os.replace(temporary, path)
        return path

    def _desired_in_system_count(self) -> int:
        min_pending = self.config.queue.min_pending_per_worker * self.worker_count
        max_pending = self.config.queue.max_pending_per_worker * self.worker_count
        if self.queue.solve_time_ewma is None:
            ahead = min_pending
        else:
            throughput = self.worker_count / max(self.queue.solve_time_ewma, 1e-9)
            ahead = int(math.ceil(self.config.queue.target_seconds_ahead * throughput))
            ahead = max(min_pending, min(max_pending, ahead))
        return self.worker_count + ahead

    def _write_status(self, deadline: float, stop_queueing: bool, final: bool = False) -> None:
        status = {
            "final": final,
            "completed_count": self.completed_count,
            "pending_count": self.queue.pending_count,
            "running_count": self.queue.running_count,
            "idle_workers": len(self.available_workers),
            "worker_count": self.worker_count,
            "stop_queueing": stop_queueing,
            "budget_remaining_s": max(0.0, deadline - time.monotonic()),
            "solve_time_ewma_s": self.queue.solve_time_ewma,
            "surrogate_version": self.surrogate_version if self.surrogate is not None else None,
            "surrogate_val_loss": self.surrogate.val_loss if self.surrogate is not None else None,
            "surrogate_training_samples": self.surrogate.training_samples if self.surrogate is not None else None,
            "surrogate_input_space": "standard_normal",
        }
        self.store.update_status(status)

    def _stop_workers(self) -> None:
        for worker_rank in self.worker_ranks:
            self.comm.send(None, dest=worker_rank, tag=TAG_STOP)


def worker_loop(config: AppConfig, comm: MPI.Comm) -> None:
    rank = comm.Get_rank()
    solver = build_factory_object(config.solver, config)
    run_dir = config.run.output_dir
    status = MPI.Status()
    while True:
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            break
        if status.Get_tag() != TAG_WORK:
            continue
        result = _evaluate_worker_job(config, solver, run_dir, rank, payload)
        comm.send(result, dest=0, tag=TAG_RESULT)


def _evaluate_worker_job(
    config: AppConfig,
    solver: Any,
    run_dir: Path,
    rank: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    job_id = str(payload["job_id"])
    theta = np.asarray(payload["theta"], dtype=float).reshape(config.problem.parameter_dim)
    if not np.all(np.isfinite(theta)):
        raise ValueError("worker received non-finite parameters")
    started_at = time.time()
    work_dir = run_dir / "workers" / f"rank_{rank}" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    context = SolverContext(
        run_dir=run_dir,
        work_dir=work_dir,
        job_id=job_id,
        worker_rank=rank,
        metadata={"attempt": int(payload.get("attempt", 0))},
    )
    try:
        raw = solver.evaluate(theta.copy(), context)
        if isinstance(raw, ForwardResult):
            forward_result = raw
        else:
            forward_result = ForwardResult(outputs=np.asarray(raw, dtype=float))
        outputs = np.asarray(forward_result.outputs, dtype=float).reshape(config.problem.output_dim)
        if not np.all(np.isfinite(outputs)):
            raise ValueError("solver returned non-finite outputs")
        auxiliary = {str(name): np.asarray(value) for name, value in forward_result.auxiliary.items()}
        HDF5Store._normalize_auxiliary_batch([auxiliary], 1)
        finished_at = time.time()
        return {
            "ok": True,
            "job_id": job_id,
            "job_sequence": int(payload["job_sequence"]),
            "theta": theta,
            "outputs": outputs,
            "auxiliary": auxiliary,
            "metadata": forward_result.metadata,
            "artifact_paths": forward_result.artifact_paths,
            "worker_rank": rank,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": finished_at - started_at,
        }
    except Exception as exc:
        finished_at = time.time()
        return {
            "ok": False,
            "job_id": job_id,
            "job_sequence": int(payload["job_sequence"]),
            "theta": theta,
            "worker_rank": rank,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": finished_at - started_at,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
