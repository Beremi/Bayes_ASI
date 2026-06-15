from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def now_seconds() -> float:
    return time.time()


def stable_theta_key(theta: np.ndarray, decimals: int = 12) -> tuple[float, ...]:
    rounded = np.round(np.asarray(theta, dtype=float).reshape(-1), decimals=decimals)
    return tuple(float(value) for value in rounded)


@dataclass(slots=True)
class CompletedData:
    theta: np.ndarray
    outputs: np.ndarray
    elapsed_s: np.ndarray
    worker_rank: np.ndarray
    job_sequence: np.ndarray

    @property
    def n(self) -> int:
        return int(self.theta.shape[0])


class HDF5Store:
    """Single-writer HDF5 store plus append-only JSONL events."""

    def __init__(self, run_dir: Path, parameter_dim: int, output_dim: int) -> None:
        self.run_dir = Path(run_dir)
        self.parameter_dim = int(parameter_dim)
        self.output_dim = int(output_dim)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.data_path = self.run_dir / "data.h5"
        self.events_path = self.run_dir / "events.jsonl"
        self.status_path = self.run_dir / "status.json"
        self._ensure_hdf5()

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def _ensure_hdf5(self) -> None:
        with h5py.File(self.data_path, "a") as h5:
            h5.attrs["parameter_dim"] = self.parameter_dim
            h5.attrs["output_dim"] = self.output_dim
            group = h5.require_group("completed")
            self._require_dataset(group, "theta", (0, self.parameter_dim), float)
            self._require_dataset(group, "outputs", (0, self.output_dim), float)
            self._require_dataset(group, "elapsed_s", (0,), float)
            self._require_dataset(group, "worker_rank", (0,), np.int64)
            self._require_dataset(group, "job_sequence", (0,), np.int64)
            self._require_dataset(group, "started_at", (0,), float)
            self._require_dataset(group, "finished_at", (0,), float)

    @staticmethod
    def _require_dataset(group: h5py.Group, name: str, shape: tuple[int, ...], dtype: Any) -> None:
        if name in group:
            return
        maxshape = (None,) + tuple(shape[1:])
        chunks = (max(1, min(1024, shape[0] or 1)),) + tuple(shape[1:])
        group.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype)

    def append_completed(
        self,
        *,
        theta: np.ndarray,
        outputs: np.ndarray,
        elapsed_s: float,
        worker_rank: int,
        job_sequence: int,
        started_at: float,
        finished_at: float,
    ) -> int:
        theta = np.asarray(theta, dtype=float).reshape(1, self.parameter_dim)
        outputs = np.asarray(outputs, dtype=float).reshape(1, self.output_dim)
        with h5py.File(self.data_path, "a") as h5:
            group = h5["completed"]
            index = int(group["theta"].shape[0])
            self._append_row(group["theta"], theta)
            self._append_row(group["outputs"], outputs)
            self._append_row(group["elapsed_s"], np.asarray([elapsed_s], dtype=float))
            self._append_row(group["worker_rank"], np.asarray([worker_rank], dtype=np.int64))
            self._append_row(group["job_sequence"], np.asarray([job_sequence], dtype=np.int64))
            self._append_row(group["started_at"], np.asarray([started_at], dtype=float))
            self._append_row(group["finished_at"], np.asarray([finished_at], dtype=float))
            h5.flush()
        return index

    @staticmethod
    def _append_row(dataset: h5py.Dataset, row: np.ndarray) -> None:
        old_n = int(dataset.shape[0])
        dataset.resize(old_n + row.shape[0], axis=0)
        dataset[old_n : old_n + row.shape[0]] = row

    def load_completed(self) -> CompletedData:
        with h5py.File(self.data_path, "r") as h5:
            group = h5["completed"]
            return CompletedData(
                theta=group["theta"][:],
                outputs=group["outputs"][:],
                elapsed_s=group["elapsed_s"][:],
                worker_rank=group["worker_rank"][:],
                job_sequence=group["job_sequence"][:],
            )

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "time": now_seconds(),
            "type": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def update_status(self, status: dict[str, Any]) -> None:
        status = {"time": now_seconds(), **status}
        tmp_path = self.status_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.status_path)

    def completed_keys(self, decimals: int = 12) -> set[tuple[float, ...]]:
        data = self.load_completed()
        return {stable_theta_key(row, decimals=decimals) for row in data.theta}

    def latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.checkpoints_dir.glob("surrogate_v*.pt"))
        return checkpoints[-1] if checkpoints else None


def replay_queue_events(
    events: Iterable[dict[str, Any]],
    completed_keys: set[tuple[float, ...]],
    decimals: int = 12,
) -> list[dict[str, Any]]:
    """Return jobs that were queued or running but never completed."""

    jobs: dict[str, dict[str, Any]] = {}
    terminal: set[str] = set()
    for event in events:
        event_type = event.get("type")
        job_id = str(event.get("job_id", ""))
        if not job_id:
            continue
        if event_type == "job_queued":
            theta = np.asarray(event["theta"], dtype=float)
            jobs[job_id] = {
                "job_id": job_id,
                "sequence": int(event.get("job_sequence", event.get("sequence", 0))),
                "theta": theta,
                "attempt": int(event.get("attempt", 0)),
                "source": event.get("source", "replayed"),
            }
        elif event_type == "job_completed":
            terminal.add(job_id)
        elif event_type == "job_failed" and int(event.get("will_retry", 0)) == 0:
            terminal.add(job_id)
    pending: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, ...]] = set()
    for job_id, job in jobs.items():
        key = stable_theta_key(job["theta"], decimals=decimals)
        if job_id in terminal or key in completed_keys or key in seen_keys:
            continue
        pending.append(job)
        seen_keys.add(key)
    return pending
