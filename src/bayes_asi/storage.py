from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Bayes ASI targets POSIX MPI systems.
    _fcntl = None

SCHEMA_VERSION = 1
_CHECKPOINT_PATTERN = re.compile(r"surrogate_v(\d+)\.pt$")
_COMPLETED_DATASETS = (
    "theta",
    "outputs",
    "elapsed_s",
    "worker_rank",
    "job_sequence",
    "started_at",
    "finished_at",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def now_seconds() -> float:
    return time.time()


@contextmanager
def exclusive_run_lock(run_dir: Path):
    """Prevent concurrent rank-0 coordinators from sharing a run directory."""

    if _fcntl is None:
        raise RuntimeError("exclusive run locking requires a POSIX platform with fcntl")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(f"run directory is already active: {run_dir} ({owner})") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"hostname": socket.gethostname(), "pid": os.getpid(), "started_at": now_seconds()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield lock_path
        finally:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


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
    """Durable single-writer sample store with an append-only event journal."""

    def __init__(self, run_dir: Path, parameter_dim: int, output_dim: int) -> None:
        self.run_dir = Path(run_dir)
        self.parameter_dim = int(parameter_dim)
        self.output_dim = int(output_dim)
        if self.parameter_dim <= 0 or self.output_dim <= 0:
            raise ValueError("parameter_dim and output_dim must be positive")
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
            self._validate_or_set_attribute(h5, "schema_version", SCHEMA_VERSION)
            self._validate_or_set_attribute(h5, "parameter_dim", self.parameter_dim)
            self._validate_or_set_attribute(h5, "output_dim", self.output_dim)
            group = h5.require_group("completed")
            self._require_dataset(group, "theta", (0, self.parameter_dim), np.float64)
            self._require_dataset(group, "outputs", (0, self.output_dim), np.float64)
            self._require_dataset(group, "elapsed_s", (0,), np.float64)
            self._require_dataset(group, "worker_rank", (0,), np.int64)
            self._require_dataset(group, "job_sequence", (0,), np.int64)
            self._require_dataset(group, "started_at", (0,), np.float64)
            self._require_dataset(group, "finished_at", (0,), np.float64)
            self._validate_completed_group(group)

    @staticmethod
    def _validate_or_set_attribute(h5: h5py.File, name: str, expected: int) -> None:
        if name in h5.attrs and int(h5.attrs[name]) != int(expected):
            raise ValueError(f"HDF5 {name}={int(h5.attrs[name])}, expected {int(expected)}")
        h5.attrs[name] = int(expected)

    @staticmethod
    def _require_dataset(group: h5py.Group, name: str, shape: tuple[int, ...], dtype: Any) -> None:
        if name not in group:
            maxshape = (None,) + tuple(shape[1:])
            chunks = (1,) + tuple(shape[1:])
            group.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype)
            return
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"completed/{name} is not a dataset")
        if dataset.ndim != len(shape) or tuple(dataset.shape[1:]) != tuple(shape[1:]):
            raise ValueError(
                f"completed/{name} has trailing shape {tuple(dataset.shape[1:])}, expected {tuple(shape[1:])}"
            )

    @staticmethod
    def _validate_completed_group(group: h5py.Group) -> int:
        lengths = {name: int(group[name].shape[0]) for name in _COMPLETED_DATASETS}
        if len(set(lengths.values())) != 1:
            detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(f"inconsistent completed dataset lengths: {detail}")
        return next(iter(lengths.values()))

    def completed_count(self) -> int:
        """Read only the row count, avoiding a full sample-data load."""

        with h5py.File(self.data_path, "r") as h5:
            return self._validate_completed_group(h5["completed"])

    def snapshot_config(self, source: Path) -> tuple[Path, str]:
        """Store an immutable content-addressed copy of a run configuration."""

        source = Path(source)
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        directory = self.run_dir / "configs"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{digest[:12]}_{source.name}"
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        return destination, digest

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
        return self.append_completed_batch(
            theta=np.asarray(theta, dtype=float).reshape(1, self.parameter_dim),
            outputs=np.asarray(outputs, dtype=float).reshape(1, self.output_dim),
            elapsed_s=np.asarray([elapsed_s], dtype=float),
            worker_rank=np.asarray([worker_rank], dtype=np.int64),
            job_sequence=np.asarray([job_sequence], dtype=np.int64),
            started_at=np.asarray([started_at], dtype=float),
            finished_at=np.asarray([finished_at], dtype=float),
        )

    def append_completed_batch(
        self,
        *,
        theta: np.ndarray,
        outputs: np.ndarray,
        elapsed_s: np.ndarray,
        worker_rank: np.ndarray,
        job_sequence: np.ndarray,
        started_at: np.ndarray,
        finished_at: np.ndarray,
    ) -> int:
        """Append one or more rows and return the first inserted HDF5 index."""

        theta = np.asarray(theta, dtype=np.float64).reshape(-1, self.parameter_dim)
        outputs = np.asarray(outputs, dtype=np.float64).reshape(-1, self.output_dim)
        rows = int(theta.shape[0])
        values = {
            "theta": theta,
            "outputs": outputs,
            "elapsed_s": np.asarray(elapsed_s, dtype=np.float64).reshape(-1),
            "worker_rank": np.asarray(worker_rank, dtype=np.int64).reshape(-1),
            "job_sequence": np.asarray(job_sequence, dtype=np.int64).reshape(-1),
            "started_at": np.asarray(started_at, dtype=np.float64).reshape(-1),
            "finished_at": np.asarray(finished_at, dtype=np.float64).reshape(-1),
        }
        mismatched = [name for name, value in values.items() if value.shape[0] != rows]
        if mismatched:
            raise ValueError(f"completed batch row-count mismatch for: {', '.join(mismatched)}")
        for name in ("theta", "outputs", "elapsed_s", "started_at", "finished_at"):
            if not np.all(np.isfinite(values[name])):
                raise ValueError(f"completed batch {name} contains non-finite values")

        with h5py.File(self.data_path, "a") as h5:
            group = h5["completed"]
            start = self._validate_completed_group(group)
            if rows == 0:
                return start
            end = start + rows
            try:
                for name in _COMPLETED_DATASETS:
                    group[name].resize(end, axis=0)
                for name, value in values.items():
                    group[name][start:end] = value
                h5.flush()
            except Exception:
                for name in _COMPLETED_DATASETS:
                    group[name].resize(start, axis=0)
                h5.flush()
                raise
        return start

    def load_completed(self) -> CompletedData:
        with h5py.File(self.data_path, "r") as h5:
            group = h5["completed"]
            self._validate_completed_group(group)
            return CompletedData(
                theta=group["theta"][:],
                outputs=group["outputs"][:],
                elapsed_s=group["elapsed_s"][:],
                worker_rank=group["worker_rank"][:],
                job_sequence=group["job_sequence"][:],
            )

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must not be empty")
        event = {**payload, "time": now_seconds(), "type": event_type}
        line = json.dumps(event, sort_keys=True, allow_nan=False, default=_json_default) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_events(self) -> list[dict[str, Any]]:
        """Read valid events, tolerating only a truncated final journal line."""

        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines(keepends=True)
        events: list[dict[str, Any]] = []
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                is_truncated_tail = index == len(lines) - 1 and not raw_line.endswith("\n")
                if is_truncated_tail:
                    break
                raise ValueError(f"invalid event JSON on line {index + 1}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"event on line {index + 1} is not a JSON object")
            events.append(event)
        return events

    def update_status(self, status: dict[str, Any]) -> None:
        status = {**status, "time": now_seconds()}
        tmp_path = self.status_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.status_path)

    def completed_keys(self, decimals: int = 12) -> set[tuple[float, ...]]:
        data = self.load_completed()
        return {stable_theta_key(row, decimals=decimals) for row in data.theta}

    def checkpoint_paths(self, *, newest_first: bool = False) -> list[Path]:
        def version(path: Path) -> int:
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            return int(match.group(1)) if match else -1

        paths = [path for path in self.checkpoints_dir.glob("surrogate_v*.pt") if version(path) >= 0]
        return sorted(paths, key=version, reverse=newest_first)

    def latest_checkpoint(self) -> Path | None:
        checkpoints = self.checkpoint_paths(newest_first=True)
        return checkpoints[0] if checkpoints else None


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
        elif event_type in {"job_completed", "job_cancelled"}:
            terminal.add(job_id)
        elif event_type == "job_failed":
            if int(event.get("will_retry", 0)) == 0:
                terminal.add(job_id)
            elif job_id in jobs:
                jobs[job_id]["attempt"] = int(event.get("attempt", jobs[job_id]["attempt"])) + 1
    pending: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, ...]] = set()
    for job_id, job in jobs.items():
        key = stable_theta_key(job["theta"], decimals=decimals)
        if job_id in terminal or key in completed_keys or key in seen_keys:
            continue
        pending.append(job)
        seen_keys.add(key)
    return pending
