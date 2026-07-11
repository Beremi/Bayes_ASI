from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from collections.abc import Iterable, Mapping, Sequence
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

SCHEMA_VERSION = 2
_CHECKPOINT_PATTERN = re.compile(r"surrogate_v(\d+)\.pt$")
_COMPLETED_DATASETS = (
    "z",
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
    z: np.ndarray
    theta: np.ndarray
    outputs: np.ndarray
    elapsed_s: np.ndarray
    worker_rank: np.ndarray
    job_sequence: np.ndarray

    @property
    def n(self) -> int:
        return int(self.theta.shape[0])

    @property
    def model_outputs(self) -> np.ndarray:
        """Dense model values used as the surrogate training target."""

        return self.outputs


@dataclass(slots=True)
class AuxiliaryData:
    """Row-aligned values and a mask identifying rows where a field is present."""

    values: np.ndarray
    present: np.ndarray


class HDF5Store:
    """Durable single-writer sample store with an append-only event journal."""

    def __init__(
        self,
        run_dir: Path,
        parameter_dim: int,
        output_dim: int,
        latent_dim: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.parameter_dim = int(parameter_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = self.parameter_dim if latent_dim is None else int(latent_dim)
        if self.latent_dim <= 0 or self.parameter_dim <= 0 or self.output_dim <= 0:
            raise ValueError("latent_dim, parameter_dim, and output_dim must be positive")
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
            existing_version = int(h5.attrs.get("schema_version", SCHEMA_VERSION))
            if existing_version not in (1, SCHEMA_VERSION):
                raise ValueError(
                    f"unsupported HDF5 schema_version={existing_version}; expected 1 or {SCHEMA_VERSION}"
                )
            self._validate_or_set_attribute(h5, "parameter_dim", self.parameter_dim)
            self._validate_or_set_attribute(h5, "output_dim", self.output_dim)
            if existing_version == SCHEMA_VERSION:
                self._validate_or_set_attribute(h5, "latent_dim", self.latent_dim)
            group = h5.require_group("completed")
            if existing_version == 1:
                old_names = tuple(name for name in _COMPLETED_DATASETS if name != "z")
                old_count = self._validate_completed_group(group, names=old_names)
                self._require_dataset(group, "z", (old_count, self.latent_dim), np.float64)
                if old_count:
                    group["z"][:] = np.nan
                h5.attrs["latent_coordinates_pending"] = int(old_count > 0)
                h5.attrs["latent_dim"] = self.latent_dim
                h5.attrs["schema_version"] = SCHEMA_VERSION
            else:
                if "latent_coordinates_pending" not in h5.attrs:
                    h5.attrs["latent_coordinates_pending"] = 0
                h5.attrs["schema_version"] = SCHEMA_VERSION
            self._require_dataset(group, "z", (0, self.latent_dim), np.float64)
            self._require_dataset(group, "theta", (0, self.parameter_dim), np.float64)
            self._require_dataset(group, "outputs", (0, self.output_dim), np.float64)
            self._require_dataset(group, "elapsed_s", (0,), np.float64)
            self._require_dataset(group, "worker_rank", (0,), np.int64)
            self._require_dataset(group, "job_sequence", (0,), np.int64)
            self._require_dataset(group, "started_at", (0,), np.float64)
            self._require_dataset(group, "finished_at", (0,), np.float64)
            count = self._validate_completed_group(group)
            auxiliary = h5.require_group("auxiliary")
            self._validate_auxiliary_group(auxiliary, count)

    @staticmethod
    def _validate_or_set_attribute(h5: h5py.File, name: str, expected: int) -> None:
        if name in h5.attrs and int(h5.attrs[name]) != int(expected):
            raise ValueError(f"HDF5 {name}={int(h5.attrs[name])}, expected {int(expected)}")
        h5.attrs[name] = int(expected)

    @staticmethod
    def _require_dataset(group: h5py.Group, name: str, shape: tuple[int, ...], dtype: Any) -> None:
        if name not in group:
            maxshape = (None,) + tuple(shape[1:])
            chunks = HDF5Store._row_chunk_shape(tuple(shape[1:]), np.dtype(dtype))
            group.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype)
            return
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"completed/{name} is not a dataset")
        if dataset.ndim != len(shape) or tuple(dataset.shape[1:]) != tuple(shape[1:]):
            raise ValueError(
                f"completed/{name} has trailing shape {tuple(dataset.shape[1:])}, expected {tuple(shape[1:])}"
            )
        if dataset.dtype != np.dtype(dtype):
            raise ValueError(f"completed/{name} has dtype {dataset.dtype}, expected {np.dtype(dtype)}")

    @staticmethod
    def _row_chunk_shape(sample_shape: tuple[int, ...], dtype: np.dtype[Any]) -> tuple[int, ...]:
        elements_per_row = max(1, int(np.prod(sample_shape, dtype=np.int64)))
        bytes_per_row = max(1, elements_per_row * dtype.itemsize)
        chunk_rows = max(1, min(1024, 1024 * 1024 // bytes_per_row))
        return (chunk_rows,) + sample_shape

    @staticmethod
    def _validate_completed_group(
        group: h5py.Group,
        *,
        names: Sequence[str] = _COMPLETED_DATASETS,
    ) -> int:
        missing = [name for name in names if name not in group or not isinstance(group[name], h5py.Dataset)]
        if missing:
            raise ValueError(f"completed group is missing dataset(s): {', '.join(missing)}")
        lengths = {name: int(group[name].shape[0]) for name in names}
        if len(set(lengths.values())) != 1:
            detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(f"inconsistent completed dataset lengths: {detail}")
        return next(iter(lengths.values()))

    @staticmethod
    def _field_identifier(name: str) -> str:
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    @classmethod
    def _auxiliary_fields(cls, root: h5py.Group) -> dict[str, h5py.Group]:
        fields: dict[str, h5py.Group] = {}
        for identifier, item in root.items():
            if not isinstance(item, h5py.Group):
                raise ValueError(f"auxiliary/{identifier} is not a group")
            raw_name = item.attrs.get("name")
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name or "")
            if not name or cls._field_identifier(name) != identifier:
                raise ValueError(f"auxiliary/{identifier} has an invalid logical name")
            if set(item) != {"present", "values"}:
                raise ValueError(f"auxiliary field {name!r} must contain present and values datasets")
            if not isinstance(item["present"], h5py.Dataset) or not isinstance(item["values"], h5py.Dataset):
                raise ValueError(f"auxiliary field {name!r} has invalid datasets")
            fields[name] = item
        return fields

    @classmethod
    def _validate_auxiliary_group(cls, root: h5py.Group, expected_rows: int) -> dict[str, h5py.Group]:
        fields = cls._auxiliary_fields(root)
        for name, group in fields.items():
            values = group["values"]
            present = group["present"]
            if int(values.shape[0]) != expected_rows or present.shape != (expected_rows,):
                raise ValueError(
                    f"auxiliary field {name!r} has {values.shape[0]} value rows and "
                    f"{present.shape[0]} mask rows; expected {expected_rows}"
                )
            if present.dtype != np.dtype(np.bool_):
                raise ValueError(f"auxiliary field {name!r} presence mask must be boolean")
        return fields

    def latent_coordinates_pending(self) -> bool:
        with h5py.File(self.data_path, "r") as h5:
            return bool(int(h5.attrs.get("latent_coordinates_pending", 0)))

    def complete_latent_migration(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.latent_dim)
        if not np.all(np.isfinite(z)):
            raise ValueError("migrated standard-normal coordinates must be finite")
        with h5py.File(self.data_path, "a") as h5:
            group = h5["completed"]
            count = self._validate_completed_group(group)
            if z.shape[0] != count:
                raise ValueError(f"migrated coordinate rows={z.shape[0]}, expected {count}")
            if not bool(int(h5.attrs.get("latent_coordinates_pending", 0))):
                if not np.allclose(group["z"][:], z, rtol=0.0, atol=0.0):
                    raise ValueError("stored standard-normal coordinates are already populated")
                return
            group["z"][:] = z
            h5.attrs["latent_coordinates_pending"] = 0
            h5.flush()

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
        destination = directory / f"{digest}_{source.name}"
        if destination.exists():
            stored_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if stored_digest != digest:
                raise RuntimeError(f"configuration snapshot hash mismatch: {destination}")
            return destination, digest

        temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, digest

    def append_completed(
        self,
        *,
        z: np.ndarray,
        theta: np.ndarray,
        outputs: np.ndarray,
        elapsed_s: float,
        worker_rank: int,
        job_sequence: int,
        started_at: float,
        finished_at: float,
        auxiliary: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        return self.append_completed_batch(
            z=np.asarray(z, dtype=float).reshape(1, self.latent_dim),
            theta=np.asarray(theta, dtype=float).reshape(1, self.parameter_dim),
            outputs=np.asarray(outputs, dtype=float).reshape(1, self.output_dim),
            elapsed_s=np.asarray([elapsed_s], dtype=float),
            worker_rank=np.asarray([worker_rank], dtype=np.int64),
            job_sequence=np.asarray([job_sequence], dtype=np.int64),
            started_at=np.asarray([started_at], dtype=float),
            finished_at=np.asarray([finished_at], dtype=float),
            auxiliary=[{} if auxiliary is None else auxiliary],
        )

    def append_completed_batch(
        self,
        *,
        z: np.ndarray,
        theta: np.ndarray,
        outputs: np.ndarray,
        elapsed_s: np.ndarray,
        worker_rank: np.ndarray,
        job_sequence: np.ndarray,
        started_at: np.ndarray,
        finished_at: np.ndarray,
        auxiliary: Sequence[Mapping[str, np.ndarray]] | None = None,
    ) -> int:
        """Append one or more rows and return the first inserted HDF5 index."""

        z = np.asarray(z, dtype=np.float64).reshape(-1, self.latent_dim)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1, self.parameter_dim)
        outputs = np.asarray(outputs, dtype=np.float64).reshape(-1, self.output_dim)
        rows = int(theta.shape[0])
        values = {
            "z": z,
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
        for name in ("z", "theta", "outputs", "elapsed_s", "started_at", "finished_at"):
            if not np.all(np.isfinite(values[name])):
                raise ValueError(f"completed batch {name} contains non-finite values")
        auxiliary_rows, auxiliary_schema = self._normalize_auxiliary_batch(auxiliary, rows)

        with h5py.File(self.data_path, "a") as h5:
            group = h5["completed"]
            start = self._validate_completed_group(group)
            auxiliary_root = h5["auxiliary"]
            fields = self._validate_auxiliary_group(auxiliary_root, start)
            if rows == 0:
                return start
            end = start + rows
            new_identifiers: list[str] = []
            try:
                for name, (shape, dtype) in auxiliary_schema.items():
                    if name in fields:
                        dataset = fields[name]["values"]
                        if tuple(dataset.shape[1:]) != shape or dataset.dtype != dtype:
                            raise ValueError(
                                f"auxiliary field {name!r} has shape/dtype "
                                f"{tuple(dataset.shape[1:])}/{dataset.dtype}, got {shape}/{dtype}"
                            )
                        continue
                    identifier = self._field_identifier(name)
                    new_identifiers.append(identifier)
                    fields[name] = self._create_auxiliary_field(
                        auxiliary_root,
                        identifier=identifier,
                        name=name,
                        existing_rows=start,
                        sample_shape=shape,
                        dtype=dtype,
                    )
                for name in _COMPLETED_DATASETS:
                    group[name].resize(end, axis=0)
                for name, value in values.items():
                    group[name][start:end] = value
                for field in fields.values():
                    field["values"].resize(end, axis=0)
                    field["present"].resize(end, axis=0)
                    field["present"][start:end] = False
                for offset, row in enumerate(auxiliary_rows):
                    for name, value in row.items():
                        fields[name]["values"][start + offset] = value
                        fields[name]["present"][start + offset] = True
                h5.flush()
            except Exception:
                for name in _COMPLETED_DATASETS:
                    group[name].resize(start, axis=0)
                for name, field in fields.items():
                    identifier = self._field_identifier(name)
                    if identifier not in new_identifiers:
                        field["values"].resize(start, axis=0)
                        field["present"].resize(start, axis=0)
                for identifier in new_identifiers:
                    if identifier in auxiliary_root:
                        del auxiliary_root[identifier]
                h5.flush()
                raise
        return start

    @staticmethod
    def _normalize_auxiliary_batch(
        auxiliary: Sequence[Mapping[str, np.ndarray]] | None,
        rows: int,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, tuple[tuple[int, ...], np.dtype[Any]]]]:
        if auxiliary is None:
            normalized: list[dict[str, np.ndarray]] = [{} for _ in range(rows)]
        else:
            if len(auxiliary) != rows:
                raise ValueError(f"auxiliary batch rows={len(auxiliary)}, expected {rows}")
            normalized = []
            for row in auxiliary:
                if not isinstance(row, Mapping):
                    raise TypeError("each auxiliary row must be a mapping")
                normalized_row: dict[str, np.ndarray] = {}
                for raw_name, raw_value in row.items():
                    name = str(raw_name)
                    if not name or name != name.strip() or "\x00" in name:
                        raise ValueError("auxiliary field names must be non-empty, trimmed strings without NUL")
                    value = np.asarray(raw_value)
                    if value.dtype.kind not in "biufc":
                        raise TypeError(
                            f"auxiliary field {name!r} must use a fixed-size numeric or boolean dtype"
                        )
                    if any(dimension <= 0 for dimension in value.shape):
                        raise ValueError(f"auxiliary field {name!r} dimensions must be non-empty")
                    normalized_row[name] = value
                normalized.append(normalized_row)

        schema: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for row in normalized:
            for name, value in row.items():
                description = (tuple(value.shape), value.dtype)
                if name in schema and schema[name] != description:
                    raise ValueError(
                        f"auxiliary field {name!r} changes shape/dtype within a batch: "
                        f"{schema[name]} versus {description}"
                    )
                schema[name] = description
        return normalized, schema

    @staticmethod
    def _create_auxiliary_field(
        root: h5py.Group,
        *,
        identifier: str,
        name: str,
        existing_rows: int,
        sample_shape: tuple[int, ...],
        dtype: np.dtype[Any],
    ) -> h5py.Group:
        field = root.create_group(identifier)
        field.attrs["name"] = name
        chunks = HDF5Store._row_chunk_shape(sample_shape, dtype)
        field.create_dataset(
            "values",
            shape=(existing_rows,) + sample_shape,
            maxshape=(None,) + sample_shape,
            chunks=chunks,
            dtype=dtype,
            compression="lzf",
            shuffle=True,
            fletcher32=True,
        )
        field.create_dataset(
            "present",
            shape=(existing_rows,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.bool_,
            compression="lzf",
            fletcher32=True,
        )
        return field

    def load_completed(self, *, allow_pending_latent: bool = False) -> CompletedData:
        with h5py.File(self.data_path, "r") as h5:
            group = h5["completed"]
            self._validate_completed_group(group)
            if bool(int(h5.attrs.get("latent_coordinates_pending", 0))) and not allow_pending_latent:
                raise RuntimeError("stored samples require migration to standard-normal coordinates")
            return CompletedData(
                z=group["z"][:],
                theta=group["theta"][:],
                outputs=group["outputs"][:],
                elapsed_s=group["elapsed_s"][:],
                worker_rank=group["worker_rank"][:],
                job_sequence=group["job_sequence"][:],
            )

    def list_auxiliary_fields(self) -> list[str]:
        with h5py.File(self.data_path, "r") as h5:
            count = self._validate_completed_group(h5["completed"])
            return sorted(self._validate_auxiliary_group(h5["auxiliary"], count))

    def load_auxiliary(self, name: str, *, rows: slice | None = None) -> AuxiliaryData:
        selection = slice(None) if rows is None else rows
        with h5py.File(self.data_path, "r") as h5:
            count = self._validate_completed_group(h5["completed"])
            fields = self._validate_auxiliary_group(h5["auxiliary"], count)
            if name not in fields:
                raise KeyError(f"unknown auxiliary field {name!r}")
            field = fields[name]
            return AuxiliaryData(
                values=np.asarray(field["values"][selection]),
                present=np.asarray(field["present"][selection], dtype=bool),
            )

    def load_all_auxiliary(self) -> dict[str, AuxiliaryData]:
        with h5py.File(self.data_path, "r") as h5:
            count = self._validate_completed_group(h5["completed"])
            fields = self._validate_auxiliary_group(h5["auxiliary"], count)
            return {
                name: AuxiliaryData(values=np.asarray(field["values"][:]), present=np.asarray(field["present"][:]))
                for name, field in fields.items()
            }

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
                "z": None if "z" not in event else np.asarray(event["z"], dtype=float),
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
