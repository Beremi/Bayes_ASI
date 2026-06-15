from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bayes_asi.config import load_config
from bayes_asi.storage import HDF5Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayes-asi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run adaptive inversion under MPI")
    run_parser.add_argument("config", help="YAML configuration path")

    status_parser = subparsers.add_parser("status", help="print run status JSON")
    status_parser.add_argument("run_dir", help="run directory")

    export_parser = subparsers.add_parser("export", help="export completed samples")
    export_parser.add_argument("run_dir", help="run directory")
    export_parser.add_argument("--format", choices=["npz", "csv"], default="npz")
    export_parser.add_argument("--output", help="output path")

    args = parser.parse_args(argv)
    if args.command == "run":
        from bayes_asi.runtime import run_mpi

        run_mpi(args.config)
        return 0
    if args.command == "status":
        return _status(Path(args.run_dir))
    if args.command == "export":
        return _export(Path(args.run_dir), args.format, Path(args.output) if args.output else None)
    raise AssertionError(args.command)


def _status(run_dir: Path) -> int:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"missing status file: {status_path}")
    print(status_path.read_text(encoding="utf-8").rstrip())
    return 0


def _export(run_dir: Path, export_format: str, output_path: Path | None) -> int:
    data_path = run_dir / "data.h5"
    if not data_path.exists():
        raise FileNotFoundError(f"missing HDF5 data file: {data_path}")
    config_path = run_dir / "resolved_config_hint.json"
    parameter_dim, output_dim = _read_dims(data_path)
    store = HDF5Store(run_dir, parameter_dim=parameter_dim, output_dim=output_dim)
    data = store.load_completed()
    if output_path is None:
        output_path = run_dir / f"completed.{export_format}"
    if export_format == "npz":
        np.savez(
            output_path,
            theta=data.theta,
            outputs=data.outputs,
            elapsed_s=data.elapsed_s,
            worker_rank=data.worker_rank,
            job_sequence=data.job_sequence,
        )
    else:
        header = [f"theta_{i}" for i in range(data.theta.shape[1])]
        header += [f"output_{i}" for i in range(data.outputs.shape[1])]
        header += ["elapsed_s", "worker_rank", "job_sequence"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(data.n):
                writer.writerow(
                    data.theta[i].tolist()
                    + data.outputs[i].tolist()
                    + [float(data.elapsed_s[i]), int(data.worker_rank[i]), int(data.job_sequence[i])]
                )
    if config_path.exists():
        pass
    print(output_path)
    return 0


def _read_dims(data_path: Path) -> tuple[int, int]:
    import h5py

    with h5py.File(data_path, "r") as h5:
        return int(h5.attrs["parameter_dim"]), int(h5.attrs["output_dim"])


if __name__ == "__main__":
    raise SystemExit(main())

