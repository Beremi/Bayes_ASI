from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from bayes_asi import __version__
from bayes_asi.config import load_config, resolve_dotted
from bayes_asi.storage import HDF5Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayes-asi")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run adaptive inversion under MPI")
    run_parser.add_argument("config", help="YAML configuration path")

    validate_parser = subparsers.add_parser("validate", help="validate configuration and factory paths")
    validate_parser.add_argument("config", help="YAML configuration path")

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
    if args.command == "validate":
        return _validate(Path(args.config))
    if args.command == "status":
        return _status(Path(args.run_dir))
    if args.command == "export":
        return _export(Path(args.run_dir), args.format, Path(args.output) if args.output else None)
    raise AssertionError(args.command)


def _status(run_dir: Path) -> int:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"missing status file: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _validate(config_path: Path) -> int:
    config = load_config(config_path)
    factories = {
        "solver": config.solver.factory,
        "posterior": config.posterior.factory,
    }
    if config.initial_data is not None:
        factories["initial_data"] = config.initial_data.factory
    for path in factories.values():
        resolve_dotted(path)
    summary = {
        "config_path": str(config.config_path),
        "factories": factories,
        "output_dir": str(config.run.output_dir),
        "problem": {
            "latent_dim": config.problem.latent_dim,
            "parameter_dim": config.problem.parameter_dim,
            "output_dim": config.problem.output_dim,
        },
        "valid": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _export(run_dir: Path, export_format: str, output_path: Path | None) -> int:
    data_path = run_dir / "data.h5"
    if not data_path.exists():
        raise FileNotFoundError(f"missing HDF5 data file: {data_path}")
    latent_dim, parameter_dim, output_dim = _read_dims(data_path)
    store = HDF5Store(
        run_dir,
        parameter_dim=parameter_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
    )
    data = store.load_completed()
    if output_path is None:
        output_path = run_dir / f"completed.{export_format}"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "npz":
        temporary = output_path.with_suffix(".tmp.npz")
        payload = {
            "z": data.z,
            "theta": data.theta,
            "model_outputs": data.outputs,
            "outputs": data.outputs,
            "elapsed_s": data.elapsed_s,
            "worker_rank": data.worker_rank,
            "job_sequence": data.job_sequence,
        }
        for name, auxiliary in store.load_all_auxiliary().items():
            payload[f"auxiliary/{name}"] = auxiliary.values
            payload[f"auxiliary_present/{name}"] = auxiliary.present
        np.savez_compressed(temporary, **payload)  # type: ignore[arg-type]
        os.replace(temporary, output_path)
    else:
        header = [f"z_{i}" for i in range(data.z.shape[1])]
        header += [f"theta_{i}" for i in range(data.theta.shape[1])]
        header += [f"output_{i}" for i in range(data.outputs.shape[1])]
        header += ["elapsed_s", "worker_rank", "job_sequence"]
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(data.n):
                writer.writerow(
                    data.z[i].tolist()
                    + data.theta[i].tolist()
                    + data.outputs[i].tolist()
                        + [float(data.elapsed_s[i]), int(data.worker_rank[i]), int(data.job_sequence[i])]
                    )
        os.replace(temporary, output_path)
    print(output_path)
    return 0


def _read_dims(data_path: Path) -> tuple[int, int, int]:
    import h5py

    with h5py.File(data_path, "r") as h5:
        parameter_dim = int(h5.attrs["parameter_dim"])
        return int(h5.attrs.get("latent_dim", parameter_dim)), parameter_dim, int(h5.attrs["output_dim"])


if __name__ == "__main__":
    raise SystemExit(main())
