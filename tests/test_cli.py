from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bayes_asi.cli import main
from bayes_asi.storage import HDF5Store

MINIMAL_CONFIG = """
problem:
  latent_dim: 2
  parameter_dim: 2
  output_dim: 2
solver:
  factory: bayes_asi.examples.synthetic:make_solver
posterior:
  factory: bayes_asi.examples.synthetic:make_posterior
"""


class CLITests(unittest.TestCase):
    def test_validate_resolves_configured_factories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(MINIMAL_CONFIG, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["validate", str(path)])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["problem"]["parameter_dim"], 2)

    def test_status_outputs_valid_pretty_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            HDF5Store(run_dir, 1, 1).update_status({"completed_count": 2})
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["status", str(run_dir)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["completed_count"], 2)

    def test_export_writes_npz_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            store = HDF5Store(run_dir, 2, 1)
            store.append_completed(
                z=np.array([-0.5, 0.5]),
                theta=np.array([1.0, 2.0]),
                outputs=np.array([3.0]),
                elapsed_s=0.5,
                worker_rank=1,
                job_sequence=4,
                started_at=1.0,
                finished_at=1.5,
                auxiliary={"experiments/source_a": np.array([4.0, 5.0], dtype=np.float32)},
            )
            npz_path = Path(tmp) / "exports" / "samples.npz"
            csv_path = Path(tmp) / "exports" / "samples.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["export", str(run_dir), "--format", "npz", "--output", str(npz_path)])
                main(["export", str(run_dir), "--format", "csv", "--output", str(csv_path)])

            with np.load(npz_path) as data:
                self.assertTrue(np.array_equal(data["z"], [[-0.5, 0.5]]))
                self.assertTrue(np.array_equal(data["theta"], [[1.0, 2.0]]))
                self.assertTrue(np.array_equal(data["model_outputs"], [[3.0]]))
                self.assertTrue(np.array_equal(data["auxiliary/experiments/source_a"], [[4.0, 5.0]]))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows[0],
                ["z_0", "z_1", "theta_0", "theta_1", "output_0", "elapsed_s", "worker_rank", "job_sequence"],
            )
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
