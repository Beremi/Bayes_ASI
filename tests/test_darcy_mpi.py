from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import h5py
import numpy as np


@unittest.skipUnless(
    os.environ.get("BAYES_ASI_DARCY_MPI_TEST") == "1",
    "set BAYES_ASI_DARCY_MPI_TEST=1 to run the expensive Darcy MPI test",
)
class DarcyMPITests(unittest.TestCase):
    def test_material_inversion_with_one_second_solver_jobs(self) -> None:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            self.skipTest("mpiexec is not available")
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            config_path = Path(tmp) / "darcy.yaml"
            config_path.write_text(self._config(run_dir), encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")

            started = time.monotonic()
            subprocess.run(
                [mpiexec, "-n", "3", sys.executable, "-m", "bayes_asi.cli", "run", str(config_path)],
                cwd=repository,
                env=environment,
                check=True,
                timeout=25,
            )
            wall_time = time.monotonic() - started

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertTrue(status["final"])
            self.assertGreaterEqual(status["completed_count"], 4)
            self.assertIsNotNone(status["surrogate_version"])
            self.assertTrue(any((run_dir / "checkpoints").glob("surrogate_v*.pt")))
            self.assertTrue(any((run_dir / "configs").glob("*_darcy.yaml")))
            with h5py.File(run_dir / "data.h5", "r") as h5:
                elapsed = np.asarray(h5["completed/elapsed_s"])
                outputs = np.asarray(h5["completed/outputs"])
            self.assertTrue(np.all(elapsed >= 1.4))
            self.assertTrue(np.all(np.isfinite(outputs)))
            self.assertGreaterEqual(wall_time, 2.8)
            self.assertLess(wall_time, 25.0)

    @staticmethod
    def _config(run_dir: Path) -> str:
        return textwrap.dedent(
            f"""
            run:
              output_dir: {run_dir}
              time_budget_seconds: 9
              status_interval_seconds: 0.25
              seed: 20260711
            problem:
              latent_dim: 3
              parameter_dim: 3
              output_dim: 13
            solver:
              factory: bayes_asi.examples.darcy_flow:make_solver
              kwargs:
                nx: 48
                ny: 48
                minimum_runtime_seconds: 1.5
            posterior:
              factory: bayes_asi.examples.darcy_flow:make_posterior
              kwargs:
                nx: 48
                ny: 48
            initial_design:
              n_points: 4
            surrogate:
              hidden_layers: [16]
              max_epochs: 35
              min_train_samples: 4
              update_after_new_samples: 2
              validation_fraction: 0.0
              failure_retry_seconds: 0.1
            hmc:
              n_chains: 2
              warmup_steps: 6
              samples_per_chain: 10
              min_leapfrog_steps: 2
              max_leapfrog_steps: 5
            active_learning:
              candidate_pool_size: 20
              batch_max_size: 2
              failure_retry_seconds: 0.1
            queue:
              min_pending_per_worker: 0
              max_pending_per_worker: 1
              refill_threshold_pending_per_worker: 0
              target_seconds_ahead: 2.0
            """
        )


if __name__ == "__main__":
    unittest.main()
