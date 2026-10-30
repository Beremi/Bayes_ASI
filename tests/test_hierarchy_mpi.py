from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml


@unittest.skipUnless(os.environ.get("BAYES_ASI_HIERARCHY_MPI_TEST") == "1", "opt-in hierarchy MPI test")
class HierarchyMPITests(unittest.TestCase):
    def test_hierarchy_and_restart_keep_solver_truth_separate(self) -> None:
        mpi = shutil.which("mpiexec")
        if mpi is None:
            self.skipTest("mpiexec unavailable")
        repo = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((repo / "examples/hierarchy.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            config["run"].update(output_dir=str(output), time_budget_seconds=4)
            config["surrogate"].update(max_epochs=20, hidden_layers=[16], min_train_samples=4)
            config["initial_design"]["n_points"] = 6
            config["hierarchy"]["student"].update(max_epochs=15, hidden_layers=[4])
            config["hierarchy"].update(distillation_points=32, teacher_screen_size=8)
            config["hmc"].update(n_chains=2, warmup_steps=4, samples_per_chain=16, max_leapfrog_steps=4)
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            previous_count = 0
            for _ in range(2):
                result = subprocess.run(
                    [mpi, "-n", "3", sys.executable, "-m", "bayes_asi.cli", "run", str(config_path)],
                    cwd=repo, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                status = json.loads((output / "status.json").read_text())
                self.assertTrue(status["final"])
                self.assertGreater(status["completed_count"], previous_count)
                previous_count = status["completed_count"]
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["type"] == "hierarchy_student_trained" for e in events))
            selected = [e for e in events if e["type"] == "active_batch_selected"]
            self.assertTrue(selected)
            self.assertTrue(any(e["hmc_surrogate_role"] == "student" for e in selected))
            self.assertTrue(all(e["teacher_screen_evaluated"] <= 8 for e in selected))
            with h5py.File(output / "data.h5") as h5:
                data = h5["completed"]
                theta, outputs = np.asarray(data["theta"]), np.asarray(data["outputs"])
                # This independent linear reference catches pseudo-label contamination.
                for row, observed in zip(theta, outputs, strict=True):
                    u, v = row
                    expected = [u + .25*v + .1, -.5*u + v - .2, .75*u - 1.25*v + .3]
                    magnitude = np.abs([[1., .25], [-.5, 1.], [.75, -1.25]]) @ np.abs(row)
                    tolerance = 8*np.finfo(np.float32).eps*(magnitude + [.1, .2, .3])
                    self.assertTrue(np.all(np.abs(observed - expected) <= tolerance))


if __name__ == "__main__":
    unittest.main()
