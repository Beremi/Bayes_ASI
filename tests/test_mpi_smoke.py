from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("BAYES_ASI_MPI_TEST") == "1", "set BAYES_ASI_MPI_TEST=1 to run MPI smoke test")
class MPISmokeTests(unittest.TestCase):
    def test_short_synthetic_run(self) -> None:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            self.skipTest("mpiexec is not available")
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "synthetic.yaml"
            run_dir = Path(tmp) / "run"
            config.write_text(
                textwrap.dedent(
                    f"""
                    run:
                      output_dir: {run_dir}
                      time_budget_seconds: 4
                      status_interval_seconds: 0.5
                      seed: 11
                    problem:
                      latent_dim: 2
                      parameter_dim: 2
                      output_dim: 2
                    solver:
                      factory: bayes_asi.examples.synthetic:make_solver
                    posterior:
                      factory: bayes_asi.examples.synthetic:make_posterior
                    initial_design:
                      n_points: 6
                    surrogate:
                      hidden_layers: [8]
                      max_epochs: 30
                      min_train_samples: 4
                      update_after_new_samples: 2
                      validation_fraction: 0.0
                    hmc:
                      n_chains: 2
                      warmup_steps: 8
                      samples_per_chain: 10
                      min_leapfrog_steps: 2
                      max_leapfrog_steps: 5
                    active_learning:
                      candidate_pool_size: 20
                      batch_max_size: 2
                    queue:
                      target_seconds_ahead: 0.5
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo / "src")
            subprocess.run(
                [mpiexec, "-n", "3", sys.executable, "-m", "bayes_asi.cli", "run", str(config)],
                cwd=repo,
                env=env,
                check=True,
                timeout=20,
            )
            self.assertTrue((run_dir / "data.h5").exists())
            self.assertTrue((run_dir / "status.json").exists())


if __name__ == "__main__":
    unittest.main()

