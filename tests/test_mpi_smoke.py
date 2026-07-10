from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXAMPLE_CASES = (
    ("nonlinear", "bayes_asi.examples.synthetic", 2, 2, 2),
    ("linear_gaussian", "bayes_asi.examples.linear_gaussian", 2, 2, 3),
    ("bounded_decay", "bayes_asi.examples.bounded_decay", 3, 3, 16),
    ("bimodal", "bayes_asi.examples.bimodal", 1, 1, 1),
)


@unittest.skipUnless(os.environ.get("BAYES_ASI_MPI_TEST") == "1", "set BAYES_ASI_MPI_TEST=1 to run MPI smoke test")
class MPISmokeTests(unittest.TestCase):
    def test_short_runs_for_all_synthetic_examples(self) -> None:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            self.skipTest("mpiexec is not available")
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for name, module, latent_dim, parameter_dim, output_dim in EXAMPLE_CASES:
                with self.subTest(example=name):
                    run_dir = Path(tmp) / f"run_{name}"
                    config = Path(tmp) / f"{name}.yaml"
                    config.write_text(
                        self._config_text(
                            run_dir=run_dir,
                            module=module,
                            latent_dim=latent_dim,
                            parameter_dim=parameter_dim,
                            output_dim=output_dim,
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
                    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                    self.assertTrue(status["final"])
                    self.assertGreaterEqual(status["completed_count"], 4)
                    self.assertIsNotNone(status["surrogate_version"])
                    self.assertTrue(any((run_dir / "checkpoints").glob("surrogate_v*.pt")))

    @staticmethod
    def _config_text(
        *,
        run_dir: Path,
        module: str,
        latent_dim: int,
        parameter_dim: int,
        output_dim: int,
    ) -> str:
        return textwrap.dedent(
            f"""
            run:
              output_dir: {run_dir}
              time_budget_seconds: 2
              status_interval_seconds: 0.25
              seed: 11
            problem:
              latent_dim: {latent_dim}
              parameter_dim: {parameter_dim}
              output_dim: {output_dim}
            solver:
              factory: {module}:make_solver
            posterior:
              factory: {module}:make_posterior
            initial_design:
              n_points: 6
            surrogate:
              hidden_layers: [8]
              max_epochs: 20
              min_train_samples: 4
              update_after_new_samples: 2
              validation_fraction: 0.0
              failure_retry_seconds: 0.1
            hmc:
              n_chains: 2
              warmup_steps: 5
              samples_per_chain: 8
              min_leapfrog_steps: 2
              max_leapfrog_steps: 4
            active_learning:
              candidate_pool_size: 16
              batch_max_size: 2
              failure_retry_seconds: 0.1
            queue:
              target_seconds_ahead: 0.25
            """
        )


if __name__ == "__main__":
    unittest.main()
