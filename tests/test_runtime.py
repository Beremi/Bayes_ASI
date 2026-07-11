from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bayes_asi.config import load_config
from bayes_asi.runtime import Coordinator, _evaluate_worker_job
from bayes_asi.types import ForwardResult


class FakeComm:
    def Get_size(self) -> int:
        return 2


def make_initial_data(config):
    del config
    return {
        "theta": np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        "outputs": np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]),
        "metadata": {"source": "unit-test"},
    }


class NonFiniteSolver:
    def evaluate(self, theta, context):
        del theta, context
        return ForwardResult(outputs=np.array([np.nan, 1.0]))


class LegacyPosteriorWithoutPriorTransform:
    pass


class RuntimeTests(unittest.TestCase):
    def _config(self, run_dir: Path, *, initial_data: bool = True, retry_seconds: float = 60.0):
        initial_section = (
            "initial_data:\n  factory: test_runtime:make_initial_data\n"
            if initial_data
            else ""
        )
        path = run_dir.parent / f"config_{run_dir.name}.yaml"
        path.write_text(
            f"""
run:
  output_dir: {run_dir}
  time_budget_seconds: 10
problem:
  latent_dim: 2
  parameter_dim: 2
  output_dim: 2
{initial_section}solver:
  factory: bayes_asi.examples.synthetic:make_solver
posterior:
  factory: bayes_asi.examples.synthetic:make_posterior
initial_design:
  n_points: 0
surrogate:
  hidden_layers: [4]
  max_epochs: 2
  min_train_samples: 2
  validation_fraction: 0.0
  failure_retry_seconds: {retry_seconds}
hmc:
  n_chains: 1
  warmup_steps: 0
  samples_per_chain: 1
""",
            encoding="utf-8",
        )
        return load_config(path)

    def test_initial_data_is_deduplicated_and_imported_as_a_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()

            self.assertEqual(coordinator.completed_count, 2)
            self.assertEqual(coordinator.store.completed_count(), 2)
            events = coordinator.store.read_events()
            loaded = [event for event in events if event["type"] == "initial_data_loaded"]
            self.assertEqual(loaded[0]["skipped_existing"], 1)

    def test_empty_run_uses_latin_hypercube_transformed_through_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(__file__).resolve().parents[1] / "examples" / "bounded_decay.yaml")
            config.run.output_dir = Path(tmp) / "run"
            coordinator = Coordinator(config, FakeComm())

            theta = coordinator._draw_initial_theta(12)
            bounds = coordinator.posterior.bounds.detach().cpu().numpy()
            unit = (theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

            for column in range(unit.shape[1]):
                strata = np.floor(unit[:, column] * len(unit)).astype(int)
                self.assertTrue(np.array_equal(np.sort(strata), np.arange(len(unit))))

    def test_initial_design_below_training_minimum_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", initial_data=False)
            coordinator = Coordinator(config, FakeComm())
            with self.assertRaisesRegex(ValueError, "cannot bootstrap surrogate"):
                coordinator._ensure_initial_design()

    def test_missing_prior_transform_fails_with_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", initial_data=False)
            coordinator = Coordinator(config, FakeComm())
            coordinator.posterior = LegacyPosteriorWithoutPriorTransform()
            with self.assertRaisesRegex(TypeError, "prior_transform"):
                coordinator._draw_initial_theta(2)

    def test_failed_training_obeys_retry_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", retry_seconds=60.0)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            with patch.object(coordinator.trainer, "train", side_effect=RuntimeError("training failed")) as train:
                coordinator._maybe_train_surrogate()
                coordinator._maybe_train_surrogate()
                self.assertEqual(train.call_count, 1)
            failures = [
                event for event in coordinator.store.read_events() if event["type"] == "surrogate_train_failed"
            ]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["retry_after_seconds"], 60.0)

    def test_corrupt_latest_checkpoint_falls_back_to_previous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            data = coordinator.store.load_completed()
            trained = coordinator.trainer.train(data.theta, data.outputs, version=1)
            trained.model.save(coordinator.store.checkpoints_dir / "surrogate_v0001.pt")
            (coordinator.store.checkpoints_dir / "surrogate_v0002.pt").write_bytes(b"not a checkpoint")
            coordinator.store.append_event(
                "job_queued",
                {
                    "job_id": "job_00000099",
                    "job_sequence": 99,
                    "theta": [2.0, 2.0],
                    "attempt": 0,
                    "source": "surrogate_v2",
                },
            )

            restored = Coordinator(config, FakeComm())

            self.assertEqual(restored.surrogate_version, 1)
            self.assertEqual(restored.completed_at_last_train, 2)
            self.assertEqual(restored.queue.pending_count, 0)
            failures = [
                event
                for event in restored.store.read_events()
                if event["type"] == "surrogate_checkpoint_load_failed"
            ]
            self.assertEqual(len(failures), 1)
            cancelled = [
                event
                for event in restored.store.read_events()
                if event["type"] == "job_cancelled" and event["job_id"] == "job_00000099"
            ]
            self.assertEqual(len(cancelled), 1)

    def test_worker_turns_non_finite_solver_output_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", initial_data=False)
            result = _evaluate_worker_job(
                config,
                NonFiniteSolver(),
                config.run.output_dir,
                1,
                {"job_id": "job_00000000", "job_sequence": 0, "theta": np.array([0.0, 0.0])},
            )
            self.assertFalse(result["ok"])
            self.assertIn("non-finite", result["error"])


if __name__ == "__main__":
    unittest.main()
