from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from scipy import special

from bayes_asi.config import load_config
from bayes_asi.hmc import HMCResult
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
        "auxiliary": {"experiments/source_a/raw": np.arange(12, dtype=np.float32).reshape(3, 2, 2)},
        "metadata": {"source": "unit-test"},
    }


class NonFiniteSolver:
    def evaluate(self, theta, context):
        del theta, context
        return ForwardResult(outputs=np.array([np.nan, 1.0]))


class LegacyPosteriorWithoutPriorTransform:
    pass


def make_legacy_posterior(config):
    del config
    return LegacyPosteriorWithoutPriorTransform()


class RuntimeTests(unittest.TestCase):
    def _config(
        self,
        run_dir: Path,
        *,
        initial_data: bool = True,
        retry_seconds: float = 60.0,
        posterior_factory: str = "bayes_asi.examples.synthetic:make_posterior",
        hierarchy: bool = False,
    ):
        initial_section = (
            f"initial_data:\n  factory: {__name__}:make_initial_data\n"
            if initial_data
            else ""
        )
        path = run_dir.parent / f"config_{run_dir.name}.yaml"
        hierarchy_section = ""
        if hierarchy:
            hierarchy_section = """
hierarchy:
  enabled: true
  distillation_points: 6
  teacher_screen_size: 4
  prediction_batch_size: 2
  exploration_fraction: 0.25
  student:
    hidden_layers: [3]
    max_epochs: 2
    min_train_samples: 2
    validation_fraction: 0.0
active_learning:
  candidate_pool_size: 16
  batch_max_size: 2
"""
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
  factory: {posterior_factory}
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
{hierarchy_section}
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
            auxiliary = coordinator.store.load_auxiliary("experiments/source_a/raw")
            self.assertEqual(auxiliary.values.shape, (2, 2, 2))
            self.assertTrue(np.array_equal(auxiliary.present, [True, True]))
            events = coordinator.store.read_events()
            loaded = [event for event in events if event["type"] == "initial_data_loaded"]
            self.assertEqual(loaded[0]["skipped_existing"], 1)

    def test_empty_run_uses_latin_hypercube_transformed_through_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(__file__).resolve().parents[1] / "examples" / "bounded_decay.yaml")
            config.run.output_dir = Path(tmp) / "run"
            coordinator = Coordinator(config, FakeComm())

            z, theta = coordinator._draw_initial_points(12)
            bounds = coordinator.posterior.bounds.detach().cpu().numpy()
            unit = (theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            latent_unit = special.ndtr(z)

            for column in range(unit.shape[1]):
                strata = np.floor(unit[:, column] * len(unit)).astype(int)
                self.assertTrue(np.array_equal(np.sort(strata), np.arange(len(unit))))
                latent_strata = np.floor(latent_unit[:, column] * len(unit)).astype(int)
                self.assertTrue(np.array_equal(np.sort(latent_strata), np.arange(len(unit))))
            self.assertTrue(np.allclose(unit, latent_unit, atol=2e-7))

    def test_initial_design_below_training_minimum_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", initial_data=False)
            coordinator = Coordinator(config, FakeComm())
            with self.assertRaisesRegex(ValueError, "cannot bootstrap surrogate"):
                coordinator._ensure_initial_design()

    def test_missing_reversible_prior_fails_with_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(
                Path(tmp) / "run",
                initial_data=False,
                posterior_factory=f"{__name__}:make_legacy_posterior",
            )
            with self.assertRaisesRegex(TypeError, "reversible prior"):
                Coordinator(config, FakeComm())

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
            trained = coordinator.trainer.train(data.z, data.outputs, version=1)
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

    def test_schema_one_samples_are_migrated_through_the_configured_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", initial_data=False)
            config.run.output_dir.mkdir(parents=True)
            with h5py.File(config.run.output_dir / "data.h5", "w") as h5:
                h5.attrs.update(schema_version=1, parameter_dim=2, output_dim=2)
                group = h5.create_group("completed")
                arrays = {
                    "theta": np.array([[2.0, -4.0], [-1.0, 3.0]]),
                    "outputs": np.array([[1.0, 2.0], [3.0, 4.0]]),
                    "elapsed_s": np.array([0.1, 0.2]),
                    "worker_rank": np.array([1, 2]),
                    "job_sequence": np.array([0, 1]),
                    "started_at": np.array([1.0, 2.0]),
                    "finished_at": np.array([1.1, 2.2]),
                }
                for name, values in arrays.items():
                    group.create_dataset(name, data=values)

            coordinator = Coordinator(config, FakeComm())
            completed = coordinator.store.load_completed()

            self.assertTrue(np.allclose(completed.z, completed.theta / 2.0, atol=1e-12))
            migrated = [
                event for event in coordinator.store.read_events() if event["type"] == "storage_schema_migrated"
            ]
            self.assertEqual(migrated[0]["rows"], 2)

    def test_restart_rejects_a_prior_inconsistent_with_stored_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            config.posterior.kwargs["prior_sd"] = 3.0

            with self.assertRaisesRegex(ValueError, "inconsistent with the configured prior"):
                Coordinator(config, FakeComm())

    def test_physical_input_checkpoint_is_not_reused_for_standard_normal_surrogate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            data = coordinator.store.load_completed()
            trained = coordinator.trainer.train(data.z, data.outputs, version=1)
            trained.model.input_space = "physical_parameters"
            trained.model.save(coordinator.store.checkpoints_dir / "surrogate_v0001.pt")
            coordinator.store.append_event(
                "job_queued",
                {
                    "job_id": "job_00000099",
                    "job_sequence": 99,
                    "z": [1.5, 1.5],
                    "theta": [3.0, 3.0],
                    "attempt": 0,
                    "source": "surrogate_v1",
                },
            )

            restored = Coordinator(config, FakeComm())

            self.assertIsNone(restored.surrogate)
            self.assertEqual(restored.queue.pending_count, 0)
            failures = [
                event
                for event in restored.store.read_events()
                if event["type"] == "surrogate_checkpoint_load_failed"
            ]
            self.assertIn("physical rather than standard-normal", failures[-1]["error"])
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

    def test_hierarchy_trains_student_without_persisting_pseudo_truth_and_restarts_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", hierarchy=True)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            coordinator._maybe_train_surrogate()

            self.assertIsNotNone(coordinator.surrogate)
            student = coordinator.student
            self.assertIsNotNone(student)
            assert student is not None
            self.assertEqual(coordinator.surrogate_version, student.version)
            self.assertEqual(coordinator.store.completed_count(), 2)
            self.assertTrue((coordinator.store.checkpoints_dir / "surrogate_v0001.pt").exists())
            self.assertTrue((coordinator.store.checkpoints_dir / "student_v0001.pt").exists())
            student_events = [
                event for event in coordinator.store.read_events() if event["type"] == "hierarchy_student_trained"
            ]
            self.assertEqual(student_events[-1]["genuine_completed_count"], 2)
            self.assertEqual(student_events[-1]["pseudo_training_count"], 8)

            restored = Coordinator(config, FakeComm())
            self.assertEqual(restored.surrogate_version, 1)
            restored_student = restored.student
            self.assertIsNotNone(restored_student)
            assert restored_student is not None
            self.assertEqual(restored_student.version, 1)
            self.assertEqual(restored.store.completed_count(), 2)
            self.assertTrue(
                any(event["type"] == "hierarchy_student_loaded" for event in restored.store.read_events())
            )

    def test_missing_student_checkpoint_is_regenerated_for_loaded_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", hierarchy=True)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            coordinator._maybe_train_surrogate()
            (coordinator.store.checkpoints_dir / "student_v0001.pt").unlink()

            restored = Coordinator(config, FakeComm())

            self.assertIsNone(restored.student)
            self.assertTrue(restored.student_regeneration_pending)
            events = restored.store.read_events()
            self.assertTrue(any(event["type"] == "hierarchy_student_regeneration_requested" for event in events))
            self.assertFalse(
                any(
                    event["type"] == "hierarchy_student_trained"
                    and event["reason"] == "restart_regeneration"
                    for event in events
                )
            )

            restored._maybe_regenerate_student(time.monotonic() + 10.0)

            restored_student = restored.student
            self.assertIsNotNone(restored_student)
            assert restored_student is not None
            self.assertEqual(restored_student.version, restored.surrogate_version)
            events = restored.store.read_events()
            self.assertTrue(
                any(
                    event["type"] == "hierarchy_student_trained"
                    and event["reason"] == "restart_regeneration"
                    for event in events
                )
            )

    def test_hierarchy_uses_student_for_hmc_and_caps_teacher_screening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", hierarchy=True)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            coordinator._maybe_train_surrogate()
            teacher = coordinator.surrogate
            self.assertIsNotNone(teacher)
            assert teacher is not None
            z = np.array(
                [[-1.5, -1.0], [-1.0, 0.5], [-0.2, 1.4], [0.4, -1.3], [1.0, 0.8], [1.6, 1.2]],
                dtype=float,
            )
            theta = 2.0 * z
            result = HMCResult(
                z=z,
                theta=theta,
                log_prob=np.linspace(-3.0, 2.0, len(z)),
                accept_rate=0.8,
                warmup_accept_rate=0.0,
                step_size=0.05,
                leapfrog_steps=2,
                mass_diag=np.ones(2),
            )
            with patch("bayes_asi.runtime.HMCSampler") as sampler_type:
                sampler_type.return_value.sample.return_value = result
                coordinator._draw_surrogate_posterior()
                self.assertIs(sampler_type.call_args.args[2], coordinator.student)
                self.assertEqual(sampler_type.call_args.kwargs["device"], coordinator._hmc_surrogate().device)

            teacher_predict_original = teacher.predict_numpy

            def teacher_predict_with_nonfinite(points):
                values = teacher_predict_original(points)
                values[0] = np.nan
                return values

            with (
                patch.object(coordinator, "_draw_surrogate_posterior", return_value=result),
                patch.object(
                    teacher,
                    "predict_numpy",
                    side_effect=teacher_predict_with_nonfinite,
                ) as teacher_predict,
            ):
                coordinator._maybe_refill_active_queue()

            screened = sum(call.args[0].shape[0] for call in teacher_predict.call_args_list)
            self.assertLessEqual(screened, 4)
            self.assertLessEqual(coordinator.queue.pending_count, 2)
            self.assertEqual(coordinator.store.completed_count(), 2)
            selected = [
                event for event in coordinator.store.read_events() if event["type"] == "active_batch_selected"
            ][-1]
            self.assertEqual(selected["hmc_surrogate_role"], "student")
            self.assertLessEqual(selected["teacher_screen_evaluated"], 4)
            self.assertLess(selected["teacher_screen_finite"], selected["teacher_screen_evaluated"])
            self.assertEqual(selected["selection_mode"], "approximate_student_teacher")

    def test_student_training_failure_falls_back_explicitly_to_teacher_hmc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", hierarchy=True)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            with patch.object(coordinator.student_trainer, "train", side_effect=RuntimeError("student failed")):
                coordinator._maybe_train_surrogate()

            self.assertIsNotNone(coordinator.surrogate)
            self.assertIsNone(coordinator.student)
            self.assertEqual(coordinator._hmc_surrogate_role(), "teacher")
            failures = [
                event
                for event in coordinator.store.read_events()
                if event["type"] == "hierarchy_student_train_failed"
            ]
            self.assertEqual(failures[-1]["fallback"], "teacher_for_hmc")

    def test_replaced_same_version_teacher_does_not_load_stale_student(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run", hierarchy=True)
            coordinator = Coordinator(config, FakeComm())
            coordinator._import_initial_data()
            coordinator._maybe_train_surrogate()
            completed = coordinator.store.load_completed()
            replacement = coordinator.trainer.train(
                completed.z,
                completed.outputs + 0.5,
                version=1,
                force_scratch=True,
            )
            replacement.model.save(coordinator.store.checkpoints_dir / "surrogate_v0001.pt")

            restored = Coordinator(config, FakeComm())

            self.assertEqual(restored.surrogate_version, 1)
            self.assertIsNone(restored.student)
            self.assertTrue(restored.student_regeneration_pending)
            pair_failures = [
                event
                for event in restored.store.read_events()
                if event["type"] == "hierarchy_student_checkpoint_load_failed"
            ]
            self.assertIn("fingerprint", pair_failures[-1]["error"])


if __name__ == "__main__":
    unittest.main()
