from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from bayes_asi.queue import Job, QueueState
from bayes_asi.storage import SCHEMA_VERSION, HDF5Store, exclusive_run_lock, replay_queue_events, stable_theta_key


class StorageTests(unittest.TestCase):
    def test_batch_append_and_count_only_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=2, output_dim=1)
            start = store.append_completed_batch(
                z=np.array([[-1.0, -2.0], [1.0, 2.0]]),
                theta=np.array([[1.0, 2.0], [3.0, 4.0]]),
                outputs=np.array([[5.0], [6.0]]),
                elapsed_s=np.array([0.1, 0.2]),
                worker_rank=np.array([1, 2]),
                job_sequence=np.array([10, 11]),
                started_at=np.array([1.0, 2.0]),
                finished_at=np.array([1.1, 2.2]),
            )
            self.assertEqual(start, 0)
            self.assertEqual(store.completed_count(), 2)
            self.assertTrue(np.array_equal(store.load_completed().job_sequence, [10, 11]))

    def test_append_completed_and_replay_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=2, output_dim=1)
            store.append_event(
                "job_queued",
                {"job_id": "job_00000000", "job_sequence": 0, "theta": [1.0, 2.0], "attempt": 0},
            )
            store.append_event(
                "job_queued",
                {"job_id": "job_00000001", "job_sequence": 1, "theta": [3.0, 4.0], "attempt": 0},
            )
            store.append_completed(
                z=np.array([-1.0, 1.0]),
                theta=np.array([1.0, 2.0]),
                outputs=np.array([5.0]),
                elapsed_s=0.2,
                worker_rank=1,
                job_sequence=0,
                started_at=10.0,
                finished_at=10.2,
            )
            store.append_event("job_completed", {"job_id": "job_00000000", "job_sequence": 0})
            data = store.load_completed()
            self.assertEqual(data.n, 1)
            self.assertTrue(np.allclose(data.theta[0], [1.0, 2.0]))
            self.assertTrue(np.allclose(data.outputs[0], [5.0]))

            pending = replay_queue_events(
                store.read_events(),
                {stable_theta_key(np.array([1.0, 2.0]))},
            )
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["job_id"], "job_00000001")

    def test_auxiliary_fields_are_columnar_optional_and_dynamic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1, latent_dim=1)
            store.append_completed_batch(
                z=np.array([[-1.0], [1.0]]),
                theta=np.array([[2.0], [3.0]]),
                outputs=np.array([[4.0], [5.0]]),
                elapsed_s=np.array([0.1, 0.2]),
                worker_rank=np.array([1, 2]),
                job_sequence=np.array([0, 1]),
                started_at=np.array([1.0, 2.0]),
                finished_at=np.array([1.1, 2.2]),
                auxiliary=[{"experiments/source_a/curve": np.arange(5, dtype=np.float32)}, {}],
            )
            store.append_completed(
                z=np.array([0.5]),
                theta=np.array([2.5]),
                outputs=np.array([4.5]),
                elapsed_s=0.3,
                worker_rank=1,
                job_sequence=2,
                started_at=3.0,
                finished_at=3.3,
                auxiliary={
                    "experiments/source_a/curve": np.arange(5, dtype=np.float32) + 10.0,
                    "diagnostics/iterations": np.array(17, dtype=np.int32),
                },
            )

            self.assertEqual(
                store.list_auxiliary_fields(),
                ["diagnostics/iterations", "experiments/source_a/curve"],
            )
            curve = store.load_auxiliary("experiments/source_a/curve")
            diagnostic = store.load_auxiliary("diagnostics/iterations")
            self.assertEqual(curve.values.shape, (3, 5))
            self.assertTrue(np.array_equal(curve.present, [True, False, True]))
            self.assertTrue(np.array_equal(curve.values[2], np.arange(5) + 10.0))
            self.assertTrue(np.array_equal(diagnostic.present, [False, False, True]))
            self.assertEqual(int(diagnostic.values[2]), 17)
            with h5py.File(store.data_path, "r") as h5:
                identifiers = list(h5["auxiliary"])
                datasets = [h5[f"auxiliary/{identifier}/values"] for identifier in identifiers]
                self.assertTrue(all(dataset.compression == "lzf" for dataset in datasets))
                self.assertTrue(all(dataset.fletcher32 for dataset in datasets))

    def test_auxiliary_schema_mismatch_rolls_back_every_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            store.append_completed(
                z=np.array([0.0]),
                theta=np.array([1.0]),
                outputs=np.array([2.0]),
                elapsed_s=0.1,
                worker_rank=1,
                job_sequence=0,
                started_at=1.0,
                finished_at=1.1,
                auxiliary={"curve": np.ones(2, dtype=np.float32)},
            )
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                store.append_completed(
                    z=np.array([0.0]),
                    theta=np.array([1.0]),
                    outputs=np.array([2.0]),
                    elapsed_s=0.1,
                    worker_rank=1,
                    job_sequence=1,
                    started_at=1.0,
                    finished_at=1.1,
                    auxiliary={"curve": np.ones(3, dtype=np.float32)},
                )

            self.assertEqual(store.completed_count(), 1)
            curve = store.load_auxiliary("curve")
            self.assertEqual(curve.values.shape, (1, 2))
            self.assertTrue(np.array_equal(curve.present, [True]))

    def test_auxiliary_rejects_object_arrays_without_modifying_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            with self.assertRaisesRegex(TypeError, "fixed-size"):
                store.append_completed(
                    z=np.array([0.0]),
                    theta=np.array([1.0]),
                    outputs=np.array([2.0]),
                    elapsed_s=0.1,
                    worker_rank=1,
                    job_sequence=0,
                    started_at=1.0,
                    finished_at=1.1,
                    auxiliary={"bad": np.array([{"value": 1}], dtype=object)},
                )
            self.assertEqual(store.completed_count(), 0)

    def test_schema_one_data_can_be_migrated_to_standard_normal_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            with h5py.File(path, "w") as h5:
                h5.attrs["schema_version"] = 1
                h5.attrs["parameter_dim"] = 1
                h5.attrs["output_dim"] = 1
                group = h5.create_group("completed")
                datasets = {
                    "theta": np.array([[2.0], [3.0]]),
                    "outputs": np.array([[4.0], [5.0]]),
                    "elapsed_s": np.array([0.1, 0.2]),
                    "worker_rank": np.array([1, 2]),
                    "job_sequence": np.array([0, 1]),
                    "started_at": np.array([1.0, 2.0]),
                    "finished_at": np.array([1.1, 2.2]),
                }
                for name, values in datasets.items():
                    group.create_dataset(name, data=values)

            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1, latent_dim=1)
            self.assertTrue(store.latent_coordinates_pending())
            with self.assertRaisesRegex(RuntimeError, "migration"):
                store.load_completed()
            store.complete_latent_migration(np.array([[-1.0], [1.0]]))
            self.assertTrue(np.array_equal(store.load_completed().z, [[-1.0], [1.0]]))
            with h5py.File(path, "r") as h5:
                self.assertEqual(int(h5.attrs["schema_version"]), SCHEMA_VERSION)
                self.assertEqual(int(h5.attrs["latent_coordinates_pending"]), 0)

    def test_cancelled_jobs_are_not_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=2, output_dim=1)
            store.append_event(
                "job_queued",
                {
                    "job_id": "job_00000000",
                    "job_sequence": 0,
                    "theta": [1.0, 2.0],
                    "attempt": 0,
                    "source": "surrogate_v1",
                },
            )
            store.append_event(
                "job_cancelled",
                {"job_id": "job_00000000", "job_sequence": 0, "reason": "new_surrogate_trained"},
            )
            pending = replay_queue_events(store.read_events(), set())
            self.assertEqual(pending, [])

    def test_retry_attempt_is_preserved_during_replay(self) -> None:
        events = [
            {
                "type": "job_queued",
                "job_id": "job_00000000",
                "job_sequence": 0,
                "theta": [1.0],
                "attempt": 0,
            },
            {
                "type": "job_failed",
                "job_id": "job_00000000",
                "attempt": 0,
                "will_retry": 1,
            },
        ]
        pending = replay_queue_events(events, set())
        self.assertEqual(pending[0]["attempt"], 1)

    def test_clear_pending_preserves_running_jobs(self) -> None:
        queue = QueueState()
        first = Job(job_id="job_00000000", sequence=0, theta=np.array([1.0, 2.0]))
        second = Job(job_id="job_00000001", sequence=1, theta=np.array([3.0, 4.0]))
        running = Job(job_id="job_00000002", sequence=2, theta=np.array([5.0, 6.0]))
        queue.add_pending(first)
        queue.add_pending(second)
        queue.running[1] = running

        cleared = queue.clear_pending()

        self.assertEqual([job.job_id for job in cleared], ["job_00000000", "job_00000001"])
        self.assertEqual(queue.pending_count, 0)
        self.assertEqual(queue.running_count, 1)
        self.assertIs(queue.running[1], running)

    def test_worker_cannot_receive_two_running_jobs(self) -> None:
        queue = QueueState()
        queue.add_pending(Job(job_id="first", sequence=0, theta=np.array([1.0])))
        queue.add_pending(Job(job_id="second", sequence=1, theta=np.array([2.0])))
        queue.pop_for_worker(1)
        with self.assertRaisesRegex(ValueError, "already has"):
            queue.pop_for_worker(1)

    def test_solve_time_validation(self) -> None:
        queue = QueueState()
        with self.assertRaisesRegex(ValueError, "elapsed_s"):
            queue.update_solve_time(np.nan)
        with self.assertRaisesRegex(ValueError, "alpha"):
            queue.update_solve_time(1.0, alpha=0.0)

    def test_status_is_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            store.update_status({"completed_count": 3})
            text = (Path(tmp) / "status.json").read_text(encoding="utf-8")
            self.assertIn('"completed_count": 3', text)

    def test_existing_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            HDF5Store(Path(tmp), parameter_dim=2, output_dim=1)
            with self.assertRaisesRegex(ValueError, "parameter_dim=2, expected 3"):
                HDF5Store(Path(tmp), parameter_dim=3, output_dim=1)

    def test_inconsistent_dataset_lengths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            with h5py.File(store.data_path, "a") as h5:
                h5["completed/theta"].resize(1, axis=0)
            with self.assertRaisesRegex(ValueError, "inconsistent completed dataset lengths"):
                store.load_completed()

    def test_non_finite_completed_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                store.append_completed(
                    z=np.array([0.0]),
                    theta=np.array([1.0]),
                    outputs=np.array([np.nan]),
                    elapsed_s=0.1,
                    worker_rank=1,
                    job_sequence=0,
                    started_at=1.0,
                    finished_at=1.1,
                )

    def test_truncated_final_event_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            store.append_event("run_started", {"value": 1})
            with store.events_path.open("a", encoding="utf-8") as handle:
                handle.write('{"type": "job_queued"')
            events = store.read_events()
            self.assertEqual([event["type"] for event in events], ["run_started"])

    def test_checkpoint_paths_use_numeric_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            for name in ("surrogate_v2.pt", "surrogate_v10.pt", "surrogate_invalid.pt"):
                (store.checkpoints_dir / name).touch()
            latest = store.latest_checkpoint()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "surrogate_v10.pt")

    def test_event_payload_supports_common_numpy_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            store.append_event(
                "metadata",
                {"scalar": np.int64(3), "array": np.array([1.0, 2.0]), "path": Path("result.out")},
            )
            event = store.read_events()[0]
            self.assertEqual(event["scalar"], 3)
            self.assertEqual(event["array"], [1.0, 2.0])
            self.assertEqual(event["path"], "result.out")

    def test_run_lock_rejects_a_second_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with exclusive_run_lock(run_dir):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with exclusive_run_lock(run_dir):
                        self.fail("second lock unexpectedly acquired")

    def test_config_snapshot_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            source = Path(tmp) / "config.yaml"
            source.write_text("problem: test\n", encoding="utf-8")
            store = HDF5Store(run_dir, parameter_dim=1, output_dim=1)
            first, digest = store.snapshot_config(source)
            second, second_digest = store.snapshot_config(source)
            self.assertEqual(first, second)
            self.assertEqual(digest, second_digest)
            self.assertEqual(first.read_text(encoding="utf-8"), "problem: test\n")

            source.write_text("problem: changed\n", encoding="utf-8")
            changed, changed_digest = store.snapshot_config(source)
            self.assertNotEqual(changed, first)
            self.assertNotEqual(changed_digest, digest)

            source.write_text("problem: test\n", encoding="utf-8")
            first.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                store.snapshot_config(source)


if __name__ == "__main__":
    unittest.main()
