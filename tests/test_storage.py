from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bayes_asi.queue import Job, QueueState
from bayes_asi.storage import HDF5Store, replay_queue_events, stable_theta_key


class StorageTests(unittest.TestCase):
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

    def test_status_is_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HDF5Store(Path(tmp), parameter_dim=1, output_dim=1)
            store.update_status({"completed_count": 3})
            text = (Path(tmp) / "status.json").read_text(encoding="utf-8")
            self.assertIn('"completed_count": 3', text)


if __name__ == "__main__":
    unittest.main()
