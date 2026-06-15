from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Job:
    job_id: str
    sequence: int
    theta: np.ndarray
    attempt: int = 0
    source: str = "active"
    queued_at: float = 0.0


@dataclass(slots=True)
class QueueState:
    pending: deque[Job] = field(default_factory=deque)
    running: dict[int, Job] = field(default_factory=dict)
    solve_time_ewma: float | None = None

    def add_pending(self, job: Job) -> None:
        self.pending.append(job)

    def pop_for_worker(self, worker_rank: int) -> Job | None:
        if not self.pending:
            return None
        job = self.pending.popleft()
        self.running[worker_rank] = job
        return job

    def finish_worker(self, worker_rank: int) -> Job | None:
        return self.running.pop(worker_rank, None)

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def running_count(self) -> int:
        return len(self.running)

    @property
    def in_system_count(self) -> int:
        return self.pending_count + self.running_count

    def pending_theta(self) -> np.ndarray:
        rows = [job.theta for job in self.pending]
        return np.vstack(rows) if rows else np.empty((0, 0))

    def running_theta(self) -> np.ndarray:
        rows = [job.theta for job in self.running.values()]
        return np.vstack(rows) if rows else np.empty((0, 0))

    def update_solve_time(self, elapsed_s: float, alpha: float = 0.2) -> None:
        elapsed_s = float(max(elapsed_s, 1e-9))
        if self.solve_time_ewma is None:
            self.solve_time_ewma = elapsed_s
        else:
            self.solve_time_ewma = alpha * elapsed_s + (1.0 - alpha) * self.solve_time_ewma

