from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayes_asi.config import ActiveLearningConfig
from bayes_asi.storage import stable_theta_key


@dataclass(slots=True)
class SelectedBatch:
    z: np.ndarray
    theta: np.ndarray
    scores: np.ndarray
    min_distances: np.ndarray


class ActiveSelector:
    """Greedy posterior-plus-distance candidate selector."""

    def __init__(self, config: ActiveLearningConfig) -> None:
        self.config = config

    def select(
        self,
        *,
        candidate_z: np.ndarray,
        candidate_theta: np.ndarray,
        candidate_log_prob: np.ndarray,
        existing_z: np.ndarray,
        queued_z: np.ndarray | None = None,
        count: int,
        known_keys: set[tuple[float, ...]] | None = None,
    ) -> SelectedBatch:
        candidate_z = np.asarray(candidate_z, dtype=float)
        candidate_theta = np.asarray(candidate_theta, dtype=float)
        candidate_log_prob = np.asarray(candidate_log_prob, dtype=float).reshape(-1)
        if candidate_z.ndim != 2 or candidate_theta.ndim != 2:
            raise ValueError("candidate_z and candidate_theta must be 2D arrays")
        if candidate_z.shape[0] != candidate_theta.shape[0] or candidate_z.shape[0] != candidate_log_prob.shape[0]:
            raise ValueError("candidate z/theta/log probability length mismatch")
        if not np.all(np.isfinite(candidate_z)) or not np.all(np.isfinite(candidate_theta)):
            raise ValueError("candidate_z and candidate_theta must contain only finite values")
        if candidate_z.shape[0] == 0 or count <= 0:
            return SelectedBatch(
                z=np.empty((0, candidate_z.shape[1] if candidate_z.ndim == 2 else 0)),
                theta=np.empty((0, candidate_theta.shape[1] if candidate_theta.ndim == 2 else 0)),
                scores=np.empty((0,)),
                min_distances=np.empty((0,)),
            )

        known_keys = set() if known_keys is None else set(known_keys)
        existing = np.asarray(existing_z, dtype=float).reshape(-1, candidate_z.shape[1])
        queued = (
            np.empty((0, candidate_z.shape[1]), dtype=float)
            if queued_z is None
            else np.asarray(queued_z, dtype=float).reshape(-1, candidate_z.shape[1])
        )
        if not np.all(np.isfinite(existing)) or not np.all(np.isfinite(queued)):
            raise ValueError("existing_z and queued_z must contain only finite values")
        posterior_scale = np.std(candidate_z, axis=0)
        posterior_scale[posterior_scale < 1e-9] = 1.0
        logp_z = self._zscore(candidate_log_prob)
        completed_distances = self._min_distances(candidate_z, existing, posterior_scale)
        if queued.size:
            queued_distances = self._min_distances(candidate_z, queued, posterior_scale)
            distances = np.minimum(
                completed_distances,
                queued_distances / self.config.queued_repulsion_radius_multiplier,
            )
        else:
            distances = completed_distances
        available = np.ones(candidate_z.shape[0], dtype=bool)
        selected_z: list[np.ndarray] = []
        selected_rows: list[np.ndarray] = []
        selected_scores: list[float] = []
        selected_distances: list[float] = []

        for _ in range(count):
            valid_indices = np.where(available)[0]
            if valid_indices.size == 0:
                break
            distances_z = self._zscore(distances)
            scores = np.zeros(candidate_z.shape[0], dtype=float)
            if self.config.posterior_weight > 0.0:
                scores += self.config.posterior_weight * logp_z
            if self.config.distance_weight > 0.0:
                scores += self.config.distance_weight * distances_z
            scores[~available] = -np.inf
            for idx in valid_indices:
                if stable_theta_key(candidate_theta[idx], self.config.duplicate_round_decimals) in known_keys:
                    scores[idx] = -np.inf
            best_index = int(np.argmax(scores))
            if not np.isfinite(scores[best_index]):
                break
            z = candidate_z[best_index].copy()
            theta = candidate_theta[best_index].copy()
            key = stable_theta_key(theta, self.config.duplicate_round_decimals)
            selected_z.append(z)
            selected_rows.append(theta)
            selected_scores.append(float(scores[best_index]))
            selected_distances.append(float(distances[best_index]))
            known_keys.add(key)
            available[best_index] = False
            new_distances = np.linalg.norm((candidate_z - z) / posterior_scale, axis=1)
            distances = np.minimum(
                distances,
                new_distances / self.config.queued_repulsion_radius_multiplier,
            )

        if not selected_rows:
            return SelectedBatch(
                z=np.empty((0, candidate_z.shape[1])),
                theta=np.empty((0, candidate_theta.shape[1])),
                scores=np.empty((0,)),
                min_distances=np.empty((0,)),
            )
        return SelectedBatch(
            z=np.vstack(selected_z),
            theta=np.vstack(selected_rows),
            scores=np.asarray(selected_scores),
            min_distances=np.asarray(selected_distances),
        )

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        z = np.full_like(values, -np.inf)
        if not np.any(finite):
            return z
        mean = float(np.mean(values[finite]))
        std = float(np.std(values[finite]))
        if std < 1e-12:
            z[finite] = 0.0
        else:
            z[finite] = (values[finite] - mean) / std
        return z

    @staticmethod
    def _min_distances(candidate_z: np.ndarray, existing_z: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if existing_z.size == 0:
            return np.ones(candidate_z.shape[0], dtype=float)
        result = np.full(candidate_z.shape[0], np.inf, dtype=float)
        max_pairs_per_chunk = 1_000_000
        reference_chunk = max(1, max_pairs_per_chunk // max(candidate_z.shape[0], 1))
        scale_3d = scale.reshape(1, 1, -1)
        for start in range(0, existing_z.shape[0], reference_chunk):
            reference = existing_z[start : start + reference_chunk]
            diff = (candidate_z[:, None, :] - reference[None, :, :]) / scale_3d
            result = np.minimum(result, np.min(np.linalg.norm(diff, axis=-1), axis=1))
        return result
