from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayes_asi.config import ActiveLearningConfig
from bayes_asi.storage import stable_theta_key


@dataclass(slots=True)
class SelectedBatch:
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
        candidate_theta: np.ndarray,
        candidate_log_prob: np.ndarray,
        existing_theta: np.ndarray,
        queued_theta: np.ndarray | None = None,
        count: int,
        known_keys: set[tuple[float, ...]] | None = None,
    ) -> SelectedBatch:
        candidate_theta = np.asarray(candidate_theta, dtype=float)
        candidate_log_prob = np.asarray(candidate_log_prob, dtype=float).reshape(-1)
        if candidate_theta.ndim != 2:
            raise ValueError("candidate_theta must be a 2D array")
        if candidate_theta.shape[0] != candidate_log_prob.shape[0]:
            raise ValueError("candidate theta/log probability length mismatch")
        if candidate_theta.shape[0] == 0 or count <= 0:
            return SelectedBatch(
                theta=np.empty((0, candidate_theta.shape[1] if candidate_theta.ndim == 2 else 0)),
                scores=np.empty((0,)),
                min_distances=np.empty((0,)),
            )

        known_keys = set() if known_keys is None else set(known_keys)
        existing = np.asarray(existing_theta, dtype=float).reshape(-1, candidate_theta.shape[1])
        queued = (
            np.empty((0, candidate_theta.shape[1]), dtype=float)
            if queued_theta is None
            else np.asarray(queued_theta, dtype=float).reshape(-1, candidate_theta.shape[1])
        )
        posterior_scale = np.std(candidate_theta, axis=0)
        posterior_scale[posterior_scale < 1e-9] = 1.0
        logp_z = self._zscore(candidate_log_prob)
        available = np.ones(candidate_theta.shape[0], dtype=bool)
        selected_rows: list[np.ndarray] = []
        selected_scores: list[float] = []
        selected_distances: list[float] = []

        for _ in range(count):
            valid_indices = np.where(available)[0]
            if valid_indices.size == 0:
                break
            distances = self._effective_min_distances(candidate_theta, existing, queued, posterior_scale)
            distances_z = self._zscore(distances)
            scores = self.config.posterior_weight * logp_z + self.config.distance_weight * distances_z
            scores[~available] = -np.inf
            for idx in valid_indices:
                if stable_theta_key(candidate_theta[idx], self.config.duplicate_round_decimals) in known_keys:
                    scores[idx] = -np.inf
            best_index = int(np.argmax(scores))
            if not np.isfinite(scores[best_index]):
                break
            theta = candidate_theta[best_index].copy()
            key = stable_theta_key(theta, self.config.duplicate_round_decimals)
            selected_rows.append(theta)
            selected_scores.append(float(scores[best_index]))
            selected_distances.append(float(distances[best_index]))
            known_keys.add(key)
            available[best_index] = False
            queued = np.vstack([queued, theta.reshape(1, -1)])

        if not selected_rows:
            return SelectedBatch(
                theta=np.empty((0, candidate_theta.shape[1])),
                scores=np.empty((0,)),
                min_distances=np.empty((0,)),
            )
        return SelectedBatch(
            theta=np.vstack(selected_rows),
            scores=np.asarray(selected_scores),
            min_distances=np.asarray(selected_distances),
        )

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(values)
        z = (values - mean) / std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _min_distances(candidate_theta: np.ndarray, existing_theta: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if existing_theta.size == 0:
            return np.ones(candidate_theta.shape[0], dtype=float)
        return np.min(ActiveSelector._pairwise_distances(candidate_theta, existing_theta, scale), axis=1)

    def _effective_min_distances(
        self,
        candidate_theta: np.ndarray,
        existing_theta: np.ndarray,
        queued_theta: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        if existing_theta.size:
            parts.append(np.min(self._pairwise_distances(candidate_theta, existing_theta, scale), axis=1))
        if queued_theta.size:
            queued_distances = np.min(self._pairwise_distances(candidate_theta, queued_theta, scale), axis=1)
            parts.append(queued_distances / self.config.queued_repulsion_radius_multiplier)
        if not parts:
            return np.ones(candidate_theta.shape[0], dtype=float)
        return np.min(np.vstack(parts), axis=0)

    @staticmethod
    def _pairwise_distances(candidate_theta: np.ndarray, reference_theta: np.ndarray, scale: np.ndarray) -> np.ndarray:
        diff = (candidate_theta[:, None, :] - reference_theta[None, :, :]) / scale.reshape(1, 1, -1)
        return np.linalg.norm(diff, axis=-1)
