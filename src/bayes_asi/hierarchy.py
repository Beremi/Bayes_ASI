from __future__ import annotations

import numpy as np

from bayes_asi.active import ActiveSelector
from bayes_asi.storage import stable_theta_key


def draw_distillation_coordinates(
    rng: np.random.Generator,
    completed_z: np.ndarray,
    count: int,
) -> np.ndarray:
    """Draw a bounded, reproducible mixture of prior and local latent points."""

    completed = np.asarray(completed_z, dtype=float)
    if completed.ndim != 2 or completed.shape[1] == 0:
        raise ValueError("completed_z must be a non-empty-width 2D array")
    if count <= 0:
        raise ValueError("distillation point count must be positive")
    dim = completed.shape[1]
    prior_count = count if completed.shape[0] == 0 else (count + 1) // 2
    points = [rng.normal(size=(prior_count, dim))]
    local_count = count - prior_count
    if local_count:
        centers = completed[rng.integers(0, completed.shape[0], size=local_count)]
        points.append(centers + 0.5 * rng.normal(size=(local_count, dim)))
    # Avoid pathological tails destabilizing a small student while retaining
    # essentially all standard-normal prior mass.
    return np.clip(np.vstack(points), -6.0, 6.0).astype(np.float32, copy=False)


def prescreen_candidates(
    *,
    selector: ActiveSelector,
    rng: np.random.Generator,
    candidate_z: np.ndarray,
    candidate_theta: np.ndarray,
    candidate_log_prob: np.ndarray,
    existing_z: np.ndarray,
    queued_z: np.ndarray,
    known_keys: set[tuple[float, ...]],
    screen_size: int,
    exploration_fraction: float,
) -> np.ndarray:
    """Return diverse unique indices, reserving a random exploration share."""

    z = np.asarray(candidate_z, dtype=float)
    theta = np.asarray(candidate_theta, dtype=float)
    log_prob = np.asarray(candidate_log_prob, dtype=float).reshape(-1)
    if z.ndim != 2 or theta.ndim != 2 or z.shape[0] != theta.shape[0] or z.shape[0] != log_prob.size:
        raise ValueError("candidate arrays must be aligned 2D z/theta and 1D log probability")
    if screen_size <= 0:
        return np.empty((0,), dtype=int)
    decimals = selector.config.duplicate_round_decimals
    unique_indices: list[int] = []
    seen = set(known_keys)
    for index, row in enumerate(theta):
        key = stable_theta_key(row, decimals)
        if key not in seen and np.all(np.isfinite(z[index])) and np.all(np.isfinite(row)):
            unique_indices.append(index)
            seen.add(key)
    if not unique_indices:
        return np.empty((0,), dtype=int)
    unique = np.asarray(unique_indices, dtype=int)
    target = min(screen_size, unique.size)
    explore_count = min(target, int(np.ceil(target * exploration_fraction)))
    exploit_count = target - explore_count
    selected_indices: list[int] = []
    if exploit_count:
        selected = selector.select(
            candidate_z=z[unique],
            candidate_theta=theta[unique],
            candidate_log_prob=log_prob[unique],
            existing_z=existing_z,
            queued_z=queued_z,
            count=exploit_count,
            known_keys=known_keys,
        )
        key_to_index = {stable_theta_key(theta[index], decimals): index for index in unique}
        selected_indices.extend(key_to_index[stable_theta_key(row, decimals)] for row in selected.theta)
    selected_set = set(selected_indices)
    remaining = np.asarray([index for index in unique if int(index) not in selected_set], dtype=int)
    random_count = min(explore_count, remaining.size)
    if random_count:
        selected_indices.extend(int(index) for index in rng.choice(remaining, size=random_count, replace=False))
    if len(selected_indices) < target:
        selected_set = set(selected_indices)
        selected_indices.extend(int(index) for index in unique if int(index) not in selected_set)
    return np.asarray(selected_indices[:target], dtype=int)
