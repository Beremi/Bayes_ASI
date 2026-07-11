from __future__ import annotations

import numpy as np


def latin_hypercube(
    rng: np.random.Generator,
    n_points: int,
    dimension: int,
) -> np.ndarray:
    """Draw a randomized Latin hypercube in the open unit hypercube."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    jitter = rng.random((n_points, dimension))
    samples = np.empty((n_points, dimension), dtype=np.float64)
    for column in range(dimension):
        strata = rng.permutation(n_points)
        samples[:, column] = (strata + jitter[:, column]) / n_points

    lower = np.finfo(np.float64).eps
    upper = 1.0 - lower
    return np.clip(samples, lower, upper)
