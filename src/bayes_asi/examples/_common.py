from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bayes_asi.config import AppConfig


def require_dimensions(
    config: AppConfig,
    *,
    latent_dim: int,
    parameter_dim: int,
    output_dim: int,
) -> None:
    expected = (latent_dim, parameter_dim, output_dim)
    actual = (
        config.problem.latent_dim,
        config.problem.parameter_dim,
        config.problem.output_dim,
    )
    if actual != expected:
        raise ValueError(
            "example requires problem dimensions "
            f"latent={latent_dim}, parameter={parameter_dim}, output={output_dim}; got {actual}"
        )


def float_vector(values: Sequence[float], *, size: int, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return array
