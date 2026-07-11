from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike

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


def float_vector(values: ArrayLike, *, size: int, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return array


def unit_matrix(values: ArrayLike, *, size: int) -> np.ndarray:
    unit = np.asarray(values, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != size:
        raise ValueError(f"unit must have shape (N, {size})")
    if not np.all(np.isfinite(unit)) or np.any(unit <= 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit values must be finite and strictly between zero and one")
    return unit


def unit_to_logit(values: ArrayLike, *, size: int) -> np.ndarray:
    unit = unit_matrix(values, size=size)
    return (np.log(unit) - np.log1p(-unit)).astype(np.float32)


def standard_normal_quantile(values: ArrayLike, *, size: int) -> np.ndarray:
    unit = unit_matrix(values, size=size)
    epsilon = np.finfo(np.float64).eps
    tensor = torch.as_tensor(np.clip(unit, epsilon, 1.0 - epsilon), dtype=torch.float64)
    quantile = np.sqrt(2.0) * torch.erfinv(2.0 * tensor - 1.0)
    return quantile.numpy().astype(np.float32)
