"""Named physical coordinates, including parameters fixed outside inference."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from bayes_asi.priors import IndependentPrior, Marginal


class NamedParameterSpace:
    """A reversible prior with explicit column order and optional constants.

    Marginals define inferred coordinates; float entries are fixed physical
    parameters. No latent coordinate or artificial prior is assigned to a constant.
    """

    def __init__(self, names: Sequence[str], components: Sequence[Marginal | float]) -> None:
        self.names = tuple(names)
        if not self.names or any(not n.strip() for n in self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be non-empty and unique")
        if len(components) != len(self.names):
            raise ValueError("names and components must have equal lengths")
        self.fixed = {i: float(c) for i, c in enumerate(components) if not isinstance(c, Marginal)}
        if not all(np.isfinite(v) for v in self.fixed.values()):
            raise ValueError("fixed parameter values must be finite")
        self.free_indices = tuple(i for i in range(len(self.names)) if i not in self.fixed)
        self.free_prior = IndependentPrior([c for c in components if isinstance(c, Marginal)])

    @property
    def latent_dim(self) -> int:
        return self.free_prior.latent_dim

    @property
    def parameter_dim(self) -> int:
        return len(self.names)

    def to_theta(self, z: torch.Tensor) -> torch.Tensor:
        free = self.free_prior.to_theta(z)
        indices = {physical: latent for latent, physical in enumerate(self.free_indices)}
        return torch.stack([
            torch.full(z.shape[:-1], self.fixed[i], dtype=z.dtype, device=z.device)
            if i in self.fixed else free[..., indices[i]]
            for i in range(self.parameter_dim)
        ], dim=-1)

    def to_standard_normal(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != self.parameter_dim or not np.all(np.isfinite(values)):
            raise ValueError(f"theta must be finite with trailing dimension {self.parameter_dim}")
        for i, constant in self.fixed.items():
            if not np.allclose(values[..., i], constant, rtol=0, atol=1e-10):
                raise ValueError(f"fixed parameter {self.names[i]} differs from {constant}")
        return self.free_prior.to_standard_normal(values[..., self.free_indices])

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        if set(values) != set(self.names):
            raise ValueError("parameter mapping does not match named schema")
        result = np.asarray([values[n] for n in self.names], dtype=np.float64)
        self.to_standard_normal(result)
        return result

    def validate_names(self, names: Sequence[str]) -> None:
        if tuple(names) != self.names:
            raise ValueError("parameter column names/order differ from the configured schema")
