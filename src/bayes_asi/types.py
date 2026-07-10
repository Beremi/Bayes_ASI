from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(slots=True)
class SolverContext:
    """Context supplied to black-box solver wrappers."""

    run_dir: Path
    work_dir: Path
    job_id: str
    worker_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForwardResult:
    """Result returned by a black-box forward solver."""

    outputs: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)


class ForwardSolver(Protocol):
    """Protocol implemented by expensive forward-model wrappers."""

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        """Evaluate one physical parameter vector."""


class Posterior(Protocol):
    """Problem-specific posterior interface evaluated on a surrogate."""

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        """Return initial unconstrained latent samples for HMC or design."""

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        """Map unconstrained HMC variables to physical solver parameters."""

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        """Return log prior density in unconstrained coordinates."""

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Return surrogate log likelihood for theta and predicted outputs."""


class DifferentiableSurrogate(Protocol):
    """Tensor-to-tensor surrogate interface required by HMC."""

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict differentiable forward-model outputs."""
