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
    """Surrogate target plus optional rich data returned by a forward solver."""

    outputs: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def model_outputs(self) -> np.ndarray:
        """Dense values that are trained by the surrogate and passed to the likelihood."""

        return self.outputs


class ForwardSolver(Protocol):
    """Protocol implemented by expensive forward-model wrappers."""

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        """Evaluate one physical parameter vector."""


class PriorTransform(Protocol):
    """Reversible map between a standard-normal reference and model parameters."""

    @property
    def latent_dim(self) -> int:
        """Dimension of the independent standard-normal reference."""

    @property
    def parameter_dim(self) -> int:
        """Dimension of the physical model parameters."""

    def to_theta(self, z: torch.Tensor) -> torch.Tensor:
        """Map standard-normal coordinates to physical solver parameters."""

    def to_standard_normal(self, theta: np.ndarray) -> np.ndarray:
        """Map physical solver parameters back to standard-normal coordinates."""


class Posterior(Protocol):
    """Problem-specific likelihood evaluated on a surrogate model output."""

    @property
    def prior(self) -> PriorTransform:
        """Physical-prior transport driven by the framework's Gaussian reference."""

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Return surrogate log likelihood for theta and predicted outputs."""


class DifferentiableSurrogate(Protocol):
    """Tensor-to-tensor surrogate interface required by HMC."""

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Predict differentiable forward-model outputs."""
