from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from bayes_asi.config import AppConfig
from bayes_asi.examples._common import require_dimensions
from bayes_asi.types import ForwardResult, SolverContext


def bimodal_forward_torch(theta: torch.Tensor) -> torch.Tensor:
    return theta.reshape(-1, 1) ** 2


@dataclass(slots=True)
class BimodalSolver:
    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        outputs = np.square(np.asarray(theta, dtype=float).reshape(1))
        return ForwardResult(
            outputs=outputs,
            metadata={"solver": "bimodal", "job_id": context.job_id},
        )


@dataclass(slots=True)
class BimodalPosterior:
    observed: float
    noise_sd: float
    prior_sd: float
    mode_location: float

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        signs = np.where(np.arange(n_chains) % 2 == 0, -1.0, 1.0)
        centers = signs * self.mode_location
        return rng.normal(centers, 0.25, size=n_chains).reshape(-1, 1).astype(np.float32)

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 1)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z.reshape(-1) / self.prior_sd) ** 2

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        residual = (y_pred.reshape(-1) - self.observed) / self.noise_sd
        return -0.5 * residual**2


def make_solver(config: AppConfig) -> BimodalSolver:
    require_dimensions(config, latent_dim=1, parameter_dim=1, output_dim=1)
    return BimodalSolver()


def make_posterior(
    config: AppConfig,
    theta_true: float = 1.5,
    noise_sd: float = 0.08,
    prior_sd: float = 2.0,
) -> BimodalPosterior:
    require_dimensions(config, latent_dim=1, parameter_dim=1, output_dim=1)
    values = np.asarray([theta_true, noise_sd, prior_sd], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("theta_true, noise_sd, and prior_sd must be finite")
    if theta_true == 0.0 or noise_sd <= 0.0 or prior_sd <= 0.0:
        raise ValueError("theta_true must be nonzero and scales must be positive")
    return BimodalPosterior(
        observed=float(theta_true**2),
        noise_sd=float(noise_sd),
        prior_sd=float(prior_sd),
        mode_location=abs(float(theta_true)),
    )
