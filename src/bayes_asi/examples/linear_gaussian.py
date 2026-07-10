from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from bayes_asi.config import AppConfig
from bayes_asi.examples._common import float_vector, require_dimensions
from bayes_asi.types import ForwardResult, SolverContext


DESIGN_MATRIX = torch.tensor(
    [
        [1.0, 0.25],
        [-0.5, 1.0],
        [0.75, -1.25],
    ],
    dtype=torch.float32,
)
BIAS = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)


def linear_forward_torch(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(-1, 2)
    design = DESIGN_MATRIX.to(device=theta.device, dtype=theta.dtype)
    bias = BIAS.to(device=theta.device, dtype=theta.dtype)
    return theta @ design.T + bias


@dataclass(slots=True)
class LinearGaussianSolver:
    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        with torch.no_grad():
            outputs = linear_forward_torch(torch.as_tensor(theta, dtype=torch.float32)).reshape(-1).numpy()
        return ForwardResult(
            outputs=outputs,
            metadata={"solver": "linear_gaussian", "job_id": context.job_id},
        )


@dataclass(slots=True)
class LinearGaussianPosterior:
    observed: torch.Tensor
    noise_sd: torch.Tensor
    prior_sd: torch.Tensor

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        return rng.normal(size=(n_chains, 2)).astype(np.float32) * self.prior_sd.numpy()

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 2)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 2)
        prior_sd = self.prior_sd.to(device=z.device, dtype=z.dtype)
        return -0.5 * torch.sum((z / prior_sd) ** 2, dim=-1)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        observed = self.observed.to(device=y_pred.device, dtype=y_pred.dtype)
        noise_sd = self.noise_sd.to(device=y_pred.device, dtype=y_pred.dtype)
        residual = (y_pred.reshape(-1, 3) - observed) / noise_sd
        return -0.5 * torch.sum(residual**2, dim=-1)


def make_solver(config: AppConfig) -> LinearGaussianSolver:
    require_dimensions(config, latent_dim=2, parameter_dim=2, output_dim=3)
    return LinearGaussianSolver()


def make_posterior(
    config: AppConfig,
    theta_true: list[float] | None = None,
    noise_sd: list[float] | None = None,
    prior_sd: list[float] | None = None,
) -> LinearGaussianPosterior:
    require_dimensions(config, latent_dim=2, parameter_dim=2, output_dim=3)
    theta_true_array = float_vector(
        [0.8, -1.1] if theta_true is None else theta_true,
        size=2,
        name="theta_true",
    )
    noise_array = float_vector(
        [0.08, 0.12, 0.1] if noise_sd is None else noise_sd,
        size=3,
        name="noise_sd",
        positive=True,
    )
    prior_array = float_vector(
        [2.0, 2.0] if prior_sd is None else prior_sd,
        size=2,
        name="prior_sd",
        positive=True,
    )
    with torch.no_grad():
        observed = linear_forward_torch(torch.as_tensor(theta_true_array)).reshape(-1)
    return LinearGaussianPosterior(
        observed=observed,
        noise_sd=torch.as_tensor(noise_array),
        prior_sd=torch.as_tensor(prior_array),
    )
