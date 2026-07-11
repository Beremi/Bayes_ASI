from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from bayes_asi.config import AppConfig
from bayes_asi.examples._common import float_vector, require_dimensions, unit_to_logit
from bayes_asi.types import ForwardResult, SolverContext

TIME_POINTS = torch.linspace(0.0, 3.0, 16, dtype=torch.float32)
DEFAULT_BOUNDS = np.array([[0.5, 2.5], [0.2, 2.0], [-0.2, 0.5]], dtype=np.float32)


def decay_forward_torch(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(-1, 3)
    time = TIME_POINTS.to(device=theta.device, dtype=theta.dtype)
    amplitude = theta[:, 0:1]
    rate = theta[:, 1:2]
    offset = theta[:, 2:3]
    return amplitude * torch.exp(-rate * time.reshape(1, -1)) + offset


@dataclass(slots=True)
class BoundedDecaySolver:
    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        with torch.no_grad():
            outputs = decay_forward_torch(torch.as_tensor(theta, dtype=torch.float32)).reshape(-1).numpy()
        return ForwardResult(
            outputs=outputs,
            metadata={"solver": "bounded_decay", "job_id": context.job_id},
        )


@dataclass(slots=True)
class BoundedDecayPosterior:
    bounds: torch.Tensor
    observed: torch.Tensor
    noise_sd: float

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        unit = np.clip(rng.random((n_chains, 3)), 1e-6, 1.0 - 1e-6)
        return self.prior_transform(unit)

    def prior_transform(self, unit: np.ndarray) -> np.ndarray:
        return unit_to_logit(unit, size=3)

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 3)
        bounds = self.bounds.to(device=z.device, dtype=z.dtype)
        lower = bounds[:, 0]
        span = bounds[:, 1] - bounds[:, 0]
        return lower + span * torch.sigmoid(z)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 3)
        return torch.sum(functional.logsigmoid(z) + functional.logsigmoid(-z), dim=-1)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        observed = self.observed.to(device=y_pred.device, dtype=y_pred.dtype)
        residual = (y_pred.reshape(-1, 16) - observed) / self.noise_sd
        return -0.5 * torch.sum(residual**2, dim=-1)


def make_solver(config: AppConfig) -> BoundedDecaySolver:
    require_dimensions(config, latent_dim=3, parameter_dim=3, output_dim=16)
    return BoundedDecaySolver()


def make_posterior(
    config: AppConfig,
    theta_true: list[float] | None = None,
    bounds: list[list[float]] | None = None,
    noise_sd: float = 0.04,
) -> BoundedDecayPosterior:
    require_dimensions(config, latent_dim=3, parameter_dim=3, output_dim=16)
    theta_true_array = float_vector(
        [1.7, 0.8, 0.1] if theta_true is None else theta_true,
        size=3,
        name="theta_true",
    )
    bounds_array = np.asarray(DEFAULT_BOUNDS if bounds is None else bounds, dtype=np.float32)
    if bounds_array.shape != (3, 2) or not np.all(np.isfinite(bounds_array)):
        raise ValueError("bounds must be a finite 3x2 array")
    if np.any(bounds_array[:, 1] <= bounds_array[:, 0]):
        raise ValueError("each upper bound must be greater than its lower bound")
    if np.any(theta_true_array <= bounds_array[:, 0]) or np.any(theta_true_array >= bounds_array[:, 1]):
        raise ValueError("theta_true must lie strictly inside bounds")
    if not np.isfinite(noise_sd) or noise_sd <= 0.0:
        raise ValueError("noise_sd must be positive and finite")
    with torch.no_grad():
        observed = decay_forward_torch(torch.as_tensor(theta_true_array)).reshape(-1)
    return BoundedDecayPosterior(
        bounds=torch.as_tensor(bounds_array),
        observed=observed,
        noise_sd=float(noise_sd),
    )
