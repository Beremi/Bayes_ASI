from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from bayes_asi.config import AppConfig
from bayes_asi.types import ForwardResult, SolverContext


def synthetic_forward_torch(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(-1, 2)
    t0 = theta[:, 0]
    t1 = theta[:, 1]
    y0 = torch.sin(t0) + 0.20 * t1 * t1
    y1 = t0 * t1 + 0.10 * t0
    return torch.stack([y0, y1], dim=-1)


def synthetic_forward_numpy(theta: np.ndarray) -> np.ndarray:
    theta_t = torch.as_tensor(theta, dtype=torch.float32).reshape(1, 2)
    with torch.no_grad():
        return synthetic_forward_torch(theta_t).reshape(-1).numpy()


@dataclass(slots=True)
class SyntheticSolver:
    sleep_seconds: float = 0.0

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        outputs = synthetic_forward_numpy(theta)
        return ForwardResult(outputs=outputs, metadata={"solver": "synthetic", "job_id": context.job_id})


@dataclass(slots=True)
class SyntheticPosterior:
    observed: torch.Tensor
    noise_sd: torch.Tensor
    prior_sd: float = 2.0

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        return rng.normal(loc=0.0, scale=self.prior_sd, size=(n_chains, 2)).astype(np.float32)

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 2)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 2)
        return -0.5 * torch.sum((z / self.prior_sd) ** 2, dim=-1)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        observed = self.observed.to(device=y_pred.device, dtype=y_pred.dtype)
        noise_sd = self.noise_sd.to(device=y_pred.device, dtype=y_pred.dtype)
        residual = (y_pred.reshape(-1, 2) - observed.reshape(1, 2)) / noise_sd.reshape(1, 2)
        return -0.5 * torch.sum(residual * residual, dim=-1)


def make_solver(config: AppConfig, sleep_seconds: float = 0.0) -> SyntheticSolver:
    del config
    return SyntheticSolver(sleep_seconds=float(sleep_seconds))


def make_posterior(
    config: AppConfig,
    theta_true: list[float] | None = None,
    noise_sd: list[float] | None = None,
    prior_sd: float = 2.0,
) -> SyntheticPosterior:
    del config
    theta_true = [0.7, -1.2] if theta_true is None else theta_true
    noise_sd = [0.12, 0.18] if noise_sd is None else noise_sd
    with torch.no_grad():
        observed = synthetic_forward_torch(torch.as_tensor(theta_true, dtype=torch.float32)).reshape(-1)
    return SyntheticPosterior(
        observed=observed,
        noise_sd=torch.as_tensor(noise_sd, dtype=torch.float32),
        prior_sd=float(prior_sd),
    )

