from __future__ import annotations

import unittest

import numpy as np
import torch

from bayes_asi.config import HMCConfig
from bayes_asi.hmc import HMCSampler


class ZeroSurrogate:
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros((theta.reshape(-1, 2).shape[0], 1), dtype=theta.dtype, device=theta.device)


class StandardNormalPosterior:
    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        return rng.normal(size=(n_chains, 2)).astype(np.float32)

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 2)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 2)
        return -0.5 * torch.sum(z * z, dim=-1)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        return torch.zeros((y_pred.shape[0],), dtype=y_pred.dtype, device=y_pred.device)


class HMCTests(unittest.TestCase):
    def test_hmc_samples_standard_normal_shape_and_acceptance(self) -> None:
        posterior = StandardNormalPosterior()
        rng = np.random.default_rng(8)
        initial = posterior.initial_z(rng, 2)
        sampler = HMCSampler(
            HMCConfig(
                n_chains=2,
                warmup_steps=20,
                samples_per_chain=40,
                initial_step_size=0.1,
                min_leapfrog_steps=2,
                max_leapfrog_steps=8,
            ),
            posterior,
            ZeroSurrogate(),
            seed=9,
        )
        result = sampler.sample(initial)
        self.assertEqual(result.z.shape, (80, 2))
        self.assertTrue(np.isfinite(result.log_prob).all())
        self.assertGreater(result.accept_rate, 0.1)
        self.assertLess(np.abs(result.z.mean(axis=0)).max(), 0.8)


if __name__ == "__main__":
    unittest.main()

