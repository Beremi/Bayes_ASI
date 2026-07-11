from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from bayes_asi.config import HMCConfig
from bayes_asi.hmc import HMCSampler
from bayes_asi.priors import IndependentPrior, beta, lognormal, normal, uniform


class ZeroSurrogate:
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros((theta.reshape(-1, 2).shape[0], 1), dtype=theta.dtype, device=theta.device)


class StandardNormalPosterior:
    prior = IndependentPrior([normal(), normal()])

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        return rng.normal(size=(n_chains, 2)).astype(np.float32)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        return torch.zeros((y_pred.shape[0],), dtype=y_pred.dtype, device=y_pred.device)


class BoundedPosterior:
    prior = IndependentPrior([uniform(10.0, 20.0), uniform(-4.0, -2.0)])

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        self.last_theta = theta.detach().clone()
        return -0.5 * torch.sum(y_pred * y_pred, dim=-1)


class MixedPriorPosterior:
    prior = IndependentPrior([beta(2.0, 5.0), lognormal(-0.2, 0.5)])

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del y_pred
        target = torch.tensor([0.3, 1.0], dtype=theta.dtype, device=theta.device)
        scale = torch.tensor([0.2, 0.3], dtype=theta.dtype, device=theta.device)
        return -0.5 * torch.sum(((theta - target) / scale) ** 2, dim=-1)


class RecordingSurrogate:
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.last_input = z.detach().clone()
        return z


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
        self.assertGreaterEqual(result.warmup_accept_rate, 0.0)
        self.assertLess(np.abs(result.z.mean(axis=0)).max(), 0.8)

    def test_non_finite_initial_state_is_rejected(self) -> None:
        sampler = HMCSampler(
            HMCConfig(n_chains=1, warmup_steps=0, samples_per_chain=1),
            StandardNormalPosterior(),
            ZeroSurrogate(),
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            sampler.sample(np.array([[np.nan, 0.0]], dtype=np.float32))

    def test_surrogate_receives_standard_normal_coordinates_not_physical_parameters(self) -> None:
        posterior = BoundedPosterior()
        surrogate = RecordingSurrogate()
        sampler = HMCSampler(
            HMCConfig(n_chains=1, warmup_steps=0, samples_per_chain=1),
            posterior,
            surrogate,
            seed=1,
        )
        z = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        log_probability, gradient = sampler._log_prob_and_grad(z)

        self.assertTrue(torch.equal(surrogate.last_input, z))
        self.assertTrue(torch.all(posterior.last_theta[:, 0] > 10.0))
        self.assertTrue(torch.all(posterior.last_theta[:, 1] < -2.0))
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_hmc_traverses_beta_and_lognormal_prior_transport(self) -> None:
        sampler = HMCSampler(
            HMCConfig(
                n_chains=2,
                warmup_steps=12,
                samples_per_chain=24,
                initial_step_size=0.08,
                min_leapfrog_steps=2,
                max_leapfrog_steps=6,
            ),
            MixedPriorPosterior(),
            ZeroSurrogate(),
            seed=31,
        )
        result = sampler.sample(np.zeros((2, 2), dtype=np.float32))

        self.assertTrue(np.isfinite(result.z).all())
        self.assertTrue(np.isfinite(result.theta).all())
        self.assertTrue(np.all((result.theta[:, 0] > 0.0) & (result.theta[:, 0] < 1.0)))
        self.assertTrue(np.all(result.theta[:, 1] > 0.0))
        self.assertGreater(result.accept_rate, 0.1)

    def test_adapted_step_size_is_bounded(self) -> None:
        self.assertAlmostEqual(HMCSampler._bounded_step_size(-1000.0), 1e-6)
        self.assertAlmostEqual(HMCSampler._bounded_step_size(1000.0), 10.0)

    def test_deadline_stops_sampling(self) -> None:
        sampler = HMCSampler(
            HMCConfig(n_chains=1, warmup_steps=0, samples_per_chain=1),
            StandardNormalPosterior(),
            ZeroSurrogate(),
            seed=1,
        )
        with patch("bayes_asi.hmc.time.monotonic", return_value=10.0):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                sampler.sample(np.zeros((1, 2), dtype=np.float32), deadline_monotonic=5.0)


if __name__ == "__main__":
    unittest.main()
