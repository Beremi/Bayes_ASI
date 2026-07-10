from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bayes_asi.config import build_factory_object, load_config
from bayes_asi.types import SolverContext


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SyntheticExampleTests(unittest.TestCase):
    def _exercise(self, config_name: str) -> tuple[object, object, torch.Tensor, torch.Tensor]:
        config = load_config(REPOSITORY_ROOT / "examples" / config_name)
        solver = build_factory_object(config.solver, config)
        posterior = build_factory_object(config.posterior, config)
        rng = np.random.default_rng(17)
        z = posterior.initial_z(rng, config.hmc.n_chains)
        z_tensor = torch.as_tensor(z, dtype=torch.float32).requires_grad_(True)
        theta = posterior.z_to_theta(z_tensor)
        with tempfile.TemporaryDirectory() as tmp:
            context = SolverContext(Path(tmp), Path(tmp), "example", 1)
            outputs = np.vstack([solver.evaluate(row, context).outputs for row in theta.detach().numpy()])
        y_pred = torch.as_tensor(outputs, dtype=torch.float32)
        log_probability = posterior.log_prior_z(z_tensor) + posterior.log_likelihood(theta, y_pred)
        gradient = torch.autograd.grad(log_probability.sum(), z_tensor)[0]

        self.assertEqual(tuple(theta.shape), (config.hmc.n_chains, config.problem.parameter_dim))
        self.assertEqual(tuple(y_pred.shape), (config.hmc.n_chains, config.problem.output_dim))
        self.assertEqual(tuple(log_probability.shape), (config.hmc.n_chains,))
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(gradient).all())
        return config, posterior, theta, log_probability

    def test_nonlinear_example_callbacks(self) -> None:
        self._exercise("synthetic.yaml")

    def test_linear_gaussian_example_callbacks(self) -> None:
        self._exercise("linear_gaussian.yaml")

    def test_bounded_decay_example_respects_parameter_bounds(self) -> None:
        _, posterior, theta, _ = self._exercise("bounded_decay.yaml")
        bounds = posterior.bounds
        self.assertTrue(torch.all(theta > bounds[:, 0]))
        self.assertTrue(torch.all(theta < bounds[:, 1]))

    def test_bimodal_example_has_symmetric_log_probability(self) -> None:
        _, posterior, _, _ = self._exercise("bimodal.yaml")
        theta = torch.tensor([[-1.5], [1.5]])
        y_pred = theta**2
        logp = posterior.log_prior_z(theta) + posterior.log_likelihood(theta, y_pred)
        self.assertAlmostEqual(float(logp[0]), float(logp[1]), places=6)


if __name__ == "__main__":
    unittest.main()
