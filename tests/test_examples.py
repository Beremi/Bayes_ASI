from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from bayes_asi.config import build_factory_object, load_config
from bayes_asi.design import latin_hypercube
from bayes_asi.types import SolverContext

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SyntheticExampleTests(unittest.TestCase):
    def _exercise(
        self,
        config_name: str,
        *,
        solver_overrides: dict[str, Any] | None = None,
        posterior_overrides: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, torch.Tensor, torch.Tensor]:
        config = load_config(REPOSITORY_ROOT / "examples" / config_name)
        config.solver.kwargs.update(solver_overrides or {})
        config.posterior.kwargs.update(posterior_overrides or {})
        solver = build_factory_object(config.solver, config)
        posterior = build_factory_object(config.posterior, config)
        rng = np.random.default_rng(17)
        unit = latin_hypercube(rng, config.hmc.n_chains, config.problem.latent_dim)
        prior_z = posterior.prior_transform(unit)
        self.assertEqual(prior_z.shape, (config.hmc.n_chains, config.problem.latent_dim))
        self.assertTrue(np.all(np.isfinite(prior_z)))
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
        _, posterior, _, _ = self._exercise("linear_gaussian.yaml")
        unit = np.array([[0.1, 0.25], [0.5, 0.75], [0.9, 0.6]])
        transformed = posterior.prior_transform(unit)
        standard = np.vectorize(NormalDist().inv_cdf)(unit)
        expected = standard * posterior.prior_sd.numpy()
        self.assertTrue(np.allclose(transformed, expected, atol=1e-6))

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

    def test_darcy_flow_example_callbacks(self) -> None:
        self._exercise(
            "darcy_flow.yaml",
            solver_overrides={"nx": 24, "ny": 20, "minimum_runtime_seconds": 0.0},
            posterior_overrides={"nx": 24, "ny": 20},
        )


if __name__ == "__main__":
    unittest.main()
