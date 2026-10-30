from __future__ import annotations

import unittest

import numpy as np
import torch
from scipy import stats

from bayes_asi.parameters import NamedParameterSpace
from bayes_asi.priors import normal, truncated_normal


class ParameterTests(unittest.TestCase):
    def test_fixed_coordinates_and_truncated_prior_round_trip(self) -> None:
        prior = NamedParameterSpace(["fixed", "k", "phi"], [4.0, normal(-18, 2), truncated_normal(.12, .04, .05, .2)])
        z = torch.tensor([[-2., .4], [1., -3.]], dtype=torch.float64, requires_grad=True)
        theta = prior.to_theta(z)
        np.testing.assert_allclose(prior.to_standard_normal(theta.detach().numpy()), z.detach(), atol=1e-10)
        theta.sum().backward()
        assert z.grad is not None
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.all(theta[:, 0] == 4))
        expected = stats.truncnorm.ppf(stats.norm.cdf(.4), (.05-.12)/.04, (.2-.12)/.04, loc=.12, scale=.04)
        self.assertAlmostEqual(float(theta[0, 2].detach()), expected)
        with self.assertRaisesRegex(ValueError, "fixed parameter"):
            prior.to_standard_normal(np.array([3., -18., .12]))
        with self.assertRaisesRegex(ValueError, "order"):
            prior.validate_names(["k", "fixed", "phi"])

    def test_truncated_transport_gradient(self) -> None:
        marginal = truncated_normal(0, 1, -1, 2)
        z = torch.tensor([-.7, .2, 1.4], dtype=torch.float64, requires_grad=True)
        out = marginal.from_standard_normal(z)
        out.sum().backward()
        assert z.grad is not None
        eps = 1e-5
        finite_difference = (marginal.from_standard_normal(z.detach()+eps)
                             - marginal.from_standard_normal(z.detach()-eps))/(2*eps)
        np.testing.assert_allclose(z.grad, finite_difference, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
