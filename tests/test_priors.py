from __future__ import annotations

import unittest

import numpy as np
import torch
from scipy import special, stats

from bayes_asi.design import latin_hypercube
from bayes_asi.priors import (
    GaussianCopulaPrior,
    IndependentPrior,
    beta,
    exponential,
    gamma,
    laplace,
    logistic,
    lognormal,
    marginal_from_spec,
    normal,
    scipy_marginal,
    standard_normal_icdf,
    standard_normal_log_prob,
    student_t,
    uniform,
    weibull,
)


class PriorTests(unittest.TestCase):
    def test_standard_marginals_match_scipy_and_round_trip(self) -> None:
        marginals = [
            uniform(-2.0, 4.0),
            beta(2.3, 4.7, low=-1.0, high=3.0),
            normal(1.2, 2.5),
            exponential(1.7, shift=-0.4),
            lognormal(0.2, 0.8, shift=0.1),
            gamma(3.2, 1.4, shift=0.3),
            weibull(1.8, 2.1, shift=-0.2),
            student_t(6.0, loc=0.4, scale=1.3),
            laplace(-0.3, 1.7),
            logistic(0.8, 0.9),
        ]
        unit = np.linspace(0.01, 0.99, 41).reshape(-1, 1)
        z = standard_normal_icdf(unit)
        for marginal in marginals:
            prior = IndependentPrior([marginal])
            theta = prior.to_theta(torch.as_tensor(z, dtype=torch.float64)).detach().numpy()
            expected = np.asarray(marginal.distribution.ppf(unit[:, 0]), dtype=np.float64)
            self.assertTrue(np.allclose(theta[:, 0], expected, rtol=2e-12, atol=2e-12), marginal.name)
            recovered = prior.to_standard_normal(theta)
            self.assertTrue(np.allclose(recovered, z, rtol=2e-12, atol=2e-12), marginal.name)

    def test_mixed_prior_latin_hypercube_is_exact_in_probability_space(self) -> None:
        prior = IndependentPrior(
            [
                uniform(-3.0, 2.0),
                beta(1.7, 3.4),
                normal(2.0, 0.5),
                exponential(0.8),
                lognormal(-0.2, 0.6),
            ]
        )
        count = 29
        unit = latin_hypercube(np.random.default_rng(20260711), count, prior.latent_dim)
        z = standard_normal_icdf(unit)
        theta = prior.to_theta(torch.as_tensor(z, dtype=torch.float64)).detach().numpy()
        recovered_unit = special.ndtr(prior.to_standard_normal(theta))

        self.assertTrue(np.allclose(recovered_unit, unit, rtol=2e-12, atol=2e-12))
        for column in range(prior.latent_dim):
            strata = np.sort(np.floor(count * recovered_unit[:, column]).astype(int))
            self.assertTrue(np.array_equal(strata, np.arange(count)))

    def test_beta_and_gamma_transforms_have_correct_implicit_gradients(self) -> None:
        for marginal in (beta(2.3, 4.7, low=-1.0, high=3.0), gamma(3.2, 1.4, shift=0.3)):
            z = torch.linspace(-2.5, 2.5, 11, dtype=torch.float64, requires_grad=True)
            theta = marginal.from_standard_normal(z)
            (theta.sum()).backward()
            assert z.grad is not None

            epsilon = 1e-5
            upper = marginal.from_standard_normal(z.detach() + epsilon).detach().numpy()
            lower = marginal.from_standard_normal(z.detach() - epsilon).detach().numpy()
            finite_difference = (upper - lower) / (2.0 * epsilon)
            self.assertTrue(np.allclose(z.grad.numpy(), finite_difference, rtol=2e-6, atol=2e-7))

    def test_gaussian_copula_round_trip_and_dependence(self) -> None:
        correlation = np.array(
            [
                [1.0, 0.65, -0.25],
                [0.65, 1.0, 0.15],
                [-0.25, 0.15, 1.0],
            ]
        )
        marginals = [beta(2.0, 5.0), lognormal(0.1, 0.5), exponential(1.2)]
        prior = GaussianCopulaPrior(marginals, correlation)
        z = np.random.default_rng(12).normal(size=(20_000, 3))
        theta = prior.to_theta(torch.as_tensor(z, dtype=torch.float64)).detach().numpy()
        recovered = prior.to_standard_normal(theta)
        marginal_scores = np.column_stack(
            [marginal.to_standard_normal(theta[:, index]) for index, marginal in enumerate(marginals)]
        )

        self.assertTrue(np.allclose(recovered, z, rtol=2e-11, atol=2e-11))
        self.assertTrue(np.allclose(np.corrcoef(marginal_scores, rowvar=False), correlation, atol=0.02))

    def test_serializable_specs_and_generic_scipy_distribution(self) -> None:
        built = IndependentPrior(
            [
                {"distribution": "uniform", "low": -1.0, "high": 2.0},
                {"distribution": "beta", "alpha": 2.0, "beta": 3.0},
                {"distribution": "lognormal", "log_mean": 0.2, "log_sd": 0.4},
            ]
        )
        self.assertEqual([item.name for item in built.marginals], ["uniform", "beta", "lognormal"])
        custom = scipy_marginal(stats.invgauss(1.5, loc=0.2, scale=0.8), name="inverse_gaussian")
        value = custom.from_standard_normal(torch.tensor([0.0], dtype=torch.float64)).item()
        self.assertAlmostEqual(value, float(custom.distribution.ppf(0.5)), places=12)
        with self.assertRaisesRegex(ValueError, "unknown distribution"):
            marginal_from_spec({"distribution": "not_a_distribution"})

    def test_standard_normal_log_density_is_normalized(self) -> None:
        z = torch.tensor([[0.0, 0.0], [1.0, -2.0]], dtype=torch.float64)
        actual = standard_normal_log_prob(z).numpy()
        expected = np.sum(stats.norm.logpdf(z.numpy()), axis=1)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-14, atol=1e-14))

    def test_invalid_distribution_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            uniform(1.0, 1.0)
        with self.assertRaises(ValueError):
            beta(0.0, 2.0)
        with self.assertRaises(ValueError):
            normal(sd=-1.0)
        with self.assertRaises(ValueError):
            GaussianCopulaPrior([normal(), normal()], np.array([[1.0, 2.0], [2.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
