import unittest

import numpy as np
import torch
from scipy.optimize import lsq_linear

from bayes_asi.likelihoods import GaussianLikelihood, ProfiledAffineLikelihood


class LikelihoodTests(unittest.TestCase):
    def test_weighted_gaussian(self) -> None:
        like = GaussianLikelihood([1.0, 2.0], [0.5, 1.0], weights=[2.0, 1.0], power_weight=2.0)
        got = like(torch.tensor([[1.5, 1.0]], dtype=torch.float64))[0]
        expected = 2 * sum(
            w * (-0.5 * ((v - o) / s) ** 2 - 0.5 * np.log(2 * np.pi * s**2))
            for w, v, o, s in zip([2, 1], [1.5, 1.0], [1, 2], [.5, 1], strict=True)
        )
        self.assertAlmostEqual(float(got), expected)
        with self.assertRaises(ValueError):
            GaussianLikelihood([1, 2], [1, 1], weights=[0, 0])

    def test_profile_boundaries_constant_and_global(self) -> None:
        observed = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        # Constant x exercises the no-divide-by-zero path and boundary choice.
        like = ProfiledAffineLikelihood(observed, 0.2, a_bounds=(-1, 1), b_bounds=(-2, 0.5))
        pred, a, b = like.fit(torch.zeros(1, 6))
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue(torch.all(a <= 1) and torch.all(b <= 0.5))
        global_like = ProfiledAffineLikelihood(observed, 0.2, a_bounds=(-10, 10), b_bounds=(-10, 10), per_support=False)
        _, ag, bg = global_like.fit(torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]))
        self.assertTrue(torch.allclose(ag[..., 0], ag[..., 1]))
        self.assertTrue(torch.allclose(bg[..., 0], bg[..., 1]))

    def test_profile_gradient_and_coefficient_mode(self) -> None:
        days = np.array([0.0, 1.0, 2.0, 3.0])
        observed = torch.tensor([[1.0, 1.2, 1.4, 1.6]])
        like = ProfiledAffineLikelihood(
            observed, 0.2, a_bounds=(-3, 3), b_bounds=(-3, 3),
            likelihood_mode="coefficient_fixed_c_d_abe", model_days=days,
            coefficient_fixed_c_d_abe=(0.3, 0.2), coefficient_sigmas=[.1, .2, .3],
        )
        x = torch.tensor([[0.2, 0.4, 0.6, 0.8]], requires_grad=True)
        value = like(x).sum()
        value.backward()
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):
            ProfiledAffineLikelihood(observed, 1.0, a_bounds=(1, 0), b_bounds=(-1, 1))

    def test_randomized_scipy_profile_parity(self) -> None:
        rng = np.random.default_rng(19)
        worst = 0.0
        for case in range(40):
            t = int(rng.integers(4, 9))
            if case % 10 == 0:
                x = np.full(t, rng.normal())
            elif case % 10 == 1:
                x = 2.0 + 1e-7 * rng.normal(size=t)
            else:
                x = rng.normal(size=t)
            y = rng.normal(size=t)
            like = ProfiledAffineLikelihood(
                torch.tensor(y[None, :], dtype=torch.float64), 0.37,
                a_bounds=(-0.35, 0.4), b_bounds=(-0.55, 0.65),
            )
            pred, a, b = like.fit(torch.tensor(x[None, :], dtype=torch.float64))
            # BVLS with a tight tolerance is a more reliable independent
            # reference than TRF for the deliberately ill-conditioned case.
            ref = lsq_linear(
                np.column_stack((np.ones(t), x)), y,
                bounds=([-0.35, -0.55], [0.4, 0.65]), method="bvls", tol=1e-13,
            )
            ref_pred = ref.x[0] + ref.x[1] * x
            ours_sse = float(torch.sum((pred[0, 0] - torch.tensor(y, dtype=torch.float64)) ** 2))
            ref_sse = float(np.sum((ref_pred - y) ** 2))
            err = abs(ours_sse - ref_sse) / max(1.0, abs(ref_sse))
            worst = max(worst, err)
            self.assertLess(err, 1e-8)
            if case % 10 in (0, 1):
                np.testing.assert_allclose(pred.detach().numpy()[0, 0], ref_pred, rtol=1e-8, atol=1e-8)
            else:
                np.testing.assert_allclose([float(a[0, 0]), float(b[0, 0])], ref.x, rtol=1e-8, atol=1e-8)
            raw = float(like(torch.tensor(x[None, :], dtype=torch.float64))[0])
            expected = -0.5 * ref_sse / 0.37**2 - 0.5 * t * np.log(2 * np.pi * 0.37**2)
            self.assertLess(abs(raw - expected), 1e-8 * max(1.0, abs(expected)))
        self.assertLess(worst, 1e-8)

    def test_global_profile_and_finite_difference_gradients(self) -> None:
        rng = np.random.default_rng(8)
        x = rng.normal(size=(3, 5))
        y = rng.normal(size=(3, 5))
        like = ProfiledAffineLikelihood(torch.tensor(y, dtype=torch.float64), 0.4,
                                        a_bounds=(-10, 10), b_bounds=(-10, 10), per_support=False)
        flat = torch.tensor(x.reshape(1, -1), dtype=torch.float64, requires_grad=True)
        value = like(flat).sum()
        value.backward()
        assert flat.grad is not None
        analytic = flat.grad.detach().numpy().copy()
        numeric = np.empty_like(analytic)
        h = 1e-6
        for i in range(flat.numel()):
            plus, minus = flat.detach().clone(), flat.detach().clone()
            plus.view(-1)[i] += h
            minus.view(-1)[i] -= h
            numeric.reshape(-1)[i] = float((like(plus) - like(minus)) / (2 * h))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)

        coeff = ProfiledAffineLikelihood(
            torch.tensor(y[:1], dtype=torch.float64), 0.4, a_bounds=(-10, 10), b_bounds=(-10, 10),
            likelihood_mode="coefficient_fixed_c_d_abe", model_days=np.arange(5.0),
            coefficient_fixed_c_d_abe=(0.3, 0.2), coefficient_sigmas=[.2, .3, .4],
        )
        one = torch.tensor(x[:1], dtype=torch.float64, requires_grad=True)
        coeff(one).backward()
        assert one.grad is not None
        analytic = one.grad.detach().numpy().copy()
        numeric = np.empty_like(analytic)
        for i in range(one.numel()):
            plus, minus = one.detach().clone(), one.detach().clone()
            plus.view(-1)[i] += h
            minus.view(-1)[i] -= h
            numeric.reshape(-1)[i] = float((coeff(plus) - coeff(minus)) / (2 * h))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)

    def test_profile_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            ProfiledAffineLikelihood([[1, 2], [3, 4]], [1, 1], a_bounds=(-1, 1), b_bounds=(-1, 1))
        with self.assertRaises(ValueError):
            ProfiledAffineLikelihood([[1, 2], [3, 4]], 1, a_bounds=(-1, 1), b_bounds=(-1, 1)).fit([[np.nan] * 4])
        with self.assertRaises(ValueError):
            ProfiledAffineLikelihood(np.empty((0, 2)), 1, a_bounds=(-1, 1), b_bounds=(-1, 1))


if __name__ == "__main__":
    unittest.main()
