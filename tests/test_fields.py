import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bayes_asi.fields import KLField, rotated_permeability_tensor


class FieldTests(unittest.TestCase):
    def test_covariance_eigendecomposition_and_scaled_npz(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T
        field = KLField.from_covariance(np.arange(4.0), cov)
        np.testing.assert_allclose(field.basis @ field.basis.T, cov, atol=1e-10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "basis.npz"
            np.savez(path, basis=np.ones((2, 1)) * 3.0)
            loaded = KLField.from_npz(path)
            np.testing.assert_allclose(loaded.evaluate([2.0]), [6.0, 6.0])
        tiny = KLField.from_covariance(np.zeros(2), np.diag([1e-24, 2e-24]))
        np.testing.assert_allclose(tiny.basis @ tiny.basis.T, np.diag([1e-24, 2e-24]), rtol=1e-10, atol=0)

    def test_invalid_covariance(self) -> None:
        with self.assertRaises(ValueError):
            KLField.from_covariance([0, 0], [[1, 2], [0, 1]])
        with self.assertRaises(ValueError):
            KLField.from_covariance([0, 0], [[1, 0], [0, -1]])

    def test_numpy_torch_batches_and_gradient(self) -> None:
        field = KLField(np.array([1.0, -2.0, 0.5]), np.array([[1.0, 0.0], [0.2, 2.0], [-1.0, 0.5]]))
        x = np.array([[0.1, -0.4], [1.0, 2.0]])
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        np.testing.assert_allclose(field.evaluate(x), field.evaluate_torch(xt).detach().numpy())
        field.evaluate_torch(xt).sum().backward()
        assert xt.grad is not None
        self.assertTrue(torch.isfinite(xt.grad).all())
        with self.assertRaises(ValueError):
            field.evaluate([np.nan, 0.0])
        with self.assertRaises(ValueError):
            field.evaluate_torch(torch.tensor([float("inf"), 0.0]))

    def test_bounded_weighted_means_and_zero_mode(self) -> None:
        field = KLField(np.zeros(3), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
        weights = np.array([1.0, 2.0, 3.0])
        values = field.bounded([[1.0, -0.5], [-2.0, 1.5]], -2.0, 3.0, weights, target=[0.2, 1.1])
        np.testing.assert_allclose(np.average(values, axis=1, weights=weights), [0.2, 1.1], atol=1e-12)
        baseline = field.bounded([0.0, 0.0], -2.0, 3.0, weights, target=0.7)
        np.testing.assert_allclose(baseline, 0.7)
        with self.assertRaises(ValueError):
            field.bounded([0.0, 0.0], -2.0, 3.0, [1, 0, 1], target=0.0)

    def test_rotated_tensor(self) -> None:
        tensor = rotated_permeability_tensor([0.0, 1.0], [0.0, 90.0], [4.0, 0.25])
        np.testing.assert_allclose(np.linalg.eigvalsh(tensor), np.array([[1, 4], [2.5, 10]]))
        np.testing.assert_allclose(tensor[0], np.array([[1, 0], [0, 4]]))
        np.testing.assert_allclose(tensor[1], np.array([[2.5, 0], [0, 10]]), atol=1e-14)
        with self.assertRaises(ValueError):
            rotated_permeability_tensor(400.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            rotated_permeability_tensor(-400.0, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
