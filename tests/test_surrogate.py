from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bayes_asi.config import SurrogateConfig
import torch

from bayes_asi.surrogate import Normalization, SurrogateModel, SurrogateTrainer, TanhMLP


class SurrogateTests(unittest.TestCase):
    def test_checkpoint_reload_and_jacobian_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.normal(size=(40, 2)).astype(np.float32)
        outputs = np.column_stack([theta[:, 0] + 2.0 * theta[:, 1], theta[:, 0] ** 2]).astype(np.float32)
        trainer = SurrogateTrainer(
            SurrogateConfig(
                hidden_layers=[12],
                max_epochs=80,
                batch_size=32,
                validation_fraction=0.0,
                patience=20,
                min_train_samples=4,
            ),
            parameter_dim=2,
            output_dim=2,
            seed=5,
        )
        result = trainer.train(theta, outputs, version=1)
        self.assertEqual(result.model.training_samples, 40)
        self.assertGreaterEqual(result.epochs, result.best_epoch)
        point = np.array([0.2, -0.3], dtype=np.float32)
        jac = result.model.jacobian_numpy(point)
        eps = 1e-3
        finite_rows = []
        for out_idx in range(2):
            row = []
            for dim in range(2):
                plus = point.copy()
                minus = point.copy()
                plus[dim] += eps
                minus[dim] -= eps
                f_plus = result.model.predict_numpy(plus.reshape(1, -1))[0, out_idx]
                f_minus = result.model.predict_numpy(minus.reshape(1, -1))[0, out_idx]
                row.append((f_plus - f_minus) / (2.0 * eps))
            finite_rows.append(row)
        self.assertTrue(np.allclose(jac, np.asarray(finite_rows), atol=2e-2))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate.pt"
            result.model.save(path)
            loaded = SurrogateModel.load(path)
            self.assertTrue(np.allclose(result.model.predict_numpy(theta[:3]), loaded.predict_numpy(theta[:3])))
            self.assertEqual(loaded.training_samples, 40)

    def test_warm_start_preserves_predictions_when_normalization_changes(self) -> None:
        trainer = SurrogateTrainer(SurrogateConfig(hidden_layers=[5]), 2, 2, seed=3)
        old = Normalization(
            x_mean=torch.tensor([1.0, -2.0]),
            x_std=torch.tensor([2.0, 0.5]),
            y_mean=torch.tensor([4.0, -1.0]),
            y_std=torch.tensor([3.0, 2.0]),
        )
        new = Normalization(
            x_mean=torch.tensor([-3.0, 5.0]),
            x_std=torch.tensor([0.7, 4.0]),
            y_mean=torch.tensor([-2.0, 8.0]),
            y_std=torch.tensor([1.5, 6.0]),
        )
        with torch.random.fork_rng():
            torch.manual_seed(7)
            network = TanhMLP(2, 2, [5])
        old_model = SurrogateModel(network, old, [5], version=1, val_loss=0.0, device="cpu")
        points = torch.tensor([[0.2, -0.3], [2.5, 1.2]])
        expected = old_model(points).detach().numpy()

        trainer._adapt_normalization(network, old, new)
        new_model = SurrogateModel(network, new, [5], version=2, val_loss=0.0, device="cpu")
        self.assertTrue(np.allclose(expected, new_model(points).detach().numpy(), atol=1e-6))

    def test_training_rejects_non_finite_data(self) -> None:
        trainer = SurrogateTrainer(SurrogateConfig(hidden_layers=[4], max_epochs=2), 1, 1, seed=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            trainer.train(np.array([[0.0], [1.0]]), np.array([[0.0], [np.nan]]), version=1)


if __name__ == "__main__":
    unittest.main()
