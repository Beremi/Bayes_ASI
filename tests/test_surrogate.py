from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bayes_asi.config import SurrogateConfig
from bayes_asi.surrogate import SurrogateModel, SurrogateTrainer


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


if __name__ == "__main__":
    unittest.main()

