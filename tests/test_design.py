from __future__ import annotations

import unittest

import numpy as np

from bayes_asi.design import latin_hypercube


class LatinHypercubeTests(unittest.TestCase):
    def test_each_dimension_occupies_every_stratum_once(self) -> None:
        n_points = 37
        samples = latin_hypercube(np.random.default_rng(123), n_points, 5)

        self.assertEqual(samples.shape, (n_points, 5))
        self.assertTrue(np.all(samples > 0.0))
        self.assertTrue(np.all(samples < 1.0))
        for column in range(samples.shape[1]):
            strata = np.floor(samples[:, column] * n_points).astype(int)
            self.assertTrue(np.array_equal(np.sort(strata), np.arange(n_points)))

    def test_fixed_seed_is_reproducible(self) -> None:
        first = latin_hypercube(np.random.default_rng(19), 8, 3)
        second = latin_hypercube(np.random.default_rng(19), 8, 3)
        self.assertTrue(np.array_equal(first, second))

    def test_invalid_dimensions_are_rejected(self) -> None:
        rng = np.random.default_rng(1)
        with self.assertRaisesRegex(ValueError, "n_points"):
            latin_hypercube(rng, 0, 2)
        with self.assertRaisesRegex(ValueError, "dimension"):
            latin_hypercube(rng, 2, 0)


if __name__ == "__main__":
    unittest.main()
