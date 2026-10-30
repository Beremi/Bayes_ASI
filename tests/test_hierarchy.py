from __future__ import annotations

import unittest

import numpy as np

from bayes_asi.active import ActiveSelector
from bayes_asi.config import ActiveLearningConfig
from bayes_asi.hierarchy import draw_distillation_coordinates, prescreen_candidates
from bayes_asi.storage import stable_theta_key


class HierarchyTests(unittest.TestCase):
    def test_distillation_mixes_prior_and_local_points_reproducibly(self) -> None:
        completed = np.array([[5.5, -5.5], [5.0, -5.0]])
        first = draw_distillation_coordinates(np.random.default_rng(41), completed, 7)
        second = draw_distillation_coordinates(np.random.default_rng(41), completed, 7)

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (7, 2))
        self.assertTrue(np.all(np.abs(first) <= 6.0))
        # The latter half is local to the far-from-origin completed cloud.
        self.assertTrue(np.all(first[4:, 0] > 3.0))

    def test_prescreen_is_capped_unique_and_reserves_exploration(self) -> None:
        selector = ActiveSelector(
            ActiveLearningConfig(
                candidate_pool_size=20,
                batch_max_size=2,
                posterior_weight=1.0,
                distance_weight=0.5,
                duplicate_round_decimals=8,
            )
        )
        theta = np.arange(10, dtype=float).reshape(-1, 1)
        theta[1] = theta[0]
        z = theta.copy()
        indices = prescreen_candidates(
            selector=selector,
            rng=np.random.default_rng(7),
            candidate_z=z,
            candidate_theta=theta,
            candidate_log_prob=np.arange(10, dtype=float),
            existing_z=np.empty((0, 1)),
            queued_z=np.empty((0, 1)),
            known_keys={stable_theta_key(np.array([2.0]), 8)},
            screen_size=4,
            exploration_fraction=0.5,
        )

        self.assertEqual(indices.shape, (4,))
        keys = {stable_theta_key(theta[index], 8) for index in indices}
        self.assertEqual(len(keys), 4)
        self.assertNotIn(stable_theta_key(np.array([2.0]), 8), keys)
        # Half the screen is random rather than exclusively the highest log density.
        self.assertTrue(any(index < 7 for index in indices))


if __name__ == "__main__":
    unittest.main()
