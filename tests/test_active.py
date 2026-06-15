from __future__ import annotations

import unittest

import numpy as np

from bayes_asi.active import ActiveSelector
from bayes_asi.config import ActiveLearningConfig


class ActiveSelectorTests(unittest.TestCase):
    def test_selects_distant_point_when_posterior_scores_match(self) -> None:
        selector = ActiveSelector(ActiveLearningConfig(distance_weight=1.0, posterior_weight=0.0))
        candidates = np.array([[0.05, 0.0], [2.0, 0.0], [0.1, 0.0]])
        selected = selector.select(
            candidate_theta=candidates,
            candidate_log_prob=np.zeros(3),
            existing_theta=np.array([[0.0, 0.0]]),
            count=1,
        )
        self.assertEqual(selected.theta.shape, (1, 2))
        self.assertTrue(np.allclose(selected.theta[0], [2.0, 0.0]))

    def test_greedy_selection_updates_distance_to_new_samples(self) -> None:
        selector = ActiveSelector(ActiveLearningConfig(distance_weight=1.0, posterior_weight=0.0))
        candidates = np.array([[5.0, 0.0], [5.1, 0.0], [-5.0, 0.0]])
        selected = selector.select(
            candidate_theta=candidates,
            candidate_log_prob=np.zeros(3),
            existing_theta=np.array([[0.0, 0.0]]),
            count=2,
        )
        self.assertEqual(selected.theta.shape, (2, 2))
        self.assertTrue(any(np.allclose(row, [-5.0, 0.0]) for row in selected.theta))


if __name__ == "__main__":
    unittest.main()

