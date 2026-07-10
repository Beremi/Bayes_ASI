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

    def test_queued_samples_have_larger_repulsion_radius(self) -> None:
        selector = ActiveSelector(
            ActiveLearningConfig(
                distance_weight=1.0,
                posterior_weight=0.0,
                queued_repulsion_radius_multiplier=2.0,
            )
        )
        candidates = np.array([[0.2, 0.0], [2.2, 0.0]])
        selected = selector.select(
            candidate_theta=candidates,
            candidate_log_prob=np.zeros(2),
            existing_theta=np.array([[0.0, 0.0]]),
            queued_theta=np.array([[2.0, 0.0]]),
            count=1,
        )
        self.assertEqual(selected.theta.shape, (1, 2))
        self.assertTrue(np.allclose(selected.theta[0], [0.2, 0.0]))

    def test_greedy_selection_treats_new_rows_as_queued(self) -> None:
        selector = ActiveSelector(
            ActiveLearningConfig(
                distance_weight=1.0,
                posterior_weight=0.0,
                queued_repulsion_radius_multiplier=2.0,
            )
        )
        candidates = np.array([[16.0, 0.0], [10.0, 0.0], [-5.0, 0.0]])
        selected = selector.select(
            candidate_theta=candidates,
            candidate_log_prob=np.zeros(3),
            existing_theta=np.array([[0.0, 0.0]]),
            count=2,
        )
        self.assertEqual(selected.theta.shape, (2, 2))
        self.assertTrue(np.allclose(selected.theta[0], [16.0, 0.0]))
        self.assertTrue(np.allclose(selected.theta[1], [-5.0, 0.0]))

    def test_queued_repulsion_multiplier_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ActiveLearningConfig(queued_repulsion_radius_multiplier=0.0)

    def test_non_finite_candidate_is_rejected(self) -> None:
        selector = ActiveSelector(ActiveLearningConfig())
        with self.assertRaisesRegex(ValueError, "finite"):
            selector.select(
                candidate_theta=np.array([[np.nan, 0.0]]),
                candidate_log_prob=np.array([0.0]),
                existing_theta=np.empty((0, 2)),
                count=1,
            )

    def test_negative_infinite_log_probability_is_not_selected(self) -> None:
        selector = ActiveSelector(ActiveLearningConfig(distance_weight=0.0, posterior_weight=1.0))
        selected = selector.select(
            candidate_theta=np.array([[0.0], [1.0], [2.0]]),
            candidate_log_prob=np.array([-np.inf, -1.0, 0.0]),
            existing_theta=np.empty((0, 1)),
            count=2,
        )
        self.assertTrue(np.array_equal(selected.theta.reshape(-1), [2.0, 1.0]))

    def test_chunked_minimum_distance_matches_direct_calculation(self) -> None:
        rng = np.random.default_rng(12)
        candidates = rng.normal(size=(20, 3))
        existing = rng.normal(size=(75, 3))
        scale = np.array([0.5, 2.0, 1.5])
        expected = np.min(
            np.linalg.norm((candidates[:, None, :] - existing[None, :, :]) / scale, axis=-1),
            axis=1,
        )
        actual = ActiveSelector._min_distances(candidates, existing, scale)
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
