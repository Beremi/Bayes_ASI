from __future__ import annotations

import math
import time
import unittest

import numpy as np
import torch

from bayes_asi import ParallelTemperedHMCConfig, ParallelTemperedHMCResult, ParallelTemperedHMCSampler


def standard_normal_target(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = -0.5 * torch.sum(z * z + math.log(2.0 * math.pi), dim=-1)
    return reference, torch.zeros_like(reference)


def symmetric_bimodal_target(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = -0.5 * torch.sum(z * z + math.log(2.0 * math.pi), dim=-1)
    coordinate = z[:, 0]
    component = torch.stack(
        (
            -0.5 * torch.square((coordinate + 2.2) / 0.28),
            -0.5 * torch.square((coordinate - 2.2) / 0.28),
        ),
        dim=-1,
    )
    tempered = torch.logsumexp(component, dim=-1) - math.log(2.0)
    return reference, tempered


class ParallelTemperedHMCTests(unittest.TestCase):
    def test_reference_normal_sampling_and_adaptation(self) -> None:
        config = ParallelTemperedHMCConfig(
            inverse_temperatures=(1.0, 0.55, 0.2, 0.0),
            warmup_steps=80,
            samples=240,
            initial_step_size=0.12,
            min_leapfrog_steps=2,
            max_leapfrog_steps=8,
        )
        sampler = ParallelTemperedHMCSampler(
            config,
            standard_normal_target,
            seed=183,
            dtype=torch.float64,
        )
        result = sampler.sample(np.zeros((4, 2), dtype=np.float64))

        self.assertEqual(result.cold_z.shape, (240, 2))
        self.assertEqual(result.replica_z.shape, (240, 4, 2))
        self.assertTrue(np.all(np.isfinite(result.cold_log_prob)))
        self.assertLess(float(np.max(np.abs(np.mean(result.cold_z, axis=0)))), 0.30)
        self.assertTrue(np.all((np.var(result.cold_z, axis=0) > 0.55) & (np.var(result.cold_z, axis=0) < 1.45)))
        self.assertTrue(np.all(result.accept_rate > 0.45))
        np.testing.assert_allclose(result.swap_accept_rate, np.ones(3), rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.isfinite(result.step_size)))
        self.assertTrue(np.all((result.leapfrog_steps >= 2) & (result.leapfrog_steps <= 8)))
        self.assertTrue(np.all(result.mass_diag > 0.0))

    def test_parallel_tempering_crosses_both_bimodal_basins(self) -> None:
        beta = (1.0, 0.82, 0.65, 0.49, 0.34, 0.22, 0.12, 0.05, 0.0)
        config = ParallelTemperedHMCConfig(
            inverse_temperatures=beta,
            warmup_steps=220,
            samples=700,
            initial_step_size=0.04,
            min_leapfrog_steps=3,
            max_leapfrog_steps=12,
        )
        initial = np.linspace(-2.2, 2.2, len(beta), dtype=np.float64).reshape(-1, 1)
        sampler = ParallelTemperedHMCSampler(
            config,
            symmetric_bimodal_target,
            seed=9182,
            dtype=torch.float64,
        )
        result = sampler.sample(initial)

        negative_fraction = float(np.mean(result.cold_z[:, 0] < 0.0))
        self.assertGreater(negative_fraction, 0.10)
        self.assertLess(negative_fraction, 0.90)
        self.assertGreater(int(np.sum(np.diff(np.signbit(result.cold_z[:, 0])) != 0)), 2)
        self.assertTrue(np.all(result.swap_accept_rate > 0.04))
        self.assertGreater(len(np.unique(result.replica_labels[:, 0])), 2)

    def test_reproducibility_thinning_and_fixed_tuning(self) -> None:
        config = ParallelTemperedHMCConfig(
            inverse_temperatures=(1.0, 0.0), warmup_steps=0, samples=12,
            thinning=3, swap_interval=100, min_leapfrog_steps=2, max_leapfrog_steps=4,
        )
        tuning = dict(
            initial_step_sizes=np.array([0.15, 0.2]),
            initial_mass_diag=np.array([[1.5], [2.0]]),
            initial_leapfrog_steps=np.array([2, 4]),
        )
        def run() -> ParallelTemperedHMCResult:
            return ParallelTemperedHMCSampler(
                config, standard_normal_target, seed=24, dtype=torch.float64,
            ).sample(
                np.zeros((2, 1)),
                initial_step_sizes=tuning["initial_step_sizes"],
                initial_mass_diag=tuning["initial_mass_diag"],
                initial_leapfrog_steps=tuning["initial_leapfrog_steps"],
            )
        first, second = run(), run()
        np.testing.assert_array_equal(first.replica_z, second.replica_z)
        np.testing.assert_array_equal(first.step_size, tuning["initial_step_sizes"])
        np.testing.assert_array_equal(first.mass_diag, tuning["initial_mass_diag"])
        np.testing.assert_array_equal(first.leapfrog_steps, tuning["initial_leapfrog_steps"])
        self.assertEqual(first.replica_z.shape, (12, 2, 1))
        self.assertTrue(np.isnan(first.swap_accept_rate).all())
        self.assertEqual(first.warmup_target_evaluations, 0)
        self.assertEqual(first.sampling_target_evaluations, first.target_evaluations)
        np.testing.assert_array_equal(first.final_z, first.replica_z[-1])
        np.testing.assert_allclose(
            first.cold_log_prob,
            first.replica_reference_log_prob[:, 0] + first.replica_tempered_log_prob[:, 0],
        )

    def test_deadline_and_invalid_initial_states(self) -> None:
        sampler = ParallelTemperedHMCSampler(
            ParallelTemperedHMCConfig(inverse_temperatures=(1.0, 0.0), samples=1),
            standard_normal_target, seed=1,
        )
        with self.assertRaises(TimeoutError):
            sampler.sample(np.zeros((2, 1)), deadline_monotonic=time.monotonic() - 1)
        for initial in (np.zeros((1, 1)), np.zeros((2, 0)), np.full((2, 1), np.nan)):
            with self.subTest(shape=initial.shape), self.assertRaises(ValueError):
                sampler.sample(initial)

    def test_swap_ratio_direction_and_state_bookkeeping(self) -> None:
        sampler = ParallelTemperedHMCSampler(
            ParallelTemperedHMCConfig(inverse_temperatures=(1.0, 0.0)),
            standard_normal_target, seed=1, dtype=torch.float64,
        )
        z = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        reference = torch.tensor([-0.5, -2.0], dtype=torch.float64)
        tempered = torch.tensor([0.0, 1000.0], dtype=torch.float64)
        labels = torch.arange(2)
        np.testing.assert_array_equal(sampler._attempt_swaps(z, reference, tempered, labels, 0), [1])
        np.testing.assert_array_equal(z.numpy(), [[2.0], [1.0]])
        np.testing.assert_array_equal(reference.numpy(), [-2.0, -0.5])
        np.testing.assert_array_equal(tempered.numpy(), [1000.0, 0.0])
        np.testing.assert_array_equal(labels.numpy(), [1, 0])
        np.testing.assert_array_equal(sampler._attempt_swaps(z, reference, tempered, labels, 0), [0])
        np.testing.assert_array_equal(z.numpy(), [[2.0], [1.0]])

    def test_nonfinite_adaptation_settings_are_rejected(self) -> None:
        for field in ("gamma", "t0", "kappa", "min_adapted_variance", "max_adapted_variance"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ParallelTemperedHMCConfig(**{field: value})  # type: ignore[arg-type]

    def test_invalid_temperature_ladder_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            ParallelTemperedHMCConfig(inverse_temperatures=(1.0, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
