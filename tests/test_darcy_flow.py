from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from bayes_asi.examples.darcy_flow import (
    DEFAULT_TRUE_LOG_CONDUCTIVITY,
    SENSOR_POSITIONS,
    DarcyFlowModel,
    DarcyFlowSolver,
)
from bayes_asi.types import SolverContext


class DarcyFlowTests(unittest.TestCase):
    def test_constant_conductivity_matches_exact_linear_head_and_flux(self) -> None:
        model = DarcyFlowModel(nx=32, ny=28)
        log_conductivity = 0.2
        solution = model.solve(np.full(3, log_conductivity))

        expected_head = 1.0 - SENSOR_POSITIONS[:, 0]
        expected_flux = np.exp(log_conductivity)
        self.assertTrue(np.allclose(solution.outputs[:-1], expected_head, atol=1e-11))
        self.assertAlmostEqual(float(solution.outputs[-1]), float(expected_flux), places=11)

    def test_each_material_parameter_changes_observations(self) -> None:
        model = DarcyFlowModel(nx=40, ny=36)
        baseline = model.solve(DEFAULT_TRUE_LOG_CONDUCTIVITY).outputs
        for index in range(3):
            perturbed = DEFAULT_TRUE_LOG_CONDUCTIVITY.copy()
            perturbed[index] += 0.2
            difference = np.linalg.norm(model.solve(perturbed).outputs - baseline)
            self.assertGreater(difference, 1e-3, f"material parameter {index} is not observable")

    def test_material_parameters_are_recovered_from_exact_observations(self) -> None:
        model = DarcyFlowModel(nx=36, ny=32)
        target = model.solve(DEFAULT_TRUE_LOG_CONDUCTIVITY).outputs
        scales = np.full(target.shape, 0.01)
        scales[-1] = abs(target[-1]) * 0.03

        result = least_squares(
            lambda theta: (model.solve(theta).outputs - target) / scales,
            x0=np.array([0.5, -3.5, -1.0]),
            bounds=([-2.5, -4.0, -1.5], [1.0, 0.5, 2.0]),
            max_nfev=50,
        )

        self.assertTrue(result.success, result.message)
        self.assertTrue(np.allclose(result.x, DEFAULT_TRUE_LOG_CONDUCTIVITY, atol=1e-5))

    def test_solver_enforces_minimum_runtime_and_can_save_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            context = SolverContext(Path(tmp), work_dir, "job_00000000", 1)
            solver = DarcyFlowSolver(
                model=DarcyFlowModel(nx=20, ny=20),
                minimum_runtime_seconds=0.12,
                save_fields=True,
            )
            started = time.monotonic()
            result = solver.evaluate(DEFAULT_TRUE_LOG_CONDUCTIVITY, context)
            elapsed = time.monotonic() - started

            self.assertGreaterEqual(elapsed, 0.11)
            self.assertEqual(result.outputs.shape, (13,))
            self.assertTrue(np.all(np.isfinite(result.outputs)))
            self.assertEqual(len(result.artifact_paths), 1)
            self.assertTrue(Path(result.artifact_paths[0]).is_file())

    def test_invalid_grid_and_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8"):
            DarcyFlowModel(nx=7, ny=8)
        model = DarcyFlowModel(nx=8, ny=8)
        with self.assertRaisesRegex(ValueError, "length 3"):
            model.solve(np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
