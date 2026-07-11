from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from bayes_asi.config import AppConfig
from bayes_asi.examples._common import float_vector, require_dimensions, unit_to_logit
from bayes_asi.types import ForwardResult, SolverContext

SENSOR_POSITIONS = np.array(
    [
        [0.15, 0.20],
        [0.15, 0.50],
        [0.15, 0.80],
        [0.35, 0.20],
        [0.35, 0.50],
        [0.35, 0.80],
        [0.55, 0.20],
        [0.55, 0.50],
        [0.55, 0.80],
        [0.78, 0.20],
        [0.78, 0.50],
        [0.78, 0.80],
    ],
    dtype=np.float64,
)
DEFAULT_BOUNDS = np.array([[-2.5, 1.0], [-4.0, 0.5], [-1.5, 2.0]], dtype=np.float32)
DEFAULT_TRUE_LOG_CONDUCTIVITY = np.array([-0.35, -2.1, 0.65], dtype=np.float32)
OUTPUT_DIM = SENSOR_POSITIONS.shape[0] + 1


def _harmonic_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return 2.0 * first * second / (first + second)


@dataclass(slots=True)
class DarcySolution:
    outputs: np.ndarray
    head: np.ndarray
    conductivity: np.ndarray


class DarcyFlowModel:
    """Cell-centered finite-volume model on the unit square."""

    def __init__(self, nx: int = 96, ny: int = 96) -> None:
        if nx < 8 or ny < 8:
            raise ValueError("Darcy grid dimensions must both be at least 8")
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = 1.0 / self.nx
        self.dy = 1.0 / self.ny
        self.cell_ids = np.arange(self.nx * self.ny, dtype=np.int64).reshape(self.ny, self.nx)

        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.material_id = self._build_material_ids()
        self.sensor_ids, self.sensor_weights = self._build_sensor_interpolation()

        self.horizontal_left = self.cell_ids[:, :-1].reshape(-1)
        self.horizontal_right = self.cell_ids[:, 1:].reshape(-1)
        self.vertical_bottom = self.cell_ids[:-1, :].reshape(-1)
        self.vertical_top = self.cell_ids[1:, :].reshape(-1)
        self.left_boundary = self.cell_ids[:, 0]
        self.right_boundary = self.cell_ids[:, -1]

    def _build_material_ids(self) -> np.ndarray:
        material = np.zeros((self.ny, self.nx), dtype=np.int8)
        lens = ((self.x_grid - 0.40) / 0.19) ** 2 + ((self.y_grid - 0.62) / 0.23) ** 2 <= 1.0
        channel_center = 0.14 + 0.68 * self.x_grid
        channel = (self.x_grid >= 0.22) & (self.x_grid <= 0.92) & (np.abs(self.y_grid - channel_center) <= 0.055)
        material[lens] = 1
        material[channel] = 2
        return material

    def _build_sensor_interpolation(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.empty((SENSOR_POSITIONS.shape[0], 4), dtype=np.int64)
        weights = np.empty((SENSOR_POSITIONS.shape[0], 4), dtype=np.float64)
        for row, (x, y) in enumerate(SENSOR_POSITIONS):
            x_index = np.clip(x / self.dx - 0.5, 0.0, self.nx - 1.0)
            y_index = np.clip(y / self.dy - 0.5, 0.0, self.ny - 1.0)
            i0 = min(int(np.floor(x_index)), self.nx - 2)
            j0 = min(int(np.floor(y_index)), self.ny - 2)
            wx = x_index - i0
            wy = y_index - j0
            ids[row] = [
                self.cell_ids[j0, i0],
                self.cell_ids[j0, i0 + 1],
                self.cell_ids[j0 + 1, i0],
                self.cell_ids[j0 + 1, i0 + 1],
            ]
            weights[row] = [
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ]
        return ids, weights

    def solve(self, log_conductivity: np.ndarray) -> DarcySolution:
        parameters = np.asarray(log_conductivity, dtype=np.float64)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise ValueError("log_conductivity must be a finite vector of length 3")
        conductivity = np.exp(parameters[self.material_id])
        flat_conductivity = conductivity.reshape(-1)

        horizontal_transmissibility = (
            _harmonic_mean(
                flat_conductivity[self.horizontal_left],
                flat_conductivity[self.horizontal_right],
            )
            * self.dy
            / self.dx
        )
        vertical_transmissibility = (
            _harmonic_mean(
                flat_conductivity[self.vertical_bottom],
                flat_conductivity[self.vertical_top],
            )
            * self.dx
            / self.dy
        )
        left_transmissibility = 2.0 * flat_conductivity[self.left_boundary] * self.dy / self.dx
        right_transmissibility = 2.0 * flat_conductivity[self.right_boundary] * self.dy / self.dx

        rows = np.concatenate(
            [
                self.horizontal_left,
                self.horizontal_left,
                self.horizontal_right,
                self.horizontal_right,
                self.vertical_bottom,
                self.vertical_bottom,
                self.vertical_top,
                self.vertical_top,
                self.left_boundary,
                self.right_boundary,
            ]
        )
        columns = np.concatenate(
            [
                self.horizontal_left,
                self.horizontal_right,
                self.horizontal_left,
                self.horizontal_right,
                self.vertical_bottom,
                self.vertical_top,
                self.vertical_bottom,
                self.vertical_top,
                self.left_boundary,
                self.right_boundary,
            ]
        )
        values = np.concatenate(
            [
                horizontal_transmissibility,
                -horizontal_transmissibility,
                -horizontal_transmissibility,
                horizontal_transmissibility,
                vertical_transmissibility,
                -vertical_transmissibility,
                -vertical_transmissibility,
                vertical_transmissibility,
                left_transmissibility,
                right_transmissibility,
            ]
        )
        matrix = coo_matrix((values, (rows, columns)), shape=(self.nx * self.ny, self.nx * self.ny)).tocsr()
        right_hand_side = np.zeros(self.nx * self.ny, dtype=np.float64)
        right_hand_side[self.left_boundary] = left_transmissibility
        head_flat = np.asarray(spsolve(matrix, right_hand_side), dtype=np.float64)
        if not np.all(np.isfinite(head_flat)):
            raise RuntimeError("Darcy solve produced non-finite hydraulic head")

        sensor_head = np.sum(head_flat[self.sensor_ids] * self.sensor_weights, axis=1)
        outlet_flux = float(np.sum(right_transmissibility * head_flat[self.right_boundary]))
        outputs = np.concatenate([sensor_head, np.asarray([outlet_flux])])
        return DarcySolution(
            outputs=outputs,
            head=head_flat.reshape(self.ny, self.nx),
            conductivity=conductivity,
        )


@dataclass(slots=True)
class DarcyFlowSolver:
    model: DarcyFlowModel
    minimum_runtime_seconds: float = 1.5
    save_fields: bool = False

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        started = time.monotonic()
        solution = self.model.solve(theta)
        artifact_paths: list[str] = []
        if self.save_fields:
            path = context.work_dir / "darcy_fields.npz"
            np.savez_compressed(
                path,
                head=solution.head,
                conductivity=solution.conductivity,
                sensor_positions=SENSOR_POSITIONS,
            )
            artifact_paths.append(str(path))
        numerical_elapsed = time.monotonic() - started
        remaining = self.minimum_runtime_seconds - numerical_elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        return ForwardResult(
            outputs=solution.outputs,
            metadata={
                "model": "steady_2d_darcy",
                "grid": [self.model.nx, self.model.ny],
                "numerical_elapsed_s": numerical_elapsed,
                "minimum_runtime_s": self.minimum_runtime_seconds,
                "conductivity_min": float(np.min(solution.conductivity)),
                "conductivity_max": float(np.max(solution.conductivity)),
            },
            artifact_paths=artifact_paths,
        )


@dataclass(slots=True)
class DarcyFlowPosterior:
    bounds: torch.Tensor
    observed: torch.Tensor
    noise_sd: torch.Tensor

    def initial_z(self, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        unit = np.clip(rng.random((n_chains, 3)), 1e-6, 1.0 - 1e-6)
        return self.prior_transform(unit)

    def prior_transform(self, unit: np.ndarray) -> np.ndarray:
        return unit_to_logit(unit, size=3)

    def z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 3)
        bounds = self.bounds.to(device=z.device, dtype=z.dtype)
        lower = bounds[:, 0]
        span = bounds[:, 1] - bounds[:, 0]
        return lower + span * torch.sigmoid(z)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, 3)
        return torch.sum(functional.logsigmoid(z) + functional.logsigmoid(-z), dim=-1)

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        observed = self.observed.to(device=y_pred.device, dtype=y_pred.dtype)
        noise_sd = self.noise_sd.to(device=y_pred.device, dtype=y_pred.dtype)
        residual = (y_pred.reshape(-1, OUTPUT_DIM) - observed) / noise_sd
        return -0.5 * torch.sum(residual**2, dim=-1)


def make_solver(
    config: AppConfig,
    nx: int = 96,
    ny: int = 96,
    minimum_runtime_seconds: float = 1.5,
    save_fields: bool = False,
) -> DarcyFlowSolver:
    require_dimensions(config, latent_dim=3, parameter_dim=3, output_dim=OUTPUT_DIM)
    if not np.isfinite(minimum_runtime_seconds) or minimum_runtime_seconds < 0.0:
        raise ValueError("minimum_runtime_seconds must be finite and non-negative")
    return DarcyFlowSolver(
        model=DarcyFlowModel(nx=nx, ny=ny),
        minimum_runtime_seconds=float(minimum_runtime_seconds),
        save_fields=bool(save_fields),
    )


def make_posterior(
    config: AppConfig,
    theta_true: list[float] | None = None,
    bounds: list[list[float]] | None = None,
    nx: int = 96,
    ny: int = 96,
    head_noise_sd: float = 0.01,
    relative_flux_noise: float = 0.03,
) -> DarcyFlowPosterior:
    require_dimensions(config, latent_dim=3, parameter_dim=3, output_dim=OUTPUT_DIM)
    true_parameters = float_vector(
        DEFAULT_TRUE_LOG_CONDUCTIVITY if theta_true is None else theta_true,
        size=3,
        name="theta_true",
    )
    bounds_array = np.asarray(DEFAULT_BOUNDS if bounds is None else bounds, dtype=np.float32)
    if bounds_array.shape != (3, 2) or not np.all(np.isfinite(bounds_array)):
        raise ValueError("bounds must be a finite 3x2 array")
    if np.any(bounds_array[:, 1] <= bounds_array[:, 0]):
        raise ValueError("each upper bound must be greater than its lower bound")
    if np.any(true_parameters <= bounds_array[:, 0]) or np.any(true_parameters >= bounds_array[:, 1]):
        raise ValueError("theta_true must lie strictly inside bounds")
    if not np.isfinite(head_noise_sd) or head_noise_sd <= 0.0:
        raise ValueError("head_noise_sd must be positive and finite")
    if not np.isfinite(relative_flux_noise) or relative_flux_noise <= 0.0:
        raise ValueError("relative_flux_noise must be positive and finite")

    reference = DarcyFlowModel(nx=nx, ny=ny).solve(true_parameters).outputs.astype(np.float32)
    noise = np.full(OUTPUT_DIM, float(head_noise_sd), dtype=np.float32)
    noise[-1] = max(abs(float(reference[-1])) * float(relative_flux_noise), 1e-6)
    return DarcyFlowPosterior(
        bounds=torch.as_tensor(bounds_array),
        observed=torch.as_tensor(reference),
        noise_sd=torch.as_tensor(noise),
    )
