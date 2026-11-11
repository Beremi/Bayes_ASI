from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

ReferenceTemperedLogDensity = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class ParallelTemperedHMCConfig:
    """Configuration for one replica-exchange ladder of adaptive HMC chains."""

    inverse_temperatures: tuple[float, ...] = (
        1.0,
        0.80,
        0.62,
        0.46,
        0.32,
        0.21,
        0.12,
        0.06,
        0.02,
        0.0,
    )
    warmup_steps: int = 300
    samples: int = 500
    thinning: int = 1
    swap_interval: int = 1
    target_accept: float = 0.8
    initial_step_size: float = 0.025
    min_leapfrog_steps: int = 3
    max_leapfrog_steps: int = 18
    leapfrog_jitter: float = 0.2
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75
    min_adapted_variance: float = 1.0e-3
    max_adapted_variance: float = 1.0e3

    def __post_init__(self) -> None:
        beta = np.asarray(self.inverse_temperatures, dtype=np.float64)
        if beta.ndim != 1 or len(beta) < 2:
            raise ValueError("parallel tempering requires at least two inverse temperatures")
        if not np.all(np.isfinite(beta)) or np.any(beta < 0.0) or np.any(beta > 1.0):
            raise ValueError("inverse temperatures must be finite and in [0, 1]")
        if not math.isclose(float(beta[0]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the first inverse temperature must be 1")
        if np.any(np.diff(beta) >= 0.0):
            raise ValueError("inverse temperatures must be strictly decreasing")
        if self.warmup_steps < 0 or self.samples <= 0 or self.thinning <= 0:
            raise ValueError("warmup must be nonnegative; samples and thinning must be positive")
        if self.swap_interval <= 0:
            raise ValueError("swap_interval must be positive")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must be in (0, 1)")
        if not math.isfinite(self.initial_step_size) or self.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be finite and positive")
        if self.min_leapfrog_steps <= 0 or self.max_leapfrog_steps < self.min_leapfrog_steps:
            raise ValueError("leapfrog bounds must satisfy 0 < min <= max")
        if not 0.0 <= self.leapfrog_jitter <= 1.0:
            raise ValueError("leapfrog_jitter must be in [0, 1]")
        if (
            not all(math.isfinite(value) for value in (self.gamma, self.t0, self.kappa))
            or self.gamma <= 0.0 or self.t0 <= 0.0 or not 0.0 < self.kappa <= 1.0
        ):
            raise ValueError("dual-averaging parameters are invalid")
        if (
            not all(math.isfinite(value) for value in (self.min_adapted_variance, self.max_adapted_variance))
            or self.min_adapted_variance <= 0.0
            or self.max_adapted_variance < self.min_adapted_variance
        ):
            raise ValueError("adapted variance bounds must satisfy 0 < min <= max")


@dataclass(slots=True)
class ParallelTemperedHMCResult:
    cold_z: np.ndarray
    cold_log_prob: np.ndarray
    replica_z: np.ndarray
    replica_reference_log_prob: np.ndarray
    replica_tempered_log_prob: np.ndarray
    replica_labels: np.ndarray
    inverse_temperatures: np.ndarray
    accept_rate: np.ndarray
    warmup_accept_rate: np.ndarray
    swap_accept_rate: np.ndarray
    warmup_swap_accept_rate: np.ndarray
    step_size: np.ndarray
    leapfrog_steps: np.ndarray
    mass_diag: np.ndarray
    leapfrog_efficiency: np.ndarray
    final_z: np.ndarray
    elapsed_seconds: float
    warmup_elapsed_seconds: float
    sampling_elapsed_seconds: float
    target_evaluations: int
    warmup_target_evaluations: int
    sampling_target_evaluations: int


class ParallelTemperedHMCSampler:
    r"""Reference-preserving replica exchange with adaptive HMC transitions.

    The callback returns ``(log_reference, log_tempered)``. Replica ``r``
    targets

    ``log pi_r(z) = log_reference(z) + beta_r * log_tempered(z)``.

    This is useful for Bayesian inversion because the hottest chain retains a
    proper prior instead of flattening the entire posterior. Both callback
    components must be finite, differentiable, and row-separable, with finite
    gradients, and the same density functions must apply to every row.
    Hard-support targets represented by negative infinity are not
    supported, including at beta zero. Step sizes are
    dual-averaged, a diagonal precision mass is estimated halfway through
    warmup, and leapfrog counts are selected by expected squared jumping
    distance per gradient evaluation.
    """

    def __init__(
        self,
        config: ParallelTemperedHMCConfig,
        log_density: ReferenceTemperedLogDensity,
        *,
        seed: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.log_density = log_density
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.beta = torch.as_tensor(
            config.inverse_temperatures,
            dtype=dtype,
            device=self.device,
        )
        self.target_evaluations = 0

    def sample(
        self,
        initial_z: np.ndarray | torch.Tensor,
        *,
        initial_step_sizes: np.ndarray | None = None,
        initial_mass_diag: np.ndarray | None = None,
        initial_leapfrog_steps: np.ndarray | None = None,
        deadline_monotonic: float | None = None,
    ) -> ParallelTemperedHMCResult:
        started = time.monotonic()
        self.target_evaluations = 0
        warmup_elapsed_seconds = 0.0
        sampling_started = started
        warmup_target_evaluations = 0
        current = torch.as_tensor(initial_z, dtype=self.dtype, device=self.device)
        replica_count = len(self.config.inverse_temperatures)
        if current.ndim != 2 or current.shape[0] != replica_count or current.shape[1] == 0:
            raise ValueError(f"initial_z must have shape ({replica_count}, dimension)")
        if not torch.isfinite(current).all():
            raise ValueError("initial_z must be finite")
        dimension = int(current.shape[1])

        step_size = self._initial_vector(
            initial_step_sizes,
            replica_count,
            self.config.initial_step_size,
            "initial_step_sizes",
        )
        mass_diag = self._initial_matrix(
            initial_mass_diag,
            replica_count,
            dimension,
            1.0,
            "initial_mass_diag",
        ).to(device=self.device, dtype=self.dtype)
        inv_mass = 1.0 / mass_diag
        tuned_leapfrog = self._initial_integer_vector(
            initial_leapfrog_steps,
            replica_count,
            (self.config.min_leapfrog_steps + self.config.max_leapfrog_steps) // 2,
        )

        log_eps = np.log(step_size)
        mu = np.log(10.0 * step_size)
        hbar = np.zeros(replica_count, dtype=np.float64)
        log_eps_bar = log_eps.copy()
        dual_iteration = 0

        mass_count = 0
        mass_mean = np.zeros((replica_count, dimension), dtype=np.float64)
        mass_m2 = np.zeros_like(mass_mean)
        mass_update_iteration = self.config.warmup_steps // 2 if self.config.warmup_steps >= 20 else 0

        leapfrog_span = self.config.max_leapfrog_steps - self.config.min_leapfrog_steps + 1
        leapfrog_score_sum = np.zeros((replica_count, leapfrog_span), dtype=np.float64)
        leapfrog_score_count = np.zeros((replica_count, leapfrog_span), dtype=np.int64)

        hmc_accepts = np.zeros(replica_count, dtype=np.float64)
        warmup_hmc_accepts = np.zeros(replica_count, dtype=np.float64)
        swap_attempts = np.zeros(replica_count - 1, dtype=np.int64)
        swap_accepts = np.zeros(replica_count - 1, dtype=np.int64)
        warmup_swap_attempts = np.zeros(replica_count - 1, dtype=np.int64)
        warmup_swap_accepts = np.zeros(replica_count - 1, dtype=np.int64)

        labels = torch.arange(replica_count, dtype=torch.int64, device=self.device)
        swap_parity = 0
        replica_samples: list[np.ndarray] = []
        reference_samples: list[np.ndarray] = []
        tempered_samples: list[np.ndarray] = []
        label_samples: list[np.ndarray] = []

        current_reference, current_tempered, current_total, _ = self._components_and_grad(current)
        if not torch.isfinite(current_total).all():
            raise ValueError("initial replica states have non-finite log density")

        total_transitions = self.config.warmup_steps + self.config.samples * self.config.thinning
        retained = 0
        for transition in range(1, total_transitions + 1):
            self._check_deadline(deadline_monotonic)
            in_warmup = transition <= self.config.warmup_steps
            if in_warmup:
                leapfrog_steps = self._warmup_leapfrog_counts(transition, replica_count)
            else:
                leapfrog_steps = self._jittered_leapfrog_counts(tuned_leapfrog)

            previous = current.detach().clone()
            current_reference, current_tempered, current_total, current_grad = self._components_and_grad(current)
            momentum = torch.randn(
                current.shape,
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            ) * torch.sqrt(mass_diag)
            proposal, proposal_momentum, proposal_reference, proposal_tempered, proposal_total = self._leapfrog(
                current,
                momentum,
                current_grad,
                step_size,
                leapfrog_steps,
                inv_mass,
                deadline_monotonic,
            )
            current_hamiltonian = -current_total + self._kinetic(momentum, inv_mass)
            proposal_hamiltonian = -proposal_total + self._kinetic(proposal_momentum, inv_mass)
            log_acceptance = current_hamiltonian - proposal_hamiltonian
            acceptance_probability = torch.exp(torch.clamp(log_acceptance, max=0.0))
            acceptance_probability = torch.nan_to_num(
                acceptance_probability,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            uniform = torch.rand(
                (replica_count,),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            accepted = uniform < acceptance_probability
            current = torch.where(accepted[:, None], proposal, current).detach()
            current_reference = torch.where(accepted, proposal_reference, current_reference).detach()
            current_tempered = torch.where(accepted, proposal_tempered, current_tempered).detach()

            accepted_numpy = accepted.detach().cpu().double().numpy()
            acceptance_numpy = acceptance_probability.detach().cpu().double().numpy()
            if in_warmup:
                warmup_hmc_accepts += accepted_numpy
                dual_iteration += 1
                hbar, log_eps, log_eps_bar = self._dual_average_update(
                    dual_iteration,
                    acceptance_numpy,
                    mu,
                    hbar,
                    log_eps_bar,
                )
                step_size = self._bounded_step_sizes(np.exp(log_eps))
            else:
                hmc_accepts += accepted_numpy

            if in_warmup and (mass_update_iteration == 0 or transition <= mass_update_iteration):
                mass_count += 1
                state_numpy = current.detach().cpu().double().numpy()
                delta = state_numpy - mass_mean
                mass_mean += delta / mass_count
                mass_m2 += delta * (state_numpy - mass_mean)

            if in_warmup and mass_update_iteration > 0 and transition == mass_update_iteration:
                if mass_count > 1:
                    variance = mass_m2 / (mass_count - 1)
                    variance = np.clip(
                        variance,
                        self.config.min_adapted_variance,
                        self.config.max_adapted_variance,
                    )
                    mass_diag = torch.as_tensor(
                        1.0 / variance,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    inv_mass = 1.0 / mass_diag
                step_size = self._bounded_step_sizes(np.exp(log_eps_bar))
                log_eps = np.log(step_size)
                mu = np.log(10.0 * step_size)
                hbar.fill(0.0)
                log_eps_bar = log_eps.copy()
                dual_iteration = 0

            if in_warmup and (mass_update_iteration == 0 or transition > mass_update_iteration):
                displacement = current.detach().cpu().double().numpy() - previous.detach().cpu().double().numpy()
                scaled_jump = np.sum(displacement * displacement * mass_diag.detach().cpu().double().numpy(), axis=1)
                for replica in range(replica_count):
                    index = int(leapfrog_steps[replica]) - self.config.min_leapfrog_steps
                    leapfrog_score_sum[replica, index] += scaled_jump[replica] / float(leapfrog_steps[replica])
                    leapfrog_score_count[replica, index] += 1

            if transition % self.config.swap_interval == 0:
                accepted_edges = self._attempt_swaps(
                    current,
                    current_reference,
                    current_tempered,
                    labels,
                    swap_parity,
                )
                attempted_edges = np.zeros(replica_count - 1, dtype=np.int64)
                attempted_edges[swap_parity::2] = 1
                if in_warmup:
                    warmup_swap_attempts += attempted_edges
                    warmup_swap_accepts += accepted_edges
                else:
                    swap_attempts += attempted_edges
                    swap_accepts += accepted_edges
                swap_parity = 1 - swap_parity

            if in_warmup and transition == self.config.warmup_steps:
                step_size = self._bounded_step_sizes(np.exp(log_eps_bar))
                tuned_leapfrog = self._select_leapfrog_counts(
                    leapfrog_score_sum,
                    leapfrog_score_count,
                    tuned_leapfrog,
                )
                warmup_elapsed_seconds = time.monotonic() - started
                warmup_target_evaluations = self.target_evaluations
                sampling_started = time.monotonic()

            if not in_warmup:
                sampling_transition = transition - self.config.warmup_steps
                if sampling_transition % self.config.thinning == 0:
                    replica_samples.append(current.detach().cpu().double().numpy().copy())
                    reference_samples.append(current_reference.detach().cpu().double().numpy().copy())
                    tempered_samples.append(current_tempered.detach().cpu().double().numpy().copy())
                    label_samples.append(labels.detach().cpu().numpy().copy())
                    retained += 1

        if retained != self.config.samples:
            raise RuntimeError(f"expected {self.config.samples} retained samples, got {retained}")

        finished = time.monotonic()
        if self.config.warmup_steps == 0:
            sampling_started = started
            warmup_target_evaluations = 0
        sampling_elapsed_seconds = finished - sampling_started

        replica_z = np.stack(replica_samples, axis=0)
        replica_reference = np.stack(reference_samples, axis=0)
        replica_tempered = np.stack(tempered_samples, axis=0)
        replica_labels = np.stack(label_samples, axis=0)
        cold_log_prob = replica_reference[:, 0] + replica_tempered[:, 0]
        leapfrog_efficiency = self._selected_leapfrog_efficiency(
            leapfrog_score_sum,
            leapfrog_score_count,
            tuned_leapfrog,
        )
        return ParallelTemperedHMCResult(
            cold_z=replica_z[:, 0, :],
            cold_log_prob=cold_log_prob,
            replica_z=replica_z,
            replica_reference_log_prob=replica_reference,
            replica_tempered_log_prob=replica_tempered,
            replica_labels=replica_labels,
            inverse_temperatures=np.asarray(self.config.inverse_temperatures, dtype=np.float64),
            accept_rate=hmc_accepts / (self.config.samples * self.config.thinning),
            warmup_accept_rate=warmup_hmc_accepts / max(self.config.warmup_steps, 1),
            swap_accept_rate=self._rates(swap_accepts, swap_attempts),
            warmup_swap_accept_rate=self._rates(warmup_swap_accepts, warmup_swap_attempts),
            step_size=np.asarray(step_size, dtype=np.float64),
            leapfrog_steps=np.asarray(tuned_leapfrog, dtype=np.int64),
            mass_diag=mass_diag.detach().cpu().double().numpy(),
            leapfrog_efficiency=leapfrog_efficiency,
            final_z=current.detach().cpu().double().numpy(),
            elapsed_seconds=finished - started,
            warmup_elapsed_seconds=warmup_elapsed_seconds,
            sampling_elapsed_seconds=sampling_elapsed_seconds,
            target_evaluations=self.target_evaluations,
            warmup_target_evaluations=warmup_target_evaluations,
            sampling_target_evaluations=self.target_evaluations - warmup_target_evaluations,
        )

    def _components_and_grad(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_variable = z.detach().clone().requires_grad_(True)
        reference, tempered = self.log_density(z_variable)
        reference = reference.reshape(z_variable.shape[0])
        tempered = tempered.reshape(z_variable.shape[0])
        total = reference + self.beta * tempered
        gradient = torch.autograd.grad(total.sum(), z_variable)[0]
        reference = torch.nan_to_num(reference, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
        tempered = torch.nan_to_num(tempered, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
        total = torch.nan_to_num(total, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
        gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_evaluations += 1
        return reference.detach(), tempered.detach(), total.detach(), gradient.detach()

    def _leapfrog(
        self,
        z: torch.Tensor,
        momentum: torch.Tensor,
        initial_gradient: torch.Tensor,
        step_size: np.ndarray,
        leapfrog_steps: np.ndarray,
        inv_mass: torch.Tensor,
        deadline_monotonic: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_new = z.detach().clone()
        momentum_new = momentum.detach().clone()
        epsilon = torch.as_tensor(step_size, dtype=self.dtype, device=self.device).reshape(-1, 1)
        momentum_new += 0.5 * epsilon * initial_gradient
        reference = torch.full_like(self.beta, -float("inf"))
        tempered = torch.full_like(self.beta, -float("inf"))
        total = torch.full_like(self.beta, -float("inf"))

        maximum_steps = int(np.max(leapfrog_steps))
        leapfrog_tensor = torch.as_tensor(leapfrog_steps, dtype=torch.int64, device=self.device)
        for step in range(maximum_steps):
            self._check_deadline(deadline_monotonic)
            active = step < leapfrog_tensor
            z_new = z_new + active[:, None] * epsilon * momentum_new * inv_mass
            reference, tempered, total, gradient = self._components_and_grad(z_new)
            continuing = step < (leapfrog_tensor - 1)
            finishing = torch.eq(leapfrog_tensor - 1, step)
            momentum_new += continuing[:, None] * epsilon * gradient
            momentum_new += 0.5 * finishing[:, None] * epsilon * gradient
        return z_new.detach(), (-momentum_new).detach(), reference, tempered, total

    def _attempt_swaps(
        self,
        z: torch.Tensor,
        reference: torch.Tensor,
        tempered: torch.Tensor,
        labels: torch.Tensor,
        parity: int,
    ) -> np.ndarray:
        accepted_edges = np.zeros(len(self.beta) - 1, dtype=np.int64)
        for left in range(parity, len(self.beta) - 1, 2):
            right = left + 1
            log_acceptance = (self.beta[left] - self.beta[right]) * (tempered[right] - tempered[left])
            probability = torch.exp(torch.clamp(log_acceptance, max=0.0))
            probability = torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
            draw = torch.rand((), generator=self.generator, device=self.device, dtype=self.dtype)
            if bool(draw < probability):
                z_pair = z[[left, right]].clone()
                reference_pair = reference[[left, right]].clone()
                tempered_pair = tempered[[left, right]].clone()
                label_pair = labels[[left, right]].clone()
                z[left], z[right] = z_pair[1], z_pair[0]
                reference[left], reference[right] = reference_pair[1], reference_pair[0]
                tempered[left], tempered[right] = tempered_pair[1], tempered_pair[0]
                labels[left], labels[right] = label_pair[1], label_pair[0]
                accepted_edges[left] = 1
        return accepted_edges

    def _dual_average_update(
        self,
        iteration: int,
        acceptance_probability: np.ndarray,
        mu: np.ndarray,
        hbar: np.ndarray,
        log_eps_bar: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta = 1.0 / (iteration + self.config.t0)
        hbar = (1.0 - eta) * hbar + eta * (self.config.target_accept - acceptance_probability)
        log_eps = mu - math.sqrt(iteration) / self.config.gamma * hbar
        weight = iteration ** (-self.config.kappa)
        log_eps_bar = weight * log_eps + (1.0 - weight) * log_eps_bar
        return hbar, log_eps, log_eps_bar

    def _warmup_leapfrog_counts(self, iteration: int, replica_count: int) -> np.ndarray:
        span = self.config.max_leapfrog_steps - self.config.min_leapfrog_steps + 1
        replica = np.arange(replica_count, dtype=np.int64)
        offsets = (iteration - 1 + 3 * replica) % span
        return self.config.min_leapfrog_steps + offsets

    def _jittered_leapfrog_counts(self, tuned: np.ndarray) -> np.ndarray:
        counts = np.empty_like(tuned)
        for replica, center in enumerate(tuned):
            low = max(
                self.config.min_leapfrog_steps,
                int(round(center * (1.0 - self.config.leapfrog_jitter))),
            )
            high = min(
                self.config.max_leapfrog_steps,
                int(round(center * (1.0 + self.config.leapfrog_jitter))),
            )
            value = torch.randint(
                low,
                high + 1,
                (1,),
                generator=self.generator,
                device=self.device,
            )
            counts[replica] = int(value.item())
        return counts

    def _select_leapfrog_counts(
        self,
        score_sum: np.ndarray,
        score_count: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        selected = fallback.copy()
        for replica in range(len(selected)):
            eligible = score_count[replica] >= 2
            if not np.any(eligible):
                continue
            mean_score = np.full(score_sum.shape[1], -np.inf, dtype=np.float64)
            mean_score[eligible] = score_sum[replica, eligible] / score_count[replica, eligible]
            selected[replica] = self.config.min_leapfrog_steps + int(np.argmax(mean_score))
        return selected

    def _selected_leapfrog_efficiency(
        self,
        score_sum: np.ndarray,
        score_count: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        efficiency = np.full(len(selected), np.nan, dtype=np.float64)
        for replica, count in enumerate(selected):
            index = int(count) - self.config.min_leapfrog_steps
            if score_count[replica, index] > 0:
                efficiency[replica] = score_sum[replica, index] / score_count[replica, index]
        return efficiency

    def _initial_vector(
        self,
        values: np.ndarray | None,
        length: int,
        default: float,
        name: str,
    ) -> np.ndarray:
        result = np.full(length, default, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
        if result.shape != (length,) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{name} must contain {length} finite positive values")
        return result.copy()

    @staticmethod
    def _initial_matrix(
        values: np.ndarray | None,
        rows: int,
        columns: int,
        default: float,
        name: str,
    ) -> torch.Tensor:
        result = (
            np.full((rows, columns), default, dtype=np.float64)
            if values is None
            else np.asarray(values, dtype=np.float64)
        )
        if result.shape != (rows, columns) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{name} must have shape ({rows}, {columns}) with finite positive values")
        return torch.as_tensor(result, dtype=torch.float64)

    def _initial_integer_vector(
        self,
        values: np.ndarray | None,
        length: int,
        default: int,
    ) -> np.ndarray:
        result = np.full(length, default, dtype=np.int64) if values is None else np.asarray(values, dtype=np.int64)
        if result.shape != (length,):
            raise ValueError(f"initial_leapfrog_steps must contain {length} values")
        if np.any(result < self.config.min_leapfrog_steps) or np.any(result > self.config.max_leapfrog_steps):
            raise ValueError("initial_leapfrog_steps are outside the configured bounds")
        return result.copy()

    def _bounded_step_sizes(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=self.config.initial_step_size, posinf=10.0, neginf=1.0e-6)
        return np.clip(values, 1.0e-6, 10.0)

    @staticmethod
    def _kinetic(momentum: torch.Tensor, inv_mass: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(momentum * momentum * inv_mass, dim=-1)

    @staticmethod
    def _rates(accepts: np.ndarray, attempts: np.ndarray) -> np.ndarray:
        rates = np.full(accepts.shape, np.nan, dtype=np.float64)
        np.divide(accepts, attempts, out=rates, where=attempts > 0)
        return rates

    @staticmethod
    def _check_deadline(deadline_monotonic: float | None) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("parallel-tempered HMC reached the deadline")
