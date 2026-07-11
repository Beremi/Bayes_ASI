from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from bayes_asi.config import HMCConfig
from bayes_asi.priors import standard_normal_log_prob
from bayes_asi.types import DifferentiableSurrogate, Posterior


@dataclass(slots=True)
class HMCResult:
    z: np.ndarray
    theta: np.ndarray
    log_prob: np.ndarray
    accept_rate: float
    warmup_accept_rate: float
    step_size: float
    leapfrog_steps: int
    mass_diag: np.ndarray


class HMCSampler:
    """Custom HMC sampler for a differentiable surrogate posterior."""

    def __init__(
        self,
        config: HMCConfig,
        posterior: Posterior,
        surrogate: DifferentiableSurrogate,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.posterior = posterior
        self.surrogate = surrogate
        self.seed = int(seed)
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)

    def sample(self, initial_z: np.ndarray, *, deadline_monotonic: float | None = None) -> HMCResult:
        current = torch.as_tensor(initial_z, dtype=torch.float32, device=self.device)
        if current.ndim == 1:
            current = current.reshape(1, -1)
        n_chains, dim = current.shape
        if n_chains != self.config.n_chains:
            raise ValueError(f"expected {self.config.n_chains} HMC chains, got {n_chains}")
        if dim == 0 or not torch.isfinite(current).all():
            raise ValueError("initial_z must be a finite, non-empty array")

        mass_diag = torch.ones(dim, dtype=torch.float32, device=self.device)
        inv_mass = torch.ones_like(mass_diag)
        step_size = float(self.config.initial_step_size)
        log_eps = math.log(step_size)
        mu = math.log(10.0 * step_size)
        hbar = 0.0
        log_eps_bar = log_eps
        min_l = int(self.config.min_leapfrog_steps)
        max_l = int(self.config.max_leapfrog_steps)
        tuned_l = max(min_l, min(max_l, (min_l + max_l) // 2))

        warmup_accept = 0.0
        sample_accept = 0.0
        samples: list[np.ndarray] = []
        logps: list[np.ndarray] = []
        warmup_states: list[np.ndarray] = []

        total_iters = int(self.config.warmup_steps + self.config.samples_per_chain)
        for iteration in range(1, total_iters + 1):
            self._check_deadline(deadline_monotonic)
            in_warmup = iteration <= self.config.warmup_steps
            if in_warmup:
                leapfrog_steps = self._randint(min_l, max_l + 1)
            else:
                leapfrog_steps = self._jittered_leapfrog_count(tuned_l, min_l, max_l)

            current_logp, _ = self._log_prob_and_grad(current)
            if not torch.isfinite(current_logp).all():
                raise ValueError("current HMC state has non-finite log probability")
            momentum = torch.randn(current.shape, generator=self.generator, device=self.device) * torch.sqrt(mass_diag)
            proposal, proposal_momentum, proposal_logp = self._leapfrog(
                current,
                momentum,
                step_size,
                leapfrog_steps,
                inv_mass,
                deadline_monotonic,
            )
            current_h = -current_logp + self._kinetic(momentum, inv_mass)
            proposal_h = -proposal_logp + self._kinetic(proposal_momentum, inv_mass)
            log_accept = current_h - proposal_h
            accept_prob = torch.exp(torch.clamp(log_accept, max=0.0))
            accept_prob = torch.nan_to_num(accept_prob, nan=0.0, posinf=1.0, neginf=0.0)
            uniform = torch.rand((n_chains,), generator=self.generator, device=self.device)
            accept = uniform < accept_prob
            current = torch.where(accept[:, None], proposal.detach(), current.detach())
            mean_accepted = float(accept.float().mean().detach().cpu())

            if in_warmup:
                warmup_accept += mean_accepted
                mean_accept = float(accept_prob.mean().detach().cpu())
                hbar, log_eps, log_eps_bar = self._dual_average_update(iteration, mean_accept, mu, hbar, log_eps_bar)
                step_size = self._bounded_step_size(log_eps)
                warmup_states.append(current.detach().cpu().numpy())
                if iteration == self.config.warmup_steps:
                    step_size = self._bounded_step_size(log_eps_bar)
                    stacked = np.concatenate(warmup_states, axis=0)
                    variance = np.nan_to_num(np.var(stacked, axis=0), nan=1.0, posinf=1.0, neginf=1.0)
                    variance[variance < 1e-4] = 1.0
                    mass_diag = torch.as_tensor(variance, dtype=torch.float32, device=self.device)
                    inv_mass = 1.0 / mass_diag
                    tuned_l = self._tune_leapfrog_count(stacked, step_size, min_l, max_l)
            else:
                sample_accept += mean_accepted
                lp, _ = self._log_prob_and_grad(current)
                samples.append(current.detach().cpu().numpy())
                logps.append(lp.detach().cpu().numpy())

        z = np.concatenate(samples, axis=0)
        log_prob = np.concatenate(logps, axis=0)
        with torch.no_grad():
            theta = self.posterior.prior.to_theta(torch.as_tensor(z, dtype=torch.float32, device=self.device))
        return HMCResult(
            z=z,
            theta=theta.detach().cpu().numpy(),
            log_prob=log_prob,
            accept_rate=sample_accept / self.config.samples_per_chain,
            warmup_accept_rate=warmup_accept / max(self.config.warmup_steps, 1),
            step_size=step_size,
            leapfrog_steps=tuned_l,
            mass_diag=mass_diag.detach().cpu().numpy(),
        )

    def _log_prob_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_var = z.detach().clone().requires_grad_(True)
        theta = self.posterior.prior.to_theta(z_var)
        y_pred = self.surrogate(z_var)
        log_prob = standard_normal_log_prob(z_var) + self.posterior.log_likelihood(theta, y_pred)
        log_prob = log_prob.reshape(z_var.shape[0])
        grad = torch.autograd.grad(log_prob.sum(), z_var)[0]
        log_prob = torch.nan_to_num(log_prob, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob.detach(), grad.detach()

    def _leapfrog(
        self,
        z: torch.Tensor,
        momentum: torch.Tensor,
        step_size: float,
        leapfrog_steps: int,
        inv_mass: torch.Tensor,
        deadline_monotonic: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_new = z.detach().clone()
        p_new = momentum.detach().clone()
        _, grad = self._log_prob_and_grad(z_new)
        p_new = p_new + 0.5 * step_size * grad
        log_prob = torch.full((z.shape[0],), -float("inf"), dtype=torch.float32, device=self.device)
        for step in range(leapfrog_steps):
            self._check_deadline(deadline_monotonic)
            z_new = z_new + step_size * p_new * inv_mass
            log_prob, grad = self._log_prob_and_grad(z_new)
            if step != leapfrog_steps - 1:
                p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        return z_new.detach(), (-p_new).detach(), log_prob.detach()

    @staticmethod
    def _kinetic(momentum: torch.Tensor, inv_mass: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(momentum * momentum * inv_mass, dim=-1)

    def _dual_average_update(
        self,
        iteration: int,
        accept_prob: float,
        mu: float,
        hbar: float,
        log_eps_bar: float,
    ) -> tuple[float, float, float]:
        eta = 1.0 / (iteration + self.config.t0)
        hbar = (1.0 - eta) * hbar + eta * (self.config.target_accept - accept_prob)
        log_eps = mu - math.sqrt(iteration) / self.config.gamma * hbar
        weight = iteration ** (-self.config.kappa)
        log_eps_bar = weight * log_eps + (1.0 - weight) * log_eps_bar
        return hbar, log_eps, log_eps_bar

    @staticmethod
    def _bounded_step_size(log_step_size: float) -> float:
        return float(math.exp(min(math.log(10.0), max(math.log(1e-6), log_step_size))))

    @staticmethod
    def _check_deadline(deadline_monotonic: float | None) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("HMC reached the run deadline")

    def _tune_leapfrog_count(self, warmup_states: np.ndarray, step_size: float, min_l: int, max_l: int) -> int:
        variance = np.var(warmup_states, axis=0)
        radius = float(np.sqrt(max(np.mean(variance), 1e-6) * warmup_states.shape[1]))
        trajectory_length = max(step_size * min_l, min(step_size * max_l, radius))
        return int(max(min_l, min(max_l, round(trajectory_length / max(step_size, 1e-12)))))

    def _jittered_leapfrog_count(self, tuned_l: int, min_l: int, max_l: int) -> int:
        low = max(min_l, int(round(tuned_l * 0.8)))
        high = min(max_l, int(round(tuned_l * 1.2)) + 1)
        return self._randint(low, max(high, low + 1))

    def _randint(self, low: int, high: int) -> int:
        value = torch.randint(low, high, (1,), generator=self.generator, device=self.device)
        return int(value.item())
