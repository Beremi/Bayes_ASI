"""Differentiable likelihoods and the legacy profiled ERT calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _finite_tensor(value: Any, name: str) -> torch.Tensor:
    # Keep problem constants in double precision; callers may still evaluate
    # in float32 and all constants are cast at the operation boundary.
    out = torch.as_tensor(value, dtype=torch.float64)
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


class GaussianLikelihood:
    """Weighted, normalized independent Gaussian log likelihood."""

    def __init__(self, observed: Any, sigma: Any, weights: Any | None = None, power_weight: float = 1.0) -> None:
        self.observed = _finite_tensor(observed, "observed")
        if self.observed.ndim != 1:
            raise ValueError("observed must be one-dimensional")
        self.sigma = _finite_tensor(sigma, "sigma")
        if self.sigma.ndim not in (0, 1) or (self.sigma.ndim == 1 and self.sigma.numel() != self.observed.numel()):
            raise ValueError("sigma must be scalar or match observed")
        if torch.any(self.sigma <= 0):
            raise ValueError("sigma must be positive")
        if weights is None:
            self.weights = torch.ones_like(self.observed)
        else:
            self.weights = _finite_tensor(weights, "weights")
            if (
                self.weights.shape != self.observed.shape
                or torch.any(self.weights < 0)
                or not torch.any(self.weights > 0)
            ):
                raise ValueError("weights must match observed, be nonnegative, and not all zero")
        if not np.isfinite(power_weight) or power_weight <= 0:
            raise ValueError("power_weight must be positive and finite")
        self.power_weight = float(power_weight)

    def __call__(self, y_pred: Any) -> torch.Tensor:
        y = torch.as_tensor(y_pred)
        if y.ndim == 0 or y.shape[-1] != self.observed.numel():
            raise ValueError("y_pred must have trailing dimension matching observed")
        obs = self.observed.to(device=y.device, dtype=y.dtype)
        sig = self.sigma.to(device=y.device, dtype=y.dtype)
        weights = self.weights.to(device=y.device, dtype=y.dtype)
        if not torch.isfinite(y).all():
            raise ValueError("y_pred must be finite")
        terms = -0.5 * ((y - obs) / sig) ** 2 - 0.5 * torch.log(2.0 * torch.pi * sig**2)
        return self.power_weight * torch.sum(weights * terms, dim=-1)


class ProfiledAffineLikelihood:
    """Box-constrained profiled ``y ~= a + b*x`` Gaussian likelihood.

    This is a profiled conditional likelihood, not a marginalized posterior.
    """

    def __init__(
        self, observed: Any, sigma: Any, *, a_bounds: tuple[float, float], b_bounds: tuple[float, float],
        per_support: bool = True, effective_rows: float | None = None, power_weight: float = 1.0,
        likelihood_mode: str = "raw_log10", model_days: Any | None = None,
        coefficient_fixed_c_d_abe: tuple[float, float] | None = None,
        coefficient_sigmas: Any | None = None,
    ) -> None:
        self.observed = _finite_tensor(observed, "observed")
        if self.observed.ndim != 2 or self.observed.shape[0] == 0 or self.observed.shape[1] < 2:
            raise ValueError("observed must have shape (S,T), T >= 2")
        self.sigma = _finite_tensor(sigma, "sigma")
        if self.sigma.ndim != 0:
            raise ValueError("sigma must be a positive scalar")
        if torch.any(self.sigma <= 0):
            raise ValueError("sigma must be positive")
        self.a_bounds = self._bounds(a_bounds, "a_bounds")
        self.b_bounds = self._bounds(b_bounds, "b_bounds")
        self.per_support = bool(per_support)
        self.effective_rows = float(self.observed.numel() if effective_rows is None else effective_rows)
        if not np.isfinite(self.effective_rows) or self.effective_rows <= 0:
            raise ValueError("effective_rows must be positive and finite")
        if not np.isfinite(power_weight) or power_weight <= 0:
            raise ValueError("power_weight must be positive and finite")
        self.power_weight = float(power_weight)
        if likelihood_mode not in {"raw_log10", "coefficient_fixed_c_d_abe"}:
            raise ValueError("unsupported likelihood_mode")
        self.likelihood_mode = likelihood_mode
        if likelihood_mode == "coefficient_fixed_c_d_abe":
            if model_days is None or coefficient_fixed_c_d_abe is None or coefficient_sigmas is None:
                raise ValueError("coefficient mode requires model_days, fixed c/d, and coefficient_sigmas")
            days = _finite_tensor(model_days, "model_days")
            if days.ndim != 1 or days.numel() != self.observed.shape[1]:
                raise ValueError("model_days must match T")
            c, d = coefficient_fixed_c_d_abe
            if not np.isfinite(c) or not np.isfinite(d):
                raise ValueError("fixed c,d must be finite")
            tt = days - torch.min(days)
            design = torch.stack((torch.ones_like(tt), torch.sin(float(c) * tt + float(d)), tt), dim=1)
            if torch.linalg.matrix_rank(design) < 3:
                raise ValueError("coefficient design must have rank 3")
            cs = _finite_tensor(coefficient_sigmas, "coefficient_sigmas")
            if cs.shape != (3,) or torch.any(cs <= 0):
                raise ValueError("coefficient_sigmas must be positive length 3")
            self.coefficient_pinv_t = torch.linalg.pinv(design).T
            self.observed_coefficients = self.observed @ self.coefficient_pinv_t
            self.coefficient_sigmas = cs

    @staticmethod
    def _bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not bounds[0] < bounds[1]:
            raise ValueError(f"{name} must be finite and increasing")
        return float(bounds[0]), float(bounds[1])

    def _fit_group(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amin, amax = self.a_bounds
        bmin, bmax = self.b_bounds
        n = x.shape[-1]
        mx, my = torch.mean(x, -1), torch.mean(y, -1)
        dx, dy = x - mx[..., None], y - my[..., None]
        sx, sy, sxy = torch.sum(x, -1), torch.sum(y, -1), torch.sum(x * y, -1)
        sxx = torch.sum(x * x, -1)
        den = torch.sum(dx * dx, -1)
        safe = torch.where(den > torch.finfo(x.dtype).eps, den, torch.ones_like(den))
        b0 = torch.sum(dx * dy, -1) / safe
        a0 = my - b0 * mx
        interior = (den > torch.finfo(x.dtype).eps) & (a0 > amin) & (a0 < amax) & (b0 > bmin) & (b0 < bmax)
        candidates = [(a0, b0, interior)]
        for a in (amin, amax):
            aa = torch.as_tensor(a, dtype=x.dtype, device=x.device)
            bb = ((sxy - aa * sx) / torch.where(sxx > 0, sxx, torch.ones_like(sxx))).clamp(bmin, bmax)
            candidates.append((aa.expand_as(bb), bb, torch.ones_like(bb, dtype=torch.bool)))
        for b in (bmin, bmax):
            bb = torch.as_tensor(b, dtype=x.dtype, device=x.device)
            aa = ((sy - bb * sx) / n).clamp(amin, amax)
            candidates.append((aa, bb.expand_as(aa), torch.ones_like(aa, dtype=torch.bool)))
        # Explicit corners protect the constant-x and zero-variance cases.
        for a in (amin, amax):
            for b in (bmin, bmax):
                candidates.append((torch.as_tensor(a, dtype=x.dtype, device=x.device).expand_as(x[..., 0]),
                                   torch.as_tensor(b, dtype=x.dtype, device=x.device).expand_as(x[..., 0]),
                                   torch.ones_like(x[..., 0], dtype=torch.bool)))
        losses = torch.stack([
            torch.where(valid, torch.sum((aa[..., None] + bb[..., None] * x - y) ** 2, -1), torch.inf)
            for aa, bb, valid in candidates
        ], -1)
        idx = torch.argmin(losses, -1)
        aa = torch.stack([v[0] for v in candidates], -1).gather(-1, idx[..., None]).squeeze(-1)
        bb = torch.stack([v[1] for v in candidates], -1).gather(-1, idx[..., None]).squeeze(-1)
        return aa, bb

    def fit(self, y_pred: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(y_pred)
        if x.ndim == 0 or x.shape[-1] != self.observed.numel():
            raise ValueError("y_pred must have trailing dimension S*T")
        if not torch.isfinite(x).all():
            raise ValueError("y_pred must be finite")
        batch = x.shape[:-1]
        xx = x.reshape(batch + tuple(self.observed.shape))
        obs = self.observed.to(device=x.device, dtype=x.dtype)
        aa_list, bb_list = [], []
        if self.per_support:
            for s in range(self.observed.shape[0]):
                aa, bb = self._fit_group(xx[..., s, :], obs[s].expand(xx.shape[:-2] + (obs.shape[1],)))
                aa_list.append(aa)
                bb_list.append(bb)
            a, b = torch.stack(aa_list, -1), torch.stack(bb_list, -1)
        else:
            flat_shape = batch + (self.observed.numel(),)
            flat_obs = obs.reshape(-1)
            a0, b0 = self._fit_group(xx.reshape(flat_shape), flat_obs.expand(xx.shape[:-2] + (self.observed.numel(),)))
            support_shape = batch + (self.observed.shape[0],)
            a, b = a0[..., None].expand(support_shape), b0[..., None].expand(support_shape)
        pred = a[..., None] + b[..., None] * xx
        return pred, a, b

    def __call__(self, y_pred: Any) -> torch.Tensor:
        pred, _, _ = self.fit(y_pred)
        obs = self.observed.to(device=pred.device, dtype=pred.dtype)
        if self.likelihood_mode == "coefficient_fixed_c_d_abe":
            coeff = pred @ self.coefficient_pinv_t.to(device=pred.device, dtype=pred.dtype)
            residual = coeff - self.observed_coefficients.to(device=pred.device, dtype=pred.dtype)
            sig = self.coefficient_sigmas.to(device=pred.device, dtype=pred.dtype)
            return torch.sum(-0.5 * (residual / sig) ** 2 - 0.5 * torch.log(2 * torch.pi * sig**2), dim=(-2, -1))
        sig = self.sigma.to(device=pred.device, dtype=pred.dtype)
        weights = torch.full_like(pred, self.effective_rows / self.observed.numel())
        terms = -0.5 * ((pred - obs) / sig) ** 2 - 0.5 * torch.log(2 * torch.pi * sig**2)
        return self.power_weight * torch.sum(weights * terms, dim=(-2, -1))


__all__ = ["GaussianLikelihood", "ProfiledAffineLikelihood"]
