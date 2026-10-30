"""Small, bounded field parameterizations used by problem adapters.

``basis`` in :class:`KLField` is the *scaled* KL basis: its columns already
contain the square roots of the retained covariance eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class KLField:
    """Evaluate a finite Karhunen--Loeve expansion.

    ``mean`` has shape ``(n_locations,)`` and ``basis`` has shape
    ``(n_locations, n_modes)``.  ``evaluate`` accepts any leading batch shape
    followed by ``n_modes``.
    """

    mean: NDArray[np.float64]
    basis: NDArray[np.float64]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        basis = np.asarray(self.basis, dtype=float)
        if mean.ndim != 1 or basis.ndim != 2 or basis.shape[0] != mean.size:
            raise ValueError("mean must be (n_locations,), basis (n_locations, n_modes)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(basis)):
            raise ValueError("mean and basis must be finite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "basis", basis)

    @property
    def n_locations(self) -> int:
        return self.mean.size

    @property
    def n_modes(self) -> int:
        return self.basis.shape[1]

    def evaluate(self, xi: ArrayLike) -> NDArray[np.float64]:
        """Return ``mean + basis @ xi`` for arbitrary leading batch dimensions."""
        x = np.asarray(xi, dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.n_modes:
            raise ValueError(f"xi must have trailing dimension {self.n_modes}")
        if not np.all(np.isfinite(x)):
            raise ValueError("xi must be finite")
        return self.mean + np.einsum("...m,nm->...n", x, self.basis)

    def evaluate_torch(self, xi: Any) -> Any:
        """Torch equivalent of :meth:`evaluate`, preserving gradient/device."""
        import torch

        if not isinstance(xi, torch.Tensor) or xi.ndim == 0 or xi.shape[-1] != self.n_modes:
            raise ValueError(f"xi must be a torch tensor with trailing dimension {self.n_modes}")
        if not torch.isfinite(xi).all():
            raise ValueError("xi must be finite")
        mean = torch.as_tensor(self.mean, dtype=xi.dtype, device=xi.device)
        basis = torch.as_tensor(self.basis, dtype=xi.dtype, device=xi.device)
        return mean + torch.matmul(xi, basis.T)

    @classmethod
    def from_covariance(
        cls, mean: ArrayLike, covariance: ArrayLike, n_modes: int | None = None,
        *, rtol: float = 1e-10,
    ) -> KLField:
        """Construct from a small symmetric PSD covariance matrix.

        Eigenpairs are ordered descending.  Eigenvalues below the numerical
        tolerance are omitted; materially negative eigenvalues are rejected.
        """
        mu = np.asarray(mean, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        if mu.ndim != 1 or mu.size == 0 or cov.ndim != 2 or cov.shape != (mu.size, mu.size):
            raise ValueError("mean/covariance dimensions do not agree")
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError("rtol must be finite and nonnegative")
        if n_modes is not None and n_modes < 0:
            raise ValueError("n_modes must be nonnegative")
        if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(mu)):
            raise ValueError("mean and covariance must be finite")
        scale = float(np.max(np.abs(cov)))
        if scale == 0.0:
            return cls(mu, np.empty((mu.size, 0), dtype=float))
        if not np.allclose(cov, cov.T, rtol=rtol, atol=rtol * scale):
            raise ValueError("covariance must be symmetric")
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
        neg_tol = rtol * scale
        if vals[0] < -neg_tol:
            raise ValueError("covariance has materially negative eigenvalues")
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        keep = vals > neg_tol
        if n_modes is not None:
            keep[np.arange(vals.size) >= n_modes] = False
        basis = vecs[:, keep] * np.sqrt(np.maximum(vals[keep], 0.0))
        return cls(mu, basis)

    @classmethod
    def from_npz(cls, path: str | Path, basis_key: str = "basis", mean: ArrayLike = 0.0) -> KLField:
        """Load an externally prepared (already scaled) basis from NPZ."""
        with np.load(path, allow_pickle=False) as data:
            if basis_key not in data:
                raise KeyError(basis_key)
            basis = np.asarray(data[basis_key], dtype=float)
        if basis.ndim != 2:
            raise ValueError("basis must be a two-dimensional array")
        mu = (
            np.full(basis.shape[0], float(np.asarray(mean).item()))
            if np.ndim(mean) == 0
            else np.asarray(mean, dtype=float)
        )
        return cls(mu, basis)

    def bounded(
        self, xi: ArrayLike, lower: float, upper: float, weights: ArrayLike,
        target: float | ArrayLike | None = None, *, logit_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Logistic field with an exact weighted mean target.

        This is a nonlinear transform, not a Gaussian KL marginal.  ``target``
        may be scalar or one value per leading batch item; by default it is the
        weighted mean of ``mean``.  The drive is centered after adding the full
        mean, so this default does not imply a spatially constant zero-mode field.
        """
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError("require finite lower < upper")
        if not np.isfinite(logit_scale):
            raise ValueError("logit_scale must be finite")
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size != self.n_locations or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must be finite and strictly positive")
        x = self.evaluate(xi)
        drive = x - np.average(x, axis=-1, weights=w, keepdims=True)
        with np.errstate(over="raise", invalid="raise"):
            scaled_drive = logit_scale * drive
        if not np.all(np.isfinite(scaled_drive)):
            raise ValueError("scaled drive must be finite")
        batch = drive.shape[:-1]
        if target is None:
            t = np.asarray(np.average(self.mean, weights=w))
        else:
            t = np.asarray(target, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t <= lower) or np.any(t >= upper):
            raise ValueError("target must be finite and strictly inside bounds")
        t = np.broadcast_to(t, batch)
        z = (t - lower) / (upper - lower)
        out = np.empty_like(drive)
        flat_d, flat_z = drive.reshape((-1, self.n_locations)), z.reshape(-1)
        flat_out = out.reshape((-1, self.n_locations))
        for i, d in enumerate(flat_d):
            zi = float(flat_z[i])
            lo, hi = -1.0, 1.0

            def f(beta: float, d=d, zi=zi) -> float:
                from scipy.special import expit

                s = expit(beta + logit_scale * d)
                return float(np.dot(w, s) / w.sum() - zi)

            for _ in range(1024):
                if f(lo) <= 0.0:
                    break
                lo *= 2.0
            else:
                raise ValueError("could not bracket bounded-field mean")
            for _ in range(1024):
                if f(hi) >= 0.0:
                    break
                hi *= 2.0
            else:
                raise ValueError("could not bracket bounded-field mean")
            for _ in range(90):
                mid = (lo + hi) / 2.0
                if f(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            from scipy.special import expit

            flat_out[i] = lower + (upper - lower) * expit((lo + hi) / 2.0 + logit_scale * d)
        return out

    bounded_field = bounded


def rotated_permeability_tensor(log10_k: ArrayLike, angle_deg: ArrayLike, ratio: ArrayLike) -> NDArray[np.float64]:
    """Return ``R diag(10**log10_k, ratio*10**log10_k) R.T`` (CCW angles)."""
    logk, angle, q = np.broadcast_arrays(
        np.asarray(log10_k, float), np.asarray(angle_deg, float), np.asarray(ratio, float)
    )
    if not np.all(np.isfinite(logk)) or not np.all(np.isfinite(angle)) or not np.all(np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("log10_k, angle_deg and ratio must be finite; ratio > 0")
    try:
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            k = np.power(10.0, logk)
            principal = np.stack((k, q * k), axis=-1)
    except FloatingPointError as exc:
        raise ValueError("permeability principal values are not representable") from exc
    if not np.all(np.isfinite(principal)) or np.any(principal <= 0):
        raise ValueError("permeability principal values must be finite and strictly positive")
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    out = np.empty(logk.shape + (2, 2), dtype=float)
    out[..., 0, 0] = k * (c * c + q * s * s)
    out[..., 1, 1] = k * (s * s + q * c * c)
    out[..., 0, 1] = out[..., 1, 0] = k * (1.0 - q) * c * s
    if not np.all(np.isfinite(out)):
        raise ValueError("permeability tensor is not finite")
    return out


__all__ = ["KLField", "rotated_permeability_tensor"]
