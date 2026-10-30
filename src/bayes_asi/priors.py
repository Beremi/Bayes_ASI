from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy import special, stats

_LOG_TWO_PI = math.log(2.0 * math.pi)
_MIN_PROBABILITY = np.nextafter(np.float64(0.0), np.float64(1.0))


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_scalar(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def standard_normal_icdf(unit: np.ndarray) -> np.ndarray:
    """Map open-unit probabilities to independent standard-normal coordinates."""

    unit = np.asarray(unit, dtype=np.float64)
    if unit.ndim != 2:
        raise ValueError("unit must be a two-dimensional array")
    if not np.all(np.isfinite(unit)) or np.any(unit <= 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit values must be finite and strictly between zero and one")
    z = special.ndtri(unit)
    if not np.all(np.isfinite(z)):
        raise ValueError("standard-normal inverse CDF returned non-finite coordinates")
    return np.asarray(z, dtype=np.float64)


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Return the normalized log density of an independent standard normal."""

    return -0.5 * torch.sum(z * z + _LOG_TWO_PI, dim=-1)


class _QuantileTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, marginal: Marginal) -> torch.Tensor:
        theta, derivative = marginal._from_standard_normal_numpy(z.detach().cpu().double().numpy())
        derivative_tensor = torch.as_tensor(derivative, dtype=z.dtype, device=z.device)
        ctx.save_for_backward(derivative_tensor)
        return torch.as_tensor(theta, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (derivative,) = ctx.saved_tensors
        return grad_output * derivative, None


@dataclass(frozen=True, slots=True)
class Marginal:
    """A continuous scalar distribution used in a Gaussian-reference transport."""

    distribution: Any = field(repr=False)
    name: str
    _torch_transform: Callable[[torch.Tensor], torch.Tensor] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("marginal name must not be empty")
        object.__setattr__(self, "name", name)
        missing = [
            method
            for method in ("cdf", "sf", "ppf", "isf", "logpdf")
            if not callable(getattr(self.distribution, method, None))
        ]
        if missing:
            raise TypeError(f"marginal distribution is missing: {', '.join(missing)}")
        quantiles = np.asarray(self.distribution.ppf([0.25, 0.5, 0.75]), dtype=np.float64)
        if quantiles.shape != (3,) or not np.all(np.isfinite(quantiles)) or not np.all(np.diff(quantiles) > 0.0):
            raise ValueError("marginal must be a finite, non-degenerate continuous distribution")

    def from_standard_normal(self, z: torch.Tensor) -> torch.Tensor:
        if self._torch_transform is not None:
            return self._torch_transform(z)
        return _QuantileTransport.apply(z, self)

    def to_standard_normal(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name} parameter values must be finite")
        median = float(self.distribution.ppf(0.5))
        lower = values < median
        tail_probability = np.empty_like(values)
        tail_probability[lower] = np.asarray(self.distribution.cdf(values[lower]), dtype=np.float64)
        tail_probability[~lower] = np.asarray(self.distribution.sf(values[~lower]), dtype=np.float64)
        if np.any(tail_probability <= 0.0) or np.any(tail_probability > 0.5 + 1e-12) or not np.all(
            np.isfinite(tail_probability)
        ):
            raise ValueError(f"values lie outside the open support of the {self.name} prior")
        tail_probability = np.minimum(tail_probability, 0.5)
        magnitude = -special.ndtri(np.maximum(tail_probability, _MIN_PROBABILITY))
        return np.where(lower, -magnitude, magnitude)

    def _from_standard_normal_numpy(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            raise ValueError("standard-normal coordinates must be finite")
        lower = z < 0.0
        tail_probability = np.maximum(special.ndtr(-np.abs(z)), _MIN_PROBABILITY)
        theta = np.empty_like(z)
        theta[lower] = np.asarray(self.distribution.ppf(tail_probability[lower]), dtype=np.float64)
        theta[~lower] = np.asarray(self.distribution.isf(tail_probability[~lower]), dtype=np.float64)
        log_derivative = -0.5 * z * z - 0.5 * _LOG_TWO_PI - np.asarray(
            self.distribution.logpdf(theta), dtype=np.float64
        )
        derivative = np.exp(np.clip(log_derivative, -745.0, 709.0))
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(derivative)) or np.any(derivative <= 0.0):
            raise ValueError(f"{self.name} prior transform returned non-finite values or derivatives")
        return theta, derivative


def scipy_marginal(
    distribution: Any,
    *,
    name: str | None = None,
    _torch_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Marginal:
    """Wrap any frozen SciPy-compatible continuous distribution."""

    inferred = getattr(getattr(distribution, "dist", None), "name", None)
    return Marginal(
        distribution=distribution,
        name=str(name or inferred or type(distribution).__name__),
        _torch_transform=_torch_transform,
    )


def uniform(low: float, high: float) -> Marginal:
    low = _finite_scalar(low, "low")
    high = _finite_scalar(high, "high")
    if high <= low:
        raise ValueError("high must be greater than low")
    span = high - low
    return scipy_marginal(
        stats.uniform(loc=low, scale=span),
        name="uniform",
        _torch_transform=lambda z: low + span * torch.special.ndtr(z),
    )


def beta(alpha: float, beta: float, *, low: float = 0.0, high: float = 1.0) -> Marginal:
    alpha = _positive_scalar(alpha, "alpha")
    beta = _positive_scalar(beta, "beta")
    low = _finite_scalar(low, "low")
    high = _finite_scalar(high, "high")
    if high <= low:
        raise ValueError("high must be greater than low")
    return scipy_marginal(stats.beta(alpha, beta, loc=low, scale=high - low), name="beta")


def normal(mean: float = 0.0, sd: float = 1.0) -> Marginal:
    mean = _finite_scalar(mean, "mean")
    sd = _positive_scalar(sd, "sd")
    return scipy_marginal(
        stats.norm(loc=mean, scale=sd),
        name="normal",
        _torch_transform=lambda z: mean + sd * z,
    )


def truncated_normal(mean: float, sd: float, low: float, high: float) -> Marginal:
    """Normal distribution conditioned on a finite open interval."""

    mean = _finite_scalar(mean, "mean")
    sd = _positive_scalar(sd, "sd")
    low = _finite_scalar(low, "low")
    high = _finite_scalar(high, "high")
    if high <= low:
        raise ValueError("high must be greater than low")
    return scipy_marginal(
        stats.truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd),
        name="truncated_normal",
    )


def exponential(rate: float = 1.0, *, shift: float = 0.0) -> Marginal:
    rate = _positive_scalar(rate, "rate")
    shift = _finite_scalar(shift, "shift")
    return scipy_marginal(
        stats.expon(loc=shift, scale=1.0 / rate),
        name="exponential",
        _torch_transform=lambda z: shift - torch.special.log_ndtr(-z) / rate,
    )


def lognormal(log_mean: float = 0.0, log_sd: float = 1.0, *, shift: float = 0.0) -> Marginal:
    log_mean = _finite_scalar(log_mean, "log_mean")
    log_sd = _positive_scalar(log_sd, "log_sd")
    shift = _finite_scalar(shift, "shift")
    return scipy_marginal(
        stats.lognorm(s=log_sd, loc=shift, scale=math.exp(log_mean)),
        name="lognormal",
        _torch_transform=lambda z: shift + torch.exp(log_mean + log_sd * z),
    )


def gamma(concentration: float, rate: float = 1.0, *, shift: float = 0.0) -> Marginal:
    concentration = _positive_scalar(concentration, "concentration")
    rate = _positive_scalar(rate, "rate")
    shift = _finite_scalar(shift, "shift")
    return scipy_marginal(
        stats.gamma(concentration, loc=shift, scale=1.0 / rate),
        name="gamma",
    )


def weibull(shape: float, scale: float = 1.0, *, shift: float = 0.0) -> Marginal:
    shape = _positive_scalar(shape, "shape")
    scale = _positive_scalar(scale, "scale")
    shift = _finite_scalar(shift, "shift")
    return scipy_marginal(
        stats.weibull_min(shape, loc=shift, scale=scale),
        name="weibull",
        _torch_transform=lambda z: shift + scale * (-torch.special.log_ndtr(-z)).pow(1.0 / shape),
    )


def student_t(df: float, *, loc: float = 0.0, scale: float = 1.0) -> Marginal:
    df = _positive_scalar(df, "df")
    loc = _finite_scalar(loc, "loc")
    scale = _positive_scalar(scale, "scale")
    return scipy_marginal(stats.t(df, loc=loc, scale=scale), name="student_t")


def laplace(loc: float = 0.0, scale: float = 1.0) -> Marginal:
    loc = _finite_scalar(loc, "loc")
    scale = _positive_scalar(scale, "scale")
    log_two = math.log(2.0)
    return scipy_marginal(
        stats.laplace(loc=loc, scale=scale),
        name="laplace",
        _torch_transform=lambda z: loc
        + torch.where(
            z < 0.0,
            scale * (log_two + torch.special.log_ndtr(z)),
            -scale * (log_two + torch.special.log_ndtr(-z)),
        ),
    )


def logistic(loc: float = 0.0, scale: float = 1.0) -> Marginal:
    loc = _finite_scalar(loc, "loc")
    scale = _positive_scalar(scale, "scale")
    return scipy_marginal(
        stats.logistic(loc=loc, scale=scale),
        name="logistic",
        _torch_transform=lambda z: loc
        + scale * (torch.special.log_ndtr(z) - torch.special.log_ndtr(-z)),
    )


_MARGINAL_FACTORIES: dict[str, Callable[..., Marginal]] = {
    "beta": beta,
    "exponential": exponential,
    "gamma": gamma,
    "laplace": laplace,
    "logistic": logistic,
    "lognormal": lognormal,
    "normal": normal,
    "truncated_normal": truncated_normal,
    "student_t": student_t,
    "uniform": uniform,
    "weibull": weibull,
}


def marginal_from_spec(specification: Mapping[str, Any]) -> Marginal:
    """Build a standard marginal from a serializable distribution specification."""

    data = {str(key): value for key, value in specification.items()}
    distribution = str(data.pop("distribution", "")).strip().lower()
    if distribution not in _MARGINAL_FACTORIES:
        supported = ", ".join(sorted(_MARGINAL_FACTORIES))
        raise ValueError(f"unknown distribution {distribution!r}; supported distributions: {supported}")
    return _MARGINAL_FACTORIES[distribution](**data)


@dataclass(frozen=True, slots=True)
class IndependentPrior:
    """Product prior represented as a map from independent standard normals."""

    marginals: tuple[Marginal, ...]

    def __init__(self, marginals: Sequence[Marginal | Mapping[str, Any]]) -> None:
        built = tuple(
            item if isinstance(item, Marginal) else marginal_from_spec(item)
            for item in marginals
        )
        if not built:
            raise ValueError("an independent prior requires at least one marginal")
        object.__setattr__(self, "marginals", built)

    @property
    def latent_dim(self) -> int:
        return len(self.marginals)

    @property
    def parameter_dim(self) -> int:
        return len(self.marginals)

    def to_theta(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 0 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have trailing dimension {self.latent_dim}")
        return torch.stack(
            [marginal.from_standard_normal(z[..., index]) for index, marginal in enumerate(self.marginals)],
            dim=-1,
        )

    def to_standard_normal(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 0 or theta.shape[-1] != self.parameter_dim:
            raise ValueError(f"theta must have trailing dimension {self.parameter_dim}")
        return np.stack(
            [marginal.to_standard_normal(theta[..., index]) for index, marginal in enumerate(self.marginals)],
            axis=-1,
        )


@dataclass(frozen=True, slots=True)
class GaussianCopulaPrior:
    """Correlated prior with arbitrary marginals and a Gaussian copula."""

    marginals: tuple[Marginal, ...]
    correlation: np.ndarray = field(repr=False)
    _cholesky: np.ndarray = field(init=False, repr=False)

    def __init__(self, marginals: Sequence[Marginal], correlation: np.ndarray) -> None:
        built = tuple(marginals)
        if not built:
            raise ValueError("a Gaussian-copula prior requires at least one marginal")
        correlation = np.asarray(correlation, dtype=np.float64)
        expected = (len(built), len(built))
        if correlation.shape != expected or not np.all(np.isfinite(correlation)):
            raise ValueError(f"correlation must be a finite array with shape {expected}")
        if not np.allclose(correlation, correlation.T, rtol=1e-12, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("correlation diagonal must contain ones")
        try:
            cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation must be positive definite") from exc
        object.__setattr__(self, "marginals", built)
        object.__setattr__(self, "correlation", correlation.copy())
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def latent_dim(self) -> int:
        return len(self.marginals)

    @property
    def parameter_dim(self) -> int:
        return len(self.marginals)

    def to_theta(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 0 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have trailing dimension {self.latent_dim}")
        cholesky = torch.as_tensor(self._cholesky, dtype=z.dtype, device=z.device)
        correlated = z @ cholesky.T
        return torch.stack(
            [marginal.from_standard_normal(correlated[..., index]) for index, marginal in enumerate(self.marginals)],
            dim=-1,
        )

    def to_standard_normal(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 0 or theta.shape[-1] != self.parameter_dim:
            raise ValueError(f"theta must have trailing dimension {self.parameter_dim}")
        correlated = np.stack(
            [marginal.to_standard_normal(theta[..., index]) for index, marginal in enumerate(self.marginals)],
            axis=-1,
        )
        flat = correlated.reshape(-1, self.parameter_dim)
        independent = np.linalg.solve(self._cholesky, flat.T).T
        return independent.reshape(*theta.shape[:-1], self.latent_dim)
