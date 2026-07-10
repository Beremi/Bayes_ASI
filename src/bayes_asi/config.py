from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown {section} option(s): {names}")


def _get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def _as_bool(value: Any, option: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{option} must be a boolean")
    return value


@dataclass(slots=True)
class FactoryConfig:
    factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.factory = self.factory.strip()
        if not self.factory:
            raise ValueError("factory path must not be empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FactoryConfig":
        data = _mapping(raw, "factory section")
        _reject_unknown(data, {"factory", "kwargs"}, "factory")
        if "factory" not in data:
            raise ValueError("factory section must contain a 'factory' dotted path")
        kwargs = _mapping(data.get("kwargs", {}), "factory.kwargs")
        return cls(factory=str(data["factory"]), kwargs=kwargs)


@dataclass(slots=True)
class RunConfig:
    output_dir: Path
    time_budget_seconds: float = 3600.0
    drain_running_on_exit: bool = True
    seed: int = 12345
    status_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.time_budget_seconds <= 0.0:
            raise ValueError("run.time_budget_seconds must be positive")
        if self.status_interval_seconds <= 0.0:
            raise ValueError("run.status_interval_seconds must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunConfig":
        data = _mapping(raw, "run")
        _reject_unknown(
            data,
            {"output_dir", "time_budget_seconds", "drain_running_on_exit", "seed", "status_interval_seconds"},
            "run",
        )
        output_dir = Path(_get(data, "output_dir", "runs/default")).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        return cls(
            output_dir=output_dir.resolve(),
            time_budget_seconds=float(_get(data, "time_budget_seconds", 3600.0)),
            drain_running_on_exit=_as_bool(_get(data, "drain_running_on_exit", True), "run.drain_running_on_exit"),
            seed=int(_get(data, "seed", 12345)),
            status_interval_seconds=float(_get(data, "status_interval_seconds", 5.0)),
        )


@dataclass(slots=True)
class ProblemConfig:
    latent_dim: int
    parameter_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        for name in ("latent_dim", "parameter_dim", "output_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"problem.{name} must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProblemConfig":
        data = _mapping(raw, "problem")
        _reject_unknown(data, {"latent_dim", "parameter_dim", "output_dim"}, "problem")
        missing = sorted({"latent_dim", "parameter_dim", "output_dim"} - set(data))
        if missing:
            raise ValueError(f"missing problem option(s): {', '.join(missing)}")
        return cls(
            latent_dim=int(data["latent_dim"]),
            parameter_dim=int(data["parameter_dim"]),
            output_dim=int(data["output_dim"]),
        )


@dataclass(slots=True)
class InitialDesignConfig:
    n_points: int = 16

    def __post_init__(self) -> None:
        if self.n_points < 0:
            raise ValueError("initial_design.n_points must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InitialDesignConfig":
        data = _mapping(raw, "initial_design")
        _reject_unknown(data, {"n_points"}, "initial_design")
        return cls(n_points=int(_get(data, "n_points", 16)))


@dataclass(slots=True)
class SurrogateConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 500
    batch_size: int = 128
    validation_fraction: float = 0.2
    patience: int = 50
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    min_train_samples: int = 16
    update_after_new_samples: int = 4
    retrain_from_scratch_every: int = 5
    scratch_improvement_rel: float = 0.02
    device: str = "cpu"
    gradient_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if any(width <= 0 for width in self.hidden_layers):
            raise ValueError("surrogate.hidden_layers widths must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("surrogate.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("surrogate.weight_decay must be non-negative")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("surrogate max_epochs, batch_size, and patience must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("surrogate.validation_fraction must be in [0, 1)")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("surrogate.scheduler_factor must be in (0, 1)")
        if self.scheduler_patience < 0:
            raise ValueError("surrogate.scheduler_patience must be non-negative")
        if self.min_train_samples < 2:
            raise ValueError("surrogate.min_train_samples must be at least 2")
        if self.update_after_new_samples <= 0:
            raise ValueError("surrogate.update_after_new_samples must be positive")
        if self.retrain_from_scratch_every < 0:
            raise ValueError("surrogate.retrain_from_scratch_every must be non-negative")
        if not 0.0 <= self.scratch_improvement_rel < 1.0:
            raise ValueError("surrogate.scratch_improvement_rel must be in [0, 1)")
        if not self.device.strip():
            raise ValueError("surrogate.device must not be empty")
        if self.gradient_epsilon <= 0.0:
            raise ValueError("surrogate.gradient_epsilon must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SurrogateConfig":
        data = _mapping(raw, "surrogate")
        _reject_unknown(data, set(cls.__dataclass_fields__), "surrogate")
        return cls(
            hidden_layers=[int(value) for value in _get(data, "hidden_layers", [64, 64])],
            learning_rate=float(_get(data, "learning_rate", 1e-3)),
            weight_decay=float(_get(data, "weight_decay", 0.0)),
            max_epochs=int(_get(data, "max_epochs", 500)),
            batch_size=int(_get(data, "batch_size", 128)),
            validation_fraction=float(_get(data, "validation_fraction", 0.2)),
            patience=int(_get(data, "patience", 50)),
            scheduler_factor=float(_get(data, "scheduler_factor", 0.5)),
            scheduler_patience=int(_get(data, "scheduler_patience", 20)),
            min_train_samples=int(_get(data, "min_train_samples", 16)),
            update_after_new_samples=int(_get(data, "update_after_new_samples", 4)),
            retrain_from_scratch_every=int(_get(data, "retrain_from_scratch_every", 5)),
            scratch_improvement_rel=float(_get(data, "scratch_improvement_rel", 0.02)),
            device=str(_get(data, "device", "cpu")),
            gradient_epsilon=float(_get(data, "gradient_epsilon", 1e-6)),
        )


@dataclass(slots=True)
class HMCConfig:
    n_chains: int = 4
    warmup_steps: int = 100
    samples_per_chain: int = 200
    target_accept: float = 0.8
    initial_step_size: float = 0.05
    min_leapfrog_steps: int = 2
    max_leapfrog_steps: int = 32
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75

    def __post_init__(self) -> None:
        if self.n_chains <= 0 or self.samples_per_chain <= 0:
            raise ValueError("hmc.n_chains and hmc.samples_per_chain must be positive")
        if self.warmup_steps < 0:
            raise ValueError("hmc.warmup_steps must be non-negative")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("hmc.target_accept must be in (0, 1)")
        if self.initial_step_size <= 0.0:
            raise ValueError("hmc.initial_step_size must be positive")
        if self.min_leapfrog_steps <= 0 or self.max_leapfrog_steps < self.min_leapfrog_steps:
            raise ValueError("hmc leapfrog bounds must satisfy 0 < min <= max")
        if self.gamma <= 0.0 or self.t0 <= 0.0:
            raise ValueError("hmc.gamma and hmc.t0 must be positive")
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError("hmc.kappa must be in (0, 1]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HMCConfig":
        data = _mapping(raw, "hmc")
        _reject_unknown(data, set(cls.__dataclass_fields__), "hmc")
        return cls(
            n_chains=int(_get(data, "n_chains", 4)),
            warmup_steps=int(_get(data, "warmup_steps", 100)),
            samples_per_chain=int(_get(data, "samples_per_chain", 200)),
            target_accept=float(_get(data, "target_accept", 0.8)),
            initial_step_size=float(_get(data, "initial_step_size", 0.05)),
            min_leapfrog_steps=int(_get(data, "min_leapfrog_steps", 2)),
            max_leapfrog_steps=int(_get(data, "max_leapfrog_steps", 32)),
            gamma=float(_get(data, "gamma", 0.05)),
            t0=float(_get(data, "t0", 10.0)),
            kappa=float(_get(data, "kappa", 0.75)),
        )


@dataclass(slots=True)
class ActiveLearningConfig:
    candidate_pool_size: int = 512
    batch_max_size: int = 8
    posterior_weight: float = 1.0
    distance_weight: float = 0.6
    queued_repulsion_radius_multiplier: float = 2.0
    duplicate_round_decimals: int = 12

    def __post_init__(self) -> None:
        if self.candidate_pool_size <= 0 or self.batch_max_size <= 0:
            raise ValueError("active_learning candidate_pool_size and batch_max_size must be positive")
        if self.posterior_weight < 0.0 or self.distance_weight < 0.0:
            raise ValueError("active_learning weights must be non-negative")
        if self.posterior_weight == 0.0 and self.distance_weight == 0.0:
            raise ValueError("at least one active_learning weight must be positive")
        if self.queued_repulsion_radius_multiplier <= 0.0:
            raise ValueError("active_learning.queued_repulsion_radius_multiplier must be positive")
        if self.duplicate_round_decimals < 0:
            raise ValueError("active_learning.duplicate_round_decimals must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActiveLearningConfig":
        data = _mapping(raw, "active_learning")
        _reject_unknown(data, set(cls.__dataclass_fields__), "active_learning")
        return cls(
            candidate_pool_size=int(_get(data, "candidate_pool_size", 512)),
            batch_max_size=int(_get(data, "batch_max_size", 8)),
            posterior_weight=float(_get(data, "posterior_weight", 1.0)),
            distance_weight=float(_get(data, "distance_weight", 0.6)),
            queued_repulsion_radius_multiplier=float(_get(data, "queued_repulsion_radius_multiplier", 2.0)),
            duplicate_round_decimals=int(_get(data, "duplicate_round_decimals", 12)),
        )


@dataclass(slots=True)
class QueueConfig:
    min_pending_per_worker: int = 1
    max_pending_per_worker: int = 2
    refill_threshold_pending_per_worker: int = 1
    target_seconds_ahead: float = 60.0
    retry_limit: int = 1

    def __post_init__(self) -> None:
        if min(self.min_pending_per_worker, self.max_pending_per_worker, self.refill_threshold_pending_per_worker) < 0:
            raise ValueError("queue pending counts must be non-negative")
        if self.max_pending_per_worker < self.min_pending_per_worker:
            raise ValueError("queue.max_pending_per_worker must be >= min_pending_per_worker")
        if self.refill_threshold_pending_per_worker > self.max_pending_per_worker:
            raise ValueError("queue.refill_threshold_pending_per_worker must not exceed max_pending_per_worker")
        if self.target_seconds_ahead < 0.0 or self.retry_limit < 0:
            raise ValueError("queue.target_seconds_ahead and retry_limit must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QueueConfig":
        data = _mapping(raw, "queue")
        _reject_unknown(data, set(cls.__dataclass_fields__), "queue")
        return cls(
            min_pending_per_worker=int(_get(data, "min_pending_per_worker", 1)),
            max_pending_per_worker=int(_get(data, "max_pending_per_worker", 2)),
            refill_threshold_pending_per_worker=int(_get(data, "refill_threshold_pending_per_worker", 1)),
            target_seconds_ahead=float(_get(data, "target_seconds_ahead", 60.0)),
            retry_limit=int(_get(data, "retry_limit", 1)),
        )


@dataclass(slots=True)
class AppConfig:
    config_path: Path
    run: RunConfig
    problem: ProblemConfig
    solver: FactoryConfig
    posterior: FactoryConfig
    initial_data: FactoryConfig | None
    initial_design: InitialDesignConfig
    surrogate: SurrogateConfig
    hmc: HMCConfig
    active_learning: ActiveLearningConfig
    queue: QueueConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    raw = _mapping(loaded, "configuration")
    allowed = {
        "run",
        "problem",
        "solver",
        "posterior",
        "initial_data",
        "initial_design",
        "surrogate",
        "hmc",
        "active_learning",
        "queue",
    }
    _reject_unknown(raw, allowed, "top-level")
    for required in ("problem", "solver", "posterior"):
        if required not in raw:
            raise ValueError(f"missing required top-level section: {required}")
    return AppConfig(
        config_path=config_path,
        run=RunConfig.from_dict(_mapping(raw.get("run"), "run")),
        problem=ProblemConfig.from_dict(_mapping(raw["problem"], "problem")),
        solver=FactoryConfig.from_dict(_mapping(raw["solver"], "solver")),
        posterior=FactoryConfig.from_dict(_mapping(raw["posterior"], "posterior")),
        initial_data=(
            FactoryConfig.from_dict(_mapping(raw["initial_data"], "initial_data"))
            if raw.get("initial_data") is not None
            else None
        ),
        initial_design=InitialDesignConfig.from_dict(_mapping(raw.get("initial_design"), "initial_design")),
        surrogate=SurrogateConfig.from_dict(_mapping(raw.get("surrogate"), "surrogate")),
        hmc=HMCConfig.from_dict(_mapping(raw.get("hmc"), "hmc")),
        active_learning=ActiveLearningConfig.from_dict(_mapping(raw.get("active_learning"), "active_learning")),
        queue=QueueConfig.from_dict(_mapping(raw.get("queue"), "queue")),
    )


def resolve_dotted(path: str) -> Callable[..., Any]:
    path = path.strip()
    if not path:
        raise ValueError("factory path must not be empty")
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        try:
            module_name, attr_name = path.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(f"factory path must include a module and callable: {path!r}") from exc
    if not module_name or not attr_name:
        raise ValueError(f"invalid factory path: {path!r}")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_name.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"{path!r} did not resolve to a callable")
    return target


def build_factory_object(factory_config: FactoryConfig, app_config: AppConfig) -> Any:
    factory = resolve_dotted(factory_config.factory)
    return factory(config=app_config, **factory_config.kwargs)
