from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


@dataclass(slots=True)
class FactoryConfig:
    factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactoryConfig":
        if "factory" not in data:
            raise ValueError("factory section must contain a 'factory' dotted path")
        return cls(factory=str(data["factory"]), kwargs=dict(data.get("kwargs", {})))


@dataclass(slots=True)
class RunConfig:
    output_dir: Path
    time_budget_seconds: float = 3600.0
    drain_running_on_exit: bool = True
    seed: int = 12345
    status_interval_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        output_dir = Path(_get(data, "output_dir", "runs/default"))
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        return cls(
            output_dir=output_dir,
            time_budget_seconds=float(_get(data, "time_budget_seconds", 3600.0)),
            drain_running_on_exit=bool(_get(data, "drain_running_on_exit", True)),
            seed=int(_get(data, "seed", 12345)),
            status_interval_seconds=float(_get(data, "status_interval_seconds", 5.0)),
        )


@dataclass(slots=True)
class ProblemConfig:
    latent_dim: int
    parameter_dim: int
    output_dim: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemConfig":
        return cls(
            latent_dim=int(data["latent_dim"]),
            parameter_dim=int(data["parameter_dim"]),
            output_dim=int(data["output_dim"]),
        )


@dataclass(slots=True)
class InitialDesignConfig:
    n_points: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InitialDesignConfig":
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SurrogateConfig":
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HMCConfig":
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
        if self.queued_repulsion_radius_multiplier <= 0.0:
            raise ValueError("active_learning.queued_repulsion_radius_multiplier must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveLearningConfig":
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueConfig":
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
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig(
        config_path=config_path,
        run=RunConfig.from_dict(raw.get("run", {})),
        problem=ProblemConfig.from_dict(raw["problem"]),
        solver=FactoryConfig.from_dict(raw["solver"]),
        posterior=FactoryConfig.from_dict(raw["posterior"]),
        initial_data=FactoryConfig.from_dict(raw["initial_data"]) if raw.get("initial_data") else None,
        initial_design=InitialDesignConfig.from_dict(raw.get("initial_design", {})),
        surrogate=SurrogateConfig.from_dict(raw.get("surrogate", {})),
        hmc=HMCConfig.from_dict(raw.get("hmc", {})),
        active_learning=ActiveLearningConfig.from_dict(raw.get("active_learning", {})),
        queue=QueueConfig.from_dict(raw.get("queue", {})),
    )


def resolve_dotted(path: str) -> Callable[..., Any]:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
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
