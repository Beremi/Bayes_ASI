"""Native adapter for the current 207-parameter Mont Terri OGS workflow.

External campaign modules and all scientific data stay in an explicitly supplied
workspace.  In particular, this module never imports the legacy ``surrDAMH``
package or the surrogate campaign wrapper that depends on it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from bayes_asi.config import AppConfig
from bayes_asi.likelihoods import ProfiledAffineLikelihood
from bayes_asi.parameters import NamedParameterSpace
from bayes_asi.priors import Marginal, normal, truncated_normal, uniform
from bayes_asi.types import ForwardResult, SolverContext

PARAMETER_COUNT = 207
SUPPORT_COUNT = 20
TIME_COUNT = 35
OUTPUT_COUNT = SUPPORT_COUNT * TIME_COUNT
SECONDS_PER_DAY = 86_400.0
OUTPUT_CONTRACT = "legacy_support_logtheta"


@dataclass(frozen=True, slots=True)
class ExternalKernels:
    allp: ModuleType
    mh: ModuleType
    prf: ModuleType
    drill: ModuleType


@dataclass(frozen=True, slots=True)
class OGSContract:
    active: Any
    feature_names: tuple[str, ...]
    support_ids: tuple[int, ...]
    model_days: np.ndarray
    dataset: dict[str, np.ndarray]
    kernels: ExternalKernels
    workspace: Path
    config_path: Path
    dataset_path: Path


@contextmanager
def _script_import_path(scripts: Path):
    original = list(sys.path)
    sys.path.insert(0, str(scripts))
    try:
        yield
    finally:
        sys.path[:] = original


def _load_external_kernels(workspace: Path) -> ExternalKernels:
    scripts = workspace / "scripts"
    required = {
        "run_ogs_allparams_porosity_rf_ert20_campaign",
        "run_ogs_mh_ert_nmr_campaign",
        "run_ogs_porosity_rf_ert20_campaign",
        "run_mh_permeability_drill_chains",
    }
    missing = sorted(name for name in required if not (scripts / f"{name}.py").is_file())
    if missing:
        raise FileNotFoundError(f"OGS workspace is missing external kernel(s): {', '.join(missing)}")
    with _script_import_path(scripts):
        modules = {name: importlib.import_module(name) for name in required}
    for name, module in modules.items():
        module_path = Path(module.__file__ or "").resolve()
        if scripts.resolve() not in module_path.parents:
            raise RuntimeError(f"external kernel {name} was already imported from another workspace: {module_path}")
    return ExternalKernels(
        allp=modules["run_ogs_allparams_porosity_rf_ert20_campaign"],
        mh=modules["run_ogs_mh_ert_nmr_campaign"],
        prf=modules["run_ogs_porosity_rf_ert20_campaign"],
        drill=modules["run_mh_permeability_drill_chains"],
    )


def _resolve_external_path(workspace: Path, value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else workspace / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    required = (
        "X",
        "feature_names",
        "support_ids",
        "model_days",
        "observed_log10_resistivity",
        "theta_log10",
    )
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in required if name in archive.files}


def _load_contract(
    config: AppConfig,
    *,
    workspace: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    source_result_dir: str | Path,
    source_candidate_id: str,
) -> OGSContract:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OGS workspace does not exist: {root}")
    external_config = _resolve_external_path(root, config_path, "config_path")
    dataset_file = _resolve_external_path(root, dataset_path, "dataset_path")
    source_dir = _resolve_external_path(root, source_result_dir, "source_result_dir")
    if not source_candidate_id.strip():
        raise ValueError("source_candidate_id must not be empty")
    kernels = _load_external_kernels(root)
    active = kernels.allp.load_context(
        argparse.Namespace(
            config=external_config,
            source_result_dir=source_dir,
            source_candidate_id=source_candidate_id,
        )
    )
    prior = active.base.mh_context.perm_prior
    if str(prior.model_type) != "domain_kl":
        raise ValueError(f"native OGS adapter requires model_type='domain_kl', got {prior.model_type!r}")
    feature_names = tuple(
        [
            *active.perm_names,
            "homogeneous_porosity",
            *active.porosity_names,
            "open_niche_curve_shift_days",
            "open_niche_curve_suction_scale",
            "open_niche_curve_suction_offset_mpa",
        ]
    )
    if len(feature_names) != PARAMETER_COUNT:
        raise ValueError(f"current native OGS contract requires {PARAMETER_COUNT} features, got {len(feature_names)}")
    dataset = _load_dataset(dataset_file)
    required_arrays = {"X", "feature_names", "support_ids", "model_days", "observed_log10_resistivity", "theta_log10"}
    missing_arrays = sorted(required_arrays - set(dataset))
    if missing_arrays:
        raise ValueError(f"OGS dataset is missing array(s): {', '.join(missing_arrays)}")
    stored_names = tuple(str(value) for value in np.asarray(dataset["feature_names"]).tolist())
    if stored_names != feature_names:
        raise ValueError("dataset feature names/order do not match the active legacy domain-KL no-Archie schema")
    support_ids = tuple(int(value) for value in np.asarray(dataset["support_ids"]).tolist())
    model_days = np.asarray(dataset["model_days"], dtype=float)
    x = np.asarray(dataset["X"], dtype=float)
    outputs = np.asarray(dataset["theta_log10"], dtype=float)
    observed = np.asarray(dataset["observed_log10_resistivity"], dtype=float)
    if len(support_ids) != SUPPORT_COUNT or model_days.shape != (TIME_COUNT,):
        raise ValueError(f"current native OGS contract requires {SUPPORT_COUNT} supports x {TIME_COUNT} times")
    if tuple(int(value) for value in active.support_ids) != support_ids:
        raise ValueError("dataset support IDs/order do not match the active OGS configuration")
    support_table = active.base.mh_context.ert_support_timeseries
    available_days = np.asarray(support_table["model_day"], dtype=float)
    tolerance_days = float(active.base.mh_context.config["likelihood"]["ert_log10_resistivity"]["time_tolerance_days"])
    if available_days.size == 0 or float(np.max(model_days)) > float(np.max(available_days)) + tolerance_days:
        raise ValueError("active OGS configuration does not cover the dataset model-day horizon")
    if x.ndim != 2 or x.shape[1] != PARAMETER_COUNT or outputs.shape != (x.shape[0], SUPPORT_COUNT, TIME_COUNT):
        raise ValueError("dataset X/theta_log10 shapes do not satisfy the 207-to-700 contract")
    if observed.shape != (SUPPORT_COUNT, TIME_COUNT):
        raise ValueError("observed_log10_resistivity must have shape (20, 35)")
    if not all(np.all(np.isfinite(value)) for value in (x, outputs, observed, model_days)):
        raise ValueError("OGS dataset arrays must be finite")
    if config.problem.parameter_dim != PARAMETER_COUNT or config.problem.output_dim != OUTPUT_COUNT:
        raise ValueError("AppConfig problem dimensions must match the native OGS 207-to-700 contract")
    return OGSContract(
        active=active,
        feature_names=feature_names,
        support_ids=support_ids,
        model_days=model_days,
        dataset=dataset,
        kernels=kernels,
        workspace=root,
        config_path=external_config,
        dataset_path=dataset_file,
    )


def _adjusted_normal(active: Any, name: str, mean: float, sd: float) -> Marginal:
    adjusted_mean = float(active.prior_mean_overrides.get(name, mean))
    adjusted_sd = float(sd) * float(active.prior_std_multipliers.get(name, 1.0))
    return normal(adjusted_mean, adjusted_sd)


def _bounded_component(
    active: Any, kernels: ExternalKernels, name: str, spec: Any | None, fixed: float
) -> Marginal | float:
    if spec is None:
        return float(fixed)
    adjusted = kernels.allp.adjusted_bounded_normal(spec, active, name)
    return truncated_normal(adjusted.mean, adjusted.std, adjusted.lower, adjusted.upper)


def _build_prior(contract: OGSContract) -> NamedParameterSpace:
    active, kernels = contract.active, contract.kernels
    perm_prior = active.base.mh_context.perm_prior
    fixed = active.base.fixed_params
    components: list[Marginal | float] = []
    for name in contract.feature_names:
        if name == "k0_log10_k_m2":
            components.append(_adjusted_normal(active, name, perm_prior.k0_mean, perm_prior.k0_std))
        elif name.startswith("kl_xi_"):
            components.append(_adjusted_normal(active, name, 0.0, 1.0))
        elif name == "bedding_angle_deg":
            components.append(_adjusted_normal(active, name, perm_prior.angle_mean_deg, perm_prior.angle_std_deg))
        elif name == "q_22_over_11":
            components.append(uniform(perm_prior.q_lower, perm_prior.q_upper))
        elif name == "homogeneous_porosity":
            components.append(_bounded_component(active, kernels, name, active.porosity_mean_prior, 0.0))
        elif name.startswith("porosity_kl_xi_"):
            components.append(_adjusted_normal(active, name, 0.0, 1.0))
        elif name == "open_niche_curve_shift_days":
            components.append(
                _bounded_component(
                    active,
                    kernels,
                    name,
                    active.base.mh_context.open_niche_curve_shift_prior,
                    fixed.get(name, 0.0),
                )
            )
        elif name == "open_niche_curve_suction_scale":
            components.append(
                _bounded_component(active, kernels, name, active.suction_scale_prior, fixed.get(name, 1.0))
            )
        elif name == "open_niche_curve_suction_offset_mpa":
            components.append(
                _bounded_component(active, kernels, name, active.suction_offset_prior, fixed.get(name, 0.0))
            )
        else:  # pragma: no cover - guarded by exact feature schema construction.
            raise ValueError(f"unsupported OGS parameter {name!r}")
    prior = NamedParameterSpace(contract.feature_names, components)
    return prior


@dataclass(slots=True)
class OGSPosterior:
    prior: NamedParameterSpace
    likelihood: ProfiledAffineLikelihood
    output_contract: str = OUTPUT_CONTRACT

    def log_likelihood(self, theta: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        del theta
        return self.likelihood(y_pred)


def make_posterior(
    config: AppConfig,
    *,
    workspace: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    source_result_dir: str | Path,
    source_candidate_id: str,
    archie_mode: str = "config",
    likelihood_mode: str = "config",
    coefficient_fixed_c_rad_per_day: float | None = None,
    coefficient_fixed_d_rad: float | None = None,
    coefficient_sigma_a: float | None = None,
    coefficient_sigma_b: float | None = None,
    coefficient_sigma_e_log10_per_day: float | None = None,
) -> OGSPosterior:
    contract = _load_contract(
        config,
        workspace=workspace,
        config_path=config_path,
        dataset_path=dataset_path,
        source_result_dir=source_result_dir,
        source_candidate_id=source_candidate_id,
    )
    prior = _build_prior(contract)
    if prior.parameter_dim != config.problem.parameter_dim or prior.latent_dim != config.problem.latent_dim:
        raise ValueError("configured OGS prior dimensions do not match AppConfig")
    likelihood_cfg = json.loads(contract.config_path.read_text(encoding="utf-8"))["likelihood"]["ert_log10_resistivity"]
    chosen_archie = str(likelihood_cfg.get("archie_mode", "per_support") if archie_mode == "config" else archie_mode)
    if chosen_archie not in {"per_support", "global"}:
        raise ValueError("archie_mode must be 'config', 'per_support', or 'global'")
    chosen_mode = str(
        likelihood_cfg.get("likelihood_mode", likelihood_cfg.get("primary_mode", "raw_log10"))
        if likelihood_mode == "config"
        else likelihood_mode
    )
    coefficient_cfg = likelihood_cfg.get("coefficient_likelihood", {})
    fixed_c = float(
        coefficient_fixed_c_rad_per_day
        if coefficient_fixed_c_rad_per_day is not None
        else coefficient_cfg.get("fixed_c_rad_per_day", 0.018299002254)
    )
    fixed_d = float(
        coefficient_fixed_d_rad
        if coefficient_fixed_d_rad is not None
        else coefficient_cfg.get("fixed_d_rad", -0.414049998782)
    )
    coefficient_sigmas = np.asarray(
        [
            coefficient_sigma_a if coefficient_sigma_a is not None else coefficient_cfg.get("sigma_a", 0.009572565142),
            coefficient_sigma_b if coefficient_sigma_b is not None else coefficient_cfg.get("sigma_b", 0.007153735148),
            coefficient_sigma_e_log10_per_day
            if coefficient_sigma_e_log10_per_day is not None
            else coefficient_cfg.get("sigma_e_log10_per_day", 2.632887523e-5),
        ],
        dtype=float,
    )
    observed = np.asarray(contract.dataset["observed_log10_resistivity"], dtype=float)
    effective = min(
        float(likelihood_cfg.get("support_effective_independent_rows", observed.size)), float(observed.size)
    )
    likelihood = ProfiledAffineLikelihood(
        observed,
        float(likelihood_cfg.get("support_sigma_log10", likelihood_cfg.get("sigma_log10", 0.18))),
        a_bounds=(-0.7, 1.3),
        b_bounds=(-2.0, -0.2),
        per_support=chosen_archie == "per_support",
        effective_rows=effective,
        power_weight=float(likelihood_cfg.get("power_weight", 1.0)),
        likelihood_mode=chosen_mode,
        model_days=contract.model_days if chosen_mode == "coefficient_fixed_c_d_abe" else None,
        coefficient_fixed_c_d_abe=(fixed_c, fixed_d) if chosen_mode == "coefficient_fixed_c_d_abe" else None,
        coefficient_sigmas=coefficient_sigmas if chosen_mode == "coefficient_fixed_c_d_abe" else None,
    )
    return OGSPosterior(prior=prior, likelihood=likelihood)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_command_success(status: dict[str, Any], phase: str) -> None:
    if status.get("skipped"):
        raise RuntimeError(f"{phase} was skipped; native adapter does not accept pseudo OGS success")
    if status.get("timed_out"):
        raise TimeoutError(f"{phase} timed out")
    if int(status.get("returncode", -1)) != 0:
        raise RuntimeError(f"{phase} failed with return code {status.get('returncode')}")


def support_theta_log10_from_theta_by_time(
    theta_by_time: dict[float, np.ndarray],
    contract: OGSContract,
) -> np.ndarray:
    """Legacy area-weighted mean of cell log10(theta), support-major."""

    cfg = contract.active.base.mh_context.config["likelihood"]["ert_log10_resistivity"]
    selected = contract.kernels.mh.choose_ert_support_model_times(
        sorted(theta_by_time), contract.active.base.mh_context, cfg
    )
    if len(selected) != TIME_COUNT:
        raise ValueError(f"OGS produced {len(selected)} matched ERT times; expected {TIME_COUNT}")
    lo, hi = (float(value) for value in cfg["theta_clip"])
    weights = contract.active.base.mh_context.ert_support_weights
    observations = contract.active.base.mh_context.ert_support_timeseries
    support_index = {support_id: index for index, support_id in enumerate(contract.support_ids)}
    values = np.full((SUPPORT_COUNT, TIME_COUNT), np.nan, dtype=float)
    selected_days: list[float] = []
    for time_index, (time_s, match) in enumerate(selected):
        timestamp = contract.kernels.mh.pd.Timestamp(match["timestamp_iso"])
        official_day = float(match.get("official_model_day", float(time_s) / SECONDS_PER_DAY))
        selected_days.append(official_day)
        theta = np.asarray(theta_by_time[time_s], dtype=float)
        for _, observation in observations[observations["timestamp"].eq(timestamp)].iterrows():
            support_id = int(observation["support_id"])
            if support_id not in support_index:
                continue
            local = weights[weights["support_id"].eq(support_id)]
            cell_ids = local["ogs_lookup_cell_id"].to_numpy(dtype=int)
            valid = (cell_ids >= 0) & (cell_ids < theta.size)
            if not np.any(valid):
                continue
            cell_ids = cell_ids[valid]
            area = np.maximum(local["support_intersection_area_m2"].to_numpy(dtype=float)[valid], 1e-12)
            clipped = np.clip(theta[cell_ids], lo, hi)
            values[support_index[support_id], time_index] = float(np.average(np.log10(clipped), weights=area))
    if not np.allclose(selected_days, contract.model_days, rtol=0.0, atol=1e-8):
        raise ValueError("selected OGS ERT model days do not match the imported training-output order")
    if not np.all(np.isfinite(values)):
        raise ValueError("OGS extraction did not produce every support/time value")
    return values.reshape(-1)


@dataclass(slots=True)
class NativeOGSSolver:
    contract: OGSContract
    horizon_days: int = 730
    timeout_s: int = 2400
    ogs_backend: str = "native"

    def evaluate(self, theta: np.ndarray, context: SolverContext) -> ForwardResult:
        vector = np.asarray(theta, dtype=float).reshape(PARAMETER_COUNT)
        if not np.all(np.isfinite(vector)):
            raise ValueError("OGS parameters must be finite")
        named = dict(zip(self.contract.feature_names, vector, strict=True))
        attempt = int(context.metadata.get("attempt", 0))
        context.work_dir.mkdir(parents=True, exist_ok=True)
        attempt_root = Path(tempfile.mkdtemp(prefix=f"attempt_{attempt}_", dir=context.work_dir))
        ogs_root = attempt_root / "ogs"
        result_root = attempt_root / "result"
        ogs_root.mkdir()
        result_root.mkdir()
        candidate_id = context.job_id
        run_dir = ogs_root / candidate_id
        active, kernels = self.contract.active, self.contract.kernels
        prepare = kernels.mh.prepare_run(
            run_dir,
            candidate_id,
            self.horizon_days,
            int(active.base.config["ogs_policy"]["output_each_steps"]),
            float(named["open_niche_curve_shift_days"]),
            float(active.base.config["ogs_policy"].get("open_niche_curve_shift_cycle_days", 365.25)),
            str(active.base.config["ogs_policy"].get("open_niche_curve_shift_method", "endpoint_blended_wrap")),
            float(active.base.config["ogs_policy"].get("open_niche_curve_wrap_blend_days", 45.0)),
            float(active.base.config["ogs_policy"].get("open_niche_curve_max_suction_jump_mpa", 3.0)),
            float(named["open_niche_curve_suction_scale"]),
            float(named["open_niche_curve_suction_offset_mpa"]),
        )
        _require_command_success(prepare, "OGS preparation")
        perm_theta = np.asarray([named[name] for name in active.perm_names], dtype=float)
        porosity_xi = np.asarray([named[name] for name in active.porosity_names], dtype=float)
        log10_k = kernels.drill.evaluate_field_on_mesh(
            perm_theta, active.base.mh_context.perm_context, active.base.mh_context.perm_prior
        )
        porosity, porosity_summary = kernels.allp.porosity_field_with_mean(
            porosity_xi, float(named["homogeneous_porosity"]), active
        )
        tensor_summary = kernels.prf.write_candidate_mesh_fields(
            run_dir,
            log10_k,
            porosity,
            float(named["bedding_angle_deg"]),
            float(named["q_22_over_11"]),
        )
        ogs = kernels.mh.run_ogs(run_dir, self.timeout_s, False, self.ogs_backend)
        _require_command_success(ogs, "OGS execution")
        theta_by_time = kernels.prf.output_theta_by_time_porosity_field(run_dir, active.base, porosity)
        if not theta_by_time:
            raise RuntimeError("OGS execution produced no extractable theta outputs")
        outputs = support_theta_log10_from_theta_by_time(theta_by_time, self.contract)
        result_path = result_root / "legacy_support_logtheta.npz"
        np.savez_compressed(
            result_path,
            theta_log10=outputs.reshape(SUPPORT_COUNT, TIME_COUNT),
            support_ids=np.asarray(self.contract.support_ids),
            model_days=self.contract.model_days,
            feature_names=np.asarray(self.contract.feature_names),
            theta=vector,
        )
        metadata = {
            "output_contract": OUTPUT_CONTRACT,
            "parameter_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "external_config_sha256": _sha256_file(self.contract.config_path),
            "dataset_sha256": _sha256_file(self.contract.dataset_path),
            "prepare": prepare,
            "ogs": ogs,
            "porosity_summary": porosity_summary,
            "permeability_tensor_summary": tensor_summary,
            "attempt": attempt,
            "attempt_root": str(attempt_root),
        }
        return ForwardResult(outputs=outputs, metadata=metadata, artifact_paths=[str(run_dir), str(result_path)])


def make_solver(
    config: AppConfig,
    *,
    workspace: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    source_result_dir: str | Path,
    source_candidate_id: str,
    horizon_days: int = 730,
    timeout_s: int = 2400,
    ogs_backend: str = "native",
) -> NativeOGSSolver:
    if horizon_days <= 0 or timeout_s <= 0:
        raise ValueError("horizon_days and timeout_s must be positive")
    if ogs_backend not in {"native", "docker", "auto"}:
        raise ValueError("unsupported OGS backend")
    contract = _load_contract(
        config,
        workspace=workspace,
        config_path=config_path,
        dataset_path=dataset_path,
        source_result_dir=source_result_dir,
        source_candidate_id=source_candidate_id,
    )
    # Solver evaluations need the validated schema and external contexts, not
    # the in-memory pretraining matrices retained by initial-data loading.
    solver_contract = replace(contract, dataset={})
    return NativeOGSSolver(
        solver_contract,
        horizon_days=int(horizon_days),
        timeout_s=int(timeout_s),
        ogs_backend=ogs_backend,
    )


def load_initial_data(
    config: AppConfig,
    *,
    workspace: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    source_result_dir: str | Path,
    source_candidate_id: str,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    contract = _load_contract(
        config,
        workspace=workspace,
        config_path=config_path,
        dataset_path=dataset_path,
        source_result_dir=source_result_dir,
        source_candidate_id=source_candidate_id,
    )
    x = np.asarray(contract.dataset["X"], dtype=float)
    outputs = np.asarray(contract.dataset["theta_log10"], dtype=float).reshape(x.shape[0], OUTPUT_COUNT)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if x.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))
            x, outputs = x[indices], outputs[indices]
    return {
        "theta": x.copy(),
        "outputs": outputs.copy(),
        "metadata": {
            "output_contract": OUTPUT_CONTRACT,
            "dataset_path": str(contract.dataset_path),
            "dataset_sha256": _sha256_file(contract.dataset_path),
            "selected_rows": int(x.shape[0]),
        },
    }


__all__ = [
    "NativeOGSSolver",
    "OGSPosterior",
    "load_initial_data",
    "make_posterior",
    "make_solver",
    "support_theta_log10_from_theta_by_time",
]
