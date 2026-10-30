#!/usr/bin/env python3
"""Verify an external OGS contract and optionally execute one bounded native job."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import lsq_linear

from bayes_asi.config import load_config
from bayes_asi.fields import KLField
from bayes_asi.integrations.ogs import load_initial_data, make_posterior, make_solver
from bayes_asi.types import SolverContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-solver", action="store_true", help="execute one 730-day native OGS evaluation")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--parity-atol", type=float, default=1e-8)
    parser.add_argument("--source-candidate-id", default="chain15_step0001_proposal")
    args = parser.parse_args()
    if not np.isfinite(args.parity_atol) or args.parity_atol <= 0:
        parser.error("--parity-atol must be finite and positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = args.workspace.expanduser().resolve()
    kwargs = dict(
        workspace=str(workspace),
        config_path="config/ogs_mh_ert_20point_730d_allparams_per_support_archie_recentered_10h_prior.json",
        dataset_path="data/nn_pretraining/ert20_current_rf_no_archie_ls/dataset_arrays.npz",
        source_result_dir=("results/ogs_allparams_porosity_rf_ert20/"
                           "ert20_support_730d_allparams_per_support_archie_recentered_10h_native_s1_v1"),
        source_candidate_id=args.source_candidate_id,
    )
    configuration = dict(
        run=dict(output_dir=str(output / "inversion"), time_budget_seconds=3600, seed=20261030),
        problem=dict(latent_dim=207, parameter_dim=207, output_dim=700),
        solver=dict(factory="bayes_asi.integrations.ogs:make_solver", kwargs={**kwargs, "timeout_s": args.timeout_s}),
        posterior=dict(factory="bayes_asi.integrations.ogs:make_posterior", kwargs=kwargs),
        initial_data=dict(factory="bayes_asi.integrations.ogs:load_initial_data", kwargs=kwargs),
        initial_design=dict(n_points=32),
        surrogate=dict(hidden_layers=[128, 128], max_epochs=300, min_train_samples=32),
        hierarchy=dict(enabled=True, student=dict(hidden_layers=[32], max_epochs=200),
                       distillation_points=512, teacher_screen_size=64, prediction_batch_size=128),
        hmc=dict(n_chains=4, warmup_steps=64, samples_per_chain=256),
        active_learning=dict(candidate_pool_size=1024, batch_max_size=4),
    )
    config_path = output / "ogs_hierarchy.yaml"
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    config = load_config(config_path)
    solver = make_solver(config, **kwargs, timeout_s=args.timeout_s)
    posterior = make_posterior(config, **kwargs)
    initial = load_initial_data(config, **kwargs)
    x, targets = initial["theta"], initial["outputs"]
    z = posterior.prior.to_standard_normal(x)
    restored = posterior.prior.to_theta(torch.as_tensor(z, dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(restored, x, rtol=1e-10, atol=1e-10)
    active = solver.contract.active
    perm_context = active.base.mh_context.perm_context
    basis = np.asarray(perm_context.cell_basis_domain)
    prior = active.base.mh_context.perm_prior
    field = KLField(np.zeros(basis.shape[0]), prior.domain_field_std * basis)
    rng = np.random.default_rng(20261030)
    chosen = rng.choice(len(x), size=min(20, len(x)), replace=False)
    likelihood_errors, field_errors = [], []
    cfg = json.loads((workspace / kwargs["config_path"]).read_text())["likelihood"]["ert_log10_resistivity"]
    obs = posterior.likelihood.observed.numpy()
    sigma = float(cfg.get("support_sigma_log10", cfg.get("sigma_log10", .18)))
    effective = min(float(cfg.get("support_effective_independent_rows", obs.size)), obs.size)
    power = float(cfg.get("power_weight", 1))
    if posterior.likelihood.likelihood_mode != "raw_log10" or not posterior.likelihood.per_support:
        raise ValueError("verification reference expects raw_log10 per-support legacy configuration")
    for index in chosen:
        named = dict(zip(posterior.prior.names, x[index], strict=True))
        perm = np.array([named[n] for n in active.perm_names])
        expected_field = solver.contract.kernels.drill.evaluate_field_on_mesh(perm, perm_context, prior)
        xi = np.array([named[f"kl_xi_{j:03d}"] for j in range(field.n_modes)])
        field_errors.append(float(np.max(np.abs(named["k0_log10_k_m2"] + field.evaluate(xi) - expected_field))))
        predictors = targets[index].reshape(obs.shape)
        predicted = np.empty_like(obs)
        for support, (xx, yy) in enumerate(zip(predictors, obs, strict=True)):
            design = np.column_stack([np.ones_like(xx), xx])
            fit = lsq_linear(design, yy, bounds=([-.7, -2.], [1.3, -.2]), tol=1e-13)
            predicted[support] = design @ fit.x
        expected = power*(effective/obs.size)*np.sum(
            -.5*((predicted-obs)/sigma)**2 - .5*np.log(2*np.pi*sigma**2))
        actual = float(posterior.log_likelihood(torch.as_tensor(x[index]), torch.as_tensor(targets[index])))
        likelihood_errors.append(abs(actual-expected))
    assert max(field_errors) < 1e-12
    assert max(likelihood_errors) < 1e-7, likelihood_errors
    with np.load(workspace / kwargs["dataset_path"], allow_pickle=False) as data:
        matches = np.flatnonzero((data["candidate_id"] == args.source_candidate_id)
                                 & (data["campaign_id"] == Path(kwargs["source_result_dir"]).name))
        if matches.size != 1:
            raise ValueError("source candidate must identify exactly one dataset row in source campaign")
        selected = int(matches[0])
    summary = dict(
        verified=datetime.now(UTC).isoformat(), parameter_dim=207, output_dim=700, archived_rows=len(x),
        prior_roundtrip_max_abs=float(np.max(np.abs(restored-x))),
        kl_reference_cases=len(chosen), kl_max_abs=max(field_errors),
        likelihood_scipy_cases=len(chosen), likelihood_max_abs=max(likelihood_errors),
        output_contract="legacy_support_logtheta",
        imports_surrdamh="surrDAMH" in sys.modules,
        new_solver_runs=0,
        dataset_sha256=hashlib.sha256((workspace / kwargs["dataset_path"]).read_bytes()).hexdigest(),
        generated_configuration=str(config_path),
    )
    assert not summary["imports_surrdamh"]
    if args.run_solver:
        started = time.monotonic()
        context = SolverContext(output, output / "native_smoke", "native_smoke", 1)
        context.work_dir.mkdir(exist_ok=True)
        result = solver.evaluate(x[selected], context)
        difference = float(np.max(np.abs(result.outputs-targets[selected])))
        if result.outputs.shape != (700,) or not np.all(np.isfinite(result.outputs)):
            raise AssertionError("native solver output contract failed")
        summary.update(new_solver_runs=1, solver_elapsed_s=time.monotonic()-started,
                       native_vs_archived_max_abs=difference,
                       native_archive_parity_atol=args.parity_atol,
                       native_archive_parity_passed=difference <= args.parity_atol,
                       artifact_paths=result.artifact_paths, native_metadata=result.metadata)
    (output / "verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"native_metadata", "artifact_paths"}}, indent=2))
    if args.run_solver and not summary["native_archive_parity_passed"]:
        raise AssertionError("native output differs from archived target; see verification.json")


if __name__ == "__main__":
    main()
