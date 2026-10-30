from __future__ import annotations

import importlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bayes_asi.config import load_config
from bayes_asi.integrations import ogs
from bayes_asi.types import SolverContext

pd = importlib.import_module("pandas")


class Bounded:
    def __init__(self, mean: float, std: float, lower: float, upper: float) -> None:
        self.mean, self.std, self.lower, self.upper = mean, std, lower, upper


class OGSAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "external"
        self.workspace.mkdir()
        self.source = self.workspace / "source"
        self.source.mkdir()
        self.config_path = self.workspace / "config.json"
        self.likelihood_cfg = {
            "archie_mode": "per_support",
            "support_sigma_log10": 0.18,
            "support_effective_independent_rows": 80,
            "power_weight": 1.0,
            "theta_clip": [0.005, 0.25],
            "time_tolerance_days": 10.0,
        }
        self.config_path.write_text(
            json.dumps({"likelihood": {"ert_log10_resistivity": self.likelihood_cfg}}),
            encoding="utf-8",
        )
        self.perm_names = [
            "k0_log10_k_m2",
            *[f"kl_xi_{i:03d}" for i in range(100)],
            "bedding_angle_deg",
            "q_22_over_11",
        ]
        self.porosity_names = [f"porosity_kl_xi_{i:03d}" for i in range(100)]
        self.feature_names = [
            *self.perm_names,
            "homogeneous_porosity",
            *self.porosity_names,
            "open_niche_curve_shift_days",
            "open_niche_curve_suction_scale",
            "open_niche_curve_suction_offset_mpa",
        ]
        self.model_days = np.arange(35, dtype=float)
        self.dataset_path = self.workspace / "dataset.npz"
        self._write_dataset()
        app_path = self.root / "app.yaml"
        app_path.write_text(
            """
problem:
  latent_dim: 207
  parameter_dim: 207
  output_dim: 700
solver:
  factory: bayes_asi.integrations.ogs:make_solver
posterior:
  factory: bayes_asi.integrations.ogs:make_posterior
""",
            encoding="utf-8",
        )
        self.app_config = load_config(app_path)
        self.active, self.kernels = self._fake_external()

    def _write_dataset(self, feature_names: list[str] | None = None) -> None:
        rng = np.random.default_rng(4)
        np.savez(
            self.dataset_path,
            X=rng.normal(size=(5, 207)),
            feature_names=np.asarray(self.feature_names if feature_names is None else feature_names),
            support_ids=np.arange(1, 21),
            model_days=self.model_days,
            observed_log10_resistivity=np.tile(np.linspace(1.0, 2.0, 35), (20, 1)),
            theta_log10=rng.normal(size=(5, 20, 35)),
        )

    def _fake_external(self):
        prior = types.SimpleNamespace(
            model_type="domain_kl",
            k0_mean=-16.0,
            k0_std=3.0,
            domain_modes=100,
            angle_mean_deg=144.0,
            angle_std_deg=5.0,
            q_lower=0.1,
            q_upper=0.9,
        )
        timestamps = pd.to_datetime(["2020-01-01"] * 35) + pd.to_timedelta(self.model_days, unit="D")
        observations = pd.DataFrame(
            [
                {"timestamp": timestamp, "support_id": support_id, "model_day": model_day}
                for timestamp, model_day in zip(timestamps, self.model_days, strict=True)
                for support_id in range(1, 21)
            ]
        )
        weights = pd.DataFrame(
            [
                {"support_id": support_id, "ogs_lookup_cell_id": cell, "support_intersection_area_m2": area}
                for support_id in range(1, 21)
                for cell, area in ((0, 1.0), (1, 3.0))
            ]
        )
        mh_context = types.SimpleNamespace(
            perm_prior=prior,
            perm_context=object(),
            config={"likelihood": {"ert_log10_resistivity": self.likelihood_cfg}},
            ert_support_timeseries=observations,
            ert_support_weights=weights,
            open_niche_curve_shift_prior=Bounded(119.58, 35.0, 0.0, 260.0),
        )
        base = types.SimpleNamespace(
            mh_context=mh_context,
            fixed_params={},
            config={
                "likelihood": {"ert_log10_resistivity": self.likelihood_cfg},
                "ogs_policy": {"output_each_steps": 1},
            },
        )
        active = types.SimpleNamespace(
            base=base,
            perm_names=self.perm_names,
            porosity_names=self.porosity_names,
            porosity_mean_prior=Bounded(0.13, 0.025, 0.05, 0.175),
            suction_scale_prior=Bounded(1.0, 0.15, 0.65, 1.35),
            suction_offset_prior=Bounded(0.0, 0.6, -2.0, 2.0),
            prior_mean_overrides={},
            prior_std_multipliers={},
            support_ids=list(range(1, 21)),
        )

        def prepare(run_dir, *args):
            del args
            run_dir.mkdir(parents=True)
            return {"returncode": 0, "timed_out": False, "skipped": False}

        def choose(available, context, cfg):
            del context, cfg
            return [
                (
                    time_s,
                    {
                        "timestamp_iso": timestamps[index].isoformat(),
                        "official_model_day": float(index),
                    },
                )
                for index, time_s in enumerate(available)
            ]

        allp = types.SimpleNamespace(
            load_context=lambda args: active,
            adjusted_bounded_normal=lambda spec, context, name: spec,
            porosity_field_with_mean=lambda xi, mean, context: (
                np.array([mean, mean]),
                {"porosity_area_mean": mean},
            ),
        )
        mh = types.SimpleNamespace(
            pd=pd,
            prepare_run=prepare,
            run_ogs=lambda *args: {"returncode": 0, "timed_out": False, "skipped": False},
            choose_ert_support_model_times=choose,
        )
        prf = types.SimpleNamespace(
            write_candidate_mesh_fields=lambda *args: {"k_min": -16.0},
            output_theta_by_time_porosity_field=lambda *args: {
                float(day * ogs.SECONDS_PER_DAY): np.array([0.01, 0.1]) for day in self.model_days
            },
        )
        drill = types.SimpleNamespace(evaluate_field_on_mesh=lambda *args: np.array([-16.0, -15.0]))
        return active, ogs.ExternalKernels(allp=allp, mh=mh, prf=prf, drill=drill)

    def _kwargs(self):
        return {
            "workspace": self.workspace,
            "config_path": self.config_path,
            "dataset_path": self.dataset_path,
            "source_result_dir": self.source,
            "source_candidate_id": "source_1",
        }

    def test_factories_validate_schema_prior_likelihood_and_initial_data(self) -> None:
        with patch.object(ogs, "_load_external_kernels", return_value=self.kernels):
            posterior = ogs.make_posterior(self.app_config, **self._kwargs())
            coefficient_posterior = ogs.make_posterior(
                self.app_config,
                **self._kwargs(),
                archie_mode="global",
                likelihood_mode="coefficient_fixed_c_d_abe",
            )
            first = ogs.load_initial_data(self.app_config, **self._kwargs(), max_samples=3, seed=11)
            second = ogs.load_initial_data(self.app_config, **self._kwargs(), max_samples=3, seed=11)

        self.assertEqual(posterior.prior.names, tuple(self.feature_names))
        self.assertEqual(posterior.prior.latent_dim, 207)
        self.assertEqual(posterior.likelihood.observed.shape, (20, 35))
        self.assertFalse(coefficient_posterior.likelihood.per_support)
        self.assertEqual(coefficient_posterior.likelihood.likelihood_mode, "coefficient_fixed_c_d_abe")
        self.assertEqual(first["theta"].shape, (3, 207))
        self.assertEqual(first["outputs"].shape, (3, 700))
        self.assertTrue(np.array_equal(first["theta"], second["theta"]))
        value = posterior.log_likelihood(torch.zeros((1, 207)), torch.zeros((1, 700)))
        self.assertEqual(value.shape, (1,))

    def test_rejects_feature_order_and_non_domain_kl(self) -> None:
        bad_names = self.feature_names.copy()
        bad_names[0], bad_names[1] = bad_names[1], bad_names[0]
        self._write_dataset(bad_names)
        with (
            patch.object(ogs, "_load_external_kernels", return_value=self.kernels),
            self.assertRaisesRegex(ValueError, "names/order"),
        ):
            ogs.load_initial_data(self.app_config, **self._kwargs())

        self._write_dataset()
        self.active.base.mh_context.perm_prior.model_type = "circle_threshold_depth"
        with (
            patch.object(ogs, "_load_external_kernels", return_value=self.kernels),
            self.assertRaisesRegex(ValueError, "domain_kl"),
        ):
            ogs.make_solver(self.app_config, **self._kwargs())

    def test_native_solver_isolates_attempt_and_averages_log_cell_theta(self) -> None:
        with patch.object(ogs, "_load_external_kernels", return_value=self.kernels):
            solver = ogs.make_solver(self.app_config, **self._kwargs(), timeout_s=10)
        work_dir = self.root / "worker" / "job"
        work_dir.mkdir(parents=True)
        context = SolverContext(
            run_dir=self.root,
            work_dir=work_dir,
            job_id="job_0001",
            worker_rank=1,
            metadata={"attempt": 2},
        )
        result = solver.evaluate(np.zeros(207), context)

        expected = (np.log10(0.01) + 3.0 * np.log10(0.1)) / 4.0
        self.assertTrue(np.allclose(result.outputs, expected, rtol=0.0, atol=1e-12))
        self.assertEqual(result.outputs.shape, (700,))
        self.assertEqual(result.metadata["output_contract"], "legacy_support_logtheta")
        first_root = Path(result.metadata["attempt_root"])
        self.assertTrue((first_root / "ogs" / "job_0001").is_dir())
        self.assertTrue((first_root / "result" / "legacy_support_logtheta.npz").is_file())

        restarted = solver.evaluate(np.zeros(207), context)
        second_root = Path(restarted.metadata["attempt_root"])
        self.assertNotEqual(first_root, second_root)
        self.assertTrue((second_root / "result" / "legacy_support_logtheta.npz").is_file())

    def test_skipped_or_failed_commands_are_not_successes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            ogs._require_command_success({"returncode": 0, "skipped": True}, "OGS")
        with self.assertRaises(TimeoutError):
            ogs._require_command_success({"returncode": 0, "timed_out": True}, "OGS")
        with self.assertRaisesRegex(RuntimeError, "return code"):
            ogs._require_command_success({"returncode": 3}, "OGS")

    @unittest.skipUnless(os.environ.get("GESA_OGS_WORKSPACE"), "real OGS workspace fixture not requested")
    def test_real_workspace_factories(self) -> None:
        workspace = Path(os.environ["GESA_OGS_WORKSPACE"])
        external_config = "config/ogs_mh_ert_20point_730d_allparams_per_support_archie_recentered_10h_prior.json"
        dataset = "data/nn_pretraining/ert20_current_rf_no_archie_ls/dataset_arrays.npz"
        source = (
            "results/ogs_allparams_porosity_rf_ert20/"
            "ert20_support_730d_allparams_per_support_archie_recentered_10h_native_s1_v1"
        )
        posterior = ogs.make_posterior(
            self.app_config,
            workspace=workspace,
            config_path=external_config,
            dataset_path=dataset,
            source_result_dir=source,
            source_candidate_id="chain15_step0001_proposal",
        )
        solver = ogs.make_solver(
            self.app_config,
            workspace=workspace,
            config_path=external_config,
            dataset_path=dataset,
            source_result_dir=source,
            source_candidate_id="chain15_step0001_proposal",
        )
        self.assertEqual(posterior.prior.parameter_dim, 207)
        self.assertEqual(posterior.prior.latent_dim, 207)
        self.assertEqual(posterior.likelihood.observed.shape, (20, 35))
        self.assertEqual(solver.contract.feature_names, posterior.prior.names)


if __name__ == "__main__":
    unittest.main()
