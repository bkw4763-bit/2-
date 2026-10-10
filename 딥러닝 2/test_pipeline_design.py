"""설계의 시간 순서·후보 비교·KNN·최종 테스트 차단을 검증한다."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import settings as cfg
from ai_protocol import lock_settings, load_locked_settings, split_xgb_training
from evaluation import score_predictions
from make_windows import load_experiment, get_window, feature_columns
from predict_timesfm import TimesFMPredictor
from prepare_weekly import knn_impute_covariates, read_supply, build_weekly
from rolling_origin import get_training_samples, run_rolling_origin


class PipelineDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.windows, cls.protocol = load_experiment()

    def test_split_and_common_candidate_origins(self):
        data, windows = self.data, self.windows
        self.assertTrue((data.date.dt.dayofweek == 5).all())
        for fuel, group in data.groupby("fuel"):
            self.assertEqual(group.date.min(), pd.Timestamp("2015-01-10"))
            self.assertTrue(np.isfinite(group[feature_columns(fuel)].to_numpy(dtype=float)).all())
        v = windows[windows.split == "validation"]
        t = windows[windows.split == "test"]
        self.assertTrue((v.target_end < pd.Timestamp(cfg.TEST_START)).all())
        self.assertTrue((t.target_start >= pd.Timestamp(cfg.TEST_START)).all())
        self.assertTrue((t.target_end < pd.Timestamp(cfg.POLICY_START)).all())
        expected = set(v.origin)
        for _, group in v.groupby(["config_id", "fuel"]):
            self.assertEqual(set(group.origin), expected)
        self.assertEqual(set(v.context_years), set(self.protocol["accepted_context_years"]))
        self.assertEqual(set(v.context_years), set(range(1, 8)))

    def test_weekly_target_and_daily_aggregation(self):
        # 실제 공급가를 유지하고, 원화 환산은 일별로 계산한 뒤 평균한다.
        weekly, daily = build_weekly()
        source = read_supply()
        for fuel, (column, _) in cfg.FUELS.items():
            table = weekly[weekly.fuel == fuel].set_index("date")
            np.testing.assert_allclose(table.supply_pretax, source[column].reindex(table.index))
            one = daily[daily.fuel == fuel].set_index("date")
            expected = one.intl_krw_l.resample(cfg.TIME_FREQ).mean().reindex(table.index)
            np.testing.assert_allclose(table.intl_krw_l, expected)
            self.assertTrue(table.daily_observations.eq(5).all())
            self.assertTrue(table.loc[table.index == "2026-03-14", "in_policy"].all())
            self.assertFalse(table.loc[table.index == "2026-03-14", "train_eligible"].any())
        self.assertEqual(feature_columns("gasoline"), ["supply_pretax", "intl_krw_l"])

    def test_knn_causality_and_initial_availability(self):
        dates = pd.bdate_range("2020-01-01", periods=8)
        raw = pd.DataFrame({"supply_pretax": np.arange(8, dtype=float),
                            "fx": [100., 110., 120., 130., np.nan, 150., 160., 170.]}, index=dates)
        a, audit = knn_impute_covariates(raw, ["fx"])
        changed = raw.copy()
        changed.iloc[5:, :] = 999999.
        b, _ = knn_impute_covariates(changed, ["fx"])
        self.assertEqual(a.loc[dates[4], "fx"], b.loc[dates[4], "fx"])
        self.assertLess(audit.source_date.iloc[0], dates[4])
        self.assertTrue(a.loc[raw.fx.notna(), "fx"].equals(raw.loc[raw.fx.notna(), "fx"]))
        self.assertTrue(self.data.groupby("date").fx.nunique().eq(1).all())
        self.assertTrue(self.data.train_eligible.any())

    def test_training_labels_and_inner_early_stopping(self):
        cutoff = pd.Timestamp(cfg.TEST_START)-pd.Timedelta(days=1)
        for horizon in (1, 4):
            samples = get_training_samples(self.data, "gasoline", cutoff, horizon)
            self.assertTrue((samples.target_date <= cutoff).all())
            self.assertTrue((samples.target_date.dt.dayofweek == 5).all())
            self.assertNotIn(pd.Timestamp(cfg.DATA_START), samples.index)
            fitting, stopping, refit = split_xgb_training(self.data, "gasoline", cutoff, horizon)
            self.assertLessEqual(fitting.target_date.max(), stopping.index.min())
            self.assertLess(fitting.target_date.max(), stopping.target_date.min())
            self.assertLessEqual(refit.target_date.max(), cutoff)

    def test_forecaster_shapes_and_calendar_year_grouping(self):
        class Fake:
            config = SimpleNamespace(quantiles=[i/10 for i in range(1, 10)])

            def predict_batch(fake, **kwargs):
                target = kwargs["contexts"][0]
                self.assertEqual(kwargs["past_future_covariates"][0].shape, (4, len(target)+4))
                return [SimpleNamespace(forecast=np.repeat(target[-1], 4), quantiles=np.tile(np.arange(9), (4, 1)))]
        w = self.windows[(self.windows.split == "validation") & (self.windows.context_years == 1)]
        w = w[w.origin == w.origin.min()]
        result = run_rolling_origin(self.data, w, TimesFMPredictor(Fake()), "timesfm3")
        self.assertEqual(len(result), 12)
        self.assertEqual(len(score_predictions(result)), 18)
        c, _, _ = get_window(self.data, w.iloc[0].to_dict())
        self.assertIn(len(c), (52, 53))  # 달력 1년의 주간 관측 수를 확인한다.

    def test_settings_lock_requires_complete_cv(self):
        windows = self.windows[self.windows.split == "validation"]
        tables = []
        for row in windows.to_dict("records"):
            dates = pd.date_range(row["target_start"], row["target_end"], freq=cfg.TIME_FREQ)
            tables.append(pd.DataFrame({"model": "timesfm3", "split": "validation", "fuel": row["fuel"],
                "config_id": row["config_id"], "context_years": row["context_years"], "origin": row["origin"],
                "context_weeks": row["context_weeks"], "date": dates, "lead": np.arange(1,5),
                "actual": 100., "prediction": 100.+row["context_years"]}))
        predictions = pd.concat(tables, ignore_index=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"locked.json"
            with self.assertRaises(ValueError):
                lock_settings(predictions.iloc[4:], path, self.protocol, self.windows)
            contaminated = predictions.copy()
            contaminated.loc[0, "split"] = "test"
            with self.assertRaises(ValueError):
                lock_settings(contaminated, path, self.protocol, self.windows)
            record = lock_settings(predictions, path, self.protocol, self.windows)
            self.assertEqual(record["context_years"], 1)
            self.assertEqual(load_locked_settings(path, self.protocol), record)
            with self.assertRaises(FileExistsError):
                lock_settings(predictions, path, self.protocol, self.windows)
            changed = {**self.protocol, "experiment_id": "different"}
            with self.assertRaises(ValueError):
                load_locked_settings(path, changed)

    def test_cli_blocks_unlocked_test_and_runs_real_smoke(self):
        script = cfg.HERE/"predict_timesfm.py"
        blocked = subprocess.run([sys.executable, str(script), "--mode", "test"], capture_output=True)
        self.assertEqual(blocked.returncode, 2)
        smoke = subprocess.run([sys.executable, str(script), "--mode", "smoke", "--years", "1"],
                               capture_output=True, timeout=90)
        self.assertEqual(smoke.returncode, 0, smoke.stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
