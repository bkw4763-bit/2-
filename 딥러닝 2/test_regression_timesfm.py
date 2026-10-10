"""회귀 공통 자료 실험의 미래 정답 누출과 입력 정렬을 점검한다."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from run_regression_timesfm import inputs, forecast, regression_forecast, policy_input, POLICY_END


class RegressionTimesFMTests(unittest.TestCase):
    def setUp(self):
        """실적 주 기준 공급가·국제가를 만들어 미래 타겟 변경을 검증한다."""
        self.dates = pd.date_range("2020-01-04", periods=110, freq="W-SAT")
        self.weekly = pd.DataFrame({"gasoline_supply_pretax": np.arange(110.)+100,
                                   "gasoline_intl_krw_l": np.arange(110.)+80,
                                   "in_policy": False}, index=self.dates)
        self.cutoff = self.dates[60]

    def test_same_inputs_when_future_target_changes(self):
        a = inputs(self.weekly, "gasoline", self.cutoff, 4, 52)
        changed = self.weekly.copy()
        changed.loc[changed.index >= self.cutoff, "gasoline_supply_pretax"] = 99999
        b = inputs(changed, "gasoline", self.cutoff, 4, 52)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[0].shape, (52,))
        self.assertEqual(a[1].shape, (1, 56))
        np.testing.assert_array_equal(a[1][0, -4:], self.weekly.gasoline_intl_krw_l.iloc[60:64])

    def test_future_prices_are_covariates_not_targets(self):
        class Fake:
            def predict_batch(fake, **kwargs):
                self.assertNotIn("past_only_covariates", kwargs)
                values = kwargs["past_future_covariates"][0][0, -4:]
                return [SimpleNamespace(forecast=values, quantiles=np.tile(values[:, None], (1, 9)))]
        dates, values, _ = forecast(Fake(), self.weekly, "gasoline", self.cutoff, 4, 52)
        self.assertTrue(dates.equals(self.dates[60:64]))
        np.testing.assert_array_equal(values, self.weekly.gasoline_intl_krw_l.iloc[60:64])

    def test_regression_uses_only_past_target(self):
        a = regression_forecast(self.weekly, "gasoline", self.cutoff, "recent_margin")
        changed = self.weekly.copy()
        changed.loc[changed.index >= self.cutoff, "gasoline_supply_pretax"] = 99999
        b = regression_forecast(changed, "gasoline", self.cutoff, "recent_margin")
        np.testing.assert_allclose(a, b)

    def test_insufficient_context_is_rejected(self):
        with self.assertRaises(ValueError):
            inputs(self.weekly, "gasoline", self.cutoff, 4, 104)

    def test_policy_extension_preserves_target_missingness(self):
        weekly = self.weekly.copy()
        for fuel in ("gasoline", "diesel", "kerosene"):
            weekly[f"{fuel}_supply_pretax"] = self.weekly.gasoline_supply_pretax
            weekly[f"{fuel}_intl_krw_l"] = self.weekly.gasoline_intl_krw_l
            weekly[f"{fuel}_tax"] = 100.
            weekly[f"{fuel}_cap"] = 500.
        expanded = policy_input(weekly)
        self.assertGreaterEqual(expanded.index.max(), POLICY_END)
        self.assertEqual(expanded.index.max(), pd.Timestamp("2027-01-02"))
        self.assertTrue(expanded.loc[expanded.index > weekly.index.max(), "gasoline_supply_pretax"].isna().all())
        self.assertEqual(expanded.gasoline_intl_krw_l.iloc[-1], weekly.gasoline_intl_krw_l.iloc[-1])

    def test_weekly_expanding_plan_is_bounded_by_split(self):
        from run_regression_timesfm import regression
        train, test, plan, spec = regression.evaluation_plan(regression.load_weekly())
        self.assertEqual(len(test), int(np.ceil(spec["total_weeks"]*.2)))
        for split, leads in plan.items():
            boundary = train.index.max() if split == "validation" else test.index.max()
            for lead, origins in leads.items():
                self.assertTrue(all(c+pd.Timedelta(weeks=lead-1) <= boundary for c in origins))
                self.assertTrue(all(b-a == pd.Timedelta(weeks=1) for a,b in zip(origins,origins[1:])))
        self.assertEqual(plan["test"][1][0], test.index.min())
        self.assertEqual(len(plan["test"][1]), 117)
        self.assertEqual(len(plan["test"][43]), 75)

    def test_expanding_context_includes_all_past_observations(self):
        from run_regression_timesfm import evaluate
        weekly = self.weekly.copy()
        for fuel in ("diesel", "kerosene"):
            weekly[f"{fuel}_supply_pretax"] = weekly.gasoline_supply_pretax
            weekly[f"{fuel}_intl_krw_l"] = weekly.gasoline_intl_krw_l
        class Fake:
            def predict_batch(fake, **kwargs):
                self.assertEqual(len(kwargs["contexts"][0]), 60)
                values = np.repeat(100., kwargs["horizon"])
                return [SimpleNamespace(forecast=values, quantiles=np.tile(values[:, None], (1, 9)))]
        predictions = evaluate(Fake(), weekly, [self.cutoff], (None,), "validation", horizon=32)
        selected = predictions[predictions.model == "timesfm3"]
        self.assertTrue(selected.context.eq("all").all())
        self.assertTrue(selected.actual_context_weeks.eq(60).all())

    def test_fixed_lead_metrics_do_not_pool_forecast_distances(self):
        from run_regression_timesfm import metrics
        table = pd.DataFrame({"split": ["test"]*4, "model": ["timesfm3"]*4,
            "context": ["all"]*4, "fuel": ["gasoline"]*4,
            "lead": [1,1,4,4], "date": self.dates[:4], "cutoff": self.dates[:4],
            "prediction": [1.,3.,10.,20.], "actual": [0.]*4})
        result = metrics(table).set_index("lead")
        self.assertEqual(result.loc[1,"mae"], 2.)
        self.assertEqual(result.loc[4,"mae"], 15.)

    def test_root_regression_matches_shared_weekly_plan(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from run_regression_timesfm import regression, cfg
        dates = pd.date_range("2015-01-10", periods=300, freq="W-SAT")
        weekly = pd.DataFrame({"in_policy": False}, index=dates)
        for fuel in cfg.FUELS:
            weekly[f"{fuel}_intl_krw_l"] = 100+np.sin(np.arange(300)/7)*20
            weekly[f"{fuel}_supply_pretax"] = weekly[f"{fuel}_intl_krw_l"]+30
        _, _, plan, _ = regression.evaluation_plan(weekly)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cfg, "OUT_DIR", Path(directory)):
                regression.evaluate_holdout(weekly)
                saved = pd.read_csv(Path(directory)/"regression_test_predictions.csv",
                                    parse_dates=["cutoff", "week_end", "train_end"])
            self.assertTrue((saved.train_end < saved.cutoff).all())
            for lead, origins in plan["test"].items():
                for _, group in saved[saved.lead.eq(lead)].groupby(["fuel", "model"]):
                    self.assertEqual(list(group.cutoff), origins)
                    self.assertFalse(group.week_end.duplicated().any())
            self.assertFalse((Path(directory)/"regression_test_tail_predictions.csv").exists())

    def test_fixed_lead_uses_only_past_targets_for_both_models(self):
        from run_regression_timesfm import evaluate
        weekly = self.weekly.copy()
        for fuel in ("diesel", "kerosene"):
            weekly[f"{fuel}_supply_pretax"] = weekly.gasoline_supply_pretax
            weekly[f"{fuel}_intl_krw_l"] = weekly.gasoline_intl_krw_l
        class Fake:
            def predict_batch(fake, **kwargs):
                self.assertEqual(len(kwargs["contexts"][0]), 60)
                values = np.repeat(kwargs["contexts"][0][-1], kwargs["horizon"])
                return [SimpleNamespace(forecast=values, quantiles=np.tile(values[:, None], (1, 9)))]
        a = evaluate(Fake(), weekly, [self.cutoff], (780,), "test", True, horizon=31, score_lead_only=True)
        weekly.loc[weekly.index >= self.cutoff, [f"{f}_supply_pretax"
                   for f in ("gasoline", "diesel", "kerosene")]] = 99999.
        b = evaluate(Fake(), weekly, [self.cutoff], (780,), "test", True, horizon=31, score_lead_only=True)
        np.testing.assert_allclose(a.prediction, b.prediction)
        self.assertTrue(a.groupby(["fuel", "model"]).size().eq(1).all())


if __name__ == "__main__":
    unittest.main()
