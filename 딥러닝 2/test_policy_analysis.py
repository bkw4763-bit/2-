"""국제가격 시나리오와 정책 이후 정답 분리·결측 격차를 검증한다."""
from types import SimpleNamespace
import unittest

import numpy as np

import analyze_policy as analysis


class PolicyAnalysisTests(unittest.TestCase):
    def setUp(self):
        """회귀와 동일한 원자료로 시나리오 입력을 점검한다."""
        self.weekly = analysis.experiment.regression.load_weekly()

    def test_scenario_changes_only_unobserved_international_prices(self):
        for direction in (-1, 0, 1):
            scenario = analysis.scenario_data(self.weekly, direction, 0.10)
            for fuel in analysis.experiment.cfg.FUELS:
                column = f"{fuel}_intl_krw_l"
                np.testing.assert_allclose(scenario.loc[self.weekly.index, column], self.weekly[column])
                self.assertAlmostEqual(scenario[column].iloc[-1], self.weekly[column].iloc[-1]*(1+direction*0.10))
                self.assertTrue(scenario.loc[scenario.index > self.weekly.index.max(), f"{fuel}_supply_pretax"].isna().all())

    def test_future_targets_do_not_affect_predictions(self):
        class Fake:
            def predict_batch(fake, **kwargs):
                values = kwargs["past_future_covariates"][0][0, -kwargs["horizon"]:]+80
                return [SimpleNamespace(forecast=values, quantiles=np.tile(values[:, None], (1, 9)))]
        a = analysis.build_paths(Fake(), self.weekly, 156, 0.10)
        changed = self.weekly.copy()
        for fuel in analysis.experiment.cfg.FUELS:
            changed.loc[changed.in_policy, f"{fuel}_supply_pretax"] = 99999
        b = analysis.build_paths(Fake(), changed, 156, 0.10)
        np.testing.assert_allclose(a.timesfm_aftertax, b.timesfm_aftertax)
        np.testing.assert_allclose(a.recent_margin_aftertax, b.recent_margin_aftertax)
        self.assertEqual(a.date.max(), analysis.experiment.POLICY_END)
        self.assertEqual(a.week_start.min().strftime("%Y-%m-%d"), "2026-03-13")
        self.assertTrue(a.loc[a.actual_aftertax.isna(), "timesfm_gap_to_actual"].isna().all())
        summary = analysis.summarize(a)
        self.assertEqual(len(summary), 27)


if __name__ == "__main__":
    unittest.main()
