"""공동 예측의 날짜 정렬·미래 정답 차단·유종별 출력 매핑을 검증한다."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from compare_timesfm_joint import FUELS, joint_inputs, joint_forecast, evaluate_joint


class JointTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-04", periods=100, freq="W-SAT")
        self.weekly = pd.DataFrame(index=self.dates)
        for i, fuel in enumerate(FUELS):
            self.weekly[f"{fuel}_supply_pretax"] = np.arange(100.) + i*100
            self.weekly[f"{fuel}_intl_krw_l"] = np.arange(100.) + i*200
        self.cutoff = self.dates[60]

    def test_alignment_and_future_target_isolation(self):
        a = joint_inputs(self.weekly, self.cutoff, 4, 52)
        changed = self.weekly.copy()
        for fuel in FUELS:
            changed.loc[self.cutoff:, f"{fuel}_supply_pretax"] = np.nan
        b = joint_inputs(changed, self.cutoff, 4, 52)
        self.assertEqual(a[0].shape, (3, 52))
        self.assertEqual(a[1].shape, (3, 56))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        for i, fuel in enumerate(FUELS):
            np.testing.assert_array_equal(a[1][i, -4:], self.weekly.loc[self.dates[60:64], f"{fuel}_intl_krw_l"])

    def test_joint_call_and_fuel_mapping(self):
        calls = []
        class Fake:
            config = SimpleNamespace(quantiles=[.1, .5, .9])
            def predict_batch(fake, **kwargs):
                calls.append(kwargs)
                self.assertEqual(kwargs["contexts"][0].shape, (3, 60))
                values = kwargs["past_future_covariates"][0][:, -kwargs["horizon"]:]
                return [SimpleNamespace(forecast=values, quantiles=np.repeat(values[:, :, None], 3, axis=2))]
        results = evaluate_joint(Fake(), self.weekly, [self.cutoff], "test")
        self.assertEqual(len(calls), 1)
        self.assertTrue(results.actual_context_weeks.eq(60).all())
        for fuel in FUELS:
            np.testing.assert_array_equal(results.loc[results.fuel.eq(fuel), "prediction"],
                self.weekly[f"{fuel}_intl_krw_l"].iloc[60:92])

    def test_internal_missing_week_is_rejected(self):
        broken = self.weekly.copy()
        broken.loc[self.dates[50], f"{FUELS[1]}_supply_pretax"] = np.nan
        with self.assertRaises(ValueError):
            joint_inputs(broken, self.cutoff, 4, 52)


if __name__ == "__main__":
    unittest.main()
