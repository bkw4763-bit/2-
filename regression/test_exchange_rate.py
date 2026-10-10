"""환율 추가에 관련된 입력·계수·문맥 검증."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'regression/full_sample'))
sys.path.insert(0, str(ROOT / '딥러닝'))
import counterfactual as cf
import load_training_set as loader
from cv_timesfm import fold_inputs, prepare_normalized_fold
from run_timesfm import forecast


class ExchangeRateTests(unittest.TestCase):
    def test_coefficient_units_and_margin(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(180, 4)), columns=['lag0', 'lag1', 'lag2', 'exchange_rate'])
        X['exchange_rate'] = 1200 + X['exchange_rate'] * 50
        y = pd.Series(100 + X.to_numpy() @ np.array([0.4, 0.3, 0.3, 0.07]))
        params = cf.fit(X, y)
        self.assertAlmostEqual(params['weights'][:3].sum(), 1)
        self.assertAlmostEqual(params['weights'][3], 0.07)
        self.assertAlmostEqual(params['const'], 100)
        np.testing.assert_allclose(cf.predict(params, X), y, atol=1e-9)

    def test_only_international_weights_normalized(self):
        np.testing.assert_allclose(cf.normalize_weights([1, 0.6, 0.4, 0.08], 3), [0.5, 0.3, 0.2, 0.08])

    def test_standardization_preserves_physical_model(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame(rng.normal(size=(200, 4)) * [100, 120, 150, 70] + [600, 610, 620, 1200],
                         columns=['lag0', 'lag1', 'lag2', 'exchange_rate'])
        y = pd.Series(80 + X.to_numpy() @ np.array([0.6, 0.4, 0.2, 0.03]))
        scale, mean = X.std(ddof=0).to_numpy(), X.mean().to_numpy()
        Z = (X - mean) / scale
        raw_params, standardized = cf.fit(X, y), cf.fit(Z, y, feature_scale=scale)
        np.testing.assert_allclose(cf.predict(raw_params, X), cf.predict(standardized, Z), atol=1e-9)
        self.assertAlmostEqual((standardized['weights'] / scale)[:3].sum(), 1)

    def test_normalized_files_and_fold_scaler_use_only_past(self):
        product = next(iter(loader.MODELS))
        data = loader.load_normalized_training_set(product)
        fold = prepare_normalized_fold(data, 151)
        np.testing.assert_allclose(fold['X'][:151].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(fold['X'][:151].std(axis=0), 1, atol=1e-12)
        changed = {**data, 'X': data['X'].copy()}
        changed['X'][151:] += 100
        next_fold = prepare_normalized_fold(changed, 151)
        np.testing.assert_allclose(fold['scaler']['mean'], next_fold['scaler']['mean'], atol=1e-12)
        np.testing.assert_allclose(fold['scaler']['scale'], next_fold['scaler']['scale'], atol=1e-12)
        self.assertEqual(fold['scaler']['train_end'], next_fold['scaler']['train_end'])
        np.testing.assert_allclose(fold['X'][:151], next_fold['X'][:151], atol=1e-12)

    def test_lag_selection_keeps_exchange(self):
        X = pd.DataFrame(np.zeros((3, 6)), columns=[f'lag{k}' for k in range(5)] + ['exchange_rate'])
        self.assertEqual(list(cf.select_lags(X, 2)), ['lag0', 'lag1', 'exchange_rate'])

    def test_shared_inputs_and_future_target_boundary(self):
        weekly = loader.load_weekly_table()
        product = next(iter(loader.MODELS))
        data = loader.load_training_set(product, weekly)
        self.assertEqual(data['X'].shape[1], 4)
        np.testing.assert_allclose(data['X'][:, 3], weekly.loc[pd.to_datetime(data['날짜']), 'exchange_rate'], rtol=1e-6)
        context, covariates, actual = fold_inputs(data, 151, 32)
        changed = {**data, 'y': data['y'].copy()}
        changed['y'][151:] += 1000
        context2, covariates2, actual2 = fold_inputs(changed, 151, 32)
        np.testing.assert_array_equal(context, context2)
        np.testing.assert_array_equal(covariates, covariates2)
        self.assertNotEqual(actual, actual2)
        self.assertEqual(covariates.shape, (4, 183))

    def test_timesfm_receives_exchange_channel(self):
        from types import SimpleNamespace
        class Model:
            config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])
            def predict_batch(self, **kwargs):
                self.received = kwargs
                return [SimpleNamespace(forecast=np.ones(2), quantiles=np.ones((2, 3)))]
        model = Model()
        features = np.arange(20).reshape(4, 5)
        forecast(model, np.ones(3), features, 2)
        np.testing.assert_array_equal(model.received['past_future_covariates'][0][3], features[3])


if __name__ == '__main__':
    unittest.main()
