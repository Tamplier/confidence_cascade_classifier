# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init

from unittest.mock import MagicMock
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from confidence_cascade import ConfidenceCascadeClassifier

TEST_PROBAS = [
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ],
    [
        [0.85, 0.15, 0],
        [1, 0, 0],
        [0.33, 0.33, 0.33]
    ],
    [
        [0.8, 0.15, 0.5],
        [0, 1, 0],
        [0.25, 0.5, 0.25]
    ]
]

class DummyClassifier(BaseEstimator):
    def fit(self, _X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba_col = X[:, 1].reshape(-1, 1)
        return np.tile(proba_col, reps=(1, 3))

    def predict(self, X):
        return X[:, 0]

def make_cascade(**kwargs):
    defaults = dict(thresholds=None, scaled_thresholds=False, classifiers=None, fit_params=None)
    defaults.update(kwargs)
    return ConfidenceCascadeClassifier(**defaults)

@pytest.mark.parametrize(
    'i,fit_params,expected',
    [
        (0, None, {}),
        (5, {'a': 1}, {'a': 1}),
        (1, [{'a': 1}, {'b': 2}], {'b': 2})
    ]
)
def test_get_fit_params(i, fit_params, expected):
    cascade = make_cascade(fit_params=fit_params)
    params = cascade._get_fit_params(i)
    assert params == expected

class TestGetThreshold:
    confidences = [0.9, 0.9, 0.8]
    thresholds = [0.9, 0.8, 0.7]

    def test_none_value(self):
        cascade = make_cascade(classifiers=[0, 0, 0, 0])
        quantiles = [0.75, 0.5, 0.25, 0]
        for i, quantile in enumerate(quantiles):
            threshold_received = cascade._get_threshold(i, self.confidences)
            assert threshold_received == np.quantile(self.confidences, quantile)

    def test_not_scaled(self):
        cascade = make_cascade(thresholds=self.thresholds)
        for i, threshold_required in enumerate(self.thresholds):
            threshold_received = cascade._get_threshold(i, self.confidences)
            assert threshold_received == threshold_required

    def test_scaled(self):
        cascade = make_cascade(thresholds=self.thresholds, scaled_thresholds=True)
        max_confidence = np.max(self.confidences)
        for i, threshold_orig in enumerate(self.thresholds):
            threshold_required = threshold_orig * max_confidence
            threshold_received = cascade._get_threshold(i, self.confidences)
            assert threshold_received == threshold_required

@pytest.mark.parametrize(
    'all_probas,expected',
    [
      (
          TEST_PROBAS,
          [
              [1, 0, 0],
              [0, 0, 1],
              [0, 1, 0]
          ]
      )
    ]
)
def test_best_confidence(all_probas, expected):
    cascade = make_cascade()
    result = cascade._best_confidence(np.array(all_probas))
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize(
    'func_name,expected',
    [
        (
            'mean',
            [
                [0.883, 0.099, 0.166],
                [0.333, 0.333, 0.333],
                [0.193, 0.61, 0.193]
            ]
        ),
        (
            'median',
            [
                [0.85, 0.15, 0.0],
                [0, 0, 0],
                [0.25, 0.5, 0.25]
            ]
        ),
        (
            'max',
            [
                [1, 0.15, 0.5],
                [1, 1, 1],
                [0.33, 1, 0.33]
            ]
        )
    ]
)
def test_aggregation_methods(func_name, expected):
    cascade = make_cascade()
    result = cascade._np_aggregation(TEST_PROBAS, func_name)
    np.testing.assert_allclose(result, expected, atol=1e-3)

@pytest.mark.parametrize(
    'mocked_return,expected',
    [
        (
            [
                [0.8, 0.1, 0.1],
                [0.25, 0.5, 0.25],
                [0, 0, 1],
                [0.3, 0.3, 0.3]
            ],
            [0, 1, 2, 0]
        )
    ]
)
def test_predict(mocked_return, expected, monkeypatch):
    cascade = make_cascade()
    def mock_predict_proba(_self, _X, _strategy):
        return mocked_return
    monkeypatch.setattr(ConfidenceCascadeClassifier, 'predict_proba', mock_predict_proba)
    y = cascade.predict(None)
    np.testing.assert_array_equal(y, expected)

@pytest.mark.parametrize(
    'thresholds,X,scale,first_fitted,expected',
    [
      (
          [0.9, 0.5, 0],
          [
              [0, 0.9],
              [0, 0.5],
              [0, 0.49],
              [0, 0]
          ],
          False,
          False,
          [4, 3, 2]
      ),
      (
          [0.9, 0.9, 0],
          [
              [0, 0.2],
              [0, 0.15],
              [0, 0.14],
              [0, 0.1],
              [0, 0]
          ],
          True,
          False,
          [5, 4, 2]
      ),
      (
          [0.9, 0],
          [
              [0, 0.91],
              [0, 0.89]
          ],
          False,
          True,
          [1] # Skip first classifier because it's already trained
      )
    ]
)
def test_fit(thresholds, X, scale, first_fitted, expected, monkeypatch):
    y = np.array(X)[:, 0]
    clfs_amount = len(thresholds)
    clfs = [DummyClassifier() for _i in range(clfs_amount)]
    if first_fitted:
        clfs[0].fit(X, y)
    mock_fit = MagicMock()
    for clf in clfs:
        monkeypatch.setattr(clf, "fit", mock_fit)
    cascade = make_cascade(thresholds=thresholds, scaled_thresholds=scale, classifiers=clfs)
    cascade.fit(X, y)
    for expected_len, call in zip(expected, mock_fit.call_args_list):
        X_arg = call.args[0]
        assert len(X_arg) == expected_len

def test_predict_proba():
    pass

def test_first_confident():
    pass
