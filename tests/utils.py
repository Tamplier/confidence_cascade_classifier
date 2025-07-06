# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init

from sklearn.base import BaseEstimator
import numpy as np
from confidence_cascade import ConfidenceCascadeClassifier

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
