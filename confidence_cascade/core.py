import numpy as np
from typing import List, Union, Optional, Protocol, Literal, runtime_checkable
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin

@runtime_checkable
class ClassifierOrOptimizer(Protocol):
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...

class ConfidenceCascadeClassifier(BaseEstimator, ClassifierMixin):
  """
  A cascade classifier where each subsequent classifier is trained on the uncertain predictions of the previous one.

  Parameters
  ----------
  classifiers : list
      A list of classifiers or optimizers that implement the `predict_proba` method.
  thresholds : list of float, optional
      Confidence thresholds (maximum predicted probability) for each classifier.
  fit_params : dict | list, optional
      Training parameters:
      - dict (all string keys): applied to all classifiers
      - list: individual parameter sets for each classifier
  verbosity : int
      Verbosity of printing messages:
        0: silent
        1: warning
        2: info
        3: debug

  Aggregation strategies (strategy)
  ---------------------------------
  - 'best_confidence' : selects the classifier with the highest confidence (max probability)
  - 'first_confident' : uses the first classifier whose confidence exceeds its threshold
  - 'mean', 'max', 'min', 'median' : NumPy-based aggregation of predicted probabilities
  """

  def __init__(
      self,
      classifiers: List[ClassifierOrOptimizer],
      thresholds: Optional[List[float]] = None,
      scaled_thresholds: bool = False,
      fit_params: Optional[Union[dict, List[Optional[dict]]]] = None,
      verbosity: Literal[0, 1, 2, 3] = 0,
      ):
    self.classifiers = classifiers
    self.thresholds = thresholds
    self.scaled_thresholds = scaled_thresholds
    self.fit_params = fit_params
    self.verbosity = verbosity

  def _log(self, level: int, msg: str):
    if self.verbosity >= level:
      print(msg)

  def _get_fit_params(self, i: int) -> dict:
    if not self.fit_params:
      return {}
    elif isinstance(self.fit_params, dict):
        return self.fit_params.get(i, {})
    elif isinstance(self.fit_params, list):
      if i < len(self.fit_params):
        return self.fit_params[i] or {}
      else:
        return {}
    else:
      raise ValueError('Incorrect fit params value')

  def _get_threshold(self, i: int, conf: np.ndarray) -> float:
    if not self.thresholds:
      threshold = np.quantile(conf, 1 - (i + 1) / len(self.classifiers))
      return threshold
    else:
      if not self.scaled_thresholds:
        return self.thresholds[i]
      else:
        return conf.max() * self.thresholds[i]

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConfidenceCascadeClassifier':
    self.trained_classifiers = []
    self.classes_ = np.unique(y)
    X = np.array(X)
    y = np.array(y)
    remain_idx = np.arange(len(X))

    for i, clf in enumerate(self.classifiers):
      df_percentage = len(remain_idx) / len(X) * 100
      self._log(2, f'Classifier #{i} uses {df_percentage:.2f}% of initial DataFrame')
      X_sub = X[remain_idx]
      y_sub = y[remain_idx]
      kwargs = self._get_fit_params(i)

      clf.fit(X_sub, y_sub, **kwargs)
      probs = clf.predict_proba(X_sub)
      if hasattr(clf, 'best_params_'):
        self._log(2, f'Classifier #{i} best params: {clf.best_params_}')
      conf = np.max(probs, axis=1)
      self._log(2, f'Classifier #{i} average confidence: {np.mean(conf):.2f}, std: {np.std(conf):.2f}')
      threshold = self._get_threshold(i, conf)
      idx_mask = conf < threshold
      self._log(
        3,
        (
          f"Classifier #{i} max conf: {np.max(conf):.2f},"
          f"min conf: {np.min(conf):.2f},"
          f"quantile: {np.mean(idx_mask)}"
        )
      )
      remain_idx = remain_idx[idx_mask]
      if len(remain_idx) == 0:
        break

      self.trained_classifiers.append(clf)

    return self

  def _best_confidence(self, all_probas: np.ndarray) -> np.ndarray:
    n_samples = all_probas.shape[1]
    confidences = np.max(all_probas, axis=2)
    best_clf_idx = np.argmax(confidences, axis=0)

    final_probas = np.array([all_probas[best_clf_idx[i], i, :] for i in range(n_samples)])
    return final_probas

  def _first_confident(self, all_probas: np.ndarray) -> np.ndarray:
    n_samples = all_probas.shape[1]
    confidences = np.max(all_probas, axis=2)
    final_probas = all_probas[-1].copy()
    remain_idx = np.arange(n_samples)
    for i, threshold in enumerate(self.thresholds):
      idx = remain_idx[confidences[i] >= threshold]
      final_probas[idx] = all_probas[i][idx]
      remain_idx = remain_idx[confidences[i] < threshold]

    return final_probas

  def _np_aggregation(self, all_probas: np.ndarray, func_name: str) -> np.ndarray:
    func = getattr(np, func_name)
    func = partial(func, axis=1)
    return func(all_probas)

  def _aggregate_probas(self, all_probas: np.ndarray, strategy: str) -> np.ndarray:
    match strategy:
      case 'best_confidence':
        return self._best_confidence(all_probas)
      case 'first_confident':
        return self._first_confident(all_probas)
      case _:
        return self._np_aggregation(all_probas, strategy)

  def predict_proba(self, X: np.ndarray, strategy: str = 'best_confidence') -> np.ndarray:
    all_probas = [clf.predict_proba(X) for clf in self.trained_classifiers]
    all_probas = np.array(all_probas)
    assert all(p.shape == all_probas[0].shape for p in all_probas)

    final_probas = self._aggregate_probas(all_probas, strategy)
    return final_probas

  def predict(self, X: np.ndarray, strategy: str ='best_confidence') -> np.ndarray:
    probas = self.predict_proba(X, strategy)
    res_classes = np.argmax(probas, axis=1)
    return res_classes
