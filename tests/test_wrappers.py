# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init

from sklearn.base import clone
import numpy as np
import pytest
from confidence_cascade import FrozenClassifier
from tests.utils import DummyClassifier, make_cascade

@pytest.mark.parametrize(
    'frozen',
    [
        ([True, True, False, False, True])
    ]
)
def test_clone_behaviour(frozen):
    clfs = [DummyClassifier().fit(None, None) for _i in range(len(frozen))]
    clfs = [FrozenClassifier(clf) if freeze else clf for clf, freeze in zip(clfs, frozen)]
    cascade = make_cascade(classifiers=clfs)
    cloned_cascade = clone(cascade)
    cloned_classifiers = [
        clf.fitted_classifier if hasattr(clf, 'fitted_classifier') else clf
        for clf in cloned_cascade.classifiers
    ]
    fitted = [cloned_cascade._is_classifier_fitted(clf) for clf in cloned_classifiers]
    assert all(isinstance(obj, DummyClassifier) for obj in cloned_classifiers)
    np.testing.assert_array_equal(fitted, frozen)
