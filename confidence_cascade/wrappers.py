# pylint: disable=invalid-name
# pylint: disable=missing-docstring

class FrozenClassifier:
    def __init__(self, fitted_classifier):
        self.fitted_classifier = fitted_classifier
        self.classes_ = None

    def fit(self, _X, _y):
        return self

    def predict(self, X):
        return self.fitted_classifier.predict(X)

    def predict_proba(self, X):
        return self.fitted_classifier.predict_proba(X)
