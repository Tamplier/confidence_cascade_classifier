
# Installation
```
pip install git+https://github.com/Tamplier/confidence_cascade_classifier.git
```
# Example
```
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from confidence_cascade import ConfidenceCascadeClassifier, FrozenClassifier

xgb_params = {
  # Search params
}
search_xgb = RandomizedSearchCV(
XGBClassifier(),
  xgb_params,
  # Other params
)

search_lgb = LGBMClassifier()
search_et = ExtraTreesClassifier()

cascade = ConfidenceCascadeClassifier(
  classifiers=[search_xgb, search_lgb, search_et],
  thresholds=[0.9, 0.8, 0.0],
  verbosity=2
)
cascade.fit(X_train, y)

probas = cascade.predict_proba(X_test)
classes = cascade.predict(X_test)
```

It's also possible to use the classifier with hyper-parameter optimizers.
But in that case every inner classifier will be retrained for all param grid combinations.

```
xgb_classifier = search_xgb.best_estimator_
# It's not required but first classifier always will use 100% of initial DataFrame
# So it will be one and the same classifier for all iterations
# And this wrapper prevents optimizer from refitting first classifier
xgb_classifier = FrozenClassifier(xgb_classifier)

cascade = ConfidenceCascadeClassifier(
  classifiers=[xgb_classifier, search_lgb, search_et],
  thresholds=[0.9, 0.8, 0.0],
  verbosity=2
)

cascade_params_grid = {
  'scaled_thresholds': [True, False],
  'thresholds': [
    None,
    [0.9, 0.9, 0.0],
    [0.9, 0.75, 0.0]
  ]
}

global_search = GridSearchCV(
  cascade,
  cascade_params_grid,
  ...
)

global_search.fit(X_train, y)
classes = global_search.predict(X_test)
```
