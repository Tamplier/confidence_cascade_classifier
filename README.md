
# Installation
```
pip install git+https://github.com/Tamplier/confidence_cascade_classifier.git
```
# Example
```
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from confidence_cascade import ConfidenceCascadeClassifier

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

probas = cascade.predict_proba(X_new)
classes = cascade.predict(X_new)
```
