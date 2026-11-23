"""Hybrid RF + ANN models for ViewCast LK.
Kept in a separate file so the notebook AND the Streamlit dashboard can both import them
(joblib needs the class definitions to load the saved models)."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_predict


class BlendRegressor(RegressorMixin, BaseEstimator):
    """Hybrid A: final prediction = w * RF + (1 - w) * ANN."""
    def __init__(self, rf=None, ann=None, w=0.5):
        self.rf = rf
        self.ann = ann
        self.w = w

    def fit(self, X, y):
        self.rf_ = clone(self.rf).fit(X, y)
        self.ann_ = clone(self.ann).fit(X, y)
        return self

    def predict(self, X):
        return self.w * self.rf_.predict(X) + (1 - self.w) * self.ann_.predict(X)


class RFtoANN(RegressorMixin, BaseEstimator):
    """Hybrid B: the RF prediction is added as an extra input feature to the ANN.
    On training data the RF prediction is out-of-fold, so the ANN never sees leaked answers."""
    def __init__(self, rf=None, ann=None, random_state=42):
        self.rf = rf
        self.ann = ann
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        inner_cv = KFold(5, shuffle=True, random_state=self.random_state)
        rf_oof = cross_val_predict(clone(self.rf), X, y, cv=inner_cv)
        self.rf_ = clone(self.rf).fit(X, y)
        self.ann_ = clone(self.ann).fit(np.column_stack([X, rf_oof]), y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.ann_.predict(np.column_stack([X, self.rf_.predict(X)]))
