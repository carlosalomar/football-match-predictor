"""Model definitions and bookmaker probabilities."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTCOMES

# Classes are always encoded H=0, D=1, A=2 so probability columns line up everywhere.
LABEL = {o: i for i, o in enumerate(OUTCOMES)}


def encode(y: pd.Series) -> np.ndarray:
    return y.map(LABEL).to_numpy()


class PriorBaseline:
    """'The home team always wins', in probabilistic form: predict the training
    frequencies of H / D / A for every match (log loss needs probabilities, and a
    hard 100% home win would score infinity on the first draw)."""

    def fit(self, X, y):
        self.p_ = np.bincount(y, minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


def logistic(C: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def lightgbm(n_estimators: int = 2000, **overrides):
    params = dict(objective="multiclass", n_estimators=n_estimators, learning_rate=0.01,
                  num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=42)
    params.update(overrides)
    return LGBMClassifier(**params)


def implied_probabilities(odds: pd.DataFrame) -> np.ndarray:
    """Turn decimal odds (H, D, A) into probabilities.

    1/odds sums to more than 1 because of the bookmaker's margin (the 'overround');
    dividing by that sum spreads the margin proportionally across the outcomes.
    """
    inv = 1.0 / odds.to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)
