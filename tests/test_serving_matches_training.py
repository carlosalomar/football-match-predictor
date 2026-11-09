"""The demo must compute exactly the same features as the training pipeline."""
import numpy as np
import pandas as pd

from src.fmp.config import DATA_PROCESSED
from src.fmp.elo import run_elo
from src.fmp.predict import Predictor


def test_fixture_features_equal_training_features():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    p = Predictor()
    target = feats[(feats.season == 2024) & (feats.league == "SP1")].iloc[100]
    # Rewind the predictor to the morning of the match.
    p.matches = p.matches[p.matches.date < target.date]
    _, p.elo_state = run_elo(p.matches, p.params)
    served = p.fixture_features(target.home_team, target.away_team, target.date)
    for col in p.features:
        assert np.isclose(float(served[col]), float(target[col]), equal_nan=True), col
