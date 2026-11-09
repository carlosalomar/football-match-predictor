"""Forecast a hypothetical fixture with the trained models (used by the Streamlit demo).

The fixture is appended to the full match history as an unplayed match, and the
*same* feature pipeline used for training computes its features. Because every
feature is lagged, the fixture only sees matches played before it.
"""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .build_dataset import load_elo_params
from .config import DATA_PROCESSED, MODELS_DIR
from .elo import run_elo
from .features import add_team_features, assemble_match_features, to_team_long


@dataclass
class Forecast:
    probs: dict            # model name -> np.array([pH, pD, pA])
    features: pd.Series    # the feature row fed to the models
    shap_home: pd.Series   # SHAP contribution of each feature to P(home win), log-odds
    shap_base: float


class Predictor:
    def __init__(self):
        self.matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
        bundle = joblib.load(MODELS_DIR / "models.joblib")
        self.models, self.features = bundle, bundle["features"]
        self.params = load_elo_params()
        _, self.elo_state = run_elo(self.matches, self.params)
        self._explainer = None

    def current_teams(self) -> dict[str, list[str]]:
        """League code -> teams in the latest season of that league."""
        latest = self.matches[self.matches.season == self.matches.season.max()]
        return {lg: sorted(set(g.home_team) | set(g.away_team)) for lg, g in latest.groupby("league")}

    def last_date(self):
        return self.matches.date.max()

    def team_ratings(self, league: str) -> pd.Series:
        teams = self.current_teams()[league]
        return pd.Series({t: self.elo_state.ratings[t] for t in teams}).sort_values(ascending=False)

    def fixture_features(self, home: str, away: str, date: pd.Timestamp) -> pd.Series:
        league = self.elo_state.league_of[home]
        fixture = pd.DataFrame([{
            "match_id": -1, "league": league, "season": int(self.matches.season.max()),
            "date": pd.Timestamp(date), "home_team": home, "away_team": away, "result": np.nan,
            "home_goals": np.nan, "away_goals": np.nan, "home_shots": np.nan, "away_shots": np.nan,
            "home_shots_target": np.nan, "away_shots_target": np.nan,
        }])
        hist = self.matches[(self.matches.home_team.isin([home, away])) | (self.matches.away_team.isin([home, away]))]
        both = pd.concat([hist, fixture], ignore_index=True)
        long = add_team_features(to_team_long(both))
        rh, ra = self.elo_state.ratings[home], self.elo_state.ratings[away]
        elo = pd.DataFrame([{"match_id": -1, "elo_home": rh, "elo_away": ra,
                             "elo_diff": rh + self.params.home_advantage - ra}])
        X = assemble_match_features(fixture, long, elo)
        return X.iloc[0]

    def forecast(self, home: str, away: str, date) -> Forecast:
        row = self.fixture_features(home, away, date)
        X = row[self.features].to_frame().T.astype(float)
        probs = {
            "LightGBM": self.models["lightgbm"].predict_proba(X)[0],
            "Logistic regression": self.models["logistic"].predict_proba(X)[0],
            "Elo only": self.models["elo_only"].predict_proba(X[["elo_diff"]])[0],
        }
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.models["lightgbm"])
        sv = np.asarray(self._explainer.shap_values(X))
        if sv.ndim == 3 and sv.shape[0] == 3:  # (classes, n, features) in older SHAP
            sv = np.transpose(sv, (1, 2, 0))
        base = np.atleast_1d(self._explainer.expected_value)[0]
        return Forecast(probs=probs, features=row,
                        shap_home=pd.Series(sv[0, :, 0], index=self.features), shap_base=float(base))
