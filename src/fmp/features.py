"""Pre-match features. The golden rule: a match's features only use earlier matches.

The pipeline is:

1. `to_team_long`  – every match becomes two rows, one per team (team / opponent view).
2. `add_team_features` – per team, rolling statistics of *previous* matches. Every
   rolling window is computed on a series that was `shift(1)`-ed first, so the
   current match never sees its own result.
3. `assemble_match_features` – the home team's row and the away team's row are
   joined back into one row per match, together with the Elo ratings.

The demo app re-uses steps 2 and 3 for a hypothetical fixture, so training and
serving share exactly the same feature definitions.
"""
import numpy as np
import pandas as pd

from .config import LEAGUES

FORM_WINDOW = 5
REST_CAP_DAYS = 21  # anything longer is an international / winter / summer break

TEAM_STATS = ["points", "gf", "ga", "shots_f", "shots_a", "sot_f", "sot_a"]
VENUE_STATS = ["points", "gf", "ga"]


def to_team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one from the home team's point of view, one from the away team's."""
    common = ["match_id", "league", "season", "date"]
    pts_home = np.select([matches.result == "H", matches.result == "D"], [3, 1], 0)
    home = matches[common].assign(
        team=matches.home_team, opponent=matches.away_team, is_home=1,
        gf=matches.home_goals, ga=matches.away_goals,
        shots_f=matches.home_shots, shots_a=matches.away_shots,
        sot_f=matches.home_shots_target, sot_a=matches.away_shots_target,
        points=pts_home,
    )
    pts_away = np.select([matches.result == "A", matches.result == "D"], [3, 1], 0)
    away = matches[common].assign(
        team=matches.away_team, opponent=matches.home_team, is_home=0,
        gf=matches.away_goals, ga=matches.home_goals,
        shots_f=matches.away_shots, shots_a=matches.home_shots,
        sot_f=matches.away_shots_target, sot_a=matches.home_shots_target,
        points=pts_away,
    )
    long = pd.concat([home, away], ignore_index=True)
    # Unplayed fixtures (used by the demo) have no result: keep their stats missing.
    unplayed = long["match_id"].isin(matches.loc[matches.result.isna(), "match_id"])
    long.loc[unplayed, ["points"] + TEAM_STATS[1:]] = np.nan
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _lagged_rolling(g: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    # shift(1) first: the value for a match is the mean of the `window` matches BEFORE it.
    return g.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def add_team_features(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["team", "date", "match_id"]).copy()
    by_team = long.groupby("team", sort=False)

    # Recent form over all matches (home and away), last 5 league matches.
    for col in TEAM_STATS:
        long[f"form_{col}"] = _lagged_rolling(by_team[col], FORM_WINDOW)

    # Venue-specific form: the home team's last 5 HOME matches, the away team's last 5 AWAY matches.
    by_team_venue = long.groupby(["team", "is_home"], sort=False)
    for col in VENUE_STATS:
        long[f"venue_{col}"] = _lagged_rolling(by_team_venue[col], FORM_WINDOW)

    # Season to date: points per game and matches played before this one.
    by_team_season = long.groupby(["team", "season"], sort=False)
    long["season_ppg"] = by_team_season["points"].transform(lambda s: s.shift(1).expanding().mean())
    long["season_played"] = by_team_season.cumcount()

    # Rest: days since the team's previous league match.
    long["prev_date"] = by_team["date"].shift(1)
    return long


FEATURE_COLS_TEAM = (
    [f"form_{c}" for c in TEAM_STATS] + [f"venue_{c}" for c in VENUE_STATS]
    + ["season_ppg", "season_played"]
)


def assemble_match_features(fixtures: pd.DataFrame, long_feats: pd.DataFrame,
                            elo: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: home features (h_*), away features (a_*), Elo and differences."""
    cols = ["match_id", "team"] + FEATURE_COLS_TEAM + ["prev_date"]
    h = long_feats[cols].rename(columns={c: f"h_{c}" for c in cols if c != "match_id"})
    a = long_feats[cols].rename(columns={c: f"a_{c}" for c in cols if c != "match_id"})
    X = (fixtures[["match_id", "league", "season", "date", "home_team", "away_team"]]
         .merge(h, left_on=["match_id", "home_team"], right_on=["match_id", "h_team"])
         .merge(a, left_on=["match_id", "away_team"], right_on=["match_id", "a_team"])
         .drop(columns=["h_team", "a_team"]))
    X = X.merge(elo[["match_id", "elo_home", "elo_away", "elo_diff"]], on="match_id")

    for side in "ha":
        rest = (X["date"] - X[f"{side}_prev_date"]).dt.days
        X[f"{side}_rest_days"] = rest.clip(upper=REST_CAP_DAYS)
        X = X.drop(columns=f"{side}_prev_date")
    X["rest_diff"] = X["h_rest_days"] - X["a_rest_days"]
    for c in ["form_points", "form_gf", "form_ga", "form_sot_f", "form_sot_a", "season_ppg"]:
        X[f"diff_{c}"] = X[f"h_{c}"] - X[f"a_{c}"]
    for code in LEAGUES:
        X[f"league_{code}"] = (X["league"] == code).astype(int)
    return X


def model_feature_names() -> list[str]:
    team = [f"{s}_{c}" for s in "ha" for c in FEATURE_COLS_TEAM + ["rest_days"]]
    diffs = ["rest_diff"] + [f"diff_{c}" for c in
                             ["form_points", "form_gf", "form_ga", "form_sot_f", "form_sot_a", "season_ppg"]]
    return ["elo_home", "elo_away", "elo_diff"] + team + diffs + [f"league_{c}" for c in LEAGUES]


def build_feature_table(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    long = add_team_features(to_team_long(matches))
    X = assemble_match_features(matches, long, elo)
    keep = ["match_id", "result", "home_goals", "away_goals"] + [c for c in matches if c.startswith("odds_")]
    return X.merge(matches[keep], on="match_id").sort_values(["date", "match_id"]).reset_index(drop=True)
