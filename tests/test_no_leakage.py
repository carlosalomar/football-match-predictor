"""Leakage tests: changing a match's result must not change that match's own features."""
import pandas as pd
import pytest

from src.fmp.config import DATA_PROCESSED
from src.fmp.elo import EloParams, run_elo
from src.fmp.features import build_feature_table, model_feature_names


@pytest.fixture(scope="module")
def matches():
    m = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    return m[(m.league == "SP1") & m.season.between(2015, 2017)].reset_index(drop=True)


def _features(m):
    elo, _ = run_elo(m, EloParams())
    elo["match_id"] = m["match_id"]
    return build_feature_table(m, elo).set_index("match_id")[model_feature_names()]


def test_flipping_a_result_does_not_change_its_own_features(matches):
    base = _features(matches)
    target = matches.index[len(matches) // 2]
    tampered = matches.copy()
    # Turn the match into a 9-0 home win with absurd shot numbers.
    tampered.loc[target, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    tampered.loc[target, ["home_shots", "home_shots_target"]] = [60, 40]
    after = _features(tampered)
    mid = matches.loc[target, "match_id"]
    pd.testing.assert_series_equal(base.loc[mid], after.loc[mid])

    # ...but later matches of the same teams DO change (the features are live).
    later = matches[(matches.date > matches.loc[target, "date"]) &
                    (matches.home_team == matches.loc[target, "home_team"])].match_id.iloc[0]
    assert not base.loc[later].equals(after.loc[later])


def test_first_match_of_each_team_has_no_form(matches):
    feats = _features(matches)
    long = pd.concat([matches.assign(team=matches.home_team, side="h"),
                      matches.assign(team=matches.away_team, side="a")])
    first = long.sort_values(["date", "match_id"]).groupby("team").head(1)
    for _, row in first.iterrows():
        # A team's first match in the sample has no history, so form must be missing.
        assert pd.isna(feats.loc[row.match_id, f"{row.side}_form_points"])
