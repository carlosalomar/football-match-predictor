"""A hand-written Elo rating system for club football.

Every team carries a single number. Before a match the expected score of the home
team is

    E_home = 1 / (1 + 10 ** ((R_away - (R_home + HFA)) / 400))

and after the match both ratings move by K * G * (S - E), where S is 1 / 0.5 / 0
for a win / draw / loss and G is an optional margin-of-victory multiplier.

Two details matter for leagues with promotion and relegation:

* A promoted team does not start at 1500 (that would make it an average side).
  It inherits the mean final rating of the teams relegated from that league the
  season before, which is a good proxy for "a team at the bottom of this league".
* Between seasons every rating is pulled part of the way back to the league mean,
  because squads change over the summer.

The ratings recorded for a match are always the ratings *before* kick-off, so
they only use information from earlier matches (no leakage).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0


@dataclass
class EloParams:
    k: float = 20.0
    home_advantage: float = 70.0
    season_reversion: float = 0.2      # share of the gap to the league mean removed each summer
    margin_of_victory: bool = True


@dataclass
class EloState:
    """Ratings at the end of the processed history (used by the demo app)."""
    ratings: dict = field(default_factory=dict)          # team -> rating
    league_of: dict = field(default_factory=dict)        # team -> league code


def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - (r_home + hfa)) / 400.0))


def _mov_multiplier(goal_diff: int, rating_gap: float) -> float:
    """Margin-of-victory multiplier (as in FiveThirtyEight's NFL/club Elo).

    Bigger wins move ratings more, damped when the favourite wins big, which
    stops strong teams' ratings from inflating.
    """
    if goal_diff == 0:
        return 1.0
    return np.log(abs(goal_diff) + 1) * (2.2 / (rating_gap * 0.001 + 2.2))


def run_elo(matches: pd.DataFrame, params: EloParams = EloParams()) -> tuple[pd.DataFrame, EloState]:
    """Walk through matches in date order and return pre-match ratings for each one.

    `matches` must contain: match_id, league, season, date, home_team, away_team,
    home_goals, away_goals. Returns a DataFrame indexed like `matches` with
    elo_home, elo_away, elo_diff (home minus away, including home advantage) and
    elo_exp_home.
    """
    m = matches.sort_values(["date", "match_id"])
    ratings: dict[str, float] = {}
    league_of: dict[str, str] = {}
    season_teams: dict[tuple[str, int], set] = {}
    for (lg, s), g in m.groupby(["league", "season"]):
        season_teams[(lg, s)] = set(g["home_team"]) | set(g["away_team"])

    current_season: dict[str, int] = {}  # league -> season being processed
    out = np.empty((len(m), 3))
    for i, row in enumerate(m[["league", "season", "home_team", "away_team",
                               "home_goals", "away_goals"]].itertuples(index=False)):
        lg, season, ht, at, hg, ag = row
        if current_season.get(lg) != season:
            _start_season(lg, season, ratings, league_of, season_teams, params)
            current_season[lg] = season
        rh, ra = ratings[ht], ratings[at]
        e = expected_home(rh, ra, params.home_advantage)
        out[i] = (rh, ra, e)
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = 1.0
        if params.margin_of_victory and hg != ag:
            winner_gap = (rh + params.home_advantage - ra) if hg > ag else (ra - rh - params.home_advantage)
            mult = _mov_multiplier(hg - ag, winner_gap)
        delta = params.k * mult * (s - e)
        ratings[ht] = rh + delta
        ratings[at] = ra - delta

    res = pd.DataFrame(out, index=m.index, columns=["elo_home", "elo_away", "elo_exp_home"])
    res["elo_diff"] = res["elo_home"] + params.home_advantage - res["elo_away"]
    return res.loc[matches.index], EloState(ratings=ratings, league_of=league_of)


def _start_season(lg, season, ratings, league_of, season_teams, params):
    teams = season_teams[(lg, season)]
    prev = season_teams.get((lg, season - 1), set())
    known = [t for t in prev if t in ratings]
    league_mean = float(np.mean([ratings[t] for t in known])) if known else INITIAL_RATING
    # Regress last season's teams toward the league mean.
    for t in known:
        ratings[t] = league_mean + (1 - params.season_reversion) * (ratings[t] - league_mean)
    relegated = [t for t in known if t not in teams]
    promoted_rating = float(np.mean([ratings[t] for t in relegated])) if relegated else INITIAL_RATING
    for t in teams:
        if t not in prev:
            # Promoted team (or very first season of the data).
            ratings[t] = promoted_rating if prev else INITIAL_RATING
        league_of[t] = lg
