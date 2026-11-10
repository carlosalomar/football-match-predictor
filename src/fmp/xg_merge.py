"""Attach Understat xG to the football-data.co.uk match table.

Team names differ between the two sources ("Ath Madrid" vs "Atletico Madrid"), so
`data/xg/team_names.csv` maps every football-data name to its Understat name. The
script builds a first draft of that file automatically (exact + fuzzy match within
each league), which is then checked by hand, and refuses to continue if any
match from 2014/15 onwards is left without xG.

    python -m src.fmp.xg_merge --draft   # (re)write the draft mapping
    python -m src.fmp.xg_merge           # merge and report
"""
import argparse
import difflib

import pandas as pd

from .config import DATA_PROCESSED, ROOT
from .xg_download import XG_FILE

TEAM_NAMES = ROOT / "data" / "xg" / "team_names.csv"
FIRST_XG_SEASON = 2014


def draft_team_names() -> pd.DataFrame:
    fd = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    us = pd.read_parquet(XG_FILE)
    fd = fd[fd.season >= FIRST_XG_SEASON]
    rows = []
    for lg in sorted(fd.league.unique()):
        fd_teams = sorted(set(fd[fd.league == lg].home_team))
        us_teams = sorted(set(us[us.league == lg].home_team_us))
        for t in fd_teams:
            if t in us_teams:
                match, how = t, "exact"
            else:
                cand = difflib.get_close_matches(t, us_teams, n=1, cutoff=0.0)
                match, how = (cand[0] if cand else ""), "fuzzy - check"
            rows.append({"league": lg, "football_data": t, "understat": match, "how": how})
    out = pd.DataFrame(rows)
    out.to_csv(TEAM_NAMES, index=False)
    return out


def attach_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """Add home_xg / away_xg columns (NaN before 2014/15 or when unavailable)."""
    if not XG_FILE.exists() or not TEAM_NAMES.exists():
        return matches.assign(home_xg=float("nan"), away_xg=float("nan"))
    names = pd.read_csv(TEAM_NAMES)
    to_us = dict(zip(zip(names.league, names.football_data), names.understat))
    us = pd.read_parquet(XG_FILE)
    us = us[us.is_result][["league", "season", "home_team_us", "away_team_us", "kickoff", "home_xg", "away_xg"]]
    m = matches.drop(columns=[c for c in ["home_xg", "away_xg"] if c in matches]).copy()
    m["home_team_us"] = [to_us.get(k) for k in zip(m.league, m.home_team)]
    m["away_team_us"] = [to_us.get(k) for k in zip(m.league, m.away_team)]
    # A home/away pairing happens once per league season, so it is a unique key.
    out = m.merge(us, on=["league", "season", "home_team_us", "away_team_us"], how="left")
    far = (out.kickoff.dt.normalize() - out.date).abs() > pd.Timedelta(days=2)
    out.loc[far, ["home_xg", "away_xg"]] = float("nan")
    return out.drop(columns=["home_team_us", "away_team_us", "kickoff"])


def report(matches: pd.DataFrame) -> pd.DataFrame:
    recent = matches[matches.season >= FIRST_XG_SEASON]
    missing = recent[recent.home_xg.isna()]
    print(f"{len(recent) - len(missing):,} of {len(recent):,} matches since 2014/15 have xG")
    if len(missing):
        print(missing[["league", "season", "date", "home_team", "away_team"]].head(30).to_string())
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()
    if args.draft:
        d = draft_team_names()
        print(d[d.how != "exact"].to_string())
    else:
        m = attach_xg(pd.read_parquet(DATA_PROCESSED / "matches.parquet"))
        if len(report(m)):
            raise SystemExit("Some matches have no xG: fix data/xg/team_names.csv")
