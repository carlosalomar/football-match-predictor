"""Download match-level expected goals (xG) from Understat with `soccerdata`.

Understat covers the big five leagues from 2014/15. soccerdata keeps a local
cache (SOCCERDATA_DIR), so finished seasons are only downloaded once; the
current season is always refreshed.

Usage:
    python -m src.fmp.xg_download            # all seasons 2014/15 -> current
    python -m src.fmp.xg_download --current  # only refresh the current season
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from .config import ROOT, SEASONS

XG_DIR = ROOT / "data" / "xg"
XG_FILE = XG_DIR / "understat_matches.parquet"

UNDERSTAT_LEAGUES = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "GER-Bundesliga": "D1",
    "ITA-Serie A": "I1",
    "FRA-Ligue 1": "F1",
}
FIRST_XG_SEASON = "1415"


class DownloadError(RuntimeError):
    """Raised when the download is incomplete, so nothing downstream runs on partial data."""


def _seasons(current_only: bool) -> list[str]:
    seasons = [s for s in SEASONS if s >= FIRST_XG_SEASON]
    return seasons[-1:] if current_only else seasons


def fetch(current_only: bool = False) -> pd.DataFrame:
    os.environ.setdefault("SOCCERDATA_DIR", str(ROOT / "data" / "cache" / "soccerdata"))
    import soccerdata as sd  # imported late so SOCCERDATA_DIR is honoured

    reader = sd.Understat(leagues=list(UNDERSTAT_LEAGUES), seasons=_seasons(current_only))
    sched = reader.read_schedule().reset_index()
    if sched.empty:
        raise DownloadError("Understat returned no matches")
    out = pd.DataFrame({
        "league": sched["league"].map(UNDERSTAT_LEAGUES),
        "season": 2000 + sched["season"].astype(str).str[:2].astype(int),
        "understat_id": sched["game_id"].astype(int),
        "kickoff": pd.to_datetime(sched["date"]),
        "home_team_us": sched["home_team"],
        "away_team_us": sched["away_team"],
        "home_goals_us": sched["home_goals"],
        "away_goals_us": sched["away_goals"],
        "home_xg": sched["home_xg"],
        "away_xg": sched["away_xg"],
        "is_result": sched["is_result"].astype(bool),
    })
    missing = sorted(set(UNDERSTAT_LEAGUES.values()) - set(out.league))
    if missing:
        raise DownloadError(f"No Understat data for leagues: {missing}")
    return out


def update_file(current_only: bool = False) -> Path:
    new = fetch(current_only)
    if current_only and XG_FILE.exists():
        old = pd.read_parquet(XG_FILE)
        old = old[~old.season.isin(new.season.unique())]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["kickoff", "league", "understat_id"]).reset_index(drop=True)
    XG_DIR.mkdir(parents=True, exist_ok=True)
    new.to_parquet(XG_FILE, index=False)
    print(new.groupby(["season", "league"]).size().unstack())
    return XG_FILE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", action="store_true", help="only refresh the current season")
    update_file(ap.parse_args().current)
