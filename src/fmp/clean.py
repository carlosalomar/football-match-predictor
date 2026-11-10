"""Unify the raw CSV files into one tidy match table.

One row per match with: league, season, date, teams, result, match statistics and
three sets of 1X2 odds (Pinnacle opening, Pinnacle closing, market average).
"""
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, LEAGUES, SEASONS

KEEP_STATS = {
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_target", "AST": "away_shots_target",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
}

# Each odds set lists candidate column names in order of preference, because
# football-data.co.uk renamed several columns over the years (e.g. BbAvH -> AvgH).
ODDS_SETS = {
    "ps": [("PSH", "PSD", "PSA")],                       # Pinnacle, published ~days before
    "psc": [("PSCH", "PSCD", "PSCA")],                   # Pinnacle closing (kick-off)
    "avg": [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")],  # market average
    "avgc": [("AvgCH", "AvgCD", "AvgCA")],               # market average, closing
    "bfec": [("BFECH", "BFECD", "BFECA")],               # Betfair Exchange, closing
}

# Team names are consistent within football-data.co.uk, except for a few spellings.
TEAM_ALIASES = {
    "Ath Bilbao": "Ath Bilbao",
}


def _read_one(code: str, season: str) -> pd.DataFrame | None:
    path = DATA_RAW / f"{code}_{season}.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])  # trailing blank lines / unplayed
    out = pd.DataFrame({
        "league": code,
        "league_name": LEAGUES[code],
        "season": int("20" + season[:2]),  # season start year
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home_team": raw["HomeTeam"].str.strip().replace(TEAM_ALIASES),
        "away_team": raw["AwayTeam"].str.strip().replace(TEAM_ALIASES),
    })
    for src, dst in KEEP_STATS.items():
        out[dst] = raw[src] if src in raw else np.nan
    for name, candidates in ODDS_SETS.items():
        for cols in candidates:
            if all(c in raw for c in cols):
                for c, suffix in zip(cols, "hda"):
                    out[f"odds_{name}_{suffix}"] = pd.to_numeric(raw[c], errors="coerce")
                break
        else:
            for suffix in "hda":
                out[f"odds_{name}_{suffix}"] = np.nan
    return out


def build_matches(seasons: list[str] = SEASONS) -> pd.DataFrame:
    frames = [f for code in LEAGUES for s in seasons if (f := _read_one(code, s)) is not None]
    df = pd.concat(frames, ignore_index=True)
    return finalize(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes, sanity checks and a stable match_id ordered by date."""
    df = df[df["result"].isin(["H", "D", "A"])].copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    # Odds below 1.01 are data-entry errors; treat as missing.
    odds_cols = [c for c in df if c.startswith("odds_")]
    df[odds_cols] = df[odds_cols].where(df[odds_cols] > 1.01)
    df = df.sort_values(["date", "league", "home_team"], kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", np.arange(len(df)))
    dupes = df.duplicated(["date", "home_team", "away_team"]).sum()
    assert dupes == 0, f"{dupes} duplicated matches"
    return df


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    m = build_matches()
    m.to_parquet(DATA_PROCESSED / "matches.parquet", index=False)
    print(m.shape)
    print(m.groupby("season").agg(n=("match_id", "size"),
                                  pinnacle=("odds_ps_h", lambda s: s.notna().mean()),
                                  closing=("odds_psc_h", lambda s: s.notna().mean()),
                                  sot=("home_shots_target", lambda s: s.notna().mean())).round(3))
