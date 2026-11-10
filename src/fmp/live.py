"""Live forecasting: predictions are published BEFORE kick-off and committed to git,
so the commit timestamp proves when each forecast was made.

Three steps, each a small script in the repository root:

* update.py   – download the current season's results and rebuild the match table.
* predict.py  – forecast the upcoming fixtures with the frozen live model and save
                predictions/YYYY-MM-DD.csv.
* evaluate.py – match past predictions with real results and score them against
                the bookmakers' closing odds.

Any download problem raises `LiveDataError`: the scripts exit with an error, the
GitHub Action fails (and emails the owner), and nothing is published.
"""
import io
import json
import urllib.request
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import metrics as M
from .build_dataset import load_elo_params
from .clean import _read_one, finalize
from .config import DATA_PROCESSED, DATA_RAW, LEAGUES, MODELS_DIR, ROOT, SEASONS
from .elo import run_elo
from .features import add_team_features, assemble_match_features, to_team_long
from .models import encode, implied_probabilities

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
CURRENT_SEASON = SEASONS[-1]                      # "2627"
CURRENT_SEASON_START = 2000 + int(CURRENT_SEASON[:2])
PREDICTIONS_DIR = ROOT / "predictions"
LIVE_RESULTS_DIR = ROOT / "results" / "live"
LIVE_MODEL = MODELS_DIR / "live_model.joblib"
MATCHES_FILE = DATA_PROCESSED / "matches.parquet"
REQUIRED_COLS = {"Div", "Date", "HomeTeam", "AwayTeam"}


class LiveDataError(RuntimeError):
    pass


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:  # network error, 404, timeout...
        raise LiveDataError(f"Download failed: {url}: {e}") from e
    if len(data) < 200:
        raise LiveDataError(f"Download suspiciously small ({len(data)} bytes): {url}")
    return data


def _read_csv(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
    df.columns = df.columns.str.replace("^(\ufeff|ï»¿)", "", regex=True)  # strip a UTF-8 BOM
    return df


# --------------------------------------------------------------------------- update
def update() -> pd.DataFrame:
    """Refresh the current season for the five leagues and rebuild matches.parquet."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for code in LEAGUES:
        raw = _download(SEASON_URL.format(season=CURRENT_SEASON, code=code))
        df = _read_csv(raw)
        if not REQUIRED_COLS.issubset(df.columns) or df["FTR"].notna().sum() == 0:
            raise LiveDataError(f"{code} {CURRENT_SEASON}: unexpected file content")
        (DATA_RAW / f"{code}_{CURRENT_SEASON}.csv").write_bytes(raw)

    current = pd.concat([_read_one(code, CURRENT_SEASON) for code in LEAGUES], ignore_index=True)
    old = pd.read_parquet(MATCHES_FILE)
    history = old[old.season < CURRENT_SEASON_START].drop(columns="match_id")
    n_before = (old.season == CURRENT_SEASON_START).sum()
    matches = finalize(pd.concat([history, current], ignore_index=True))
    n_after = (matches.season == CURRENT_SEASON_START).sum()
    if n_after < n_before:
        raise LiveDataError(f"Current season shrank from {n_before} to {n_after} matches")
    if not matches.equals(old):  # avoid an empty "binary change" commit when nothing new was played
        matches.to_parquet(MATCHES_FILE, index=False)
    print(f"Current season: {n_before} -> {n_after} played matches. Last match: {matches.date.max():%Y-%m-%d}")
    return matches


# --------------------------------------------------------------------------- predict
def load_fixtures(today: pd.Timestamp) -> pd.DataFrame:
    raw = _read_csv(_download(FIXTURES_URL))
    if not REQUIRED_COLS.issubset(raw.columns):
        raise LiveDataError("fixtures.csv has an unexpected format")
    fx = raw[raw["Div"].isin(LEAGUES)].copy()
    fx["date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed")
    fx = fx[fx["date"] >= today.normalize()]
    out = pd.DataFrame({
        "league": fx["Div"], "date": fx["date"], "time": fx.get("Time"),
        "home_team": fx["HomeTeam"].str.strip(), "away_team": fx["AwayTeam"].str.strip(),
    })
    for c, src in zip("hda", ["AvgH", "AvgD", "AvgA"]):
        out[f"odds_avg_open_{c}"] = pd.to_numeric(fx.get(src), errors="coerce")
    return out.sort_values(["date", "time", "league"]).reset_index(drop=True)


def _already_predicted() -> set:
    seen = set()
    for f in PREDICTIONS_DIR.glob("*.csv"):
        p = pd.read_csv(f)
        seen |= set(zip(p.league, p.home_team, p.away_team, p.date))
    return seen


def predict(today: pd.Timestamp | None = None) -> pd.DataFrame | None:
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    bundle = joblib.load(LIVE_MODEL)
    matches = pd.read_parquet(MATCHES_FILE)
    fixtures = load_fixtures(today)

    # Every team must be a known team of its league this season, or the names are off.
    current = matches[matches.season == CURRENT_SEASON_START]
    known = {lg: set(g.home_team) | set(g.away_team) for lg, g in current.groupby("league")}
    unknown = [(r.league, t) for r in fixtures.itertuples() for t in (r.home_team, r.away_team)
               if t not in known.get(r.league, set())]
    if unknown:
        raise LiveDataError(f"Unknown team names in fixtures: {sorted(set(unknown))}")

    # Only a team's NEXT match: its later fixtures depend on a result we don't know yet.
    teams_seen, keep = set(), []
    for r in fixtures.itertuples():
        keep.append(r.home_team not in teams_seen and r.away_team not in teams_seen)
        teams_seen |= {r.home_team, r.away_team}
    fixtures = fixtures[keep]
    done = _already_predicted()
    fixtures = fixtures[[(r.league, r.home_team, r.away_team, f"{r.date:%Y-%m-%d}") not in done
                         for r in fixtures.itertuples()]]
    if fixtures.empty:
        print("No new fixtures to predict.")
        return None

    fx = fixtures.assign(match_id=-np.arange(1, len(fixtures) + 1), season=CURRENT_SEASON_START,
                         result=np.nan)
    params = load_elo_params()
    _, state = run_elo(matches, params)
    elo = pd.DataFrame({"match_id": fx.match_id,
                        "elo_home": fx.home_team.map(state.ratings).to_numpy(),
                        "elo_away": fx.away_team.map(state.ratings).to_numpy()})
    elo["elo_diff"] = elo.elo_home + params.home_advantage - elo.elo_away
    teams = set(fx.home_team) | set(fx.away_team)
    hist = matches[matches.home_team.isin(teams) | matches.away_team.isin(teams)]
    long = add_team_features(to_team_long(pd.concat([hist, fx], ignore_index=True)))
    X = assemble_match_features(fx, long, elo).set_index("match_id").loc[fx.match_id]
    probs = bundle["model"].predict_proba(X[bundle["features"]].astype(float))

    out = fx[["league", "date", "time", "home_team", "away_team"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out[["p_home", "p_draw", "p_away"]] = probs.round(4)
    out[["elo_home", "elo_away"]] = X[["elo_home", "elo_away"]].round(1).to_numpy()
    out[["odds_avg_open_h", "odds_avg_open_d", "odds_avg_open_a"]] = fx[
        ["odds_avg_open_h", "odds_avg_open_d", "odds_avg_open_a"]].to_numpy()
    out["model_version"] = bundle["version"]
    out["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = PREDICTIONS_DIR / f"{today:%Y-%m-%d}.csv"
    if path.exists():  # second run on the same day: append, never rewrite earlier rows
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} predictions to {path.name}")
    return out


# --------------------------------------------------------------------------- evaluate
BENCHMARKS = {"Market average (closing)": "odds_avgc", "Betfair Exchange (closing)": "odds_bfec"}


def evaluate() -> pd.DataFrame | None:
    files = sorted(PREDICTIONS_DIR.glob("*.csv"))
    if not files:
        print("No predictions yet.")
        return None
    preds = pd.concat([pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True)
    matches = pd.read_parquet(MATCHES_FILE)
    cur = matches[matches.season == CURRENT_SEASON_START]
    # Join on teams only: a postponed match keeps its prediction, which was made before it was played.
    df = preds.merge(cur, on=["league", "home_team", "away_team"], suffixes=("", "_played"))
    df = df[df.date_played >= pd.to_datetime(df.date) - pd.Timedelta(days=3)]
    if df.empty:
        print("No predicted match has been played yet.")
        return None
    y = encode(df.result)
    rows = df[["league", "date_played", "home_team", "away_team", "result",
               "p_home", "p_draw", "p_away"]].rename(columns={"date_played": "date"}).copy()
    rows["ll_model"] = M.per_match_log_loss(df[["p_home", "p_draw", "p_away"]].to_numpy(), y)
    for name, prefix in BENCHMARKS.items():
        p = implied_probabilities(df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]])
        rows[f"ll_{prefix}"] = M.per_match_log_loss(np.nan_to_num(p, nan=1 / 3), y)
        rows.loc[np.isnan(p).any(axis=1), f"ll_{prefix}"] = np.nan
    rows = rows.sort_values("date").reset_index(drop=True)
    rows["matchday_week"] = pd.to_datetime(rows.date).dt.to_period("W-MON").dt.start_time.dt.strftime("%Y-%m-%d")

    common = rows.dropna()
    weekly = common.groupby("matchday_week").agg(
        n=("ll_model", "size"), model=("ll_model", "sum"),
        **{k: (f"ll_{v}", "sum") for k, v in {"market_closing": "odds_avgc", "betfair_closing": "odds_bfec"}.items()})
    cum = weekly.cumsum()
    for c in ["model", "market_closing", "betfair_closing"]:
        weekly[f"cum_{c}"] = cum[c] / cum["n"]
    weekly["cum_n"] = cum["n"]

    LIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(LIVE_RESULTS_DIR / "matches.csv", index=False)
    weekly.reset_index().to_csv(LIVE_RESULTS_DIR / "cumulative.csv", index=False)
    summary = {
        "matches_scored": int(len(common)),
        "log_loss": {"model": float(common.ll_model.mean()),
                     "market_average_closing": float(common.ll_odds_avgc.mean()),
                     "betfair_exchange_closing": float(common.ll_odds_bfec.mean())},
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (LIVE_RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return rows
