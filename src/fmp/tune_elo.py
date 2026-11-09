"""Grid-search the Elo parameters on the TRAINING seasons only.

Criterion: mean squared error between the Elo expected score of the home team and
the actual score (1 / 0.5 / 0), over 2008/09-2020/21. The validation and test
seasons are never looked at here.
"""
import itertools
import json

import pandas as pd

from .config import DATA_PROCESSED, MODELS_DIR, TRAIN_END, WARMUP_END
from .elo import EloParams, run_elo


def score(matches: pd.DataFrame, params: EloParams) -> float:
    elo, _ = run_elo(matches, params)
    actual = matches["result"].map({"H": 1.0, "D": 0.5, "A": 0.0})
    mask = matches["season"].between(WARMUP_END + 1, TRAIN_END)
    return float(((elo["elo_exp_home"] - actual)[mask] ** 2).mean())


def main():
    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    history = matches[matches["season"] <= TRAIN_END]
    grid = itertools.product([8, 10, 12, 15, 20, 25], [55, 65, 75], [0.0, 0.1, 0.2, 0.33], [False, True])
    rows = []
    for k, hfa, rev, mov in grid:
        p = EloParams(k=k, home_advantage=hfa, season_reversion=rev, margin_of_victory=mov)
        rows.append({"k": k, "home_advantage": hfa, "season_reversion": rev,
                     "margin_of_victory": mov, "mse": score(history, p)})
    res = pd.DataFrame(rows).sort_values("mse")
    print(res.head(10).to_string(index=False))
    best = res.iloc[0].drop("mse").to_dict()
    best["margin_of_victory"] = bool(best["margin_of_victory"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "elo_params.json").write_text(json.dumps(best, indent=2))
    res.to_csv(DATA_PROCESSED / "elo_grid.csv", index=False)


if __name__ == "__main__":
    main()
