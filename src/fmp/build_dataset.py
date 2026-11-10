"""Run Elo with the tuned parameters and build the model-ready feature table."""
import json

import pandas as pd

from .config import DATA_PROCESSED, MODELS_DIR
from .elo import EloParams, run_elo
from .features import build_feature_table
from .xg_merge import attach_xg


def load_elo_params() -> EloParams:
    return EloParams(**json.loads((MODELS_DIR / "elo_params.json").read_text()))


def main():
    matches = attach_xg(pd.read_parquet(DATA_PROCESSED / "matches.parquet"))
    elo, _ = run_elo(matches, load_elo_params())
    elo["match_id"] = matches["match_id"]
    feats = build_feature_table(matches, elo)
    feats.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    print(feats.shape)


if __name__ == "__main__":
    main()
