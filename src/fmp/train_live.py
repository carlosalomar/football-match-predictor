"""Train the frozen model used for live predictions during a season.

Run ONCE before (or early in) a season. The model is trained on every completed
season up to the previous one and is NOT retrained week to week, so the live
track record is a clean out-of-sample test.

    python -m src.fmp.train_live
"""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import DATA_PROCESSED, MODELS_DIR, WARMUP_END
from .features import model_feature_names
from .live import CURRENT_SEASON_START, LIVE_MODEL
from .models import encode, lightgbm


def main(with_xg: bool = False):
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    train = feats[feats.season.between(WARMUP_END + 1, CURRENT_SEASON_START - 1)]
    features = model_feature_names(with_xg)
    n_trees = json.loads((MODELS_DIR / "training_report.json").read_text())["lightgbm"]["best_iteration"]
    model = lightgbm(n_estimators=n_trees).fit(train[features], encode(train.result))
    version = f"lightgbm-{CURRENT_SEASON_START}-{(CURRENT_SEASON_START + 1) % 100:02d}" + ("-xg" if with_xg else "")
    joblib.dump({"model": model, "features": features, "version": version,
                 "trained_on": f"{WARMUP_END + 1}/{(WARMUP_END + 2) % 100:02d}-"
                               f"{CURRENT_SEASON_START - 1}/{CURRENT_SEASON_START % 100:02d}",
                 "n_matches": len(train),
                 "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, LIVE_MODEL)
    print(f"{version}: trained on {len(train):,} matches -> {LIVE_MODEL}")


if __name__ == "__main__":
    main()
