"""Fit every model on the training seasons, pick hyper-parameters on the validation
season, refit on train + validation, and predict the held-out test seasons.

Output: data/processed/test_predictions.parquet with one probability triple per
model (and per bookmaker) for every test match, plus the fitted models in models/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.metrics import log_loss

from .config import (DATA_PROCESSED, MODELS_DIR, TEST_END, TEST_START, TRAIN_END,
                     VALID_SEASON, WARMUP_END)
from .features import model_feature_names
from .models import PriorBaseline, encode, implied_probabilities, lightgbm, logistic

FEATURES = model_feature_names()
ELO_ONLY = ["elo_diff"]


def split(feats: pd.DataFrame):
    train = feats[feats.season.between(WARMUP_END + 1, TRAIN_END)]
    valid = feats[feats.season == VALID_SEASON]
    test = feats[feats.season.between(TEST_START, TEST_END)]
    return train, valid, test


def main():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    train, valid, test = split(feats)
    full = pd.concat([train, valid])
    y_tr, y_va, y_full = encode(train.result), encode(valid.result), encode(full.result)
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}")

    report = {}

    # 1. Logistic regression: choose the regularisation strength on validation.
    scores = {}
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]:
        m = logistic(C).fit(train[FEATURES], y_tr)
        scores[C] = log_loss(y_va, m.predict_proba(valid[FEATURES]))
    best_C = min(scores, key=scores.get)
    report["logistic_C"] = {"chosen": best_C, "valid_log_loss": scores}

    # 2. LightGBM: number of trees by early stopping on validation.
    gbm = lightgbm().fit(train[FEATURES], y_tr, eval_X=(valid[FEATURES],), eval_y=(y_va,),
                         callbacks=[early_stopping(200, verbose=False)])
    best_iter = int(gbm.best_iteration_)
    report["lightgbm"] = {"best_iteration": best_iter,
                          "valid_log_loss": float(gbm.best_score_["valid_0"]["multi_logloss"])}

    # Refit every model on train + validation with the chosen settings.
    models = {
        "Home-win prior (baseline)": (PriorBaseline().fit(None, y_full), None),
        "Elo only": (logistic(C=1.0).fit(full[ELO_ONLY], y_full), ELO_ONLY),
        "Logistic regression": (logistic(best_C).fit(full[FEATURES], y_full), FEATURES),
        "LightGBM": (lightgbm(n_estimators=best_iter).fit(full[FEATURES], y_full), FEATURES),
    }

    pred = test[["match_id", "league", "season", "date", "home_team", "away_team", "result"]].copy()
    for name, (model, cols) in models.items():
        X = test if cols is None else test[cols]
        pred[[f"{name}|{o}" for o in "HDA"]] = model.predict_proba(X)
    for name, prefix in [("Market average", "odds_avg"), ("Pinnacle (opening)", "odds_ps"),
                         ("Pinnacle (closing)", "odds_psc")]:
        odds = test[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]]
        pred[[f"{name}|{o}" for o in "HDA"]] = implied_probabilities(odds)
        report[f"overround_{prefix}"] = float((1 / odds).dropna().sum(axis=1).mean() - 1)

    MODELS_DIR.mkdir(exist_ok=True)
    pred.to_parquet(DATA_PROCESSED / "test_predictions.parquet", index=False)
    joblib.dump({"lightgbm": models["LightGBM"][0], "logistic": models["Logistic regression"][0],
                 "elo_only": models["Elo only"][0], "features": FEATURES}, MODELS_DIR / "models.joblib")
    (MODELS_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
