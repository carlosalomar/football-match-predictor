"""Fair test of how much xG adds: the same models, the same matches, with and without xG.

xG only exists from 2014/15, so both versions are trained on 2014/15-2020/21,
tuned on 2021/22 and tested on 2022/23-2025/26 (the same test seasons as the
main results). The only difference between the two runs is the feature list.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.metrics import log_loss

from . import metrics as M
from .config import DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR, TEST_END, TEST_START, VALID_SEASON
from .features import model_feature_names
from .models import encode, implied_probabilities, lightgbm, logistic
from .plotting import COLORS, INK, style
from .xg_merge import covered_league_seasons

XG_TRAIN_START = 2014


def fit_predict(train, valid, test, features):
    y_tr, y_va = encode(train.result), encode(valid.result)
    full = pd.concat([train, valid])
    y_full = encode(full.result)
    Cs = [0.01, 0.1, 1.0]
    best_C = min(Cs, key=lambda C: log_loss(y_va, logistic(C).fit(train[features], y_tr).predict_proba(valid[features])))
    gbm = lightgbm().fit(train[features], y_tr, eval_X=(valid[features],), eval_y=(y_va,),
                         callbacks=[early_stopping(200, verbose=False)])
    return {
        "Logistic regression": logistic(best_C).fit(full[features], y_full).predict_proba(test[features]),
        "LightGBM": lightgbm(n_estimators=int(gbm.best_iteration_)).fit(full[features], y_full)
        .predict_proba(test[features]),
    }


def main():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    # Only league-seasons with xG, identical rows for both versions of every model.
    have = covered_league_seasons()
    feats = feats[[(lg, s) in have for lg, s in zip(feats.league, feats.season)]]
    train = feats[feats.season.between(XG_TRAIN_START, VALID_SEASON - 1)]
    valid = feats[feats.season == VALID_SEASON]
    test = feats[feats.season.between(TEST_START, TEST_END)].dropna(
        subset=["odds_psc_h", "odds_psc_d", "odds_psc_a"])
    y = encode(test.result)

    preds = {}
    for label, with_xg in [("without xG", False), ("with xG", True)]:
        for model, p in fit_predict(train, valid, test, model_feature_names(with_xg)).items():
            preds[f"{model}, {label}"] = p
    preds["Pinnacle (closing)"] = implied_probabilities(test[["odds_psc_h", "odds_psc_d", "odds_psc_a"]])

    table = pd.DataFrame([{"model": k, "log_loss": M.log_loss(p, y), "brier": M.brier(p, y),
                           "rps": M.rps(p, y), "accuracy": M.accuracy(p, y)} for k, p in preds.items()])
    boot = {}
    for model in ["Logistic regression", "LightGBM"]:
        d, lo, hi = M.paired_bootstrap_diff(M.per_match_log_loss(preds[f"{model}, with xG"], y),
                                            M.per_match_log_loss(preds[f"{model}, without xG"], y))
        boot[model] = {"with_minus_without": d, "ci95": [lo, hi]}
    out = {"league_seasons_used": len(have),
           "train_seasons": f"{XG_TRAIN_START}/15-{VALID_SEASON - 1}/{VALID_SEASON % 100:02d} (+{VALID_SEASON}/22 refit)",
           "n_train": len(train), "n_test": len(test),
           "table": table.round(4).to_dict(orient="records"), "bootstrap": boot}
    (REPORTS_DIR / "xg_comparison.json").write_text(json.dumps(out, indent=2))
    (REPORTS_DIR / "xg_comparison.md").write_text(table.round(4).to_markdown(index=False) + "\n")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    pinn = table.set_index("model").loc["Pinnacle (closing)", "log_loss"]
    labels, vals, colors = [], [], []
    for i, model in enumerate(["Logistic regression", "LightGBM"]):
        for j, label in enumerate(["without xG", "with xG"]):
            labels.append(f"{model}\n{label}")
            vals.append(table.set_index("model").loc[f"{model}, {label}", "log_loss"] - pinn)
            colors.append(COLORS[0] if label == "with xG" else "#9a998f")
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Log loss minus Pinnacle closing")
    ax.set_title("Does xG close the gap to the bookmaker? (same matches, same seasons)", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xg_comparison.png", dpi=150)
    plt.close(fig)
    print(table.round(4).to_string(index=False))
    print(json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
