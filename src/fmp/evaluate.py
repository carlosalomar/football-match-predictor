"""Score every model on the held-out test seasons and write tables + figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as M
from .config import DATA_PROCESSED, FIGURES_DIR, LEAGUES, REPORTS_DIR
from .models import encode
from .plotting import COLORS, INK, MUTED, style

MODELS = ["Home-win prior (baseline)", "Elo only", "Logistic regression", "LightGBM",
          "Market average", "Pinnacle (opening)", "Pinnacle (closing)"]


def probs(df, name):
    return df[[f"{name}|{o}" for o in "HDA"]].to_numpy()


def score_table(df):
    y = encode(df.result)
    rows = []
    for name in MODELS:
        p = probs(df, name)
        rows.append({"model": name, "log_loss": M.log_loss(p, y), "brier": M.brier(p, y),
                     "rps": M.rps(p, y), "accuracy": M.accuracy(p, y)})
    return pd.DataFrame(rows)


def calibration_plot(df, path):
    y = encode(df.result)
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for k, (ax, label) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
        for name, color in [("LightGBM", COLORS[0]), ("Logistic regression", COLORS[2]),
                            ("Pinnacle (closing)", COLORS[1])]:
            p = probs(df, name)[:, k]
            hit = (y == k).astype(float)
            b = np.clip(np.digitize(p, bins) - 1, 0, 9)
            g = pd.DataFrame({"b": b, "p": p, "hit": hit}).groupby("b").agg(
                p=("p", "mean"), hit=("hit", "mean"), n=("hit", "size"))
            g = g[g.n >= 30]
            ax.plot(g.p, g.hit, marker="o", ms=5, lw=2, color=color, label=name)
        ax.set_title(label, color=INK, loc="left")
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 0.9 if k != 1 else 0.45)
        ax.set_ylim(0, 0.9)
        style(ax)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Calibration on the test seasons (2022/23-2025/26): when the model says 60%, does it happen 60% of the time?",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gap_by_league_plot(per_league, path):
    """Log-loss gap to Pinnacle closing, per league."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    leagues = list(LEAGUES.values())
    x = np.arange(len(leagues))
    w = 0.38
    for i, (name, color) in enumerate([("Logistic regression", COLORS[2]), ("LightGBM", COLORS[0])]):
        gap = [per_league.loc[(lg, name), "log_loss"] - per_league.loc[(lg, "Pinnacle (closing)"), "log_loss"]
               for lg in leagues]
        bars = ax.bar(x + (i - 0.5) * w, gap, width=w - 0.02, color=color, label=name)
        for b, v in zip(bars, gap):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x, leagues)
    ax.set_ylabel("Log loss minus Pinnacle closing")
    ax.set_title("How far behind the sharpest bookmaker, by league (lower is closer)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(DATA_PROCESSED / "test_predictions.parquet")
    # Compare everyone on the same matches: those with all three bookmaker prices.
    odds_cols = [c for c in pred if c.startswith(("Pinnacle", "Market"))]
    common = pred.dropna(subset=odds_cols)

    main_table = score_table(common)
    full_table = score_table(pred.dropna(subset=[c for c in odds_cols if c.startswith("Market")]))
    full_table = full_table[~full_table.model.str.startswith("Pinnacle")]

    per_league = (common.groupby("league").apply(score_table, include_groups=False)
                  .reset_index(level=1, drop=True).reset_index())
    per_league["league"] = per_league["league"].map(LEAGUES)
    per_league = per_league.set_index(["league", "model"])
    per_season = (common.groupby("season").apply(score_table, include_groups=False)
                  .reset_index(level=1, drop=True).reset_index())

    y = encode(common.result)
    loss = {m: M.per_match_log_loss(probs(common, m), y) for m in MODELS}
    boot = {}
    for model in ["LightGBM", "Logistic regression", "Elo only"]:
        for ref in ["Pinnacle (closing)", "Pinnacle (opening)", "Market average"]:
            d, lo, hi = M.paired_bootstrap_diff(loss[model], loss[ref])
            boot[f"{model} - {ref}"] = {"mean": d, "ci95": [lo, hi]}
    d, lo, hi = M.paired_bootstrap_diff(loss["LightGBM"], loss["Logistic regression"])
    boot["LightGBM - Logistic regression"] = {"mean": d, "ci95": [lo, hi]}

    outcome_freq = common.result.value_counts(normalize=True).reindex(list("HDA")).to_dict()
    draws_predicted = {m: float((probs(common, m).argmax(1) == 1).mean()) for m in MODELS}

    calibration_plot(common, FIGURES_DIR / "calibration.png")
    gap_by_league_plot(per_league, FIGURES_DIR / "gap_to_pinnacle_by_league.png")

    out = {
        "n_test_matches": len(pred), "n_common_matches": len(common),
        "main_table": main_table.round(4).to_dict(orient="records"),
        "all_test_matches_table": full_table.round(4).to_dict(orient="records"),
        "bootstrap_log_loss_differences": boot,
        "outcome_frequency": outcome_freq,
        "share_of_matches_where_draw_is_the_top_pick": draws_predicted,
    }
    (REPORTS_DIR / "results.json").write_text(json.dumps(out, indent=2))
    with open(REPORTS_DIR / "results.md", "w") as f:
        f.write(f"## Test seasons 2022/23-2025/26, {len(common):,} matches with bookmaker odds\n\n")
        f.write(main_table.round(4).to_markdown(index=False))
        f.write(f"\n\n## All {len(pred):,} test matches (Pinnacle stopped publishing mid-2025/26)\n\n")
        f.write(full_table.round(4).to_markdown(index=False))
        f.write("\n\n## By league\n\n")
        f.write(per_league.round(4).reset_index().to_markdown(index=False))
        f.write("\n\n## By season\n\n")
        f.write(per_season.round(4).to_markdown(index=False))
        f.write("\n\n## Paired bootstrap of log-loss differences (2,000 resamples)\n\n")
        for k, v in boot.items():
            f.write(f"- {k}: {v['mean']:+.4f} (95% CI {v['ci95'][0]:+.4f} to {v['ci95'][1]:+.4f})\n")
    print(main_table.round(4).to_string(index=False))
    print(full_table.round(4).to_string(index=False))
    print(json.dumps(boot, indent=1))
    print(draws_predicted, outcome_freq)


if __name__ == "__main__":
    main()
