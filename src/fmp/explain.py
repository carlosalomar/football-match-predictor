"""Interpretability and context figures: SHAP, home advantage per league, Elo history."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .build_dataset import load_elo_params
from .config import DATA_PROCESSED, FIGURES_DIR, LEAGUES, MODELS_DIR, REPORTS_DIR, TEST_END, TEST_START
from .elo import run_elo
from .plotting import COLORS, INK, MUTED, SECONDARY, style

PRETTY = {
    "elo_diff": "Elo difference (incl. home advantage)", "elo_home": "Elo home team",
    "elo_away": "Elo away team", "diff_season_ppg": "Season points/game difference",
    "diff_form_points": "Points last 5 difference", "diff_form_sot_f": "Shots on target last 5 difference",
    "diff_form_sot_a": "Shots on target conceded last 5 difference",
    "diff_form_gf": "Goals scored last 5 difference", "diff_form_ga": "Goals conceded last 5 difference",
    "h_venue_points": "Home team: points last 5 at home", "a_venue_points": "Away team: points last 5 away",
    "rest_diff": "Rest days difference",
}


def pretty(name):
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("league_"):
        return f"League = {LEAGUES[name[7:]]}"
    side = {"h": "Home team", "a": "Away team"}[name[0]]
    stat = {"form_points": "points last 5", "form_gf": "goals scored last 5", "form_ga": "goals conceded last 5",
            "form_shots_f": "shots last 5", "form_shots_a": "shots conceded last 5",
            "form_sot_f": "shots on target last 5", "form_sot_a": "shots on target conceded last 5",
            "venue_points": "points last 5 at this venue", "venue_gf": "goals scored last 5 at this venue",
            "venue_ga": "goals conceded last 5 at this venue", "season_ppg": "season points/game",
            "season_played": "matches played this season", "rest_days": "days of rest"}[name[2:]]
    return f"{side}: {stat}"


def shap_figures(feats, bundle):
    test = feats[feats.season.between(TEST_START, TEST_END)]
    sample = test.sample(3000, random_state=0)
    X = sample[bundle["features"]]
    explainer = shap.TreeExplainer(bundle["lightgbm"])
    sv = explainer.shap_values(X)  # (n, features, classes) in recent SHAP versions
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == 3:
        sv = np.transpose(sv, (1, 2, 0))
    # Mean |SHAP| per feature and outcome (log-odds scale of the multiclass model).
    imp = pd.DataFrame({o: np.abs(sv[:, :, k]).mean(0) for k, o in enumerate(["Home win", "Draw", "Away win"])},
                       index=X.columns)
    imp["total"] = imp.sum(axis=1)
    imp = imp.sort_values("total", ascending=False)
    top = imp.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    left = np.zeros(len(top))
    for k, o in enumerate(["Home win", "Draw", "Away win"]):
        ax.barh([pretty(i) for i in top.index], top[o], left=left, color=COLORS[k], label=o,
                edgecolor="white", linewidth=1.5, height=0.7)
        left += top[o].to_numpy()
    ax.set_xlabel("Mean |SHAP value| (log-odds), test seasons")
    ax.set_title("What drives the LightGBM forecast", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    # Beeswarm for the home-win output.
    plt.figure()
    shap.summary_plot(sv[:, :, 0], X.rename(columns=pretty), max_display=12, show=False, plot_size=(9, 5.5))
    plt.title("SHAP values for P(home win)", loc="left", color=INK, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm_home_win.png", dpi=150)
    plt.close()

    share = (imp["total"] / imp["total"].sum())
    groups = {
        "Elo ratings": share[[c for c in share.index if c.startswith("elo")]].sum(),
        "Recent form & season record": share[[c for c in share.index if "form" in c or "ppg" in c or "venue" in c or "played" in c]].sum(),
        "Rest days": share[[c for c in share.index if "rest" in c]].sum(),
        "League": share[[c for c in share.index if c.startswith("league")]].sum(),
    }
    return {"top_features": imp.head(12)["total"].round(4).to_dict(),
            "share_by_group": {k: round(float(v), 3) for k, v in groups.items()}}


def home_advantage_figure(matches):
    """Home advantage = home win rate minus away win rate, per league and season."""
    m = matches[matches.season <= TEST_END]
    g = m.groupby(["league", "season"]).agg(
        home=("result", lambda r: (r == "H").mean()), away=("result", lambda r: (r == "A").mean()),
        gd=("home_goals", "mean"), ga=("away_goals", "mean")).reset_index()
    g["edge"] = g.home - g.away
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, (code, name) in enumerate(LEAGUES.items()):
        d = g[g.league == code]
        ax.plot(d.season, d.edge * 100, lw=2, color=COLORS[k], label=name)
        nudge = {"E0": 0.8, "D1": -0.8}.get(code, 0)
        ax.text(d.season.iloc[-1] + 0.3, d.edge.iloc[-1] * 100 + nudge, name, color=SECONDARY, fontsize=8, va="center")
    ax.axvspan(2019.6, 2020.9, color="#e6e5e0", alpha=0.6, lw=0)
    ax.text(2020.25, ax.get_ylim()[1] * 0.97, "Covid:\nempty\nstadiums", ha="center", va="top", fontsize=8, color=SECONDARY)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(2005, TEST_END + 1, 2), [f"{y}/{(y + 1) % 100:02d}" for y in range(2005, TEST_END + 1, 2)])
    ax.set_xlim(2004.5, TEST_END + 2.5)
    ax.set_ylabel("Home win % minus away win %")
    ax.set_title("Home advantage by league and season", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "home_advantage.png", dpi=150)
    plt.close(fig)
    summary = g[g.season <= TEST_END].groupby("league").agg(home_win=("home", "mean"), away_win=("away", "mean"),
                                                            edge=("edge", "mean"))
    covid = g[g.season == 2020].set_index("league").edge
    summary["edge_2020_21"] = covid
    summary.index = summary.index.map(LEAGUES)
    return summary.round(3).to_dict(orient="index")


def elo_history_figure(matches):
    elo, _ = run_elo(matches, load_elo_params())
    long = pd.concat([
        pd.DataFrame({"date": matches.date, "team": matches.home_team, "elo": elo.elo_home}),
        pd.DataFrame({"date": matches.date, "team": matches.away_team, "elo": elo.elo_away})])
    teams = ["Barcelona", "Real Madrid", "Bayern Munich", "Man City", "Leicester"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, t in enumerate(teams):
        d = long[long.team == t].sort_values("date")
        d = d.set_index("date").elo.rolling("60D").mean()
        # Break the line while a team is outside the top flight (e.g. Leicester 2023/24).
        gaps = d.index.to_series().diff().dt.days > 150
        d[gaps] = np.nan
        ax.plot(d.index, d.values, lw=2, color=COLORS[k], label=t)
    ax.axhline(1500, color=MUTED, lw=1, ls="--")
    ax.set_ylabel("Elo rating (pre-match)")
    ax.set_title("Elo ratings over time (60-day average; gaps = seasons outside the top flight)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elo_history.png", dpi=150)
    plt.close(fig)


def main():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    bundle = joblib.load(MODELS_DIR / "models.joblib")
    out = {"shap": shap_figures(feats, bundle), "home_advantage": home_advantage_figure(matches)}
    elo_history_figure(matches)
    (REPORTS_DIR / "explain.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
