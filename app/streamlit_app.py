"""European Football Match Predictor: interactive demo.

Pick two teams from the same league; the app computes their current Elo and
recent form from every league match played so far, and shows the predicted
probabilities of a home win, draw and away win, plus a SHAP explanation.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fmp.config import LEAGUES, REPORTS_DIR  # noqa: E402
from src.fmp.explain import pretty  # noqa: E402
from src.fmp.predict import Predictor  # noqa: E402

COLORS = {"Home win": "#2a78d6", "Draw": "#eb6834", "Away win": "#1baf7a"}

st.set_page_config(page_title="European Football Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource
def load_predictor():
    return Predictor()


P = load_predictor()
teams_by_league = P.current_teams()

st.title("European Football Match Predictor")
st.caption("Probabilistic forecasts (home win / draw / away win) for the big five European leagues, "
           f"built from public match data up to {P.last_date():%d %b %Y}. "
           "A machine-learning portfolio project, not betting advice.")

tab_predict, tab_results, tab_ratings = st.tabs(["Predict a match", "How good is it?", "Elo table"])

with tab_predict:
    c1, c2, c3, c4 = st.columns([1.2, 1.5, 1.5, 1])
    league = c1.selectbox("League", list(LEAGUES), format_func=LEAGUES.get, index=1)
    teams = teams_by_league[league]
    ratings = P.team_ratings(league)
    home = c2.selectbox("Home team", teams, index=teams.index(ratings.index[0]))
    away_options = [t for t in teams if t != home]
    away = c3.selectbox("Away team", away_options, index=away_options.index(ratings.index[1]) if ratings.index[1] in away_options else 0)
    date = c4.date_input("Match date", value=(P.last_date() + pd.Timedelta(days=7)).date(),
                         min_value=(P.last_date() + pd.Timedelta(days=1)).date())

    fc = P.forecast(home, away, pd.Timestamp(date))
    p = fc.probs["LightGBM"]

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home} win", f"{p[0]:.0%}")
    m2.metric("Draw", f"{p[1]:.0%}")
    m3.metric(f"{away} win", f"{p[2]:.0%}")

    fig = go.Figure()
    for name in ["LightGBM", "Logistic regression", "Elo only"]:
        probs = fc.probs[name]
        left = 0
        for k, (label, color) in enumerate(COLORS.items()):
            fig.add_bar(y=[name], x=[probs[k]], orientation="h", name=label, marker_color=color,
                        marker_line=dict(color="white", width=2), showlegend=(name == "LightGBM"),
                        text=f"{probs[k]:.0%}", textposition="inside", insidetextanchor="middle",
                        hovertemplate=f"{name}<br>{label}: %{{x:.1%}}<extra></extra>")
            left += probs[k]
    fig.update_layout(barmode="stack", height=230, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.25, traceorder="normal"))
    st.plotly_chart(fig, width="stretch")

    left_col, right_col = st.columns([1.1, 1])
    with left_col:
        st.subheader("Why? SHAP explanation of P(home win)")
        contrib = fc.shap_home.copy()
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        bar = go.Figure(go.Bar(
            y=[pretty(i) for i in top.index], x=top.values, orientation="h",
            marker_color=["#2a78d6" if v > 0 else "#e34948" for v in top.values],
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        bar.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Push towards (+) or away from (−) a home win, log-odds")
        st.plotly_chart(bar, width="stretch")
        st.caption("Blue features make a home win more likely, red ones less likely, relative to an average match.")
    with right_col:
        st.subheader("What the model sees")
        f = fc.features
        rows = [
            ("Elo rating", f.elo_home, f.elo_away, "{:.0f}"),
            ("Points per game, last 5", f.h_form_points, f.a_form_points, "{:.1f}"),
            ("Goals scored, last 5 (avg)", f.h_form_gf, f.a_form_gf, "{:.1f}"),
            ("Goals conceded, last 5 (avg)", f.h_form_ga, f.a_form_ga, "{:.1f}"),
            ("Shots on target, last 5 (avg)", f.h_form_sot_f, f.a_form_sot_f, "{:.1f}"),
            ("Points per game at this venue, last 5", f.h_venue_points, f.a_venue_points, "{:.1f}"),
            ("Season points per game", f.h_season_ppg, f.a_season_ppg, "{:.2f}"),
            ("Days of rest", f.h_rest_days, f.a_rest_days, "{:.0f}"),
        ]
        table = pd.DataFrame([(r[0], r[3].format(r[1]) if pd.notna(r[1]) else "–",
                               r[3].format(r[2]) if pd.notna(r[2]) else "–") for r in rows],
                             columns=["", home, away]).set_index("")
        st.table(table)
        st.caption(f"Home advantage in the Elo model: +{P.params.home_advantage:.0f} points. "
                   "Rest days only count league matches (cup and European games are not in the data).")

with tab_results:
    res = json.loads((REPORTS_DIR / "results.json").read_text())
    st.subheader("Out-of-sample results, seasons 2022/23–2025/26")
    tbl = pd.DataFrame(res["main_table"]).set_index("model")
    tbl.columns = ["Log loss", "Brier score", "RPS", "Accuracy"]
    st.dataframe(tbl.style.format({"Log loss": "{:.4f}", "Brier score": "{:.4f}", "RPS": "{:.4f}",
                                   "Accuracy": "{:.1%}"}), width="stretch")
    st.caption(f"{res['n_common_matches']:,} test matches with bookmaker odds available. Lower is better for every "
               "score except accuracy. Models were trained on 2008/09–2021/22 only.")
    for img, cap in [("gap_to_pinnacle_by_league.png", None), ("calibration.png", None),
                     ("shap_importance.png", None), ("home_advantage.png", None)]:
        st.image(str(REPORTS_DIR / "figures" / img), width="stretch")

with tab_ratings:
    lg = st.selectbox("League ", list(LEAGUES), format_func=LEAGUES.get, index=1, key="elo_league")
    r = P.team_ratings(lg).round(0).astype(int).rename("Elo").to_frame()
    r.index.name = "Team"
    r.insert(0, "Rank", range(1, len(r) + 1))
    st.dataframe(r, width="stretch", height=740)
