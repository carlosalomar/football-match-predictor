"""Rebuild Understat match xG from the Internet Archive (Wayback Machine).

Understat has been offline since November 2026. Until late 2025 every Understat
league page embedded the full fixture list with xG as JSON
(`var datesData = JSON.parse('...')`). For each league and season we take an
archived copy of that page captured AFTER the season ended (so it contains every
match) and parse the JSON. The output has the same format as `xg_download.py`.

    python -m src.fmp.xg_wayback
"""
import codecs
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from .xg_download import XG_DIR, XG_FILE

SLUGS = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1", "Serie_A": "I1", "Ligue_1": "F1"}
SEASONS = range(2014, 2026)  # season start years
CDX = "https://web.archive.org/cdx/search/cdx"
PATTERN = re.compile(r"datesData\s*=\s*JSON\.parse\('([^']+)'\)")


def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def snapshots(slug: str, season: int) -> list[str]:
    """Timestamps of archived copies taken after the season's last matchday, oldest first."""
    q = urllib.parse.urlencode({
        "url": f"understat.com/league/{slug}/{season}", "output": "json",
        "filter": "statuscode:200", "from": f"{season + 1}0601", "collapse": "timestamp:8"})
    raw = _get(f"{CDX}?{q}")
    rows = json.loads(raw) if raw.strip() else []
    return [r[1] for r in rows[1:]]


def parse(html: str, code: str, season: int) -> pd.DataFrame | None:
    m = PATTERN.search(html)
    if not m:
        return None
    data = json.loads(codecs.decode(m.group(1), "unicode_escape"))
    return pd.DataFrame([{
        "league": code, "season": season, "understat_id": int(d["id"]),
        "kickoff": pd.to_datetime(d["datetime"]),
        "home_team_us": d["h"]["title"], "away_team_us": d["a"]["title"],
        "home_goals_us": pd.to_numeric(d["goals"]["h"]), "away_goals_us": pd.to_numeric(d["goals"]["a"]),
        "home_xg": pd.to_numeric(d["xG"]["h"]), "away_xg": pd.to_numeric(d["xG"]["a"]),
        "is_result": bool(d["isResult"]),
    } for d in data])


CACHE = XG_DIR / "wayback"


def _complete(df: pd.DataFrame | None) -> bool:
    # Snapshots are taken after the season ended, so the season is over. Accept it if at least
    # 270 matches have a result (Ligue 1 2019/20 was stopped by Covid after 279).
    return df is not None and df.is_result.sum() >= 270 and (df.is_result.all() or df.is_result.sum() < 300)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, coverage = [], []
    for slug, code in SLUGS.items():
        for season in SEASONS:
            cached = CACHE / f"{code}_{season}.parquet"
            got = pd.read_parquet(cached) if cached.exists() else None
            if got is None:
                try:
                    stamps = snapshots(slug, season)[:15]
                except Exception as e:
                    print(f"  {code} {season}: snapshot index failed: {e}", flush=True)
                    stamps = []
                for ts in stamps:
                    try:
                        html = _get(f"https://web.archive.org/web/{ts}id_/https://understat.com/league/{slug}/{season}")
                    except Exception as e:
                        print(f"  {code} {season} {ts}: {e}", flush=True)
                        continue
                    df = parse(html.decode("utf-8", "replace"), code, season)
                    time.sleep(1.5)
                    if _complete(df):
                        got = df.assign(source_snapshot=ts)
                        got.to_parquet(cached, index=False)  # resume point if the job is interrupted
                        break
            n = 0 if got is None else int(got.is_result.sum())
            coverage.append({"league": code, "season": season, "matches_with_result": n,
                             "snapshot": None if got is None else got.source_snapshot.iloc[0]})
            print(code, season, n, flush=True)
            if got is not None:
                frames.append(got)
            time.sleep(1.5)
    XG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(XG_DIR / "wayback_coverage.csv", index=False)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["kickoff", "league"]).reset_index(drop=True)
        out.to_parquet(XG_FILE, index=False)
        print(f"Saved {len(out):,} matches to {XG_FILE}")


if __name__ == "__main__":
    main()
