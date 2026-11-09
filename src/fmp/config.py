"""Project-wide constants: paths, leagues, seasons and the temporal split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Season codes as used by football-data.co.uk ("0506" = 2005/06).
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2005, 2027)]

# Temporal split, by season start year.
#   2005/06-2007/08 : Elo / rolling-form warm-up only (never used to fit models)
#   2008/09-2020/21 : training
#   2021/22         : validation (early stopping, hyper-parameters, Elo tuning check)
#   2022/23-2025/26 : test (held out until the very end)
#   2026/27         : in progress, only used to bring team ratings up to date for the demo
WARMUP_END = 2007
TRAIN_END = 2020
VALID_SEASON = 2021
TEST_START, TEST_END = 2022, 2025

OUTCOMES = ["H", "D", "A"]
