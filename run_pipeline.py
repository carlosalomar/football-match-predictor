"""Run the whole project end to end: download -> clean -> Elo -> features -> train -> evaluate -> explain."""
from src.fmp import build_dataset, clean, download, evaluate, explain, train, tune_elo
from src.fmp.config import DATA_PROCESSED

if __name__ == "__main__":
    download.download_all()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean.build_matches().to_parquet(DATA_PROCESSED / "matches.parquet", index=False)
    tune_elo.main()
    build_dataset.main()
    train.main()
    evaluate.main()
    explain.main()
