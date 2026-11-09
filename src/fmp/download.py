"""Download the raw football-data.co.uk CSV files (top division of the big five leagues)."""
import time
import urllib.request

from .config import BASE_URL, DATA_RAW, LEAGUES, SEASONS


def download_all(force: bool = False) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for code in LEAGUES:
            path = DATA_RAW / f"{code}_{season}.csv"
            if path.exists() and not force:
                continue
            url = BASE_URL.format(season=season, code=code)
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
            time.sleep(0.5)  # be polite to a free data source


if __name__ == "__main__":
    download_all()
