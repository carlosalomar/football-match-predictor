"""Live pipeline step: see src/fmp/live.py. Exits with an error if any download fails."""
import sys

from src.fmp.live import LiveDataError, update

if __name__ == "__main__":
    try:
        update()
    except LiveDataError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
