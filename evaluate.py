"""Live pipeline step: see src/fmp/live.py. Exits with an error if any download fails."""
import sys

from src.fmp.live import LiveDataError, evaluate

if __name__ == "__main__":
    try:
        evaluate()
    except LiveDataError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
