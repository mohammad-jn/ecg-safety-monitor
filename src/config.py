from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MITBIH_DIR = BASE_DIR / "mitbih_database"

SAMPLING_RATE_HZ = 360
DEFAULT_LEAD = "MLII"