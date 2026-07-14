from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONTHLY_BATCH_DIR = DATA_DIR / "monthly_batches"
REPORTS_DIR = DATA_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "accepted_2007_to_2018Q4.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_lending_club.csv"

TARGET_COLUMN = "target"
DATE_COLUMN = "issue_d"

RANDOM_STATE = 42