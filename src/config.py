"""Configuration settings for the project"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
GAMES_FILE = DATA_DIR / "games.csv"
TURNS_FILE = DATA_DIR / "turns.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering parameters
TURNS_AGGREGATIONS = {
    'points': ['mean', 'max', 'min', 'std', 'sum'],
    'score': ['mean', 'max', 'std'],
    'turn_number': ['max', 'mean']
}

