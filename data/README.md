# data/

Dataset files are not committed to this repository.

## Getting the data

This project uses the Kaggle "Digit Recognizer" competition dataset.

1. Install dependencies: `poetry install` (the Kaggle CLI is already a dependency in `pyproject.toml`).
2. Create a Kaggle API token: kaggle.com -> Account -> Create New API Token.
   Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
   (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
3. Join the competition (accept its rules) at
   https://www.kaggle.com/c/digit-recognizer
4. Run `poetry run python main.py` (or any `src/v*` module) - it automatically
   downloads and extracts `train.csv` into `data/raw/` on first run if it's missing.

Alternatively, download `train.csv` manually from the competition's Data tab
and place it at `data/raw/train.csv`.
