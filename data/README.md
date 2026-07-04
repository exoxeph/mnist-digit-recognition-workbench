# data/

Dataset files are not committed to this repository.

## Getting the data

This project uses the Kaggle "Digit Recognizer" competition dataset.

1. Install dependencies: `poetry install` (the Kaggle CLI is already a dependency in `pyproject.toml`).
2. Create a Kaggle API token: kaggle.com -> Account -> Create New API Token
   (downloads a `kaggle.json` containing your username and key).
3. Copy `.env.example` to `.env` and fill in `KAGGLE_USERNAME` / `KAGGLE_KEY`
   from that `kaggle.json`. `.env` is git-ignored and never committed.
4. Join the competition (accept its rules) at
   https://www.kaggle.com/c/digit-recognizer
5. Run `poetry run python main.py` (or any `src/v*` module) - it automatically
   loads `.env` and downloads/extracts `train.csv` into `data/raw/` on first
   run if it's missing.

Alternatively, download `train.csv` manually from the competition's Data tab
and place it at `data/raw/train.csv`.
