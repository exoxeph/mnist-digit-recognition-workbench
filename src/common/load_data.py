import subprocess
import zipfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.common import config

KAGGLE_COMPETITION = "digit-recognizer"

load_dotenv(config.PROJECT_ROOT / ".env")


def download_dataset():
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", KAGGLE_COMPETITION,
            "-p", str(config.DATA_RAW_DIR),
        ],
        check=True,
    )
    zip_path = config.DATA_RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_RAW_DIR)


def download_if_missing():
    if config.TRAIN_CSV_PATH.exists():
        return
    try:
        download_dataset()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FileNotFoundError(
            f"{config.TRAIN_CSV_PATH} not found and automatic Kaggle download "
            f"failed ({exc}). See data/README.md for manual download instructions."
        ) from exc
    if not config.TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_CSV_PATH} still not found after download. "
            "See data/README.md for manual download instructions."
        )


def dataframe_to_arrays(df):
    labels = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    images = pixels.reshape(-1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    return images, labels


def load_split():
    download_if_missing()
    df = pd.read_csv(config.TRAIN_CSV_PATH)
    images, labels = dataframe_to_arrays(df)

    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels,
        train_size=config.TRAIN_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        train_size=val_ratio,
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
