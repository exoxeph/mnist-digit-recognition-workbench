# Handwritten Digit Recognition Workbench

A production-inspired computer vision project built on the classic MNIST digit
recognition task. Rather than stopping at "train a model, report accuracy," this
project is staged into four versions that mirror how a real CV project matures:
a baseline, a proper model, rigorous evaluation, and a live demo.

## What is MNIST?

MNIST is a dataset of 28x28 grayscale images of handwritten digits (0-9). This
project uses the Kaggle "Digit Recognizer" competition's CSV format, where each
row is a label plus 784 flattened pixel values (0-255).

## Roadmap

- **V1 - Baseline**: logistic regression on flattened, normalized pixels.
- **V2 - CNN**: a small PyTorch convolutional network, compared directly against V1.
- **V3 - Analysis**: confusion matrix, per-class accuracy, misclassified examples,
  confidence scoring with high/medium/low bands, and robustness testing under
  rotation/shift/noise/blur.
- **V4 - Demo**: a Streamlit app where you draw a digit and get a live prediction
  with confidence and top-3 breakdown.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

Get the dataset (see `data/README.md` for full instructions): create a Kaggle API
token, copy `.env.example` to `.env` and fill in `KAGGLE_USERNAME`/`KAGGLE_KEY`,
then any training command below will auto-download `train.csv` into `data/raw/`
if it's missing.

## Training

Run the full pipeline:

```bash
poetry run python main.py
```

Or run each stage individually:

```bash
poetry run python -m src.v1_baseline
poetry run python -m src.v2_cnn
poetry run python -m src.v3_error_analysis
poetry run python -m src.v3_robustness
```

## Running the app

```bash
poetry run streamlit run app/streamlit_app.py
```

Draw a digit, click Predict, and see the predicted digit, confidence score,
top-3 predictions, and the exact 28x28 image the model sees.

## Results

| Model | Test Accuracy |
|---|---|
| V1 - Logistic Regression | 0.9126 |
| V2 - CNN | 0.9845 |

See `reports/confusion_matrix.png` and `reports/per_class_accuracy.csv` for the
full breakdown.

## Error analysis

See `reports/error_analysis.md` for the most common confusions and
`reports/misclassified_examples.png` for a visual grid of mistakes.

## Robustness

`reports/robustness_results.csv` shows CNN accuracy under rotation, shift,
additive noise, and gaussian blur, compared to the unperturbed baseline.

## Limitations

- Trained only on Kaggle's `train.csv` (~42k images); Kaggle's own `test.csv` has
  no labels and isn't used for evaluation.
- Confidence-band thresholds (0.9 / 0.6) are reasonable defaults, not
  empirically tuned.
- The Streamlit canvas input differs from MNIST's centered, anti-aliased digits,
  so live predictions can be less accurate than the reported test metrics.

## Future improvements

- Data augmentation targeted at the most-confused digit pairs.
- A deeper CNN with batch normalization/dropout.
- A "model audit assistant" that automatically summarizes `reports/` into a
  human-readable health check (not built in this version - noted here as a
  future direction only).
