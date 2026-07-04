from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import config, load_data, preprocess
from src.v2_cnn import load_trained_model


def find_misclassified(model, x_test, y_test):
    x_cnn = preprocess.to_cnn_input(x_test)
    with torch.no_grad():
        preds = model(torch.from_numpy(x_cnn)).argmax(dim=1).numpy()
    mismatch_indices = np.where(preds != y_test)[0]
    return mismatch_indices, preds


def save_misclassified_grid(x_test, y_test, preds, indices, path, max_examples=25):
    n = min(max_examples, len(indices))
    cols = 5
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for idx, ax in zip(indices[:n], axes.flat):
        ax.imshow(x_test[idx], cmap="gray")
        ax.set_title(f"true={y_test[idx]} pred={preds[idx]}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def top_confusions(y_test, preds, top_n=3):
    counter = Counter()
    for true, pred in zip(y_test, preds):
        if true != pred:
            counter[(int(true), int(pred))] += 1
    return counter.most_common(top_n)


def write_error_analysis_report(y_test, preds, path):
    confusions = top_confusions(y_test, preds)
    accuracy = float((y_test == preds).mean())
    lines = [
        "# Error Analysis",
        "",
        f"Overall test accuracy: {accuracy:.4f}",
        "",
        "## Most common confusions",
        "",
    ]
    for (true_digit, pred_digit), count in confusions:
        lines.append(f"- True `{true_digit}` predicted as `{pred_digit}`: {count} times")
    lines += [
        "",
        "## Why this matters",
        "",
        "Confusions between visually similar digit shapes indicate the model "
        "relies on stroke geometry that overlaps between certain digit pairs "
        "(e.g. closed loops, similar curvature, or shared vertical strokes).",
        "",
        "## Possible future improvements",
        "",
        "- Data augmentation (rotation/elastic distortion) targeted at the most-confused pairs",
        "- A deeper CNN or additional regularization (dropout, batch norm)",
        "- Ensembling the V1 baseline and V2 CNN predictions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run():
    model = load_trained_model()
    (_, _), (_, _), (x_test, y_test) = load_data.load_split()
    indices, preds = find_misclassified(model, x_test, y_test)
    save_misclassified_grid(x_test, y_test, preds, indices, config.MISCLASSIFIED_EXAMPLES_PATH)
    write_error_analysis_report(y_test, preds, config.ERROR_ANALYSIS_PATH)
    print(f"Found {len(indices)} misclassified examples out of {len(y_test)}")


if __name__ == "__main__":
    run()
