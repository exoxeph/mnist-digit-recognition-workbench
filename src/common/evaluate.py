import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.common import config


def compute_accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))


def save_metrics_json(path, model_name, y_true, y_pred):
    accuracy = compute_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "num_test_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def save_confusion_matrix_plot(y_true, y_pred, path):
    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_per_class_accuracy_csv(y_true, y_pred, path):
    cm = compute_confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    df = pd.DataFrame({
        "digit": list(range(config.NUM_CLASSES)),
        "accuracy": per_class,
        "num_samples": cm.sum(axis=1),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
