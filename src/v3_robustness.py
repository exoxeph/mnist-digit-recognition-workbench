import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate, shift, gaussian_filter

from src.common import config, load_data, preprocess
from src.v2_cnn import load_trained_model


def apply_rotation(images, angle=15):
    return np.stack([
        rotate(img, angle, reshape=False, mode="constant", cval=0.0)
        for img in images
    ])


def apply_shift(images, shift_pixels=3):
    return np.stack([
        shift(img, shift_pixels, mode="constant", cval=0.0)
        for img in images
    ])


def apply_noise(images, sigma=25.0, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, images.shape)
    return np.clip(images + noise, 0, 255)


def apply_blur(images, sigma=1.5):
    return np.stack([
        gaussian_filter(img, sigma=sigma)
        for img in images
    ])


PERTURBATIONS = {
    "rotation": apply_rotation,
    "shift": apply_shift,
    "noise": lambda images: apply_noise(images, seed=config.RANDOM_SEED),
    "blur": apply_blur,
}


def evaluate_perturbation(model, x_test, y_test, perturb_fn):
    perturbed = perturb_fn(x_test)
    x_cnn = preprocess.to_cnn_input(perturbed)
    with torch.no_grad():
        preds = model(torch.from_numpy(x_cnn)).argmax(dim=1).numpy()
    return float((preds == y_test).mean())


def run():
    model = load_trained_model()
    (_, _), (_, _), (x_test, y_test) = load_data.load_split()

    baseline_x_cnn = preprocess.to_cnn_input(x_test)
    with torch.no_grad():
        baseline_preds = model(torch.from_numpy(baseline_x_cnn)).argmax(dim=1).numpy()
    baseline_accuracy = float((baseline_preds == y_test).mean())

    results = [{"perturbation": "none", "accuracy": baseline_accuracy}]
    for name, fn in PERTURBATIONS.items():
        accuracy = evaluate_perturbation(model, x_test, y_test, fn)
        results.append({"perturbation": name, "accuracy": accuracy})
        print(f"{name}: {accuracy:.4f}")

    df = pd.DataFrame(results)
    config.ROBUSTNESS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ROBUSTNESS_RESULTS_PATH, index=False)
    return df


if __name__ == "__main__":
    run()
