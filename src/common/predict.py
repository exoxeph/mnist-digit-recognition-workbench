import numpy as np
import torch

from src.common import config


def confidence_band(probability):
    if probability >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if probability >= config.CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def predict_with_confidence(model, image_28x28, preprocess_fn):
    model.eval()
    x = preprocess_fn(image_28x28)
    x_tensor = torch.from_numpy(x)
    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [(int(i), float(probabilities[i])) for i in top3_indices]
    top1_digit, top1_confidence = top3[0]

    return {
        "predicted_digit": top1_digit,
        "confidence": top1_confidence,
        "confidence_band": confidence_band(top1_confidence),
        "top3": top3,
        "probabilities": probabilities.tolist(),
    }
