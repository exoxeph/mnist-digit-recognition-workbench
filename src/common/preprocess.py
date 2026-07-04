import numpy as np

from src.common import config


def normalize(images):
    images = np.asarray(images, dtype=np.float32)
    return images / 255.0


def to_cnn_input(images):
    normalized = normalize(images)
    return normalized.reshape(-1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)


def to_flat_input(images):
    normalized = normalize(images)
    return normalized.reshape(-1, config.IMAGE_SIZE * config.IMAGE_SIZE)


def preprocess_single_image(image_28x28):
    image = np.asarray(image_28x28, dtype=np.float32)
    if image.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE):
        raise ValueError(
            f"Expected {config.IMAGE_SIZE}x{config.IMAGE_SIZE} image, got {image.shape}"
        )
    normalized = image / 255.0 if image.max() > 1.0 else image
    return normalized.reshape(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32)
