import numpy as np
import pytest

from src.common import preprocess


def test_normalize_scales_to_0_1_range():
    images = np.array([[0, 255], [128, 64]], dtype=np.float32)
    result = preprocess.normalize(images)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert result.dtype == np.float32


def test_to_cnn_input_shape():
    images = np.zeros((5, 28, 28), dtype=np.float32)
    result = preprocess.to_cnn_input(images)
    assert result.shape == (5, 1, 28, 28)
    assert result.max() <= 1.0


def test_to_flat_input_shape():
    images = np.zeros((5, 28, 28), dtype=np.float32)
    result = preprocess.to_flat_input(images)
    assert result.shape == (5, 28 * 28)


def test_preprocess_single_image_valid_shape():
    image = (np.random.rand(28, 28) * 255).astype(np.float32)
    result = preprocess.preprocess_single_image(image)
    assert result.shape == (1, 1, 28, 28)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_single_image_rejects_wrong_shape():
    image = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocess.preprocess_single_image(image)
