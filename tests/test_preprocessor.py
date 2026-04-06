import numpy as np
import pytest

from src.preprocessor import ECGPreprocessor


def test_normalize_signal():
    preprocessor = ECGPreprocessor(360)
    signal = np.array([1.0, 2.0, 3.0, 4.0])

    normalized = preprocessor.normalize_signal(signal)

    assert np.isclose(np.mean(normalized), 0.0)
    assert np.isclose(np.std(normalized), 1.0)


def test_normalize_signal_empty():
    preprocessor = ECGPreprocessor(360)

    with pytest.raises(ValueError):
        preprocessor.normalize_signal(np.array([]))


def test_bandpass_filter_length_preserved():
    preprocessor = ECGPreprocessor(360)
    signal = np.random.randn(1000)

    filtered = preprocessor.bandpass_filter(signal)

    assert len(filtered) == len(signal)


def test_bandpass_filter_empty():
    preprocessor = ECGPreprocessor(360)

    with pytest.raises(ValueError):
        preprocessor.bandpass_filter(np.array([]))