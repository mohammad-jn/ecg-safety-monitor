import numpy as np
import pytest

from src.peak_detector import RPeakDetector


def test_detect_peaks_basic():
    detector = RPeakDetector(360)

    # Simple synthetic signal with peaks
    signal = np.zeros(1000)
    signal[100] = 1
    signal[300] = 1
    signal[500] = 1

    peaks, _ = detector.detect_peaks(signal)

    assert len(peaks) >= 3


def test_compute_heart_rate():
    detector = RPeakDetector(360)

    peaks = np.array([0, 360, 720])  # 1 second apart

    hr = detector.compute_heart_rate(peaks)

    assert np.isclose(hr, 60.0)


def test_compute_heart_rate_not_enough_peaks():
    detector = RPeakDetector(360)

    with pytest.raises(ValueError):
        detector.compute_heart_rate(np.array([100]))