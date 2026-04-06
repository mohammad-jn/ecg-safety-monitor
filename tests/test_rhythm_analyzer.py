import numpy as np
import pytest

from src.rhythm_analyzer import RhythmAnalyzer


def test_rr_intervals():
    analyzer = RhythmAnalyzer(360)

    peaks = np.array([0, 360, 720])
    rr = analyzer.compute_rr_intervals(peaks)

    assert np.allclose(rr, [1.0, 1.0])


def test_hrv_zero_for_regular():
    analyzer = RhythmAnalyzer(360)

    rr = np.array([1.0, 1.0, 1.0])
    hrv = analyzer.compute_hrv(rr)

    assert np.isclose(hrv, 0.0)


def test_irregular_detection():
    analyzer = RhythmAnalyzer(360)

    rr = np.array([0.8, 1.2, 0.7, 1.3])
    assert analyzer.detect_irregular_rhythm(rr) is True


def test_empty_rr():
    analyzer = RhythmAnalyzer(360)

    with pytest.raises(ValueError):
        analyzer.compute_hrv(np.array([]))