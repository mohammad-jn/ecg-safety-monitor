import numpy as np
import pytest

from src.evaluator import PeakEvaluator


def test_match_peaks_basic():
    evaluator = PeakEvaluator(tolerance_samples=5)

    detected = np.array([100, 200, 300])
    reference = np.array([102, 198, 400])

    metrics = evaluator.match_peaks(detected, reference)

    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1


def test_match_peaks_empty_detected():
    evaluator = PeakEvaluator()

    with pytest.raises(ValueError):
        evaluator.match_peaks(np.array([]), np.array([100, 200]))


def test_match_peaks_empty_reference():
    evaluator = PeakEvaluator()

    with pytest.raises(ValueError):
        evaluator.match_peaks(np.array([100, 200]), np.array([]))