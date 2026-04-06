import numpy as np

from src.safety_controller import SafetyController


def test_signal_quality():
    controller = SafetyController()

    good_signal = np.random.randn(1000)
    poor_signal = np.ones(1000)

    assert controller.check_signal_quality(good_signal) == "GOOD"
    assert controller.check_signal_quality(poor_signal) == "POOR"


def test_heart_rate_status():
    controller = SafetyController()

    assert controller.check_heart_rate(50) == "LOW"
    assert controller.check_heart_rate(75) == "NORMAL"
    assert controller.check_heart_rate(120) == "HIGH"


def test_overall_status():
    controller = SafetyController()

    status = controller.overall_status("GOOD", "NORMAL", "OK")
    assert status == "NORMAL"

    status = controller.overall_status("POOR", "NORMAL", "OK")
    assert status == "WARNING"

    status = controller.overall_status("GOOD", "CRITICAL_HIGH", "OK")
    assert status == "CRITICAL"