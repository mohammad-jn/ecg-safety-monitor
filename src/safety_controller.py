import numpy as np


class SafetyController:
    def __init__(self) -> None:
        pass

    def check_signal_quality(self, signal: np.ndarray) -> str:
        if signal.size == 0:
            return "INVALID"

        std = np.std(signal)

        if std < 0.1:
            return "POOR"

        return "GOOD"

    def check_heart_rate(self, heart_rate: float) -> str:
        if heart_rate < 40:
            return "CRITICAL_LOW"
        elif heart_rate < 60:
            return "LOW"
        elif heart_rate <= 100:
            return "NORMAL"
        elif heart_rate <= 140:
            return "HIGH"
        else:
            return "CRITICAL_HIGH"

    def check_peak_count(self, peaks: np.ndarray) -> str:
        if len(peaks) < 10:
            return "TOO_FEW"
        elif len(peaks) > 10000:
            return "TOO_MANY"
        return "OK"

    def overall_status(
        self,
        signal_quality: str,
        heart_rate_status: str,
        peak_status: str,
    ) -> str:

        if signal_quality == "INVALID":
            return "ERROR"

        if "CRITICAL" in heart_rate_status:
            return "CRITICAL"

        if signal_quality == "POOR":
            return "WARNING"

        if peak_status != "OK":
            return "WARNING"

        return "NORMAL"