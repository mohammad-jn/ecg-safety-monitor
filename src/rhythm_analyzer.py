import numpy as np


class RhythmAnalyzer:
    def __init__(self, sampling_rate_hz: int) -> None:
        self.sampling_rate_hz = sampling_rate_hz

    def compute_rr_intervals(self, peaks: np.ndarray) -> np.ndarray:
        if len(peaks) < 2:
            raise ValueError("Not enough peaks to compute RR intervals.")

        rr_intervals = np.diff(peaks) / self.sampling_rate_hz  # seconds
        return rr_intervals

    def compute_hrv(self, rr_intervals: np.ndarray) -> float:
        if len(rr_intervals) == 0:
            raise ValueError("RR intervals array is empty.")

        return np.std(rr_intervals)

    def detect_irregular_rhythm(self, rr_intervals: np.ndarray) -> bool:
        if len(rr_intervals) == 0:
            raise ValueError("RR intervals array is empty.")

        hrv = self.compute_hrv(rr_intervals)

        # Simple threshold (can tune later)
        return bool(hrv > 0.1)