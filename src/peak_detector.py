import numpy as np
from scipy.signal import find_peaks


class RPeakDetector:
    def __init__(self, sampling_rate_hz: int) -> None:
        self.sampling_rate_hz = sampling_rate_hz

    def detect_peaks(self, signal: np.ndarray):
        if signal.size == 0:
            raise ValueError("Signal is empty.")

        # Minimum distance between peaks (~200 ms)
        min_distance = int(0.2 * self.sampling_rate_hz)

        # Detect peaks
        peaks, properties = find_peaks(
            signal,
            distance=min_distance,
            height=np.mean(signal),  # simple threshold
        )

        return peaks, properties

    def compute_heart_rate(self, peaks: np.ndarray) -> float:
        if len(peaks) < 2:
            raise ValueError("Not enough peaks to compute heart rate.")

        rr_intervals = np.diff(peaks) / self.sampling_rate_hz  # seconds
        mean_rr = np.mean(rr_intervals)

        heart_rate = 60.0 / mean_rr  # BPM

        return heart_rate