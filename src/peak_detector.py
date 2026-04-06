import numpy as np
from scipy.signal import find_peaks


class RPeakDetector:
    def __init__(self, sampling_rate_hz: int) -> None:
        self.sampling_rate_hz = sampling_rate_hz

    def detect_peaks(self, signal: np.ndarray):
        if signal.size == 0:
            raise ValueError("Signal is empty.")

        min_distance = int(0.25 * self.sampling_rate_hz)

        signal_mean = np.mean(signal)
        signal_std = np.std(signal)

        min_height = signal_mean + 0.5 * signal_std
        min_prominence = 0.6 * signal_std

        peaks, properties = find_peaks(
            signal,
            distance=min_distance,
            height=min_height,
            prominence=min_prominence,
        )

        return peaks, properties

    def compute_heart_rate(self, peaks: np.ndarray) -> float:
        if len(peaks) < 2:
            raise ValueError("Not enough peaks to compute heart rate.")

        rr_intervals = np.diff(peaks) / self.sampling_rate_hz
        mean_rr = np.mean(rr_intervals)

        if mean_rr <= 0:
            raise ValueError("Invalid RR interval encountered.")

        return 60.0 / mean_rr