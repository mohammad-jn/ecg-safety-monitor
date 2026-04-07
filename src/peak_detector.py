import numpy as np
from scipy.signal import find_peaks


class RPeakDetector:
    def __init__(
        self,
        sampling_rate_hz: int,
        min_distance_sec: float = 0.25,
        height_factor: float = 0.5,
        prominence_factor: float = 0.6,
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.min_distance_sec = min_distance_sec
        self.height_factor = height_factor
        self.prominence_factor = prominence_factor

    def detect_peaks(self, signal: np.ndarray):
        if signal.size == 0:
            raise ValueError("Signal is empty.")

        # Convert distance from seconds → samples
        min_distance = int(self.min_distance_sec * self.sampling_rate_hz)

        signal_mean = np.mean(signal)
        signal_std = np.std(signal)

        min_height = signal_mean + self.height_factor * signal_std
        min_prominence = self.prominence_factor * signal_std

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