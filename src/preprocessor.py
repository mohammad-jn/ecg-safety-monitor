import numpy as np
from scipy.signal import butter, filtfilt


class ECGPreprocessor:
    def __init__(self, sampling_rate_hz: int) -> None:
        self.sampling_rate_hz = sampling_rate_hz

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            raise ValueError("Signal is empty.")

        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:
            raise ValueError("Signal standard deviation is zero; cannot normalize.")

        return (signal - mean) / std

    def bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 40.0,
        order: int = 4,
    ) -> np.ndarray:
        if signal.size == 0:
            raise ValueError("Signal is empty.")

        nyquist = 0.5 * self.sampling_rate_hz
        low = lowcut_hz / nyquist
        high = highcut_hz / nyquist

        if not 0 < low < high < 1:
            raise ValueError("Invalid bandpass cutoff frequencies.")

        b, a = butter(order, [low, high], btype="band")
        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal