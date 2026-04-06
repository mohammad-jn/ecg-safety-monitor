import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_ecg_signal(
    signal: np.ndarray,
    sample_indices: Optional[np.ndarray] = None,
    title: str = "ECG Signal",
    start: int = 0,
    end: Optional[int] = None,
) -> None:
    if end is None:
        end = len(signal)

    signal_segment = signal[start:end]

    plt.figure(figsize=(12, 4))

    if sample_indices is None:
        plt.plot(signal_segment)
        plt.xlabel("Sample Index")
    else:
        sample_segment = sample_indices[start:end]
        plt.plot(sample_segment, signal_segment)
        plt.xlabel("Sample Number")

    plt.title(title)
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_ecg_with_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    sample_indices: Optional[np.ndarray] = None,
    title: str = "ECG with R-peaks",
    start: int = 0,
    end: Optional[int] = None,
) -> None:
    if end is None:
        end = len(signal)

    signal_segment = signal[start:end]

    plt.figure(figsize=(12, 4))

    if sample_indices is None:
        x = np.arange(start, end)
    else:
        x = sample_indices[start:end]

    plt.plot(x, signal_segment, label="ECG")

    # Only plot peaks in this window
    mask = (peaks >= start) & (peaks < end)
    peak_x = peaks[mask]
    peak_y = signal[peak_x]

    if sample_indices is not None:
        peak_x = sample_indices[peak_x]

    plt.scatter(peak_x, peak_y, color="red", label="R-peaks")

    plt.title(title)
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()