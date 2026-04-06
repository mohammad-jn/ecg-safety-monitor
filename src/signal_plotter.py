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