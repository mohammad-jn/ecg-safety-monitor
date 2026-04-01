import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_ecg_signal(
    signal: np.ndarray,
    sample_indices: Optional[np.ndarray] = None,
    title: str = "ECG Signal",
) -> None:
    plt.figure(figsize=(12, 4))

    if sample_indices is None:
        plt.plot(signal)
        plt.xlabel("Sample Index")
    else:
        plt.plot(sample_indices, signal)
        plt.xlabel("Sample Number")

    plt.title(title)
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()