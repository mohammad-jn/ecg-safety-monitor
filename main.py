from src.config import MITBIH_DIR, SAMPLING_RATE_HZ, DEFAULT_LEAD
from src.data_loader import ECGDataLoader
from src.signal_plotter import plot_ecg_signal
from src.preprocessor import ECGPreprocessor
from src.peak_detector import RPeakDetector
from src.signal_plotter import plot_ecg_with_peaks

def main() -> None:
    loader = ECGDataLoader(MITBIH_DIR)
    preprocessor = ECGPreprocessor(SAMPLING_RATE_HZ)

    record_files = loader.list_record_files()

    if not record_files:
        raise RuntimeError("No ECG CSV files found in mitbih_database.")

    file_path = record_files[0]
    print(f"Loading record: {file_path.name}")

    df = loader.load_record(file_path)

    sample_indices = loader.get_sample_indices(df)
    raw_signal = loader.get_signal(df, lead=DEFAULT_LEAD)

    normalized_signal = preprocessor.normalize_signal(raw_signal)
    filtered_signal = preprocessor.bandpass_filter(normalized_signal)

    print(f"Columns: {list(df.columns)}")
    print(f"Number of samples: {len(raw_signal)}")
    print(f"First sample index: {sample_indices[0]}")
    print(f"Last sample index: {sample_indices[-1]}")

    start = 0
    end = 1200

    detector = RPeakDetector(SAMPLING_RATE_HZ)

    peaks, _ = detector.detect_peaks(filtered_signal)
    heart_rate = detector.compute_heart_rate(peaks)

    print(f"Detected {len(peaks)} peaks")
    print(f"Estimated heart rate: {heart_rate:.2f} BPM")

    plot_ecg_signal(
        signal=raw_signal,
        sample_indices=sample_indices,
        title=f"Raw ECG Record {file_path.stem} - {DEFAULT_LEAD}",
        start=start,
        end=end,
    )

    plot_ecg_signal(
        signal=normalized_signal,
        sample_indices=sample_indices,
        title=f"Normalized ECG Record {file_path.stem} - {DEFAULT_LEAD}",
        start=start,
        end=end,
    )

    plot_ecg_signal(
        signal=filtered_signal,
        sample_indices=sample_indices,
        title=f"Filtered ECG Record {file_path.stem} - {DEFAULT_LEAD}",
        start=start,
        end=end,
    )

    plot_ecg_with_peaks(
        signal=filtered_signal,
        peaks=peaks,
        sample_indices=sample_indices,
        title=f"R-Peaks - Record {file_path.stem}",
        start=0,
        end=2000,
    )


if __name__ == "__main__":
    main()