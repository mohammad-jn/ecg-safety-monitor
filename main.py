from src.config import MITBIH_DIR
from src.data_loader import ECGDataLoader
from src.signal_plotter import plot_ecg_signal


def main() -> None:
    loader = ECGDataLoader(MITBIH_DIR)
    record_files = loader.list_record_files()

    if not record_files:
        raise RuntimeError("No ECG CSV files found in mitbih_database.")

    file_path = record_files[0]
    print(f"Loading record: {file_path.name}")

    df = loader.load_record(file_path)

    sample_indices = loader.get_sample_indices(df)
    signal = loader.get_signal(df, lead="'MLII'")

    print(f"Columns: {list(df.columns)}")
    print(f"Number of samples: {len(signal)}")
    print(f"First sample index: {sample_indices[0]}")
    print(f"Last sample index: {sample_indices[-1]}")

    plot_ecg_signal(
        signal=signal,
        sample_indices=sample_indices,
        title=f"ECG Record {file_path.stem} - Lead MLII",
    )


if __name__ == "__main__":
    main()