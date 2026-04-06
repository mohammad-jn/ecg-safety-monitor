from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import ECGDataLoader


def test_load_record_file_not_found():
    loader = ECGDataLoader(Path("missing_folder"))
    with pytest.raises(FileNotFoundError):
        loader.load_record(Path("missing.csv"))


def test_get_signal_and_sample_indices():
    df = pd.DataFrame(
        {
            "sample #": [0, 1, 2],
            "MLII": [995, 996, 997],
            "V5": [1011, 1010, 1009],
        }
    )

    loader = ECGDataLoader(Path("."))

    signal = loader.get_signal(df, lead="MLII")
    sample_indices = loader.get_sample_indices(df)

    assert len(signal) == 3
    assert signal[0] == 995.0
    assert sample_indices[2] == 2


def test_get_signal_invalid_lead():
    df = pd.DataFrame(
        {
            "sample #": [0, 1],
            "MLII": [995, 996],
            "V5": [1011, 1010],
        }
    )

    loader = ECGDataLoader(Path("."))

    with pytest.raises(ValueError):
        loader.get_signal(df, lead="XYZ")