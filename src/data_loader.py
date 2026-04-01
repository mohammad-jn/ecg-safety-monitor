from pathlib import Path
import pandas as pd


class ECGDataLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def list_record_files(self) -> list:
        return sorted(self.data_dir.glob("*.csv"))

    def load_record(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path.name}")

        expected_columns = {"'sample #'", "'MLII'", "'V5'"}
        actual_columns = set(df.columns)

        if not expected_columns.issubset(actual_columns):
            raise ValueError(
                f"Unexpected columns in {file_path.name}. "
                f"Expected at least {expected_columns}, got {actual_columns}"
            )

        return df

    def get_signal(self, df: pd.DataFrame, lead: str = "'MLII'"):
        if lead not in df.columns:
            raise ValueError(
                f"Lead {lead} not found. Available columns: {list(df.columns)}"
            )

        return df[lead].astype(float).to_numpy()

    def get_sample_indices(self, df: pd.DataFrame):
        sample_col = "'sample #'"
        if sample_col not in df.columns:
            raise ValueError(f"Column {sample_col} not found.")

        return df[sample_col].to_numpy()