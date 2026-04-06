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

        df.columns = [col.strip().strip("'").strip('"') for col in df.columns]

        if "sample #" not in df.columns:
            raise ValueError(
                f"Missing required column 'sample #' in {file_path.name}. "
                f"Available columns: {list(df.columns)}"
            )

        if len(df.columns) < 2:
            raise ValueError(
                f"No ECG lead columns found in {file_path.name}. "
                f"Available columns: {list(df.columns)}"
            )

        return df

    def get_available_leads(self, df: pd.DataFrame) -> list:
        return [col for col in df.columns if col != "sample #"]

    def get_signal(self, df: pd.DataFrame, lead: str = "MLII"):
        if lead not in df.columns:
            available_leads = self.get_available_leads(df)
            raise ValueError(
                f"Lead {lead} not found. Available leads: {available_leads}"
            )

        return df[lead].astype(float).to_numpy()

    def get_sample_indices(self, df: pd.DataFrame):
        sample_col = "sample #"
        if sample_col not in df.columns:
            raise ValueError(f"Column {sample_col} not found.")

        return df[sample_col].to_numpy()