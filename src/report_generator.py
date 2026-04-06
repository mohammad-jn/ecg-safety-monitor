import json
from pathlib import Path


class ReportGenerator:
    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def save_report(self, data: dict, filename: str) -> Path:
        file_path = self.output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return file_path