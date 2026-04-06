from pathlib import Path
import numpy as np


class AnnotationLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def get_annotation_file(self, record_stem: str) -> Path:
        file_path = self.data_dir / f"{record_stem}annotations.txt"

        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")

        return file_path

    def load_annotation_samples(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")

        samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # Skip header row
                if parts[0] == "Time":
                    continue

                # Expected format:
                # Time, Sample#, Type, Sub, Chan, Num, Aux(optional)
                if len(parts) < 3:
                    continue

                try:
                    sample_index = int(parts[1])
                    samples.append(sample_index)
                except ValueError:
                    continue

        if not samples:
            raise ValueError(f"No annotation samples found in {file_path.name}")

        return np.array(samples, dtype=int)