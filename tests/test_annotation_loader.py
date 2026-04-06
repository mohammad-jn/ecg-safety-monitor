from pathlib import Path
import numpy as np
import pytest

from src.annotation_loader import AnnotationLoader


def test_get_annotation_file_not_found():
    loader = AnnotationLoader(Path("missing_dir"))

    with pytest.raises(FileNotFoundError):
        loader.get_annotation_file("100")


def test_load_annotation_samples(tmp_path: Path):
    annotation_file = tmp_path / "100annotations.txt"
    annotation_file.write_text(
        "Time   Sample #  Type  Sub Chan  Num      Aux\n"
        "0:00.050       18     +    0    0    0      (N\n"
        "0:00.214       77     N    0    0    0\n"
        "0:01.028      370     V    0    0    0\n",
        encoding="utf-8",
    )

    loader = AnnotationLoader(tmp_path)
    samples = loader.load_annotation_samples(annotation_file)

    assert np.array_equal(samples, np.array([18, 77, 370]))