import json
from pathlib import Path

import numpy as np

from src.config import MITBIH_DIR, SAMPLING_RATE_HZ
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor
from src.peak_detector import RPeakDetector
from src.annotation_loader import AnnotationLoader
from src.evaluator import PeakEvaluator


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():
    loader = ECGDataLoader(MITBIH_DIR)
    preprocessor = ECGPreprocessor(SAMPLING_RATE_HZ)
    annotation_loader = AnnotationLoader(MITBIH_DIR)
    evaluator = PeakEvaluator(tolerance_samples=18)

    record_files = loader.list_record_files()

    best_score = 0.0
    best_params = None
    iteration_results = []

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    for distance in [0.35, 0.4, 0.5]:
        for height_factor in [0.9, 1.1, 1.3]:
            for prominence_factor in [0.9, 1.1, 1.3]:

                f1_scores = []
                precision_scores = []
                recall_scores = []
                records_processed = 0
                records_skipped = []

                print(f"\nTesting d={distance}, h={height_factor}, p={prominence_factor}")

                for file_path in record_files:
                    record = file_path.stem

                    try:
                        df = loader.load_record(file_path)

                        # Skip records without MLII
                        if "MLII" not in df.columns:
                            records_skipped.append(
                                {"record": record, "reason": "MLII lead not available"}
                            )
                            continue

                        signal = loader.get_signal(df, lead="MLII")

                        normalized = preprocessor.normalize_signal(signal)
                        filtered = preprocessor.bandpass_filter(normalized)

                        annotation_file = annotation_loader.get_annotation_file(record)
                        reference_peaks = annotation_loader.load_annotation_samples(
                            annotation_file
                        )

                        detector = RPeakDetector(
                            sampling_rate_hz=SAMPLING_RATE_HZ,
                            min_distance_sec=distance,
                            height_factor=height_factor,
                            prominence_factor=prominence_factor,
                        )

                        peaks, _ = detector.detect_peaks(filtered)
                        metrics = evaluator.match_peaks(peaks, reference_peaks)

                        precision = metrics["precision"]
                        recall = metrics["recall"]
                        f1 = compute_f1(precision, recall)

                        f1_scores.append(f1)
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        records_processed += 1

                    except Exception as e:
                        print(f"Skipping {record}: {e}")
                        records_skipped.append({"record": record, "reason": str(e)})
                        continue

                if not f1_scores:
                    continue

                avg_f1 = float(np.mean(f1_scores))
                avg_precision = float(np.mean(precision_scores))
                avg_recall = float(np.mean(recall_scores))

                result = {
                    "distance": distance,
                    "height_factor": height_factor,
                    "prominence_factor": prominence_factor,
                    "avg_precision": round(avg_precision, 4),
                    "avg_recall": round(avg_recall, 4),
                    "avg_f1": round(avg_f1, 4),
                    "records_processed": records_processed,
                    "records_skipped": records_skipped,
                }

                iteration_results.append(result)

                print(
                    f"Average Precision: {avg_precision:.4f} | "
                    f"Average Recall: {avg_recall:.4f} | "
                    f"Average F1: {avg_f1:.4f}"
                )

                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = {
                        "distance": distance,
                        "height_factor": height_factor,
                        "prominence_factor": prominence_factor,
                        "avg_precision": round(avg_precision, 4),
                        "avg_recall": round(avg_recall, 4),
                        "avg_f1": round(avg_f1, 4),
                    }

    iteration_results.sort(key=lambda x: x["avg_f1"], reverse=True)

    output_data = {
        "best_params": best_params,
        "all_iterations": iteration_results,
    }

    output_path = output_dir / "tuning_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print("\n🔥 BEST GLOBAL PARAMETERS:")
    print(best_params)
    print(f"\nSaved tuning results to: {output_path}")


if __name__ == "__main__":
    main()