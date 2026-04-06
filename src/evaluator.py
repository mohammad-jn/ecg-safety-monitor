import numpy as np


class PeakEvaluator:
    def __init__(self, tolerance_samples: int = 18) -> None:
        self.tolerance_samples = tolerance_samples

    def match_peaks(
        self,
        detected_peaks: np.ndarray,
        reference_peaks: np.ndarray,
    ) -> dict:
        if detected_peaks.size == 0:
            raise ValueError("Detected peaks array is empty.")

        if reference_peaks.size == 0:
            raise ValueError("Reference peaks array is empty.")

        matched_detected = set()
        matched_reference = set()

        for ref_idx, ref_peak in enumerate(reference_peaks):
            for det_idx, det_peak in enumerate(detected_peaks):
                if det_idx in matched_detected:
                    continue

                if abs(det_peak - ref_peak) <= self.tolerance_samples:
                    matched_reference.add(ref_idx)
                    matched_detected.add(det_idx)
                    break

        true_positives = len(matched_reference)
        false_negatives = len(reference_peaks) - true_positives
        false_positives = len(detected_peaks) - len(matched_detected)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        return {
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
        }