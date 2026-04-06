import numpy as np
import argparse

from src.config import MITBIH_DIR, SAMPLING_RATE_HZ, DEFAULT_LEAD
from src.data_loader import ECGDataLoader
from src.diagnostic_engine import DiagnosticEngine
from src.report_generator import ReportGenerator
from src.rhythm_analyzer import RhythmAnalyzer
from src.safety_controller import SafetyController
from src.signal_plotter import plot_ecg_signal
from src.preprocessor import ECGPreprocessor
from src.peak_detector import RPeakDetector
from src.signal_plotter import plot_ecg_with_peaks
from src.annotation_loader import AnnotationLoader
from src.evaluator import PeakEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Safety Monitor")

    parser.add_argument("--record", type=str, default="100", help="Record ID (e.g., 100)")
    parser.add_argument("--lead", type=str, default="MLII", help="ECG lead (MLII or V5)")
    parser.add_argument("--start", type=int, default=0, help="Start index for plotting")
    parser.add_argument("--end", type=int, default=2000, help="End index for plotting")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    loader = ECGDataLoader(MITBIH_DIR)
    preprocessor = ECGPreprocessor(SAMPLING_RATE_HZ)

    record_files = loader.list_record_files()

    if not record_files:
        raise RuntimeError("No ECG CSV files found in mitbih_database.")

    file_path = MITBIH_DIR / f"{args.record}.csv"
    print(f"Loading record: {file_path.name}")

    df = loader.load_record(file_path)

    available_leads = loader.get_available_leads(df)
    print(f"Available leads: {available_leads}")
    sample_indices = loader.get_sample_indices(df)
    raw_signal = loader.get_signal(df, lead=args.lead)

    normalized_signal = preprocessor.normalize_signal(raw_signal)
    filtered_signal = preprocessor.bandpass_filter(normalized_signal)

    print(f"Columns: {list(df.columns)}")
    print(f"Number of samples: {len(raw_signal)}")
    print(f"First sample index: {sample_indices[0]}")
    print(f"Last sample index: {sample_indices[-1]}")

    detector = RPeakDetector(SAMPLING_RATE_HZ)

    peaks, _ = detector.detect_peaks(filtered_signal)
    heart_rate = detector.compute_heart_rate(peaks)

    print(f"Detected {len(peaks)} peaks")
    print(f"Estimated heart rate: {heart_rate:.2f} BPM")

    analyzer = RhythmAnalyzer(SAMPLING_RATE_HZ)

    rr_intervals = analyzer.compute_rr_intervals(peaks)
    hrv = analyzer.compute_hrv(rr_intervals)
    is_irregular = analyzer.detect_irregular_rhythm(rr_intervals)

    print(f"Average RR interval: {np.mean(rr_intervals):.3f} sec")
    print(f"HRV (std of RR): {hrv:.4f}")
    print(f"Irregular rhythm detected: {is_irregular}")

    controller = SafetyController()

    signal_quality = controller.check_signal_quality(filtered_signal)
    heart_rate_status = controller.check_heart_rate(heart_rate)
    peak_status = controller.check_peak_count(peaks)

    overall = controller.overall_status(
        signal_quality,
        heart_rate_status,
        peak_status,
    )

    print(f"Signal quality: {signal_quality}")
    print(f"Heart rate status: {heart_rate_status}")
    print(f"Peak detection status: {peak_status}")
    print(f"Overall system status: {overall}")

    annotation_loader = AnnotationLoader(MITBIH_DIR)
    evaluator = PeakEvaluator(tolerance_samples=18)

    annotation_file = annotation_loader.get_annotation_file(file_path.stem)
    reference_peaks = annotation_loader.load_annotation_samples(annotation_file)

    metrics = evaluator.match_peaks(peaks, reference_peaks)

    print(f"Reference beats: {len(reference_peaks)}")
    print(f"True positives: {metrics['true_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    diagnostic_engine = DiagnosticEngine()
    diagnostic_result = diagnostic_engine.classify_record(
    heart_rate=heart_rate,
    hrv=hrv,
    irregular_rhythm=is_irregular,
    signal_quality=signal_quality,
    peak_status=peak_status,
    overall_status=overall,
    )

    print(f"Record interpretation: {diagnostic_result['label']}")
    print(f"Interpretation note: {diagnostic_result['explanation']}")

    report_generator = ReportGenerator()

    report_data = {
        "record": args.record,
        "heart_rate": round(heart_rate, 2),
        "hrv": round(hrv, 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "signal_quality": signal_quality,
        "heart_rate_status": heart_rate_status,
        "overall_status": overall,
        "interpretation": diagnostic_result["label"],
    }

    report_path = report_generator.save_report(
        report_data,
        filename=f"{args.record}_report.json",
    )

    print(f"Report saved to: {report_path}")

    plot_ecg_signal(
        signal=raw_signal,
        sample_indices=sample_indices,
        title=f"Raw ECG Record {file_path.stem} - {args.lead}",
        start=args.start,
        end=args.end,
    )

    plot_ecg_signal(
        signal=normalized_signal,
        sample_indices=sample_indices,
        title=f"Normalized ECG Record {file_path.stem} - {args.lead}",
        start=args.start,
        end=args.end,
    )

    plot_ecg_signal(
        signal=filtered_signal,
        sample_indices=sample_indices,
        title=f"Filtered ECG Record {file_path.stem} - {args.lead}",
        start=args.start,
        end=args.end,
    )

    plot_ecg_with_peaks(
        signal=filtered_signal,
        peaks=peaks,
        sample_indices=sample_indices,
        title=f"R-Peaks - Record {file_path.stem}",
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()