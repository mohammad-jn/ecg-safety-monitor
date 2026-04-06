from src.diagnostic_engine import DiagnosticEngine


def test_normal_rhythm_like():
    engine = DiagnosticEngine()

    result = engine.classify_record(
        heart_rate=75.0,
        hrv=0.04,
        irregular_rhythm=False,
        signal_quality="GOOD",
        peak_status="OK",
        overall_status="NORMAL",
    )

    assert result["label"] == "NORMAL_RHYTHM_LIKE"


def test_irregular_rhythm_review():
    engine = DiagnosticEngine()

    result = engine.classify_record(
        heart_rate=80.0,
        hrv=0.15,
        irregular_rhythm=True,
        signal_quality="GOOD",
        peak_status="OK",
        overall_status="NORMAL",
    )

    assert result["label"] == "IRREGULAR_RHYTHM_REVIEW"


def test_tachycardia_like():
    engine = DiagnosticEngine()

    result = engine.classify_record(
        heart_rate=120.0,
        hrv=0.03,
        irregular_rhythm=False,
        signal_quality="GOOD",
        peak_status="OK",
        overall_status="NORMAL",
    )

    assert result["label"] == "TACHYCARDIA_LIKE"


def test_low_confidence_review():
    engine = DiagnosticEngine()

    result = engine.classify_record(
        heart_rate=78.0,
        hrv=0.03,
        irregular_rhythm=False,
        signal_quality="POOR",
        peak_status="OK",
        overall_status="WARNING",
    )

    assert result["label"] == "LOW_CONFIDENCE_REVIEW"