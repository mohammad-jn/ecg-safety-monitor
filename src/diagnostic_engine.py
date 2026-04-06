class DiagnosticEngine:
    def __init__(self) -> None:
        pass

    def classify_record(
        self,
        heart_rate: float,
        hrv: float,
        irregular_rhythm: bool,
        signal_quality: str,
        peak_status: str,
        overall_status: str,
    ) -> dict:
        if signal_quality != "GOOD":
            return {
                "label": "LOW_CONFIDENCE_REVIEW",
                "explanation": "Signal quality is not sufficient for confident rhythm interpretation.",
            }

        if peak_status != "OK":
            return {
                "label": "LOW_CONFIDENCE_REVIEW",
                "explanation": "Peak detection quality is not sufficient for confident rhythm interpretation.",
            }

        if overall_status == "CRITICAL":
            if heart_rate > 140:
                return {
                    "label": "TACHYCARDIA_LIKE",
                    "explanation": "Heart rate is critically elevated and requires review.",
                }

            if heart_rate < 40:
                return {
                    "label": "BRADYCARDIA_LIKE",
                    "explanation": "Heart rate is critically low and requires review.",
                }

        if heart_rate > 100:
            return {
                "label": "TACHYCARDIA_LIKE",
                "explanation": "Heart rate is above the normal resting range.",
            }

        if heart_rate < 60:
            return {
                "label": "BRADYCARDIA_LIKE",
                "explanation": "Heart rate is below the normal resting range.",
            }

        if irregular_rhythm or hrv > 0.1:
            return {
                "label": "IRREGULAR_RHYTHM_REVIEW",
                "explanation": "Rhythm variability is elevated and may indicate an irregular pattern.",
            }

        return {
            "label": "NORMAL_RHYTHM_LIKE",
            "explanation": "Rhythm features are stable and within expected limits.",
        }