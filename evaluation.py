from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from confidence_layer import ConfidenceRiskEngine


@dataclass
class EvalExample:
    text: str
    risk_label: str
    is_ood: bool


@dataclass
class EvalRow:
    text: str
    expected_risk_label: str
    predicted_risk_label: str
    expected_is_ood: bool
    predicted_is_ood: bool
    risk_score: float
    uncertainty_score: float


class EvaluationHarness:
    def __init__(self, engine: ConfidenceRiskEngine | None = None) -> None:
        self.engine = engine or ConfidenceRiskEngine()

    def evaluate(self, examples: Iterable[EvalExample], profile: str = "balanced") -> Dict[str, object]:
        rows: List[EvalRow] = []
        for ex in examples:
            pred = self.engine.predict(ex.text, profile=profile)
            rows.append(
                EvalRow(
                    text=ex.text,
                    expected_risk_label=ex.risk_label,
                    predicted_risk_label=pred.risk_label,
                    expected_is_ood=ex.is_ood,
                    predicted_is_ood=pred.is_ood,
                    risk_score=pred.risk_score,
                    uncertainty_score=pred.uncertainty_score,
                )
            )

        return {
            "count": len(rows),
            "risk_macro_f1": self._risk_macro_f1(rows),
            "ood_accuracy": self._ood_accuracy(rows),
            "error_analysis": self._error_analysis(rows),
            "dashboard_markdown": self._dashboard(rows),
        }

    def _risk_macro_f1(self, rows: List[EvalRow]) -> float:
        labels = ["safe", "caution", "high_risk"]
        if not rows:
            return 0.0

        f1s: List[float] = []
        for label in labels:
            tp = sum(1 for r in rows if r.expected_risk_label == label and r.predicted_risk_label == label)
            fp = sum(1 for r in rows if r.expected_risk_label != label and r.predicted_risk_label == label)
            fn = sum(1 for r in rows if r.expected_risk_label == label and r.predicted_risk_label != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            if precision + recall == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * precision * recall / (precision + recall))
        return round(sum(f1s) / len(f1s), 4)

    def _ood_accuracy(self, rows: List[EvalRow]) -> float:
        if not rows:
            return 0.0
        correct = sum(1 for r in rows if r.expected_is_ood == r.predicted_is_ood)
        return round(correct / len(rows), 4)

    def _error_analysis(self, rows: List[EvalRow]) -> Dict[str, List[Dict[str, object]]]:
        risk_fp: List[Dict[str, object]] = []
        risk_fn: List[Dict[str, object]] = []
        ood_fp: List[Dict[str, object]] = []
        ood_fn: List[Dict[str, object]] = []

        for r in rows:
            if r.predicted_risk_label == "high_risk" and r.expected_risk_label != "high_risk":
                risk_fp.append(self._row_dict(r))
            if r.predicted_risk_label != "high_risk" and r.expected_risk_label == "high_risk":
                risk_fn.append(self._row_dict(r))
            if r.predicted_is_ood and not r.expected_is_ood:
                ood_fp.append(self._row_dict(r))
            if not r.predicted_is_ood and r.expected_is_ood:
                ood_fn.append(self._row_dict(r))

        return {
            "risk_false_positives": risk_fp,
            "risk_false_negatives": risk_fn,
            "ood_false_positives": ood_fp,
            "ood_false_negatives": ood_fn,
        }

    def _dashboard(self, rows: List[EvalRow]) -> str:
        risk_macro_f1 = self._risk_macro_f1(rows)
        ood_accuracy = self._ood_accuracy(rows)
        errors = self._error_analysis(rows)
        return (
            "# Confidence Layer Evaluation Dashboard\n\n"
            f"- Samples: {len(rows)}\n"
            f"- Risk macro-F1: {risk_macro_f1}\n"
            f"- OoD accuracy: {ood_accuracy}\n"
            f"- Risk FP/FN: {len(errors['risk_false_positives'])}/{len(errors['risk_false_negatives'])}\n"
            f"- OoD FP/FN: {len(errors['ood_false_positives'])}/{len(errors['ood_false_negatives'])}\n"
        )

    @staticmethod
    def _row_dict(row: EvalRow) -> Dict[str, object]:
        return {
            "text": row.text,
            "expected_risk_label": row.expected_risk_label,
            "predicted_risk_label": row.predicted_risk_label,
            "expected_is_ood": row.expected_is_ood,
            "predicted_is_ood": row.predicted_is_ood,
            "risk_score": row.risk_score,
            "uncertainty_score": row.uncertainty_score,
        }
