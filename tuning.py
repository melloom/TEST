from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from confidence_layer import ConfidenceRiskEngine


@dataclass
class TuningExample:
    text: str
    risk_label: str
    is_ood: bool


class ProductionThresholdTuner:
    """Tune profile thresholds from labeled validation examples."""

    def __init__(self, engine: ConfidenceRiskEngine) -> None:
        self.engine = engine

    def tune_profile(
        self,
        profile: str,
        examples: Iterable[TuningExample],
        target_id_tpr: float = 0.95,
        target_high_risk_precision: float = 0.85,
    ) -> dict:
        rows = [self.engine.score_message(ex.text, profile=profile) | {"expected_risk": ex.risk_label, "expected_is_ood": ex.is_ood} for ex in examples]
        if not rows:
            return self.engine.get_runtime_config()["profiles"][profile]

        id_scores = [r["in_domain_score"] for r in rows if not r["expected_is_ood"]]
        ood_uncertainties = [r["uncertainty_score"] for r in rows if r["expected_is_ood"]]

        risk_threshold = self._find_high_risk_threshold(rows, target_high_risk_precision)
        caution_threshold = max(0.05, min(risk_threshold - 0.05, risk_threshold * 0.65))

        ood_threshold = self._id_quantile_threshold(id_scores, target_id_tpr)
        uncertainty_warn_threshold = self._quantile(ood_uncertainties, 0.5, default=0.75)

        tuned = {
            "ood_threshold": ood_threshold,
            "uncertainty_warn_threshold": uncertainty_warn_threshold,
            "caution": caution_threshold,
            "high_risk": risk_threshold,
        }
        self.engine.apply_profile_thresholds(profile, **tuned)
        return tuned

    def _find_high_risk_threshold(self, rows: List[dict], target_precision: float) -> float:
        candidates = sorted({round(float(r["risk_score"]), 4) for r in rows}, reverse=True)
        if not candidates:
            return 0.7

        best = 0.7
        for threshold in candidates:
            tp = sum(1 for r in rows if r["risk_score"] >= threshold and r["expected_risk"] == "high_risk")
            fp = sum(1 for r in rows if r["risk_score"] >= threshold and r["expected_risk"] != "high_risk")
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if precision >= target_precision:
                return threshold
            if precision > 0:
                best = threshold
        return best

    def _id_quantile_threshold(self, id_scores: List[float], target_tpr: float) -> float:
        if not id_scores:
            return 0.68
        q = max(0.0, min(1.0, 1.0 - target_tpr))
        return self._quantile(id_scores, q, default=0.68)

    @staticmethod
    def _quantile(values: List[float], q: float, default: float) -> float:
        if not values:
            return default
        vals = sorted(max(0.0, min(1.0, v)) for v in values)
        idx = int(round(max(0.0, min(1.0, q)) * (len(vals) - 1)))
        return vals[idx]
