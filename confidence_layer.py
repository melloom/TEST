from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Tuple


@dataclass
class Prediction:
    is_ood: bool
    uncertainty_score: float
    risk_label: str
    risk_score: float
    reasons: List[str]
    action: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_ood": self.is_ood,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "risk_label": self.risk_label,
            "risk_score": round(self.risk_score, 4),
            "reasons": self.reasons,
            "action": self.action,
        }


class ConfidenceRiskEngine:
    """Simple no-dependency confidence layer.

    This is intentionally heuristic for v0 so we can ship fast and iterate.
    """

    def __init__(self) -> None:
        # baseline corpus-style tokens representing expected in-distribution messages
        baseline = (
            "hello thanks update status meeting project docs review "
            "help question task timeline note summary"
        )
        self._baseline_vocab = set(self._tokenize(baseline))

        self._high_risk_patterns: List[Tuple[str, str, float]] = [
            (r"\b(password|otp|one\s*time\s*passcode|2fa\s*code)\b", "requests sensitive authentication information", 0.40),
            (r"\b(urgent|immediately|asap|right\s*now)\b", "uses urgency language", 0.15),
            (r"\b(click\s+here|verify\s+account|suspend(?:ed)?\s+account)\b", "contains phishing-style call to action", 0.30),
            (r"\b(bitcoin|gift\s*card|wire\s*transfer|bank\s*account)\b", "references payment transfer patterns", 0.30),
            (r"\b(ssn|social\s*security|credit\s*card|cvv)\b", "asks for private personal data", 0.45),
        ]

    def predict(self, text: str) -> Prediction:
        tokens = self._tokenize(text)
        token_set = set(tokens)

        is_ood, uncertainty, ood_reason = self._ood_uncertainty(token_set)
        risk_score, reasons = self._risk_score(text)

        if ood_reason:
            reasons.insert(0, ood_reason)

        risk_label = self._risk_label(risk_score)
        action = self._action(risk_label, is_ood, uncertainty)

        return Prediction(
            is_ood=is_ood,
            uncertainty_score=uncertainty,
            risk_label=risk_label,
            risk_score=risk_score,
            reasons=reasons or ["no strong risk signals were found"],
            action=action,
        )

    def _ood_uncertainty(self, token_set: set[str]) -> Tuple[bool, float, str]:
        if not token_set:
            return True, 1.0, "input is empty or unparseable"

        overlap = len(token_set & self._baseline_vocab)
        ratio = overlap / max(len(token_set), 1)

        # low overlap => higher uncertainty / likely OOD
        uncertainty = 1.0 - ratio
        is_ood = ratio < 0.2
        reason = "" if not is_ood else "low vocabulary overlap with expected message domain"
        return is_ood, self._clamp(uncertainty), reason

    def _risk_score(self, text: str) -> Tuple[float, List[str]]:
        lowered = text.lower()
        score = 0.0
        reasons: List[str] = []

        for pattern, reason, weight in self._high_risk_patterns:
            if re.search(pattern, lowered):
                score += weight
                reasons.append(reason)

        # structural uncertainty cues
        if len(text) > 280:
            score += 0.05
            reasons.append("long-form message can hide mixed intent")

        punctuation_burst = lowered.count("!") + lowered.count("?")
        if punctuation_burst >= 4:
            score += 0.10
            reasons.append("high punctuation intensity")

        # squash to [0,1] with smooth curve
        score = 1 - math.exp(-score)
        return self._clamp(score), reasons

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.70:
            return "high_risk"
        if score >= 0.35:
            return "caution"
        return "safe"

    @staticmethod
    def _action(risk_label: str, is_ood: bool, uncertainty: float) -> str:
        if risk_label == "high_risk":
            return "block"
        if risk_label == "caution" or (is_ood and uncertainty > 0.75):
            return "warn"
        if is_ood:
            return "escalate"
        return "allow"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
