from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

from preprocessing import HashingEmbedder, PreprocessedMessage, Preprocessor
from message_risk import MessageRiskClassifier, default_bootstrap_samples
from ood_ue import OodUeDetector, OodUeThresholds
from schema import Prediction, ProfileLabel, validate_profile


class ConfidenceRiskEngine:
    """Configurable confidence layer with preprocessing and embedding pipeline."""

    _PROFILES: Dict[str, Dict[str, float]] = {
        "strict": {"ood_threshold": 0.62, "uncertainty_warn_threshold": 0.70, "caution": 0.28, "high_risk": 0.62},
        "balanced": {"ood_threshold": 0.68, "uncertainty_warn_threshold": 0.75, "caution": 0.35, "high_risk": 0.70},
        "lenient": {"ood_threshold": 0.74, "uncertainty_warn_threshold": 0.80, "caution": 0.45, "high_risk": 0.80},
    }

    def __init__(self, baseline_messages: Iterable[str] | None = None, embed_dim: int = 128) -> None:
        self.preprocessor = Preprocessor()
        self.embedder = HashingEmbedder(dim=embed_dim)

        if baseline_messages is None:
            baseline_messages = [
                "hello thanks update status meeting project docs review",
                "help question task timeline note summary",
            ]

        self._baseline_messages: List[PreprocessedMessage] = []
        self._baseline_vocab: set[str] = set()
        self._baseline_embeddings: List[List[float]] = []
        self._centroid: List[float] = [0.0] * embed_dim

        self.fit_reference(baseline_messages)
        self._ood_detectors = {
            name: OodUeDetector(
                thresholds=OodUeThresholds(
                    ood_threshold=cfg["ood_threshold"],
                    uncertainty_warn_threshold=cfg["uncertainty_warn_threshold"],
                )
            )
            for name, cfg in self._PROFILES.items()
        }

        self._risk_classifier = MessageRiskClassifier(embed_dim=embed_dim)
        self._risk_classifier.fit(default_bootstrap_samples(), lr=0.4, epochs=40)

        self._high_risk_patterns: List[Tuple[str, str, float]] = [
            (r"\b(password|otp|one\s*time\s*passcode|2fa\s*code)\b", "requests sensitive authentication information", 0.42),
            (r"\b(urgent|immediately|asap|right\s*now)\b", "uses urgency language", 0.16),
            (r"\b(click\s+here|verify\s+account|suspend(?:ed)?\s+account)\b", "contains phishing-style call to action", 0.32),
            (r"\b(bitcoin|gift\s*card|wire\s*transfer|bank\s*account)\b", "references payment transfer patterns", 0.34),
            (r"\b(ssn|social\s*security|credit\s*card|cvv)\b", "asks for private personal data", 0.46),
        ]

    def fit_reference(self, messages: Iterable[str]) -> None:
        processed = [self.preprocessor.process(m) for m in messages if m.strip()]
        if not processed:
            return

        self._baseline_messages = processed
        self._baseline_vocab = set()
        self._baseline_embeddings = []

        for msg in processed:
            self._baseline_vocab.update(msg.tokens)
            self._baseline_embeddings.append(self.embedder.embed_tokens(msg.tokens))

        self._centroid = self._compute_centroid(self._baseline_embeddings)

    def predict(self, text: str, profile: str = "balanced") -> Prediction:
        selected: ProfileLabel = validate_profile(profile)
        cfg = self._PROFILES[selected]

        msg = self.preprocessor.process(text)
        embedding = self.embedder.embed_tokens(msg.tokens)

        ood_eval, ood_reason = self._ood_uncertainty(msg.tokens, embedding, selected)
        is_ood = ood_eval.is_ood
        uncertainty = ood_eval.uncertainty_score
        fused_risk_score, reasons = self._risk_score(msg)

        if ood_reason:
            reasons.insert(0, ood_reason)

        risk_label = self._risk_label(fused_risk_score, caution=cfg["caution"], high_risk=cfg["high_risk"])
        warn_threshold = self._ood_detectors[selected].thresholds.uncertainty_warn_threshold
        action = self._action(risk_label, is_ood, uncertainty, warn_threshold)

        return Prediction(
            is_ood=is_ood,
            uncertainty_score=uncertainty,
            risk_label=risk_label,
            risk_score=fused_risk_score,
            reasons=reasons or ["no strong risk signals were found"],
            action=action,
            profile=selected,
        )

    def _ood_uncertainty(self, tokens: List[str], embedding: List[float], profile: ProfileLabel):
        if not tokens:
            eval_result = self._ood_detectors[profile].evaluate(0.0, 1.0)
            return eval_result, "input is empty or unparseable"

        overlap = len(set(tokens) & self._baseline_vocab)
        overlap_ratio = overlap / max(len(set(tokens)), 1)

        centroid_similarity = self._cosine_similarity(embedding, self._centroid)
        nearest_similarity = max((self._cosine_similarity(embedding, b) for b in self._baseline_embeddings), default=0.0)

        alpha_tokens = sum(1 for t in tokens if re.search(r"[a-zA-Z]", t))
        alpha_ratio = alpha_tokens / max(len(tokens), 1)

        in_domain_raw = 0.40 * overlap_ratio + 0.35 * centroid_similarity + 0.25 * nearest_similarity
        uncertainty_raw = (1.0 - in_domain_raw) * 0.85 + (1.0 - alpha_ratio) * 0.15

        eval_result = self._ood_detectors[profile].evaluate(in_domain_raw, uncertainty_raw)
        reason = "" if not eval_result.is_ood else "low reference similarity in vocabulary/embedding space"
        return eval_result, reason


    def fit_ood_thresholds(self, profile: ProfileLabel, id_in_domain_scores: List[float], target_tpr: float = 0.95) -> None:
        self._ood_detectors[profile].fit_thresholds(id_in_domain_scores, target_tpr=target_tpr)

    def fit_ood_calibrator(self, profile: ProfileLabel, id_scores: List[float], ood_scores: List[float]) -> None:
        self._ood_detectors[profile].fit_calibrator(id_scores, ood_scores)

    def _risk_score(self, msg: PreprocessedMessage) -> Tuple[float, List[str]]:
        lowered = msg.normalized
        rule_score = 0.0
        reasons: List[str] = []

        for pattern, reason, weight in self._high_risk_patterns:
            if re.search(pattern, lowered):
                rule_score += weight
                reasons.append(reason)

        if len(msg.raw) > 280:
            rule_score += 0.05
            reasons.append("long-form message can hide mixed intent")

        punctuation_burst = msg.raw.count("!") + msg.raw.count("?")
        if punctuation_burst >= 4:
            rule_score += 0.10
            reasons.append("high punctuation intensity")

        if re.search(r"<url>", lowered) and re.search(r"\b(login|verify|account|secure)\b", lowered):
            rule_score += 0.15
            reasons.append("link combined with account-security prompt")

        rule_prob = self._clamp(1 - math.exp(-rule_score))
        clf_prob = self._risk_classifier.predict_score_from_message(msg)

        # fused score: classifier captures semantic context; rules enforce explicit safety signals
        fused = self._clamp(0.6 * clf_prob + 0.4 * rule_prob)

        if clf_prob >= 0.65 and "classifier semantic risk signal" not in reasons:
            reasons.append("classifier semantic risk signal")

        return fused, reasons

    @staticmethod
    def _risk_label(score: float, caution: float, high_risk: float) -> str:
        if score >= high_risk:
            return "high_risk"
        if score >= caution:
            return "caution"
        return "safe"

    @staticmethod
    def _action(risk_label: str, is_ood: bool, uncertainty: float, uncertainty_warn_threshold: float) -> str:
        if risk_label == "high_risk":
            return "block"
        if risk_label == "caution" or (is_ood and uncertainty > uncertainty_warn_threshold):
            return "warn"
        if is_ood:
            return "escalate"
        return "allow"

    @staticmethod
    def _compute_centroid(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        centroid = [0.0] * dim
        for v in vectors:
            for i, value in enumerate(v):
                centroid[i] += value
        n = len(vectors)
        centroid = [x / n for x in centroid]
        norm = math.sqrt(sum(x * x for x in centroid))
        if norm > 0:
            centroid = [x / norm for x in centroid]
        return centroid

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
