from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


@dataclass
class OodUeResult:
    in_domain_score: float
    uncertainty_score: float
    is_ood: bool


@dataclass
class OodUeThresholds:
    ood_threshold: float
    uncertainty_warn_threshold: float


class OodUeCalibrator:
    """Simple affine calibrator for in-domain score and uncertainty."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, temperature: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.temperature = max(1e-6, temperature)

    def calibrate_in_domain(self, raw_score: float) -> float:
        logits = self.alpha * raw_score + self.beta
        scaled = logits / self.temperature
        return self._clamp(self._sigmoid(scaled))

    def calibrate_uncertainty(self, raw_uncertainty: float) -> float:
        # keep monotonic shape while allowing smoothing via temperature
        centered = (raw_uncertainty - 0.5) / self.temperature + 0.5
        return self._clamp(centered)

    @staticmethod
    def fit_from_scores(id_scores: Sequence[float], ood_scores: Sequence[float]) -> "OodUeCalibrator":
        """Choose simple beta shift that separates ID and OOD means."""
        if not id_scores or not ood_scores:
            return OodUeCalibrator()
        id_mean = sum(id_scores) / len(id_scores)
        ood_mean = sum(ood_scores) / len(ood_scores)
        midpoint = (id_mean + ood_mean) / 2
        beta = -midpoint
        spread = max(1e-3, abs(id_mean - ood_mean))
        temperature = max(0.25, min(2.0, 0.5 / spread))
        return OodUeCalibrator(alpha=1.0, beta=beta, temperature=temperature)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))


class OodUeDetector:
    def __init__(self, thresholds: OodUeThresholds | None = None, calibrator: OodUeCalibrator | None = None) -> None:
        self.thresholds = thresholds or OodUeThresholds(ood_threshold=0.68, uncertainty_warn_threshold=0.75)
        self.calibrator = calibrator or OodUeCalibrator()

    def evaluate(self, in_domain_raw: float, uncertainty_raw: float) -> OodUeResult:
        in_domain_score = self.calibrator.calibrate_in_domain(in_domain_raw)
        uncertainty_score = self.calibrator.calibrate_uncertainty(uncertainty_raw)
        is_ood = in_domain_score < self.thresholds.ood_threshold
        return OodUeResult(
            in_domain_score=in_domain_score,
            uncertainty_score=uncertainty_score,
            is_ood=is_ood,
        )

    def fit_thresholds(self, id_scores: Iterable[float], target_tpr: float = 0.95) -> None:
        sorted_scores = sorted(id_scores)
        if not sorted_scores:
            return
        n = len(sorted_scores)
        idx = max(0, min(n - 1, int((1 - target_tpr) * (n - 1))))
        self.thresholds.ood_threshold = sorted_scores[idx]

    def fit_calibrator(self, id_scores: Sequence[float], ood_scores: Sequence[float]) -> None:
        self.calibrator = OodUeCalibrator.fit_from_scores(id_scores, ood_scores)
