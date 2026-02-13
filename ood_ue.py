from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence


_MIN_EPS = 1e-6


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class OodUeResult:
    in_domain_score: float
    uncertainty_score: float
    is_ood: bool


@dataclass
class OodUeThresholds:
    ood_threshold: float
    uncertainty_warn_threshold: float

    def __post_init__(self) -> None:
        self.ood_threshold = _clamp01(self.ood_threshold)
        self.uncertainty_warn_threshold = _clamp01(self.uncertainty_warn_threshold)


class OodUeCalibrator:
    """Simple affine calibrator for in-domain score and uncertainty."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, temperature: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.temperature = max(_MIN_EPS, temperature)

    def calibrate_in_domain(self, raw_score: float) -> float:
        raw_score = _clamp01(raw_score)
        logits = self.alpha * raw_score + self.beta
        scaled = logits / self.temperature
        return _clamp01(self._sigmoid(scaled))

    def calibrate_uncertainty(self, raw_uncertainty: float) -> float:
        raw_uncertainty = _clamp01(raw_uncertainty)
        centered = (raw_uncertainty - 0.5) / self.temperature + 0.5
        return _clamp01(centered)

    @staticmethod
    def fit_from_scores(id_scores: Sequence[float], ood_scores: Sequence[float]) -> "OodUeCalibrator":
        """Fit robust defaults from score means with bounded temperature."""
        if len(id_scores) < 2 or len(ood_scores) < 2:
            return OodUeCalibrator()

        id_scores = [_clamp01(s) for s in id_scores]
        ood_scores = [_clamp01(s) for s in ood_scores]

        id_mean = sum(id_scores) / len(id_scores)
        ood_mean = sum(ood_scores) / len(ood_scores)

        midpoint = (id_mean + ood_mean) / 2.0
        beta = -midpoint

        spread = max(_MIN_EPS, abs(id_mean - ood_mean))
        temperature = max(0.5, min(1.5, 0.5 / spread))
        return OodUeCalibrator(alpha=1.0, beta=beta, temperature=temperature)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)


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

    def fit_thresholds(self, id_scores: Iterable[float], target_tpr: float = 0.95, min_samples: int = 10) -> None:
        safe_target_tpr = _clamp01(target_tpr)
        sorted_scores = sorted(_clamp01(s) for s in id_scores)
        if len(sorted_scores) < max(1, min_samples):
            return

        # Lower quantile to preserve desired true-positive rate on in-domain examples.
        q = 1.0 - safe_target_tpr
        idx = int(round(q * (len(sorted_scores) - 1)))
        idx = max(0, min(len(sorted_scores) - 1, idx))
        self.thresholds.ood_threshold = sorted_scores[idx]

    def fit_calibrator(self, id_scores: Sequence[float], ood_scores: Sequence[float]) -> None:
        self.calibrator = OodUeCalibrator.fit_from_scores(id_scores, ood_scores)
