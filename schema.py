from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, TypedDict, cast

SCHEMA_VERSION = "1.0.0"

RiskLabel = Literal["safe", "caution", "high_risk"]
ActionLabel = Literal["allow", "warn", "block", "escalate"]
ProfileLabel = Literal["strict", "balanced", "lenient"]


class PredictionPayload(TypedDict):
    schema_version: str
    is_ood: bool
    uncertainty_score: float
    risk_label: RiskLabel
    risk_score: float
    reasons: List[str]
    action: ActionLabel
    profile: ProfileLabel


@dataclass
class Prediction:
    is_ood: bool
    uncertainty_score: float
    risk_label: RiskLabel
    risk_score: float
    reasons: List[str]
    action: ActionLabel
    profile: ProfileLabel

    def to_dict(self) -> PredictionPayload:
        payload: Dict[str, object] = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["uncertainty_score"] = round(float(payload["uncertainty_score"]), 4)
        payload["risk_score"] = round(float(payload["risk_score"]), 4)
        return cast(PredictionPayload, payload)


def validate_profile(profile: str) -> ProfileLabel:
    if profile in {"strict", "balanced", "lenient"}:
        return cast(ProfileLabel, profile)
    return "balanced"
