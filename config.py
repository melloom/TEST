from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CONFIG_VERSION = "1.0.0"


@dataclass
class ProfileThresholds:
    ood_threshold: float
    uncertainty_warn_threshold: float
    caution: float
    high_risk: float


@dataclass
class EngineConfig:
    config_version: str = CONFIG_VERSION
    embed_dim: int = 128
    baseline_messages: Tuple[str, ...] = (
        "hello thanks update status meeting project docs review",
        "help question task timeline note summary",
    )
    profiles: Dict[str, ProfileThresholds] = field(
        default_factory=lambda: {
            "strict": ProfileThresholds(ood_threshold=0.62, uncertainty_warn_threshold=0.70, caution=0.28, high_risk=0.62),
            "balanced": ProfileThresholds(ood_threshold=0.68, uncertainty_warn_threshold=0.75, caution=0.35, high_risk=0.70),
            "lenient": ProfileThresholds(ood_threshold=0.74, uncertainty_warn_threshold=0.80, caution=0.45, high_risk=0.80),
        }
    )
    high_risk_patterns: List[Tuple[str, str, float]] = field(
        default_factory=lambda: [
            (
                r"\b(password|otp|one\s*time\s*passcode|2fa\s*code)\b",
                "The message asks for sensitive authentication details (for example password or OTP).",
                0.42,
            ),
            (
                r"\b(urgent|immediately|asap|right\s*now)\b",
                "The message uses urgency pressure that is common in social engineering.",
                0.16,
            ),
            (
                r"\b(click\s+here|verify\s+account|suspend(?:ed)?\s+account)\b",
                "The message contains a phishing-style call to action.",
                0.32,
            ),
            (
                r"\b(bitcoin|gift\s*card|wire\s*transfer|bank\s*account)\b",
                "The message references money transfer patterns associated with scams.",
                0.34,
            ),
            (
                r"\b(ssn|social\s*security|credit\s*card|cvv)\b",
                "The message asks for highly private personal data.",
                0.46,
            ),
        ]
    )


def default_config() -> EngineConfig:
    return EngineConfig()
