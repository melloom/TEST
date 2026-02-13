from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from confidence_layer import ConfidenceRiskEngine


@dataclass
class ReusablePredictorBundle:
    message_predictor: "MessageRiskPredictor"
    mood_predictor: "MoodRiskPredictor"
    code_predictor: "CodeRiskPredictor"
    assistant_predictor: "AssistantSafetyPredictor"


class MessageRiskPredictor:
    def __init__(self, engine: ConfidenceRiskEngine | None = None, profile: str = "balanced") -> None:
        self.engine = engine or ConfidenceRiskEngine()
        self.profile = profile

    def predict(self, text: str) -> Dict[str, object]:
        return self.engine.predict(text, profile=self.profile).to_dict()


class MoodRiskPredictor:
    def __init__(self, engine: ConfidenceRiskEngine | None = None, profile: str = "balanced") -> None:
        self.engine = engine or ConfidenceRiskEngine()
        self.profile = profile

    def predict(self, mood_text: str, context: str = "") -> Dict[str, object]:
        text = f"{mood_text} {context}".strip()
        result = self.engine.predict(text, profile=self.profile).to_dict()
        result["domain"] = "mood"
        return result


class CodeRiskPredictor:
    def __init__(self, engine: ConfidenceRiskEngine | None = None, profile: str = "strict") -> None:
        self.engine = engine or ConfidenceRiskEngine()
        self.profile = profile

    def predict(self, code_snippet: str, task_context: str = "") -> Dict[str, object]:
        text = f"{task_context}\n{code_snippet}".strip()
        result = self.engine.predict(text, profile=self.profile).to_dict()
        result["domain"] = "code"
        return result


class AssistantSafetyPredictor:
    def __init__(self, engine: ConfidenceRiskEngine | None = None, profile: str = "balanced") -> None:
        self.engine = engine or ConfidenceRiskEngine()
        self.profile = profile

    def predict(self, user_message: str, assistant_intent: str = "") -> Dict[str, object]:
        text = f"{user_message} {assistant_intent}".strip()
        result = self.engine.predict(text, profile=self.profile).to_dict()
        result["domain"] = "assistant"
        return result


def build_reusable_bundle(engine: ConfidenceRiskEngine | None = None) -> ReusablePredictorBundle:
    shared_engine = engine or ConfidenceRiskEngine()
    return ReusablePredictorBundle(
        message_predictor=MessageRiskPredictor(shared_engine, profile="balanced"),
        mood_predictor=MoodRiskPredictor(shared_engine, profile="balanced"),
        code_predictor=CodeRiskPredictor(shared_engine, profile="strict"),
        assistant_predictor=AssistantSafetyPredictor(shared_engine, profile="balanced"),
    )
