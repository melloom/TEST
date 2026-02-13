from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

from preprocessing import HashingEmbedder, PreprocessedMessage, Preprocessor


@dataclass
class MessageRiskModel:
    weights: List[float]
    bias: float


class MessageRiskClassifier:
    """Lightweight linear classifier over hashed embeddings."""

    def __init__(self, embed_dim: int = 128, model: MessageRiskModel | None = None) -> None:
        self.preprocessor = Preprocessor()
        self.embedder = HashingEmbedder(dim=embed_dim)
        if model is None:
            self.model = MessageRiskModel(weights=[0.0] * embed_dim, bias=-0.2)
        else:
            self.model = model

    def predict_score(self, text: str) -> float:
        msg = self.preprocessor.process(text)
        return self.predict_score_from_message(msg)

    def predict_score_from_message(self, msg: PreprocessedMessage) -> float:
        x = self.embedder.embed_tokens(msg.tokens)
        logit = self.model.bias + sum(w * xi for w, xi in zip(self.model.weights, x))
        return self._sigmoid(logit)

    def fit(self, samples: Sequence[Tuple[str, int]], lr: float = 0.5, epochs: int = 50) -> None:
        if not samples:
            return
        for _ in range(epochs):
            grad_w = [0.0] * len(self.model.weights)
            grad_b = 0.0
            for text, y in samples:
                y_true = 1.0 if y else 0.0
                msg = self.preprocessor.process(text)
                x = self.embedder.embed_tokens(msg.tokens)
                pred = self.predict_score_from_message(msg)
                err = pred - y_true
                for i, xi in enumerate(x):
                    grad_w[i] += err * xi
                grad_b += err

            n = float(len(samples))
            for i in range(len(self.model.weights)):
                self.model.weights[i] -= lr * (grad_w[i] / n)
            self.model.bias -= lr * (grad_b / n)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)


def default_bootstrap_samples() -> List[Tuple[str, int]]:
    return [
        ("hello team project update and timeline", 0),
        ("please review the meeting notes", 0),
        ("urgent verify your account now click here", 1),
        ("send your otp and password immediately", 1),
        ("wire transfer gift card needed asap", 1),
        ("thanks for the status summary", 0),
    ]
