from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, List


@dataclass
class PreprocessedMessage:
    raw: str
    normalized: str
    tokens: List[str]


class Preprocessor:
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
    URL_PATTERN = re.compile(r"https?://\S+")
    SPACE_PATTERN = re.compile(r"\s+")

    def process(self, text: str) -> PreprocessedMessage:
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        return PreprocessedMessage(raw=text, normalized=normalized, tokens=tokens)

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = self.URL_PATTERN.sub(" <url> ", text)
        text = self.SPACE_PATTERN.sub(" ", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return self.TOKEN_PATTERN.findall(text)


class HashingEmbedder:
    """Deterministic dependency-free hashed bag-of-words embedder with L2 normalization."""

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_tokens(self, tokens: Iterable[str]) -> List[float]:
        vec = [0.0] * self.dim
        for token in tokens:
            idx = self._stable_index(token)
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_text(self, text: str, preprocessor: Preprocessor | None = None) -> List[float]:
        p = preprocessor or Preprocessor()
        msg = p.process(text)
        return self.embed_tokens(msg.tokens)

    def _stable_index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim
