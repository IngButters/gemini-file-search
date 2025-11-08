"""Semantic matching utilities for normalized table rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import math

from .table_normalization import NormalizedRow


EmbeddingFunction = Callable[[Sequence[str]], List[List[float]]]


@dataclass
class MatchCandidate:
    """Represents a potential match between two rows."""

    base: NormalizedRow
    peer: NormalizedRow
    score: float
    price_delta: Optional[float]


class SemanticMatcher:
    """Semantic matcher combining embeddings with rule enforcement."""

    def __init__(
        self,
        embed_fn: EmbeddingFunction,
        price_column: str = "price",
        max_candidates: int = 5,
        minimum_score: float = 0.5,
        price_threshold: float = 0.25,
    ) -> None:
        self.embed_fn = embed_fn
        self.price_column = price_column
        self.max_candidates = max_candidates
        self.minimum_score = minimum_score
        self.price_threshold = price_threshold

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _price_delta(self, base: NormalizedRow, peer: NormalizedRow) -> Optional[float]:
        base_price = base.get_numeric(self.price_column)
        peer_price = peer.get_numeric(self.price_column)
        if base_price is None or peer_price is None or peer_price == 0:
            return None
        return abs(base_price - peer_price) / peer_price

    def _passes_rules(self, base: NormalizedRow, peer: NormalizedRow) -> Tuple[bool, Optional[float]]:
        if base.unit and peer.unit and base.unit != peer.unit:
            return False, None
        if base.currency and peer.currency and base.currency != peer.currency:
            return False, None
        delta = self._price_delta(base, peer)
        if delta is not None and delta > self.price_threshold:
            return False, delta
        return True, delta

    def match(self, base_rows: Iterable[NormalizedRow], peer_rows: Iterable[NormalizedRow]) -> List[MatchCandidate]:
        base_rows = list(base_rows)
        peer_rows = list(peer_rows)

        base_embeddings = self.embed_fn([row.descriptor for row in base_rows])
        peer_embeddings = self.embed_fn([row.descriptor for row in peer_rows])

        candidates: List[MatchCandidate] = []
        for base_idx, base_row in enumerate(base_rows):
            scored: List[Tuple[float, NormalizedRow, Optional[float]]] = []
            for peer_idx, peer_row in enumerate(peer_rows):
                score = self._cosine_similarity(base_embeddings[base_idx], peer_embeddings[peer_idx])
                if score < self.minimum_score:
                    continue
                passes, delta = self._passes_rules(base_row, peer_row)
                if not passes:
                    continue
                scored.append((score, peer_row, delta))
            scored.sort(key=lambda item: item[0], reverse=True)
            for score, peer_row, delta in scored[: self.max_candidates]:
                candidates.append(MatchCandidate(base=base_row, peer=peer_row, score=score, price_delta=delta))

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates


__all__ = ["MatchCandidate", "SemanticMatcher"]
