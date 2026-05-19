from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompressedContext:
    content: str
    score: float
    source: str
    metadata: dict[str, Any]


class ContextCompressionService:
    def compress(self, query: str, contexts: list[dict[str, Any]], max_items: int = 6) -> list[CompressedContext]:
        keywords = {w.strip("，。？！,. ") for w in query.lower().split() if len(w.strip("，。？！,. ")) > 1}
        scored: list[CompressedContext] = []
        for ctx in contexts:
            content = str(ctx.get("content", ""))
            base = float(ctx.get("score", 0.0))
            overlap = sum(1 for k in keywords if k and k in content.lower())
            score = base + overlap * 0.15
            scored.append(
                CompressedContext(
                    content=self._trim(content),
                    score=score,
                    source=str(ctx.get("source", "")),
                    metadata=dict(ctx.get("metadata", {})),
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:max_items]

    @staticmethod
    def _trim(content: str, limit: int = 350) -> str:
        content = " ".join(content.split())
        return content[:limit]

