from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.context_compression import ContextCompressionService
from services.graph_rag import GraphRAGPipeline


@dataclass
class RetrievalBundle:
    raw_contexts: list[dict[str, Any]] = field(default_factory=list)
    compressed_contexts: list[dict[str, Any]] = field(default_factory=list)
    answer_contexts: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_steps: list[str] = field(default_factory=list)


class RetrievalOrchestrator:
    def __init__(self, graph_rag: GraphRAGPipeline, compressor: ContextCompressionService) -> None:
        self.graph_rag = graph_rag
        self.compressor = compressor

    async def retrieve(self, query: str, top_k: int = 8) -> RetrievalBundle:
        raw = await self.graph_rag.retrieve(query, top_k=top_k)
        raw_dicts = [
            {
                "content": c.content,
                "source": c.source_type,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in raw
        ]
        compressed = self.compressor.compress(query, raw_dicts, max_items=top_k)
        return RetrievalBundle(
            raw_contexts=raw_dicts,
            compressed_contexts=[c.__dict__ for c in compressed],
            answer_contexts=[c.__dict__ for c in compressed[:top_k]],
            confidence=min(1.0, sum(c.score for c in compressed[:top_k]) / max(len(compressed[:top_k]), 1)),
            reasoning_steps=[
                f"raw={len(raw_dicts)}",
                f"compressed={len(compressed)}",
            ],
        )

