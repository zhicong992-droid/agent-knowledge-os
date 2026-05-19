"""
多模态服务 — 统一处理不同模态数据的嵌入与检索

职责:
  1. 文本嵌入
  2. 图像嵌入（通过 LLM 视觉描述后再嵌入）
  3. 表格嵌入（结构化 → 自然语言 → 嵌入）
  4. 跨模态检索时的分数加权融合
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocType, DocumentChunk
from config import settings


@dataclass
class MultimodalSearchResult:
    content: str
    modality: str
    score: float
    metadata: dict[str, Any]


class MultimodalService:
    """
    多模态处理服务

    策略: 各模态先转为文本表示，再做统一嵌入
    不同模态在检索时根据与查询的匹配度施加不同权重
    """

    MODALITY_WEIGHTS: dict[str, float] = {
        DocType.TEXT.value: 1.0,
        DocType.MARKDOWN.value: 1.0,
        DocType.PDF.value: 0.95,
        DocType.TABLE.value: 0.9,
        DocType.IMAGE.value: 0.85,
    }

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """批量嵌入文档块"""
        texts = [c.content for c in chunks]
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        """嵌入查询文本"""
        return await self.embeddings.aembed_query(query)

    def weighted_rerank(
        self,
        results: list[tuple[DocumentChunk, float]],
    ) -> list[MultimodalSearchResult]:
        """
        跨模态加权重排序
        对不同模态的检索结果施加不同权重后统一排序
        """
        reranked: list[MultimodalSearchResult] = []
        for chunk, score in results:
            weight = self.MODALITY_WEIGHTS.get(chunk.doc_type.value, 1.0)
            reranked.append(MultimodalSearchResult(
                content=chunk.content,
                modality=chunk.doc_type.value,
                score=score * weight,
                metadata=chunk.metadata,
            ))
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
