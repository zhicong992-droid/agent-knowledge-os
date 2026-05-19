"""
问答 Agent — 混合检索 (Vector + Graph) + 多跳推理 + 答案生成

核心能力:
  1. 意图识别 & 查询改写
  2. 向量检索 (语义相似度)
  3. 图谱检索 (Cypher 查询 / 子图遍历)
  4. 混合排序 & 重排序
  5. 基于检索结果的答案生成（带引用来源）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.retrieval_pipeline import RetrievalBundle, RetrievalOrchestrator


class QueryIntent(str, Enum):
    FACTOID = "factoid"           # 事实型问题
    ANALYTICAL = "analytical"     # 分析型问题
    COMPARATIVE = "comparative"   # 对比型问题
    PROCEDURAL = "procedural"     # 流程型问题
    EXPLORATORY = "exploratory"   # 探索型问题


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # "vector" | "graph" | "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


INTENT_PROMPT = """\
你是一个查询意图分类器。根据用户问题，返回意图类别（只返回类别名）：
- factoid: 事实型（谁/什么/哪里/何时）
- analytical: 分析型（为什么/怎么理解）
- comparative: 对比型（A和B有什么区别）
- procedural: 流程型（怎么做/步骤）
- exploratory: 探索型（有哪些/概述）
"""

QUERY_REWRITE_PROMPT = """\
你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
3. 返回 JSON: {"queries": ["查询1", "查询2"], "entities": ["实体1"], "keywords": ["关键词1"]}
"""

CYPHER_GENERATION_PROMPT = """\
你是一个 Neo4j Cypher 查询生成专家。根据用户问题和提取的实体，生成 Cypher 查询。

知识图谱 Schema:
- 节点标签: Person, Organization, Technology, Product, Concept, Location
- 关系类型: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- 节点属性: name, type, description, created_at, version

生成 1-2 条 Cypher 查询，返回 JSON: {"queries": ["MATCH ...", "MATCH ..."]}
只返回 JSON，不要其他文字。
"""

ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
"""


class QAAgent:
    """
    问答 Agent

    工作流:
      query → intent_classify → rewrite → parallel_retrieve → rerank → generate_answer
    """

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        retrieval_orchestrator: RetrievalOrchestrator | None = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.retrieval_orchestrator = retrieval_orchestrator

    # ── public API ───────────────────────────────────────────

    async def answer(self, question: str) -> QAResult:
        """完整问答流程"""
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        if self.retrieval_orchestrator:
            bundle = await self.retrieval_orchestrator.retrieve(question, top_k=settings.retrieval_top_k)
            contexts = [
                RetrievedContext(
                    content=c["content"],
                    source=c["source"],
                    score=float(c["score"]),
                    retrieval_type=c.get("metadata", {}).get("type", "hybrid"),
                    metadata=c.get("metadata", {}),
                )
                for c in bundle.answer_contexts
            ]
            reasoning = bundle.reasoning_steps + [f"confidence={bundle.confidence:.2f}"]
        else:
            vector_contexts = await self._vector_retrieve(rewritten)
            graph_contexts = await self._graph_retrieve(question, rewritten)
            if not vector_contexts and not graph_contexts:
                graph_contexts = await self._keyword_graph_fallback(question, rewritten)
            contexts = self._hybrid_rerank(vector_contexts + graph_contexts)[:8]
            reasoning = []

        if not contexts:
            contexts = await self._keyword_graph_fallback(question, rewritten)

        answer_text, generated_reasoning = await self._generate_answer(question, contexts, intent)
        reasoning.extend(generated_reasoning)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=contexts,
            intent=intent,
            confidence=self._calc_confidence(contexts),
            reasoning_steps=reasoning,
        )

    # ── intent classification ────────────────────────────────

    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        raw = resp.content.strip().lower()
        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    # ── query rewriting ──────────────────────────────────────

    async def _rewrite_query(self, question: str) -> dict:
        import json
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return {"queries": [question], "entities": [], "keywords": []}

    # ── vector retrieval ─────────────────────────────────────

    async def _vector_retrieve(self, rewritten: dict) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in rewritten.get("queries", []):
            results = await self.vector_store.search(query, top_k=5)
            for doc, score in results:
                contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=doc.get("metadata", {}),
                ))
        return contexts

    # ── graph retrieval ──────────────────────────────────────

    async def _graph_retrieve(self, question: str, rewritten: dict) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        import json
        entities = rewritten.get("entities", [])
        messages = [
            SystemMessage(content=CYPHER_GENERATION_PROMPT),
            HumanMessage(content=f"问题: {question}\n实体: {entities}"),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            cypher_data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            cypher_data = {"queries": []}

        contexts: list[RetrievedContext] = []
        for cypher in cypher_data.get("queries", []):
            try:
                records = await self.knowledge_graph.execute_cypher(cypher)
                for record in records:
                    contexts.append(RetrievedContext(
                        content=str(record),
                        source="knowledge_graph",
                        score=0.8,
                        retrieval_type="graph",
                        metadata={"cypher": cypher},
                    ))
            except Exception:
                continue
        return contexts

    async def _keyword_graph_fallback(self, question: str, rewritten: dict) -> list[RetrievedContext]:
        """当 LLM 生成实体失败时，使用关键词直接在图谱中做兜底召回。"""
        if not self.knowledge_graph:
            return []

        keywords: list[str] = []
        for item in rewritten.get("entities", []) + rewritten.get("keywords", []):
            if item and item not in keywords:
                keywords.append(str(item))

        for token in question.replace("？", " ").replace("?", " ").replace("，", " ").split():
            token = token.strip()
            if len(token) > 1 and token not in keywords:
                keywords.append(token)

        contexts: list[RetrievedContext] = []
        for keyword in keywords[:5]:
            try:
                records = await self.knowledge_graph.search_entities(keyword, limit=5)
            except Exception:
                records = []
            for record in records:
                content = record.get("description") or record.get("name") or str(record)
                contexts.append(RetrievedContext(
                    content=content,
                    source=f"knowledge_graph:{record.get('name', keyword)}",
                    score=0.72,
                    retrieval_type="graph",
                    metadata={"keyword": keyword, "matched": record.get("name", keyword)},
                ))
        return contexts

    # ── hybrid reranking ─────────────────────────────────────

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        混合重排序：向量分数 + 图谱分数加权
        图谱检索结果天然带有结构化关系，给予略高权重
        """
        weight_map = {"vector": 1.0, "graph": 1.2, "hybrid": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)

        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

    # ── answer generation ────────────────────────────────────

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        if contexts:
            top = contexts[0]
            answer = (
                f"根据检索结果，相关信息来自 {top.source}，"
                f"内容指向：{top.content[:160]}"
            )
            return answer, [
                f"识别问题意图: {intent.value}",
                f"检索到 {len(contexts)} 条相关上下文",
                "命中本地检索结果，直接生成引用式回答",
            ]

        context_text = "\n\n".join(
            f"[来源 {i+1}: {c.source} | 类型: {c.retrieval_type} | 分数: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"识别问题意图: {intent.value}",
            f"检索到 {len(contexts)} 条相关上下文",
            f"向量检索: {sum(1 for c in contexts if c.retrieval_type == 'vector')} 条",
            f"图谱检索: {sum(1 for c in contexts if c.retrieval_type == 'graph')} 条",
        ]

        messages = [
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(content=f"上下文信息:\n{context_text}\n\n用户问题: {question}"),
        ]
        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("答案生成完成")
        return resp.content, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
