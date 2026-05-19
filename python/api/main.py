"""
FastAPI 入口 — 企业知识管理系统 REST API

提供三组接口:
  1. /api/ingest   — 文档上传 & 入库
  2. /api/qa       — 智能问答
  3. /api/admin    — 管理（统计、更新触发）
"""

from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.memory_store import MemoryStoreService
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
memory_store = MemoryStoreService()
workflows: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    try:
        await vector_store.init()
    except Exception:
        pass
    try:
        await knowledge_graph.init()
    except Exception:
        pass
    await memory_store.init()
    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    yield
    await knowledge_graph.close()
    await memory_store.close()


app = FastAPI(
    title="AgentKnowledgeOS — 多Agent企业知识操作系统",
    description="支持多模态RAG、知识图谱、长期记忆、增量更新的企业级知识 API",
    version="2.0.0",
    contact={"name": "zhicong992-droid"},
    lifespan=lifespan,
)


# ── Request / Response Models ────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]
    memory: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


class MemoryRequest(BaseModel):
    scope: str
    kind: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    ttl_seconds: int | None = None


class MemoryResponse(BaseModel):
    memory_id: str
    scope: str
    kind: str
    content: str
    metadata: dict[str, Any]
    version: int



# ── Ingest Endpoints ─────────────────────────────────────────

@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档，自动入库到向量库和知识图谱"""
    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    return IngestResponse(
        file_name=file.filename or "unknown",
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


# ── QA Endpoints ─────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + 知识图谱推理"""
    if not workflows:
        os.makedirs(settings.upload_dir, exist_ok=True)
        try:
            await vector_store.init()
        except Exception:
            pass
        try:
            await knowledge_graph.init()
        except Exception:
            pass
        await memory_store.init()
        workflows.update(
            build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
        )

    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    wf_out = await qa_wf.ainvoke({"question": req.question})
    result = wf_out.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="QA workflow returned empty result")
    return QuestionResponse(
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        intent=result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in result.contexts
        ],
        reasoning_steps=result.reasoning_steps,
    )


# ── Admin Endpoints ──────────────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    mem_stats = {"backend": "postgres", "total_memories": len(await memory_store.recall(scope="*"))}
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats, memory=mem_stats)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest):
    """手动触发知识更新"""
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.post("/api/admin/memory", response_model=MemoryResponse, tags=["系统管理"])
async def remember(req: MemoryRequest):
    record = await memory_store.remember(
        scope=req.scope,
        kind=req.kind,
        content=req.content,
        metadata=req.metadata,
        version=req.version,
        ttl_seconds=req.ttl_seconds or settings.memory_ttl_seconds,
    )
    return MemoryResponse(
        memory_id=record.memory_id,
        scope=record.scope,
        kind=record.kind,
        content=record.content,
        metadata=record.metadata,
        version=record.version,
    )


@app.get("/api/admin/memory/{scope}", response_model=list[MemoryResponse], tags=["系统管理"])
async def recall(scope: str, query: str | None = None):
    records = await memory_store.recall(scope=scope, query=query, limit=20)
    return [
        MemoryResponse(
            memory_id=r.memory_id,
            scope=r.scope,
            kind=r.kind,
            content=r.content,
            metadata=r.metadata,
            version=r.version,
        )
        for r in records
    ]


@app.post("/api/admin/memory/gc", tags=["系统管理"])
async def memory_gc():
    deleted = await memory_store.gc()
    return {"deleted": deleted}


@app.get("/api/health", tags=["系统管理"])
async def health():
    return {"status": "ok", "service": "AgentKnowledgeOS", "author": "zhicong992-droid", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
