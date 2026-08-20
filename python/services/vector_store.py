"""
向量存储服务 — 支持 ChromaDB / PGVector 双后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""

from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocumentChunk
from config import settings


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._store: Any = None
        self._backend = settings.vector_store_type

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._store = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并存储文档块"""
        if not chunks:
            return 0
        if self._store is None:
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"doc_id": c.doc_id, "doc_type": c.doc_type.value, "source": c.metadata.get("source", ""), "chunk_index": c.chunk_index}
            for c in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表"""
        if self._store is None:
            return []
        if self._backend == "chroma":
            try:
                q_vec = await self.embeddings.aembed_query(query)
                results = self._store.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
                out: list[tuple[dict, float]] = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    score = 1.0 - dist  # cosine distance → similarity
                    out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
                return out
            except Exception:
                return []
        else:
            try:
                results = await self._store.asimilarity_search_with_score(query, k=top_k)
                return [
                    ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
                    for doc, score in results
                ]
            except Exception:
                return []

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if self._store is None:
            return 0
        if self._backend == "chroma":
            existing = self._store.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)

        # PGVector stores metadata as JSONB.  Use the store's SQLAlchemy
        # binding when available so deletion remains scoped to this collection.
        try:
            from sqlalchemy import text

            engine = getattr(self._store, "_bind", None) or getattr(self._store, "_engine", None)
            if engine is None:
                connection_string = getattr(self._store, "connection_string", None) or settings.pgvector_dsn
                from sqlalchemy import create_engine
                engine = create_engine(connection_string)

            collection_name = getattr(self._store, "collection_name", self.COLLECTION_NAME)
            with engine.begin() as connection:
                result = connection.execute(
                    text(
                        """
                        DELETE FROM langchain_pg_embedding AS e
                        USING langchain_pg_collection AS c
                        WHERE e.collection_id = c.uuid
                          AND c.name = :collection_name
                          AND e.cmetadata ->> 'doc_id' = :doc_id
                        """
                    ),
                    {"collection_name": collection_name, "doc_id": doc_id},
                )
                return int(result.rowcount or 0)
        except Exception:
            return 0

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if self._store is None:
            return {"backend": self._backend, "total_vectors": 0, "status": "unavailable"}
        if self._backend == "chroma":
            count = self._store.count()
            return {"backend": "chroma", "total_vectors": count, "collection": self.COLLECTION_NAME}
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
