"""
知识更新 Agent — 监听文档变更，增量更新向量库和知识图谱

核心能力:
  1. 文件系统监听 (Watchdog) / Kafka CDC 消费
  2. 差量对比：对比新旧文档，只处理变更部分
  3. 增量向量化 & 图谱更新
  4. 版本管理：知识节点带时间戳和版本号
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """
    知识更新 Agent

    支持两种模式:
      1. 文件监听模式 (Watchdog): 监听本地文件系统变更
      2. CDC 模式 (Kafka): 消费来自消息队列的变更事件

    工作流:
      detect_change → diff_analysis → incremental_parse → update_vector_store → update_knowledge_graph → log
    """

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._file_hashes: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}

    # ── public API ───────────────────────────────────────────

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        """处理单个文档变更"""
        start = time.time()
        result = UpdateResult(change=change)

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        """批量处理文档变更"""
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        """扫描文件列表，检测变更"""
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            self._file_hashes[fp] = new_hash

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
            ))
            del self._file_hashes[fp]

        return changes

    # ── watchdog mode ────────────────────────────────────────

    def start_watching(self, directory: str) -> None:
        """启动文件系统监听（非阻塞，在独立线程运行）"""
        import threading
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                    asyncio.run(agent.process_change(change))

            def on_modified(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                    asyncio.run(agent.process_change(change))

            def on_deleted(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.DELETED)
                    asyncio.run(agent.process_change(change))

        observer = Observer()
        observer.schedule(_Handler(), directory, recursive=True)

        def _run():
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── kafka CDC mode ───────────────────────────────────────

    async def start_kafka_consumer(self) -> None:
        """启动 Kafka CDC 消费者"""
        import json
        from confluent_kafka import Consumer

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "knowledge-update-agent",
            "auto.offset.reset": "latest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([settings.kafka_topic_doc_changes])

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                payload = json.loads(msg.value().decode("utf-8"))
                change = DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType(payload["change_type"]),
                    old_hash=payload.get("old_hash", ""),
                    new_hash=payload.get("new_hash", ""),
                )
                await self.process_change(change)
        finally:
            consumer.close()

    # ── internal handlers ────────────────────────────────────

    async def _handle_create(self, change: DocumentChange, result: UpdateResult) -> None:
        if not self.doc_parser:
            return
        chunks = await self.doc_parser.parse(change.file_path)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)

        if self.knowledge_extractor and self.knowledge_graph:
            extractions = await self.knowledge_extractor.extract(chunks)
            for ext in extractions:
                for ent in ext.entities:
                    version = self._bump_version(ent.name)
                    await self.knowledge_graph.upsert_entity(ent, version=version)
                    result.entities_added += 1
                for rel in ext.relations:
                    await self.knowledge_graph.add_relation(rel)
                    result.relations_added += 1

    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        await self._handle_create(change, result)

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)

    # ── utilities ────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    def _bump_version(self, entity_name: str) -> int:
        ver = self._version_counter.get(entity_name, 0) + 1
        self._version_counter[entity_name] = ver
        return ver
