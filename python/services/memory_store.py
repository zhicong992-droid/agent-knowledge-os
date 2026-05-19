from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, select, text
from sqlalchemy.engine import Engine

from config import settings


@dataclass
class MemoryRecord:
    memory_id: str
    scope: str
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    ttl_seconds: int = 86400
    created_at: float = field(default_factory=time.time)

    @property
    def expired_at(self) -> float:
        return self.created_at + self.ttl_seconds


class MemoryStoreService:
    def __init__(self) -> None:
        self.engine: Engine | None = None
        self.metadata = MetaData(schema=settings.memory_schema)
        self.table = Table(
            "memories",
            self.metadata,
            Column("memory_id", String(128), primary_key=True),
            Column("scope", String(128), index=True, nullable=False),
            Column("kind", String(128), index=True, nullable=False),
            Column("content", Text, nullable=False),
            Column("metadata", Text, nullable=False, default="{}"),
            Column("version", Integer, nullable=False, default=1),
            Column("ttl_seconds", Integer, nullable=False, default=86400),
            Column("created_at", Float, nullable=False),
            Column("expired_at", Float, nullable=False, index=True),
        )

    async def init(self) -> None:
        self.engine = create_engine(settings.memory_dsn, future=True)
        with self.engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.memory_schema}"'))
        self.metadata.create_all(self.engine)

    async def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def _ensure(self) -> Engine:
        if self.engine is None:
            self.engine = create_engine(settings.memory_dsn, future=True)
            with self.engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.memory_schema}"'))
            self.metadata.create_all(self.engine)
        return self.engine

    async def remember(
        self,
        scope: str,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        version: int = 1,
        ttl_seconds: int = 86400,
    ) -> MemoryRecord:
        engine = self._ensure()
        now = time.time()
        record = MemoryRecord(
            memory_id=f"{scope}:{kind}:{int(now * 1000)}",
            scope=scope,
            kind=kind,
            content=content,
            metadata=metadata or {},
            version=version,
            ttl_seconds=ttl_seconds,
            created_at=now,
        )
        row = {
            "memory_id": record.memory_id,
            "scope": record.scope,
            "kind": record.kind,
            "content": record.content,
            "metadata": json.dumps(record.metadata, ensure_ascii=False),
            "version": record.version,
            "ttl_seconds": record.ttl_seconds,
            "created_at": record.created_at,
            "expired_at": record.expired_at,
        }
        with engine.begin() as conn:
            conn.execute(self.table.insert().values(**row))
        return record

    async def recall(
        self,
        scope: str,
        query: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        engine = self._ensure()
        now = time.time()
        stmt = select(self.table).where(self.table.c.expired_at > now)
        if scope != "*":
            stmt = stmt.where(self.table.c.scope == scope)
        if query:
            like = f"%{query}%"
            stmt = stmt.where((self.table.c.content.ilike(like)) | (self.table.c.metadata.ilike(like)))
        stmt = stmt.order_by(self.table.c.version.desc(), self.table.c.created_at.desc()).limit(limit)
        with engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        records: list[MemoryRecord] = []
        for row in rows:
            records.append(
                MemoryRecord(
                    memory_id=row["memory_id"],
                    scope=row["scope"],
                    kind=row["kind"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    version=row["version"],
                    ttl_seconds=row["ttl_seconds"],
                    created_at=row["created_at"],
                )
            )
        return records

    async def gc(self) -> int:
        engine = self._ensure()
        now = time.time()
        with engine.begin() as conn:
            result = conn.execute(delete(self.table).where(self.table.c.expired_at <= now))
            return result.rowcount or 0
