"""
知识图谱服务 — Neo4j 图数据库操作

职责:
  1. 实体 (Node) CRUD — 带版本号和时间戳
  2. 关系 (Relationship) CRUD
  3. Cypher 查询执行
  4. 子图检索（多跳遍历）
  5. 按来源删除（支持增量更新）
"""

from __future__ import annotations

import time
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings


class KnowledgeGraphService:
    """Neo4j 知识图谱服务"""

    def __init__(self) -> None:
        self._driver: Any = None

    # ── lifecycle ────────────────────────────────────────────

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        """创建常用索引以加速查询"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                await session.run(q)

    # ── entity operations ────────────────────────────────────

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        """
        创建或更新实体节点 — MERGE 语义
        带版本号和时间戳，支持增量更新追踪
        """
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.version = $version,
            e.source = $source,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "version": version,
                "source": source,
                "now": int(time.time()),
            })

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        """创建实体间关系"""
        rel_type = relation.relation.upper().replace(" ", "_")
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head,
                "tail": relation.tail,
                "confidence": relation.confidence,
                "source": source,
                "now": int(time.time()),
            })

    # ── query operations ─────────────────────────────────────

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """执行任意 Cypher 查询"""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def get_entity(self, name: str) -> dict | None:
        """查询单个实体"""
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """
        多跳子图检索 — GraphRAG 的核心能力
        从指定实体出发，遍历 N 跳内的所有关联实体和关系
        """
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
        RETURN
            start.name AS source,
            [r IN relationships(path) | type(r)] AS relations,
            neighbor.name AS target,
            neighbor.type AS target_type,
            neighbor.description AS target_desc
        LIMIT 50
        """
        return await self.execute_cypher(cypher, {"name": entity_name})

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        """模糊搜索实体"""
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    # ── delete operations ────────────────────────────────────

    async def delete_by_source(self, source: str) -> int:
        """按来源删除所有实体和关系（增量更新时先删再建）"""
        cypher = """
        MATCH (e:Entity {source: $source})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        records = await self.execute_cypher(cypher, {"source": source})
        return records[0].get("deleted", 0) if records else 0

    # ── stats ────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """获取图谱统计信息"""
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }
