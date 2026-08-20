import asyncio
import pytest

pytest.importorskip("langchain_core")

from services.graph_rag import GraphRAGPipeline


@pytest.mark.asyncio
async def test_retrieve_runs_independent_branches_concurrently():
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    active = 0
    peak = 0

    async def mark(name, result):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return result

    pipeline._vector_search = lambda query, top_k: mark("vector", [])
    pipeline._entity_linking = lambda query: mark("entities", ["A", "B"])
    pipeline._subgraph_search = lambda entities: mark("subgraph", [])
    pipeline._path_search = lambda entities: mark("path", [])

    result = await pipeline.retrieve("question")

    assert result == []
    assert peak >= 2
