import pytest

pytest.importorskip("langchain_openai")

from services.vector_store import VectorStoreService


class _Result:
    rowcount = 3


class _Connection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params):
        self.calls.append((str(statement), params))
        return _Result()


class _Begin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *args):
        return False


class _Engine:
    def __init__(self):
        self.connection = _Connection()

    def begin(self):
        return _Begin(self.connection)


@pytest.mark.asyncio
async def test_pgvector_delete_by_doc_id_is_scoped_and_returns_count():
    service = VectorStoreService.__new__(VectorStoreService)
    service._backend = "pgvector"
    service._store = type("Store", (), {"_bind": _Engine(), "collection_name": "knowledge_chunks"})()

    deleted = await service.delete_by_doc_id("doc-123")

    assert deleted == 3
    call = service._store._bind.connection.calls[0]
    assert "langchain_pg_embedding" in call[0]
    assert call[1] == {"collection_name": "knowledge_chunks", "doc_id": "doc-123"}
