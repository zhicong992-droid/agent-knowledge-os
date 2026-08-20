import pytest

pytest.importorskip("langchain_openai")

from services.vector_store import VectorStoreService


class _Collection:
    def __init__(self):
        self.deleted = None

    def get(self, where, include):
        assert where == {"doc_id": "doc-123"}
        return {"ids": ["chunk-1", "chunk-2"]}

    def delete(self, ids):
        self.deleted = ids


@pytest.mark.asyncio
async def test_chroma_delete_by_doc_id_removes_all_matching_chunks():
    service = VectorStoreService.__new__(VectorStoreService)
    service._backend = "chroma"
    service._store = _Collection()

    deleted = await service.delete_by_doc_id("doc-123")

    assert deleted == 2
    assert service._store.deleted == ["chunk-1", "chunk-2"]
