from types import SimpleNamespace

import pytest

from backend.app.models import ChunkEmbedding, ChunkingResult, EmbeddingBatchResult, RawTextChunk
from backend.app.storage.actian_cortex_store import (
    ActianCortexConfig,
    ActianCortexStore,
    chunk_id_to_point_id,
)


class FakeCortexClient:
    created_collections: list[dict] = []
    upserts: list[dict] = []
    flush_calls: list[str] = []
    search_calls: list[dict] = []

    def __init__(self, address: str) -> None:
        self.address = address

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def health_check(self):
        return "fake-cortex", 1

    def collection_exists(self, name: str) -> bool:
        return False

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)

    def batch_upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def flush(self, collection_name: str) -> None:
        self.flush_calls.append(collection_name)

    def search(
        self,
        collection_name: str,
        query: list[float],
        top_k: int,
        with_payload: bool = False,
        with_vectors: bool = False,
    ):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "top_k": top_k,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        payload = (
            {
                "chunk_id": "doc1:vision_text:00000",
                "doc_id": "doc1",
                "model_name": "BAAI/bge-m3",
                "text": "Limits and derivatives.",
            }
            if with_payload
            else {}
        )
        return [
            SimpleNamespace(
                id=123,
                score=0.91,
                payload=payload,
            )
        ][:top_k]


def _build_sample_batch() -> tuple[ChunkingResult, EmbeddingBatchResult]:
    chunking = ChunkingResult(
        doc_id="doc1",
        chunks=[
            RawTextChunk(
                chunk_id="doc1:vision_text:00000",
                doc_id="doc1",
                source_type="vision_text",
                order=0,
                text="Limits and derivatives.",
                token_estimate=4,
            )
        ],
    )
    embeddings = EmbeddingBatchResult(
        doc_id="doc1",
        model_name="BAAI/bge-m3",
        embeddings=[
            ChunkEmbedding(
                chunk_id="doc1:vision_text:00000",
                vector=[0.1, 0.2, 0.3],
                vector_dim=3,
                model_name="BAAI/bge-m3",
            )
        ],
    )
    return chunking, embeddings


def test_chunk_id_to_point_id_is_deterministic() -> None:
    point_a = chunk_id_to_point_id("doc1:chunk:00001")
    point_b = chunk_id_to_point_id("doc1:chunk:00001")
    point_c = chunk_id_to_point_id("doc1:chunk:00002")
    assert point_a == point_b
    assert point_a > 0
    assert point_c > 0
    assert point_a != point_c


def test_upsert_rejects_doc_id_mismatch_before_network_calls() -> None:
    chunking, embeddings = _build_sample_batch()
    embeddings = EmbeddingBatchResult(
        doc_id="different_doc",
        model_name=embeddings.model_name,
        embeddings=embeddings.embeddings,
    )
    store = ActianCortexStore(config=ActianCortexConfig(server_address="localhost:50051"))
    with pytest.raises(ValueError):
        store.upsert_chunks_and_embeddings(chunking=chunking, embeddings=embeddings)


def test_upsert_and_similarity_search_with_fake_client(monkeypatch) -> None:
    FakeCortexClient.created_collections = []
    FakeCortexClient.upserts = []
    FakeCortexClient.flush_calls = []
    FakeCortexClient.search_calls = []

    chunking, embeddings = _build_sample_batch()
    store = ActianCortexStore(config=ActianCortexConfig(server_address="localhost:50051"))
    monkeypatch.setattr(store, "_get_client_class", lambda: FakeCortexClient)

    inserted_chunks, inserted_embeddings = store.upsert_chunks_and_embeddings(
        chunking=chunking,
        embeddings=embeddings,
    )
    assert inserted_chunks == 1
    assert inserted_embeddings == 1
    assert len(FakeCortexClient.created_collections) == 1
    assert len(FakeCortexClient.upserts) == 1

    results = store.similarity_search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=1,
        model_name="BAAI/bge-m3",
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1:vision_text:00000"
    assert results[0]["similarity"] == pytest.approx(0.91)
    assert len(FakeCortexClient.search_calls) == 1
    assert FakeCortexClient.search_calls[0]["with_payload"] is True
    assert FakeCortexClient.search_calls[0]["with_vectors"] is False
