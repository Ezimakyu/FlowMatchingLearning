from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from backend.app.models import ChunkingResult, EmbeddingBatchResult


def _slugify(value: str, *, max_len: int = 32) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "default"
    return cleaned[:max_len]


def chunk_id_to_point_id(chunk_id: str) -> int:
    digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    point_id = int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
    return point_id or 1


@dataclass(frozen=True)
class ActianCortexConfig:
    server_address: str = "localhost:50051"
    collection_prefix: str = "course_chunks"
    distance_metric: str = "COSINE"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 50

    @classmethod
    def from_env(cls) -> "ActianCortexConfig":
        import os

        return cls(
            server_address=os.getenv("ACTIAN_VECTORAI_ADDR", "localhost:50051"),
            collection_prefix=os.getenv("ACTIAN_COLLECTION_PREFIX", "course_chunks"),
            distance_metric=os.getenv("ACTIAN_DISTANCE_METRIC", "COSINE"),
            hnsw_m=int(os.getenv("ACTIAN_HNSW_M", "16")),
            hnsw_ef_construct=int(os.getenv("ACTIAN_HNSW_EF_CONSTRUCT", "200")),
            hnsw_ef_search=int(os.getenv("ACTIAN_HNSW_EF_SEARCH", "50")),
        )


class ActianCortexStore:
    def __init__(self, config: ActianCortexConfig) -> None:
        self.config = config

    def _build_collection_name(self, *, model_name: str, vector_dim: int) -> str:
        prefix = _slugify(self.config.collection_prefix, max_len=24)
        model_slug = _slugify(model_name, max_len=24)
        return f"{prefix}_{model_slug}_{vector_dim}"

    def _get_client_class(self):
        try:
            from cortex import CortexClient
        except ImportError as exc:
            raise RuntimeError(
                "cortex client is required for ActianCortexStore. "
                "Install the Actian wheel: pip install ./actiancortex-0.1.0b1-py3-none-any.whl"
            ) from exc
        return CortexClient

    def ensure_schema(self) -> None:
        # Cortex collections are created lazily per model/dimension, so schema setup is a health check.
        logger = logging.getLogger(__name__)
        logger.info("actian.ensure_schema_start addr=%s", self.config.server_address)
        CortexClient = self._get_client_class()
        with CortexClient(self.config.server_address) as client:
            client.health_check()
        logger.info("actian.ensure_schema_finish addr=%s", self.config.server_address)

    def _collection_exists(self, client, collection_name: str) -> bool:
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(collection_name))
        if hasattr(client, "has_collection"):
            return bool(client.has_collection(collection_name))
        try:
            client.open_collection(collection_name)
            if hasattr(client, "close_collection"):
                client.close_collection(collection_name)
            return True
        except Exception:
            return False

    def _ensure_collection(
        self,
        *,
        client,
        collection_name: str,
        vector_dim: int,
    ) -> None:
        if hasattr(client, "get_or_create_collection"):
            try:
                client.get_or_create_collection(
                    name=collection_name,
                    dimension=vector_dim,
                    distance_metric=self.config.distance_metric,
                )
                return
            except TypeError:
                # Fallback for client versions where kwargs are not accepted.
                client.get_or_create_collection(
                    collection_name,
                    vector_dim,
                    self.config.distance_metric,
                )
                return

        if self._collection_exists(client, collection_name):
            return

        create_kwargs = {
            "name": collection_name,
            "dimension": vector_dim,
            "distance_metric": self.config.distance_metric,
            "hnsw_m": self.config.hnsw_m,
            "hnsw_ef_construct": self.config.hnsw_ef_construct,
            "hnsw_ef_search": self.config.hnsw_ef_search,
        }
        try:
            client.create_collection(**create_kwargs)
        except TypeError:
            # Fallback for positional-only signatures in early beta SDKs.
            client.create_collection(
                collection_name,
                vector_dim,
                self.config.distance_metric,
            )

    def upsert_chunks_and_embeddings(
        self,
        *,
        chunking: ChunkingResult,
        embeddings: EmbeddingBatchResult,
    ) -> tuple[int, int]:
        logger = logging.getLogger(__name__)
        if chunking.doc_id != embeddings.doc_id:
            raise ValueError(
                f"doc_id mismatch: chunking={chunking.doc_id}, embeddings={embeddings.doc_id}"
            )

        embedding_by_chunk = {item.chunk_id: item for item in embeddings.embeddings}
        missing_embeddings = [
            chunk.chunk_id for chunk in chunking.chunks if chunk.chunk_id not in embedding_by_chunk
        ]
        if missing_embeddings:
            raise ValueError(
                "Embeddings are missing for chunk ids: " + ", ".join(sorted(missing_embeddings))
            )

        if not chunking.chunks:
            return 0, 0

        vector_dims = {embedding.vector_dim for embedding in embeddings.embeddings}
        if len(vector_dims) != 1:
            raise ValueError("All embeddings in a batch must share the same vector dimension.")
        vector_dim = next(iter(vector_dims))
        collection_name = self._build_collection_name(
            model_name=embeddings.model_name,
            vector_dim=vector_dim,
        )
        logger.info(
            "actian.upsert_start addr=%s collection=%s chunks=%d embeddings=%d",
            self.config.server_address,
            collection_name,
            len(chunking.chunks),
            len(embeddings.embeddings),
        )

        point_ids: list[int] = []
        vectors: list[list[float]] = []
        payloads: list[dict[str, Any]] = []

        for chunk in chunking.chunks:
            embedding = embedding_by_chunk[chunk.chunk_id]
            point_ids.append(chunk_id_to_point_id(chunk.chunk_id))
            vectors.append(embedding.vector)
            payloads.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "model_name": embedding.model_name,
                    "chunk_order": chunk.order,
                    "source_type": chunk.source_type,
                    "section_hint": chunk.section_hint,
                    "source_page": chunk.source_page,
                    "source_time_start_seconds": chunk.source_time_start_seconds,
                    "source_time_end_seconds": chunk.source_time_end_seconds,
                    "token_estimate": chunk.token_estimate,
                    "text": chunk.text,
                    "chunk_metadata": chunk.metadata,
                    "embedding_metadata": embeddings.metadata,
                }
            )

        CortexClient = self._get_client_class()
        with CortexClient(self.config.server_address) as client:
            self._ensure_collection(
                client=client,
                collection_name=collection_name,
                vector_dim=vector_dim,
            )
            client.batch_upsert(
                collection_name=collection_name,
                ids=point_ids,
                vectors=vectors,
                payloads=payloads,
            )
            if hasattr(client, "flush"):
                client.flush(collection_name)

        logger.info(
            "actian.upsert_finish addr=%s collection=%s",
            self.config.server_address,
            collection_name,
        )
        return len(chunking.chunks), len(embeddings.embeddings)

    def similarity_search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        model_name: str = "BAAI/bge-m3",
        candidate_limit: int = 0,
    ) -> list[dict[str, Any]]:
        logger = logging.getLogger(__name__)
        if not query_vector:
            raise ValueError("query_vector must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # candidate_limit is intentionally unused in Cortex mode because search API is top_k-based.
        _ = candidate_limit

        collection_name = self._build_collection_name(
            model_name=model_name,
            vector_dim=len(query_vector),
        )
        logger.info(
            "actian.search_start addr=%s collection=%s top_k=%d min_similarity=%.3f",
            self.config.server_address,
            collection_name,
            top_k,
            min_similarity,
        )
        CortexClient = self._get_client_class()
        with CortexClient(self.config.server_address) as client:
            try:
                results = client.search(
                    collection_name,
                    query_vector,
                    top_k=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            except TypeError:
                # Fallback for early beta signatures.
                results = client.search(collection_name, query_vector, top_k=top_k)

        ranked: list[dict[str, Any]] = []
        for result in results:
            score = float(getattr(result, "score", 0.0))
            if score < min_similarity:
                continue
            payload = getattr(result, "payload", None) or {}
            ranked.append(
                {
                    "point_id": getattr(result, "id", None),
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "model_name": payload.get("model_name"),
                    "raw_text": payload.get("text", ""),
                    "similarity": score,
                    "payload": payload,
                }
            )
        logger.info(
            "actian.search_finish addr=%s collection=%s results=%d",
            self.config.server_address,
            collection_name,
            len(ranked),
        )
        return ranked


__all__ = [
    "ActianCortexConfig",
    "ActianCortexStore",
    "chunk_id_to_point_id",
]
