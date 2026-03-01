from __future__ import annotations

from functools import lru_cache

import modal

from backend.app.models import ChunkEmbedding, EmbeddingBatchResult, RawTextChunk

app = modal.App("phase-a-embedding")

embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "sentence-transformers==5.1.0",
        "torch==2.8.0",
    )
)


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cuda")


@app.function(
    image=embedding_image,
    gpu="L4",
    timeout=60 * 20,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=2.0),
)
def embed_chunks(
    *,
    doc_id: str,
    chunks: list[dict],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 128,
) -> dict:
    model = _load_embedding_model(model_name)
    validated_chunks = [RawTextChunk.model_validate(chunk) for chunk in chunks]
    text_batch = [chunk.text for chunk in validated_chunks]
    vectors = model.encode(
        text_batch,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings: list[ChunkEmbedding] = []
    for chunk, vector in zip(validated_chunks, vectors):
        vector_list = [float(value) for value in vector.tolist()]
        embeddings.append(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                vector=vector_list,
                vector_dim=len(vector_list),
                model_name=model_name,
            )
        )

    result = EmbeddingBatchResult(
        doc_id=doc_id,
        model_name=model_name,
        embeddings=embeddings,
        metadata={"chunk_count": len(validated_chunks)},
    )
    return result.model_dump(mode="json")


@app.local_entrypoint()
def main(chunks_json: str, doc_id: str) -> None:
    import json

    chunks = json.loads(chunks_json)
    payload = embed_chunks.remote(doc_id=doc_id, chunks=chunks)
    print(payload)
