from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.ingestion.chunking import (
    build_chunking_result,
    chunk_transcription_result,
    chunk_vision_extraction_result,
)
from backend.app.models import (
    ChunkingResult,
    EmbeddingBatchResult,
    PhaseAIngestionResult,
    RawTextChunk,
    TranscriptionResult,
    VisionExtractionResult,
)


class ModalIngestionClient(Protocol):
    def transcribe_media(
        self,
        *,
        media_bytes: bytes,
        doc_id: str,
        media_id: str,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def extract_vision(
        self,
        *,
        file_bytes: bytes,
        doc_id: str,
        source_file_id: str,
    ) -> VisionExtractionResult:
        raise NotImplementedError

    def embed_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[RawTextChunk],
    ) -> EmbeddingBatchResult:
        raise NotImplementedError


class ActianStorageClient(Protocol):
    def ensure_schema(self) -> None:
        raise NotImplementedError

    def upsert_chunks_and_embeddings(
        self,
        *,
        chunking: ChunkingResult,
        embeddings: EmbeddingBatchResult,
    ) -> tuple[int, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class PhaseAIngestionConfig:
    max_vision_chunk_tokens: int = 260
    max_transcript_chunk_tokens: int = 220
    include_transcript_chunks: bool = True


class PhaseAIngestionPipeline:
    def __init__(
        self,
        *,
        ingestion_client: ModalIngestionClient,
        storage_client: ActianStorageClient,
        config: PhaseAIngestionConfig | None = None,
        ingestion_provider: str = "modal",
        storage_provider: str = "actian",
    ) -> None:
        self.ingestion_client = ingestion_client
        self.storage_client = storage_client
        self.config = config or PhaseAIngestionConfig()
        self.ingestion_provider = ingestion_provider
        self.storage_provider = storage_provider

    def run(
        self,
        *,
        doc_id: str,
        source_file_id: str,
        source_file_bytes: bytes,
        media_id: str | None = None,
        media_bytes: bytes | None = None,
    ) -> PhaseAIngestionResult:
        assert_compute_boundary("ingestion.vision_extraction", self.ingestion_provider)
        vision_result = self.ingestion_client.extract_vision(
            file_bytes=source_file_bytes,
            doc_id=doc_id,
            source_file_id=source_file_id,
        )

        transcription_result: TranscriptionResult | None = None
        if media_bytes is not None and media_id is not None:
            assert_compute_boundary("ingestion.transcription", self.ingestion_provider)
            transcription_result = self.ingestion_client.transcribe_media(
                media_bytes=media_bytes,
                doc_id=doc_id,
                media_id=media_id,
            )

        chunks = self._build_chunks(
            doc_id=doc_id,
            vision_result=vision_result,
            transcription_result=transcription_result,
        )
        chunking = build_chunking_result(doc_id=doc_id, chunks=chunks)

        assert_compute_boundary("ingestion.embedding", self.ingestion_provider)
        embeddings = self.ingestion_client.embed_chunks(doc_id=doc_id, chunks=chunking.chunks)

        assert_compute_boundary("storage.write_chunks_and_vectors", self.storage_provider)
        self.storage_client.ensure_schema()
        stored_chunk_count, stored_embedding_count = self.storage_client.upsert_chunks_and_embeddings(
            chunking=chunking,
            embeddings=embeddings,
        )

        return PhaseAIngestionResult(
            doc_id=doc_id,
            chunking=chunking,
            embeddings=embeddings,
            stored_chunk_count=stored_chunk_count,
            stored_embedding_count=stored_embedding_count,
            metadata={
                "source_file_id": source_file_id,
                "media_id": media_id,
                "ingestion_provider": self.ingestion_provider,
                "storage_provider": self.storage_provider,
            },
        )

    def _build_chunks(
        self,
        *,
        doc_id: str,
        vision_result: VisionExtractionResult,
        transcription_result: TranscriptionResult | None = None,
    ) -> list[RawTextChunk]:
        all_chunks: list[RawTextChunk] = []
        vision_chunks = chunk_vision_extraction_result(
            vision_result,
            max_tokens=self.config.max_vision_chunk_tokens,
            start_order=0,
        )
        all_chunks.extend(vision_chunks)

        if self.config.include_transcript_chunks and transcription_result is not None:
            transcript_chunks = chunk_transcription_result(
                transcription_result,
                max_tokens=self.config.max_transcript_chunk_tokens,
                start_order=len(all_chunks),
            )
            all_chunks.extend(transcript_chunks)

        # Ensure the output ordering is contiguous for deterministic storage.
        normalized_chunks: list[RawTextChunk] = []
        for order, chunk in enumerate(all_chunks):
            normalized_chunks.append(
                RawTextChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=doc_id,
                    source_type=chunk.source_type,
                    order=order,
                    text=chunk.text,
                    token_estimate=chunk.token_estimate,
                    section_hint=chunk.section_hint,
                    source_page=chunk.source_page,
                    source_time_start_seconds=chunk.source_time_start_seconds,
                    source_time_end_seconds=chunk.source_time_end_seconds,
                    metadata=chunk.metadata,
                )
            )

        return normalized_chunks


__all__ = [
    "ActianStorageClient",
    "ModalIngestionClient",
    "PhaseAIngestionConfig",
    "PhaseAIngestionPipeline",
]
