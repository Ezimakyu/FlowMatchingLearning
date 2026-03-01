from __future__ import annotations

from dataclasses import dataclass
import logging
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


@dataclass(frozen=True)
class PhaseAIngestionInput:
    source_file_id: str
    source_file_bytes: bytes
    media_id: str | None = None
    media_bytes: bytes | None = None


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
        return self.run_batch(
            doc_id=doc_id,
            inputs=[
                PhaseAIngestionInput(
                    source_file_id=source_file_id,
                    source_file_bytes=source_file_bytes,
                    media_id=media_id,
                    media_bytes=media_bytes,
                )
            ],
        )

    def run_batch(
        self,
        *,
        doc_id: str,
        inputs: list[PhaseAIngestionInput],
    ) -> PhaseAIngestionResult:
        logger = logging.getLogger(__name__)
        if not inputs:
            raise ValueError("inputs must contain at least one source file.")

        logger.info(
            "phase_a.pipeline_start doc_id=%s inputs=%d",
            doc_id,
            len(inputs),
        )
        all_chunks: list[RawTextChunk] = []
        source_file_ids: list[str] = []
        media_ids: list[str] = []

        for input_index, ingestion_input in enumerate(inputs):
            if (ingestion_input.media_id is None) != (ingestion_input.media_bytes is None):
                raise ValueError(
                    "media_id and media_bytes must either both be set or both be omitted "
                    f"(source_file_id={ingestion_input.source_file_id})."
                )

            logger.info(
                "phase_a.input_start doc_id=%s input_index=%d source_file_id=%s has_media=%s",
                doc_id,
                input_index,
                ingestion_input.source_file_id,
                ingestion_input.media_id is not None,
            )
            assert_compute_boundary("ingestion.vision_extraction", self.ingestion_provider)
            vision_result = self.ingestion_client.extract_vision(
                file_bytes=ingestion_input.source_file_bytes,
                doc_id=doc_id,
                source_file_id=ingestion_input.source_file_id,
            )
            logger.info(
                "phase_a.vision_complete doc_id=%s source_file_id=%s pages=%d",
                doc_id,
                ingestion_input.source_file_id,
                len(vision_result.pages),
            )
            source_file_ids.append(ingestion_input.source_file_id)

            transcription_result: TranscriptionResult | None = None
            if ingestion_input.media_bytes is not None and ingestion_input.media_id is not None:
                assert_compute_boundary("ingestion.transcription", self.ingestion_provider)
                transcription_result = self.ingestion_client.transcribe_media(
                    media_bytes=ingestion_input.media_bytes,
                    doc_id=doc_id,
                    media_id=ingestion_input.media_id,
                )
                media_ids.append(ingestion_input.media_id)
                logger.info(
                    "phase_a.transcription_complete doc_id=%s media_id=%s segments=%d",
                    doc_id,
                    ingestion_input.media_id,
                    len(transcription_result.segments),
                )

            input_chunks = self._build_chunks(
                doc_id=doc_id,
                vision_result=vision_result,
                transcription_result=transcription_result,
                start_order=len(all_chunks),
            )
            all_chunks.extend(
                self._attach_chunk_metadata(
                    chunks=input_chunks,
                    source_file_id=ingestion_input.source_file_id,
                    media_id=ingestion_input.media_id,
                    input_index=input_index,
                )
            )
            logger.info(
                "phase_a.input_complete doc_id=%s input_index=%d chunks_added=%d total_chunks=%d",
                doc_id,
                input_index,
                len(input_chunks),
                len(all_chunks),
            )

        chunking = build_chunking_result(doc_id=doc_id, chunks=all_chunks)
        logger.info(
            "phase_a.chunking_complete doc_id=%s chunks=%d",
            doc_id,
            len(chunking.chunks),
        )

        assert_compute_boundary("ingestion.embedding", self.ingestion_provider)
        embeddings = self.ingestion_client.embed_chunks(doc_id=doc_id, chunks=chunking.chunks)
        logger.info(
            "phase_a.embedding_complete doc_id=%s embeddings=%d model=%s",
            doc_id,
            len(embeddings.embeddings),
            embeddings.model_name,
        )

        assert_compute_boundary("storage.write_chunks_and_vectors", self.storage_provider)
        logger.info("phase_a.actian_start doc_id=%s", doc_id)
        self.storage_client.ensure_schema()
        stored_chunk_count, stored_embedding_count = self.storage_client.upsert_chunks_and_embeddings(
            chunking=chunking,
            embeddings=embeddings,
        )
        logger.info(
            "phase_a.actian_complete doc_id=%s stored_chunks=%d stored_embeddings=%d",
            doc_id,
            stored_chunk_count,
            stored_embedding_count,
        )

        output = PhaseAIngestionResult(
            doc_id=doc_id,
            chunking=chunking,
            embeddings=embeddings,
            stored_chunk_count=stored_chunk_count,
            stored_embedding_count=stored_embedding_count,
            metadata=self._build_output_metadata(
                source_file_ids=source_file_ids,
                media_ids=media_ids,
            ),
        )
        logger.info("phase_a.pipeline_finish doc_id=%s", doc_id)
        return output

    def _build_output_metadata(
        self,
        *,
        source_file_ids: list[str],
        media_ids: list[str],
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "source_file_ids": source_file_ids,
            "media_ids": media_ids,
            "input_count": len(source_file_ids),
            "inputs_with_media_count": len(media_ids),
            "ingestion_provider": self.ingestion_provider,
            "storage_provider": self.storage_provider,
        }
        if len(source_file_ids) == 1:
            metadata["source_file_id"] = source_file_ids[0]
            metadata["media_id"] = media_ids[0] if media_ids else None
        elif len(media_ids) == 1:
            metadata["media_id"] = media_ids[0]
        return metadata

    def _attach_chunk_metadata(
        self,
        *,
        chunks: list[RawTextChunk],
        source_file_id: str,
        media_id: str | None,
        input_index: int,
    ) -> list[RawTextChunk]:
        output: list[RawTextChunk] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["source_file_id"] = source_file_id
            metadata["input_index"] = input_index
            if media_id is not None and chunk.source_type == "transcript":
                metadata["media_id"] = media_id
            output.append(
                RawTextChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    order=chunk.order,
                    text=chunk.text,
                    token_estimate=chunk.token_estimate,
                    section_hint=chunk.section_hint,
                    source_page=chunk.source_page,
                    source_time_start_seconds=chunk.source_time_start_seconds,
                    source_time_end_seconds=chunk.source_time_end_seconds,
                    metadata=metadata,
                )
            )
        return output

    def _build_chunks(
        self,
        *,
        doc_id: str,
        vision_result: VisionExtractionResult,
        transcription_result: TranscriptionResult | None = None,
        start_order: int = 0,
    ) -> list[RawTextChunk]:
        all_chunks: list[RawTextChunk] = []
        vision_chunks = chunk_vision_extraction_result(
            vision_result,
            max_tokens=self.config.max_vision_chunk_tokens,
            start_order=start_order,
        )
        all_chunks.extend(vision_chunks)

        if self.config.include_transcript_chunks and transcription_result is not None:
            transcript_chunks = chunk_transcription_result(
                transcription_result,
                max_tokens=self.config.max_transcript_chunk_tokens,
                start_order=start_order + len(all_chunks),
            )
            all_chunks.extend(transcript_chunks)

        # Ensure output ordering remains contiguous and anchored to start_order.
        normalized_chunks: list[RawTextChunk] = []
        for order, chunk in enumerate(all_chunks, start=start_order):
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
    "PhaseAIngestionInput",
    "PhaseAIngestionPipeline",
]
