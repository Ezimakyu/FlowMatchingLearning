import pytest

from backend.app.compute_boundaries import BoundaryViolationError
from backend.app.models import (
    ChunkEmbedding,
    EmbeddingBatchResult,
    RawTextChunk,
    TranscriptSegment,
    TranscriptionResult,
    VisionExtractionResult,
    VisionPageExtraction,
)
from backend.app.pipelines.phase_a import PhaseAIngestionPipeline


class FakeModalIngestionClient:
    def __init__(self) -> None:
        self.transcription_calls = 0
        self.vision_calls = 0
        self.embedding_calls = 0

    def transcribe_media(
        self,
        *,
        media_bytes: bytes,
        doc_id: str,
        media_id: str,
    ) -> TranscriptionResult:
        self.transcription_calls += 1
        return TranscriptionResult(
            doc_id=doc_id,
            media_id=media_id,
            model_name="fake-whisper",
            segments=[
                TranscriptSegment(
                    segment_id=f"{doc_id}:seg:00000",
                    start_seconds=0.0,
                    end_seconds=2.5,
                    text="Transcript text about prerequisites.",
                )
            ],
            transcript_text="Transcript text about prerequisites.",
        )

    def extract_vision(
        self,
        *,
        file_bytes: bytes,
        doc_id: str,
        source_file_id: str,
    ) -> VisionExtractionResult:
        self.vision_calls += 1
        return VisionExtractionResult(
            doc_id=doc_id,
            source_file_id=source_file_id,
            model_name="fake-vlm",
            pages=[
                VisionPageExtraction(
                    page_number=1,
                    raw_text="Limits lead to derivatives in introductory calculus.",
                    image_descriptions=["A graph of a function approaching a tangent line."],
                    chunk_ids=[],
                )
            ],
        )

    def embed_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[RawTextChunk],
    ) -> EmbeddingBatchResult:
        self.embedding_calls += 1
        embeddings = [
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                vector=[float(index + 1), float(index + 2), float(index + 3)],
                vector_dim=3,
                model_name="fake-bge",
            )
            for index, chunk in enumerate(chunks)
        ]
        return EmbeddingBatchResult(
            doc_id=doc_id,
            model_name="fake-bge",
            embeddings=embeddings,
        )


class FakeActianStorageClient:
    def __init__(self) -> None:
        self.ensure_schema_calls = 0
        self.last_doc_id: str | None = None
        self.last_chunk_count = 0
        self.last_embedding_count = 0

    def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    def upsert_chunks_and_embeddings(self, *, chunking, embeddings) -> tuple[int, int]:
        self.last_doc_id = chunking.doc_id
        self.last_chunk_count = len(chunking.chunks)
        self.last_embedding_count = len(embeddings.embeddings)
        return self.last_chunk_count, self.last_embedding_count


def test_phase_a_pipeline_with_media_runs_end_to_end() -> None:
    ingestion_client = FakeModalIngestionClient()
    storage_client = FakeActianStorageClient()
    pipeline = PhaseAIngestionPipeline(
        ingestion_client=ingestion_client,
        storage_client=storage_client,
    )

    result = pipeline.run(
        doc_id="calculus_101_lecture_01",
        source_file_id="slides_01.pdf",
        source_file_bytes=b"%PDF-1.7 fake",
        media_id="lecture_01.mp4",
        media_bytes=b"fake-media",
    )

    assert ingestion_client.vision_calls == 1
    assert ingestion_client.transcription_calls == 1
    assert ingestion_client.embedding_calls == 1
    assert storage_client.ensure_schema_calls == 1
    assert result.stored_chunk_count == storage_client.last_chunk_count
    assert result.stored_embedding_count == storage_client.last_embedding_count
    assert result.metadata["ingestion_provider"] == "modal"
    assert result.metadata["storage_provider"] == "actian"


def test_phase_a_pipeline_without_media_skips_transcription() -> None:
    ingestion_client = FakeModalIngestionClient()
    storage_client = FakeActianStorageClient()
    pipeline = PhaseAIngestionPipeline(
        ingestion_client=ingestion_client,
        storage_client=storage_client,
    )

    result = pipeline.run(
        doc_id="calculus_101_lecture_02",
        source_file_id="slides_02.pdf",
        source_file_bytes=b"%PDF-1.7 fake",
    )

    assert ingestion_client.transcription_calls == 0
    assert result.stored_chunk_count >= 1


def test_phase_a_pipeline_rejects_wrong_ingestion_provider() -> None:
    ingestion_client = FakeModalIngestionClient()
    storage_client = FakeActianStorageClient()
    pipeline = PhaseAIngestionPipeline(
        ingestion_client=ingestion_client,
        storage_client=storage_client,
        ingestion_provider="openai",
    )

    with pytest.raises(BoundaryViolationError):
        pipeline.run(
            doc_id="calculus_101_lecture_03",
            source_file_id="slides_03.pdf",
            source_file_bytes=b"%PDF-1.7 fake",
        )
