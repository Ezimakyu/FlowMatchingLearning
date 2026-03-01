import pytest

from backend.app.compute_boundaries import BoundaryViolationError
from backend.app.models import (
    ChunkingResult,
    LLMCallMetadata,
    RawTextChunk,
    TOCData,
    TOCSection,
)
from backend.app.pipelines.phase_b_toc import (
    PhaseBTOCPipeline,
    build_toc_input_text,
)
from backend.app.reasoning import TOCGenerationOutput


class FakeTOCReasoningClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_doc_id: str | None = None
        self.last_document_text: str | None = None

    def generate_toc(self, *, doc_id: str, document_text: str) -> TOCGenerationOutput:
        self.calls += 1
        self.last_doc_id = doc_id
        self.last_document_text = document_text
        toc = TOCData(
            doc_id=doc_id,
            sections=[
                TOCSection(
                    section_id="intro",
                    title="Introduction",
                    order=0,
                    chunk_ids=[],
                    key_terms=[],
                    children=[],
                )
            ],
        )
        llm_call = LLMCallMetadata(
            provider="openai",
            prompt_name="toc_generation",
            prompt_version="2026-02-28.v2",
            model="gpt-4.1-mini",
        )
        return TOCGenerationOutput(
            toc=toc,
            llm_call=llm_call,
            prompt_tag="toc_generation:2026-02-28.v2",
            prompt_checksum="fake-checksum",
            raw_response_text='{"doc_id":"x","sections":[]}',
        )


def _sample_chunking(doc_id: str = "doc_1") -> ChunkingResult:
    return ChunkingResult(
        doc_id=doc_id,
        chunks=[
            RawTextChunk(
                chunk_id=f"{doc_id}:vision_text:00000",
                doc_id=doc_id,
                source_type="vision_text",
                order=0,
                text="Limits are introduced before derivatives.",
                token_estimate=8,
            ),
            RawTextChunk(
                chunk_id=f"{doc_id}:vision_text:00001",
                doc_id=doc_id,
                source_type="vision_text",
                order=1,
                text="Derivatives are defined through limits.",
                token_estimate=8,
            ),
        ],
    )


def test_build_toc_input_text_contains_chunk_metadata() -> None:
    chunking = _sample_chunking()
    text = build_toc_input_text(chunking=chunking, max_chars=10000)
    assert "chunk_id=doc_1:vision_text:00000" in text
    assert "source=vision_text" in text
    assert "Limits are introduced before derivatives." in text


def test_build_toc_input_text_truncates_when_needed() -> None:
    chunking = _sample_chunking()
    text = build_toc_input_text(chunking=chunking, max_chars=120)
    assert "[TRUNCATED_FOR_CONTEXT_LIMIT]" in text


def test_phase_b_toc_pipeline_runs_with_fake_client() -> None:
    fake_client = FakeTOCReasoningClient()
    pipeline = PhaseBTOCPipeline(reasoning_client=fake_client)
    chunking = _sample_chunking("doc_calc")

    result = pipeline.run(doc_id="doc_calc", chunking=chunking)
    assert fake_client.calls == 1
    assert fake_client.last_doc_id == "doc_calc"
    assert "chunk_id=doc_calc:vision_text:00000" in (fake_client.last_document_text or "")
    assert result.toc.doc_id == "doc_calc"
    assert result.llm_call.provider == "openai"


def test_phase_b_toc_pipeline_rejects_wrong_provider() -> None:
    fake_client = FakeTOCReasoningClient()
    pipeline = PhaseBTOCPipeline(
        reasoning_client=fake_client,
        reasoning_provider="anthropic",
    )
    with pytest.raises(BoundaryViolationError):
        pipeline.run(doc_id="doc_1", chunking=_sample_chunking("doc_1"))


def test_phase_b_toc_pipeline_rejects_doc_id_mismatch() -> None:
    fake_client = FakeTOCReasoningClient()
    pipeline = PhaseBTOCPipeline(reasoning_client=fake_client)
    with pytest.raises(ValueError):
        pipeline.run(doc_id="doc_other", chunking=_sample_chunking("doc_1"))
