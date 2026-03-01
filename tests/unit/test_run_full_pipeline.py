import json
from argparse import Namespace
from pathlib import Path

from backend.app.models import (
    ChunkEmbedding,
    ChunkingResult,
    EmbeddingBatchResult,
    GraphData,
    LLMCallMetadata,
    PhaseAIngestionResult,
    RawTextChunk,
    RollingState,
    SectionParseResult,
    TOCData,
    TOCSection,
)
from backend.app.pipelines import PhaseBGraphOutput
from backend.app.reasoning import TOCGenerationOutput
from backend.tools import run_full_pipeline


class FakeIngestionClient:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class FakeStorageClient:
    def __init__(self, config=None) -> None:
        self.config = config


class FakeTOCReasoningClient:
    def __init__(self, config=None) -> None:
        self.config = config


class FakeSectionReasoningClient:
    def __init__(self, config=None) -> None:
        self.config = config


class FakePhaseAPipeline:
    def __init__(self, *, ingestion_client, storage_client, config) -> None:
        self.ingestion_client = ingestion_client
        self.storage_client = storage_client
        self.config = config

    def run_batch(self, *, doc_id: str, inputs) -> PhaseAIngestionResult:
        _ = inputs
        chunking = ChunkingResult(
            doc_id=doc_id,
            chunks=[
                RawTextChunk(
                    chunk_id=f"{doc_id}:vision_text:00000",
                    doc_id=doc_id,
                    source_type="vision_text",
                    order=0,
                    text="Limits introduction.",
                    token_estimate=4,
                )
            ],
        )
        embeddings = EmbeddingBatchResult(
            doc_id=doc_id,
            model_name="BAAI/bge-m3",
            embeddings=[
                ChunkEmbedding(
                    chunk_id=f"{doc_id}:vision_text:00000",
                    vector=[0.1, 0.2, 0.3],
                    vector_dim=3,
                    model_name="BAAI/bge-m3",
                )
            ],
        )
        return PhaseAIngestionResult(
            doc_id=doc_id,
            chunking=chunking,
            embeddings=embeddings,
            stored_chunk_count=1,
            stored_embedding_count=1,
        )


class FakeTOCPipeline:
    def __init__(self, *, reasoning_client, config, reasoning_provider) -> None:
        self.reasoning_client = reasoning_client
        self.config = config
        self.reasoning_provider = reasoning_provider

    def run(self, *, doc_id: str, chunking) -> TOCGenerationOutput:
        _ = chunking
        toc = TOCData(
            doc_id=doc_id,
            sections=[
                TOCSection(
                    section_id="intro",
                    title="Intro",
                    order=0,
                    chunk_ids=[f"{doc_id}:vision_text:00000"],
                    key_terms=[],
                    children=[],
                )
            ],
        )
        return TOCGenerationOutput(
            toc=toc,
            llm_call=LLMCallMetadata(
                provider="openai",
                prompt_name="toc_generation",
                prompt_version="2026-02-28.v2",
                model="gpt-4.1-mini",
                request_id="req_toc",
            ),
            prompt_tag="toc_generation:2026-02-28.v2",
            prompt_checksum="checksum_toc",
            raw_response_text="{}",
        )


class FakeGraphPipeline:
    def __init__(
        self,
        *,
        reasoning_client,
        storage_client,
        config,
        reasoning_provider,
        storage_provider,
    ) -> None:
        self.reasoning_client = reasoning_client
        self.storage_client = storage_client
        self.config = config
        self.reasoning_provider = reasoning_provider
        self.storage_provider = storage_provider

    def run(self, *, doc_id: str, toc, chunking, embeddings, job_id=None) -> PhaseBGraphOutput:
        _ = toc
        _ = chunking
        _ = embeddings
        graph = GraphData(graph_id=f"graph_{doc_id}", metadata={"doc_id": doc_id})
        rolling_state = RollingState(job_id=job_id or "job_full", doc_id=doc_id)
        section_result = SectionParseResult(
            job_id=rolling_state.job_id,
            doc_id=doc_id,
            section_id="intro",
            section_order=0,
            section_title="Intro",
        )
        return PhaseBGraphOutput(
            graph=graph,
            rolling_state=rolling_state,
            section_results=[section_result],
        )


def test_run_full_pipeline_main_writes_all_artifacts(monkeypatch, tmp_path: Path) -> None:
    source_file = tmp_path / "slides.pdf"
    source_file.write_bytes(b"%PDF-1.7 fake")

    hp_json = tmp_path / "hyperparameters.json"
    hp_json.write_text(
        json.dumps(
            {
                "phase_a": {
                    "max_vision_chunk_tokens": 260,
                    "max_transcript_chunk_tokens": 220,
                    "include_transcript_chunks": True,
                },
                "phase_b": {
                    "toc_generation": {
                        "max_input_chars": 10000,
                        "model_test": "gpt-4.1-mini",
                        "model_demo": "gpt-5.1",
                        "prompt_version": "2026-02-28.v2",
                        "temperature": 0.0,
                        "max_output_tokens": 4000,
                    },
                    "iteration_loop": {
                        "top_k_historical_matches": 12,
                        "similarity_threshold": 0.72,
                        "similarity_fallback_threshold": 0.62,
                        "edge_acceptance_confidence_threshold": 0.6,
                        "retrieval_overfetch_multiplier": 4,
                        "max_section_chars_per_call": 30000,
                        "max_state_nodes_in_context": 200,
                        "max_historical_nodes_for_local_similarity": 400,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "full_outputs"
    monkeypatch.setattr(run_full_pipeline, "ModalRemoteIngestionClient", FakeIngestionClient)
    monkeypatch.setattr(run_full_pipeline, "ActianCortexStore", FakeStorageClient)
    monkeypatch.setattr(run_full_pipeline, "OpenAITOCReasoningClient", FakeTOCReasoningClient)
    monkeypatch.setattr(
        run_full_pipeline, "OpenAISectionReasoningClient", FakeSectionReasoningClient
    )
    monkeypatch.setattr(run_full_pipeline, "PhaseAIngestionPipeline", FakePhaseAPipeline)
    monkeypatch.setattr(run_full_pipeline, "PhaseBTOCPipeline", FakeTOCPipeline)
    monkeypatch.setattr(run_full_pipeline, "PhaseBGraphPipeline", FakeGraphPipeline)
    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            doc_id="doc_full",
            source_file=[str(source_file)],
            source_file_id=["slides_01"],
            media_file=[],
            media_id=[],
            output_dir=str(output_dir),
            output_phase_a_json=None,
            output_toc_json=None,
            output_toc_meta_json=None,
            output_graph_json=None,
            output_rolling_state_json=None,
            output_section_results_json=None,
            actian_addr=None,
            model=None,
            model_profile="test",
            toc_prompt_version=None,
            section_prompt_version=None,
            edge_prompt_version=None,
            job_id="job_full",
            hyperparams_json=str(hp_json),
            env_file=".env",
            no_env_file=True,
        ),
    )

    run_full_pipeline.main()

    phase_a_path = output_dir / "phase_a_result.json"
    toc_path = output_dir / "toc.json"
    toc_meta_path = output_dir / "toc_meta.json"
    graph_path = output_dir / "graph_data.json"
    rolling_path = output_dir / "rolling_state.json"
    section_results_path = output_dir / "section_parse_results.json"

    assert phase_a_path.exists()
    assert toc_path.exists()
    assert toc_meta_path.exists()
    assert graph_path.exists()
    assert rolling_path.exists()
    assert section_results_path.exists()

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    rolling_payload = json.loads(rolling_path.read_text(encoding="utf-8"))
    section_payload = json.loads(section_results_path.read_text(encoding="utf-8"))
    assert graph_payload["graph_id"] == "graph_doc_full"
    assert rolling_payload["job_id"] == "job_full"
    assert section_payload[0]["section_id"] == "intro"
