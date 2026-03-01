from pathlib import Path

from backend.app.api.models import JobStage, JobStatus, StartJobRequest
from backend.app.api.orchestrator import JobOrchestrator, OrchestratorConfig
from backend.app.api.store import JobStore
from backend.app.models import (
    ChunkEmbedding,
    ChunkingResult,
    ConceptEdge,
    ConceptNode,
    EmbeddingBatchResult,
    GraphData,
    LLMCallMetadata,
    PhaseAIngestionResult,
    RawTextChunk,
    RollingState,
    SectionConcept,
    SectionParseResult,
    SourceMaterial,
    TOCData,
    TOCSection,
)
from backend.app.pipelines import PhaseBGraphOutput
from backend.app.reasoning import TOCGenerationOutput


def _sample_phase_a(doc_id: str) -> PhaseAIngestionResult:
    chunk = RawTextChunk(
        chunk_id=f"{doc_id}:vision_text:00000",
        doc_id=doc_id,
        source_type="vision_text",
        order=0,
        text="Limits are introduced before derivatives.",
        token_estimate=8,
    )
    embedding = ChunkEmbedding(
        chunk_id=chunk.chunk_id,
        vector=[0.1, 0.2, 0.3],
        vector_dim=3,
        model_name="BAAI/bge-m3",
    )
    return PhaseAIngestionResult(
        doc_id=doc_id,
        chunking=ChunkingResult(doc_id=doc_id, chunks=[chunk]),
        embeddings=EmbeddingBatchResult(
            doc_id=doc_id,
            model_name="BAAI/bge-m3",
            embeddings=[embedding],
        ),
        stored_chunk_count=1,
        stored_embedding_count=1,
    )


def _sample_toc_output(doc_id: str) -> TOCGenerationOutput:
    toc = TOCData(
        doc_id=doc_id,
        sections=[
            TOCSection(
                section_id="sec_limits",
                title="Limits",
                order=0,
                chunk_ids=[f"{doc_id}:vision_text:00000"],
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
        prompt_checksum="checksum",
        raw_response_text="{}",
    )


def _sample_graph_output(doc_id: str, job_id: str) -> PhaseBGraphOutput:
    limits = ConceptNode(
        id=f"{doc_id}:sec_limits:limits",
        label="Limits",
        summary="Foundational limit behavior.",
        aliases=["limit"],
        source_material=SourceMaterial(
            doc_id=doc_id,
            section_id="sec_limits",
            chunk_ids=[f"{doc_id}:vision_text:00000"],
        ),
    )
    derivatives = ConceptNode(
        id=f"{doc_id}:sec_derivatives:derivatives",
        label="Derivatives",
        summary="Rate of change via limits.",
        aliases=["differentiation"],
        source_material=SourceMaterial(
            doc_id=doc_id,
            section_id="sec_derivatives",
            chunk_ids=[f"{doc_id}:vision_text:00000"],
        ),
    )
    edge = ConceptEdge(
        id="edge:1234abcd",
        source=limits.id,
        target=derivatives.id,
        explanation="Derivatives are defined from limits.",
        confidence=0.9,
    )
    graph = GraphData(graph_id=f"graph_{doc_id}", nodes=[limits, derivatives], edges=[edge])
    rolling_state = RollingState(
        job_id=job_id,
        doc_id=doc_id,
        nodes=[limits, derivatives],
        edges=[edge],
        concept_alias_index={"limits": limits.id, "derivatives": derivatives.id},
    )
    section_result = SectionParseResult(
        job_id=job_id,
        doc_id=doc_id,
        section_id="sec_derivatives",
        section_order=0,
        section_title="Derivatives",
        concepts=[
            SectionConcept(
                concept_id=derivatives.id,
                label=derivatives.label,
                summary=derivatives.summary,
                aliases=derivatives.aliases,
                source_chunk_ids=[f"{doc_id}:vision_text:00000"],
                confidence=0.9,
            )
        ],
    )
    return PhaseBGraphOutput(
        graph=graph,
        rolling_state=rolling_state,
        section_results=[section_result],
    )


def test_orchestrator_runs_state_machine_and_export(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "runtime"
    orchestrator = JobOrchestrator(
        config=OrchestratorConfig(
            storage_dir=str(runtime_dir),
            hyperparams_json="backend/config/hyperparameters.json",
            run_jobs_async=False,
        ),
        store=JobStore(root_dir=runtime_dir),
    )
    upload = orchestrator.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=b"%PDF fake",
    )

    monkeypatch.setattr(
        orchestrator,
        "_run_ingesting_stage",
        lambda *, job, upload: _sample_phase_a(job.doc_id),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_toc_stage",
        lambda *, job, phase_a: _sample_toc_output(job.doc_id),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_section_parsing_stage",
        lambda *, job, phase_a, toc: _sample_graph_output(job.doc_id, job.job_id),
    )

    job = orchestrator.start_job(
        StartJobRequest(upload_id=upload.upload_id, model_profile="test")
    )
    assert job.status == JobStatus.COMPLETED
    assert job.stage == JobStage.GRAPH_FINALIZED
    assert Path(job.artifacts.phase_a_json).exists()
    assert Path(job.artifacts.toc_json).exists()
    assert Path(job.artifacts.graph_json).exists()

    status = orchestrator.get_job_status(job_id=job.job_id, events_limit=200)
    completed_stages = [event.stage for event in status.events if event.event_type == "stage_complete"]
    assert completed_stages == [JobStage.INGESTING, JobStage.TOC, JobStage.SECTION_PARSING]

    export_path = tmp_path / "graph_export.json"
    export = orchestrator.export_graph(job_id=job.job_id, output_path=str(export_path))
    assert export.stage == JobStage.EXPORTED
    assert export_path.exists()


def test_orchestrator_start_is_idempotent_for_completed_job(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_dir = tmp_path / "runtime"
    orchestrator = JobOrchestrator(
        config=OrchestratorConfig(
            storage_dir=str(runtime_dir),
            hyperparams_json="backend/config/hyperparameters.json",
            run_jobs_async=False,
        ),
        store=JobStore(root_dir=runtime_dir),
    )
    upload = orchestrator.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=b"%PDF fake",
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_ingesting_stage",
        lambda *, job, upload: _sample_phase_a(job.doc_id),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_toc_stage",
        lambda *, job, phase_a: _sample_toc_output(job.doc_id),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_section_parsing_stage",
        lambda *, job, phase_a, toc: _sample_graph_output(job.doc_id, job.job_id),
    )

    first = orchestrator.start_job(StartJobRequest(upload_id=upload.upload_id, model_profile="test"))
    second = orchestrator.start_job(StartJobRequest(upload_id=upload.upload_id, model_profile="test"))
    assert first.job_id == second.job_id
    assert second.status == JobStatus.COMPLETED
