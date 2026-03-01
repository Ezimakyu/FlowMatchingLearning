from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Literal

from backend.app.api.models import (
    ExportGraphResponse,
    JobEvent,
    JobRecord,
    JobStage,
    JobStatus,
    JobStatusResponse,
    StartCombinedJobRequest,
    StartJobRequest,
    UploadRecord,
    utc_now,
)
from backend.app.api.retry import RetryPolicy, run_with_retry
from backend.app.api.store import JobStore
from backend.app.config import load_hyperparameters
from backend.app.ingestion import ModalRemoteIngestionClient
from backend.app.models import (
    GraphData,
    PhaseAIngestionResult,
    RollingState,
    SectionParseResult,
    TOCData,
)
from backend.app.pipelines import (
    PhaseAIngestionConfig,
    PhaseAIngestionInput,
    PhaseAIngestionPipeline,
    PhaseBGraphConfig,
    PhaseBGraphOutput,
    PhaseBGraphPipeline,
    PhaseBTOCConfig,
    PhaseBTOCPipeline,
)
from backend.app.reasoning import (
    OpenAISectionReasoningClient,
    OpenAISectionReasoningConfig,
    OpenAITOCReasoningClient,
    OpenAITOCReasoningConfig,
    TOCGenerationOutput,
)
from backend.app.storage import ActianCortexConfig, ActianCortexStore


class JobNotFoundError(RuntimeError):
    pass


class JobConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestratorConfig:
    storage_dir: str = field(
        default_factory=lambda: os.getenv("FLOW_JOB_STORAGE_DIR", "artifacts/job_runtime")
    )
    hyperparams_json: str = field(
        default_factory=lambda: os.getenv(
            "FLOW_HYPERPARAMS_JSON", "backend/config/hyperparameters.json"
        )
    )
    run_jobs_async: bool = True
    llm_retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
        )
    )
    db_retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.8,
            backoff_multiplier=2.0,
            max_backoff_seconds=6.0,
        )
    )


class RetryingTOCReasoningClient:
    def __init__(self, *, delegate: OpenAITOCReasoningClient, policy: RetryPolicy) -> None:
        self._delegate = delegate
        self._policy = policy
        self._logger = logging.getLogger(__name__)

    def generate_toc(self, *, doc_id: str, document_text: str) -> TOCGenerationOutput:
        return run_with_retry(
            operation_name="llm.generate_toc",
            fn=lambda: self._delegate.generate_toc(doc_id=doc_id, document_text=document_text),
            policy=self._policy,
            logger=self._logger,
        )


class RetryingSectionReasoningClient:
    def __init__(self, *, delegate: OpenAISectionReasoningClient, policy: RetryPolicy) -> None:
        self._delegate = delegate
        self._policy = policy
        self._logger = logging.getLogger(__name__)

    def extract_section_concepts(
        self,
        *,
        doc_id: str,
        section_id: str,
        section_title: str,
        section_text: str,
        rolling_state_json: str,
    ):
        return run_with_retry(
            operation_name="llm.extract_section_concepts",
            fn=lambda: self._delegate.extract_section_concepts(
                doc_id=doc_id,
                section_id=section_id,
                section_title=section_title,
                section_text=section_text,
                rolling_state_json=rolling_state_json,
            ),
            policy=self._policy,
            logger=self._logger,
        )

    def validate_edge_candidate(
        self,
        *,
        new_concept_json: str,
        historical_concept_json: str,
        supporting_evidence_json: str,
    ):
        return run_with_retry(
            operation_name="llm.validate_edge_candidate",
            fn=lambda: self._delegate.validate_edge_candidate(
                new_concept_json=new_concept_json,
                historical_concept_json=historical_concept_json,
                supporting_evidence_json=supporting_evidence_json,
            ),
            policy=self._policy,
            logger=self._logger,
        )


class RetryingActianStore:
    def __init__(self, *, delegate: ActianCortexStore, policy: RetryPolicy) -> None:
        self._delegate = delegate
        self._policy = policy
        self._logger = logging.getLogger(__name__)

    def ensure_schema(self) -> None:
        run_with_retry(
            operation_name="db.ensure_schema",
            fn=lambda: self._delegate.ensure_schema(),
            policy=self._policy,
            logger=self._logger,
        )

    def upsert_chunks_and_embeddings(self, *, chunking, embeddings) -> tuple[int, int]:
        return run_with_retry(
            operation_name="db.upsert_chunks_and_embeddings",
            fn=lambda: self._delegate.upsert_chunks_and_embeddings(
                chunking=chunking,
                embeddings=embeddings,
            ),
            policy=self._policy,
            logger=self._logger,
        )

    def similarity_search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        model_name: str = "BAAI/bge-m3",
        candidate_limit: int = 0,
    ) -> list[dict[str, Any]]:
        return run_with_retry(
            operation_name="db.similarity_search",
            fn=lambda: self._delegate.similarity_search(
                query_vector=query_vector,
                top_k=top_k,
                min_similarity=min_similarity,
                model_name=model_name,
                candidate_limit=candidate_limit,
            ),
            policy=self._policy,
            logger=self._logger,
        )


class JobOrchestrator:
    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        store: JobStore | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.store = store or JobStore(root_dir=self.config.storage_dir)
        self._logger = logging.getLogger(__name__)
        self._active_threads: dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()

    def register_upload(
        self,
        *,
        doc_id: str,
        source_file_id: str,
        source_filename: str,
        source_bytes: bytes,
        media_id: str | None = None,
        media_filename: str | None = None,
        media_bytes: bytes | None = None,
    ) -> UploadRecord:
        return self.store.register_upload(
            doc_id=doc_id,
            source_file_id=source_file_id,
            source_filename=source_filename,
            source_bytes=source_bytes,
            media_id=media_id,
            media_filename=media_filename,
            media_bytes=media_bytes,
        )

    def start_job(self, request: StartJobRequest) -> JobRecord:
        upload = self._load_upload_or_raise(request.upload_id)
        job = self.store.get_or_create_job(upload=upload, model_profile=request.model_profile)
        if request.force_restart and job.status == JobStatus.RUNNING and self._is_thread_active(job.job_id):
            raise JobConflictError(
                f"Job '{job.job_id}' is currently running and cannot be restarted."
            )
        if request.force_restart:
            job = self._reset_for_restart(job)

        if job.status == JobStatus.RUNNING and self._is_thread_active(job.job_id):
            return job

        if job.status == JobStatus.COMPLETED and job.stage in {
            JobStage.GRAPH_FINALIZED,
            JobStage.EXPORTED,
        }:
            return job

        started_at = job.started_at or utc_now()
        job = self.store.save_job(
            job.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "error": None,
                    "started_at": started_at,
                }
            )
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="job_started",
            message="Job started.",
            stage=job.stage,
        )

        if self.config.run_jobs_async:
            self._start_background_worker(job.job_id)
            return self.store.load_job(job_id=job.job_id)

        self._run_job_worker(job.job_id)
        return self.store.load_job(job_id=job.job_id)

    def start_combined_job(self, request: StartCombinedJobRequest) -> JobRecord:
        normalized_upload_ids = sorted({item.strip() for item in request.upload_ids if item.strip()})
        if not normalized_upload_ids:
            raise ValueError("upload_ids must include at least one upload id.")
        # Ensure all upload ids exist before creating or starting combined job.
        for upload_id in normalized_upload_ids:
            self._load_upload_or_raise(upload_id)

        job = self.store.get_or_create_combined_job(
            upload_ids=normalized_upload_ids,
            doc_id=request.doc_id,
            model_profile=request.model_profile,
        )
        if request.force_restart and job.status == JobStatus.RUNNING and self._is_thread_active(job.job_id):
            raise JobConflictError(
                f"Job '{job.job_id}' is currently running and cannot be restarted."
            )
        if request.force_restart:
            job = self._reset_for_restart(job)

        if job.status == JobStatus.RUNNING and self._is_thread_active(job.job_id):
            return job

        if job.status == JobStatus.COMPLETED and job.stage in {
            JobStage.GRAPH_FINALIZED,
            JobStage.EXPORTED,
        }:
            return job

        started_at = job.started_at or utc_now()
        job = self.store.save_job(
            job.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "error": None,
                    "started_at": started_at,
                }
            )
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="job_started",
            message="Combined job started.",
            stage=job.stage,
            payload={"upload_count": len(job.upload_ids or [job.upload_id])},
        )

        if self.config.run_jobs_async:
            self._start_background_worker(job.job_id)
            return self.store.load_job(job_id=job.job_id)

        self._run_job_worker(job.job_id)
        return self.store.load_job(job_id=job.job_id)

    def get_job_status(self, *, job_id: str, events_limit: int = 100) -> JobStatusResponse:
        job = self._load_job_or_raise(job_id)
        events = self.store.list_events(job_id=job_id, limit=events_limit)
        return JobStatusResponse(job=job, events=events)

    def get_graph(self, *, job_id: str) -> GraphData:
        job = self._load_job_or_raise(job_id)
        graph_path = Path(job.artifacts.graph_json)
        if not graph_path.exists():
            raise ValueError(
                f"Graph artifact is not available yet for job '{job_id}'."
            )
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        return GraphData.model_validate(payload)

    def export_graph(self, *, job_id: str, output_path: str | None = None) -> ExportGraphResponse:
        job = self._load_job_or_raise(job_id)
        graph = self.get_graph(job_id=job_id)
        if output_path:
            target_path = Path(output_path)
            if not target_path.is_absolute():
                target_path = Path.cwd() / target_path
        else:
            target_path = Path(job.artifacts.graph_json).with_name("export_graph.json")

        self.store.write_json_artifact(
            path=str(target_path),
            payload=graph.model_dump(mode="json"),
        )

        updated_artifacts = job.artifacts.model_copy(update={"export_json": str(target_path)})
        job = self.store.save_job(
            job.model_copy(
                update={
                    "stage": JobStage.EXPORTED,
                    "status": JobStatus.COMPLETED,
                    "error": None,
                    "artifacts": updated_artifacts,
                    "completed_at": utc_now(),
                }
            )
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="graph_exported",
            message="Graph export completed.",
            stage=JobStage.EXPORTED,
            payload={"export_path": str(target_path)},
        )
        return ExportGraphResponse(
            job_id=job.job_id,
            stage=job.stage,
            export_path=str(target_path),
            graph=graph,
        )

    def _start_background_worker(self, job_id: str) -> None:
        with self._thread_lock:
            existing = self._active_threads.get(job_id)
            if existing is not None and existing.is_alive():
                return
            thread = threading.Thread(
                target=self._run_job_worker,
                args=(job_id,),
                name=f"job-worker-{job_id[:12]}",
                daemon=True,
            )
            self._active_threads[job_id] = thread
            thread.start()

    def _is_thread_active(self, job_id: str) -> bool:
        with self._thread_lock:
            thread = self._active_threads.get(job_id)
            return bool(thread and thread.is_alive())

    def _run_job_worker(self, job_id: str) -> None:
        stage: JobStage | None = None
        try:
            job = self._load_job_or_raise(job_id)
            upload_ids = job.upload_ids if job.upload_ids else [job.upload_id]
            uploads = [self._load_upload_or_raise(upload_id) for upload_id in upload_ids]

            job = self.store.save_job(
                job.model_copy(
                    update={
                        "status": JobStatus.RUNNING,
                        "attempt_count": job.attempt_count + 1,
                        "error": None,
                    }
                )
            )
            stage = JobStage.INGESTING
            phase_a = self._ensure_phase_a_artifact(job=job, uploads=uploads)
            stage = JobStage.TOC
            toc = self._ensure_toc_artifact(job=job, phase_a=phase_a)
            stage = JobStage.SECTION_PARSING
            graph_output = self._ensure_graph_artifacts(job=job, phase_a=phase_a, toc=toc)

            finished = self._load_job_or_raise(job_id)
            finished = self.store.save_job(
                finished.model_copy(
                    update={
                        "stage": JobStage.GRAPH_FINALIZED,
                        "status": JobStatus.COMPLETED,
                        "error": None,
                        "completed_at": utc_now(),
                    }
                )
            )
            self._emit_event(
                job_id=job_id,
                event_type="job_completed",
                stage=JobStage.GRAPH_FINALIZED,
                message="Job completed with finalized graph.",
                payload={
                    "nodes": len(graph_output.graph.nodes),
                    "edges": len(graph_output.graph.edges),
                },
            )
        except Exception as exc:
            self._logger.exception("job.execution_failed job_id=%s", job_id)
            job = self._load_job_or_raise(job_id)
            failed = self.store.save_job(
                job.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "error": str(exc),
                        "completed_at": utc_now(),
                    }
                )
            )
            self._emit_event(
                job_id=job_id,
                event_type="job_failed",
                stage=stage or failed.stage,
                level="ERROR",
                message="Job failed.",
                payload={"error": str(exc), "error_type": exc.__class__.__name__},
            )
        finally:
            with self._thread_lock:
                self._active_threads.pop(job_id, None)

    def _ensure_phase_a_artifact(
        self,
        *,
        job: JobRecord,
        uploads: list[UploadRecord],
    ) -> PhaseAIngestionResult:
        artifact_path = Path(job.artifacts.phase_a_json)
        if artifact_path.exists():
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                phase_a = PhaseAIngestionResult.model_validate(payload)
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_resume",
                    stage=JobStage.INGESTING,
                    message="INGESTING artifact exists; reused for recovery.",
                    payload={"artifact": str(artifact_path)},
                )
                return phase_a
            except Exception:
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_artifact_invalid",
                    stage=JobStage.INGESTING,
                    level="WARNING",
                    message="INGESTING artifact invalid; recomputing.",
                    payload={"artifact": str(artifact_path)},
                )

        self._set_job_stage(job_id=job.job_id, stage=JobStage.INGESTING)
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_start",
            stage=JobStage.INGESTING,
            message="INGESTING stage started.",
            payload={"input_count": len(uploads)},
        )
        phase_a = self._run_ingesting_stage(job=job, uploads=uploads)
        self.store.write_json_artifact(
            path=job.artifacts.phase_a_json,
            payload=phase_a.model_dump(mode="json"),
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_complete",
            stage=JobStage.INGESTING,
            message="INGESTING stage completed.",
            payload={
                "stored_chunk_count": phase_a.stored_chunk_count,
                "stored_embedding_count": phase_a.stored_embedding_count,
                "input_count": len(uploads),
            },
        )
        return phase_a

    def _ensure_toc_artifact(self, *, job: JobRecord, phase_a: PhaseAIngestionResult) -> TOCData:
        toc_path = Path(job.artifacts.toc_json)
        if toc_path.exists():
            try:
                payload = json.loads(toc_path.read_text(encoding="utf-8"))
                toc = TOCData.model_validate(payload)
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_resume",
                    stage=JobStage.TOC,
                    message="TOC artifact exists; reused for recovery.",
                    payload={"artifact": str(toc_path)},
                )
                return toc
            except Exception:
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_artifact_invalid",
                    stage=JobStage.TOC,
                    level="WARNING",
                    message="TOC artifact invalid; recomputing.",
                    payload={"artifact": str(toc_path)},
                )

        self._set_job_stage(job_id=job.job_id, stage=JobStage.TOC)
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_start",
            stage=JobStage.TOC,
            message="TOC stage started.",
        )
        toc_output = self._run_toc_stage(job=job, phase_a=phase_a)
        self.store.write_json_artifact(
            path=job.artifacts.toc_json,
            payload=toc_output.toc.model_dump(mode="json"),
        )
        self.store.write_json_artifact(
            path=job.artifacts.toc_meta_json,
            payload={
                "llm_call": toc_output.llm_call.model_dump(mode="json"),
                "prompt_tag": toc_output.prompt_tag,
                "prompt_checksum": toc_output.prompt_checksum,
            },
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_complete",
            stage=JobStage.TOC,
            message="TOC stage completed.",
            payload={"top_level_sections": len(toc_output.toc.sections)},
        )
        return toc_output.toc

    def _ensure_graph_artifacts(
        self,
        *,
        job: JobRecord,
        phase_a: PhaseAIngestionResult,
        toc: TOCData,
    ) -> PhaseBGraphOutput:
        graph_path = Path(job.artifacts.graph_json)
        rolling_path = Path(job.artifacts.rolling_state_json)
        section_path = Path(job.artifacts.section_results_json)
        if graph_path.exists() and rolling_path.exists() and section_path.exists():
            try:
                graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
                graph = GraphData.model_validate(graph_payload)
                rolling_payload = json.loads(rolling_path.read_text(encoding="utf-8"))
                section_payload = json.loads(section_path.read_text(encoding="utf-8"))
                # Keep validation strict for recovery.
                output = PhaseBGraphOutput(
                    graph=graph,
                    rolling_state=RollingState.model_validate(rolling_payload),
                    section_results=[
                        SectionParseResult.model_validate(item) for item in section_payload
                    ],
                )
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_resume",
                    stage=JobStage.SECTION_PARSING,
                    message="SECTION_PARSING artifacts exist; reused for recovery.",
                    payload={"graph_artifact": str(graph_path)},
                )
                return output
            except Exception:
                self._emit_event(
                    job_id=job.job_id,
                    event_type="stage_artifact_invalid",
                    stage=JobStage.SECTION_PARSING,
                    level="WARNING",
                    message="SECTION_PARSING artifacts invalid; recomputing.",
                    payload={"graph_artifact": str(graph_path)},
                )

        self._set_job_stage(job_id=job.job_id, stage=JobStage.SECTION_PARSING)
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_start",
            stage=JobStage.SECTION_PARSING,
            message="SECTION_PARSING stage started.",
        )
        graph_output = self._run_section_parsing_stage(job=job, phase_a=phase_a, toc=toc)
        self.store.write_json_artifact(
            path=job.artifacts.graph_json,
            payload=graph_output.graph.model_dump(mode="json"),
        )
        self.store.write_json_artifact(
            path=job.artifacts.rolling_state_json,
            payload=graph_output.rolling_state.model_dump(mode="json"),
        )
        self.store.write_json_artifact(
            path=job.artifacts.section_results_json,
            payload=[item.model_dump(mode="json") for item in graph_output.section_results],
        )
        self._emit_event(
            job_id=job.job_id,
            event_type="stage_complete",
            stage=JobStage.SECTION_PARSING,
            message="SECTION_PARSING stage completed.",
            payload={
                "nodes": len(graph_output.graph.nodes),
                "edges": len(graph_output.graph.edges),
                "parse_log_tail": graph_output.rolling_state.parse_log[-10:],
            },
        )
        return graph_output

    def _run_ingesting_stage(self, *, job: JobRecord, uploads: list[UploadRecord]) -> PhaseAIngestionResult:
        hyperparams = load_hyperparameters(self.config.hyperparams_json)
        if not uploads:
            raise ValueError("Combined ingestion requires at least one upload.")
        ingestion_inputs: list[PhaseAIngestionInput] = []
        for upload in uploads:
            items = upload.input_items
            if not items:
                source_file_id = upload.source_file_id
                source_file_path = upload.source_file_path
                media_id = upload.media_id
                media_file_path = upload.media_file_path
                source_bytes = Path(source_file_path).read_bytes()
                media_bytes = Path(media_file_path).read_bytes() if media_file_path else None
                ingestion_inputs.append(
                    PhaseAIngestionInput(
                        source_file_id=source_file_id,
                        source_file_bytes=source_bytes,
                        media_id=media_id,
                        media_bytes=media_bytes,
                    )
                )
                continue

            for item in items:
                source_bytes = Path(item.source_file_path).read_bytes()
                media_bytes = (
                    Path(item.media_file_path).read_bytes() if item.media_file_path else None
                )
                ingestion_inputs.append(
                    PhaseAIngestionInput(
                        source_file_id=item.source_file_id,
                        source_file_bytes=source_bytes,
                        media_id=item.media_id,
                        media_bytes=media_bytes,
                    )
                )
        storage_client = RetryingActianStore(
            delegate=ActianCortexStore(config=ActianCortexConfig.from_env()),
            policy=self.config.db_retry_policy,
        )
        pipeline = PhaseAIngestionPipeline(
            ingestion_client=ModalRemoteIngestionClient(),
            storage_client=storage_client,
            config=PhaseAIngestionConfig(
                max_vision_chunk_tokens=hyperparams.phase_a.max_vision_chunk_tokens,
                max_transcript_chunk_tokens=hyperparams.phase_a.max_transcript_chunk_tokens,
                include_transcript_chunks=hyperparams.phase_a.include_transcript_chunks,
            ),
            ingestion_provider="modal",
            storage_provider="actian",
        )
        return pipeline.run_batch(doc_id=job.doc_id, inputs=ingestion_inputs)

    def _run_toc_stage(self, *, job: JobRecord, phase_a: PhaseAIngestionResult) -> TOCGenerationOutput:
        hyperparams = load_hyperparameters(self.config.hyperparams_json)
        toc_hp = hyperparams.phase_b.toc_generation
        env_model = os.getenv("OPENAI_REASONING_MODEL", "").strip()
        env_prompt = os.getenv("TOC_PROMPT_VERSION", "").strip()
        env_temperature = os.getenv("OPENAI_REASONING_TEMPERATURE", "").strip()
        env_max_tokens = os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "").strip()

        profile_model = toc_hp.model_test if job.model_profile == "test" else toc_hp.model_demo
        model = env_model or profile_model
        prompt_version = env_prompt or toc_hp.prompt_version
        temperature = float(env_temperature) if env_temperature else toc_hp.temperature
        max_output_tokens = int(env_max_tokens) if env_max_tokens else toc_hp.max_output_tokens

        reasoning = RetryingTOCReasoningClient(
            delegate=OpenAITOCReasoningClient(
                config=OpenAITOCReasoningConfig(
                    model=model,
                    prompt_version=prompt_version,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            ),
            policy=self.config.llm_retry_policy,
        )
        pipeline = PhaseBTOCPipeline(
            reasoning_client=reasoning,
            config=PhaseBTOCConfig(max_input_chars=toc_hp.max_input_chars),
            reasoning_provider="openai",
        )
        return pipeline.run(doc_id=job.doc_id, chunking=phase_a.chunking)

    def _run_section_parsing_stage(
        self,
        *,
        job: JobRecord,
        phase_a: PhaseAIngestionResult,
        toc: TOCData,
    ) -> PhaseBGraphOutput:
        hyperparams = load_hyperparameters(self.config.hyperparams_json)
        toc_hp = hyperparams.phase_b.toc_generation
        loop_hp = hyperparams.phase_b.iteration_loop
        env_model = os.getenv("OPENAI_REASONING_MODEL", "").strip()
        env_temperature = os.getenv("OPENAI_REASONING_TEMPERATURE", "").strip()
        env_max_tokens = os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "").strip()
        env_section_prompt = os.getenv("SECTION_CONCEPT_PROMPT_VERSION", "").strip()
        env_edge_prompt = os.getenv("EDGE_VALIDATION_PROMPT_VERSION", "").strip()

        profile_model = toc_hp.model_test if job.model_profile == "test" else toc_hp.model_demo
        model = env_model or profile_model
        temperature = float(env_temperature) if env_temperature else toc_hp.temperature
        max_output_tokens = int(env_max_tokens) if env_max_tokens else toc_hp.max_output_tokens
        section_prompt_version = env_section_prompt or "2026-03-01.v3"
        edge_prompt_version = env_edge_prompt or "2026-03-01.v3"

        reasoning_client = RetryingSectionReasoningClient(
            delegate=OpenAISectionReasoningClient(
                config=OpenAISectionReasoningConfig(
                    model=model,
                    section_prompt_version=section_prompt_version,
                    edge_prompt_version=edge_prompt_version,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            ),
            policy=self.config.llm_retry_policy,
        )
        storage_client = RetryingActianStore(
            delegate=ActianCortexStore(config=ActianCortexConfig.from_env()),
            policy=self.config.db_retry_policy,
        )
        pipeline = PhaseBGraphPipeline(
            reasoning_client=reasoning_client,
            storage_client=storage_client,
            config=PhaseBGraphConfig(
                top_k_historical_matches=loop_hp.top_k_historical_matches,
                similarity_threshold=loop_hp.similarity_threshold,
                similarity_fallback_threshold=loop_hp.similarity_fallback_threshold,
                edge_acceptance_confidence_threshold=loop_hp.edge_acceptance_confidence_threshold,
                retrieval_overfetch_multiplier=loop_hp.retrieval_overfetch_multiplier,
                max_section_chars_per_call=loop_hp.max_section_chars_per_call,
                max_sections_to_parse=loop_hp.max_sections_to_parse,
                max_llm_concepts_per_section=loop_hp.max_llm_concepts_per_section,
                max_state_nodes_in_context=loop_hp.max_state_nodes_in_context,
                max_historical_nodes_for_local_similarity=loop_hp.max_historical_nodes_for_local_similarity,
                seed_core_nodes_from_toc=loop_hp.seed_core_nodes_from_toc,
                max_seed_core_nodes=loop_hp.max_seed_core_nodes,
                freeze_node_set_after_seed=loop_hp.freeze_node_set_after_seed,
            ),
            reasoning_provider="openai",
            storage_provider="actian",
        )
        return pipeline.run(
            doc_id=job.doc_id,
            toc=toc,
            chunking=phase_a.chunking,
            embeddings=phase_a.embeddings,
            job_id=job.job_id,
        )

    def _set_job_stage(self, *, job_id: str, stage: JobStage) -> JobRecord:
        job = self._load_job_or_raise(job_id)
        return self.store.save_job(
            job.model_copy(
                update={
                    "stage": stage,
                    "status": JobStatus.RUNNING,
                    "error": None,
                }
            )
        )

    def _reset_for_restart(self, job: JobRecord) -> JobRecord:
        self.store.clear_events(job_id=job.job_id)
        for path in [
            job.artifacts.phase_a_json,
            job.artifacts.toc_json,
            job.artifacts.toc_meta_json,
            job.artifacts.graph_json,
            job.artifacts.rolling_state_json,
            job.artifacts.section_results_json,
            job.artifacts.export_json,
        ]:
            if not path:
                continue
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
        restarted = self.store.save_job(
            job.model_copy(
                update={
                    "stage": JobStage.INGESTING,
                    "status": JobStatus.PENDING,
                    "error": None,
                    "completed_at": None,
                }
            )
        )
        self._emit_event(
            job_id=restarted.job_id,
            event_type="job_restart",
            stage=JobStage.INGESTING,
            message="Job restart requested.",
        )
        return restarted

    def _load_upload_or_raise(self, upload_id: str) -> UploadRecord:
        try:
            return self.store.load_upload(upload_id=upload_id)
        except FileNotFoundError as exc:
            raise JobNotFoundError(f"Upload '{upload_id}' was not found.") from exc

    def _load_job_or_raise(self, job_id: str) -> JobRecord:
        try:
            return self.store.load_job(job_id=job_id)
        except FileNotFoundError as exc:
            raise JobNotFoundError(f"Job '{job_id}' was not found.") from exc

    def _emit_event(
        self,
        *,
        job_id: str,
        event_type: str,
        message: str,
        stage: JobStage | None = None,
        level: Literal["INFO", "WARNING", "ERROR"] = "INFO",
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = JobEvent(
            job_id=job_id,
            event_type=event_type,
            message=message,
            stage=stage,
            level=level,
            payload=payload or {},
        )
        self.store.append_event(event)


__all__ = [
    "JobConflictError",
    "JobNotFoundError",
    "JobOrchestrator",
    "OrchestratorConfig",
]
