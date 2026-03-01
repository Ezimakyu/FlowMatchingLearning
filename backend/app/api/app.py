from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from backend.app.api.models import (
    ExportGraphRequest,
    ExportGraphResponse,
    JobStatusResponse,
    StartJobRequest,
    UploadRecord,
)
from backend.app.api.orchestrator import JobConflictError, JobNotFoundError, JobOrchestrator


def create_app(*, orchestrator: JobOrchestrator | None = None) -> FastAPI:
    app = FastAPI(
        title="Flow Matching Learning API",
        version="0.1.0",
        description="Job orchestration API for ingestion, graph generation, and export.",
    )
    runtime_orchestrator = orchestrator or JobOrchestrator()
    logger = logging.getLogger(__name__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/upload", response_model=UploadRecord)
    async def upload_document(
        doc_id: str = Form(...),
        source_file_id: str = Form(...),
        source_file: UploadFile = File(...),
        media_id: str | None = Form(default=None),
        media_file: UploadFile | None = File(default=None),
    ) -> UploadRecord:
        source_bytes = await source_file.read()
        media_bytes = await media_file.read() if media_file is not None else None
        if not source_bytes:
            raise HTTPException(status_code=400, detail="source_file is empty.")
        if media_file is not None and not media_bytes:
            raise HTTPException(status_code=400, detail="media_file is empty.")
        resolved_media_id = media_id
        if media_file is not None and not resolved_media_id:
            resolved_media_id = Path(media_file.filename or "media").stem
        try:
            return runtime_orchestrator.register_upload(
                doc_id=doc_id,
                source_file_id=source_file_id,
                source_filename=source_file.filename or "source.bin",
                source_bytes=source_bytes,
                media_id=resolved_media_id,
                media_filename=(media_file.filename if media_file is not None else None),
                media_bytes=media_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/jobs/start", response_model=JobStatusResponse)
    def start_job(request: StartJobRequest) -> JobStatusResponse:
        try:
            job = runtime_orchestrator.start_job(request)
            return runtime_orchestrator.get_job_status(job_id=job.job_id, events_limit=50)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(
        job_id: str,
        events_limit: int = Query(default=100, ge=1, le=1000),
    ) -> JobStatusResponse:
        try:
            return runtime_orchestrator.get_job_status(job_id=job_id, events_limit=events_limit)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/graph")
    def get_graph(job_id: str) -> dict:
        try:
            graph = runtime_orchestrator.get_graph(job_id=job_id)
            return graph.model_dump(mode="json")
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/export", response_model=ExportGraphResponse)
    def export_graph(job_id: str, request: ExportGraphRequest) -> ExportGraphResponse:
        try:
            return runtime_orchestrator.export_graph(
                job_id=job_id,
                output_path=request.output_path,
            )
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("api.export_graph_failed job_id=%s", job_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


__all__ = ["app", "create_app"]
