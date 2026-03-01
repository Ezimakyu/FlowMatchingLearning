from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models import GraphData


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobStage(str, Enum):
    INGESTING = "INGESTING"
    TOC = "TOC"
    SECTION_PARSING = "SECTION_PARSING"
    GRAPH_FINALIZED = "GRAPH_FINALIZED"
    EXPORTED = "EXPORTED"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UploadInputItem(StrictAPIModel):
    source_file_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    source_file_path: str = Field(min_length=1)
    media_id: str | None = None
    media_filename: str | None = None
    media_sha256: str | None = None
    media_file_path: str | None = None


class UploadRecord(StrictAPIModel):
    upload_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source_file_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    source_file_path: str = Field(min_length=1)
    media_id: str | None = None
    media_filename: str | None = None
    media_sha256: str | None = None
    media_file_path: str | None = None
    input_items: list[UploadInputItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobArtifactPaths(StrictAPIModel):
    phase_a_json: str = Field(min_length=1)
    toc_json: str = Field(min_length=1)
    toc_meta_json: str = Field(min_length=1)
    graph_json: str = Field(min_length=1)
    rolling_state_json: str = Field(min_length=1)
    section_results_json: str = Field(min_length=1)
    export_json: str | None = None


class JobRecord(StrictAPIModel):
    job_id: str = Field(min_length=1)
    deterministic_key: str = Field(min_length=1)
    upload_id: str = Field(min_length=1)
    upload_ids: list[str] = Field(default_factory=list)
    doc_id: str = Field(min_length=1)
    model_profile: Literal["test", "demo"] = "test"
    stage: JobStage = JobStage.INGESTING
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifacts: JobArtifactPaths
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobEvent(StrictAPIModel):
    job_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    level: Literal["INFO", "WARNING", "ERROR"] = "INFO"
    stage: JobStage | None = None
    message: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class StartJobRequest(StrictAPIModel):
    upload_id: str = Field(min_length=1)
    model_profile: Literal["test", "demo"] = "test"
    force_restart: bool = False


class StartCombinedJobRequest(StrictAPIModel):
    upload_ids: list[str] = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    model_profile: Literal["test", "demo"] = "test"
    force_restart: bool = False


class JobStatusResponse(StrictAPIModel):
    job: JobRecord
    events: list[JobEvent] = Field(default_factory=list)


class ExportGraphRequest(StrictAPIModel):
    output_path: str | None = None


class ExportGraphResponse(StrictAPIModel):
    job_id: str = Field(min_length=1)
    stage: JobStage
    export_path: str = Field(min_length=1)
    graph: GraphData


__all__ = [
    "ExportGraphRequest",
    "ExportGraphResponse",
    "JobArtifactPaths",
    "JobEvent",
    "JobRecord",
    "JobStage",
    "JobStatus",
    "JobStatusResponse",
    "StartCombinedJobRequest",
    "StartJobRequest",
    "UploadInputItem",
    "UploadRecord",
    "utc_now",
]
