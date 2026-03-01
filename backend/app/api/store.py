from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

from backend.app.api.models import (
    JobArtifactPaths,
    JobEvent,
    JobRecord,
    JobStage,
    JobStatus,
    UploadRecord,
    utc_now,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _deterministic_upload_id(
    *,
    doc_id: str,
    source_file_id: str,
    source_sha256: str,
    media_id: str | None,
    media_sha256: str | None,
) -> str:
    payload = "|".join(
        [
            doc_id.strip(),
            source_file_id.strip(),
            source_sha256.strip(),
            (media_id or "").strip(),
            (media_sha256 or "").strip(),
        ]
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()
    return f"upl_{digest}"


def _deterministic_job_id(*, deterministic_key: str) -> str:
    digest = hashlib.blake2b(deterministic_key.encode("utf-8"), digest_size=10).hexdigest()
    return f"job_{digest}"


class JobStore:
    def __init__(self, *, root_dir: str | Path = "artifacts/job_runtime") -> None:
        self.root_dir = Path(root_dir)
        self.uploads_dir = self.root_dir / "uploads"
        self.jobs_dir = self.root_dir / "jobs"
        self._lock = threading.Lock()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

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
        normalized_doc_id = doc_id.strip()
        normalized_source_file_id = source_file_id.strip()
        if not normalized_doc_id:
            raise ValueError("doc_id must be non-empty.")
        if not normalized_source_file_id:
            raise ValueError("source_file_id must be non-empty.")
        if not source_bytes:
            raise ValueError("source_bytes must be non-empty.")
        if (media_bytes is None) != (media_id is None):
            raise ValueError("media_id and media_bytes must be provided together.")

        source_sha256 = _sha256_bytes(source_bytes)
        media_sha256 = _sha256_bytes(media_bytes) if media_bytes is not None else None
        upload_id = _deterministic_upload_id(
            doc_id=normalized_doc_id,
            source_file_id=normalized_source_file_id,
            source_sha256=source_sha256,
            media_id=media_id,
            media_sha256=media_sha256,
        )

        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_path = upload_dir / "source.bin"
        media_path = upload_dir / "media.bin"
        upload_json_path = upload_dir / "upload.json"

        if not source_path.exists():
            source_path.write_bytes(source_bytes)
        if media_bytes is not None and not media_path.exists():
            media_path.write_bytes(media_bytes)

        record = UploadRecord(
            upload_id=upload_id,
            doc_id=normalized_doc_id,
            source_file_id=normalized_source_file_id,
            source_filename=source_filename.strip() or "source.bin",
            source_sha256=source_sha256,
            source_file_path=str(source_path),
            media_id=media_id.strip() if media_id else None,
            media_filename=(media_filename.strip() if media_filename else None),
            media_sha256=media_sha256,
            media_file_path=str(media_path) if media_bytes is not None else None,
            metadata={
                "source_bytes": len(source_bytes),
                "media_bytes": len(media_bytes) if media_bytes is not None else 0,
            },
        )
        self._atomic_write_json(upload_json_path, record.model_dump(mode="json"))
        return record

    def load_upload(self, *, upload_id: str) -> UploadRecord:
        path = self.uploads_dir / upload_id / "upload.json"
        payload = self._read_json(path)
        return UploadRecord.model_validate(payload)

    def get_or_create_job(
        self,
        *,
        upload: UploadRecord,
        model_profile: str,
    ) -> JobRecord:
        if model_profile not in {"test", "demo"}:
            raise ValueError("model_profile must be either 'test' or 'demo'.")
        deterministic_key = f"{upload.upload_id}:{upload.doc_id}:{model_profile}"
        job_id = _deterministic_job_id(deterministic_key=deterministic_key)
        job_dir = self.jobs_dir / job_id
        artifacts_dir = job_dir / "artifacts"
        job_path = job_dir / "job.json"

        if job_path.exists():
            return JobRecord.model_validate(self._read_json(job_path))

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifacts = JobArtifactPaths(
            phase_a_json=str(artifacts_dir / "phase_a_result.json"),
            toc_json=str(artifacts_dir / "toc.json"),
            toc_meta_json=str(artifacts_dir / "toc_meta.json"),
            graph_json=str(artifacts_dir / "graph_data.json"),
            rolling_state_json=str(artifacts_dir / "rolling_state.json"),
            section_results_json=str(artifacts_dir / "section_parse_results.json"),
        )
        job = JobRecord(
            job_id=job_id,
            deterministic_key=deterministic_key,
            upload_id=upload.upload_id,
            doc_id=upload.doc_id,
            model_profile=model_profile,
            stage=JobStage.INGESTING,
            status=JobStatus.PENDING,
            artifacts=artifacts,
        )
        self.save_job(job)
        return job

    def save_job(self, job: JobRecord) -> JobRecord:
        updated = job.model_copy(update={"updated_at": utc_now()})
        path = self.jobs_dir / updated.job_id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(path, updated.model_dump(mode="json"))
        return updated

    def load_job(self, *, job_id: str) -> JobRecord:
        path = self.jobs_dir / job_id / "job.json"
        payload = self._read_json(path)
        return JobRecord.model_validate(payload)

    def append_event(self, event: JobEvent) -> None:
        path = self.jobs_dir / event.job_id / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event.model_dump(mode="json"), ensure_ascii=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")

    def list_events(self, *, job_id: str, limit: int = 100) -> list[JobEvent]:
        path = self.jobs_dir / job_id / "events.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if limit > 0:
            lines = lines[-limit:]
        return [JobEvent.model_validate(json.loads(line)) for line in lines]

    def write_json_artifact(self, *, path: str, payload: dict | list) -> None:
        self._atomic_write_json(Path(path), payload)

    def _atomic_write_json(self, path: Path, payload: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        serialized = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
        with self._lock:
            temp_path.write_text(serialized, encoding="utf-8")
            temp_path.replace(path)

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

