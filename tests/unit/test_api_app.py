from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.app import create_app
from backend.app.api.models import (
    ExportGraphResponse,
    JobArtifactPaths,
    JobRecord,
    JobStage,
    JobStatus,
    JobStatusResponse,
    UploadRecord,
)
from backend.app.models import ConceptNode, GraphData, SourceMaterial


class FakeOrchestrator:
    def __init__(self, tmp_path: Path) -> None:
        self.upload = UploadRecord(
            upload_id="upl_fake_1",
            doc_id="doc_calc",
            source_file_id="slides_01",
            source_filename="slides_01.pdf",
            source_sha256="a" * 64,
            source_file_path=str(tmp_path / "source.bin"),
        )
        artifacts = JobArtifactPaths(
            phase_a_json=str(tmp_path / "phase_a_result.json"),
            toc_json=str(tmp_path / "toc.json"),
            toc_meta_json=str(tmp_path / "toc_meta.json"),
            graph_json=str(tmp_path / "graph_data.json"),
            rolling_state_json=str(tmp_path / "rolling_state.json"),
            section_results_json=str(tmp_path / "section_parse_results.json"),
        )
        self.job = JobRecord(
            job_id="job_fake_1",
            deterministic_key="upl_fake_1:doc_calc:test",
            upload_id=self.upload.upload_id,
            doc_id=self.upload.doc_id,
            model_profile="test",
            stage=JobStage.GRAPH_FINALIZED,
            status=JobStatus.COMPLETED,
            artifacts=artifacts,
        )
        self.graph = GraphData(
            graph_id="graph_doc_calc",
            nodes=[
                ConceptNode(
                    id="doc_calc:sec_limits:limits",
                    label="Limits",
                    summary="Foundational limits.",
                    source_material=SourceMaterial(
                        doc_id="doc_calc",
                        section_id="sec_limits",
                        chunk_ids=["doc_calc:vision_text:00000"],
                    ),
                )
            ],
            edges=[],
        )

    def register_upload(self, **kwargs) -> UploadRecord:
        _ = kwargs
        return self.upload

    def start_job(self, request) -> JobRecord:
        _ = request
        return self.job

    def get_job_status(self, *, job_id: str, events_limit: int = 100) -> JobStatusResponse:
        _ = job_id
        _ = events_limit
        return JobStatusResponse(job=self.job, events=[])

    def get_graph(self, *, job_id: str) -> GraphData:
        _ = job_id
        return self.graph

    def export_graph(self, *, job_id: str, output_path: str | None = None) -> ExportGraphResponse:
        _ = output_path
        return ExportGraphResponse(
            job_id=job_id,
            stage=JobStage.EXPORTED,
            export_path=str(Path(self.job.artifacts.graph_json).with_name("export_graph.json")),
            graph=self.graph,
        )


def test_api_endpoints_basic_flow(tmp_path: Path) -> None:
    orchestrator = FakeOrchestrator(tmp_path)
    app = create_app(orchestrator=orchestrator)
    client = TestClient(app)

    upload_response = client.post(
        "/api/v1/upload",
        data={
            "doc_id": "doc_calc",
            "source_file_id": "slides_01",
        },
        files={"source_file": ("slides_01.pdf", b"%PDF-1.7 fake", "application/pdf")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["upload_id"] == "upl_fake_1"

    start_response = client.post(
        "/api/v1/jobs/start",
        json={"upload_id": "upl_fake_1", "model_profile": "test"},
    )
    assert start_response.status_code == 200
    assert start_response.json()["job"]["job_id"] == "job_fake_1"

    status_response = client.get("/api/v1/jobs/job_fake_1")
    assert status_response.status_code == 200
    assert status_response.json()["job"]["stage"] == "GRAPH_FINALIZED"

    graph_response = client.get("/api/v1/jobs/job_fake_1/graph")
    assert graph_response.status_code == 200
    assert graph_response.json()["graph_id"] == "graph_doc_calc"

    export_response = client.post("/api/v1/jobs/job_fake_1/export", json={})
    assert export_response.status_code == 200
    assert export_response.json()["stage"] == "EXPORTED"
