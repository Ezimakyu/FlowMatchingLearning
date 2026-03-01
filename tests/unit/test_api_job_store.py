from pathlib import Path

from backend.app.api.models import JobEvent, JobStage
from backend.app.api.store import JobStore


def test_job_store_register_upload_is_deterministic(tmp_path: Path) -> None:
    store = JobStore(root_dir=tmp_path / "runtime")
    payload = b"%PDF fake content"

    first = store.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=payload,
    )
    second = store.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=payload,
    )

    assert first.upload_id == second.upload_id
    assert Path(first.source_file_path).exists()
    loaded = store.load_upload(upload_id=first.upload_id)
    assert loaded.source_sha256 == first.source_sha256
    assert loaded.doc_id == "doc_calc"
    assert len(loaded.input_items) == 1
    assert loaded.input_items[0].source_file_id == "slides_01"


def test_job_store_job_and_event_persistence(tmp_path: Path) -> None:
    store = JobStore(root_dir=tmp_path / "runtime")
    upload = store.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=b"%PDF fake content",
    )
    job = store.get_or_create_job(upload=upload, model_profile="test")
    duplicate = store.get_or_create_job(upload=upload, model_profile="test")
    assert duplicate.job_id == job.job_id

    store.append_event(
        JobEvent(
            job_id=job.job_id,
            event_type="stage_start",
            stage=JobStage.INGESTING,
            message="started",
        )
    )
    events = store.list_events(job_id=job.job_id)
    assert len(events) == 1
    assert events[0].event_type == "stage_start"
    store.clear_events(job_id=job.job_id)
    assert store.list_events(job_id=job.job_id) == []


def test_job_store_combined_job_is_deterministic(tmp_path: Path) -> None:
    store = JobStore(root_dir=tmp_path / "runtime")
    first_upload = store.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_01",
        source_filename="slides_01.pdf",
        source_bytes=b"%PDF fake content A",
    )
    second_upload = store.register_upload(
        doc_id="doc_calc",
        source_file_id="slides_02",
        source_filename="slides_02.pdf",
        source_bytes=b"%PDF fake content B",
    )
    combined_one = store.get_or_create_combined_job(
        upload_ids=[first_upload.upload_id, second_upload.upload_id],
        doc_id="doc_calc_combined",
        model_profile="test",
    )
    combined_two = store.get_or_create_combined_job(
        upload_ids=[second_upload.upload_id, first_upload.upload_id],
        doc_id="doc_calc_combined",
        model_profile="test",
    )
    assert combined_one.job_id == combined_two.job_id
    assert sorted(combined_one.upload_ids) == sorted(
        [first_upload.upload_id, second_upload.upload_id]
    )
