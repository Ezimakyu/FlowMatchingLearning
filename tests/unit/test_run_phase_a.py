from argparse import Namespace
from pathlib import Path

import pytest

from backend.tools import run_phase_a


def test_build_ingestion_inputs_single_source_without_media(tmp_path: Path) -> None:
    source_file = tmp_path / "slides.pdf"
    source_file.write_bytes(b"%PDF-1.7 fake")
    args = Namespace(
        source_file=[str(source_file)],
        source_file_id=["slides_01"],
        media_file=[],
        media_id=[],
    )

    inputs = run_phase_a.build_ingestion_inputs(args)

    assert len(inputs) == 1
    assert inputs[0].source_file_id == "slides_01"
    assert inputs[0].media_id is None
    assert inputs[0].media_bytes is None


def test_build_ingestion_inputs_multiple_sources_with_media(tmp_path: Path) -> None:
    source_1 = tmp_path / "slides_01.pdf"
    source_2 = tmp_path / "slides_02.pdf"
    media_1 = tmp_path / "lecture_01.mp4"
    media_2 = tmp_path / "lecture_02.mp4"
    source_1.write_bytes(b"%PDF-1.7 fake 1")
    source_2.write_bytes(b"%PDF-1.7 fake 2")
    media_1.write_bytes(b"fake-media-1")
    media_2.write_bytes(b"fake-media-2")

    args = Namespace(
        source_file=[str(source_1), str(source_2)],
        source_file_id=["slides_01", "slides_02"],
        media_file=[str(media_1), str(media_2)],
        media_id=["video_01", "video_02"],
    )

    inputs = run_phase_a.build_ingestion_inputs(args)

    assert len(inputs) == 2
    assert [item.source_file_id for item in inputs] == ["slides_01", "slides_02"]
    assert [item.media_id for item in inputs] == ["video_01", "video_02"]


def test_build_ingestion_inputs_rejects_mismatched_source_counts(tmp_path: Path) -> None:
    source_file = tmp_path / "slides.pdf"
    source_file.write_bytes(b"%PDF-1.7 fake")
    args = Namespace(
        source_file=[str(source_file)],
        source_file_id=["slides_01", "slides_02"],
        media_file=[],
        media_id=[],
    )

    with pytest.raises(ValueError):
        run_phase_a.build_ingestion_inputs(args)


def test_build_ingestion_inputs_rejects_partial_media_lists(tmp_path: Path) -> None:
    source_1 = tmp_path / "slides_01.pdf"
    source_2 = tmp_path / "slides_02.pdf"
    media_1 = tmp_path / "lecture_01.mp4"
    source_1.write_bytes(b"%PDF-1.7 fake 1")
    source_2.write_bytes(b"%PDF-1.7 fake 2")
    media_1.write_bytes(b"fake-media-1")

    args = Namespace(
        source_file=[str(source_1), str(source_2)],
        source_file_id=["slides_01", "slides_02"],
        media_file=[str(media_1)],
        media_id=["video_01"],
    )

    with pytest.raises(ValueError):
        run_phase_a.build_ingestion_inputs(args)
