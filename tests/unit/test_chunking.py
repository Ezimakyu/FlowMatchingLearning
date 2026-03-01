from backend.app.ingestion.chunking import (
    build_chunking_result,
    chunk_text_units,
    chunk_transcription_result,
    chunk_vision_extraction_result,
    estimate_token_count,
    split_logical_units,
)
from backend.app.models import (
    TranscriptSegment,
    TranscriptionResult,
    VisionExtractionResult,
    VisionPageExtraction,
)


def test_split_logical_units_collapses_blank_lines() -> None:
    text = "Intro paragraph.\n\n\nSecond paragraph.\n\nThird paragraph."
    units = split_logical_units(text)
    assert units == ["Intro paragraph.", "Second paragraph.", "Third paragraph."]


def test_chunk_text_units_respects_order_and_token_limit() -> None:
    units = [
        "Linear algebra foundations and vectors.",
        "Matrix multiplication and inversion details.",
        "Eigenvalues and eigenvectors interpretation.",
    ]
    chunks = chunk_text_units(
        doc_id="doc_1",
        units=units,
        source_type="vision_text",
        max_tokens=8,
    )
    assert len(chunks) >= 2
    assert [chunk.order for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.token_estimate >= 1


def test_chunk_transcription_result_carries_time_bounds() -> None:
    transcription = TranscriptionResult(
        doc_id="doc_2",
        media_id="lecture.mp3",
        model_name="test-whisper",
        segments=[
            TranscriptSegment(
                segment_id="seg_1",
                start_seconds=0.0,
                end_seconds=3.0,
                text="A derivative describes local rate of change.",
            ),
            TranscriptSegment(
                segment_id="seg_2",
                start_seconds=3.0,
                end_seconds=8.0,
                text="It can be derived from the limit of a difference quotient.",
            ),
        ],
        transcript_text="A derivative describes local rate of change. It can be derived...",
    )

    chunks = chunk_transcription_result(transcription, max_tokens=200)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_type == "transcript"
    assert chunk.source_time_start_seconds == 0.0
    assert chunk.source_time_end_seconds == 8.0


def test_chunk_vision_result_emits_text_and_description_chunks() -> None:
    vision = VisionExtractionResult(
        doc_id="doc_3",
        source_file_id="slides.pdf",
        model_name="test-vlm",
        pages=[
            VisionPageExtraction(
                page_number=1,
                raw_text="Gradient descent updates parameters iteratively.",
                image_descriptions=["Diagram of a loss surface with contour lines."],
                chunk_ids=[],
            )
        ],
    )

    chunks = chunk_vision_extraction_result(vision, max_tokens=200)
    source_types = {chunk.source_type for chunk in chunks}
    assert "vision_text" in source_types
    assert "vision_image_description" in source_types


def test_build_chunking_result_validates_sort_order() -> None:
    chunks = chunk_text_units(
        doc_id="doc_4",
        units=["A short segment.", "Another short segment."],
        source_type="manual",
        max_tokens=200,
    )
    result = build_chunking_result("doc_4", chunks)
    assert result.doc_id == "doc_4"
    assert len(result.chunks) == len(chunks)
    assert estimate_token_count(result.chunks[0].text) >= 1
