from __future__ import annotations

import re
from typing import Iterable, Literal

from backend.app.models import (
    ChunkingResult,
    RawTextChunk,
    TranscriptSegment,
    TranscriptionResult,
    VisionExtractionResult,
)

WHITESPACE_RE = re.compile(r"\s+")
SourceType = Literal["transcript", "vision_text", "vision_image_description", "manual"]


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def estimate_token_count(text: str) -> int:
    # Fast heuristic: most English words are roughly 1.3 tokens.
    words = [token for token in re.split(r"\s+", text.strip()) if token]
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


def split_logical_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    raw_units = re.split(r"\n{2,}", normalized)
    units: list[str] = []
    for raw_unit in raw_units:
        unit = normalize_whitespace(raw_unit)
        if unit:
            units.append(unit)
    return units


def _make_chunk_id(doc_id: str, source_type: SourceType, order: int) -> str:
    safe_doc = re.sub(r"[^A-Za-z0-9._:-]", "_", doc_id)
    return f"{safe_doc}:{source_type}:{order:05d}"


def chunk_text_units(
    *,
    doc_id: str,
    units: Iterable[str],
    source_type: SourceType,
    start_order: int = 0,
    max_tokens: int = 240,
    section_hint: str | None = None,
    source_page: int | None = None,
) -> list[RawTextChunk]:
    if max_tokens < 8:
        raise ValueError("max_tokens must be at least 8.")

    chunks: list[RawTextChunk] = []
    current_order = start_order
    current_parts: list[str] = []
    current_token_count = 0

    for unit in units:
        unit_text = normalize_whitespace(unit)
        if not unit_text:
            continue
        unit_tokens = estimate_token_count(unit_text)
        if unit_tokens == 0:
            continue

        would_overflow = current_parts and current_token_count + unit_tokens > max_tokens
        if would_overflow:
            chunk_text = " ".join(current_parts)
            chunks.append(
                RawTextChunk(
                    chunk_id=_make_chunk_id(doc_id=doc_id, source_type=source_type, order=current_order),
                    doc_id=doc_id,
                    source_type=source_type,
                    order=current_order,
                    text=chunk_text,
                    token_estimate=estimate_token_count(chunk_text),
                    section_hint=section_hint,
                    source_page=source_page,
                )
            )
            current_order += 1
            current_parts = [unit_text]
            current_token_count = unit_tokens
            continue

        current_parts.append(unit_text)
        current_token_count += unit_tokens

    if current_parts:
        chunk_text = " ".join(current_parts)
        chunks.append(
            RawTextChunk(
                chunk_id=_make_chunk_id(doc_id=doc_id, source_type=source_type, order=current_order),
                doc_id=doc_id,
                source_type=source_type,
                order=current_order,
                text=chunk_text,
                token_estimate=estimate_token_count(chunk_text),
                section_hint=section_hint,
                source_page=source_page,
            )
        )

    return chunks


def chunk_transcription_result(
    transcription: TranscriptionResult,
    *,
    max_tokens: int = 220,
    start_order: int = 0,
) -> list[RawTextChunk]:
    chunks: list[RawTextChunk] = []
    current_text_parts: list[str] = []
    current_token_count = 0
    current_start: float | None = None
    current_end: float | None = None
    order = start_order

    segments: list[TranscriptSegment] = transcription.segments

    for segment in segments:
        segment_text = normalize_whitespace(segment.text)
        if not segment_text:
            continue
        segment_tokens = estimate_token_count(segment_text)
        if segment_tokens == 0:
            continue

        if current_start is None:
            current_start = segment.start_seconds
        current_end = segment.end_seconds

        if current_text_parts and current_token_count + segment_tokens > max_tokens:
            merged = " ".join(current_text_parts)
            chunks.append(
                RawTextChunk(
                    chunk_id=_make_chunk_id(doc_id=transcription.doc_id, source_type="transcript", order=order),
                    doc_id=transcription.doc_id,
                    source_type="transcript",
                    order=order,
                    text=merged,
                    token_estimate=estimate_token_count(merged),
                    source_time_start_seconds=current_start,
                    source_time_end_seconds=current_end,
                )
            )
            order += 1
            current_text_parts = [segment_text]
            current_token_count = segment_tokens
            current_start = segment.start_seconds
            current_end = segment.end_seconds
            continue

        current_text_parts.append(segment_text)
        current_token_count += segment_tokens

    if current_text_parts:
        merged = " ".join(current_text_parts)
        chunks.append(
            RawTextChunk(
                chunk_id=_make_chunk_id(doc_id=transcription.doc_id, source_type="transcript", order=order),
                doc_id=transcription.doc_id,
                source_type="transcript",
                order=order,
                text=merged,
                token_estimate=estimate_token_count(merged),
                source_time_start_seconds=current_start,
                source_time_end_seconds=current_end,
            )
        )

    return chunks


def chunk_vision_extraction_result(
    vision_result: VisionExtractionResult,
    *,
    max_tokens: int = 260,
    start_order: int = 0,
) -> list[RawTextChunk]:
    chunks: list[RawTextChunk] = []
    order = start_order

    for page in vision_result.pages:
        text_units = split_logical_units(page.raw_text)
        page_chunks = chunk_text_units(
            doc_id=vision_result.doc_id,
            units=text_units,
            source_type="vision_text",
            start_order=order,
            max_tokens=max_tokens,
            source_page=page.page_number,
        )
        chunks.extend(page_chunks)
        order += len(page_chunks)

        description_chunks = chunk_text_units(
            doc_id=vision_result.doc_id,
            units=page.image_descriptions,
            source_type="vision_image_description",
            start_order=order,
            max_tokens=max_tokens,
            source_page=page.page_number,
        )
        chunks.extend(description_chunks)
        order += len(description_chunks)

    return chunks


def build_chunking_result(doc_id: str, chunks: list[RawTextChunk]) -> ChunkingResult:
    ordered = sorted(chunks, key=lambda chunk: chunk.order)
    return ChunkingResult(doc_id=doc_id, chunks=ordered)


__all__ = [
    "build_chunking_result",
    "chunk_text_units",
    "chunk_transcription_result",
    "chunk_vision_extraction_result",
    "estimate_token_count",
    "normalize_whitespace",
    "split_logical_units",
]
