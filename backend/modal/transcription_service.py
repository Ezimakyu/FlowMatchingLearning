from __future__ import annotations

import logging
import os
import tempfile

import modal

from backend.app.models import TranscriptSegment, TranscriptionResult

app = modal.App("phase-a-transcription")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transcription_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install("faster-whisper==1.1.1", "pydantic")
    .add_local_python_source("backend")
)


@app.function(
    image=transcription_image,
    gpu="T4",
    timeout=60 * 20,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=2.0),
)
def transcribe_media(
    *,
    media_bytes: bytes,
    doc_id: str,
    media_id: str,
    model_name: str = "large-v3",
    language: str | None = None,
) -> dict:
    from faster_whisper import WhisperModel

    logger.info("transcription.start doc_id=%s media_id=%s model=%s", doc_id, media_id, model_name)
    suffix = os.path.splitext(media_id)[1] or ".bin"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(media_bytes)
            temp_path = tmp.name

        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        raw_segments, _ = model.transcribe(
            temp_path,
            language=language,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segments: list[TranscriptSegment] = []
        transcript_parts: list[str] = []
        for index, segment in enumerate(raw_segments):
            segment_text = segment.text.strip()
            if not segment_text:
                continue
            segments.append(
                TranscriptSegment(
                    segment_id=f"{doc_id}:seg:{index:05d}",
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=segment_text,
                )
            )
            transcript_parts.append(segment_text)

        result = TranscriptionResult(
            doc_id=doc_id,
            media_id=media_id,
            model_name=model_name,
            language=language,
            segments=segments,
            transcript_text=" ".join(transcript_parts).strip(),
            metadata={"segment_count": len(segments)},
        )
        logger.info("transcription.finish doc_id=%s segments=%d", doc_id, len(segments))
        return result.model_dump(mode="json")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.local_entrypoint()
def main(input_path: str, doc_id: str, media_id: str) -> None:
    with open(input_path, "rb") as file_handle:
        media_bytes = file_handle.read()
    payload = transcribe_media.remote(media_bytes=media_bytes, doc_id=doc_id, media_id=media_id)
    print(payload)
