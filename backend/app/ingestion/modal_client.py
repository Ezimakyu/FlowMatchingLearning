from __future__ import annotations

from dataclasses import dataclass

from backend.app.models import (
    EmbeddingBatchResult,
    RawTextChunk,
    TranscriptionResult,
    VisionExtractionResult,
)


@dataclass(frozen=True)
class ModalFunctionConfig:
    transcription_app_name: str = "phase-a-transcription"
    transcription_function_name: str = "transcribe_media"
    vision_app_name: str = "phase-a-vision-extraction"
    vision_function_name: str = "extract_document_vision"
    embedding_app_name: str = "phase-a-embedding"
    embedding_function_name: str = "embed_chunks"


class ModalRemoteIngestionClient:
    def __init__(self, config: ModalFunctionConfig | None = None) -> None:
        self.config = config or ModalFunctionConfig()
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError(
                "modal SDK is required for ModalRemoteIngestionClient. Install with `pip install modal`."
            ) from exc

        self._transcribe_fn = modal.Function.from_name(
            self.config.transcription_app_name,
            self.config.transcription_function_name,
        )
        self._vision_fn = modal.Function.from_name(
            self.config.vision_app_name,
            self.config.vision_function_name,
        )
        self._embedding_fn = modal.Function.from_name(
            self.config.embedding_app_name,
            self.config.embedding_function_name,
        )

    def transcribe_media(
        self,
        *,
        media_bytes: bytes,
        doc_id: str,
        media_id: str,
    ) -> TranscriptionResult:
        payload = self._transcribe_fn.remote(
            media_bytes=media_bytes,
            doc_id=doc_id,
            media_id=media_id,
        )
        return TranscriptionResult.model_validate(payload)

    def extract_vision(
        self,
        *,
        file_bytes: bytes,
        doc_id: str,
        source_file_id: str,
    ) -> VisionExtractionResult:
        payload = self._vision_fn.remote(
            file_bytes=file_bytes,
            doc_id=doc_id,
            source_file_id=source_file_id,
        )
        return VisionExtractionResult.model_validate(payload)

    def embed_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[RawTextChunk],
    ) -> EmbeddingBatchResult:
        payload = self._embedding_fn.remote(
            doc_id=doc_id,
            chunks=[chunk.model_dump(mode="json") for chunk in chunks],
        )
        return EmbeddingBatchResult.model_validate(payload)


__all__ = [
    "ModalFunctionConfig",
    "ModalRemoteIngestionClient",
]
