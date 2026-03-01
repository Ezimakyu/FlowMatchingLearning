from __future__ import annotations

from dataclasses import dataclass
import logging

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.models import ChunkingResult
from backend.app.reasoning import TOCGenerationOutput, TOCReasoningClient


def build_toc_input_text(*, chunking: ChunkingResult, max_chars: int = 120_000) -> str:
    if max_chars < 100:
        raise ValueError("max_chars must be >= 100.")

    lines: list[str] = []
    total_chars = 0
    for chunk in chunking.chunks:
        prefix = f"[chunk_id={chunk.chunk_id} source={chunk.source_type} order={chunk.order}] "
        line = prefix + chunk.text.strip()
        projected = total_chars + len(line) + 1
        if projected > max_chars:
            lines.append("[TRUNCATED_FOR_CONTEXT_LIMIT]")
            break
        lines.append(line)
        total_chars = projected
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class PhaseBTOCConfig:
    max_input_chars: int = 120_000


class PhaseBTOCPipeline:
    def __init__(
        self,
        *,
        reasoning_client: TOCReasoningClient,
        config: PhaseBTOCConfig | None = None,
        reasoning_provider: str = "openai",
    ) -> None:
        self.reasoning_client = reasoning_client
        self.config = config or PhaseBTOCConfig()
        self.reasoning_provider = reasoning_provider

    def run(self, *, doc_id: str, chunking: ChunkingResult) -> TOCGenerationOutput:
        logger = logging.getLogger(__name__)
        if chunking.doc_id != doc_id:
            raise ValueError(f"doc_id mismatch: run={doc_id}, chunking={chunking.doc_id}")

        logger.info(
            "phase_b_toc.pipeline_start doc_id=%s chunks=%d",
            doc_id,
            len(chunking.chunks),
        )
        assert_compute_boundary("reasoning.toc_generation", self.reasoning_provider)
        document_text = build_toc_input_text(
            chunking=chunking,
            max_chars=self.config.max_input_chars,
        )
        if not document_text:
            raise ValueError("Chunking input is empty; cannot generate TOC.")
        logger.info(
            "phase_b_toc.input_built doc_id=%s chars=%d",
            doc_id,
            len(document_text),
        )
        output = self.reasoning_client.generate_toc(
            doc_id=doc_id,
            document_text=document_text,
        )
        logger.info(
            "phase_b_toc.pipeline_finish doc_id=%s top_level_sections=%d",
            doc_id,
            len(output.toc.sections),
        )
        return output


__all__ = [
    "PhaseBTOCConfig",
    "PhaseBTOCPipeline",
    "build_toc_input_text",
]
