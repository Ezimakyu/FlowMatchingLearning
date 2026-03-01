from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.models import (
    LLMCallMetadata,
    SCHEMA_VERSION,
    SectionConcept,
    SectionEdgeCandidate,
)
from backend.app.prompts import render_prompt
from backend.app.reasoning.toc_reasoning import extract_first_json_object


class SectionConceptStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    aliases: list[str]
    source_chunk_ids: list[str]
    evidence_text: str | None
    confidence: float


class SectionConceptExtractionStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concepts: list[SectionConceptStructuredOutput]
    warnings: list[str]


class ConceptEdgeEvidenceStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    historical_doc_id: str | None
    current_doc_id: str | None
    historical_chunk_ids: list[str]
    current_chunk_ids: list[str]


class SectionEdgeCandidateStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_concept_id: str = Field(min_length=1)
    target_concept_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    accepted: bool
    confidence: float
    explanation: str = Field(min_length=1)
    evidence: ConceptEdgeEvidenceStructuredOutput


def _schema_text_config(*, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def _looks_like_schema_capability_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "json_schema",
        "response_format",
        "text.format",
        "unknown parameter",
        "unsupported",
    )
    return any(marker in message for marker in markers)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    output_blocks = getattr(response, "output", []) or []
    parts: list[str] = []
    for block in output_blocks:
        content_items = getattr(block, "content", []) or []
        for content in content_items:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def _sanitize_identifier(value: str, *, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "_", value.strip().lower()).strip("_")
    if not cleaned:
        cleaned = prefix
    return cleaned


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class SectionConceptExtractionOutput:
    concepts: list[SectionConcept]
    warnings: list[str]
    llm_call: LLMCallMetadata
    prompt_tag: str
    prompt_checksum: str
    raw_response_text: str


@dataclass(frozen=True)
class EdgeValidationOutput:
    candidate: SectionEdgeCandidate
    llm_call: LLMCallMetadata
    prompt_tag: str
    prompt_checksum: str
    raw_response_text: str


class SectionReasoningClient(Protocol):
    def extract_section_concepts(
        self,
        *,
        doc_id: str,
        section_id: str,
        section_title: str,
        section_text: str,
        rolling_state_json: str,
    ) -> SectionConceptExtractionOutput:
        raise NotImplementedError

    def validate_edge_candidate(
        self,
        *,
        new_concept_json: str,
        historical_concept_json: str,
        supporting_evidence_json: str,
    ) -> EdgeValidationOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenAISectionReasoningConfig:
    model: str = "gpt-4.1-mini"
    section_prompt_version: str = "2026-03-01.v3"
    edge_prompt_version: str = "2026-03-01.v3"
    temperature: float = 0.0
    max_output_tokens: int = 2500

    @classmethod
    def from_env(cls) -> "OpenAISectionReasoningConfig":
        return cls(
            model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1-mini"),
            section_prompt_version=os.getenv(
                "SECTION_CONCEPT_PROMPT_VERSION", "2026-03-01.v3"
            ),
            edge_prompt_version=os.getenv("EDGE_VALIDATION_PROMPT_VERSION", "2026-03-01.v3"),
            temperature=float(os.getenv("OPENAI_REASONING_TEMPERATURE", "0.0")),
            max_output_tokens=int(os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "2500")),
        )


class OpenAISectionReasoningClient:
    def __init__(
        self,
        *,
        config: OpenAISectionReasoningConfig | None = None,
        provider: str = "openai",
    ) -> None:
        assert_compute_boundary("reasoning.section_concept_extraction", provider)
        assert_compute_boundary("reasoning.edge_validation", provider)
        self.provider = provider
        self.config = config or OpenAISectionReasoningConfig.from_env()
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK is required for Phase B section reasoning. Install with `pip install openai`."
            ) from exc
        return OpenAI()

    def _call_model_with_schema(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[str, str | None]:
        logger = logging.getLogger(__name__)
        request_id: str | None = None

        if hasattr(self._client, "responses"):
            create_kwargs = {
                "model": self.config.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            }
            try:
                response = self._client.responses.create(
                    **create_kwargs,
                    text=_schema_text_config(name=schema_name, schema=schema),
                )
                logger.info(
                    "phase_b.structured_output api=responses mode=json_schema strict=true schema=%s",
                    schema_name,
                )
            except TypeError:
                logger.warning(
                    "phase_b.structured_output_fallback api=responses reason=type_error schema=%s",
                    schema_name,
                )
                response = self._client.responses.create(**create_kwargs)
            except Exception as exc:
                if not _looks_like_schema_capability_error(exc):
                    raise
                logger.warning(
                    "phase_b.structured_output_fallback api=responses reason=%s schema=%s",
                    exc.__class__.__name__,
                    schema_name,
                )
                response = self._client.responses.create(**create_kwargs)
            request_id = getattr(response, "id", None)
            return _extract_response_text(response), request_id

        chat_kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        try:
            response = self._client.chat.completions.create(
                **chat_kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            logger.info(
                "phase_b.structured_output api=chat_completions mode=json_schema strict=true schema=%s",
                schema_name,
            )
        except TypeError:
            logger.warning(
                "phase_b.structured_output_fallback api=chat_completions reason=type_error schema=%s",
                schema_name,
            )
            response = self._client.chat.completions.create(
                **chat_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if not _looks_like_schema_capability_error(exc):
                raise
            logger.warning(
                "phase_b.structured_output_fallback api=chat_completions reason=%s schema=%s",
                exc.__class__.__name__,
                schema_name,
            )
            response = self._client.chat.completions.create(
                **chat_kwargs,
                response_format={"type": "json_object"},
            )
        request_id = getattr(response, "id", None)
        message = response.choices[0].message
        content = message.content or ""
        return content, request_id

    def extract_section_concepts(
        self,
        *,
        doc_id: str,
        section_id: str,
        section_title: str,
        section_text: str,
        rolling_state_json: str,
    ) -> SectionConceptExtractionOutput:
        logger = logging.getLogger(__name__)
        rendered = render_prompt(
            name="section_concept_extraction",
            version=self.config.section_prompt_version,
            doc_id=doc_id,
            section_id=section_id,
            section_title=section_title,
            schema_version=SCHEMA_VERSION,
            rolling_state_json=rolling_state_json,
            section_text=section_text,
        )
        logger.info(
            "phase_b.section_concept_call_start doc_id=%s section_id=%s model=%s prompt=%s",
            doc_id,
            section_id,
            self.config.model,
            rendered["prompt_tag"],
        )
        raw_text, request_id = self._call_model_with_schema(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
            schema_name="section_concept_extraction_output",
            schema=SectionConceptExtractionStructuredOutput.model_json_schema(),
        )
        parsed = extract_first_json_object(raw_text)
        try:
            structured = SectionConceptExtractionStructuredOutput.model_validate(parsed)
        except ValidationError:
            logger.exception(
                "phase_b.section_concept_parse_error doc_id=%s section_id=%s request_id=%s",
                doc_id,
                section_id,
                request_id,
            )
            logger.error(
                "phase_b.section_concept_raw_response_excerpt doc_id=%s section_id=%s excerpt=%r",
                doc_id,
                section_id,
                raw_text[:2000],
            )
            raise

        concepts: list[SectionConcept] = []
        for item in structured.concepts:
            concept_id = _sanitize_identifier(item.concept_id or item.label, prefix="concept")
            concepts.append(
                SectionConcept(
                    concept_id=concept_id,
                    label=item.label,
                    summary=item.summary,
                    aliases=item.aliases,
                    source_chunk_ids=item.source_chunk_ids,
                    evidence_text=item.evidence_text,
                    confidence=_clamp01(float(item.confidence)),
                )
            )

        llm_call = LLMCallMetadata(
            provider="openai",
            prompt_name=rendered["prompt_name"],
            prompt_version=rendered["prompt_version"],
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            request_id=request_id,
        )
        logger.info(
            "phase_b.section_concept_call_finish doc_id=%s section_id=%s concepts=%d",
            doc_id,
            section_id,
            len(concepts),
        )
        return SectionConceptExtractionOutput(
            concepts=concepts,
            warnings=[item.strip() for item in structured.warnings if item.strip()],
            llm_call=llm_call,
            prompt_tag=rendered["prompt_tag"],
            prompt_checksum=rendered["prompt_checksum"],
            raw_response_text=raw_text,
        )

    def validate_edge_candidate(
        self,
        *,
        new_concept_json: str,
        historical_concept_json: str,
        supporting_evidence_json: str,
    ) -> EdgeValidationOutput:
        logger = logging.getLogger(__name__)
        rendered = render_prompt(
            name="edge_validation",
            version=self.config.edge_prompt_version,
            schema_version=SCHEMA_VERSION,
            new_concept_json=new_concept_json,
            historical_concept_json=historical_concept_json,
            supporting_evidence_json=supporting_evidence_json,
        )
        raw_text, request_id = self._call_model_with_schema(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
            schema_name="edge_validation_output",
            schema=SectionEdgeCandidateStructuredOutput.model_json_schema(),
        )
        parsed = extract_first_json_object(raw_text)
        try:
            structured = SectionEdgeCandidateStructuredOutput.model_validate(parsed)
        except ValidationError:
            logger.exception("phase_b.edge_validation_parse_error request_id=%s", request_id)
            logger.error("phase_b.edge_validation_raw_response_excerpt excerpt=%r", raw_text[:2000])
            raise

        candidate = SectionEdgeCandidate(
            source_concept_id=_sanitize_identifier(structured.source_concept_id, prefix="concept"),
            target_concept_id=_sanitize_identifier(structured.target_concept_id, prefix="concept"),
            relation="prerequisite_for",
            accepted=bool(structured.accepted),
            confidence=_clamp01(float(structured.confidence)),
            explanation=structured.explanation,
            evidence={
                "historical_doc_id": structured.evidence.historical_doc_id,
                "current_doc_id": structured.evidence.current_doc_id,
                "historical_chunk_ids": structured.evidence.historical_chunk_ids,
                "current_chunk_ids": structured.evidence.current_chunk_ids,
            },
        )
        llm_call = LLMCallMetadata(
            provider="openai",
            prompt_name=rendered["prompt_name"],
            prompt_version=rendered["prompt_version"],
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            request_id=request_id,
        )
        return EdgeValidationOutput(
            candidate=candidate,
            llm_call=llm_call,
            prompt_tag=rendered["prompt_tag"],
            prompt_checksum=rendered["prompt_checksum"],
            raw_response_text=raw_text,
        )


__all__ = [
    "EdgeValidationOutput",
    "OpenAISectionReasoningClient",
    "OpenAISectionReasoningConfig",
    "SectionConceptExtractionOutput",
    "SectionReasoningClient",
]
