from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.models import LLMCallMetadata, SCHEMA_VERSION, TOCData
from backend.app.prompts import render_prompt


class TOCSectionStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    order: int = Field(ge=0)
    chunk_ids: list[str]
    summary: str | None
    key_terms: list[str]
    children: list["TOCSectionStructuredOutput"]


TOCSectionStructuredOutput.model_rebuild()


class TOCStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    sections: list[TOCSectionStructuredOutput]


def normalize_toc_payload(parsed: dict[str, Any], *, doc_id: str) -> dict[str, Any]:
    normalized = dict(parsed)
    if "doc_id" not in normalized or not str(normalized.get("doc_id", "")).strip():
        normalized["doc_id"] = doc_id
    if "schema_version" not in normalized:
        normalized["schema_version"] = SCHEMA_VERSION
    return normalized


def build_toc_json_schema() -> dict[str, Any]:
    return TOCStructuredOutput.model_json_schema()


def build_responses_json_schema_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "toc_data",
            "schema": build_toc_json_schema(),
            "strict": True,
        }
    }


def build_chat_completions_json_schema_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "toc_data",
            "schema": build_toc_json_schema(),
            "strict": True,
        },
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


def extract_first_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response; expected JSON object.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("Model response did not contain a JSON object.")
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


@dataclass(frozen=True)
class TOCGenerationOutput:
    toc: TOCData
    llm_call: LLMCallMetadata
    prompt_tag: str
    prompt_checksum: str
    raw_response_text: str


class TOCReasoningClient(Protocol):
    def generate_toc(
        self,
        *,
        doc_id: str,
        document_text: str,
    ) -> TOCGenerationOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenAITOCReasoningConfig:
    model: str = "gpt-4.1-mini"
    prompt_version: str = "2026-02-28.v2"
    temperature: float = 0.0
    max_output_tokens: int = 4000

    @classmethod
    def from_env(cls) -> "OpenAITOCReasoningConfig":
        return cls(
            model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1-mini"),
            prompt_version=os.getenv("TOC_PROMPT_VERSION", "2026-02-28.v2"),
            temperature=float(os.getenv("OPENAI_REASONING_TEMPERATURE", "0.0")),
            max_output_tokens=int(os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "4000")),
        )


class OpenAITOCReasoningClient:
    def __init__(
        self,
        *,
        config: OpenAITOCReasoningConfig | None = None,
        provider: str = "openai",
    ) -> None:
        assert_compute_boundary("reasoning.toc_generation", provider)
        self.provider = provider
        self.config = config or OpenAITOCReasoningConfig.from_env()
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK is required for Phase B reasoning. Install with `pip install openai`."
            ) from exc
        return OpenAI()

    def _call_model(self, *, system_prompt: str, user_prompt: str) -> tuple[str, str | None]:
        logger = logging.getLogger(__name__)
        request_id: str | None = None

        # Prefer the modern Responses API when available.
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
                    text=build_responses_json_schema_text_config(),
                )
                logger.info(
                    "phase_b_toc.structured_output api=responses mode=json_schema strict=true"
                )
            except TypeError:
                logger.warning(
                    "phase_b_toc.structured_output_fallback api=responses reason=type_error"
                )
                response = self._client.responses.create(**create_kwargs)
            except Exception as exc:
                if not _looks_like_schema_capability_error(exc):
                    raise
                logger.warning(
                    "phase_b_toc.structured_output_fallback api=responses reason=%s",
                    exc.__class__.__name__,
                )
                response = self._client.responses.create(**create_kwargs)
            request_id = getattr(response, "id", None)
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text, request_id

            # Fallback: stitch text from output blocks when output_text is unavailable.
            output_blocks = getattr(response, "output", []) or []
            parts: list[str] = []
            for block in output_blocks:
                content_items = getattr(block, "content", []) or []
                for content in content_items:
                    value = getattr(content, "text", None)
                    if value:
                        parts.append(value)
            return "\n".join(parts).strip(), request_id

        # Compatibility fallback: chat.completions API.
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
                response_format=build_chat_completions_json_schema_response_format(),
            )
            logger.info(
                "phase_b_toc.structured_output api=chat_completions mode=json_schema strict=true"
            )
        except TypeError:
            logger.warning(
                "phase_b_toc.structured_output_fallback api=chat_completions reason=type_error"
            )
            response = self._client.chat.completions.create(
                **chat_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if not _looks_like_schema_capability_error(exc):
                raise
            logger.warning(
                "phase_b_toc.structured_output_fallback api=chat_completions reason=%s",
                exc.__class__.__name__,
            )
            response = self._client.chat.completions.create(
                **chat_kwargs,
                response_format={"type": "json_object"},
            )
        request_id = getattr(response, "id", None)
        message = response.choices[0].message
        content = message.content or ""
        return content, request_id

    def generate_toc(
        self,
        *,
        doc_id: str,
        document_text: str,
    ) -> TOCGenerationOutput:
        logger = logging.getLogger(__name__)
        rendered = render_prompt(
            name="toc_generation",
            version=self.config.prompt_version,
            doc_id=doc_id,
            schema_version=SCHEMA_VERSION,
            document_text=document_text,
        )
        logger.info(
            "phase_b_toc.reasoning_call_start doc_id=%s model=%s prompt=%s",
            doc_id,
            self.config.model,
            rendered["prompt_tag"],
        )
        raw_text, request_id = self._call_model(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
        )
        logger.info(
            "phase_b_toc.reasoning_call_finish doc_id=%s request_id=%s response_chars=%d",
            doc_id,
            request_id,
            len(raw_text),
        )
        parsed = extract_first_json_object(raw_text)
        normalized_payload = normalize_toc_payload(parsed, doc_id=doc_id)
        if "doc_id" not in parsed:
            logger.warning("phase_b_toc.reasoning_missing_doc_id using_fallback=%s", doc_id)
        try:
            toc = TOCData.model_validate(normalized_payload)
        except ValidationError:
            logger.exception(
                "phase_b_toc.reasoning_parse_error doc_id=%s request_id=%s keys=%s",
                doc_id,
                request_id,
                sorted(normalized_payload.keys()),
            )
            logger.error(
                "phase_b_toc.reasoning_raw_response_excerpt doc_id=%s excerpt=%r",
                doc_id,
                raw_text[:2000],
            )
            raise
        llm_call = LLMCallMetadata(
            provider="openai",
            prompt_name=rendered["prompt_name"],
            prompt_version=rendered["prompt_version"],
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            request_id=request_id,
        )
        output = TOCGenerationOutput(
            toc=toc,
            llm_call=llm_call,
            prompt_tag=rendered["prompt_tag"],
            prompt_checksum=rendered["prompt_checksum"],
            raw_response_text=raw_text,
        )
        logger.info(
            "phase_b_toc.reasoning_parse_finish doc_id=%s top_level_sections=%d",
            doc_id,
            len(toc.sections),
        )
        return output


__all__ = [
    "build_chat_completions_json_schema_response_format",
    "build_responses_json_schema_text_config",
    "build_toc_json_schema",
    "OpenAITOCReasoningClient",
    "OpenAITOCReasoningConfig",
    "TOCGenerationOutput",
    "TOCReasoningClient",
    "extract_first_json_object",
    "normalize_toc_payload",
]
