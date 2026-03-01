from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.models import LLMCallMetadata, SCHEMA_VERSION, TOCData
from backend.app.prompts import render_prompt


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
        request_id: str | None = None

        # Prefer the modern Responses API when available.
        if hasattr(self._client, "responses"):
            response = self._client.responses.create(
                model=self.config.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
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
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
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
        rendered = render_prompt(
            name="toc_generation",
            version=self.config.prompt_version,
            doc_id=doc_id,
            schema_version=SCHEMA_VERSION,
            document_text=document_text,
        )
        raw_text, request_id = self._call_model(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
        )
        parsed = extract_first_json_object(raw_text)
        toc = TOCData.model_validate(parsed)
        llm_call = LLMCallMetadata(
            provider="openai",
            prompt_name=rendered["prompt_name"],
            prompt_version=rendered["prompt_version"],
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            request_id=request_id,
        )
        return TOCGenerationOutput(
            toc=toc,
            llm_call=llm_call,
            prompt_tag=rendered["prompt_tag"],
            prompt_checksum=rendered["prompt_checksum"],
            raw_response_text=raw_text,
        )


__all__ = [
    "OpenAITOCReasoningClient",
    "OpenAITOCReasoningConfig",
    "TOCGenerationOutput",
    "TOCReasoningClient",
    "extract_first_json_object",
]
