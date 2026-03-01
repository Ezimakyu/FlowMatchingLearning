from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PhaseAHyperparameters(StrictConfigModel):
    max_vision_chunk_tokens: int = Field(default=260, ge=64)
    max_transcript_chunk_tokens: int = Field(default=220, ge=64)
    include_transcript_chunks: bool = True


class TOCGenerationHyperparameters(StrictConfigModel):
    max_input_chars: int = Field(default=120000, ge=1000)
    model_test: str = "gpt-4.1-mini"
    model_demo: str = "gpt-5.1"
    prompt_version: str = "2026-02-28.v2"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4000, ge=256)


class IterationLoopHyperparameters(StrictConfigModel):
    top_k_historical_matches: int = Field(default=12, ge=1)
    similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    similarity_fallback_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    edge_acceptance_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    retrieval_overfetch_multiplier: int = Field(default=4, ge=1, le=10)
    max_section_chars_per_call: int = Field(default=30000, ge=1000)
    max_sections_to_parse: int = Field(default=0, ge=0)
    max_llm_concepts_per_section: int = Field(default=6, ge=1, le=20)
    max_state_nodes_in_context: int = Field(default=200, ge=10)
    max_historical_nodes_for_local_similarity: int = Field(default=400, ge=50)
    seed_core_nodes_from_toc: bool = True
    max_seed_core_nodes: int = Field(default=12, ge=1, le=50)
    freeze_node_set_after_seed: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> "IterationLoopHyperparameters":
        if self.similarity_fallback_threshold > self.similarity_threshold:
            raise ValueError(
                "similarity_fallback_threshold must be <= similarity_threshold."
            )
        return self


class PhaseBHyperparameters(StrictConfigModel):
    toc_generation: TOCGenerationHyperparameters = Field(
        default_factory=TOCGenerationHyperparameters
    )
    iteration_loop: IterationLoopHyperparameters = Field(
        default_factory=IterationLoopHyperparameters
    )


class Hyperparameters(StrictConfigModel):
    phase_a: PhaseAHyperparameters = Field(default_factory=PhaseAHyperparameters)
    phase_b: PhaseBHyperparameters = Field(default_factory=PhaseBHyperparameters)


def _strip_comment_fields(value):
    if isinstance(value, dict):
        cleaned: dict = {}
        for key, item in value.items():
            if key.startswith("_comment"):
                continue
            cleaned[key] = _strip_comment_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_comment_fields(item) for item in value]
    return value


def _extract_nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        nested = _extract_nested_model(arg)
        if nested is not None:
            return nested
    return None


def _strip_unknown_fields_for_model(model_cls: type[BaseModel], value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        field = model_cls.model_fields.get(key)
        if field is None:
            continue
        nested_model = _extract_nested_model(field.annotation)
        if nested_model is None:
            cleaned[key] = item
            continue
        cleaned[key] = _strip_unknown_fields_for_model(nested_model, item)
    return cleaned


def load_hyperparameters(path: str | Path) -> Hyperparameters:
    logger = logging.getLogger(__name__)
    target = Path(path)
    if not target.exists():
        return Hyperparameters()

    raw_payload = json.loads(target.read_text(encoding="utf-8"))
    normalized = _strip_comment_fields(raw_payload)
    try:
        return Hyperparameters.model_validate(normalized)
    except ValidationError as exc:
        if not exc.errors() or any(err.get("type") != "extra_forbidden" for err in exc.errors()):
            raise
        # Compatibility fallback: ignore unknown fields rather than failing hard.
        filtered = _strip_unknown_fields_for_model(Hyperparameters, normalized)
        dropped_count = len(exc.errors())
        logger.warning(
            "hyperparameters.ignored_unknown_fields path=%s dropped=%d",
            target,
            dropped_count,
        )
        return Hyperparameters.model_validate(filtered)


__all__ = [
    "Hyperparameters",
    "IterationLoopHyperparameters",
    "PhaseAHyperparameters",
    "PhaseBHyperparameters",
    "TOCGenerationHyperparameters",
    "load_hyperparameters",
]
