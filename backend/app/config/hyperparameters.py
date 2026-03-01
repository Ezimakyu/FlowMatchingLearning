from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
    top_k_historical_matches: int = Field(default=8, ge=1)
    similarity_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    edge_acceptance_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_section_chars_per_call: int = Field(default=30000, ge=1000)
    max_state_nodes_in_context: int = Field(default=200, ge=10)


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


def load_hyperparameters(path: str | Path) -> Hyperparameters:
    target = Path(path)
    if not target.exists():
        return Hyperparameters()

    raw_payload = json.loads(target.read_text(encoding="utf-8"))
    normalized = _strip_comment_fields(raw_payload)
    return Hyperparameters.model_validate(normalized)


__all__ = [
    "Hyperparameters",
    "IterationLoopHyperparameters",
    "PhaseAHyperparameters",
    "PhaseBHyperparameters",
    "TOCGenerationHyperparameters",
    "load_hyperparameters",
]
