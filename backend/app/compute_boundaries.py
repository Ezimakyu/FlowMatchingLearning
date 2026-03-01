from __future__ import annotations

from typing import Final

from .models import ComputeProvider


PIPELINE_BOUNDARIES: Final[dict[str, ComputeProvider]] = {
    # Phase A ingestion (Modal serverless GPU functions)
    "ingestion.transcription": ComputeProvider.modal,
    "ingestion.vision_extraction": ComputeProvider.modal,
    "ingestion.embedding": ComputeProvider.modal,
    # Storage and retrieval (Actian VectorDB)
    "storage.write_chunks_and_vectors": ComputeProvider.actian,
    "storage.similarity_search": ComputeProvider.actian,
    # Inner parsing loop reasoning (OpenAI API only)
    "reasoning.toc_generation": ComputeProvider.openai,
    "reasoning.section_concept_extraction": ComputeProvider.openai,
    "reasoning.edge_validation": ComputeProvider.openai,
    # Export layer (Supermemory API)
    "export.supermemory": ComputeProvider.supermemory,
}


class BoundaryViolationError(ValueError):
    """Raised when a pipeline stage uses an invalid compute provider."""


def required_provider_for_stage(stage: str) -> ComputeProvider:
    if stage not in PIPELINE_BOUNDARIES:
        known_stages = ", ".join(sorted(PIPELINE_BOUNDARIES))
        raise BoundaryViolationError(
            f"Unknown stage '{stage}'. Known stages: {known_stages}"
        )
    return PIPELINE_BOUNDARIES[stage]


def assert_compute_boundary(stage: str, provider: str | ComputeProvider) -> None:
    required_provider = required_provider_for_stage(stage)
    try:
        actual_provider = (
            provider if isinstance(provider, ComputeProvider) else ComputeProvider(provider)
        )
    except ValueError as exc:
        valid_providers = ", ".join(member.value for member in ComputeProvider)
        raise BoundaryViolationError(
            f"Unknown provider '{provider}'. Valid providers: {valid_providers}"
        ) from exc
    if actual_provider != required_provider:
        raise BoundaryViolationError(
            f"Stage '{stage}' requires provider '{required_provider.value}', "
            f"got '{actual_provider.value}'."
        )


__all__ = [
    "BoundaryViolationError",
    "PIPELINE_BOUNDARIES",
    "assert_compute_boundary",
    "required_provider_for_stage",
]
