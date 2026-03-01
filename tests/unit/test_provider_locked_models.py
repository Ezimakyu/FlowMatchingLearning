import pytest
from pydantic import ValidationError

from backend.app.models import (
    LLMCallMetadata,
    SimilarityMatch,
)


def test_llm_call_metadata_provider_locked_to_openai() -> None:
    metadata = LLMCallMetadata(
        prompt_name="toc_generation",
        prompt_version="2026-02-28.v2",
        model="gpt-5-mini",
    )
    assert metadata.provider == "openai"


def test_similarity_match_provider_locked_to_actian() -> None:
    match = SimilarityMatch(
        historical_concept_id="limits",
        similarity=0.82,
    )
    assert match.retrieval_provider == "actian"


def test_llm_call_metadata_rejects_non_openai_provider() -> None:
    with pytest.raises(ValidationError):
        LLMCallMetadata(
            provider="anthropic",  # type: ignore[arg-type]
            prompt_name="toc_generation",
            prompt_version="2026-02-28.v2",
            model="gpt-5-mini",
        )
