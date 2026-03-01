import pytest

from backend.app.compute_boundaries import (
    BoundaryViolationError,
    assert_compute_boundary,
    required_provider_for_stage,
)


def test_required_provider_mapping() -> None:
    assert required_provider_for_stage("ingestion.embedding").value == "modal"
    assert required_provider_for_stage("reasoning.toc_generation").value == "openai"
    assert required_provider_for_stage("storage.similarity_search").value == "actian"


def test_assert_compute_boundary_accepts_valid_mapping() -> None:
    assert_compute_boundary("ingestion.transcription", "modal")
    assert_compute_boundary("reasoning.edge_validation", "openai")


def test_assert_compute_boundary_rejects_invalid_provider() -> None:
    with pytest.raises(BoundaryViolationError):
        assert_compute_boundary("ingestion.vision_extraction", "openai")


def test_assert_compute_boundary_rejects_unknown_provider() -> None:
    with pytest.raises(BoundaryViolationError):
        assert_compute_boundary("ingestion.vision_extraction", "not-a-provider")


def test_required_provider_for_unknown_stage() -> None:
    with pytest.raises(BoundaryViolationError):
        required_provider_for_stage("ingestion.unknown_stage")
