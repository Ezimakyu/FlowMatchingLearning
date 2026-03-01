import json
from pathlib import Path

from backend.app.config import load_hyperparameters


def test_load_hyperparameters_from_defaults_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    hyperparams = load_hyperparameters(missing)
    assert hyperparams.phase_a.max_vision_chunk_tokens == 260
    assert hyperparams.phase_b.toc_generation.model_test == "gpt-4.1-mini"
    assert hyperparams.phase_b.iteration_loop.top_k_historical_matches == 12
    assert hyperparams.phase_b.iteration_loop.similarity_fallback_threshold == 0.62


def test_load_hyperparameters_ignores_comment_fields(tmp_path: Path) -> None:
    payload = {
        "_comment": "root comment",
        "phase_a": {
            "_comment": "phase a comment",
            "max_vision_chunk_tokens": 300,
            "max_transcript_chunk_tokens": 230,
            "include_transcript_chunks": False,
        },
        "phase_b": {
            "toc_generation": {
                "_comment": "toc comment",
                "max_input_chars": 90000,
                "model_test": "gpt-4.1-mini",
                "model_demo": "gpt-5.1",
                "prompt_version": "2026-02-28.v2",
                "temperature": 0.1,
                "max_output_tokens": 3000,
            },
            "iteration_loop": {
                "top_k_historical_matches": 9,
                "similarity_threshold": 0.8,
                "similarity_fallback_threshold": 0.58,
                "edge_acceptance_confidence_threshold": 0.7,
                "retrieval_overfetch_multiplier": 5,
                "max_section_chars_per_call": 25000,
                "max_state_nodes_in_context": 150,
                "max_historical_nodes_for_local_similarity": 300,
            },
        },
    }
    config_path = tmp_path / "hyperparameters.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    hyperparams = load_hyperparameters(config_path)
    assert hyperparams.phase_a.max_vision_chunk_tokens == 300
    assert hyperparams.phase_a.include_transcript_chunks is False
    assert hyperparams.phase_b.toc_generation.max_input_chars == 90000
    assert hyperparams.phase_b.iteration_loop.top_k_historical_matches == 9
    assert hyperparams.phase_b.iteration_loop.similarity_fallback_threshold == 0.58
    assert hyperparams.phase_b.iteration_loop.retrieval_overfetch_multiplier == 5
