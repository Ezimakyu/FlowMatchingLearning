import json
from argparse import Namespace
from pathlib import Path

from backend.app.models import LLMCallMetadata, TOCData, TOCSection
from backend.app.reasoning import TOCGenerationOutput
from backend.tools import run_phase_b_toc


class FakeReasoningClient:
    def __init__(self, config=None) -> None:
        self.config = config

    def generate_toc(self, *, doc_id: str, document_text: str) -> TOCGenerationOutput:
        _ = document_text
        toc = TOCData(
            doc_id=doc_id,
            sections=[
                TOCSection(
                    section_id="intro",
                    title="Intro",
                    order=0,
                    chunk_ids=[],
                    key_terms=[],
                    children=[],
                )
            ],
        )
        return TOCGenerationOutput(
            toc=toc,
            llm_call=LLMCallMetadata(
                provider="openai",
                prompt_name="toc_generation",
                prompt_version="2026-02-28.v2",
                model="gpt-4.1-mini",
                request_id="req_1",
            ),
            prompt_tag="toc_generation:2026-02-28.v2",
            prompt_checksum="checksum",
            raw_response_text='{"doc_id":"x","sections":[]}',
        )


def test_run_phase_b_toc_main_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    chunking_payload = {
        "schema_version": "1.0.0",
        "doc_id": "doc_cli",
        "chunks": [
            {
                "chunk_id": "doc_cli:vision_text:00000",
                "doc_id": "doc_cli",
                "source_type": "vision_text",
                "order": 0,
                "text": "Limits and derivatives.",
                "token_estimate": 5,
                "metadata": {},
            }
        ],
        "created_at": "2026-03-01T00:00:00Z",
        "metadata": {},
    }
    chunking_json = tmp_path / "chunking.json"
    chunking_json.write_text(json.dumps(chunking_payload), encoding="utf-8")

    output_toc = tmp_path / "toc.json"
    output_meta = tmp_path / "toc_meta.json"
    hp_json = tmp_path / "hyperparameters.json"
    hp_json.write_text(
        json.dumps(
            {
                "phase_b": {
                    "toc_generation": {
                        "max_input_chars": 10000,
                        "model_test": "gpt-4.1-mini",
                        "model_demo": "gpt-5.1",
                        "prompt_version": "2026-02-28.v2",
                        "temperature": 0.0,
                        "max_output_tokens": 4000,
                    },
                    "iteration_loop": {
                        "top_k_historical_matches": 8,
                        "similarity_threshold": 0.78,
                        "edge_acceptance_confidence_threshold": 0.65,
                        "max_section_chars_per_call": 30000,
                        "max_state_nodes_in_context": 200,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_phase_b_toc, "OpenAITOCReasoningClient", FakeReasoningClient)
    monkeypatch.setattr(
        run_phase_b_toc,
        "parse_args",
        lambda: Namespace(
            chunking_json=str(chunking_json),
            phase_a_json=None,
            doc_id=None,
            output_toc_json=str(output_toc),
            output_meta_json=str(output_meta),
            model=None,
            model_profile="test",
            prompt_version=None,
            max_input_chars=None,
            hyperparams_json=str(hp_json),
            env_file=".env",
            no_env_file=True,
        ),
    )

    run_phase_b_toc.main()

    toc_payload = json.loads(output_toc.read_text(encoding="utf-8"))
    meta_payload = json.loads(output_meta.read_text(encoding="utf-8"))
    assert toc_payload["doc_id"] == "doc_cli"
    assert len(toc_payload["sections"]) == 1
    assert meta_payload["llm_call"]["provider"] == "openai"


def test_run_phase_b_toc_accepts_phase_a_json(monkeypatch, tmp_path: Path) -> None:
    phase_a_payload = {
        "schema_version": "1.0.0",
        "doc_id": "doc_cli_phasea",
        "chunking": {
            "schema_version": "1.0.0",
            "doc_id": "doc_cli_phasea",
            "chunks": [
                {
                    "chunk_id": "doc_cli_phasea:vision_text:00000",
                    "doc_id": "doc_cli_phasea",
                    "source_type": "vision_text",
                    "order": 0,
                    "text": "A test chunk for TOC.",
                    "token_estimate": 6,
                    "metadata": {},
                }
            ],
            "created_at": "2026-03-01T00:00:00Z",
            "metadata": {},
        },
        "embeddings": {
            "schema_version": "1.0.0",
            "doc_id": "doc_cli_phasea",
            "provider": "modal",
            "runtime": "serverless_gpu",
            "model_name": "BAAI/bge-m3",
            "embeddings": [],
            "created_at": "2026-03-01T00:00:00Z",
            "metadata": {},
        },
        "stored_chunk_count": 1,
        "stored_embedding_count": 0,
        "created_at": "2026-03-01T00:00:00Z",
        "metadata": {},
    }
    phase_a_json = tmp_path / "phase_a.json"
    phase_a_json.write_text(json.dumps(phase_a_payload), encoding="utf-8")

    output_toc = tmp_path / "toc_from_phasea.json"
    hp_json = tmp_path / "hyperparameters.json"
    hp_json.write_text(
        json.dumps(
            {
                "phase_b": {
                    "toc_generation": {
                        "max_input_chars": 10000,
                        "model_test": "gpt-4.1-mini",
                        "model_demo": "gpt-5.1",
                        "prompt_version": "2026-02-28.v2",
                        "temperature": 0.0,
                        "max_output_tokens": 4000,
                    },
                    "iteration_loop": {
                        "top_k_historical_matches": 8,
                        "similarity_threshold": 0.78,
                        "edge_acceptance_confidence_threshold": 0.65,
                        "max_section_chars_per_call": 30000,
                        "max_state_nodes_in_context": 200,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_phase_b_toc, "OpenAITOCReasoningClient", FakeReasoningClient)
    monkeypatch.setattr(
        run_phase_b_toc,
        "parse_args",
        lambda: Namespace(
            chunking_json=None,
            phase_a_json=str(phase_a_json),
            doc_id=None,
            output_toc_json=str(output_toc),
            output_meta_json=None,
            model=None,
            model_profile="test",
            prompt_version=None,
            max_input_chars=None,
            hyperparams_json=str(hp_json),
            env_file=".env",
            no_env_file=True,
        ),
    )

    run_phase_b_toc.main()
    toc_payload = json.loads(output_toc.read_text(encoding="utf-8"))
    assert toc_payload["doc_id"] == "doc_cli_phasea"
