import json
from argparse import Namespace
from pathlib import Path

from backend.app.models import (
    GraphData,
    RollingState,
    SectionParseResult,
)
from backend.app.pipelines.phase_b_graph import PhaseBGraphOutput
from backend.tools import run_phase_b_graph


class FakeReasoningClient:
    def __init__(self, config=None) -> None:
        self.config = config


class FakeStorageClient:
    def __init__(self, config=None) -> None:
        self.config = config


class FakePipeline:
    def __init__(
        self,
        *,
        reasoning_client,
        storage_client,
        config,
        reasoning_provider,
        storage_provider,
    ) -> None:
        self.reasoning_client = reasoning_client
        self.storage_client = storage_client
        self.config = config
        self.reasoning_provider = reasoning_provider
        self.storage_provider = storage_provider

    def run(self, *, doc_id, toc, chunking, embeddings, job_id=None) -> PhaseBGraphOutput:
        _ = toc
        _ = chunking
        _ = embeddings
        graph = GraphData(
            graph_id=f"graph_{doc_id}",
            metadata={"doc_id": doc_id},
        )
        rolling_state = RollingState(
            job_id=job_id or "job_test",
            doc_id=doc_id,
        )
        section_result = SectionParseResult(
            job_id=rolling_state.job_id,
            doc_id=doc_id,
            section_id="intro",
            section_order=0,
            section_title="Intro",
        )
        return PhaseBGraphOutput(
            graph=graph,
            rolling_state=rolling_state,
            section_results=[section_result],
        )


def test_run_phase_b_graph_main_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    phase_a_payload = {
        "schema_version": "1.0.0",
        "doc_id": "doc_phaseb",
        "chunking": {
            "schema_version": "1.0.0",
            "doc_id": "doc_phaseb",
            "chunks": [
                {
                    "chunk_id": "doc_phaseb:vision_text:00000",
                    "doc_id": "doc_phaseb",
                    "source_type": "vision_text",
                    "order": 0,
                    "text": "Limits introduction.",
                    "token_estimate": 5,
                    "metadata": {},
                }
            ],
            "created_at": "2026-03-01T00:00:00Z",
            "metadata": {},
        },
        "embeddings": {
            "schema_version": "1.0.0",
            "doc_id": "doc_phaseb",
            "provider": "modal",
            "runtime": "serverless_gpu",
            "model_name": "BAAI/bge-m3",
            "embeddings": [
                {
                    "chunk_id": "doc_phaseb:vision_text:00000",
                    "vector": [0.1, 0.2, 0.3],
                    "vector_dim": 3,
                    "model_name": "BAAI/bge-m3",
                }
            ],
            "created_at": "2026-03-01T00:00:00Z",
            "metadata": {},
        },
        "stored_chunk_count": 1,
        "stored_embedding_count": 1,
        "created_at": "2026-03-01T00:00:00Z",
        "metadata": {},
    }
    toc_payload = {
        "schema_version": "1.0.0",
        "doc_id": "doc_phaseb",
        "sections": [
            {
                "section_id": "intro",
                "title": "Intro",
                "order": 0,
                "chunk_ids": ["doc_phaseb:vision_text:00000"],
                "key_terms": [],
                "children": [],
            }
        ],
    }
    phase_a_json = tmp_path / "phase_a.json"
    toc_json = tmp_path / "toc.json"
    phase_a_json.write_text(json.dumps(phase_a_payload), encoding="utf-8")
    toc_json.write_text(json.dumps(toc_payload), encoding="utf-8")

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

    output_graph = tmp_path / "graph.json"
    output_state = tmp_path / "rolling_state.json"
    output_sections = tmp_path / "section_results.json"

    monkeypatch.setattr(run_phase_b_graph, "OpenAISectionReasoningClient", FakeReasoningClient)
    monkeypatch.setattr(run_phase_b_graph, "ActianCortexStore", FakeStorageClient)
    monkeypatch.setattr(run_phase_b_graph, "PhaseBGraphPipeline", FakePipeline)
    monkeypatch.setattr(
        run_phase_b_graph,
        "parse_args",
        lambda: Namespace(
            phase_a_json=str(phase_a_json),
            toc_json=str(toc_json),
            doc_id=None,
            job_id="job_cli",
            output_graph_json=str(output_graph),
            output_rolling_state_json=str(output_state),
            output_section_results_json=str(output_sections),
            actian_addr=None,
            model=None,
            model_profile="test",
            section_prompt_version=None,
            edge_prompt_version=None,
            hyperparams_json=str(hp_json),
            env_file=".env",
            no_env_file=True,
        ),
    )

    run_phase_b_graph.main()

    graph_payload = json.loads(output_graph.read_text(encoding="utf-8"))
    state_payload = json.loads(output_state.read_text(encoding="utf-8"))
    section_payload = json.loads(output_sections.read_text(encoding="utf-8"))
    assert graph_payload["graph_id"] == "graph_doc_phaseb"
    assert state_payload["doc_id"] == "doc_phaseb"
    assert section_payload[0]["section_id"] == "intro"
