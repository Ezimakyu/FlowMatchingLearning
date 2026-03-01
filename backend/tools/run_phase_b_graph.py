"""
python backend/tools/run_phase_b_graph.py \
  --phase-a-json artifacts/phase_a_result.json \
  --toc-json artifacts/toc.json \
  --output-graph-json artifacts/graph_data.json \
  --output-rolling-state-json artifacts/rolling_state.json \
  --output-section-results-json artifacts/section_parse_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from backend.app.config import load_env_file, load_hyperparameters
from backend.app.logging_utils import configure_logging
from backend.app.models import PhaseAIngestionResult, TOCData
from backend.app.pipelines import PhaseBGraphConfig, PhaseBGraphPipeline
from backend.app.reasoning import OpenAISectionReasoningClient, OpenAISectionReasoningConfig
from backend.app.storage import ActianCortexConfig, ActianCortexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase B section loop and graph generation.")
    parser.add_argument(
        "--phase-a-json",
        required=True,
        help="Path to PhaseAIngestionResult JSON artifact.",
    )
    parser.add_argument(
        "--toc-json",
        required=True,
        help="Path to TOCData JSON artifact.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional doc_id override (defaults to phase_a.doc_id).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Optional job id override.",
    )
    parser.add_argument(
        "--output-graph-json",
        required=True,
        help="Path to write GraphData JSON artifact.",
    )
    parser.add_argument(
        "--output-rolling-state-json",
        default=None,
        help="Optional path to write RollingState JSON artifact.",
    )
    parser.add_argument(
        "--output-section-results-json",
        default=None,
        help="Optional path to write list[SectionParseResult] JSON artifact.",
    )
    parser.add_argument(
        "--actian-addr",
        default=None,
        help="Optional Actian VectorAI address, e.g. localhost:50051.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override (otherwise OPENAI_REASONING_MODEL/env default).",
    )
    parser.add_argument(
        "--model-profile",
        choices=["test", "demo"],
        default="test",
        help="Model profile from hyperparameters JSON when --model is not provided.",
    )
    parser.add_argument(
        "--section-prompt-version",
        default=None,
        help="Optional section concept extraction prompt version override.",
    )
    parser.add_argument(
        "--edge-prompt-version",
        default=None,
        help="Optional edge validation prompt version override.",
    )
    parser.add_argument(
        "--hyperparams-json",
        default="backend/config/hyperparameters.json",
        help="Path to hyperparameters JSON file.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to load before execution (default: .env).",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Disable automatic env-file loading.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    if not args.no_env_file:
        loaded = load_env_file(args.env_file, override=False)
        logger.info("phase_b_graph.env_loaded keys=%d path=%s", len(loaded), args.env_file)

    phase_a_payload = json.loads(Path(args.phase_a_json).read_text(encoding="utf-8"))
    toc_payload = json.loads(Path(args.toc_json).read_text(encoding="utf-8"))
    phase_a = PhaseAIngestionResult.model_validate(phase_a_payload)
    toc = TOCData.model_validate(toc_payload)

    doc_id = args.doc_id or phase_a.doc_id
    if toc.doc_id != doc_id:
        raise ValueError(f"doc_id mismatch: run={doc_id}, toc={toc.doc_id}")
    if phase_a.doc_id != doc_id:
        raise ValueError(f"doc_id mismatch: run={doc_id}, phase_a={phase_a.doc_id}")

    hyperparams = load_hyperparameters(args.hyperparams_json)
    toc_hp = hyperparams.phase_b.toc_generation
    loop_hp = hyperparams.phase_b.iteration_loop
    env_model = os.getenv("OPENAI_REASONING_MODEL", "").strip()
    env_temperature = os.getenv("OPENAI_REASONING_TEMPERATURE", "").strip()
    env_max_tokens = os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "").strip()
    profile_model = toc_hp.model_test if args.model_profile == "test" else toc_hp.model_demo

    model = args.model or env_model or profile_model
    section_prompt_version = (
        args.section_prompt_version
        or os.getenv("SECTION_CONCEPT_PROMPT_VERSION", "").strip()
        or "2026-03-01.v3"
    )
    edge_prompt_version = (
        args.edge_prompt_version
        or os.getenv("EDGE_VALIDATION_PROMPT_VERSION", "").strip()
        or "2026-03-01.v3"
    )
    temperature = float(env_temperature) if env_temperature else toc_hp.temperature
    max_output_tokens = int(env_max_tokens) if env_max_tokens else toc_hp.max_output_tokens

    logger.info(
        "phase_b_graph.start doc_id=%s model=%s section_prompt=%s edge_prompt=%s",
        doc_id,
        model,
        section_prompt_version,
        edge_prompt_version,
    )

    reasoning_client = OpenAISectionReasoningClient(
        config=OpenAISectionReasoningConfig(
            model=model,
            section_prompt_version=section_prompt_version,
            edge_prompt_version=edge_prompt_version,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )
    storage_config = ActianCortexConfig.from_env()
    if args.actian_addr:
        storage_config = ActianCortexConfig(
            server_address=args.actian_addr,
            collection_prefix=storage_config.collection_prefix,
            distance_metric=storage_config.distance_metric,
            hnsw_m=storage_config.hnsw_m,
            hnsw_ef_construct=storage_config.hnsw_ef_construct,
            hnsw_ef_search=storage_config.hnsw_ef_search,
        )
    storage_client = ActianCortexStore(config=storage_config)
    pipeline = PhaseBGraphPipeline(
        reasoning_client=reasoning_client,
        storage_client=storage_client,
        config=PhaseBGraphConfig(
            top_k_historical_matches=loop_hp.top_k_historical_matches,
            similarity_threshold=loop_hp.similarity_threshold,
            similarity_fallback_threshold=loop_hp.similarity_fallback_threshold,
            edge_acceptance_confidence_threshold=loop_hp.edge_acceptance_confidence_threshold,
            retrieval_overfetch_multiplier=loop_hp.retrieval_overfetch_multiplier,
            max_section_chars_per_call=loop_hp.max_section_chars_per_call,
            max_state_nodes_in_context=loop_hp.max_state_nodes_in_context,
            max_historical_nodes_for_local_similarity=loop_hp.max_historical_nodes_for_local_similarity,
        ),
        reasoning_provider="openai",
        storage_provider="actian",
    )
    result = pipeline.run(
        doc_id=doc_id,
        toc=toc,
        chunking=phase_a.chunking,
        embeddings=phase_a.embeddings,
        job_id=args.job_id,
    )

    Path(args.output_graph_json).write_text(
        json.dumps(result.graph.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("phase_b_graph.output_graph_written path=%s", args.output_graph_json)

    if args.output_rolling_state_json:
        Path(args.output_rolling_state_json).write_text(
            json.dumps(result.rolling_state.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "phase_b_graph.output_rolling_state_written path=%s",
            args.output_rolling_state_json,
        )

    if args.output_section_results_json:
        Path(args.output_section_results_json).write_text(
            json.dumps([item.model_dump(mode="json") for item in result.section_results], indent=2)
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "phase_b_graph.output_section_results_written path=%s",
            args.output_section_results_json,
        )

    print(
        "Wrote graph "
        f"(nodes={len(result.graph.nodes)}, edges={len(result.graph.edges)}) "
        f"to {args.output_graph_json}"
    )
    logger.info(
        "phase_b_graph.finish doc_id=%s nodes=%d edges=%d",
        result.rolling_state.doc_id,
        len(result.graph.nodes),
        len(result.graph.edges),
    )


if __name__ == "__main__":
    main()
