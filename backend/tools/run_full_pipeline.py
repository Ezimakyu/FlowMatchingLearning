"""
Run end-to-end pipeline in one command:

python backend/tools/run_full_pipeline.py \
  --doc-id sample \
  --source-file sample_data/sample_lecture.pdf \
  --source-file-id sample_lecture \
  --output-dir artifacts/full_run \
  --model-profile test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from backend.app.config import load_env_file, load_hyperparameters
from backend.app.ingestion import ModalRemoteIngestionClient
from backend.app.logging_utils import configure_logging
from backend.app.pipelines import (
    PhaseAIngestionConfig,
    PhaseAIngestionPipeline,
    PhaseBGraphConfig,
    PhaseBGraphPipeline,
    PhaseBTOCConfig,
    PhaseBTOCPipeline,
)
from backend.app.reasoning import (
    OpenAISectionReasoningClient,
    OpenAISectionReasoningConfig,
    OpenAITOCReasoningClient,
    OpenAITOCReasoningConfig,
)
from backend.app.storage import ActianCortexConfig, ActianCortexStore
from backend.tools.run_phase_a import build_ingestion_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline: Phase A -> TOC -> Graph.")
    parser.add_argument("--doc-id", required=True, help="Stable document identifier.")
    parser.add_argument(
        "--source-file",
        action="append",
        required=True,
        help="Path to PDF/slide file. Repeat for multi-file runs.",
    )
    parser.add_argument(
        "--source-file-id",
        action="append",
        required=True,
        help="External source file identifier. Repeat in the same order as --source-file.",
    )
    parser.add_argument(
        "--media-file",
        action="append",
        default=[],
        help=(
            "Optional media file path for Whisper transcription. "
            "When used, provide one entry per --source-file in matching order."
        ),
    )
    parser.add_argument(
        "--media-id",
        action="append",
        default=[],
        help=(
            "Optional media identifier. Required for each --media-file and must align by position."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for all generated artifacts.",
    )
    parser.add_argument(
        "--output-phase-a-json",
        default=None,
        help="Optional path override for PhaseAIngestionResult artifact.",
    )
    parser.add_argument(
        "--output-toc-json",
        default=None,
        help="Optional path override for TOCData artifact.",
    )
    parser.add_argument(
        "--output-toc-meta-json",
        default=None,
        help="Optional path override for TOC metadata artifact.",
    )
    parser.add_argument(
        "--output-graph-json",
        default=None,
        help="Optional path override for GraphData artifact.",
    )
    parser.add_argument(
        "--output-rolling-state-json",
        default=None,
        help="Optional path override for RollingState artifact.",
    )
    parser.add_argument(
        "--output-section-results-json",
        default=None,
        help="Optional path override for section parse results artifact.",
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
        "--toc-prompt-version",
        default=None,
        help="Optional TOC prompt version override.",
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
        "--job-id",
        default=None,
        help="Optional job id override for graph stage.",
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


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    return {
        "phase_a": Path(args.output_phase_a_json)
        if args.output_phase_a_json
        else output_dir / "phase_a_result.json",
        "toc": Path(args.output_toc_json) if args.output_toc_json else output_dir / "toc.json",
        "toc_meta": Path(args.output_toc_meta_json)
        if args.output_toc_meta_json
        else output_dir / "toc_meta.json",
        "graph": Path(args.output_graph_json)
        if args.output_graph_json
        else output_dir / "graph_data.json",
        "rolling_state": Path(args.output_rolling_state_json)
        if args.output_rolling_state_json
        else output_dir / "rolling_state.json",
        "section_results": Path(args.output_section_results_json)
        if args.output_section_results_json
        else output_dir / "section_parse_results.json",
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    if not args.no_env_file:
        loaded = load_env_file(args.env_file, override=False)
        logger.info("full_pipeline.env_loaded keys=%d path=%s", len(loaded), args.env_file)

    output_paths = _resolve_output_paths(args)
    inputs = build_ingestion_inputs(args)
    hyperparams = load_hyperparameters(args.hyperparams_json)
    toc_hp = hyperparams.phase_b.toc_generation
    loop_hp = hyperparams.phase_b.iteration_loop

    env_model = os.getenv("OPENAI_REASONING_MODEL", "").strip()
    env_temperature = os.getenv("OPENAI_REASONING_TEMPERATURE", "").strip()
    env_max_tokens = os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "").strip()
    env_toc_prompt = os.getenv("TOC_PROMPT_VERSION", "").strip()
    env_section_prompt = os.getenv("SECTION_CONCEPT_PROMPT_VERSION", "").strip()
    env_edge_prompt = os.getenv("EDGE_VALIDATION_PROMPT_VERSION", "").strip()

    profile_model = toc_hp.model_test if args.model_profile == "test" else toc_hp.model_demo
    model = args.model or env_model or profile_model
    temperature = float(env_temperature) if env_temperature else toc_hp.temperature
    max_output_tokens = int(env_max_tokens) if env_max_tokens else toc_hp.max_output_tokens
    toc_prompt_version = args.toc_prompt_version or env_toc_prompt or toc_hp.prompt_version
    section_prompt_version = args.section_prompt_version or env_section_prompt or "2026-03-01.v3"
    edge_prompt_version = args.edge_prompt_version or env_edge_prompt or "2026-03-01.v3"

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

    logger.info(
        "full_pipeline.start doc_id=%s source_files=%d model=%s model_profile=%s",
        args.doc_id,
        len(inputs),
        model,
        args.model_profile,
    )

    # Phase A
    logger.info("full_pipeline.phase_a_start doc_id=%s", args.doc_id)
    phase_a_pipeline = PhaseAIngestionPipeline(
        ingestion_client=ModalRemoteIngestionClient(),
        storage_client=ActianCortexStore(config=storage_config),
        config=PhaseAIngestionConfig(
            max_vision_chunk_tokens=hyperparams.phase_a.max_vision_chunk_tokens,
            max_transcript_chunk_tokens=hyperparams.phase_a.max_transcript_chunk_tokens,
            include_transcript_chunks=hyperparams.phase_a.include_transcript_chunks,
        ),
    )
    phase_a_result = phase_a_pipeline.run_batch(doc_id=args.doc_id, inputs=inputs)
    _write_json(output_paths["phase_a"], phase_a_result.model_dump(mode="json"))
    logger.info("full_pipeline.phase_a_finish path=%s", output_paths["phase_a"])

    # Phase B TOC
    logger.info("full_pipeline.toc_start doc_id=%s prompt_version=%s", args.doc_id, toc_prompt_version)
    toc_pipeline = PhaseBTOCPipeline(
        reasoning_client=OpenAITOCReasoningClient(
            config=OpenAITOCReasoningConfig(
                model=model,
                prompt_version=toc_prompt_version,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        ),
        config=PhaseBTOCConfig(max_input_chars=toc_hp.max_input_chars),
        reasoning_provider="openai",
    )
    toc_result = toc_pipeline.run(doc_id=args.doc_id, chunking=phase_a_result.chunking)
    _write_json(output_paths["toc"], toc_result.toc.model_dump(mode="json"))
    _write_json(
        output_paths["toc_meta"],
        {
            "llm_call": toc_result.llm_call.model_dump(mode="json"),
            "prompt_tag": toc_result.prompt_tag,
            "prompt_checksum": toc_result.prompt_checksum,
        },
    )
    logger.info("full_pipeline.toc_finish path=%s", output_paths["toc"])

    # Phase B graph
    logger.info(
        "full_pipeline.graph_start doc_id=%s section_prompt=%s edge_prompt=%s",
        args.doc_id,
        section_prompt_version,
        edge_prompt_version,
    )
    graph_pipeline = PhaseBGraphPipeline(
        reasoning_client=OpenAISectionReasoningClient(
            config=OpenAISectionReasoningConfig(
                model=model,
                section_prompt_version=section_prompt_version,
                edge_prompt_version=edge_prompt_version,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        ),
        storage_client=ActianCortexStore(config=storage_config),
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
    graph_result = graph_pipeline.run(
        doc_id=args.doc_id,
        toc=toc_result.toc,
        chunking=phase_a_result.chunking,
        embeddings=phase_a_result.embeddings,
        job_id=args.job_id,
    )
    _write_json(output_paths["graph"], graph_result.graph.model_dump(mode="json"))
    _write_json(output_paths["rolling_state"], graph_result.rolling_state.model_dump(mode="json"))
    _write_json(
        output_paths["section_results"],
        [item.model_dump(mode="json") for item in graph_result.section_results],
    )
    logger.info("full_pipeline.graph_finish path=%s", output_paths["graph"])

    print(
        "Full pipeline complete: "
        f"phase_a={output_paths['phase_a']} toc={output_paths['toc']} "
        f"graph={output_paths['graph']} "
        f"(nodes={len(graph_result.graph.nodes)}, edges={len(graph_result.graph.edges)})"
    )


if __name__ == "__main__":
    main()
