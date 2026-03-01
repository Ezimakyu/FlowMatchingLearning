"""
Run end-to-end pipeline in one command:

python backend/tools/run_full_pipeline.py \
  --doc-id sample \
  --source-file sample_data/sample_lecture.pdf \
  --source-file-id sample_lecture \
  --output-dir artifacts/full_run \
  --model-profile test

(directory mode: auto-discover source/media files)
python backend/tools/run_full_pipeline.py \
  --doc-id sample \
  --input-dir sample_data/course_pack \
  --output-dir artifacts/full_run \
  --model-profile test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re

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

SOURCE_FILE_EXTENSIONS = {".pdf", ".ppt", ".pptx"}
MEDIA_FILE_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".mov", ".webm", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline: Phase A -> TOC -> Graph.")
    parser.add_argument("--doc-id", required=True, help="Stable document identifier.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Optional directory mode. Auto-discovers source files "
            f"({', '.join(sorted(SOURCE_FILE_EXTENSIONS))}) and media files "
            f"({', '.join(sorted(MEDIA_FILE_EXTENSIONS))})."
        ),
    )
    parser.add_argument(
        "--inputs-json",
        default=None,
        help=(
            "Optional JSON path describing input file list. "
            "Supported shapes: list[items] or {\"items\": [...]}."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --input-dir, include nested files recursively.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        help="Path to PDF/slide file. Repeat for multi-file runs.",
    )
    parser.add_argument(
        "--source-file-id",
        action="append",
        default=[],
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
        "--output-input-manifest-json",
        default=None,
        help="Optional path override for resolved input manifest artifact.",
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
        "input_manifest": Path(args.output_input_manifest_json)
        if args.output_input_manifest_json
        else output_dir / "input_manifest.json",
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


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "_", value).strip("_").lower()
    return cleaned or "file"


def _parse_inputs_json(path: Path) -> list[dict[str, str | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    else:
        raise ValueError("inputs-json must be list[...] or {\"items\": [...]} payload.")

    normalized: list[dict[str, str | None]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"inputs-json item at index {index} must be an object.")
        source_file = str(item.get("source_file", "")).strip()
        if not source_file:
            raise ValueError(f"inputs-json item at index {index} missing source_file.")
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_file_id = str(item.get("source_file_id", "")).strip() or _safe_identifier(
            source_path.stem
        )

        media_file_raw = item.get("media_file")
        media_id_raw = item.get("media_id")
        media_file = str(media_file_raw).strip() if media_file_raw is not None else None
        media_id = str(media_id_raw).strip() if media_id_raw is not None else None
        if media_file and not Path(media_file).exists():
            raise FileNotFoundError(media_file)
        if media_file and not media_id:
            media_id = _safe_identifier(Path(media_file).stem)
        normalized.append(
            {
                "source_file": str(source_path),
                "source_file_id": source_file_id,
                "media_file": media_file,
                "media_id": media_id,
            }
        )
    return normalized


def _discover_inputs_in_directory(
    *,
    input_dir: Path,
    recursive: bool,
) -> list[dict[str, str | None]]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input-dir does not exist or is not a directory: {input_dir}")

    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    source_files: list[Path] = []
    media_by_key: dict[tuple[str, str], Path] = {}
    media_by_stem: dict[str, Path] = {}

    for candidate in iterator:
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in SOURCE_FILE_EXTENSIONS:
            source_files.append(candidate)
            continue
        if suffix not in MEDIA_FILE_EXTENSIONS:
            continue
        parent_key = str(candidate.parent.resolve())
        stem_key = candidate.stem.lower()
        media_by_key[(parent_key, stem_key)] = candidate
        media_by_stem.setdefault(stem_key, candidate)

    if not source_files:
        raise ValueError(
            f"No source files found in {input_dir} with extensions {sorted(SOURCE_FILE_EXTENSIONS)}."
        )

    source_files.sort(key=lambda item: str(item.relative_to(input_dir)))
    used_source_ids: dict[str, int] = {}
    discovered: list[dict[str, str | None]] = []

    for source_file in source_files:
        relative_no_suffix = source_file.relative_to(input_dir).with_suffix("")
        raw_source_id = _safe_identifier(str(relative_no_suffix).replace(os.sep, "_"))
        suffix_count = used_source_ids.get(raw_source_id, 0)
        used_source_ids[raw_source_id] = suffix_count + 1
        source_file_id = (
            raw_source_id if suffix_count == 0 else f"{raw_source_id}_{suffix_count + 1}"
        )
        source_parent_key = str(source_file.parent.resolve())
        source_stem_key = source_file.stem.lower()
        media_file = media_by_key.get((source_parent_key, source_stem_key)) or media_by_stem.get(
            source_stem_key
        )
        discovered.append(
            {
                "source_file": str(source_file),
                "source_file_id": source_file_id,
                "media_file": str(media_file) if media_file else None,
                "media_id": _safe_identifier(media_file.stem) if media_file else None,
            }
        )
    return discovered


def _resolve_input_rows(args: argparse.Namespace) -> list[dict[str, str | None]]:
    has_input_dir = bool(args.input_dir)
    has_inputs_json = bool(args.inputs_json)
    has_explicit_sources = bool(args.source_file)
    selected_modes = sum([has_input_dir, has_inputs_json, has_explicit_sources])
    if selected_modes == 0:
        raise ValueError(
            "Provide one input mode: explicit --source-file, --input-dir, or --inputs-json."
        )
    if selected_modes > 1:
        raise ValueError(
            "Input modes are mutually exclusive. Use only one of --source-file, --input-dir, or --inputs-json."
        )

    if has_input_dir:
        return _discover_inputs_in_directory(
            input_dir=Path(args.input_dir),
            recursive=bool(args.recursive),
        )

    if has_inputs_json:
        return _parse_inputs_json(Path(args.inputs_json))

    source_files = [str(item).strip() for item in args.source_file if str(item).strip()]
    source_ids = [str(item).strip() for item in args.source_file_id if str(item).strip()]
    if source_ids and len(source_files) != len(source_ids):
        raise ValueError(
            "When --source-file-id is provided, it must match --source-file count."
        )
    if not source_ids:
        source_ids = [_safe_identifier(Path(path).stem) for path in source_files]

    media_files = [str(item).strip() for item in args.media_file if str(item).strip()]
    media_ids = [str(item).strip() for item in args.media_id if str(item).strip()]
    if media_files and len(media_files) != len(source_files):
        raise ValueError(
            "When media is enabled, provide one --media-file per --source-file."
        )
    if media_ids and len(media_ids) != len(media_files):
        raise ValueError("When --media-id is provided, it must match --media-file count.")
    if media_files and not media_ids:
        media_ids = [_safe_identifier(Path(path).stem) for path in media_files]

    rows: list[dict[str, str | None]] = []
    for index, source_file in enumerate(source_files):
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        media_file = media_files[index] if media_files else None
        media_id = media_ids[index] if media_ids else None
        if media_file and not Path(media_file).exists():
            raise FileNotFoundError(media_file)
        rows.append(
            {
                "source_file": source_file,
                "source_file_id": source_ids[index],
                "media_file": media_file,
                "media_id": media_id,
            }
        )
    return rows


def _build_phase_a_input_namespace(rows: list[dict[str, str | None]]) -> argparse.Namespace:
    media_files = [item["media_file"] for item in rows]
    media_ids = [item["media_id"] for item in rows]
    return argparse.Namespace(
        source_file=[item["source_file"] for item in rows],
        source_file_id=[item["source_file_id"] for item in rows],
        media_file=media_files if any(media_files) else [],
        media_id=media_ids if any(media_files) else [],
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    if not args.no_env_file:
        loaded = load_env_file(args.env_file, override=False)
        logger.info("full_pipeline.env_loaded keys=%d path=%s", len(loaded), args.env_file)

    output_paths = _resolve_output_paths(args)
    input_rows = _resolve_input_rows(args)
    phase_a_args = _build_phase_a_input_namespace(input_rows)
    inputs = build_ingestion_inputs(phase_a_args)
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
    _write_json(
        output_paths["input_manifest"],
        {
            "doc_id": args.doc_id,
            "input_mode": (
                "input_dir"
                if args.input_dir
                else ("inputs_json" if args.inputs_json else "explicit")
            ),
            "items": input_rows,
        },
    )
    logger.info("full_pipeline.input_manifest_written path=%s", output_paths["input_manifest"])

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
            max_sections_to_parse=loop_hp.max_sections_to_parse,
            max_llm_concepts_per_section=loop_hp.max_llm_concepts_per_section,
            max_state_nodes_in_context=loop_hp.max_state_nodes_in_context,
            max_historical_nodes_for_local_similarity=loop_hp.max_historical_nodes_for_local_similarity,
            seed_core_nodes_from_toc=loop_hp.seed_core_nodes_from_toc,
            max_seed_core_nodes=loop_hp.max_seed_core_nodes,
            freeze_node_set_after_seed=loop_hp.freeze_node_set_after_seed,
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
