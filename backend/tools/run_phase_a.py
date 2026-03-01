"""
(PDF only, simplest first run)
python backend/tools/run_phase_a.py \
  --doc-id sample \
  --source-file "sample_data/sample_lecture.pdf" \
  --source-file-id sample_lecture \
  --output-json artifacts/phase_a_result.json

(PDF + video test)
python backend/tools/run_phase_a.py \
  --doc-id sample \
  --source-file "sample_data/sample_lecture.pdf" \
  --source-file-id sample_lecture \
  --media-file "sample_data/sample_lecture.mp4" \
  --media-id sample_video \
  --output-json artifacts/phase_a_result.json

(Multi-source + multi-video batch run; positional pairing by index)
python backend/tools/run_phase_a.py \
  --doc-id sample_batch \
  --source-file "sample_data/lecture_01.pdf" \
  --source-file-id lecture_01_slides \
  --media-file "sample_data/lecture_01.mp4" \
  --media-id lecture_01_video \
  --source-file "sample_data/lecture_02.pdf" \
  --source-file-id lecture_02_slides \
  --media-file "sample_data/lecture_02.mp4" \
  --media-id lecture_02_video \
  --output-json artifacts/phase_a_result.json

python backend/tools/run_phase_b_toc.py \
  --phase-a-json artifacts/phase_a_result.json \
  --output-toc-json artifacts/toc.json \
  --output-meta-json artifacts/toc_meta.json \
  --model-profile demo
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backend.app.config import load_env_file, load_hyperparameters
from backend.app.ingestion import ModalRemoteIngestionClient
from backend.app.logging_utils import configure_logging
from backend.app.pipelines import (
    PhaseAIngestionConfig,
    PhaseAIngestionInput,
    PhaseAIngestionPipeline,
)
from backend.app.storage import ActianCortexConfig, ActianCortexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A ingestion pipeline.")
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
        "--output-json",
        default=None,
        help="Optional path to write PhaseAIngestionResult as JSON.",
    )
    parser.add_argument(
        "--actian-addr",
        default=None,
        help="Optional Actian VectorAI address, e.g. localhost:50051.",
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


def _normalize_to_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_ingestion_inputs(args: argparse.Namespace) -> list[PhaseAIngestionInput]:
    source_files = _normalize_to_list(args.source_file)
    source_file_ids = _normalize_to_list(args.source_file_id)
    media_files = _normalize_to_list(args.media_file)
    media_ids = _normalize_to_list(args.media_id)

    if len(source_files) != len(source_file_ids):
        raise ValueError(
            "Expected equal counts for --source-file and --source-file-id. "
            f"Got source_files={len(source_files)} source_file_ids={len(source_file_ids)}."
        )

    if media_files or media_ids:
        if len(media_files) != len(source_files):
            raise ValueError(
                "When media is enabled, provide one --media-file per --source-file "
                f"(source_files={len(source_files)} media_files={len(media_files)})."
            )
        if len(media_ids) != len(media_files):
            raise ValueError(
                "Expected equal counts for --media-file and --media-id "
                f"(media_files={len(media_files)} media_ids={len(media_ids)})."
            )

    inputs: list[PhaseAIngestionInput] = []
    for index, (source_file, source_file_id) in enumerate(zip(source_files, source_file_ids)):
        source_bytes = Path(source_file).read_bytes()
        media_file = media_files[index] if media_files else None
        media_id = media_ids[index] if media_ids else None
        media_bytes = Path(media_file).read_bytes() if media_file else None
        inputs.append(
            PhaseAIngestionInput(
                source_file_id=source_file_id,
                source_file_bytes=source_bytes,
                media_id=media_id,
                media_bytes=media_bytes,
            )
        )

    if not inputs:
        raise ValueError("No input files were provided.")
    return inputs


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    if not args.no_env_file:
        loaded = load_env_file(args.env_file, override=False)
        logger.info("phase_a.env_loaded keys=%d path=%s", len(loaded), args.env_file)

    inputs = build_ingestion_inputs(args)
    logger.info(
        "phase_a.start doc_id=%s source_files=%d media_files=%d",
        args.doc_id,
        len(inputs),
        sum(1 for item in inputs if item.media_id is not None),
    )

    hyperparams = load_hyperparameters(args.hyperparams_json)
    ingestion_client = ModalRemoteIngestionClient()
    config = ActianCortexConfig.from_env()
    if args.actian_addr:
        config = ActianCortexConfig(
            server_address=args.actian_addr,
            collection_prefix=config.collection_prefix,
            distance_metric=config.distance_metric,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construct=config.hnsw_ef_construct,
            hnsw_ef_search=config.hnsw_ef_search,
        )
    storage_client = ActianCortexStore(config=config)
    pipeline = PhaseAIngestionPipeline(
        ingestion_client=ingestion_client,
        storage_client=storage_client,
        config=PhaseAIngestionConfig(
            max_vision_chunk_tokens=hyperparams.phase_a.max_vision_chunk_tokens,
            max_transcript_chunk_tokens=hyperparams.phase_a.max_transcript_chunk_tokens,
            include_transcript_chunks=hyperparams.phase_a.include_transcript_chunks,
        ),
    )
    result = pipeline.run_batch(
        doc_id=args.doc_id,
        inputs=inputs,
    )

    serialized = json.dumps(result.model_dump(mode="json"), indent=2)
    if args.output_json:
        Path(args.output_json).write_text(serialized + "\n", encoding="utf-8")
        logger.info("phase_a.output_written path=%s", args.output_json)
    else:
        print(serialized)
        logger.info("phase_a.output_printed_to_stdout")

    logger.info(
        "phase_a.finish doc_id=%s stored_chunks=%d stored_embeddings=%d",
        result.doc_id,
        result.stored_chunk_count,
        result.stored_embedding_count,
    )


if __name__ == "__main__":
    main()
