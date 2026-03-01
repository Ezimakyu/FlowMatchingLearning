from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.ingestion import ModalRemoteIngestionClient
from backend.app.pipelines import PhaseAIngestionPipeline
from backend.app.storage import ActianCortexConfig, ActianCortexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A ingestion pipeline.")
    parser.add_argument("--doc-id", required=True, help="Stable document identifier.")
    parser.add_argument("--source-file", required=True, help="Path to PDF/slide file.")
    parser.add_argument(
        "--source-file-id",
        required=True,
        help="External source file identifier.",
    )
    parser.add_argument(
        "--media-file",
        default=None,
        help="Optional media file path for Whisper transcription.",
    )
    parser.add_argument(
        "--media-id",
        default=None,
        help="Optional media identifier (required when --media-file is provided).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_file_bytes = Path(args.source_file).read_bytes()

    media_bytes: bytes | None = None
    if args.media_file:
        if not args.media_id:
            raise ValueError("--media-id is required when --media-file is provided.")
        media_bytes = Path(args.media_file).read_bytes()

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
    )
    result = pipeline.run(
        doc_id=args.doc_id,
        source_file_id=args.source_file_id,
        source_file_bytes=source_file_bytes,
        media_id=args.media_id,
        media_bytes=media_bytes,
    )

    serialized = json.dumps(result.model_dump(mode="json"), indent=2)
    if args.output_json:
        Path(args.output_json).write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
