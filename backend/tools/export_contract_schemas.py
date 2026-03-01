from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from backend.app.models import (
    ChunkingResult,
    EmbeddingBatchResult,
    GraphData,
    PhaseAIngestionResult,
    RollingState,
    SectionParseResult,
    TOCData,
    TranscriptionResult,
    VisionExtractionResult,
)


SCHEMA_MODELS: dict[str, Type[BaseModel]] = {
    "graph_data.json": GraphData,
    "rolling_state.json": RollingState,
    "toc.json": TOCData,
    "section_parse_result.json": SectionParseResult,
    "transcription_result.json": TranscriptionResult,
    "vision_extraction_result.json": VisionExtractionResult,
    "embedding_batch_result.json": EmbeddingBatchResult,
    "chunking_result.json": ChunkingResult,
    "phase_a_ingestion_result.json": PhaseAIngestionResult,
}


def export_contract_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, model_type in SCHEMA_MODELS.items():
        schema = model_type.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = filename
        target_file = output_dir / filename
        target_file.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {target_file}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    export_contract_schemas(project_root / "backend" / "contracts")
