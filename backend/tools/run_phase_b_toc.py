from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.models import ChunkingResult
from backend.app.pipelines import PhaseBTOCConfig, PhaseBTOCPipeline
from backend.app.reasoning import OpenAITOCReasoningClient, OpenAITOCReasoningConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase B TOC generation.")
    parser.add_argument(
        "--chunking-json",
        required=True,
        help="Path to input ChunkingResult JSON artifact.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional doc_id override (defaults to chunking.doc_id).",
    )
    parser.add_argument(
        "--output-toc-json",
        required=True,
        help="Path to write TOCData JSON artifact.",
    )
    parser.add_argument(
        "--output-meta-json",
        default=None,
        help="Optional path to write LLM metadata JSON artifact.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override (otherwise OPENAI_REASONING_MODEL/env default).",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Optional TOC prompt version override.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=120000,
        help="Max chars to pass into reasoning stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunking_payload = json.loads(Path(args.chunking_json).read_text(encoding="utf-8"))
    chunking = ChunkingResult.model_validate(chunking_payload)
    doc_id = args.doc_id or chunking.doc_id

    config = OpenAITOCReasoningConfig.from_env()
    if args.model or args.prompt_version:
        config = OpenAITOCReasoningConfig(
            model=args.model or config.model,
            prompt_version=args.prompt_version or config.prompt_version,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

    pipeline = PhaseBTOCPipeline(
        reasoning_client=OpenAITOCReasoningClient(config=config),
        config=PhaseBTOCConfig(max_input_chars=args.max_input_chars),
        reasoning_provider="openai",
    )
    result = pipeline.run(doc_id=doc_id, chunking=chunking)

    Path(args.output_toc_json).write_text(
        json.dumps(result.toc.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    if args.output_meta_json:
        Path(args.output_meta_json).write_text(
            json.dumps(
                {
                    "llm_call": result.llm_call.model_dump(mode="json"),
                    "prompt_tag": result.prompt_tag,
                    "prompt_checksum": result.prompt_checksum,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"Wrote TOC ({len(result.toc.sections)} top-level sections) to {args.output_toc_json}"
    )


if __name__ == "__main__":
    main()
