"""
python backend/tools/run_phase_b_toc.py \
  --phase-a-json artifacts/phase_a_result.json \
  --output-toc-json artifacts/toc.json \
  --output-meta-json artifacts/toc_meta.json 
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from backend.app.config import load_env_file, load_hyperparameters
from backend.app.logging_utils import configure_logging
from backend.app.models import ChunkingResult
from backend.app.pipelines import PhaseBTOCConfig, PhaseBTOCPipeline
from backend.app.reasoning import OpenAITOCReasoningClient, OpenAITOCReasoningConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase B TOC generation.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--chunking-json",
        default=None,
        help="Path to input ChunkingResult JSON artifact.",
    )
    input_group.add_argument(
        "--phase-a-json",
        default=None,
        help="Path to input PhaseAIngestionResult JSON artifact (contains `chunking`).",
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
        "--model-profile",
        choices=["test", "demo"],
        default="test",
        help="Model profile from hyperparameters JSON when --model is not provided.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Optional TOC prompt version override.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=None,
        help="Max chars to pass into reasoning stage.",
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
        logger.info("phase_b_toc.env_loaded keys=%d path=%s", len(loaded), args.env_file)

    logger.info("phase_b_toc.start model_profile=%s", args.model_profile)
    if args.phase_a_json:
        phase_a_payload = json.loads(Path(args.phase_a_json).read_text(encoding="utf-8"))
        if "chunking" not in phase_a_payload:
            raise ValueError("phase-a-json payload is missing `chunking` key.")
        chunking_payload = phase_a_payload["chunking"]
        logger.info("phase_b_toc.input_source=phase_a_json path=%s", args.phase_a_json)
    else:
        chunking_payload = json.loads(Path(args.chunking_json).read_text(encoding="utf-8"))
        logger.info("phase_b_toc.input_source=chunking_json path=%s", args.chunking_json)
    chunking = ChunkingResult.model_validate(chunking_payload)
    doc_id = args.doc_id or chunking.doc_id

    hyperparams = load_hyperparameters(args.hyperparams_json)
    toc_hp = hyperparams.phase_b.toc_generation
    env_model = os.getenv("OPENAI_REASONING_MODEL", "").strip()
    env_prompt_version = os.getenv("TOC_PROMPT_VERSION", "").strip()
    env_temperature = os.getenv("OPENAI_REASONING_TEMPERATURE", "").strip()
    env_max_tokens = os.getenv("OPENAI_REASONING_MAX_OUTPUT_TOKENS", "").strip()

    profile_model = toc_hp.model_test if args.model_profile == "test" else toc_hp.model_demo
    model = args.model or env_model or profile_model
    prompt_version = args.prompt_version or env_prompt_version or toc_hp.prompt_version
    temperature = float(env_temperature) if env_temperature else toc_hp.temperature
    max_output_tokens = int(env_max_tokens) if env_max_tokens else toc_hp.max_output_tokens

    config = OpenAITOCReasoningConfig(
        model=model,
        prompt_version=prompt_version,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    max_input_chars = args.max_input_chars or toc_hp.max_input_chars
    logger.info(
        "phase_b_toc.config model=%s prompt_version=%s max_input_chars=%d",
        model,
        prompt_version,
        max_input_chars,
    )

    pipeline = PhaseBTOCPipeline(
        reasoning_client=OpenAITOCReasoningClient(config=config),
        config=PhaseBTOCConfig(max_input_chars=max_input_chars),
        reasoning_provider="openai",
    )
    result = pipeline.run(doc_id=doc_id, chunking=chunking)

    Path(args.output_toc_json).write_text(
        json.dumps(result.toc.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("phase_b_toc.output_toc_written path=%s", args.output_toc_json)

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
        logger.info("phase_b_toc.output_meta_written path=%s", args.output_meta_json)

    print(
        f"Wrote TOC ({len(result.toc.sections)} top-level sections) to {args.output_toc_json}"
    )
    logger.info(
        "phase_b_toc.finish doc_id=%s top_level_sections=%d",
        result.toc.doc_id,
        len(result.toc.sections),
    )


if __name__ == "__main__":
    main()
