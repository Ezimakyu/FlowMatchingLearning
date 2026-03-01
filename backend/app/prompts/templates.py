from __future__ import annotations

from hashlib import sha256
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt_template: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)

    @property
    def tag(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def checksum(self) -> str:
        payload = "||".join(
            [
                self.name,
                self.version,
                self.system_prompt,
                self.user_prompt_template,
                self.output_contract,
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def render_user_prompt(self, **kwargs: Any) -> str:
        return Template(self.user_prompt_template).substitute(**kwargs)


TOC_GENERATION_V1 = PromptTemplate(
    name="toc_generation",
    version="2026-02-28.v1",
    description="Generate a strict hierarchical TOC from a raw course document.",
    system_prompt=(
        "You are a curriculum parser. Extract a hierarchical table of contents from the "
        "provided course material. Return JSON only. Do not include markdown, prose, or "
        "backticks."
    ),
    user_prompt_template=(
        "Document id: $doc_id\n"
        "Schema version: $schema_version\n"
        "Output contract: TOCData\n\n"
        "You must output valid JSON matching TOCData fields.\n"
        "Rules:\n"
        "- section_id must be stable snake_case.\n"
        "- order must be ascending at each level.\n"
        "- children should preserve parent-child hierarchy.\n"
        "- Keep titles faithful to source material.\n\n"
        "Document text:\n"
        "$document_text\n"
    ),
    output_contract="TOCData",
)

TOC_GENERATION_V2 = PromptTemplate(
    name="toc_generation",
    version="2026-02-28.v2",
    description=(
        "Generate a strict hierarchical TOC from extracted course text. "
        "This is a reasoning-only prompt for OpenAI."
    ),
    system_prompt=(
        "You are the reasoning engine in a hybrid pipeline. Your provider is OpenAI. "
        "Do not perform transcription, vision/OCR extraction, or embedding generation. "
        "Those are pre-computed by Modal ingestion services. "
        "Extract a hierarchical table of contents from the provided text and return JSON only."
    ),
    user_prompt_template=(
        "Document id: $doc_id\n"
        "Schema version: $schema_version\n"
        "Output contract: TOCData\n"
        "Boundary: reasoning-only (OpenAI), no ingestion tasks.\n\n"
        "You must output valid JSON matching TOCData fields.\n"
        "Rules:\n"
        "- section_id must be stable snake_case.\n"
        "- order must be ascending at each level.\n"
        "- children should preserve parent-child hierarchy.\n"
        "- Keep titles faithful to source material.\n\n"
        "Extracted document text:\n"
        "$document_text\n"
    ),
    output_contract="TOCData",
)

SECTION_CONCEPT_EXTRACTION_V1 = PromptTemplate(
    name="section_concept_extraction",
    version="2026-02-28.v1",
    description="Extract concepts from one section using rolling state context.",
    system_prompt=(
        "You are a concept extraction engine for a dependency graph pipeline. "
        "Output JSON only and strictly follow the requested contract."
    ),
    user_prompt_template=(
        "Document id: $doc_id\n"
        "Section id: $section_id\n"
        "Section title: $section_title\n"
        "Schema version: $schema_version\n"
        "Output contract: SectionParseResult\n\n"
        "Use only the provided section text plus rolling state summary.\n"
        "Do not hallucinate concepts outside the section.\n\n"
        "Rolling state JSON:\n"
        "$rolling_state_json\n\n"
        "Section text:\n"
        "$section_text\n"
    ),
    output_contract="SectionParseResult",
)

SECTION_CONCEPT_EXTRACTION_V2 = PromptTemplate(
    name="section_concept_extraction",
    version="2026-02-28.v2",
    description=(
        "Extract concepts from one section using rolling state context. "
        "Reasoning-only prompt for OpenAI."
    ),
    system_prompt=(
        "You are the OpenAI reasoning stage in a hybrid microservices architecture. "
        "Ingestion (transcription, VLM extraction, embeddings) is already done by Modal GPU services. "
        "Use only supplied section text and rolling state. Return strict JSON only."
    ),
    user_prompt_template=(
        "Document id: $doc_id\n"
        "Section id: $section_id\n"
        "Section title: $section_title\n"
        "Schema version: $schema_version\n"
        "Output contract: SectionParseResult\n"
        "Boundary: reasoning-only (OpenAI), no ingestion tasks.\n\n"
        "Use only the provided section text plus rolling state summary.\n"
        "Do not hallucinate concepts outside the section.\n\n"
        "Rolling state JSON:\n"
        "$rolling_state_json\n\n"
        "Section text:\n"
        "$section_text\n"
    ),
    output_contract="SectionParseResult",
)

SECTION_CONCEPT_EXTRACTION_V3 = PromptTemplate(
    name="section_concept_extraction",
    version="2026-03-01.v3",
    description=(
        "Extract canonical concepts from one section with rolling-state context. "
        "Tuned for recall while preserving strict evidence grounding."
    ),
    system_prompt=(
        "You are the OpenAI reasoning stage for section parsing in a prerequisite-graph pipeline. "
        "Ingestion and embeddings are already done by separate services. "
        "Use only the provided section text and rolling state summary. Return strict JSON only."
    ),
    user_prompt_template=(
        "Document id: $doc_id\n"
        "Section id: $section_id\n"
        "Section title: $section_title\n"
        "Schema version: $schema_version\n"
        "Output contract: SectionParseResult\n"
        "Boundary: reasoning-only (OpenAI), no ingestion tasks.\n\n"
        "Task:\n"
        "- Extract canonical concepts introduced or materially used in this section.\n"
        "- Prefer broad lecture-level topics over micro-techniques or one-off details.\n"
        "- Prefer stable, reusable labels (avoid overly local phrasing).\n"
        "- Include direct evidence_text snippets when available.\n\n"
        "Rules:\n"
        "- Use only the provided section text plus rolling state summary.\n"
        "- Do not hallucinate concepts outside this section text.\n"
        "- Exclude trivial terms unless they are true instructional prerequisites.\n"
        "- Keep the concept list sparse: usually 3-6 core concepts for a typical section.\n"
        "- confidence must be in [0.0, 1.0].\n\n"
        "Rolling state JSON:\n"
        "$rolling_state_json\n\n"
        "Section text:\n"
        "$section_text\n"
    ),
    output_contract="SectionParseResult",
)

EDGE_VALIDATION_V1 = PromptTemplate(
    name="edge_validation",
    version="2026-02-28.v1",
    description="Validate whether a historical concept is a prerequisite of a new concept.",
    system_prompt=(
        "You decide if a prerequisite edge exists between concepts. "
        "Direction convention: source=prerequisite, target=dependent. "
        "Return JSON only."
    ),
    user_prompt_template=(
        "Schema version: $schema_version\n"
        "Output contract: SectionEdgeCandidate\n\n"
        "New concept JSON:\n"
        "$new_concept_json\n\n"
        "Historical concept JSON:\n"
        "$historical_concept_json\n\n"
        "Supporting evidence JSON:\n"
        "$supporting_evidence_json\n\n"
        "Rules:\n"
        "- accepted=true only if evidence supports a dependency.\n"
        "- source_concept_id must be the prerequisite concept id.\n"
        "- target_concept_id must be the dependent concept id.\n"
        "- confidence must be in [0.0, 1.0].\n"
        "- explanation must cite why dependency exists.\n"
    ),
    output_contract="SectionEdgeCandidate",
)

EDGE_VALIDATION_V2 = PromptTemplate(
    name="edge_validation",
    version="2026-02-28.v2",
    description=(
        "Validate prerequisite dependency between concepts from Actian matches. "
        "Reasoning-only prompt for OpenAI."
    ),
    system_prompt=(
        "You are the OpenAI reasoning engine for dependency validation. "
        "Similarity retrieval results come from Actian VectorDB. "
        "Do not perform embedding, transcription, or vision extraction. Return JSON only."
    ),
    user_prompt_template=(
        "Schema version: $schema_version\n"
        "Output contract: SectionEdgeCandidate\n"
        "Boundary: reasoning-only (OpenAI).\n\n"
        "New concept JSON:\n"
        "$new_concept_json\n\n"
        "Historical concept JSON (retrieved via Actian):\n"
        "$historical_concept_json\n\n"
        "Supporting evidence JSON:\n"
        "$supporting_evidence_json\n\n"
        "Rules:\n"
        "- accepted=true only if evidence supports a dependency.\n"
        "- source_concept_id must be the prerequisite concept id.\n"
        "- target_concept_id must be the dependent concept id.\n"
        "- confidence must be in [0.0, 1.0].\n"
        "- explanation must cite why dependency exists.\n"
    ),
    output_contract="SectionEdgeCandidate",
)

EDGE_VALIDATION_V3 = PromptTemplate(
    name="edge_validation",
    version="2026-03-01.v3",
    description=(
        "Validate prerequisite dependency between a historical concept and a new concept. "
        "Tuned to reduce false negatives while keeping evidence-based decisions."
    ),
    system_prompt=(
        "You are the OpenAI reasoning engine for dependency validation in a DAG. "
        "Direction convention is strict: source=prerequisite, target=dependent. "
        "Be evidence-grounded, but do not be overly conservative when a dependency is clear. "
        "Return JSON only."
    ),
    user_prompt_template=(
        "Schema version: $schema_version\n"
        "Output contract: SectionEdgeCandidate\n"
        "Boundary: reasoning-only (OpenAI).\n\n"
        "New concept JSON:\n"
        "$new_concept_json\n\n"
        "Historical concept JSON (retrieved via Actian):\n"
        "$historical_concept_json\n\n"
        "Supporting evidence JSON:\n"
        "$supporting_evidence_json\n\n"
        "Decision guidance:\n"
        "- accepted=true when historical concept is needed to define, derive, apply, or correctly "
        "interpret the new concept.\n"
        "- accepted=false when concepts are merely related, parallel, or reversed in direction.\n"
        "- source_concept_id must be the prerequisite concept id.\n"
        "- target_concept_id must be the dependent concept id.\n"
        "- confidence must be in [0.0, 1.0].\n"
        "- explanation should briefly cite why the dependency does or does not hold.\n"
        "- Do not reject solely due to moderate similarity if conceptual dependency is explicit.\n"
    ),
    output_contract="SectionEdgeCandidate",
)


PROMPTS: tuple[PromptTemplate, ...] = (
    TOC_GENERATION_V1,
    TOC_GENERATION_V2,
    SECTION_CONCEPT_EXTRACTION_V1,
    SECTION_CONCEPT_EXTRACTION_V2,
    SECTION_CONCEPT_EXTRACTION_V3,
    EDGE_VALIDATION_V1,
    EDGE_VALIDATION_V2,
    EDGE_VALIDATION_V3,
)

PROMPT_REGISTRY: dict[str, PromptTemplate] = {prompt.tag: prompt for prompt in PROMPTS}
LATEST_PROMPTS: dict[str, PromptTemplate] = {prompt.name: prompt for prompt in PROMPTS}


def get_prompt(name: str, version: str | None = None) -> PromptTemplate:
    if version is None:
        if name not in LATEST_PROMPTS:
            raise KeyError(f"Unknown prompt name '{name}'.")
        return LATEST_PROMPTS[name]

    key = f"{name}:{version}"
    if key not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt tag '{key}'.")
    return PROMPT_REGISTRY[key]


def render_prompt(name: str, version: str | None = None, **kwargs: Any) -> dict[str, str]:
    prompt = get_prompt(name=name, version=version)
    return {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_tag": prompt.tag,
        "prompt_checksum": prompt.checksum,
        "output_contract": prompt.output_contract,
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.render_user_prompt(**kwargs),
    }


__all__ = [
    "EDGE_VALIDATION_V1",
    "EDGE_VALIDATION_V2",
    "EDGE_VALIDATION_V3",
    "LATEST_PROMPTS",
    "PROMPT_REGISTRY",
    "PROMPTS",
    "PromptTemplate",
    "SECTION_CONCEPT_EXTRACTION_V1",
    "SECTION_CONCEPT_EXTRACTION_V2",
    "SECTION_CONCEPT_EXTRACTION_V3",
    "TOC_GENERATION_V1",
    "TOC_GENERATION_V2",
    "get_prompt",
    "render_prompt",
]
