from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
ID_PATTERN = r"^[A-Za-z0-9._:-]+$"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)


class ParseStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class SectionResultStatus(str, Enum):
    ok = "ok"
    partial = "partial"
    failed = "failed"


class ComputeProvider(str, Enum):
    modal = "modal"
    actian = "actian"
    openai = "openai"
    supermemory = "supermemory"


class SourceMaterial(StrictModel):
    doc_id: str = Field(min_length=1)
    section_id: str | None = Field(default=None, min_length=1)
    chunk_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    transcript_timestamps: list[str] = Field(default_factory=list)
    snippet: str | None = None

    @field_validator("chunk_ids", "transcript_timestamps")
    @classmethod
    def dedupe_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)

    @field_validator("page_numbers")
    @classmethod
    def validate_pages(cls, value: list[int]) -> list[int]:
        deduped: list[int] = []
        seen: set[int] = set()
        for page in value:
            if page < 1:
                raise ValueError("page_numbers must be >= 1")
            if page in seen:
                continue
            seen.add(page)
            deduped.append(page)
        return deduped


class ConceptNode(StrictModel):
    id: str = Field(min_length=1, pattern=ID_PATTERN)
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_material: SourceMaterial
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    deep_dive: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)


class ConceptEdgeEvidence(StrictModel):
    historical_doc_id: str | None = Field(default=None, min_length=1)
    current_doc_id: str | None = Field(default=None, min_length=1)
    historical_chunk_ids: list[str] = Field(default_factory=list)
    current_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("historical_chunk_ids", "current_chunk_ids")
    @classmethod
    def normalize_chunk_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)


class ConceptEdge(StrictModel):
    id: str = Field(min_length=1, pattern=ID_PATTERN)
    source: str = Field(min_length=1, pattern=ID_PATTERN)
    target: str = Field(min_length=1, pattern=ID_PATTERN)
    relation: Literal["prerequisite_for"] = "prerequisite_for"
    explanation: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: ConceptEdgeEvidence = Field(default_factory=ConceptEdgeEvidence)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_not_self_edge(self) -> ConceptEdge:
        if self.source == self.target:
            raise ValueError("Edge source and target cannot be the same node.")
        return self


def _build_adjacency(node_ids: set[str], edges: list[ConceptEdge]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    return adjacency


def _find_cycle_path(adjacency: dict[str, list[str]]) -> list[str] | None:
    state: dict[str, int] = {node_id: 0 for node_id in adjacency}
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    cycle_path: list[str] | None = None

    def dfs(node_id: str) -> None:
        nonlocal cycle_path
        state[node_id] = 1
        stack_index[node_id] = len(stack)
        stack.append(node_id)

        for neighbor in adjacency.get(node_id, []):
            if cycle_path is not None:
                return
            neighbor_state = state.get(neighbor, 0)
            if neighbor_state == 0:
                dfs(neighbor)
            elif neighbor_state == 1:
                start = stack_index[neighbor]
                cycle_path = stack[start:] + [neighbor]
                return

        stack.pop()
        stack_index.pop(node_id, None)
        state[node_id] = 2

    for node_id in adjacency:
        if state[node_id] != 0:
            continue
        dfs(node_id)
        if cycle_path is not None:
            break

    return cycle_path


def assert_is_dag(node_ids: set[str], edges: list[ConceptEdge]) -> None:
    adjacency = _build_adjacency(node_ids, edges)
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        indegree[edge.target] = indegree.get(edge.target, 0) + 1

    queue = sorted(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered: list[str] = []

    while queue:
        node_id = queue.pop(0)
        ordered.append(node_id)
        for neighbor in adjacency.get(node_id, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
        queue.sort()

    if len(ordered) != len(node_ids):
        cycle = _find_cycle_path(adjacency)
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise ValueError(f"Cycle detected in graph. Path: {cycle_str}")
        raise ValueError("Cycle detected in graph.")


class GraphData(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    graph_id: str | None = Field(default=None, min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)
    nodes: list[ConceptNode] = Field(default_factory=list)
    edges: list[ConceptEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, nodes: list[ConceptNode]) -> list[ConceptNode]:
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node ids must be unique.")
        return nodes

    @field_validator("edges")
    @classmethod
    def unique_edge_ids(cls, edges: list[ConceptEdge]) -> list[ConceptEdge]:
        edge_ids = [edge.id for edge in edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Edge ids must be unique.")
        return edges

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> GraphData:
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' is not present in nodes.")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' is not present in nodes.")
        assert_is_dag(node_ids=node_ids, edges=self.edges)
        return self


class TOCSection(StrictModel):
    section_id: str = Field(min_length=1, pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    order: int = Field(ge=0)
    chunk_ids: list[str] = Field(default_factory=list)
    source_anchor: str | None = None
    summary: str | None = None
    key_terms: list[str] = Field(default_factory=list)
    children: list["TOCSection"] = Field(default_factory=list)

    @field_validator("chunk_ids", "key_terms")
    @classmethod
    def normalize_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)

    @model_validator(mode="after")
    def validate_children(self) -> TOCSection:
        child_ids = [child.section_id for child in self.children]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("Child section ids must be unique per parent section.")
        child_orders = [child.order for child in self.children]
        if child_orders != sorted(child_orders):
            raise ValueError("Child section order values must be sorted ascending.")
        return self


class TOCData(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)
    sections: list[TOCSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sections")
    @classmethod
    def top_level_sections_must_be_sorted(cls, sections: list[TOCSection]) -> list[TOCSection]:
        orders = [section.order for section in sections]
        if orders != sorted(orders):
            raise ValueError("Top-level section order values must be sorted ascending.")
        return sections

    @model_validator(mode="after")
    def unique_section_ids(self) -> TOCData:
        all_ids: list[str] = []

        def walk(items: list[TOCSection]) -> None:
            for item in items:
                all_ids.append(item.section_id)
                walk(item.children)

        walk(self.sections)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Section ids must be globally unique in TOC.")
        return self


class LLMCallMetadata(StrictModel):
    provider: Literal["openai"] = "openai"
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    request_id: str | None = Field(default=None, min_length=1)


class SimilarityMatch(StrictModel):
    retrieval_provider: Literal["actian"] = "actian"
    historical_concept_id: str = Field(min_length=1, pattern=ID_PATTERN)
    similarity: float = Field(ge=0.0, le=1.0)
    historical_doc_id: str | None = Field(default=None, min_length=1)
    historical_section_id: str | None = Field(default=None, min_length=1)
    historical_chunk_id: str | None = Field(default=None, min_length=1)


class TranscriptSegment(StrictModel):
    segment_id: str = Field(min_length=1, pattern=ID_PATTERN)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_segment_bounds(self) -> TranscriptSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self


class TranscriptionResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    media_id: str = Field(min_length=1)
    provider: Literal["modal"] = "modal"
    runtime: Literal["serverless_gpu"] = "serverless_gpu"
    model_name: str = Field(min_length=1)
    language: str | None = Field(default=None, min_length=1)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    transcript_text: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionPageExtraction(StrictModel):
    page_number: int = Field(ge=1)
    raw_text: str = Field(default="")
    image_descriptions: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("image_descriptions", "chunk_ids")
    @classmethod
    def normalize_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)


class VisionExtractionResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    source_file_id: str = Field(min_length=1)
    provider: Literal["modal"] = "modal"
    runtime: Literal["serverless_gpu"] = "serverless_gpu"
    model_name: str = Field(min_length=1)
    pages: list[VisionPageExtraction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkEmbedding(StrictModel):
    chunk_id: str = Field(min_length=1, pattern=ID_PATTERN)
    vector: list[float] = Field(min_length=1)
    vector_dim: int = Field(ge=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vector_dimension(self) -> ChunkEmbedding:
        if len(self.vector) != self.vector_dim:
            raise ValueError("vector_dim must equal len(vector).")
        return self


class EmbeddingBatchResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    provider: Literal["modal"] = "modal"
    runtime: Literal["serverless_gpu"] = "serverless_gpu"
    model_name: str = Field(min_length=1)
    embeddings: list[ChunkEmbedding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawTextChunk(StrictModel):
    chunk_id: str = Field(min_length=1, pattern=ID_PATTERN)
    doc_id: str = Field(min_length=1)
    source_type: Literal["transcript", "vision_text", "vision_image_description", "manual"] = (
        "manual"
    )
    order: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_estimate: int = Field(ge=1)
    section_hint: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    source_time_start_seconds: float | None = Field(default=None, ge=0.0)
    source_time_end_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_bounds(self) -> RawTextChunk:
        if (
            self.source_time_start_seconds is not None
            and self.source_time_end_seconds is not None
            and self.source_time_end_seconds <= self.source_time_start_seconds
        ):
            raise ValueError("source_time_end_seconds must be > source_time_start_seconds.")
        return self


class ChunkingResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    chunks: list[RawTextChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunks")
    @classmethod
    def validate_chunk_uniqueness_and_order(
        cls, chunks: list[RawTextChunk]
    ) -> list[RawTextChunk]:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk ids must be unique.")

        order_values = [chunk.order for chunk in chunks]
        if order_values != sorted(order_values):
            raise ValueError("Chunks must be sorted in ascending order.")
        return chunks


class PhaseAIngestionResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    doc_id: str = Field(min_length=1)
    chunking: ChunkingResult
    embeddings: EmbeddingBatchResult
    stored_chunk_count: int = Field(ge=0)
    stored_embedding_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionConcept(StrictModel):
    concept_id: str = Field(min_length=1, pattern=ID_PATTERN)
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_text: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    historical_matches: list[SimilarityMatch] = Field(default_factory=list)

    @field_validator("aliases", "source_chunk_ids")
    @classmethod
    def normalize_list_fields(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)


class SectionEdgeCandidate(StrictModel):
    source_concept_id: str = Field(min_length=1, pattern=ID_PATTERN)
    target_concept_id: str = Field(min_length=1, pattern=ID_PATTERN)
    relation: Literal["prerequisite_for"] = "prerequisite_for"
    accepted: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    evidence: ConceptEdgeEvidence = Field(default_factory=ConceptEdgeEvidence)

    @model_validator(mode="after")
    def validate_not_self_edge(self) -> SectionEdgeCandidate:
        if self.source_concept_id == self.target_concept_id:
            raise ValueError("source_concept_id and target_concept_id cannot be identical.")
        return self


class SectionParseResult(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1, pattern=ID_PATTERN)
    section_order: int = Field(ge=0)
    section_title: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(default_factory=list)
    concepts: list[SectionConcept] = Field(default_factory=list)
    edge_candidates: list[SectionEdgeCandidate] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    status: SectionResultStatus = SectionResultStatus.ok
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_chunk_ids", "warnings", "errors")
    @classmethod
    def normalize_string_list_fields(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return dedupe_preserve_order(cleaned)

    @field_validator("concepts")
    @classmethod
    def unique_concept_ids(cls, concepts: list[SectionConcept]) -> list[SectionConcept]:
        concept_ids = [concept.concept_id for concept in concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("Section concept ids must be unique.")
        return concepts

    @model_validator(mode="after")
    def validate_edge_candidates(self) -> SectionParseResult:
        concept_ids = {concept.concept_id for concept in self.concepts}
        for edge in self.edge_candidates:
            if edge.target_concept_id not in concept_ids:
                raise ValueError(
                    "Each edge candidate target_concept_id must reference a concept from this section."
                )
        return self


class SectionProgress(StrictModel):
    section_id: str = Field(min_length=1, pattern=ID_PATTERN)
    order: int = Field(ge=0)
    status: ParseStatus = ParseStatus.pending
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_time_bounds(self) -> SectionProgress:
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at.")
        return self


class RollingState(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=utc_now)
    current_section_index: int = Field(default=0, ge=0)
    sections: list[SectionProgress] = Field(default_factory=list)
    nodes: list[ConceptNode] = Field(default_factory=list)
    edges: list[ConceptEdge] = Field(default_factory=list)
    concept_alias_index: dict[str, str] = Field(default_factory=dict)
    parse_log: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("concept_alias_index")
    @classmethod
    def validate_alias_index(cls, alias_index: dict[str, str]) -> dict[str, str]:
        for alias, concept_id in alias_index.items():
            if not alias.strip():
                raise ValueError("Alias keys in concept_alias_index cannot be empty.")
            if not concept_id.strip():
                raise ValueError("Alias values in concept_alias_index cannot be empty.")
        return alias_index

    @field_validator("parse_log")
    @classmethod
    def normalize_parse_log(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned

    @model_validator(mode="after")
    def validate_state(self) -> RollingState:
        if self.current_section_index > len(self.sections):
            raise ValueError("current_section_index cannot exceed number of known sections.")

        in_progress_count = sum(
            1 for section in self.sections if section.status == ParseStatus.in_progress
        )
        if in_progress_count > 1:
            raise ValueError("Only one section may be in progress at a time.")

        graph_snapshot = GraphData(
            schema_version=self.schema_version,
            generated_at=self.updated_at,
            nodes=self.nodes,
            edges=self.edges,
        )
        node_ids = {node.id for node in graph_snapshot.nodes}
        for alias, concept_id in self.concept_alias_index.items():
            if concept_id not in node_ids:
                raise ValueError(
                    f"Alias '{alias}' points to unknown concept id '{concept_id}'."
                )

        return self


TOCSection.model_rebuild()


__all__ = [
    "SCHEMA_VERSION",
    "ChunkingResult",
    "ChunkEmbedding",
    "ConceptEdge",
    "ConceptEdgeEvidence",
    "ConceptNode",
    "ComputeProvider",
    "EmbeddingBatchResult",
    "GraphData",
    "LLMCallMetadata",
    "ParseStatus",
    "PhaseAIngestionResult",
    "RawTextChunk",
    "RollingState",
    "SectionConcept",
    "SectionEdgeCandidate",
    "SectionParseResult",
    "SectionProgress",
    "SectionResultStatus",
    "SimilarityMatch",
    "SourceMaterial",
    "TranscriptSegment",
    "TranscriptionResult",
    "TOCData",
    "TOCSection",
    "VisionExtractionResult",
    "VisionPageExtraction",
    "assert_is_dag",
]
