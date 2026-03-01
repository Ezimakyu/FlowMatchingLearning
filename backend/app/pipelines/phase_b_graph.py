from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
import uuid
from typing import Any, Protocol

from backend.app.compute_boundaries import assert_compute_boundary
from backend.app.models import (
    ChunkEmbedding,
    ChunkingResult,
    ConceptEdge,
    ConceptNode,
    EmbeddingBatchResult,
    GraphData,
    ParseStatus,
    RollingState,
    SectionConcept,
    SectionEdgeCandidate,
    SectionParseResult,
    SectionProgress,
    SimilarityMatch,
    SourceMaterial,
    TOCData,
    TOCSection,
)
from backend.app.reasoning.section_reasoning import (
    EdgeValidationOutput,
    SectionConceptExtractionOutput,
    SectionReasoningClient,
)


@dataclass(frozen=True)
class FlatTOCSection:
    section_id: str
    title: str
    order: int
    chunk_ids: list[str]
    path: str


@dataclass(frozen=True)
class PhaseBGraphConfig:
    top_k_historical_matches: int = 12
    similarity_threshold: float = 0.72
    similarity_fallback_threshold: float = 0.62
    edge_acceptance_confidence_threshold: float = 0.60
    retrieval_overfetch_multiplier: int = 4
    max_section_chars_per_call: int = 30_000
    max_sections_to_parse: int = 0
    max_llm_concepts_per_section: int = 6
    max_state_nodes_in_context: int = 200
    max_historical_nodes_for_local_similarity: int = 400
    seed_core_nodes_from_toc: bool = True
    max_seed_core_nodes: int = 12
    freeze_node_set_after_seed: bool = True


@dataclass(frozen=True)
class PhaseBGraphOutput:
    graph: GraphData
    rolling_state: RollingState
    section_results: list[SectionParseResult]


class ActianSimilaritySearchClient(Protocol):
    def similarity_search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        model_name: str = "BAAI/bge-m3",
        candidate_limit: int = 0,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def flatten_toc_sections(sections: list[TOCSection]) -> list[FlatTOCSection]:
    flat: list[FlatTOCSection] = []

    def visit(items: list[TOCSection], *, parent_path: str) -> None:
        ordered_items = sorted(items, key=lambda item: item.order)
        for section in ordered_items:
            path = f"{parent_path}/{section.section_id}" if parent_path else section.section_id
            flat.append(
                FlatTOCSection(
                    section_id=section.section_id,
                    title=section.title,
                    order=section.order,
                    chunk_ids=list(section.chunk_ids),
                    path=path,
                )
            )
            visit(section.children, parent_path=path)

    visit(sections, parent_path="")
    return flat


def build_section_text(
    *,
    chunking: ChunkingResult,
    chunk_ids: list[str],
    max_chars: int,
) -> str:
    if max_chars < 200:
        raise ValueError("max_chars must be >= 200.")

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunking.chunks}
    lines: list[str] = []
    total_chars = 0
    for chunk_id in chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        prefix = f"[chunk_id={chunk.chunk_id} source={chunk.source_type} order={chunk.order}] "
        line = prefix + chunk.text.strip()
        projected = total_chars + len(line) + 1
        if projected > max_chars:
            lines.append("[TRUNCATED_FOR_CONTEXT_LIMIT]")
            break
        lines.append(line)
        total_chars = projected
    return "\n".join(lines).strip()


def _safe_slug(value: str, *, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "concept"
    return cleaned[:max_len]


def _build_edge_id(*, source: str, target: str) -> str:
    digest = hashlib.blake2b(f"{source}->{target}".encode("utf-8"), digest_size=8).hexdigest()
    return f"edge:{digest}"


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average an empty list of vectors.")
    dim = len(vectors[0])
    if dim == 0:
        raise ValueError("Vectors must be non-empty.")
    totals = [0.0] * dim
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError("All vectors must share the same dimension.")
        for index, value in enumerate(vector):
            totals[index] += float(value)
    count = float(len(vectors))
    return [value / count for value in totals]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left[index] * right[index] for index in range(len(left)))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    similarity = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, similarity))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


class PhaseBGraphPipeline:
    def __init__(
        self,
        *,
        reasoning_client: SectionReasoningClient,
        storage_client: ActianSimilaritySearchClient,
        config: PhaseBGraphConfig | None = None,
        reasoning_provider: str = "openai",
        storage_provider: str = "actian",
    ) -> None:
        self.reasoning_client = reasoning_client
        self.storage_client = storage_client
        self.config = config or PhaseBGraphConfig()
        self.reasoning_provider = reasoning_provider
        self.storage_provider = storage_provider

    def run(
        self,
        *,
        doc_id: str,
        toc: TOCData,
        chunking: ChunkingResult,
        embeddings: EmbeddingBatchResult,
        job_id: str | None = None,
    ) -> PhaseBGraphOutput:
        logger = logging.getLogger(__name__)
        self._validate_inputs(doc_id=doc_id, toc=toc, chunking=chunking, embeddings=embeddings)
        assert_compute_boundary("reasoning.section_concept_extraction", self.reasoning_provider)
        assert_compute_boundary("reasoning.edge_validation", self.reasoning_provider)
        assert_compute_boundary("storage.similarity_search", self.storage_provider)

        flat_sections = flatten_toc_sections(toc.sections)
        if not flat_sections:
            raise ValueError("TOC contains no sections; cannot build graph.")
        original_section_count = len(flat_sections)
        if self.config.max_sections_to_parse > 0 and len(flat_sections) > self.config.max_sections_to_parse:
            flat_sections = flat_sections[: self.config.max_sections_to_parse]
            logger.info(
                "phase_b_graph.section_limit_applied doc_id=%s selected_sections=%d total_sections=%d",
                doc_id,
                len(flat_sections),
                original_section_count,
            )
        resolved_job_id = job_id or f"job_{doc_id}_{uuid.uuid4().hex[:8]}"

        rolling_state = RollingState(
            job_id=resolved_job_id,
            doc_id=doc_id,
            sections=[
                SectionProgress(section_id=section.section_id, order=index)
                for index, section in enumerate(flat_sections)
            ],
            metadata={
                "toc_section_count": original_section_count,
                "toc_section_count_active": len(flat_sections),
                "reasoning_provider": self.reasoning_provider,
                "storage_provider": self.storage_provider,
            },
        )
        section_results: list[SectionParseResult] = []
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunking.chunks}
        embedding_by_chunk = {item.chunk_id: item for item in embeddings.embeddings}
        seeded_node_count = 0
        if self.config.seed_core_nodes_from_toc:
            seeded_node_count = self._seed_core_nodes_from_toc(
                state=rolling_state,
                flat_sections=flat_sections,
                chunk_by_id=chunk_by_id,
            )
            if seeded_node_count:
                logger.info(
                    "phase_b_graph.seeded_core_nodes doc_id=%s seeded_nodes=%d",
                    doc_id,
                    seeded_node_count,
                )

        logger.info(
            (
                "phase_b_graph.pipeline_start doc_id=%s job_id=%s sections=%d chunks=%d "
                "seeded_nodes=%d freeze_node_set=%s"
            ),
            doc_id,
            resolved_job_id,
            len(flat_sections),
            len(chunking.chunks),
            seeded_node_count,
            self.config.freeze_node_set_after_seed,
        )
        for index, section in enumerate(flat_sections):
            self._mark_section_in_progress(state=rolling_state, index=index)
            section_result = self._run_single_section(
                state=rolling_state,
                section=section,
                section_index=index,
                total_sections=len(flat_sections),
                chunking=chunking,
                chunk_by_id=chunk_by_id,
                embedding_by_chunk=embedding_by_chunk,
                embedding_model_name=embeddings.model_name,
            )
            section_results.append(section_result)

            progress = rolling_state.sections[index]
            if progress.status == ParseStatus.failed:
                logger.warning(
                    "phase_b_graph.section_failed doc_id=%s section_id=%s",
                    doc_id,
                    section.section_id,
                )
            else:
                logger.info(
                    "phase_b_graph.section_complete doc_id=%s section_id=%s concepts=%d edges=%d",
                    doc_id,
                    section.section_id,
                    len(section_result.concepts),
                    len(section_result.edge_candidates),
                )
            rolling_state.current_section_index = index + 1
            rolling_state.updated_at = _utc_now()

        graph = GraphData(
            graph_id=f"graph_{doc_id}",
            nodes=rolling_state.nodes,
            edges=rolling_state.edges,
            metadata={
                "job_id": rolling_state.job_id,
                "doc_id": rolling_state.doc_id,
                "sections_total": len(rolling_state.sections),
                "sections_completed": sum(
                    1 for section in rolling_state.sections if section.status == ParseStatus.completed
                ),
                "sections_failed": sum(
                    1 for section in rolling_state.sections if section.status == ParseStatus.failed
                ),
            },
        )
        logger.info(
            "phase_b_graph.pipeline_finish doc_id=%s nodes=%d edges=%d",
            doc_id,
            len(graph.nodes),
            len(graph.edges),
        )
        return PhaseBGraphOutput(
            graph=graph,
            rolling_state=rolling_state,
            section_results=section_results,
        )

    def _run_single_section(
        self,
        *,
        state: RollingState,
        section: FlatTOCSection,
        section_index: int,
        total_sections: int,
        chunking: ChunkingResult,
        chunk_by_id: dict[str, Any],
        embedding_by_chunk: dict[str, ChunkEmbedding],
        embedding_model_name: str,
    ) -> SectionParseResult:
        logger = logging.getLogger(__name__)
        warnings: list[str] = []
        errors: list[str] = []
        llm_calls: list[Any] = []
        concepts: list[SectionConcept] = []
        concepts_by_id: dict[str, SectionConcept] = {}
        concept_order: list[str] = []
        edge_candidates: list[SectionEdgeCandidate] = []
        edge_validation_attempts = 0
        accepted_edge_count = 0
        rejected_by_model_count = 0
        rejected_by_confidence_count = 0
        rejected_duplicate_count = 0
        rejected_cycle_count = 0

        resolved_chunk_ids = self._resolve_section_chunk_ids(
            section=section,
            section_index=section_index,
            total_sections=total_sections,
            chunking=chunking,
            chunk_by_id=chunk_by_id,
        )
        if not resolved_chunk_ids:
            warnings.append("No chunks resolved for section; using empty text context.")
        section_text = build_section_text(
            chunking=chunking,
            chunk_ids=resolved_chunk_ids,
            max_chars=self.config.max_section_chars_per_call,
        )
        if not section_text:
            section_text = "[NO_SECTION_TEXT_AVAILABLE]"

        historical_node_ids = [
            node.id for node in state.nodes if not bool(node.metadata.get("toc_seed"))
        ]
        historical_node_id_set = set(historical_node_ids)
        historical_chunk_index = self._build_chunk_to_concept_index(
            nodes=state.nodes,
            allowed_concept_ids=historical_node_id_set,
        )
        node_by_id = {node.id: node for node in state.nodes}
        historical_concept_vectors = self._build_historical_concept_vectors(
            historical_concept_ids=historical_node_ids,
            node_by_id=node_by_id,
            embedding_by_chunk=embedding_by_chunk,
        )
        allow_node_creation = not self.config.freeze_node_set_after_seed or not state.nodes

        try:
            extraction = self._extract_concepts_for_section(
                state=state,
                section=section,
                section_text=section_text,
            )
            llm_calls.append(extraction.llm_call)
            warnings.extend(extraction.warnings)
            concept_cap = max(1, self.config.max_llm_concepts_per_section)
            extracted_concepts = extraction.concepts[:concept_cap]
            if len(extraction.concepts) > concept_cap:
                warnings.append(
                    (
                        "Trimmed extracted concepts from "
                        f"{len(extraction.concepts)} to {concept_cap} to keep core-topic focus."
                    )
                )
                logger.info(
                    "phase_b_graph.section_concept_cap section_id=%s extracted=%d capped_to=%d",
                    section.section_id,
                    len(extraction.concepts),
                    concept_cap,
                )

            for extracted_concept in extracted_concepts:
                concept = self._normalize_section_concept(
                    concept=extracted_concept,
                    fallback_chunk_ids=resolved_chunk_ids,
                    chunk_by_id=chunk_by_id,
                )
                canonical_id = self._resolve_or_create_node(
                    state=state,
                    node_by_id=node_by_id,
                    concept=concept,
                    section=section,
                    chunk_by_id=chunk_by_id,
                    allow_node_creation=allow_node_creation,
                )
                concept = SectionConcept(
                    concept_id=canonical_id,
                    label=concept.label,
                    summary=concept.summary,
                    aliases=concept.aliases,
                    source_chunk_ids=concept.source_chunk_ids,
                    evidence_text=concept.evidence_text,
                    confidence=concept.confidence,
                    historical_matches=[],
                )
                if canonical_id in concepts_by_id:
                    concepts_by_id[canonical_id] = self._merge_section_concepts(
                        existing=concepts_by_id[canonical_id],
                        incoming=concept,
                    )
                    continue
                concept_matches = self._retrieve_historical_matches(
                    concept=concept,
                    historical_chunk_index=historical_chunk_index,
                    node_by_id=node_by_id,
                    embedding_by_chunk=embedding_by_chunk,
                    embedding_model_name=embedding_model_name,
                    historical_concept_vectors=historical_concept_vectors,
                )
                concept.historical_matches = concept_matches
                concepts_by_id[canonical_id] = concept
                concept_order.append(canonical_id)

                for match in concept_matches:
                    historical_node = node_by_id.get(match.historical_concept_id)
                    if historical_node is None:
                        continue
                    edge_validation_attempts += 1
                    edge_output = self._validate_edge(
                        doc_id=state.doc_id,
                        concept=concept,
                        historical_node=historical_node,
                        match=match,
                    )
                    llm_calls.append(edge_output.llm_call)

                    normalized_candidate, candidate_warning = self._normalize_edge_candidate(
                        candidate=edge_output.candidate,
                        expected_source=historical_node.id,
                        expected_target=concept.concept_id,
                        doc_id=state.doc_id,
                        historical_chunk_id=match.historical_chunk_id,
                        current_chunk_ids=concept.source_chunk_ids,
                    )
                    if candidate_warning:
                        warnings.append(candidate_warning)
                    edge_candidates.append(normalized_candidate)

                    if not normalized_candidate.accepted:
                        rejected_by_model_count += 1
                        continue
                    if (
                        normalized_candidate.confidence
                        < self.config.edge_acceptance_confidence_threshold
                    ):
                        rejected_by_confidence_count += 1
                        continue
                    edge = ConceptEdge(
                        id=_build_edge_id(
                            source=normalized_candidate.source_concept_id,
                            target=normalized_candidate.target_concept_id,
                        ),
                        source=normalized_candidate.source_concept_id,
                        target=normalized_candidate.target_concept_id,
                        relation="prerequisite_for",
                        explanation=normalized_candidate.explanation,
                        confidence=normalized_candidate.confidence,
                        evidence=normalized_candidate.evidence,
                        metadata={"section_id": section.section_id},
                    )
                    edge_added, reason = self._try_add_edge(state=state, edge=edge)
                    if edge_added:
                        accepted_edge_count += 1
                    if not edge_added and reason:
                        warnings.append(reason)
                        if reason.startswith("Duplicate edge skipped"):
                            rejected_duplicate_count += 1
                        elif reason.startswith("Edge skipped due to DAG constraint"):
                            rejected_cycle_count += 1

            concepts = [concepts_by_id[concept_id] for concept_id in concept_order]
            status = "partial" if warnings else "ok"
            self._mark_section_complete(state=state, index=section_index)
            concepts_with_matches = sum(1 for item in concepts if item.historical_matches)
            logger.info(
                (
                    "phase_b_graph.section_metrics doc_id=%s section_id=%s concepts=%d "
                    "concepts_with_matches=%d edge_validations=%d accepted_edges=%d "
                    "rejected_by_model=%d rejected_by_confidence=%d rejected_duplicate=%d "
                    "rejected_cycle=%d"
                ),
                state.doc_id,
                section.section_id,
                len(concepts),
                concepts_with_matches,
                edge_validation_attempts,
                accepted_edge_count,
                rejected_by_model_count,
                rejected_by_confidence_count,
                rejected_duplicate_count,
                rejected_cycle_count,
            )
            state.parse_log.append(
                f"section={section.section_id} status={status} concepts={len(concepts)} edges={len(edge_candidates)}"
            )
        except Exception as exc:
            errors.append(str(exc))
            status = "failed"
            self._mark_section_failed(state=state, index=section_index, error=str(exc))
            state.parse_log.append(f"section={section.section_id} status=failed error={exc}")
            logger.exception(
                "phase_b_graph.section_exception doc_id=%s section_id=%s",
                state.doc_id,
                section.section_id,
            )

        return SectionParseResult(
            job_id=state.job_id,
            doc_id=state.doc_id,
            section_id=section.section_id,
            section_order=section_index,
            section_title=section.title,
            source_chunk_ids=resolved_chunk_ids,
            concepts=concepts,
            edge_candidates=edge_candidates,
            llm_calls=llm_calls,
            status=status,
            warnings=warnings,
            errors=errors,
            metadata={"section_path": section.path},
        )

    def _extract_concepts_for_section(
        self,
        *,
        state: RollingState,
        section: FlatTOCSection,
        section_text: str,
    ) -> SectionConceptExtractionOutput:
        context_json = self._build_state_context_json(state=state)
        return self.reasoning_client.extract_section_concepts(
            doc_id=state.doc_id,
            section_id=section.section_id,
            section_title=section.title,
            section_text=section_text,
            rolling_state_json=context_json,
        )

    def _validate_edge(
        self,
        *,
        doc_id: str,
        concept: SectionConcept,
        historical_node: ConceptNode,
        match: SimilarityMatch,
    ) -> EdgeValidationOutput:
        new_concept_json = json.dumps(concept.model_dump(mode="json"), ensure_ascii=True)
        historical_concept_json = json.dumps(
            historical_node.model_dump(mode="json"), ensure_ascii=True
        )
        supporting_evidence_json = json.dumps(
            {
                "similarity": match.similarity,
                "historical_chunk_id": match.historical_chunk_id,
                "historical_section_id": match.historical_section_id,
                "current_chunk_ids": concept.source_chunk_ids,
                "doc_id": doc_id,
            },
            ensure_ascii=True,
        )
        return self.reasoning_client.validate_edge_candidate(
            new_concept_json=new_concept_json,
            historical_concept_json=historical_concept_json,
            supporting_evidence_json=supporting_evidence_json,
        )

    def _validate_inputs(
        self,
        *,
        doc_id: str,
        toc: TOCData,
        chunking: ChunkingResult,
        embeddings: EmbeddingBatchResult,
    ) -> None:
        if toc.doc_id != doc_id:
            raise ValueError(f"doc_id mismatch: run={doc_id}, toc={toc.doc_id}")
        if chunking.doc_id != doc_id:
            raise ValueError(f"doc_id mismatch: run={doc_id}, chunking={chunking.doc_id}")
        if embeddings.doc_id != doc_id:
            raise ValueError(f"doc_id mismatch: run={doc_id}, embeddings={embeddings.doc_id}")

    def _build_state_context_json(self, *, state: RollingState) -> str:
        summary_nodes = state.nodes[-self.config.max_state_nodes_in_context :]
        payload = {
            "job_id": state.job_id,
            "doc_id": state.doc_id,
            "current_section_index": state.current_section_index,
            "known_nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "aliases": node.aliases,
                    "summary": node.summary,
                    "section_id": node.source_material.section_id,
                }
                for node in summary_nodes
            ],
            "known_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                }
                for edge in state.edges[-200:]
            ],
            "parse_log_tail": state.parse_log[-20:],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _resolve_section_chunk_ids(
        self,
        *,
        section: FlatTOCSection,
        section_index: int,
        total_sections: int,
        chunking: ChunkingResult,
        chunk_by_id: dict[str, Any],
    ) -> list[str]:
        explicit = [chunk_id for chunk_id in section.chunk_ids if chunk_id in chunk_by_id]
        if explicit:
            return _dedupe_preserve_order(explicit)

        if not chunking.chunks:
            return []
        start = int((section_index * len(chunking.chunks)) / total_sections)
        end = int(((section_index + 1) * len(chunking.chunks)) / total_sections)
        if end <= start:
            end = min(start + 1, len(chunking.chunks))
        return [chunk.chunk_id for chunk in chunking.chunks[start:end]]

    def _normalize_section_concept(
        self,
        *,
        concept: SectionConcept,
        fallback_chunk_ids: list[str],
        chunk_by_id: dict[str, Any],
    ) -> SectionConcept:
        valid_source_chunk_ids = [
            chunk_id for chunk_id in concept.source_chunk_ids if chunk_id in chunk_by_id
        ]
        if not valid_source_chunk_ids:
            valid_source_chunk_ids = fallback_chunk_ids[:]
        if not valid_source_chunk_ids and fallback_chunk_ids:
            valid_source_chunk_ids = [fallback_chunk_ids[0]]

        return SectionConcept(
            concept_id=concept.concept_id,
            label=concept.label,
            summary=concept.summary,
            aliases=concept.aliases,
            source_chunk_ids=valid_source_chunk_ids,
            evidence_text=concept.evidence_text,
            confidence=concept.confidence,
            historical_matches=[],
        )

    def _merge_section_concepts(
        self, *, existing: SectionConcept, incoming: SectionConcept
    ) -> SectionConcept:
        merged_aliases = _dedupe_preserve_order(existing.aliases + incoming.aliases)
        merged_chunk_ids = _dedupe_preserve_order(
            existing.source_chunk_ids + incoming.source_chunk_ids
        )
        merged_historical = list(existing.historical_matches)
        known_historical_ids = {match.historical_concept_id for match in merged_historical}
        for match in incoming.historical_matches:
            if match.historical_concept_id in known_historical_ids:
                continue
            merged_historical.append(match)
            known_historical_ids.add(match.historical_concept_id)
        return SectionConcept(
            concept_id=existing.concept_id,
            label=existing.label,
            summary=incoming.summary if len(incoming.summary) > len(existing.summary) else existing.summary,
            aliases=merged_aliases,
            source_chunk_ids=merged_chunk_ids,
            evidence_text=existing.evidence_text or incoming.evidence_text,
            confidence=max(existing.confidence, incoming.confidence),
            historical_matches=merged_historical,
        )

    def _resolve_or_create_node(
        self,
        *,
        state: RollingState,
        node_by_id: dict[str, ConceptNode],
        concept: SectionConcept,
        section: FlatTOCSection,
        chunk_by_id: dict[str, Any],
        allow_node_creation: bool,
    ) -> str:
        alias_keys = self._build_alias_keys(label=concept.label, aliases=concept.aliases)
        existing_id = next(
            (
                state.concept_alias_index[key]
                for key in alias_keys
                if key in state.concept_alias_index
            ),
            None,
        )
        if existing_id and existing_id in node_by_id:
            self._merge_node_with_concept(
                state=state,
                node_by_id=node_by_id,
                target_node_id=existing_id,
                concept=concept,
                alias_keys=alias_keys,
                chunk_by_id=chunk_by_id,
                section_id=section.section_id,
            )
            return existing_id

        if not allow_node_creation and node_by_id:
            fallback_node_id = self._select_fallback_node_id(
                state=state,
                node_by_id=node_by_id,
                section=section,
                concept=concept,
            )
            if fallback_node_id:
                self._merge_node_with_concept(
                    state=state,
                    node_by_id=node_by_id,
                    target_node_id=fallback_node_id,
                    concept=concept,
                    alias_keys=alias_keys,
                    chunk_by_id=chunk_by_id,
                    section_id=section.section_id,
                )
                return fallback_node_id

        node_id = self._generate_node_id(
            doc_id=state.doc_id,
            section_id=section.section_id,
            label=concept.label,
            existing_ids=set(node_by_id.keys()),
        )
        source_material = SourceMaterial(
            doc_id=state.doc_id,
            section_id=section.section_id,
            chunk_ids=concept.source_chunk_ids,
            page_numbers=[
                chunk_by_id[cid].source_page
                for cid in concept.source_chunk_ids
                if cid in chunk_by_id and chunk_by_id[cid].source_page is not None
            ],
            transcript_timestamps=self._chunk_timestamps(concept.source_chunk_ids, chunk_by_id),
            snippet=concept.evidence_text
            or self._build_snippet_from_chunks(chunk_ids=concept.source_chunk_ids, chunk_by_id=chunk_by_id),
        )
        node = ConceptNode(
            id=node_id,
            label=concept.label,
            summary=concept.summary,
            aliases=concept.aliases,
            confidence=concept.confidence,
            source_material=source_material,
            metadata={"created_from_section": section.section_id},
        )
        state.nodes.append(node)
        node_by_id[node.id] = node
        for alias_key in alias_keys:
            state.concept_alias_index[alias_key] = node.id
        return node.id

    def _merge_node_with_concept(
        self,
        *,
        state: RollingState,
        node_by_id: dict[str, ConceptNode],
        target_node_id: str,
        concept: SectionConcept,
        alias_keys: list[str],
        chunk_by_id: dict[str, Any],
        section_id: str,
    ) -> None:
        existing_node = node_by_id[target_node_id]
        merged_aliases = _dedupe_preserve_order(existing_node.aliases + concept.aliases)
        merged_chunk_ids = _dedupe_preserve_order(
            existing_node.source_material.chunk_ids + concept.source_chunk_ids
        )
        merged_pages = _dedupe_preserve_order(
            [str(page) for page in existing_node.source_material.page_numbers]
            + [
                str(chunk_by_id[cid].source_page)
                for cid in concept.source_chunk_ids
                if cid in chunk_by_id and chunk_by_id[cid].source_page is not None
            ]
        )
        merged_timestamps = _dedupe_preserve_order(
            existing_node.source_material.transcript_timestamps
            + self._chunk_timestamps(concept.source_chunk_ids, chunk_by_id)
        )
        use_incoming_summary = (
            len(concept.summary) > len(existing_node.summary)
            or bool(existing_node.metadata.get("toc_seed"))
        )
        updated_metadata = dict(existing_node.metadata)
        updated_metadata["last_observed_section"] = section_id
        if bool(updated_metadata.get("toc_seed")):
            updated_metadata["toc_seed"] = False
        updated_node = ConceptNode(
            id=existing_node.id,
            label=existing_node.label,
            summary=concept.summary if use_incoming_summary else existing_node.summary,
            aliases=merged_aliases,
            confidence=max(existing_node.confidence, concept.confidence),
            deep_dive=existing_node.deep_dive,
            source_material=SourceMaterial(
                doc_id=existing_node.source_material.doc_id,
                section_id=existing_node.source_material.section_id,
                chunk_ids=merged_chunk_ids,
                page_numbers=[int(page) for page in merged_pages],
                transcript_timestamps=merged_timestamps,
                snippet=existing_node.source_material.snippet or concept.evidence_text,
            ),
            metadata=updated_metadata,
        )
        self._replace_node_in_state(state=state, node_id=existing_node.id, node=updated_node)
        node_by_id[existing_node.id] = updated_node
        for alias_key in alias_keys:
            state.concept_alias_index[alias_key] = existing_node.id

    def _select_fallback_node_id(
        self,
        *,
        state: RollingState,
        node_by_id: dict[str, ConceptNode],
        section: FlatTOCSection,
        concept: SectionConcept,
    ) -> str | None:
        top_section_id = self._top_level_section_id(section.path)
        anchored_candidates = [
            node.id
            for node in state.nodes
            if node.metadata.get("seed_section_id") == top_section_id
        ]
        if len(anchored_candidates) == 1:
            return anchored_candidates[0]
        if len(anchored_candidates) > 1:
            concept_tokens = set(_safe_slug(concept.label, max_len=128).split("_"))
            best_id: str | None = None
            best_score = -1
            for node_id in anchored_candidates:
                node = node_by_id.get(node_id)
                if node is None:
                    continue
                node_tokens = set(_safe_slug(node.label, max_len=128).split("_"))
                score = len(concept_tokens & node_tokens)
                if score > best_score:
                    best_score = score
                    best_id = node_id
            if best_id:
                return best_id
            return anchored_candidates[0]
        if state.nodes:
            return state.nodes[0].id
        return None

    def _top_level_section_id(self, path: str) -> str:
        cleaned = path.strip()
        if not cleaned:
            return ""
        return cleaned.split("/", 1)[0]

    def _seed_core_nodes_from_toc(
        self,
        *,
        state: RollingState,
        flat_sections: list[FlatTOCSection],
        chunk_by_id: dict[str, Any],
    ) -> int:
        top_level_sections: list[FlatTOCSection] = [
            section for section in flat_sections if "/" not in section.path
        ]
        if not top_level_sections:
            top_level_sections = list(flat_sections)
        seed_limit = max(1, self.config.max_seed_core_nodes)
        seeded = 0
        existing_ids = {node.id for node in state.nodes}
        for section in top_level_sections[:seed_limit]:
            label = section.title.strip()
            if not label:
                continue
            node_id = self._generate_node_id(
                doc_id=state.doc_id,
                section_id=section.section_id,
                label=label,
                existing_ids=existing_ids,
            )
            existing_ids.add(node_id)
            valid_chunk_ids = [
                chunk_id for chunk_id in section.chunk_ids if chunk_id in chunk_by_id
            ]
            snippet = self._build_snippet_from_chunks(
                chunk_ids=valid_chunk_ids,
                chunk_by_id=chunk_by_id,
            )
            node = ConceptNode(
                id=node_id,
                label=label,
                summary=f"Core topic seeded from TOC section '{label}'.",
                aliases=_dedupe_preserve_order([label]),
                confidence=0.8,
                source_material=SourceMaterial(
                    doc_id=state.doc_id,
                    section_id=section.section_id,
                    chunk_ids=valid_chunk_ids,
                    page_numbers=[
                        chunk_by_id[cid].source_page
                        for cid in valid_chunk_ids
                        if chunk_by_id[cid].source_page is not None
                    ],
                    transcript_timestamps=self._chunk_timestamps(valid_chunk_ids, chunk_by_id),
                    snippet=snippet,
                ),
                metadata={
                    "core_concept": True,
                    "toc_seed": True,
                    "seed_section_id": section.section_id,
                },
            )
            state.nodes.append(node)
            alias_keys = self._build_alias_keys(label=node.label, aliases=node.aliases)
            for alias_key in alias_keys:
                state.concept_alias_index[alias_key] = node.id
            seeded += 1
        return seeded

    def _retrieve_historical_matches(
        self,
        *,
        concept: SectionConcept,
        historical_chunk_index: dict[str, list[str]],
        node_by_id: dict[str, ConceptNode],
        embedding_by_chunk: dict[str, ChunkEmbedding],
        embedding_model_name: str,
        historical_concept_vectors: dict[str, list[float]],
    ) -> list[SimilarityMatch]:
        logger = logging.getLogger(__name__)
        vectors = [
            embedding_by_chunk[chunk_id].vector
            for chunk_id in concept.source_chunk_ids
            if chunk_id in embedding_by_chunk
        ]
        if not vectors:
            logger.debug(
                "phase_b_graph.retrieval_skip concept_id=%s reason=no_query_vectors",
                concept.concept_id,
            )
            return []

        query_vector = _mean_vector(vectors)
        top_k = max(1, self.config.top_k_historical_matches)
        overfetch = max(top_k, top_k * max(1, self.config.retrieval_overfetch_multiplier))
        primary_threshold = max(0.0, min(1.0, self.config.similarity_threshold))
        fallback_threshold = max(
            0.0,
            min(primary_threshold, self.config.similarity_fallback_threshold),
        )

        raw_results = self.storage_client.similarity_search(
            query_vector=query_vector,
            top_k=overfetch,
            min_similarity=primary_threshold,
            model_name=embedding_model_name,
        )
        best_by_concept: dict[str, dict[str, Any]] = {}
        mapped_primary = self._ingest_similarity_results(
            concept=concept,
            raw_results=raw_results,
            historical_chunk_index=historical_chunk_index,
            destination=best_by_concept,
            origin="actian_primary",
        )

        mapped_relaxed = 0
        if len(best_by_concept) < top_k and fallback_threshold < primary_threshold:
            relaxed_results = self.storage_client.similarity_search(
                query_vector=query_vector,
                top_k=overfetch,
                min_similarity=fallback_threshold,
                model_name=embedding_model_name,
            )
            mapped_relaxed = self._ingest_similarity_results(
                concept=concept,
                raw_results=relaxed_results,
                historical_chunk_index=historical_chunk_index,
                destination=best_by_concept,
                origin="actian_relaxed",
            )

        local_added = 0
        if len(best_by_concept) < top_k:
            local_candidates = list(historical_concept_vectors.items())
            local_limit = max(1, self.config.max_historical_nodes_for_local_similarity)
            if len(local_candidates) > local_limit:
                local_candidates = local_candidates[-local_limit:]
            for historical_concept_id, historical_vector in local_candidates:
                if historical_concept_id in best_by_concept or historical_concept_id == concept.concept_id:
                    continue
                similarity = _cosine_similarity(query_vector, historical_vector)
                if similarity < fallback_threshold:
                    continue
                historical_node = node_by_id.get(historical_concept_id)
                if historical_node is None:
                    continue
                fallback_chunk_id = next(
                    (
                        chunk_id
                        for chunk_id in historical_node.source_material.chunk_ids
                        if chunk_id in embedding_by_chunk
                    ),
                    None,
                )
                if fallback_chunk_id is None and historical_node.source_material.chunk_ids:
                    fallback_chunk_id = historical_node.source_material.chunk_ids[0]
                best_by_concept[historical_concept_id] = {
                    "similarity": similarity,
                    "chunk_id": fallback_chunk_id,
                    "origin": "local_concept_fallback",
                }
                local_added += 1

        ranked = sorted(
            best_by_concept.items(),
            key=lambda item: item[1]["similarity"],
            reverse=True,
        )[:top_k]
        matches: list[SimilarityMatch] = []
        for historical_concept_id, payload in ranked:
            historical_node = node_by_id.get(historical_concept_id)
            matches.append(
                SimilarityMatch(
                    historical_concept_id=historical_concept_id,
                    similarity=float(payload["similarity"]),
                    historical_doc_id=(
                        historical_node.source_material.doc_id if historical_node else None
                    ),
                    historical_section_id=(
                        historical_node.source_material.section_id if historical_node else None
                    ),
                    historical_chunk_id=payload["chunk_id"],
                )
            )
        logger.info(
            (
                "phase_b_graph.retrieval concept_id=%s source_chunks=%d "
                "actian_results=%d mapped_primary=%d mapped_relaxed=%d mapped_local=%d returned=%d "
                "threshold_primary=%.3f threshold_fallback=%.3f"
            ),
            concept.concept_id,
            len(concept.source_chunk_ids),
            len(raw_results),
            mapped_primary,
            mapped_relaxed,
            local_added,
            len(matches),
            primary_threshold,
            fallback_threshold,
        )
        return matches

    def _ingest_similarity_results(
        self,
        *,
        concept: SectionConcept,
        raw_results: list[dict[str, Any]],
        historical_chunk_index: dict[str, list[str]],
        destination: dict[str, dict[str, Any]],
        origin: str,
    ) -> int:
        mapped = 0
        for result in raw_results:
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            linked_concepts = historical_chunk_index.get(chunk_id, [])
            if not linked_concepts:
                continue
            similarity = float(result.get("similarity", 0.0))
            for concept_id in linked_concepts:
                if concept_id == concept.concept_id:
                    continue
                current = destination.get(concept_id)
                if current is None:
                    mapped += 1
                if current is None or similarity > current["similarity"]:
                    destination[concept_id] = {
                        "similarity": similarity,
                        "chunk_id": chunk_id,
                        "origin": origin,
                    }
        return mapped

    def _normalize_edge_candidate(
        self,
        *,
        candidate: SectionEdgeCandidate,
        expected_source: str,
        expected_target: str,
        doc_id: str,
        historical_chunk_id: str | None,
        current_chunk_ids: list[str],
    ) -> tuple[SectionEdgeCandidate, str | None]:
        warning: str | None = None
        if candidate.source_concept_id != expected_source or candidate.target_concept_id != expected_target:
            warning = (
                "Edge candidate concept IDs did not match expected direction; "
                "normalized to historical->current."
            )
        evidence_historical_chunks = (
            [historical_chunk_id] if historical_chunk_id else candidate.evidence.historical_chunk_ids
        )
        normalized = SectionEdgeCandidate(
            source_concept_id=expected_source,
            target_concept_id=expected_target,
            relation="prerequisite_for",
            accepted=candidate.accepted and expected_source != expected_target,
            confidence=candidate.confidence,
            explanation=candidate.explanation,
            evidence={
                "historical_doc_id": candidate.evidence.historical_doc_id or doc_id,
                "current_doc_id": candidate.evidence.current_doc_id or doc_id,
                "historical_chunk_ids": evidence_historical_chunks,
                "current_chunk_ids": current_chunk_ids,
            },
        )
        return normalized, warning

    def _try_add_edge(self, *, state: RollingState, edge: ConceptEdge) -> tuple[bool, str | None]:
        existing_pairs = {(item.source, item.target) for item in state.edges}
        if (edge.source, edge.target) in existing_pairs:
            return False, f"Duplicate edge skipped: {edge.source} -> {edge.target}"

        tentative_edges = state.edges + [edge]
        try:
            GraphData(nodes=state.nodes, edges=tentative_edges)
        except ValueError as exc:
            return False, f"Edge skipped due to DAG constraint: {exc}"
        state.edges.append(edge)
        return True, None

    def _build_chunk_to_concept_index(
        self,
        *,
        nodes: list[ConceptNode],
        allowed_concept_ids: set[str],
    ) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for node in nodes:
            if node.id not in allowed_concept_ids:
                continue
            for chunk_id in node.source_material.chunk_ids:
                index.setdefault(chunk_id, []).append(node.id)
        return index

    def _build_historical_concept_vectors(
        self,
        *,
        historical_concept_ids: list[str],
        node_by_id: dict[str, ConceptNode],
        embedding_by_chunk: dict[str, ChunkEmbedding],
    ) -> dict[str, list[float]]:
        concept_vectors: dict[str, list[float]] = {}
        for concept_id in historical_concept_ids:
            node = node_by_id.get(concept_id)
            if node is None:
                continue
            source_vectors = [
                embedding_by_chunk[chunk_id].vector
                for chunk_id in node.source_material.chunk_ids
                if chunk_id in embedding_by_chunk
            ]
            if not source_vectors:
                continue
            try:
                concept_vectors[concept_id] = _mean_vector(source_vectors)
            except ValueError:
                continue
        return concept_vectors

    def _replace_node_in_state(self, *, state: RollingState, node_id: str, node: ConceptNode) -> None:
        for index, existing in enumerate(state.nodes):
            if existing.id == node_id:
                state.nodes[index] = node
                return

    def _build_alias_keys(self, *, label: str, aliases: list[str]) -> list[str]:
        values = [label, *aliases]
        keys = [_safe_slug(value, max_len=96) for value in values if value.strip()]
        return _dedupe_preserve_order(keys)

    def _generate_node_id(
        self,
        *,
        doc_id: str,
        section_id: str,
        label: str,
        existing_ids: set[str],
    ) -> str:
        base = f"{_safe_slug(doc_id, max_len=24)}:{_safe_slug(section_id, max_len=24)}:{_safe_slug(label)}"
        candidate = base
        suffix = 2
        while candidate in existing_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _chunk_timestamps(self, chunk_ids: list[str], chunk_by_id: dict[str, Any]) -> list[str]:
        timestamps: list[str] = []
        for chunk_id in chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            if (
                chunk.source_time_start_seconds is None
                or chunk.source_time_end_seconds is None
            ):
                continue
            timestamps.append(
                f"{chunk.source_time_start_seconds:.2f}-{chunk.source_time_end_seconds:.2f}"
            )
        return timestamps

    def _build_snippet_from_chunks(
        self,
        *,
        chunk_ids: list[str],
        chunk_by_id: dict[str, Any],
        max_chars: int = 300,
    ) -> str | None:
        parts: list[str] = []
        for chunk_id in chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            text = chunk.text.strip()
            if text:
                parts.append(text)
        if not parts:
            return None
        snippet = " ".join(parts)
        return snippet[:max_chars]

    def _mark_section_in_progress(self, *, state: RollingState, index: int) -> None:
        progress = state.sections[index]
        progress.status = ParseStatus.in_progress
        progress.started_at = _utc_now()
        progress.error = None
        state.updated_at = _utc_now()

    def _mark_section_complete(self, *, state: RollingState, index: int) -> None:
        progress = state.sections[index]
        progress.status = ParseStatus.completed
        progress.completed_at = _utc_now()
        progress.error = None
        state.updated_at = _utc_now()

    def _mark_section_failed(self, *, state: RollingState, index: int, error: str) -> None:
        progress = state.sections[index]
        progress.status = ParseStatus.failed
        progress.completed_at = _utc_now()
        progress.error = error
        state.updated_at = _utc_now()


__all__ = [
    "ActianSimilaritySearchClient",
    "FlatTOCSection",
    "PhaseBGraphConfig",
    "PhaseBGraphOutput",
    "PhaseBGraphPipeline",
    "build_section_text",
    "flatten_toc_sections",
]
