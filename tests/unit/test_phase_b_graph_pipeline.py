from backend.app.models import (
    ChunkEmbedding,
    ChunkingResult,
    EmbeddingBatchResult,
    LLMCallMetadata,
    RawTextChunk,
    SectionConcept,
    SectionEdgeCandidate,
    TOCData,
    TOCSection,
)
from backend.app.pipelines.phase_b_graph import PhaseBGraphConfig, PhaseBGraphPipeline
from backend.app.reasoning.section_reasoning import (
    EdgeValidationOutput,
    SectionConceptExtractionOutput,
)


class FakeSectionReasoningClient:
    def extract_section_concepts(
        self,
        *,
        doc_id: str,
        section_id: str,
        section_title: str,
        section_text: str,
        rolling_state_json: str,
    ) -> SectionConceptExtractionOutput:
        _ = section_title
        _ = section_text
        _ = rolling_state_json
        if section_id == "limits_section":
            concepts = [
                SectionConcept(
                    concept_id="limits",
                    label="Limits",
                    summary="Foundational behavior as x approaches a point.",
                    aliases=["limit"],
                    source_chunk_ids=[f"{doc_id}:vision_text:00000"],
                    confidence=0.95,
                )
            ]
        else:
            concepts = [
                SectionConcept(
                    concept_id="derivatives",
                    label="Derivatives",
                    summary="Rate of change defined from limits.",
                    aliases=["differentiation"],
                    source_chunk_ids=[f"{doc_id}:vision_text:00001"],
                    confidence=0.92,
                )
            ]
        return SectionConceptExtractionOutput(
            concepts=concepts,
            warnings=[],
            llm_call=LLMCallMetadata(
                provider="openai",
                prompt_name="section_concept_extraction",
                prompt_version="2026-02-28.v2",
                model="gpt-4.1-mini",
                request_id=f"req_extract_{section_id}",
            ),
            prompt_tag="section_concept_extraction:2026-02-28.v2",
            prompt_checksum="checksum_extract",
            raw_response_text='{"concepts":[]}',
        )

    def validate_edge_candidate(
        self,
        *,
        new_concept_json: str,
        historical_concept_json: str,
        supporting_evidence_json: str,
    ) -> EdgeValidationOutput:
        _ = new_concept_json
        _ = historical_concept_json
        _ = supporting_evidence_json
        return EdgeValidationOutput(
            candidate=SectionEdgeCandidate(
                source_concept_id="placeholder_historical",
                target_concept_id="placeholder_current",
                accepted=True,
                confidence=0.86,
                explanation="Derivatives depend on the definition of limits.",
                evidence={
                    "historical_doc_id": "doc_calc",
                    "current_doc_id": "doc_calc",
                    "historical_chunk_ids": ["doc_calc:vision_text:00000"],
                    "current_chunk_ids": ["doc_calc:vision_text:00001"],
                },
            ),
            llm_call=LLMCallMetadata(
                provider="openai",
                prompt_name="edge_validation",
                prompt_version="2026-02-28.v2",
                model="gpt-4.1-mini",
                request_id="req_edge",
            ),
            prompt_tag="edge_validation:2026-02-28.v2",
            prompt_checksum="checksum_edge",
            raw_response_text='{"accepted":true}',
        )


class FakeSpecificConceptReasoningClient(FakeSectionReasoningClient):
    def extract_section_concepts(
        self,
        *,
        doc_id: str,
        section_id: str,
        section_title: str,
        section_text: str,
        rolling_state_json: str,
    ) -> SectionConceptExtractionOutput:
        _ = section_title
        _ = section_text
        _ = rolling_state_json
        if section_id == "limits_section":
            concepts = [
                SectionConcept(
                    concept_id="epsilon_delta",
                    label="Epsilon Delta Formalism",
                    summary="Formal limit definitions and notation details.",
                    aliases=["eps-delta argument"],
                    source_chunk_ids=[f"{doc_id}:vision_text:00000"],
                    confidence=0.74,
                )
            ]
        else:
            concepts = [
                SectionConcept(
                    concept_id="notation_tricks",
                    label="Derivative Notation Tricks",
                    summary="Surface notation choices for derivative expressions.",
                    aliases=["symbol shortcuts"],
                    source_chunk_ids=[f"{doc_id}:vision_text:00001"],
                    confidence=0.7,
                )
            ]
        return SectionConceptExtractionOutput(
            concepts=concepts,
            warnings=[],
            llm_call=LLMCallMetadata(
                provider="openai",
                prompt_name="section_concept_extraction",
                prompt_version="2026-02-28.v2",
                model="gpt-4.1-mini",
                request_id=f"req_extract_specific_{section_id}",
            ),
            prompt_tag="section_concept_extraction:2026-02-28.v2",
            prompt_checksum="checksum_extract_specific",
            raw_response_text='{"concepts":[]}',
        )


class FakeActianSimilarityClient:
    def similarity_search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        model_name: str = "BAAI/bge-m3",
        candidate_limit: int = 0,
    ) -> list[dict]:
        _ = query_vector
        _ = top_k
        _ = min_similarity
        _ = model_name
        _ = candidate_limit
        return [
            {
                "chunk_id": "doc_calc:vision_text:00000",
                "doc_id": "doc_calc",
                "similarity": 0.91,
                "raw_text": "Limits precede derivatives.",
            }
        ]


class FakeNoActianSimilarityClient:
    def similarity_search(
        self,
        *,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        model_name: str = "BAAI/bge-m3",
        candidate_limit: int = 0,
    ) -> list[dict]:
        _ = query_vector
        _ = top_k
        _ = min_similarity
        _ = model_name
        _ = candidate_limit
        return []


def _sample_chunking() -> ChunkingResult:
    return ChunkingResult(
        doc_id="doc_calc",
        chunks=[
            RawTextChunk(
                chunk_id="doc_calc:vision_text:00000",
                doc_id="doc_calc",
                source_type="vision_text",
                order=0,
                text="Limits are introduced first.",
                token_estimate=6,
                source_page=1,
            ),
            RawTextChunk(
                chunk_id="doc_calc:vision_text:00001",
                doc_id="doc_calc",
                source_type="vision_text",
                order=1,
                text="Derivatives are defined via limits.",
                token_estimate=7,
                source_page=2,
            ),
        ],
    )


def _sample_embeddings() -> list[ChunkEmbedding]:
    return [
        ChunkEmbedding(
            chunk_id="doc_calc:vision_text:00000",
            vector=[1.0, 0.0, 0.0],
            vector_dim=3,
            model_name="BAAI/bge-m3",
        ),
        ChunkEmbedding(
            chunk_id="doc_calc:vision_text:00001",
            vector=[0.8, 0.2, 0.0],
            vector_dim=3,
            model_name="BAAI/bge-m3",
        ),
    ]


def _sample_toc() -> TOCData:
    return TOCData(
        doc_id="doc_calc",
        sections=[
            TOCSection(
                section_id="limits_section",
                title="Limits",
                order=0,
                chunk_ids=["doc_calc:vision_text:00000"],
                key_terms=[],
                children=[],
            ),
            TOCSection(
                section_id="derivatives_section",
                title="Derivatives",
                order=1,
                chunk_ids=["doc_calc:vision_text:00001"],
                key_terms=[],
                children=[],
            ),
        ],
    )


def test_phase_b_graph_pipeline_generates_nodes_and_edges() -> None:
    pipeline = PhaseBGraphPipeline(
        reasoning_client=FakeSectionReasoningClient(),
        storage_client=FakeActianSimilarityClient(),
    )
    embeddings = EmbeddingBatchResult(
        doc_id="doc_calc",
        model_name="BAAI/bge-m3",
        embeddings=_sample_embeddings(),
    )
    result = pipeline.run(
        doc_id="doc_calc",
        toc=_sample_toc(),
        chunking=_sample_chunking(),
        embeddings=embeddings,
        job_id="job_graph_test",
    )
    assert result.graph.graph_id == "graph_doc_calc"
    assert len(result.graph.nodes) == 2
    assert len(result.graph.edges) == 1
    assert result.graph.edges[0].source != result.graph.edges[0].target
    assert result.rolling_state.current_section_index == 2
    assert all(section.status == "completed" for section in result.rolling_state.sections)
    assert len(result.section_results) == 2


def test_phase_b_graph_pipeline_uses_local_similarity_fallback() -> None:
    pipeline = PhaseBGraphPipeline(
        reasoning_client=FakeSectionReasoningClient(),
        storage_client=FakeNoActianSimilarityClient(),
        config=PhaseBGraphConfig(
            top_k_historical_matches=4,
            similarity_threshold=0.9,
            similarity_fallback_threshold=0.5,
            edge_acceptance_confidence_threshold=0.6,
        ),
    )
    embeddings = EmbeddingBatchResult(
        doc_id="doc_calc",
        model_name="BAAI/bge-m3",
        embeddings=_sample_embeddings(),
    )
    result = pipeline.run(
        doc_id="doc_calc",
        toc=_sample_toc(),
        chunking=_sample_chunking(),
        embeddings=embeddings,
        job_id="job_graph_test_fallback",
    )
    assert len(result.graph.edges) == 1
    second_section = result.section_results[1]
    assert len(second_section.concepts) == 1
    assert len(second_section.concepts[0].historical_matches) >= 1
    assert (
        second_section.concepts[0].historical_matches[0].historical_concept_id
        == result.graph.edges[0].source
    )


def test_phase_b_graph_pipeline_freezes_node_set_after_toc_seed() -> None:
    pipeline = PhaseBGraphPipeline(
        reasoning_client=FakeSpecificConceptReasoningClient(),
        storage_client=FakeNoActianSimilarityClient(),
    )
    embeddings = EmbeddingBatchResult(
        doc_id="doc_calc",
        model_name="BAAI/bge-m3",
        embeddings=_sample_embeddings(),
    )
    result = pipeline.run(
        doc_id="doc_calc",
        toc=_sample_toc(),
        chunking=_sample_chunking(),
        embeddings=embeddings,
        job_id="job_graph_test_freeze",
    )
    assert len(result.graph.nodes) == 2
    seeded_ids = {node.id for node in result.graph.nodes}
    assert seeded_ids
    for section_result in result.section_results:
        for concept in section_result.concepts:
            assert concept.concept_id in seeded_ids
