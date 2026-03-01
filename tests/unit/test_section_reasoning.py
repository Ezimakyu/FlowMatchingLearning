from backend.app.reasoning.section_reasoning import (
    OpenAISectionReasoningClient,
    OpenAISectionReasoningConfig,
)


class StubOpenAISectionReasoningClient(OpenAISectionReasoningClient):
    def _build_client(self):
        return object()

    def _call_model_with_schema(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict,
    ):
        _ = system_prompt
        _ = user_prompt
        _ = schema
        if schema_name == "section_concept_extraction_output":
            return (
                '{"concepts":[{"concept_id":"limits","label":"Limits","summary":"Limit behavior",'
                '"aliases":["limit"],"source_chunk_ids":["doc:vision_text:00000"],'
                '"evidence_text":"limit definition","confidence":0.9}],"warnings":[]}',
                "req_section_1",
            )
        return (
            '{"source_concept_id":"historical_limits","target_concept_id":"derivatives",'
            '"relation":"prerequisite_for","accepted":true,"confidence":0.88,'
            '"explanation":"Derivative definition builds on limits.",'
            '"evidence":{"historical_doc_id":"doc","current_doc_id":"doc",'
            '"historical_chunk_ids":["doc:vision_text:00000"],'
            '"current_chunk_ids":["doc:vision_text:00010"]}}',
            "req_edge_1",
        )


def test_extract_section_concepts_parses_result() -> None:
    client = StubOpenAISectionReasoningClient(
        config=OpenAISectionReasoningConfig(
            model="gpt-4.1-mini",
            section_prompt_version="2026-02-28.v2",
            edge_prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1200,
        )
    )
    output = client.extract_section_concepts(
        doc_id="doc",
        section_id="sec_intro",
        section_title="Intro",
        section_text="Limits section text",
        rolling_state_json='{"nodes":[],"edges":[]}',
    )
    assert len(output.concepts) == 1
    assert output.concepts[0].concept_id == "limits"
    assert output.llm_call.request_id == "req_section_1"


def test_validate_edge_candidate_parses_result() -> None:
    client = StubOpenAISectionReasoningClient(
        config=OpenAISectionReasoningConfig(
            model="gpt-4.1-mini",
            section_prompt_version="2026-02-28.v2",
            edge_prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1200,
        )
    )
    output = client.validate_edge_candidate(
        new_concept_json='{"concept_id":"derivatives"}',
        historical_concept_json='{"id":"historical_limits"}',
        supporting_evidence_json='{"similarity":0.9}',
    )
    assert output.candidate.source_concept_id == "historical_limits"
    assert output.candidate.target_concept_id == "derivatives"
    assert output.candidate.accepted is True
    assert output.llm_call.request_id == "req_edge_1"
