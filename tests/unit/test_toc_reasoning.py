import pytest

from backend.app.reasoning.toc_reasoning import (
    OpenAITOCReasoningClient,
    OpenAITOCReasoningConfig,
    extract_first_json_object,
)


class StubOpenAITOCReasoningClient(OpenAITOCReasoningClient):
    def _build_client(self):
        return object()

    def _call_model(self, *, system_prompt: str, user_prompt: str):
        _ = system_prompt
        _ = user_prompt
        return (
            '{"doc_id":"doc_abc","sections":[{"section_id":"intro","title":"Intro","order":0,'
            '"chunk_ids":[],"key_terms":[],"children":[]}],"metadata":{}}',
            "req_test_001",
        )


def test_extract_first_json_object_with_wrapped_response() -> None:
    wrapped = "Some preface\n```json\n{\"a\": 1, \"b\": 2}\n```\n"
    parsed = extract_first_json_object(wrapped)
    assert parsed == {"a": 1, "b": 2}


def test_extract_first_json_object_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        extract_first_json_object("this is not json")


def test_openai_toc_reasoning_client_parses_toc() -> None:
    client = StubOpenAITOCReasoningClient(
        config=OpenAITOCReasoningConfig(
            model="gpt-4.1-mini",
            prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )
    result = client.generate_toc(
        doc_id="doc_abc",
        document_text="Limits then derivatives.",
    )
    assert result.toc.doc_id == "doc_abc"
    assert len(result.toc.sections) == 1
    assert result.toc.sections[0].section_id == "intro"
    assert result.llm_call.provider == "openai"
    assert result.llm_call.request_id == "req_test_001"
