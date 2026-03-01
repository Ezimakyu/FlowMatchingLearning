import pytest
from types import SimpleNamespace

from backend.app.reasoning.toc_reasoning import (
    OpenAITOCReasoningClient,
    OpenAITOCReasoningConfig,
    build_responses_json_schema_text_config,
    build_toc_json_schema,
    extract_first_json_object,
    normalize_toc_payload,
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


class StubOpenAITOCReasoningClientMissingDocId(OpenAITOCReasoningClient):
    def _build_client(self):
        return object()

    def _call_model(self, *, system_prompt: str, user_prompt: str):
        _ = system_prompt
        _ = user_prompt
        return (
            '{"sections":[{"section_id":"intro","title":"Intro","order":0,'
            '"chunk_ids":[],"key_terms":[],"children":[]}],"metadata":{}}',
            "req_test_002",
        )


class FakeResponsesEndpoint:
    def __init__(self, *, fail_on_text: bool = False) -> None:
        self.fail_on_text = fail_on_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_text and "text" in kwargs:
            raise TypeError("unexpected keyword argument: text")
        return SimpleNamespace(
            id="resp_schema_test",
            output_text='{"doc_id":"doc_schema","sections":[],"metadata":{}}',
        )


class StubOpenAITOCReasoningClientWithResponses(OpenAITOCReasoningClient):
    def __init__(self, *, fail_on_text: bool = False, config: OpenAITOCReasoningConfig):
        self._fail_on_text = fail_on_text
        super().__init__(config=config)

    def _build_client(self):
        self.responses_endpoint = FakeResponsesEndpoint(fail_on_text=self._fail_on_text)
        return SimpleNamespace(responses=self.responses_endpoint)


def test_extract_first_json_object_with_wrapped_response() -> None:
    wrapped = "Some preface\n```json\n{\"a\": 1, \"b\": 2}\n```\n"
    parsed = extract_first_json_object(wrapped)
    assert parsed == {"a": 1, "b": 2}


def test_extract_first_json_object_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        extract_first_json_object("this is not json")


def test_normalize_toc_payload_injects_missing_doc_id() -> None:
    normalized = normalize_toc_payload({"sections": []}, doc_id="doc_fix")
    assert normalized["doc_id"] == "doc_fix"
    assert normalized["schema_version"] == "1.0.0"


def test_build_toc_json_schema_requires_doc_id() -> None:
    schema = build_toc_json_schema()
    assert "doc_id" in schema["required"]


def test_responses_json_schema_text_config_is_strict() -> None:
    config = build_responses_json_schema_text_config()
    assert config["format"]["type"] == "json_schema"
    assert config["format"]["strict"] is True
    assert "doc_id" in config["format"]["schema"]["required"]


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


def test_openai_toc_reasoning_client_uses_custom_model() -> None:
    client = StubOpenAITOCReasoningClient(
        config=OpenAITOCReasoningConfig(
            model="gpt-5.1",
            prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )
    result = client.generate_toc(
        doc_id="doc_demo",
        document_text="Sequence and series.",
    )
    assert result.llm_call.model == "gpt-5.1"


def test_openai_toc_reasoning_client_fallback_doc_id_when_missing() -> None:
    client = StubOpenAITOCReasoningClientMissingDocId(
        config=OpenAITOCReasoningConfig(
            model="gpt-4.1-mini",
            prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )
    result = client.generate_toc(
        doc_id="doc_from_call",
        document_text="Limits then derivatives.",
    )
    assert result.toc.doc_id == "doc_from_call"
    assert result.llm_call.request_id == "req_test_002"


def test_openai_toc_reasoning_client_uses_responses_json_schema_when_available() -> None:
    client = StubOpenAITOCReasoningClientWithResponses(
        fail_on_text=False,
        config=OpenAITOCReasoningConfig(
            model="gpt-4.1-mini",
            prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )

    result = client.generate_toc(
        doc_id="doc_schema",
        document_text="Foundations then operators.",
    )
    assert result.toc.doc_id == "doc_schema"
    assert len(client.responses_endpoint.calls) == 1
    assert "text" in client.responses_endpoint.calls[0]
    assert client.responses_endpoint.calls[0]["text"]["format"]["type"] == "json_schema"


def test_openai_toc_reasoning_client_falls_back_when_responses_text_unsupported() -> None:
    client = StubOpenAITOCReasoningClientWithResponses(
        fail_on_text=True,
        config=OpenAITOCReasoningConfig(
            model="gpt-4.1-mini",
            prompt_version="2026-02-28.v2",
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )

    result = client.generate_toc(
        doc_id="doc_schema",
        document_text="Foundations then operators.",
    )
    assert result.toc.doc_id == "doc_schema"
    assert len(client.responses_endpoint.calls) == 2
    assert "text" in client.responses_endpoint.calls[0]
    assert "text" not in client.responses_endpoint.calls[1]
