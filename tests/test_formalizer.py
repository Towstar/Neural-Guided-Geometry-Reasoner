from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

from geometry_reasoner.formalizer import (
    FormalizerConfig,
    FormalizationAPIError,
    FormalizationRefusalError,
    IncompleteFormalizationError,
    MissingAPIKeyError,
    MissingModelError,
    OllamaFormalizer,
    OpenAIFormalizer,
    create_formalizer,
    openai_api_key_from_environment,
    materialize_problem,
)
from geometry_reasoner.schema import FormalizationResult


def _parsed_result() -> FormalizationResult:
    return FormalizationResult.model_validate(
        {
            "status": "ok",
            "problem": {
                "statement": "D is the midpoint of AB. Prove A, D, and B are collinear.",
                "givens": [{"pred": "midpoint", "args": ["D", "A", "B"]}],
                "goal": {"pred": "collinear", "args": ["A", "D", "B"]},
            },
            "unsupported_reason": None,
        }
    )


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict | None = None

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class RaisingResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def parse(self, **kwargs: object) -> object:
        raise self.error


class RaisingClient:
    def __init__(self, error: Exception) -> None:
        self.responses = RaisingResponses(error)


class FakeClient:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


def _response(parsed: object | None = None, status: str = "completed") -> object:
    return SimpleNamespace(
        id="resp_test",
        model="gpt-5.6-luna",
        status=status,
        output_parsed=parsed,
        output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )


class FakeOllamaClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict | None = None

    def chat(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


def _ollama_response(content: str) -> object:
    return SimpleNamespace(
        model="geometry-local",
        done=True,
        message=SimpleNamespace(content=content),
        prompt_eval_count=10,
        eval_count=20,
    )


def test_text_request_uses_structured_output_without_storage() -> None:
    client = FakeClient(_response(_parsed_result()))
    formalizer = OpenAIFormalizer(client=client)

    result = formalizer.formalize_text("D is the midpoint of AB.")

    assert result.status == "ok"
    assert client.responses.kwargs is not None
    assert client.responses.kwargs["text_format"] is FormalizationResult
    assert client.responses.kwargs["store"] is False
    assert client.responses.kwargs["model"] == "gpt-5.6-luna"
    assert formalizer.last_call_metadata is not None
    assert formalizer.last_call_metadata.total_tokens == 30
    assert formalizer.last_call_metadata.provider == "openai"


def test_ollama_text_request_uses_schema_and_validates_json() -> None:
    expected = _parsed_result()
    client = FakeOllamaClient(_ollama_response(expected.model_dump_json()))
    formalizer = OllamaFormalizer(client=client, model="geometry-local")

    result = formalizer.formalize_text("D is the midpoint of AB.")

    assert result == expected
    assert client.kwargs is not None
    assert client.kwargs["model"] == "geometry-local"
    assert client.kwargs["format"] == FormalizationResult.model_json_schema()
    assert client.kwargs["options"] == {"temperature": 0}
    assert client.kwargs["stream"] is False
    assert formalizer.last_call_metadata is not None
    assert formalizer.last_call_metadata.provider == "ollama"
    assert formalizer.last_call_metadata.total_tokens == 30


def test_factory_selects_ollama_and_requires_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geometry_reasoner.formalizer.load_dotenv", lambda: None)
    selected = create_formalizer(FormalizerConfig(provider="ollama", model="local"))
    assert isinstance(selected, OllamaFormalizer)
    assert selected.model == "local"

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(MissingModelError, match="Ollama model is missing"):
        OllamaFormalizer()


def test_image_request_uses_base64_and_original_detail() -> None:
    image_path = Path(__file__).resolve().parents[1] / "data/images/geom_0002.png"
    client = FakeClient(_response(_parsed_result()))
    formalizer = OpenAIFormalizer(client=client)

    result = formalizer.formalize_image(image_path)

    assert result.status == "ok"
    assert client.responses.kwargs is not None
    content = client.responses.kwargs["input"][1]["content"]
    image_content = content[1]
    assert image_content["detail"] == "original"
    assert image_content["image_url"].startswith("data:image/png;base64,")


def test_incomplete_response_is_reported_safely() -> None:
    formalizer = OpenAIFormalizer(client=FakeClient(_response(status="incomplete")))
    with pytest.raises(IncompleteFormalizationError, match="not completed"):
        formalizer.formalize_text("A geometry question")


def test_completed_response_without_parsed_output_is_reported() -> None:
    formalizer = OpenAIFormalizer(client=FakeClient(_response()))
    with pytest.raises(IncompleteFormalizationError, match="parsed formalization"):
        formalizer.formalize_text("A geometry question")


def test_refusal_is_reported_without_returning_model_text() -> None:
    refusal = SimpleNamespace(content=[SimpleNamespace(refusal="private refusal")])
    response = _response()
    response.output = [refusal]
    formalizer = OpenAIFormalizer(client=FakeClient(response))

    with pytest.raises(FormalizationRefusalError, match="model refused") as error:
        formalizer.formalize_text("A geometry question")

    assert "private refusal" not in str(error.value)


def test_timeout_is_mapped_to_safe_typed_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    formalizer = OpenAIFormalizer(
        client=RaisingClient(APITimeoutError(request=request))
    )

    with pytest.raises(FormalizationAPIError, match="timed out"):
        formalizer.formalize_text("A geometry question")


def test_authentication_error_does_not_expose_provider_message() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(401, request=request)
    provider_error = AuthenticationError(
        "provider details must stay private",
        response=response,
        body=None,
    )
    formalizer = OpenAIFormalizer(client=RaisingClient(provider_error))

    with pytest.raises(FormalizationAPIError, match="authentication failed") as error:
        formalizer.formalize_text("A geometry question")

    assert "provider details" not in str(error.value)


def test_malformed_parsed_result_is_rejected() -> None:
    formalizer = OpenAIFormalizer(client=FakeClient(_response({"status": "ok"})))

    with pytest.raises(FormalizationAPIError, match="failed geometry validation"):
        formalizer.formalize_text("A geometry question")


def test_materialize_problem_generates_id_entities_and_canonical_facts() -> None:
    problem = materialize_problem(_parsed_result(), b"stable input")
    assert problem["id"].startswith("input_")
    assert problem["entities"]["points"] == ["A", "B", "D"]
    assert problem["goal"] == {"pred": "collinear", "args": ["A", "B", "D"]}


def test_api_key_prefers_standard_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geometry_reasoner.formalizer.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")
    monkeypatch.setenv("OPEN_AI_KEY", "legacy-key")
    assert openai_api_key_from_environment() == "standard-key"


def test_missing_api_key_has_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geometry_reasoner.formalizer.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        openai_api_key_from_environment()
