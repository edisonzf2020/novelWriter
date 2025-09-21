"""Unit tests for the OpenAI SDK-backed provider implementation."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Optional

import pytest

from novelwriter.ai.providers import ProviderSettings
from novelwriter.ai.providers.openai_sdk import OpenAISDKProvider

import novelwriter.ai.providers.openai_sdk as openai_sdk


class StubAPIStatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class StubStream:
    def __init__(self, events: Iterable[Any]) -> None:
        self._events = list(events)
        self.closed = False

    def __enter__(self) -> Iterable[Any]:
        return iter(self._events)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterable[Any]:
        return iter(self._events)

    def close(self) -> None:
        self.closed = True


class StubResponses:
    def __init__(
        self,
        *,
        create_result: Any,
        stream_events: Iterable[Any] | None = None,
        create_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._create_result = create_result
        self._stream_events = list(stream_events or [])
        self._create_error = create_error
        self._stream_error = stream_error
        self.with_raw_response = StubWithRawResponse(self)

    def create(self, **_: Any) -> Any:
        if self._create_error is not None:
            raise self._create_error
        return self._create_result

    def stream(self, **_: Any) -> StubStream:
        if self._stream_error is not None:
            raise self._stream_error
        return StubStream(self._stream_events)


class StubRawResponse:
    def __init__(self, result: Any, headers: dict[str, str] | None = None) -> None:
        self._result = result
        self.headers = headers or {}

    def parse(self) -> Any:
        return self._result


class StubWithRawResponse:
    def __init__(self, parent: Any) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> StubRawResponse:
        result = self._parent.create(**kwargs)
        # Extract max output tokens from the result's usage if available
        max_tokens = "4096"  # default
        if hasattr(result, 'usage') and result.usage and 'output_tokens' in result.usage:
            max_tokens = str(result.usage['output_tokens'])
        elif isinstance(result, dict) and 'usage' in result and 'completion_tokens' in result['usage']:
            max_tokens = str(result['usage']['completion_tokens'])
        
        headers = {"x-openai-limit-max-output-tokens": max_tokens}
        return StubRawResponse(result, headers)


class StubChatCompletions:
    def __init__(
        self,
        *,
        create_result: Any,
        stream_events: Iterable[Any] | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self._create_result = create_result
        self._stream_events = list(stream_events or [])
        self._create_error = create_error
        self.with_raw_response = StubWithRawResponse(self)

    def create(self, *, stream: bool = False, **_: Any) -> Any:
        if self._create_error is not None:
            raise self._create_error
        if stream:
            return StubStream(self._stream_events)
        return self._create_result


class StubModels:
    def __init__(
        self,
        *,
        list_data: list[Any] | None = None,
        metadata_by_id: dict[str, Any] | None = None,
    ) -> None:
        self._list_data = list_data or []
        self._metadata = metadata_by_id or {}

    def list(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(data=list(self._list_data))

    def retrieve(self, model_id: str, **_: Any) -> Any:
        return self._metadata.get(model_id, {})


class StubOpenAIClient:
    def __init__(
        self,
        *,
        responses: StubResponses,
        chat_completions: StubChatCompletions,
        models: StubModels,
        **_: Any,
    ) -> None:
        self.responses = responses
        self.chat = SimpleNamespace(completions=chat_completions)
        self.models = models
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _install_stub_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: StubResponses,
    chat_completions: StubChatCompletions,
    models: StubModels,
) -> StubOpenAIClient:
    stub_client = StubOpenAIClient(
        responses=responses,
        chat_completions=chat_completions,
        models=models,
    )
    monkeypatch.setattr(openai_sdk, "_OpenAIClient", lambda **_: stub_client, raising=False)
    monkeypatch.setattr(openai_sdk, "_APIStatusError", StubAPIStatusError, raising=False)
    monkeypatch.setattr(openai_sdk, "_OpenAIError", Exception, raising=False)
    monkeypatch.setattr(openai_sdk, "_OPENAI_IMPORT_ERROR", None, raising=False)
    return stub_client


def _provider_settings() -> ProviderSettings:
    return ProviderSettings(
        base_url="https://api.openai.com/v1",
        api_key="token",
        model="gpt-4o",
        timeout=30.0,
        organisation=None,
        extra_headers=None,
        user_agent="novelWriter-Test",
        transport=None,
    )


def _responses_object(text: str, output_tokens: int) -> Any:
    return SimpleNamespace(output_text=[text], usage={"output_tokens": output_tokens})


def _chat_response(text: str, completion_tokens: int) -> Any:
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                }
            }
        ],
        "usage": {"completion_tokens": completion_tokens},
    }


def test_openai_sdk_provider_detects_responses_support(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = StubResponses(create_result=_responses_object("pong", 24))
    chat = StubChatCompletions(create_result=_chat_response("pong", 12))
    models = StubModels(
        list_data=[{"id": "gpt-4o", "display_name": "GPT-4o", "owned_by": "openai"}],
        metadata_by_id={"gpt-4o": {"id": "gpt-4o", "output_token_limit": 4096}},
    )
    _install_stub_client(monkeypatch, responses=responses, chat_completions=chat, models=models)

    provider = OpenAISDKProvider(_provider_settings())
    capabilities = provider.ensure_capabilities()

    assert capabilities.supports_responses is True
    assert capabilities.preferred_endpoint == "responses"
    assert capabilities.max_output_tokens == 24
    assert "responses_probe" in capabilities.metadata


def test_openai_sdk_provider_detects_chat_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    responses_error = StubAPIStatusError(404, "not found")
    responses = StubResponses(create_result=None, create_error=responses_error)
    chat = StubChatCompletions(create_result=_chat_response("pong", 18))
    models = StubModels()
    _install_stub_client(monkeypatch, responses=responses, chat_completions=chat, models=models)

    provider = OpenAISDKProvider(_provider_settings())
    capabilities = provider.ensure_capabilities()

    assert capabilities.supports_responses is False
    assert capabilities.supports_chat_completions is True
    assert capabilities.preferred_endpoint == "chat_completions"
    assert capabilities.metadata["chat_probe"]["success"] is True


def test_openai_sdk_provider_streams_via_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    stream_events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(type="response.completed"),
    ]
    responses = StubResponses(
        create_result=_responses_object("hello", 10),
        stream_events=stream_events,
    )
    chat = StubChatCompletions(create_result=_chat_response("fallback", 5))
    models = StubModels()
    _install_stub_client(monkeypatch, responses=responses, chat_completions=chat, models=models)

    provider = OpenAISDKProvider(_provider_settings())
    provider.ensure_capabilities()

    session = provider.generate(
        messages=[{"role": "user", "content": "Say hi"}],
        stream=True,
    )
    chunks = list(session.iter_text())
    assert "".join(chunks) == "Hello"
    session.close()


def test_openai_sdk_provider_falls_back_to_chat_on_stream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = StubResponses(
        create_result=_responses_object("", 4),
        stream_error=StubAPIStatusError(500, "boom"),
    )
    chat_stream_events = [
        {
            "choices": [
                {"delta": {"content": [{"type": "text", "text": "Hi"}]}},
            ]
        },
        {
            "choices": [
                {"delta": {"content": [{"type": "text", "text": " there"}]}}
            ]
        },
    ]
    chat = StubChatCompletions(
        create_result=_chat_response("Hi there", 6),
        stream_events=chat_stream_events,
    )
    models = StubModels()
    _install_stub_client(monkeypatch, responses=responses, chat_completions=chat, models=models)

    provider = OpenAISDKProvider(_provider_settings())
    provider.ensure_capabilities()

    session = provider.generate(
        messages=[{"role": "user", "content": "Greet"}],
        stream=True,
    )
    chunks = list(session.iter_text())
    assert "".join(chunks) == "Hi there"
    session.close()


def test_openai_sdk_provider_lists_and_normalises_models(monkeypatch: pytest.MonkeyPatch) -> None:
    list_data = [
        {
            "id": "gpt-4o-mini",
            "display_name": "GPT-4o mini",
            "owned_by": "openai",
            "metadata": {"input_token_limit": "2048", "output_token_limit": "1024"},
        },
    ]
    models = StubModels(list_data=list_data, metadata_by_id={})
    responses = StubResponses(create_result=_responses_object("pong", 12))
    chat = StubChatCompletions(create_result=_chat_response("pong", 6))
    _install_stub_client(monkeypatch, responses=responses, chat_completions=chat, models=models)

    provider = OpenAISDKProvider(_provider_settings())
    entries = provider.list_models(force=True)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "gpt-4o-mini"
    assert entry["input_token_limit"] == 2048
    assert entry["output_token_limit"] == 1024
