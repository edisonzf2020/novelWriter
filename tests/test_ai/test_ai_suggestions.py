"""Tests for AI suggestion streaming utilities."""
from __future__ import annotations

from typing import Any, Iterable, List

import pytest

from novelwriter.ai import NWAiApi
from novelwriter.core.project import NWProject

from tests.tools import buildTestProject


class FakeResponse:
    """Mimic the subset of httpx.Response used in tests."""

    def __init__(self, chunks: Iterable[str]) -> None:
        self._chunks = [chunk for chunk in chunks]
        self.closed = False
        self.text = "".join(self._chunks)

    def iter_text(self, chunk_size: int = 256) -> Iterable[str]:  # pragma: no cover - trivial generator
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class FakeStreamResponse(FakeResponse):
    def __enter__(self) -> "FakeStreamResponse":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()


class FakeProvider:
    """Simple provider stub capturing invocation data."""

    def __init__(self, chunks: Iterable[str]) -> None:
        self._chunks = list(chunks)
        self.last_request: dict[str, Any] | None = None
        self.last_response: FakeResponse | None = None

    def generate(self, messages: List[dict[str, Any]], *, stream: bool, **kwargs: Any) -> Any:
        self.last_request = {"messages": messages, "stream": stream, "kwargs": kwargs}
        if stream:
            response = FakeStreamResponse(self._chunks)
        else:
            response = FakeResponse(["".join(self._chunks)])
        self.last_response = response
        return response


@pytest.fixture()
def api_with_project(projPath, mockRnd, mockGUIwithTheme):
    project = NWProject()
    buildTestProject(project, projPath)
    return NWAiApi(project)


def _patch_provider(monkeypatch, api: NWAiApi, provider: FakeProvider) -> None:
    monkeypatch.setattr(api, "_ensure_provider", lambda: provider, raising=False)


def test_stream_chat_completion_yields_chunks(monkeypatch, api_with_project) -> None:
    provider = FakeProvider(["Hello", " world"])  # noqa: RUF005 - test literal spacing
    _patch_provider(monkeypatch, api_with_project, provider)

    stream = api_with_project.streamChatCompletion(
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert hasattr(stream, "close")

    chunks = list(stream)

    assert chunks == ["Hello", " world"]
    assert provider.last_request == {
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True,
        "kwargs": {"tools": None},
    }
    assert provider.last_response is not None
    assert provider.last_response.closed is True

    audit_operations = {entry["operation"] for entry in api_with_project.get_audit_log()}
    assert "provider.request.dispatched" in audit_operations
    assert "provider.request.succeeded" in audit_operations


def test_stream_chat_completion_non_stream(monkeypatch, api_with_project) -> None:
    provider = FakeProvider(["All good"])
    _patch_provider(monkeypatch, api_with_project, provider)

    stream = api_with_project.streamChatCompletion(
        messages=[{"role": "user", "content": "Check status"}],
        stream=False,
        extra={"max_output_tokens": 128},
    )

    assert hasattr(stream, "close")

    chunks = list(stream)

    assert chunks == ["All good"]
    assert provider.last_request is not None
    assert provider.last_request["stream"] is False
    assert provider.last_request["kwargs"] == {"max_output_tokens": 128, "tools": None}
    assert provider.last_response is not None
    assert provider.last_response.closed is True


def test_stream_chat_completion_close_releases_resources(monkeypatch, api_with_project) -> None:
    provider = FakeProvider(["chunk"])
    _patch_provider(monkeypatch, api_with_project, provider)

    stream = api_with_project.streamChatCompletion(
        messages=[{"role": "user", "content": "Ping"}],
    )

    assert hasattr(stream, "close")

    # Prime the iterator so the provider session is created.
    assert next(stream) == "chunk"

    stream.close()

    assert provider.last_response is not None
    assert provider.last_response.closed is True

    # Closing again should be a no-op.
    stream.close()
    assert provider.last_response.closed is True
