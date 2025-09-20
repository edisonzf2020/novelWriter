"""Provider implementation backed by the official OpenAI Python SDK."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Sequence

from novelwriter.ai.errors import NWAiProviderError
from novelwriter.ai.performance import current_span

from .base import BaseProvider, ProviderCapabilities, ProviderSettings

try:  # Optional dependency (novelWriter[ai])
    from openai import APIStatusError as _APIStatusError
    from openai import OpenAI as _OpenAIClient
    from openai import OpenAIError as _OpenAIError
except ImportError as exc:  # pragma: no cover - optional dependency missing
    _OpenAIClient = None
    _APIStatusError = None
    _OpenAIError = None
    _OPENAI_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency available
    _OPENAI_IMPORT_ERROR = None

try:  # Optional injection for tests
    import httpx as _httpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _httpx = None

logger = logging.getLogger(__name__)


_MODEL_CACHE_TTL = timedelta(seconds=300)
_PROBE_TIMEOUT = 10.0


@dataclass(slots=True)
class _ProbeResult:
    """Internal helper describing the result of a detection probe."""

    success: bool
    status_code: int | None = None
    stream_supported: bool = False
    tool_call_supported: bool = False
    max_output_tokens: int | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


class _NonStreamingResponse:
    """Minimal wrapper exposing a ``text`` attribute and ``close`` method."""

    def __init__(self, text: str) -> None:
        self.text = text

    def close(self) -> None:  # noqa: D401 - interface parity with httpx.Response
        """Match httpx response close semantics."""

        return None


class _ResponsesStreamAdapter:
    """Adapter exposing ``iter_text`` for Responses streaming events."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._iterator: Iterable[Any] | None = None
        self._closed = False

    def __enter__(self) -> "_ResponsesStreamAdapter":
        iterator = getattr(self._stream, "__enter__", None)
        if callable(iterator):
            self._iterator = iterator()
        else:
            self._iterator = self._stream
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        closer = getattr(self._stream, "__exit__", None)
        if callable(closer):
            return bool(closer(exc_type, exc, tb))
        return False

    def iter_text(self, chunk_size: int = 256) -> Iterator[str]:  # noqa: D401 - parity with httpx
        events: Iterable[Any] | Any
        if self._iterator is not None:
            events = self._iterator
        else:
            iterator = getattr(self._stream, "__iter__", None)
            events = iterator() if callable(iterator) else self._stream
        yield from _yield_responses_text(events)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._stream, "close", None)
        if callable(closer):
            closer()


class _ChatStreamAdapter:
    """Adapter exposing ``iter_text`` for Chat Completions streaming chunks."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._iterator: Iterable[Any] | None = None
        self._closed = False

    def __enter__(self) -> "_ChatStreamAdapter":
        iterator = getattr(self._stream, "__enter__", None)
        if callable(iterator):
            self._iterator = iterator()
        else:
            self._iterator = self._stream
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        closer = getattr(self._stream, "__exit__", None)
        if callable(closer):
            return bool(closer(exc_type, exc, tb))
        return False

    def iter_text(self, chunk_size: int = 256) -> Iterator[str]:  # noqa: D401 - parity with httpx
        events: Iterable[Any] | Any
        if self._iterator is not None:
            events = self._iterator
        else:
            iterator = getattr(self._stream, "__iter__", None)
            events = iterator() if callable(iterator) else self._stream
        yield from _yield_chat_text(events)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._stream, "close", None)
        if callable(closer):
            closer()


class OpenAISDKProvider(BaseProvider):
    """Provider implementation delegating to ``openai.OpenAI`` client."""

    def __init__(self, settings: ProviderSettings) -> None:
        if _OpenAIClient is None:
            message = (
                "OpenAI SDK provider requires the optional dependency 'openai'. "
                "Install novelWriter[ai] to enable AI Copilot network access."
            )
            raise NWAiProviderError(message) from _OPENAI_IMPORT_ERROR

        super().__init__(settings)
        self._client: Any | None = None
        self._http_client: Any | None = None
        self._models_cache: list[dict[str, Any]] | None = None
        self._models_cache_fetched_at: datetime | None = None

    # ------------------------------------------------------------------
    # BaseProvider overrides
    # ------------------------------------------------------------------
    def _detect_capabilities(self) -> ProviderCapabilities:
        client = self._ensure_client()

        responses_probe = self._probe_responses_endpoint(client)
        chat_probe: _ProbeResult | None = None

        supports_responses = responses_probe.success
        supports_chat = False
        preferred_endpoint = "responses" if supports_responses else "chat_completions"
        supports_stream = responses_probe.stream_supported
        supports_tools = responses_probe.tool_call_supported
        max_tokens = responses_probe.max_output_tokens
        detection_meta: dict[str, Any] = {
            "responses_probe": _probe_to_dict(responses_probe),
        }

        if not supports_responses:
            chat_probe = self._probe_chat_endpoint(client)
            detection_meta["chat_probe"] = _probe_to_dict(chat_probe)
            supports_chat = chat_probe.success
            supports_stream = supports_stream or chat_probe.stream_supported
            supports_tools = supports_tools or chat_probe.tool_call_supported
            if not max_tokens:
                max_tokens = chat_probe.max_output_tokens
            if not supports_chat:
                reason = chat_probe.error or responses_probe.error or "No compatible endpoint detected."
                raise NWAiProviderError(f"OpenAI SDK endpoint detection failed: {reason}")
        else:
            try:
                chat_probe = self._probe_chat_endpoint(client, lightweight=True)
                detection_meta["chat_probe"] = _probe_to_dict(chat_probe)
                supports_chat = chat_probe.success
                supports_stream = supports_stream or chat_probe.stream_supported
                supports_tools = supports_tools or chat_probe.tool_call_supported
                if not max_tokens:
                    max_tokens = chat_probe.max_output_tokens
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                logger.debug("Chat endpoint probe failed (diagnostic only): %s", exc)
                supports_chat = False

        model_meta = self._fetch_model_metadata(client)
        if model_meta:
            detection_meta["model_metadata"] = model_meta
            if not max_tokens:
                max_tokens = _extract_output_limit(model_meta)

        detection_meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        return ProviderCapabilities(
            preferred_endpoint=preferred_endpoint,
            supports_responses=supports_responses,
            supports_chat_completions=supports_chat,
            supports_stream=supports_stream,
            supports_tool_calls=supports_tools,
            max_output_tokens=max_tokens,
            detected_at=datetime.now(timezone.utc),
            metadata=detection_meta,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        client = self._ensure_client()
        capabilities = self.capabilities
        extra = dict(kwargs)
        timeout_override = extra.pop("timeout", None)

        try:
            if capabilities.preferred_endpoint == "responses":
                return self._dispatch_via_responses(
                    client,
                    messages,
                    tools=tools,
                    stream=stream,
                    extra=extra,
                    timeout=timeout_override,
                )
            return self._dispatch_via_chat(
                client,
                messages,
                tools=tools,
                stream=stream,
                extra=extra,
                timeout=timeout_override,
            )
        except NWAiProviderError as exc:
            if capabilities.preferred_endpoint == "responses":
                logger.debug("Responses endpoint failed, falling back to chat completions: %s", exc)
                span = current_span()
                if span is not None:
                    span.mark_degraded("responses", "chat_completions")
                return self._dispatch_via_chat(
                    client,
                    messages,
                    tools=tools,
                    stream=stream,
                    extra=extra,
                    timeout=timeout_override,
                )
            raise

    def list_models(self, *, force: bool = False) -> list[dict[str, Any]]:
        client = self._ensure_client()
        if (
            not force
            and self._models_cache is not None
            and self._models_cache_fetched_at is not None
            and datetime.now(timezone.utc) - self._models_cache_fetched_at <= _MODEL_CACHE_TTL
        ):
            return [copy.deepcopy(entry) for entry in self._models_cache]

        models = self._request_model_list(client)
        self._models_cache = [copy.deepcopy(entry) for entry in models]
        self._models_cache_fetched_at = datetime.now(timezone.utc)
        return [copy.deepcopy(entry) for entry in models]

    def get_model_metadata(self, model_id: str, *, force: bool = False) -> dict[str, Any] | None:
        client = self._ensure_client()
        if not force and self._models_cache is not None:
            cached = self._get_cached_model(model_id)
            if cached is not None:
                return cached

        metadata = self._request_model_metadata(client, model_id)
        if metadata is not None:
            self._store_model_cache_entry(metadata)
        return metadata

    def close(self) -> None:
        client = self._client
        if client is not None:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()
            self._client = None
        if self._http_client is not None:
            closer = getattr(self._http_client, "close", None)
            if callable(closer):
                closer()
            self._http_client = None
        self._models_cache = None
        self._models_cache_fetched_at = None

    # ------------------------------------------------------------------
    # OpenAI client helpers
    # ------------------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        if _OpenAIClient is None:
            message = (
                "OpenAI SDK provider requires the optional dependency 'openai'. "
                "Install novelWriter[ai] to enable AI Copilot network access."
            )
            raise NWAiProviderError(message) from _OPENAI_IMPORT_ERROR

        client_kwargs: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "base_url": self.settings.base_url,
        }
        if self.settings.organisation:
            client_kwargs["organization"] = self.settings.organisation
        if self.settings.timeout:
            client_kwargs["timeout"] = float(self.settings.timeout)

        headers: dict[str, str] = {}
        if self.settings.user_agent:
            headers["User-Agent"] = self.settings.user_agent
        if self.settings.extra_headers:
            headers.update(self.settings.extra_headers)
        if headers:
            client_kwargs["default_headers"] = headers

        if self.settings.transport is not None:
            if _httpx is None:
                raise NWAiProviderError(
                    "OpenAI SDK provider requires httpx for custom transports."
                )
            http_client = _httpx.Client(
                transport=self.settings.transport,
                timeout=self.settings.timeout or _PROBE_TIMEOUT,
            )
            self._http_client = http_client
            client_kwargs["http_client"] = http_client

        self._client = _OpenAIClient(**client_kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Capability probing
    # ------------------------------------------------------------------
    def _probe_responses_endpoint(self, client: Any) -> _ProbeResult:
        messages = [{"role": "user", "content": "Reply with the word 'pong'."}]
        payload = {
            "model": self.settings.model,
            "input": _build_responses_input(messages),
            "max_output_tokens": 32,
            "temperature": 0,
        }

        try:
            response = client.responses.create(timeout=_PROBE_TIMEOUT, **payload)
        except Exception as exc:  # noqa: BLE001 - capture and normalise
            return _normalise_exception(exc, endpoint="responses")

        output_text = _collapse_responses_output(response)
        details = {
            "output_text": output_text,
            "usage": _safe_as_dict(getattr(response, "usage", None)),
        }
        max_tokens = _extract_usage_limit(details["usage"], "output_tokens")
        return _ProbeResult(
            success=True,
            stream_supported=True,
            tool_call_supported=True,
            max_output_tokens=max_tokens,
            details=details,
        )

    def _probe_chat_endpoint(self, client: Any, *, lightweight: bool = False) -> _ProbeResult:
        messages = [
            {"role": "system", "content": "Reply concisely."},
            {"role": "user", "content": "Return the word 'pong'."},
        ]
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        }
        try:
            response = client.chat.completions.create(timeout=_PROBE_TIMEOUT, **payload)
        except Exception as exc:  # noqa: BLE001 - capture and normalise
            return _normalise_exception(exc, endpoint="chat.completions")

        output_text = _collapse_chat_output(response)
        details = {
            "output_text": output_text,
            "usage": _safe_as_dict(getattr(response, "usage", None)),
        }
        max_tokens = _extract_usage_limit(details["usage"], "completion_tokens")
        return _ProbeResult(
            success=True,
            stream_supported=not lightweight,
            tool_call_supported=True,
            max_output_tokens=max_tokens,
            details=details,
        )

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------
    def _dispatch_via_responses(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
        timeout: float | None,
    ) -> Any:
        payload = {
            "model": self.settings.model,
            "input": _build_responses_input(messages),
        }
        payload.update(_remap_responses_extra(extra))
        if tools:
            payload["tools"] = tools
        timeout_value = timeout if timeout is not None else self.settings.timeout

        try:
            if stream:
                session = client.responses.stream(timeout=timeout_value, **payload)
                return _ResponsesStreamAdapter(session)
            response = client.responses.create(timeout=timeout_value, **payload)
        except Exception as exc:  # noqa: BLE001 - normalise to provider error
            raise _wrap_exception(exc, endpoint="responses") from exc

        text = _collapse_responses_output(response)
        return _NonStreamingResponse(text)

    def _dispatch_via_chat(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
        timeout: float | None,
    ) -> Any:
        payload = {
            "model": self.settings.model,
            "messages": messages,
        }
        payload.update(_remap_chat_extra(extra))
        if tools:
            payload["tools"] = tools
        timeout_value = timeout if timeout is not None else self.settings.timeout

        try:
            if stream:
                session = client.chat.completions.create(
                    stream=True,
                    timeout=timeout_value,
                    **payload,
                )
                return _ChatStreamAdapter(session)
            response = client.chat.completions.create(
                stream=False,
                timeout=timeout_value,
                **payload,
            )
        except Exception as exc:  # noqa: BLE001 - normalise to provider error
            raise _wrap_exception(exc, endpoint="chat.completions") from exc

        text = _collapse_chat_output(response)
        return _NonStreamingResponse(text)

    # ------------------------------------------------------------------
    # Model catalogue helpers
    # ------------------------------------------------------------------
    def _request_model_list(self, client: Any) -> list[dict[str, Any]]:
        try:
            response = client.models.list(timeout=_PROBE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - normalise to provider error
            raise _wrap_exception(exc, endpoint="models.list") from exc

        data = getattr(response, "data", None) or []
        normalised: list[dict[str, Any]] = []
        for entry in data:
            normalised.append(_normalise_model_entry(_safe_as_dict(entry)))
        return normalised

    def _request_model_metadata(self, client: Any, model_id: str) -> dict[str, Any] | None:
        try:
            metadata = client.models.retrieve(model_id, timeout=_PROBE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - convert to provider error
            probe = _normalise_exception(exc, endpoint="models.retrieve")
            if probe.success:
                return None
            logger.debug("Failed to retrieve model metadata for '%s': %s", model_id, probe.error)
            return None
        details = _safe_as_dict(metadata)
        return _normalise_model_entry(details)

    def _store_model_cache_entry(self, metadata: dict[str, Any]) -> None:
        if self._models_cache is None:
            self._models_cache = []
        existing = [entry for entry in self._models_cache if entry.get("id") != metadata.get("id")]
        existing.append(copy.deepcopy(metadata))
        self._models_cache = existing
        self._models_cache_fetched_at = datetime.now(timezone.utc)

    def _get_cached_model(self, model_id: str) -> dict[str, Any] | None:
        if self._models_cache is None:
            return None
        for entry in self._models_cache:
            if entry.get("id") == model_id:
                return copy.deepcopy(entry)
        return None

    def _fetch_model_metadata(self, client: Any) -> dict[str, Any] | None:
        try:
            return self._request_model_metadata(client, self.settings.model)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.debug("Model metadata probe failed: %s", exc)
            return None


def _wrap_exception(exc: Exception, *, endpoint: str) -> NWAiProviderError:
    probe = _normalise_exception(exc, endpoint=endpoint)
    message = probe.error or f"OpenAI SDK error while calling {endpoint}."
    return NWAiProviderError(message)


def _normalise_exception(exc: Exception, *, endpoint: str) -> _ProbeResult:
    status_code: int | None = None
    message = str(exc) or exc.__class__.__name__
    if _APIStatusError is not None and isinstance(exc, _APIStatusError):
        status_code = getattr(exc, "status_code", None)
        message = getattr(exc, "message", None) or message
    elif _OpenAIError is not None and isinstance(exc, _OpenAIError):
        message = getattr(exc, "message", None) or message
    logger.debug("OpenAI SDK call to %s failed: %s", endpoint, message)
    return _ProbeResult(success=False, status_code=status_code, error=message)


def _probe_to_dict(probe: _ProbeResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": probe.success,
        "status_code": probe.status_code,
        "stream_supported": probe.stream_supported,
        "tool_call_supported": probe.tool_call_supported,
        "max_output_tokens": probe.max_output_tokens,
    }
    if probe.error:
        payload["error"] = probe.error
    if probe.details:
        payload["details"] = probe.details
    return payload


def _safe_as_dict(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        return dict(data)
    converter = getattr(data, "model_dump", None)
    if callable(converter):
        return converter()
    converter = getattr(data, "dict", None)
    if callable(converter):
        return converter()
    if hasattr(data, "__dict__"):
        return {key: value for key, value in vars(data).items() if not key.startswith("_")}
    return {"value": data}


def _flatten_text_payload(payload: Any) -> Iterator[str]:
    if payload is None:
        return
    if isinstance(payload, str):
        yield payload
        return
    if isinstance(payload, Mapping):
        if payload.get("type") == "text" and isinstance(payload.get("text"), str):
            yield payload["text"]
            return
        if "content" in payload:
            yield from _flatten_text_payload(payload.get("content"))
        return
    if isinstance(payload, Sequence):
        for item in payload:
            yield from _flatten_text_payload(item)
        return
    yield str(payload)


def _yield_responses_text(events: Iterable[Any]) -> Iterator[str]:
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type is None:
            event_type = _safe_as_dict(event).get("type")
        if event_type in {"response.output_text.delta", "response.output_text.done"}:
            delta = getattr(event, "delta", None)
            if delta is None:
                delta = _safe_as_dict(event).get("delta")
            yield from _flatten_text_payload(delta)
        elif event_type in {"response.error", "error"}:
            message = getattr(event, "message", None) or str(event)
            raise NWAiProviderError(message)
        elif event_type == "response.completed":
            break


def _yield_chat_text(events: Iterable[Any]) -> Iterator[str]:
    for chunk in events:
        payload = _safe_as_dict(chunk)
        choices = payload.get("choices")
        if not isinstance(choices, Sequence):
            continue
        for choice in choices:
            choice_payload = _safe_as_dict(choice)
            delta = choice_payload.get("delta")
            if delta is None:
                continue
            yield from _flatten_text_payload(delta.get("content"))


def _build_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts = list(_flatten_text_payload(content))
        if not parts:
            parts = [""]
        adapted.append(
            {
                "role": role,
                "content": [{"type": "text", "text": part} for part in parts],
            }
        )
    return adapted


def _collapse_responses_output(response: Any) -> str:
    if response is None:
        return ""
    output_text = getattr(response, "output_text", None)
    if output_text:
        if isinstance(output_text, str):
            return output_text
        if isinstance(output_text, Sequence):
            return "".join(str(item) for item in output_text if item is not None)
    output = getattr(response, "output", None)
    if output:
        combined: list[str] = []
        for item in output:
            combined.extend(list(_flatten_text_payload(_safe_as_dict(item))))
        return "".join(combined)
    return ""


def _collapse_chat_output(response: Any) -> str:
    if response is None:
        return ""
    payload = _safe_as_dict(response)
    choices = payload.get("choices")
    if not isinstance(choices, Sequence):
        return ""
    combined: list[str] = []
    for choice in choices:
        choice_payload = _safe_as_dict(choice)
        message = choice_payload.get("message")
        if message is None:
            continue
        combined.extend(list(_flatten_text_payload(_safe_as_dict(message).get("content"))))
    return "".join(combined)


def _extract_usage_limit(usage: Mapping[str, Any] | None, key: str) -> int | None:
    if not usage:
        return None
    try:
        value = usage.get(key)
    except AttributeError:  # pragma: no cover - defensive guard
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_output_limit(metadata: Mapping[str, Any] | None) -> int | None:
    if not metadata:
        return None
    output_limit = metadata.get("output_token_limit")
    if isinstance(output_limit, int):
        return output_limit
    if isinstance(output_limit, str) and output_limit.isdigit():
        return int(output_limit)
    return None


def _normalise_model_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    model_id = str(entry.get("id", "")).strip() or "unknown"
    description = entry.get("description")
    owned_by = entry.get("owned_by")
    display_name = entry.get("display_name") or model_id

    metadata = _safe_as_dict(entry.get("metadata"))
    input_limit = entry.get("input_token_limit") or metadata.get("input_token_limit")
    output_limit = entry.get("output_token_limit") or metadata.get("output_token_limit")

    def _ensure_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    return {
        "id": model_id,
        "display_name": display_name,
        "description": description,
        "input_token_limit": _ensure_int(input_limit),
        "output_token_limit": _ensure_int(output_limit),
        "owned_by": owned_by,
        "capabilities": entry.get("capabilities") or metadata.get("capabilities", {}),
        "metadata": metadata or {},
    }


def _remap_responses_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(extra)
    if "max_tokens" in payload and "max_output_tokens" not in payload:
        payload["max_output_tokens"] = payload.pop("max_tokens")
    return payload


def _remap_chat_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(extra)
    if "max_output_tokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("max_output_tokens")
    return payload
