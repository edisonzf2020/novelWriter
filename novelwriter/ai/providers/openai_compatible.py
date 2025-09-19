"""OpenAI API compatible provider with endpoint capability detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from novelwriter.ai.errors import NWAiProviderError

try:  # Optional dependency (novelWriter[ai])
    import httpx as _httpx  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency missing
    _httpx = None
    _HTTPX_IMPORT_ERROR = exc
else:  # pragma: no branch - executed when dependency available
    _HTTPX_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

from .base import BaseProvider, ProviderCapabilities, ProviderSettings

logger = logging.getLogger(__name__)


_RESPONSES_ENDPOINT = "/v1/responses"
_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
_MODELS_ENDPOINT = "/v1/models/{model}"

_DETECTION_TIMEOUT = _httpx.Timeout(10.0) if _httpx is not None else None
_USER_AGENT = "novelWriter-AI-Provider/1.0"


@dataclass(slots=True)
class _ProbeOutcome:
    """Internal helper describing the result of a detection probe."""

    success: bool
    status_code: int | None
    stream_supported: bool
    tool_call_supported: bool
    max_output_tokens: int | None
    details: dict[str, Any]
    error: str | None = None


class OpenAICompatibleProvider(BaseProvider):
    """Provider implementation for OpenAI compatible HTTP endpoints."""

    def __init__(self, settings: ProviderSettings) -> None:
        if _httpx is None:
            message = (
                "OpenAI-compatible provider requires the optional dependency 'httpx'. "
                "Install novelWriter[ai] to enable AI Copilot network access."
            )
            raise NWAiProviderError(message) from _HTTPX_IMPORT_ERROR

        super().__init__(settings)
        self._client: "httpx.Client | None" = None

    # ------------------------------------------------------------------
    # BaseProvider overrides
    # ------------------------------------------------------------------
    def _detect_capabilities(self) -> ProviderCapabilities:
        client = self._ensure_client()

        responses_probe = self._probe_responses_endpoint(client)
        chat_probe: _ProbeOutcome | None = None

        supports_responses = responses_probe.success
        supports_chat = False
        preferred_endpoint = "responses" if supports_responses else "chat_completions"
        supports_stream = responses_probe.stream_supported
        supports_tools = responses_probe.tool_call_supported
        max_tokens = responses_probe.max_output_tokens
        detection_meta: dict[str, Any] = {
            "responses_probe": self._normalise_probe(responses_probe),
        }

        if not supports_responses:
            chat_probe = self._probe_chat_completions(client)
            detection_meta["chat_probe"] = self._normalise_probe(chat_probe)
            supports_chat = chat_probe.success
            supports_stream = supports_stream or chat_probe.stream_supported
            supports_tools = supports_tools or chat_probe.tool_call_supported
            max_tokens = max_tokens or chat_probe.max_output_tokens
            if not supports_chat:
                reason = chat_probe.error or responses_probe.error or "No compatible endpoint available."
                raise NWAiProviderError(f"OpenAI-compatible endpoint detection failed: {reason}")
        else:
            # Even when responses works we attempt a lightweight chat probe for diagnostics.
            try:
                chat_probe = self._probe_chat_completions(client, lightweight=True)
                detection_meta["chat_probe"] = self._normalise_probe(chat_probe)
                supports_chat = chat_probe.success
                supports_stream = supports_stream or chat_probe.stream_supported
                supports_tools = supports_tools or chat_probe.tool_call_supported
                max_tokens = max_tokens or chat_probe.max_output_tokens
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                logger.debug("Chat completion probe skipped: %s", exc)
                supports_chat = False

        model_meta = self._fetch_model_metadata(client)
        if model_meta:
            detection_meta["model_metadata"] = model_meta
            if not max_tokens:
                max_tokens = self._extract_output_limit(model_meta)

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
        """Execute a generation request using the preferred endpoint."""

        client = self._ensure_client()
        capabilities = self.capabilities

        extra = dict(kwargs)
        timeout_override = extra.pop("timeout", None)

        if capabilities.preferred_endpoint == "responses":
            payload = self._build_responses_payload(messages, tools=tools, stream=stream, extra=extra)
            url = _RESPONSES_ENDPOINT
        else:
            payload = self._build_chat_payload(messages, tools=tools, stream=stream, extra=extra)
            url = _CHAT_COMPLETIONS_ENDPOINT

        timeout = timeout_override if timeout_override is not None else self.settings.timeout

        logger.debug(
            "Dispatching OpenAI-compatible request via %s (stream=%s)",
            url,
            stream,
        )

        if stream:
            return client.stream("POST", url, json=payload, timeout=timeout)
        return client.post(url, json=payload, timeout=timeout)


    def close(self) -> None:
        client = self._client
        if client is not None:
            client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_client(self) -> "httpx.Client":
        if _httpx is None:
            raise NWAiProviderError(
                "OpenAI-compatible provider requires the optional dependency 'httpx'."
            )
        if self._client is not None:
            return self._client

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.settings.organisation:
            headers["OpenAI-Organization"] = self.settings.organisation
        if self.settings.user_agent:
            headers["User-Agent"] = self.settings.user_agent
        else:
            headers["User-Agent"] = _USER_AGENT
        if self.settings.extra_headers:
            headers.update(self.settings.extra_headers)

        assert _httpx is not None  # Runtime guard for optional dependency

        base_url = _httpx.URL(self.settings.base_url)
        client_base = base_url.copy_with(path="/", query=None, fragment=None)

        self._client = _httpx.Client(
            base_url=client_base,
            headers=headers,
            timeout=self.settings.timeout,
            transport=self.settings.transport,
        )
        return self._client

    def _probe_responses_endpoint(self, client: "httpx.Client") -> _ProbeOutcome:
        if _httpx is None:
            raise NWAiProviderError(
                "OpenAI-compatible provider requires the optional dependency 'httpx'."
            )
        payload = {
            "model": self.settings.model,
            "input": "ping",
            "max_output_tokens": 1,
            "metadata": {"origin": "novelwriter-capability-probe"},
        }

        try:
            response = client.post(_RESPONSES_ENDPOINT, json=payload, timeout=_DETECTION_TIMEOUT)
        except _httpx.HTTPError as exc:
            return _ProbeOutcome(
                success=False,
                status_code=None,
                stream_supported=False,
                tool_call_supported=False,
                max_output_tokens=None,
                details={"error": str(exc)},
                error=str(exc),
            )

        if 200 <= response.status_code < 300:
            data = self._safe_json(response)
            max_tokens = self._extract_header_limit(response) or self._extract_usage_limit(data)
            return _ProbeOutcome(
                success=True,
                status_code=response.status_code,
                stream_supported=True,
                tool_call_supported=True,
                max_output_tokens=max_tokens,
                details={"headers": dict(response.headers), "body": data},
            )

        error_message = self._build_error_message("/v1/responses", response)
        return _ProbeOutcome(
            success=False,
            status_code=response.status_code,
            stream_supported=False,
            tool_call_supported=False,
            max_output_tokens=None,
            details={"headers": dict(response.headers), "body": self._safe_json(response)},
            error=error_message,
        )

    def _probe_chat_completions(
        self,
        client: "httpx.Client",
        *,
        lightweight: bool = False,
    ) -> _ProbeOutcome:
        if _httpx is None:
            raise NWAiProviderError(
                "OpenAI-compatible provider requires the optional dependency 'httpx'."
            )
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }

        if not lightweight:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Capability detection placeholder.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ping": {"type": "string"},
                            },
                        },
                    },
                }
            ]

        try:
            response = client.post(_CHAT_COMPLETIONS_ENDPOINT, json=payload, timeout=_DETECTION_TIMEOUT)
        except _httpx.HTTPError as exc:
            return _ProbeOutcome(
                success=False,
                status_code=None,
                stream_supported=False,
                tool_call_supported=False,
                max_output_tokens=None,
                details={"error": str(exc)},
                error=str(exc),
            )

        if 200 <= response.status_code < 300:
            data = self._safe_json(response)
            tool_supported = not bool(data.get("error"))
            max_tokens = self._extract_header_limit(response) or self._extract_usage_limit(data)
            return _ProbeOutcome(
                success=True,
                status_code=response.status_code,
                stream_supported=True,
                tool_call_supported=tool_supported,
                max_output_tokens=max_tokens,
                details={"headers": dict(response.headers), "body": data},
            )

        error_message = self._build_error_message("/v1/chat/completions", response)
        return _ProbeOutcome(
            success=False,
            status_code=response.status_code,
            stream_supported=False,
            tool_call_supported=False,
            max_output_tokens=None,
            details={"headers": dict(response.headers), "body": self._safe_json(response)},
            error=error_message,
        )

    def _fetch_model_metadata(self, client: "httpx.Client") -> dict[str, Any] | None:
        if _httpx is None:
            raise NWAiProviderError(
                "OpenAI-compatible provider requires the optional dependency 'httpx'."
            )
        endpoint = _MODELS_ENDPOINT.format(model=self.settings.model)
        try:
            response = client.get(endpoint, timeout=_DETECTION_TIMEOUT)
        except _httpx.HTTPError as exc:
            logger.debug("Model metadata fetch failed: %s", exc)
            return None

        if 200 <= response.status_code < 300:
            data = self._safe_json(response)
            if isinstance(data, dict):
                return data
        else:
            logger.debug(
                "Model metadata probe failed (%s): %s",
                response.status_code,
                self._safe_json(response),
            )
        return None

    def _build_responses_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": self._adapt_messages_for_responses(messages),
        }
        if tools:
            payload["tools"] = tools
        payload.update(extra)
        if stream:
            payload.setdefault("stream", True)
        return payload

    def _build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        payload.update(extra)
        if stream:
            payload.setdefault("stream", True)
        return payload

    @staticmethod
    def _adapt_messages_for_responses(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        if not messages:
            return ""

        content: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            text = message.get("content", "")
            if isinstance(text, str):
                content.append({"role": role, "content": text})
            else:
                content.append({"role": role, "content": text})
        return content

    @staticmethod
    def _extract_usage_limit(payload: Any) -> int | None:
        if isinstance(payload, dict):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                for key in ("output_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int) and value > 0:
                        return max(value, usage.get("output_tokens", value))
        return None

    @staticmethod
    def _extract_header_limit(response: "httpx.Response") -> int | None:
        for header in ("x-openai-limit-max-output-tokens", "x-openai-max-output-tokens"):
            if header in response.headers:
                try:
                    return int(response.headers[header])
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_output_limit(model_meta: dict[str, Any]) -> int | None:
        for key in ("output_token_limit", "max_output_tokens", "max_output_tokens_per_request", "context_length"):
            value = model_meta.get(key)
            if isinstance(value, int):
                return value
        limits = model_meta.get("limits")
        if isinstance(limits, dict):
            for key in ("output_tokens", "max_output_tokens"):
                value = limits.get(key)
                if isinstance(value, int):
                    return value
        return None

    @staticmethod
    def _normalise_probe(probe: _ProbeOutcome | None) -> dict[str, Any] | None:
        if probe is None:
            return None
        payload = {
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

    @staticmethod
    def _safe_json(response: "httpx.Response") -> Any:
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    @staticmethod
    def _build_error_message(endpoint: str, response: "httpx.Response") -> str:
        payload = OpenAICompatibleProvider._safe_json(response)
        if isinstance(payload, dict) and "error" in payload:
            detail = payload["error"]
            if isinstance(detail, dict) and "message" in detail:
                return f"{endpoint} {response.status_code}: {detail['message']}"
            return f"{endpoint} {response.status_code}: {detail}"
        return f"{endpoint} {response.status_code}"
