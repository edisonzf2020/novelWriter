"""Smoke tests ensuring the AI domain package is importable."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from novelwriter.ai import (
    BuildResult,
    DocumentRef,
    NWAiApi,
    NWAiApiError,
    NWAiConfigError,
    NWAiError,
    NWAiProviderError,
    Suggestion,
    TextRange,
)

if TYPE_CHECKING:
    from novelwriter.core.project import NWProject


def test_ai_models_can_be_instantiated() -> None:
    """The exported dataclasses should accept basic constructor arguments."""

    document = DocumentRef(handle="doc-1", name="Doc", parent=None)
    range_ = TextRange(start=0, end=10)
    suggestion = Suggestion(id="sug-1", handle=document.handle, preview="text", diff=None)
    result = BuildResult(format="html", outputPath="/tmp/out.html", success=True)

    assert document.handle == "doc-1"
    assert range_.start == 0
    assert suggestion.preview == "text"
    assert result.success is True


@pytest.mark.parametrize(
    "error_type",
    [NWAiError, NWAiApiError, NWAiProviderError, NWAiConfigError],
)
def test_error_hierarchy_is_available(error_type: type[BaseException]) -> None:
    """Ensure each error type can be constructed."""

    error = error_type("boom")
    assert isinstance(error, Exception)


def test_transaction_methods_are_not_implemented_yet() -> None:
    """Placeholder transaction helpers should still raise until implemented."""

    api = NWAiApi(project=cast("NWProject", object()))

    with pytest.raises(NotImplementedError):
        api.begin_transaction()
