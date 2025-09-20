"""Tests for the AI threading helpers."""

from __future__ import annotations

import time

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from novelwriter.ai.threading import AiCancellationToken, get_ai_executor


class _TestWorker(QObject):
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._token = None
        self._deadline = None
        self.events: list[str] = []

    def attach_execution(self, token: AiCancellationToken, deadline: float | None) -> None:
        self._token = token
        self._deadline = deadline

    def run(self) -> None:  # noqa: D401 - interface required
        while True:
            if self._token.cancelled():
                self.events.append("cancelled")
                self.cancelled.emit()
                return
            if self._deadline is not None and time.monotonic() >= self._deadline:
                self.events.append("timeout")
                self.finished.emit()
                return
            time.sleep(0.02)


@pytest.mark.usefixtures("qtbot")
def test_executor_handles_cancellation(qtbot) -> None:
    worker = _TestWorker()
    executor = get_ai_executor()

    handle = executor.submit_worker(worker, timeout=2.0)
    qtbot.waitUntil(handle.is_running, timeout=1000)

    handle.cancel()
    qtbot.waitUntil(lambda: "cancelled" in worker.events, timeout=2000)
    assert not handle.is_running()


@pytest.mark.usefixtures("qtbot")
def test_executor_enforces_timeout(qtbot) -> None:
    worker = _TestWorker()
    executor = get_ai_executor()

    handle = executor.submit_worker(worker, timeout=0.25)
    qtbot.waitUntil(lambda: "timeout" in worker.events, timeout=2000)
    assert not handle.is_running()
    assert "timeout" in worker.events
