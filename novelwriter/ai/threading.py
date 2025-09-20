"""Threading utilities for the novelWriter AI background tasks."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Protocol

from PyQt6.QtCore import QObject, QRunnable, QThreadPool

__all__ = [
    "AiCancellationToken",
    "AiTaskHandle",
    "AiThreadPoolExecutor",
    "get_ai_executor",
]


class AiTaskAborted(RuntimeError):
    """Raised when a background task is cancelled."""


class AiCancellationToken:
    """Cancellation helper shared between the executor and workers."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal the associated task to stop as soon as practical."""

        self._event.set()

    def cancelled(self) -> bool:
        """Return ``True`` when cancellation was requested."""

        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until cancellation is requested or timeout expires."""

        return self._event.wait(timeout)


class AiWorker(Protocol):  # pragma: no cover - structural typing only
    """Protocol implemented by background workers executed via the pool."""

    def attach_execution(self, token: AiCancellationToken, deadline: Optional[float]) -> None:
        ...

    def run(self) -> None:
        ...


class AiTaskHandle:
    """Track the lifecycle of a submitted background task."""

    __slots__ = ("_token", "_done", "_lock")

    def __init__(self, token: AiCancellationToken) -> None:
        self._token = token
        self._done = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Request cancellation for the underlying task."""

        self._token.cancel()

    def is_running(self) -> bool:
        """Return ``True`` while the task is active."""

        with self._lock:
            return not self._done

    def _mark_done(self) -> None:
        with self._lock:
            self._done = True

    @property
    def token(self) -> AiCancellationToken:
        """Expose the cancellation token for advanced integrations."""

        return self._token


class AiThreadPoolExecutor(QObject):
    """Submit background work to the global Qt thread pool."""

    def __init__(self, *, pool: Optional[QThreadPool] = None) -> None:
        super().__init__()
        self._pool = pool or QThreadPool.globalInstance()
        self._active: set[AiTaskHandle] = set()
        self._lock = threading.Lock()

    def submit_worker(
        self,
        worker: AiWorker,
        *,
        timeout: Optional[float] = None,
    ) -> AiTaskHandle:
        """Submit a worker object to the pool and return a task handle."""

        token = AiCancellationToken()
        handle = AiTaskHandle(token)
        runnable = _WorkerRunnable(worker, handle, timeout, self._finalise_task)
        with self._lock:
            self._active.add(handle)
        self._pool.start(runnable)
        return handle

    def has_running_tasks(self) -> bool:
        """Return ``True`` if at least one submitted task is still running."""

        with self._lock:
            return any(handle.is_running() for handle in self._active)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _finalise_task(self, handle: AiTaskHandle) -> None:
        handle._mark_done()
        with self._lock:
            self._active.discard(handle)


class _WorkerRunnable(QRunnable):
    """Execute a worker inside a thread pool slot."""

    def __init__(
        self,
        worker: AiWorker,
        handle: AiTaskHandle,
        timeout: Optional[float],
        finaliser: Callable[[AiTaskHandle], None],
    ) -> None:
        super().__init__()
        self._worker = worker
        self._handle = handle
        self._timeout = timeout
        self._finaliser = finaliser
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D401 - Qt compatible signature
        deadline = None
        if self._timeout is not None and self._timeout > 0:
            deadline = time.monotonic() + self._timeout

        attach = getattr(self._worker, "attach_execution", None)
        if callable(attach):
            attach(self._handle.token, deadline)
        else:  # pragma: no cover - defensive fallback
            setattr(self._worker, "_ai_executor_token", self._handle.token)
            setattr(self._worker, "_ai_executor_deadline", deadline)

        try:
            self._worker.run()
        finally:
            self._finaliser(self._handle)


_executor_lock = threading.Lock()
_executor_instance: Optional[AiThreadPoolExecutor] = None


def get_ai_executor() -> AiThreadPoolExecutor:
    """Return a lazily instantiated shared executor instance."""

    global _executor_instance
    with _executor_lock:
        if _executor_instance is None:
            _executor_instance = AiThreadPoolExecutor()
        return _executor_instance
