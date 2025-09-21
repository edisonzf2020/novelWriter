"""Timeline utilities for AI transaction and audit history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "HistoryEvent",
    "HistoryOperation",
    "HistoryTransaction",
    "HistorySnapshot",
    "HistoryManager",
]


@dataclass(frozen=True)
class HistoryEvent:
    """Single audit entry describing an AI-related event."""

    event_id: str
    timestamp: datetime
    transaction_id: Optional[str]
    operation: str
    target: Optional[str]
    summary: Optional[str]
    level: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable representation of the event."""

        payload: Dict[str, object] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "target": self.target,
            "summary": self.summary,
            "level": self.level,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class HistoryOperation:
    """Represents a committed operation that can be rolled back."""

    operation: str
    target: Optional[str]
    summary: Optional[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    undo: Optional[Callable[[], None]] = None

    def clone(self) -> "HistoryOperation":
        """Return a shallow copy suitable for rollback processing."""

        return HistoryOperation(
            operation=self.operation,
            target=self.target,
            summary=self.summary,
            metadata=dict(self.metadata) if self.metadata else {},
            undo=self.undo,
        )

    def to_summary(self) -> Dict[str, Any]:
        """Return a metadata-only representation for UI purposes."""

        payload: Dict[str, Any] = {
            "operation": self.operation,
            "target": self.target,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class HistoryTransaction:
    """Aggregated timeline information for a transaction."""

    transaction_id: str
    started_at: datetime
    status: str = "pending"
    completed_at: Optional[datetime] = None
    events: List[HistoryEvent] = field(default_factory=list)
    operations_summary: List[Dict[str, Any]] = field(default_factory=list)
    rollback_available: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable representation of the transaction timeline."""

        payload: Dict[str, object] = {
            "transaction_id": self.transaction_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "events": [event.to_dict() for event in self.events],
            "rollback_available": self.rollback_available,
        }
        if self.completed_at is not None:
            payload["completed_at"] = self.completed_at.isoformat()
        if self.operations_summary:
            payload["operations"] = [dict(item) for item in self.operations_summary]
        return payload


@dataclass
class HistorySnapshot:
    """Container returned by the history manager for UI consumption."""

    transactions: List[HistoryTransaction]
    standalone_events: List[HistoryEvent]

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable mapping suitable for JSON transport."""

        return {
            "transactions": [txn.to_dict() for txn in self.transactions],
            "events": [event.to_dict() for event in self.standalone_events],
        }


class HistoryManager:
    """Collect and aggregate audit events into transaction timelines."""

    def __init__(self, *, max_events: int = 1000) -> None:
        self._events: deque[HistoryEvent] = deque(maxlen=max_events)
        self._lock = RLock()
        self._operations: Dict[str, List[HistoryOperation]] = {}
        self._operation_summaries_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def add_event(
        self,
        *,
        event_id: str,
        timestamp: datetime,
        transaction_id: Optional[str],
        operation: str,
        target: Optional[str],
        summary: Optional[str],
        level: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Persist a new history event."""

        payload = dict(metadata) if metadata else {}
        event = HistoryEvent(
            event_id=event_id,
            timestamp=timestamp,
            transaction_id=transaction_id,
            operation=operation,
            target=target,
            summary=summary,
            level=level,
            metadata=payload,
        )
        with self._lock:
            self._events.append(event)

    def register_operations(
        self,
        transaction_id: str,
        operations: Iterable[HistoryOperation],
    ) -> None:
        """Persist the committed operations for later rollback."""

        clones = [op.clone() for op in operations]
        summaries = [op.to_summary() for op in clones]
        with self._lock:
            self._operations[transaction_id] = clones
            self._operation_summaries_cache[transaction_id] = summaries

    def get_operations_for_rollback(self, transaction_id: str) -> List[HistoryOperation]:
        """Return rollback-capable operations for the transaction."""

        with self._lock:
            stored = self._operations.get(transaction_id, [])
            return [op.clone() for op in stored]

    def clear_operations(self, transaction_id: str) -> None:
        """Remove stored rollback operations for a transaction."""

        with self._lock:
            self._operations.pop(transaction_id, None)

    def has_operations(self, transaction_id: str) -> bool:
        """Return True when rollback metadata exists for the transaction."""

        with self._lock:
            return transaction_id in self._operations

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def snapshot(
        self,
        *,
        transaction_limit: Optional[int] = None,
        event_limit: int = 200,
    ) -> HistorySnapshot:
        """Build a snapshot of recent transactions and standalone events."""

        with self._lock:
            events: Sequence[HistoryEvent] = list(self._events)

        transactions = self._aggregate_transactions(events)
        standalone = [event for event in events if not event.transaction_id]

        # Present most recent items first for UI convenience
        transactions.sort(key=lambda item: item.started_at, reverse=True)
        standalone.sort(key=lambda item: item.timestamp, reverse=True)

        if transaction_limit is not None:
            limit = max(0, transaction_limit)
            transactions = transactions[:limit]
        if event_limit >= 0:
            standalone = standalone[:event_limit]

        return HistorySnapshot(transactions=transactions, standalone_events=standalone)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_transactions(
        self,
        events: Iterable[HistoryEvent],
    ) -> List[HistoryTransaction]:
        buckets: Dict[str, HistoryTransaction] = {}
        for event in events:
            if not event.transaction_id:
                continue
            txn = buckets.get(event.transaction_id)
            if txn is None:
                txn = HistoryTransaction(
                    transaction_id=event.transaction_id,
                    started_at=event.timestamp,
                )
                buckets[event.transaction_id] = txn
            else:
                if event.timestamp < txn.started_at:
                    txn.started_at = event.timestamp
            txn.events.append(event)
            self._update_transaction_state(txn, event)

        for txn in buckets.values():
            txn.events.sort(key=lambda item: item.timestamp)
            txn.operations_summary = self._operation_summaries(txn.transaction_id)
            txn.rollback_available = self.has_operations(txn.transaction_id)

        return list(buckets.values())

    def _operation_summaries(self, transaction_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            operations = self._operations.get(transaction_id)
            if operations:
                return [op.to_summary() for op in operations]
            cached = self._operation_summaries_cache.get(transaction_id, [])
            return [dict(item) for item in cached]

    def _update_transaction_state(
        self,
        txn: HistoryTransaction,
        event: HistoryEvent,
    ) -> None:
        operation = event.operation
        if operation == "transaction.begin":
            txn.started_at = event.timestamp
            txn.status = "pending"
            return
        if operation.startswith("transaction.commit"):
            txn.status = "committed"
            txn.completed_at = event.timestamp
            return
        if operation.startswith("transaction.rollback"):
            txn.status = "rolled_back"
            txn.completed_at = event.timestamp
            return
