"""Tests for the AI performance tracker."""

from __future__ import annotations

from pathlib import Path

from novelwriter.ai.performance import get_tracker


def test_performance_tracker_records_and_logs(tmp_path) -> None:
    tracker = get_tracker()
    log_path = Path(tmp_path) / "debug-log.md"
    tracker.configure(enabled=True, log_path=log_path, max_samples=5)
    tracker.reset()

    with tracker.start_request("provider-test", stream=True, timeout=1.5) as span:
        span.add_output(128)
        span.finish()

    snapshot = tracker.snapshot()
    assert snapshot["entries"], "Expected at least one recorded performance sample"
    assert snapshot["entries"][0].provider_id == "provider-test"
    assert log_path.exists(), "Performance tracker should write debug log entries"
    log_text = log_path.read_text(encoding="utf-8")
    assert "provider-test" in log_text

    tracker.configure(enabled=False, log_path=log_path, max_samples=5)
    tracker.reset()
