"""
novelWriter – Audit Logging System
===================================

File History:
Created: 2025-09-25 [James - Dev Agent]

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import os
import json
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from datetime import datetime, timedelta
from threading import Lock
import shutil

from novelwriter.api.base.security import AuditLogEntry, RiskLevel

logger = logging.getLogger(__name__)


class AuditLogManager:
    """Manages audit log storage, rotation, and retrieval."""
    
    def __init__(
        self,
        log_dir: Path,
        max_file_size_mb: int = 100,
        retention_days: int = 30,
        compress_old_logs: bool = True
    ):
        """Initialize audit log manager.
        
        Args:
            log_dir: Directory for log files
            max_file_size_mb: Maximum size per log file
            retention_days: Days to retain logs
            compress_old_logs: Whether to compress old logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.compress_old_logs = compress_old_logs
        
        self._current_file: Optional[Path] = None
        self._current_handle = None
        self._lock = Lock()
        
        self._initialize_log_file()
    
    def _initialize_log_file(self) -> None:
        """Initialize current log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_dir / f"audit_{timestamp}.jsonl"
        logger.info(f"Initialized audit log file: {self._current_file}")
    
    def write(self, entry: AuditLogEntry) -> None:
        """Write audit log entry.
        
        Args:
            entry: Log entry to write
        """
        with self._lock:
            try:
                # Check if rotation needed
                if self._should_rotate():
                    self._rotate_log()
                
                # Write entry
                with open(self._current_file, 'a') as f:
                    f.write(entry.model_dump_json() + '\n')
                
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
    
    def _should_rotate(self) -> bool:
        """Check if log rotation is needed.
        
        Returns:
            True if rotation needed
        """
        if not self._current_file or not self._current_file.exists():
            return True
        
        return self._current_file.stat().st_size >= self.max_file_size
    
    def _rotate_log(self) -> None:
        """Rotate current log file."""
        if self._current_file and self._current_file.exists():
            # Compress if enabled
            if self.compress_old_logs:
                self._compress_log(self._current_file)
            
            logger.info(f"Rotated log file: {self._current_file}")
        
        # Create new log file
        self._initialize_log_file()
    
    def _compress_log(self, log_file: Path) -> None:
        """Compress log file.
        
        Args:
            log_file: Log file to compress
        """
        try:
            compressed_file = log_file.with_suffix('.jsonl.gz')
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            log_file.unlink()
            logger.info(f"Compressed log file: {compressed_file}")
            
        except Exception as e:
            logger.error(f"Failed to compress log: {e}")
    
    def cleanup_old_logs(self) -> None:
        """Clean up old log files based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        with self._lock:
            for log_file in self.log_dir.glob("audit_*.jsonl*"):
                try:
                    # Get file creation time
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_time < cutoff_date:
                        log_file.unlink()
                        logger.info(f"Deleted old log file: {log_file}")
                        
                except Exception as e:
                    logger.error(f"Failed to delete old log: {e}")
    
    def read_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditLogEntry]:
        """Read audit logs within time range.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum entries to return
            
        Returns:
            List of audit log entries
        """
        entries = []
        
        with self._lock:
            # Get all log files
            log_files = sorted(self.log_dir.glob("audit_*.jsonl*"))
            
            for log_file in log_files:
                if len(entries) >= limit:
                    break
                
                try:
                    # Read entries from file
                    for entry_dict in self._read_file(log_file):
                        entry = AuditLogEntry(**entry_dict)
                        
                        # Apply time filters
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        
                        entries.append(entry)
                        
                        if len(entries) >= limit:
                            break
                            
                except Exception as e:
                    logger.error(f"Failed to read log file {log_file}: {e}")
        
        return entries
    
    def _read_file(self, log_file: Path) -> Iterator[Dict[str, Any]]:
        """Read entries from log file.
        
        Args:
            log_file: Log file to read
            
        Yields:
            Log entry dictionaries
        """
        if log_file.suffix == '.gz':
            # Compressed file
            with gzip.open(log_file, 'rt') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            # Regular file
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit log statistics.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_entries": 0,
            "by_operation": {},
            "by_result": {},
            "by_risk_level": {},
            "average_execution_time_ms": 0,
            "total_size_mb": 0
        }
        
        total_time = 0
        entries = self.read_logs(start_time, end_time, limit=10000)
        
        for entry in entries:
            stats["total_entries"] += 1
            
            # By operation
            stats["by_operation"][entry.operation] = \
                stats["by_operation"].get(entry.operation, 0) + 1
            
            # By result
            stats["by_result"][entry.result] = \
                stats["by_result"].get(entry.result, 0) + 1
            
            # By risk level
            stats["by_risk_level"][entry.risk_level.value] = \
                stats["by_risk_level"].get(entry.risk_level.value, 0) + 1
            
            # Execution time
            total_time += entry.execution_time_ms
        
        # Calculate averages
        if stats["total_entries"] > 0:
            stats["average_execution_time_ms"] = total_time / stats["total_entries"]
        
        # Calculate total size
        for log_file in self.log_dir.glob("audit_*.jsonl*"):
            stats["total_size_mb"] += log_file.stat().st_size / (1024 * 1024)
        
        return stats


class AuditLogViewer:
    """Viewer for audit logs with filtering and search."""
    
    def __init__(self, log_manager: AuditLogManager):
        """Initialize audit log viewer.
        
        Args:
            log_manager: Audit log manager instance
        """
        self.log_manager = log_manager
    
    def search(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        result: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Search audit logs.
        
        Args:
            query: Search query string
            start_time: Start time filter
            end_time: End time filter
            operation: Operation filter
            risk_level: Risk level filter
            result: Result filter
            limit: Maximum results
            
        Returns:
            Matching audit log entries
        """
        # Get all logs in time range
        entries = self.log_manager.read_logs(start_time, end_time, limit * 10)
        
        results = []
        query_lower = query.lower() if query else ""
        
        for entry in entries:
            # Apply filters
            if operation and entry.operation != operation:
                continue
            if risk_level and entry.risk_level != risk_level:
                continue
            if result and entry.result != result:
                continue
            
            # Apply text search
            if query:
                entry_dict = entry.model_dump()
                # Convert datetime to string for JSON serialization
                for key, value in entry_dict.items():
                    if isinstance(value, datetime):
                        entry_dict[key] = value.isoformat()
                entry_text = json.dumps(entry_dict).lower()
                if query_lower not in entry_text:
                    continue
            
            results.append(entry)
            
            if len(results) >= limit:
                break
        
        return results
    
    def export(
        self,
        output_file: Path,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Export audit logs to file.
        
        Args:
            output_file: Output file path
            format: Export format (json, csv)
            start_time: Start time filter
            end_time: End time filter
        """
        entries = self.log_manager.read_logs(start_time, end_time, limit=100000)
        
        if format == "json":
            self._export_json(entries, output_file)
        elif format == "csv":
            self._export_csv(entries, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, entries: List[AuditLogEntry], output_file: Path) -> None:
        """Export entries as JSON.
        
        Args:
            entries: Log entries
            output_file: Output file
        """
        with open(output_file, 'w') as f:
            json.dump(
                [entry.model_dump() for entry in entries],
                f,
                indent=2,
                default=str
            )
    
    def _export_csv(self, entries: List[AuditLogEntry], output_file: Path) -> None:
        """Export entries as CSV.
        
        Args:
            entries: Log entries
            output_file: Output file
        """
        import csv
        
        with open(output_file, 'w', newline='') as f:
            if not entries:
                return
            
            # Get field names from first entry
            fieldnames = list(entries[0].model_dump().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in entries:
                row = entry.model_dump()
                # Convert complex types to strings
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    elif isinstance(value, datetime):
                        row[key] = value.isoformat()
                writer.writerow(row)
