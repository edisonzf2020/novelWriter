"""
novelWriter – External MCP Cache Management
============================================

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

import time
import json
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio
from threading import Lock

from novelwriter.api.external_mcp.exceptions import ExternalMCPCacheError

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        return time.time() - self.timestamp > self.ttl_seconds
    
    def access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class ExternalMCPCache:
    """Cache manager for external MCP tool results.
    
    Implements LRU cache with TTL support and intelligent eviction.
    """
    
    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 300,
        max_entries: int = 10000
    ):
        """Initialize cache manager.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl_seconds: Default time-to-live for entries
            max_entries: Maximum number of cache entries
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._current_size_bytes = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = False
        
        logger.debug(
            f"ExternalMCPCache initialized: max_size={max_size_mb}MB, "
            f"default_ttl={default_ttl_seconds}s, max_entries={max_entries}"
        )
    
    def generate_key(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        connection_id: str
    ) -> str:
        """Generate cache key for tool call.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            connection_id: Connection identifier
            
        Returns:
            Cache key string
        """
        # Create deterministic key from inputs
        key_data = {
            "tool": tool_name,
            "params": parameters,
            "connection": connection_id
        }
        
        # Sort keys for consistency
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
        
        return f"{connection_id}:{tool_name}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._evict_entry(key)
                self._misses += 1
                return None
            
            # Update access info and move to end (most recent)
            entry.access()
            self._cache.move_to_end(key)
            
            self._hits += 1
            logger.debug(f"Cache hit for key: {key}")
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        
        # Estimate size
        try:
            value_json = json.dumps(value)
            size_bytes = len(value_json.encode())
        except (TypeError, ValueError):
            # If not JSON serializable, estimate size
            size_bytes = len(str(value).encode())
        
        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                self._evict_entry(key)
            
            # Check if we need to evict entries
            self._ensure_capacity(size_bytes)
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            
            logger.debug(
                f"Cached key: {key}, size: {size_bytes} bytes, ttl: {ttl}s"
            )
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._evict_entry(key)
                logger.debug(f"Invalidated cache key: {key}")
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all entries matching pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            Number of entries invalidated
        """
        import fnmatch
        
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in keys_to_remove:
                self._evict_entry(key)
            
            if keys_to_remove:
                logger.debug(
                    f"Invalidated {len(keys_to_remove)} entries matching: {pattern}"
                )
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self._evictions = 0
            logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "entries": len(self._cache),
                "size_bytes": self._current_size_bytes,
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "max_entries": self._max_entries
            }
    
    def _evict_entry(self, key: str) -> None:
        """Evict single cache entry.
        
        Args:
            key: Cache key to evict
        """
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]
            self._evictions += 1
    
    def _ensure_capacity(self, required_bytes: int) -> None:
        """Ensure cache has capacity for new entry.
        
        Args:
            required_bytes: Bytes needed for new entry
        """
        # Check entry count limit
        while len(self._cache) >= self._max_entries:
            # Remove least recently used (first item)
            if self._cache:
                oldest_key = next(iter(self._cache))
                self._evict_entry(oldest_key)
                logger.debug(f"Evicted LRU entry: {oldest_key}")
        
        # Check size limit
        while self._current_size_bytes + required_bytes > self._max_size_bytes:
            if not self._cache:
                break
            
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            self._evict_entry(oldest_key)
            logger.debug(f"Evicted entry for size limit: {oldest_key}")
    
    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._stop_cleanup = False
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Cache cleanup task started")
    
    async def stop_cleanup(self) -> None:
        """Stop background cleanup task."""
        self._stop_cleanup = True
        if self._cleanup_task:
            await self._cleanup_task
            self._cleanup_task = None
            logger.debug("Cache cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean expired entries."""
        while not self._stop_cleanup:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._evict_entry(key)
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                        
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")


class CacheManager:
    """Global cache manager for external MCP tools."""
    
    _instance: Optional[CacheManager] = None
    _lock = Lock()
    
    def __new__(cls) -> CacheManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache manager (only once)."""
        if not self._initialized:
            self._caches: Dict[str, ExternalMCPCache] = {}
            self._default_cache = ExternalMCPCache()
            self._initialized = True
            logger.debug("CacheManager initialized")
    
    def get_cache(self, namespace: str = "default") -> ExternalMCPCache:
        """Get cache for namespace.
        
        Args:
            namespace: Cache namespace
            
        Returns:
            Cache instance
        """
        if namespace == "default":
            return self._default_cache
        
        if namespace not in self._caches:
            self._caches[namespace] = ExternalMCPCache()
            logger.debug(f"Created cache for namespace: {namespace}")
        
        return self._caches[namespace]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self._default_cache.clear()
        for cache in self._caches.values():
            cache.clear()
        logger.info("All caches cleared")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches.
        
        Returns:
            Dictionary with global cache statistics
        """
        stats = {
            "default": self._default_cache.get_statistics(),
            "namespaces": {}
        }
        
        for namespace, cache in self._caches.items():
            stats["namespaces"][namespace] = cache.get_statistics()
        
        # Calculate totals
        total_entries = stats["default"]["entries"]
        total_size_mb = stats["default"]["size_mb"]
        total_hits = stats["default"]["hits"]
        total_misses = stats["default"]["misses"]
        
        for ns_stats in stats["namespaces"].values():
            total_entries += ns_stats["entries"]
            total_size_mb += ns_stats["size_mb"]
            total_hits += ns_stats["hits"]
            total_misses += ns_stats["misses"]
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        stats["global"] = {
            "total_entries": total_entries,
            "total_size_mb": total_size_mb,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "global_hit_rate": global_hit_rate
        }
        
        return stats
