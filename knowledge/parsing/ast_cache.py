"""
LRU Cache system for parsed ASTs with intelligent memory management.

This module provides efficient caching of parsed abstract syntax trees
with automatic eviction, file change detection, and memory monitoring.
"""

import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict

from .exceptions import CacheError


class ASTCacheEntry:
    """
    Represents a single cached AST entry with metadata.
    """
    
    def __init__(self, file_path: str, language: str, tree, parse_result: Dict[str, Any]):
        self.file_path = file_path
        self.language = language
        self.tree = tree
        self.parse_result = parse_result
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 1
        
        # File metadata for change detection
        try:
            stat = os.stat(file_path)
            self.file_size = stat.st_size
            self.file_mtime = stat.st_mtime
            self.file_hash = self._compute_file_hash(file_path)
        except OSError:
            # File doesn't exist or is inaccessible
            self.file_size = 0
            self.file_mtime = 0
            self.file_hash = ""
    
    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute SHA-256 hash of file contents for change detection.
        
        Args:
            file_path: Path to file to hash
            chunk_size: Size of chunks to read
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""
    
    def is_valid(self) -> bool:
        """
        Check if cached entry is still valid (file hasn't changed).
        
        Returns:
            True if file hasn't changed since caching
        """
        try:
            stat = os.stat(self.file_path)
            
            # Quick check: modification time
            if stat.st_mtime != self.file_mtime:
                return False
            
            # Quick check: file size
            if stat.st_size != self.file_size:
                return False
            
            # Thorough check: file hash (only if mtime/size match)
            current_hash = self._compute_file_hash(self.file_path)
            return current_hash == self.file_hash
            
        except OSError:
            # File no longer exists or is inaccessible
            return False
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of this cache entry in bytes.
        
        Returns:
            Estimated memory usage in bytes
        """
        # Base overhead for the entry object
        base_size = 200  # Approximate object overhead
        
        # String data
        base_size += len(self.file_path) * 2  # UTF-16 in Python
        base_size += len(self.language) * 2
        base_size += len(self.file_hash) * 2
        
        # Parse result dictionary (approximate)
        result_size = len(str(self.parse_result)) * 2
        base_size += result_size
        
        # Tree object is harder to estimate, use file size as approximation
        # ASTs are typically 2-5x the source size
        estimated_tree_size = self.file_size * 3
        
        return base_size + estimated_tree_size


class ASTCache:
    """
    LRU cache for parsed ASTs with intelligent memory management.
    
    Features:
    - LRU eviction when cache reaches maximum capacity
    - File change detection with hash-based verification
    - Memory usage monitoring and limits
    - Thread-safe operations
    - Cache hit/miss metrics
    """
    
    def __init__(self, max_entries: int = 100, max_memory_mb: int = 200):
        """
        Initialize AST cache.
        
        Args:
            max_entries: Maximum number of cached entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Thread-safe cache storage using OrderedDict for LRU
        self._cache: OrderedDict[str, ASTCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0
        
        # Memory tracking
        self._current_memory_bytes = 0
    
    def _generate_cache_key(self, file_path: str, language: str) -> str:
        """
        Generate unique cache key for file and language combination.
        
        Args:
            file_path: Absolute path to file
            language: Language identifier
            
        Returns:
            Cache key string
        """
        # Normalize path and create key
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        return f"{language}:{normalized_path}"
    
    def get(self, file_path: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached parse result if valid.
        
        Args:
            file_path: Path to file
            language: Language identifier
            
        Returns:
            Cached parse result or None if not found/invalid
        """
        cache_key = self._generate_cache_key(file_path, language)
        
        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if entry is still valid
            if not entry.is_valid():
                # File changed, invalidate entry
                self._invalidate_entry(cache_key)
                self._misses += 1
                self._invalidations += 1
                return None
            
            # Move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(cache_key)
            self._hits += 1
            
            return entry.parse_result.copy()  # Return copy to prevent mutation
    
    def put(self, file_path: str, language: str, tree, parse_result: Dict[str, Any]) -> bool:
        """
        Store parse result in cache.
        
        Args:
            file_path: Path to file
            language: Language identifier
            tree: Parsed tree object
            parse_result: Complete parse result dictionary
            
        Returns:
            True if successfully cached
        """
        cache_key = self._generate_cache_key(file_path, language)
        
        try:
            # Create cache entry
            entry = ASTCacheEntry(file_path, language, tree, parse_result)
            entry_memory = entry.estimate_memory_usage()
            
            with self._lock:
                # Check if we need to make space
                self._ensure_capacity(entry_memory)
                
                # Remove old entry if exists
                if cache_key in self._cache:
                    old_entry = self._cache[cache_key]
                    self._current_memory_bytes -= old_entry.estimate_memory_usage()
                    del self._cache[cache_key]
                
                # Add new entry
                self._cache[cache_key] = entry
                self._current_memory_bytes += entry_memory
                
                return True
                
        except Exception as e:
            raise CacheError(f"Failed to cache entry for {file_path}", cache_key=cache_key) from e
    
    def _ensure_capacity(self, new_entry_size: int):
        """
        Ensure cache has capacity for new entry by evicting old entries.
        
        Args:
            new_entry_size: Size of entry being added
        """
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            self._evict_lru()
        
        # Check memory limit
        while (self._current_memory_bytes + new_entry_size) > self.max_memory_bytes:
            if len(self._cache) == 0:
                # Can't make more space
                raise CacheError(
                    f"Entry size {new_entry_size} bytes exceeds maximum cache memory "
                    f"{self.max_memory_bytes} bytes"
                )
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Get LRU entry (first in OrderedDict)
        lru_key = next(iter(self._cache))
        lru_entry = self._cache[lru_key]
        
        # Update memory tracking
        self._current_memory_bytes -= lru_entry.estimate_memory_usage()
        
        # Remove entry
        del self._cache[lru_key]
        self._evictions += 1
    
    def _invalidate_entry(self, cache_key: str):
        """Remove invalid entry from cache."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self._current_memory_bytes -= entry.estimate_memory_usage()
            del self._cache[cache_key]
    
    def invalidate(self, file_path: str, language: Optional[str] = None):
        """
        Invalidate cache entries for a file.
        
        Args:
            file_path: Path to file to invalidate
            language: Specific language to invalidate, or None for all
        """
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        
        with self._lock:
            keys_to_remove = []
            
            for key in self._cache:
                if language:
                    # Specific language
                    target_key = f"{language}:{normalized_path}"
                    if key == target_key:
                        keys_to_remove.append(key)
                else:
                    # All languages for this file
                    if key.endswith(f":{normalized_path}"):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._invalidate_entry(key)
                self._invalidations += 1
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
            self._evictions += len(self._cache)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'invalidations': self._invalidations,
                'current_entries': len(self._cache),
                'max_entries': self.max_entries,
                'current_memory_mb': self._current_memory_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_usage_percent': (self._current_memory_bytes / self.max_memory_bytes) * 100
            }
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about cached entries.
        
        Returns:
            List of cache entry information
        """
        with self._lock:
            info = []
            for key, entry in self._cache.items():
                info.append({
                    'key': key,
                    'file_path': entry.file_path,
                    'language': entry.language,
                    'created_at': entry.created_at,
                    'accessed_at': entry.accessed_at,
                    'access_count': entry.access_count,
                    'file_size': entry.file_size,
                    'estimated_memory_mb': entry.estimate_memory_usage() / (1024 * 1024),
                    'age_seconds': time.time() - entry.created_at,
                    'valid': entry.is_valid()
                })
            return info
    
    def cleanup_invalid(self) -> int:
        """
        Remove all invalid entries from cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            invalid_keys = []
            
            for key, entry in self._cache.items():
                if not entry.is_valid():
                    invalid_keys.append(key)
            
            for key in invalid_keys:
                self._invalidate_entry(key)
                self._invalidations += 1
            
            return len(invalid_keys)