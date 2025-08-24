"""
Caching mechanisms for pattern extraction performance optimization.

This module provides intelligent caching for pattern extraction components
to improve performance and reduce redundant computations.
"""

import json
import hashlib
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from functools import wraps

# ExtractedPattern import moved to avoid circular import


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without value for serialization)."""
        data = asdict(self)
        data.pop('value', None)  # Don't serialize the actual value
        data['created_at'] = data['created_at'].isoformat()
        data['accessed_at'] = data['accessed_at'].isoformat()
        return data


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'total_bytes': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.access_order.remove(key)
                self.stats['misses'] += 1
                self.stats['size'] -= 1
                self.stats['total_bytes'] -= entry.size_bytes
                return None
            
            # Update access information
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
            # Move to end of access order (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self.lock:
            now = datetime.now()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value).encode('utf-8'))
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats['total_bytes'] -= old_entry.size_bytes
                self.access_order.remove(key)
            else:
                self.stats['size'] += 1
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.stats['total_bytes'] += size_bytes
            
            # Evict if necessary
            self._evict_if_necessary()
    
    def _evict_if_necessary(self) -> None:
        """Evict least recently used entries if cache is full."""
        while len(self.cache) > self.max_size:
            # Remove least recently used
            lru_key = self.access_order[0]
            lru_entry = self.cache[lru_key]
            
            del self.cache[lru_key]
            self.access_order.remove(lru_key)
            
            self.stats['evictions'] += 1
            self.stats['size'] -= 1
            self.stats['total_bytes'] -= lru_entry.size_bytes
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'size': 0,
                'total_bytes': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['hits'] / max(stats['hits'] + stats['misses'], 1)
            stats['average_size'] = stats['total_bytes'] / max(stats['size'], 1)
            return stats


class PatternExtractionCache:
    """Specialized cache for pattern extraction operations."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_memory_cache_size: int = 500):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Memory cache for frequently accessed items
        self.memory_cache = LRUCache(max_memory_cache_size, default_ttl=3600)  # 1 hour TTL
        
        # Disk cache for persistent storage
        self.disk_cache_dir = self.cache_dir / 'patterns'
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.config = {
            'code_analysis_ttl': 7200,      # 2 hours
            'workflow_analysis_ttl': 3600,  # 1 hour
            'decision_analysis_ttl': 3600,  # 1 hour
            'metrics_calculation_ttl': 1800, # 30 minutes
            'ast_parsing_ttl': 7200,        # 2 hours
            'similarity_search_ttl': 1800   # 30 minutes
        }
    
    def get_code_analysis_cache_key(self, code_content: str, file_path: str) -> str:
        """Generate cache key for code analysis."""
        content_hash = hashlib.sha256(code_content.encode()).hexdigest()
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()
        return f"code_analysis_{content_hash}_{path_hash}"
    
    def get_workflow_cache_key(self, issue_data: Dict[str, Any]) -> str:
        """Generate cache key for workflow analysis."""
        # Use issue number and history hash
        issue_id = issue_data.get('issue_number', 'unknown')
        history = json.dumps(issue_data.get('history', []), sort_keys=True)
        history_hash = hashlib.sha256(history.encode()).hexdigest()
        return f"workflow_{issue_id}_{history_hash}"
    
    def get_decision_cache_key(self, decision_data: Dict[str, Any]) -> str:
        """Generate cache key for decision analysis."""
        decision_json = json.dumps(decision_data, sort_keys=True)
        decision_hash = hashlib.sha256(decision_json.encode()).hexdigest()
        return f"decision_{decision_hash}"
    
    def get_metrics_cache_key(self, pattern_id: str, application_data_hash: str) -> str:
        """Generate cache key for metrics calculation."""
        return f"metrics_{pattern_id}_{application_data_hash}"
    
    def get_ast_cache_key(self, code_content: str) -> str:
        """Generate cache key for AST parsing."""
        content_hash = hashlib.sha256(code_content.encode()).hexdigest()
        return f"ast_{content_hash}"
    
    def cache_code_analysis(self, code_content: str, file_path: str, 
                           analysis_result: List[Any]) -> None:
        """Cache code analysis result."""
        cache_key = self.get_code_analysis_cache_key(code_content, file_path)
        ttl = self.config['code_analysis_ttl']
        
        # Store in memory cache
        self.memory_cache.put(cache_key, analysis_result, ttl)
        
        # Store in disk cache for persistence
        self._store_to_disk(cache_key, analysis_result, ttl)
    
    def get_cached_code_analysis(self, code_content: str, file_path: str) -> Optional[List[Any]]:
        """Get cached code analysis result."""
        cache_key = self.get_code_analysis_cache_key(code_content, file_path)
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Try disk cache
        result = self._load_from_disk(cache_key)
        if result is not None:
            # Store in memory cache for faster future access
            self.memory_cache.put(cache_key, result, self.config['code_analysis_ttl'])
            return result
        
        return None
    
    def cache_workflow_analysis(self, issue_data: Dict[str, Any],
                               analysis_result: List[Any]) -> None:
        """Cache workflow analysis result."""
        cache_key = self.get_workflow_cache_key(issue_data)
        ttl = self.config['workflow_analysis_ttl']
        
        self.memory_cache.put(cache_key, analysis_result, ttl)
        self._store_to_disk(cache_key, analysis_result, ttl)
    
    def get_cached_workflow_analysis(self, issue_data: Dict[str, Any]) -> Optional[List[Any]]:
        """Get cached workflow analysis result."""
        cache_key = self.get_workflow_cache_key(issue_data)
        
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        result = self._load_from_disk(cache_key)
        if result is not None:
            self.memory_cache.put(cache_key, result, self.config['workflow_analysis_ttl'])
            return result
        
        return None
    
    def cache_decision_analysis(self, decision_data: Dict[str, Any],
                               analysis_result: List[Any]) -> None:
        """Cache decision analysis result."""
        cache_key = self.get_decision_cache_key(decision_data)
        ttl = self.config['decision_analysis_ttl']
        
        self.memory_cache.put(cache_key, analysis_result, ttl)
        self._store_to_disk(cache_key, analysis_result, ttl)
    
    def get_cached_decision_analysis(self, decision_data: Dict[str, Any]) -> Optional[List[Any]]:
        """Get cached decision analysis result."""
        cache_key = self.get_decision_cache_key(decision_data)
        
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        result = self._load_from_disk(cache_key)
        if result is not None:
            self.memory_cache.put(cache_key, result, self.config['decision_analysis_ttl'])
            return result
        
        return None
    
    def cache_ast_parsing(self, code_content: str, ast_result: Any) -> None:
        """Cache AST parsing result."""
        cache_key = self.get_ast_cache_key(code_content)
        ttl = self.config['ast_parsing_ttl']
        
        self.memory_cache.put(cache_key, ast_result, ttl)
        # AST results are typically large, so only cache in memory
    
    def get_cached_ast_parsing(self, code_content: str) -> Optional[Any]:
        """Get cached AST parsing result."""
        cache_key = self.get_ast_cache_key(code_content)
        return self.memory_cache.get(cache_key)
    
    def _store_to_disk(self, cache_key: str, data: Any, ttl: int) -> None:
        """Store data to disk cache."""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
            
            cache_data = {
                'data': data,
                'created_at': datetime.now().isoformat(),
                'ttl_seconds': ttl
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to store cache entry to disk: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check expiration
            created_at = datetime.fromisoformat(cache_data['created_at'])
            ttl_seconds = cache_data['ttl_seconds']
            
            if (datetime.now() - created_at).total_seconds() > ttl_seconds:
                # Cache entry expired, remove file
                cache_file.unlink()
                return None
            
            return cache_data['data']
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache entry from disk: {e}")
            return None
    
    def clear_memory_cache(self) -> None:
        """Clear memory cache."""
        self.memory_cache.clear()
    
    def clear_disk_cache(self) -> None:
        """Clear disk cache."""
        try:
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to clear disk cache: {e}")
    
    def clear_all(self) -> None:
        """Clear both memory and disk caches."""
        self.clear_memory_cache()
        self.clear_disk_cache()
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from disk cache."""
        cleaned_count = 0
        try:
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    created_at = datetime.fromisoformat(cache_data['created_at'])
                    ttl_seconds = cache_data['ttl_seconds']
                    
                    if (datetime.now() - created_at).total_seconds() > ttl_seconds:
                        cache_file.unlink()
                        cleaned_count += 1
                        
                except Exception:
                    # If we can't read the file, remove it
                    cache_file.unlink()
                    cleaned_count += 1
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup expired cache entries: {e}")
        
        return cleaned_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        # Get disk cache stats
        disk_files = 0
        disk_size = 0
        try:
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                disk_files += 1
                disk_size += cache_file.stat().st_size
        except Exception:
            pass
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': {
                'files': disk_files,
                'total_size_bytes': disk_size,
                'average_size_bytes': disk_size / max(disk_files, 1)
            },
            'total_cache_size_mb': (memory_stats['total_bytes'] + disk_size) / (1024 * 1024)
        }


def cached_pattern_extraction(cache_type: str, ttl: Optional[int] = None):
    """Decorator for caching pattern extraction methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if instance has cache
            if not hasattr(self, '_cache'):
                return func(self, *args, **kwargs)
            
            cache = self._cache
            
            # Generate cache key based on function arguments
            cache_key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            cache_key_json = json.dumps(cache_key_data, sort_keys=True)
            cache_key = f"{cache_type}_{hashlib.sha256(cache_key_json.encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache.memory_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            cache.memory_cache.put(cache_key, result, ttl or 3600)
            
            return result
        return wrapper
    return decorator


class BatchProcessor:
    """Batch processor for efficient pattern extraction."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_code_files_batch(self, file_contents: List[Tuple[str, str]], 
                                extractor, cache: PatternExtractionCache) -> List[Any]:
        """Process multiple code files efficiently."""
        all_patterns = []
        
        # Group files into batches
        for i in range(0, len(file_contents), self.batch_size):
            batch = file_contents[i:i + self.batch_size]
            batch_patterns = self._process_code_batch(batch, extractor, cache)
            all_patterns.extend(batch_patterns)
        
        return all_patterns
    
    def _process_code_batch(self, batch: List[Tuple[str, str]], 
                           extractor, cache: PatternExtractionCache) -> List[Any]:
        """Process a single batch of code files."""
        patterns = []
        
        for file_path, content in batch:
            try:
                # Check cache first
                cached_patterns = cache.get_cached_code_analysis(content, file_path)
                if cached_patterns is not None:
                    patterns.extend(cached_patterns)
                    continue
                
                # Analyze code
                file_patterns = extractor._analyze_code_content(content, file_path, f"batch_{int(time.time())}")
                
                # Cache results
                cache.cache_code_analysis(content, file_path, file_patterns)
                
                patterns.extend(file_patterns)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path} in batch: {e}")
                continue
        
        return patterns
    
    def process_decision_batch(self, decisions: List[Dict[str, Any]], 
                              extractor, cache: PatternExtractionCache) -> List[Any]:
        """Process multiple decisions efficiently."""
        all_patterns = []
        
        for i, decision in enumerate(decisions):
            try:
                # Check cache first
                cached_patterns = cache.get_cached_decision_analysis(decision)
                if cached_patterns is not None:
                    all_patterns.extend(cached_patterns)
                    continue
                
                # Analyze decision
                decision_pattern = extractor._parse_decision_record(decision, f"batch_decision_{i}")
                if decision_pattern:
                    # Convert to ExtractedPattern format
                    extracted_pattern = self._convert_decision_to_extracted_pattern(decision_pattern, decision)
                    decision_patterns = [extracted_pattern] if extracted_pattern else []
                    
                    # Cache results
                    cache.cache_decision_analysis(decision, decision_patterns)
                    all_patterns.extend(decision_patterns)
                
            except Exception as e:
                self.logger.error(f"Error processing decision in batch: {e}")
                continue
        
        return all_patterns
    
    def _convert_decision_to_extracted_pattern(self, decision_pattern, decision_data: Dict[str, Any]) -> Optional[Any]:
        """Convert DecisionPattern to ExtractedPattern."""
        try:
            from .discovery_engine import PatternSignature, ExtractedPattern
            
            pattern = ExtractedPattern(
                title=decision_pattern.title,
                description=f"Decision pattern: {decision_pattern.title}",
                pattern_type='decision',
                source='batch_processing',
                content=decision_pattern.to_dict(),
                context={
                    'decision_type': decision_pattern.decision_type,
                    'stakeholder_count': len(decision_pattern.context.stakeholders),
                    'alternative_count': len(decision_pattern.alternatives),
                    'risk_level': decision_pattern.context.risk_level
                },
                signature=PatternSignature.from_pattern({
                    'title': decision_pattern.title,
                    'description': decision_pattern.outcome.rationale,
                    'complexity': decision_pattern.context.risk_level,
                    'domain': 'decision_making'
                }),
                extraction_method='batch_processing',
                confidence=0.8,
                created_at=datetime.now()
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error converting decision pattern: {e}")
            return None


# Global cache instance
_global_cache: Optional[PatternExtractionCache] = None


def get_global_cache() -> PatternExtractionCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PatternExtractionCache()
    return _global_cache


def initialize_global_cache(cache_dir: Optional[str] = None, 
                          max_memory_cache_size: int = 500) -> None:
    """Initialize global cache with custom settings."""
    global _global_cache
    _global_cache = PatternExtractionCache(cache_dir, max_memory_cache_size)