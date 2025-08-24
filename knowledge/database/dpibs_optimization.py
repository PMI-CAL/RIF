#!/usr/bin/env python3
"""
DPIBS Performance Optimization Layer
Issue #120: DPIBS Architecture Phase 2 - Database Schema + Performance Optimization Layer

Provides high-performance database operations for DPIBS with:
- Multi-level caching strategy
- Connection pooling optimization  
- Query performance monitoring
- Intelligent cache invalidation
- Sub-200ms context API response times
- Sub-100ms cached query performance
"""

import json
import time
import hashlib
import logging
import pickle
import zlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import threading
from functools import wraps
import psutil
import statistics

from .connection_manager import DuckDBConnectionManager
from .database_config import DatabaseConfig


@dataclass
class PerformanceMetrics:
    """Performance tracking for DPIBS operations"""
    operation: str
    duration_ms: float
    cache_hit: bool = False
    query_complexity: str = "simple"  # simple, medium, complex
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class DPIBSCacheManager:
    """
    Enhanced Multi-level cache manager for DPIBS performance optimization
    
    Cache Levels:
    1. L1 Memory Cache: Hot data, 5-minute TTL, <20ms access
    2. L2 Context Cache: Agent contexts, 30-minute TTL, <50ms access
    3. L3 Persistent Cache: Long-term storage, 24-hour TTL, <100ms access
    """
    
    def __init__(self, max_memory_entries: int = 1000, max_l3_entries: int = 10000, 
                 connection_manager: Optional[DuckDBConnectionManager] = None):
        self.max_memory_entries = max_memory_entries
        self.max_l3_entries = max_l3_entries
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # L1 Memory Cache - OrderedDict for LRU behavior
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.memory_cache_lock = threading.RLock()
        
        # L2 Context Cache - Specialized for agent contexts
        self.context_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.context_cache_lock = threading.RLock()
        
        # Cache statistics with per-level tracking
        self.cache_stats = {
            'l1': {'hits': 0, 'misses': 0, 'evictions': 0, 'invalidations': 0},
            'l2': {'hits': 0, 'misses': 0, 'evictions': 0, 'invalidations': 0},
            'l3': {'hits': 0, 'misses': 0, 'evictions': 0, 'invalidations': 0},
            'total_hits': 0,
            'total_misses': 0
        }
        
        # Initialize L3 persistent cache table if connection manager available
        self._initialize_l3_cache()
        
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key from operation and parameters"""
        key_data = f"{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _initialize_l3_cache(self) -> None:
        """Initialize L3 persistent cache table in DuckDB"""
        if not self.connection_manager:
            return
            
        try:
            with self.connection_manager.get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS l3_persistent_cache (
                        cache_key VARCHAR PRIMARY KEY,
                        operation VARCHAR NOT NULL,
                        data BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_size INTEGER NOT NULL
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_cache_operation ON l3_persistent_cache(operation)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_cache_expires ON l3_persistent_cache(expires_at)")
                
                self.logger.info("L3 persistent cache initialized")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize L3 cache: {e}")
    
    def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Enhanced get with multi-level cache hierarchy"""
        cache_key = self._generate_cache_key(operation, params)
        
        # Try L1 Memory Cache first
        result = self._get_l1(cache_key, operation)
        if result is not None:
            return result
        
        # Try L2 Context Cache for context operations
        if 'context' in operation:
            result = self._get_l2(cache_key, operation)
            if result is not None:
                # Promote to L1
                self._put_l1(cache_key, operation, result, ttl_minutes=5)
                return result
        
        # Try L3 Persistent Cache
        result = self._get_l3(cache_key, operation)
        if result is not None:
            # Promote to L1 and L2 if applicable
            self._put_l1(cache_key, operation, result, ttl_minutes=5)
            if 'context' in operation:
                self._put_l2(cache_key, operation, result, ttl_minutes=30)
            return result
        
        self.cache_stats['total_misses'] += 1
        self.logger.debug(f"Multi-level cache miss for {operation} (key: {cache_key[:8]})")
        return None
    
    def _get_l1(self, cache_key: str, operation: str) -> Optional[Any]:
        """Get from L1 memory cache"""
        with self.memory_cache_lock:
            if cache_key in self.memory_cache:
                cache_entry = self.memory_cache[cache_key]
                
                # Check TTL
                if cache_entry['expires_at'] > datetime.utcnow():
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(cache_key)
                    self.cache_stats['l1']['hits'] += 1
                    self.cache_stats['total_hits'] += 1
                    self.logger.debug(f"L1 cache hit for {operation} (key: {cache_key[:8]})")
                    return cache_entry['data']
                else:
                    # Expired entry
                    del self.memory_cache[cache_key]
                    self.cache_stats['l1']['evictions'] += 1
        
        self.cache_stats['l1']['misses'] += 1
        return None
    
    def _get_l2(self, cache_key: str, operation: str) -> Optional[Any]:
        """Get from L2 context cache"""
        with self.context_cache_lock:
            if cache_key in self.context_cache:
                cache_entry = self.context_cache[cache_key]
                
                # Check TTL
                if cache_entry['expires_at'] > datetime.utcnow():
                    # Move to end (most recently used)
                    self.context_cache.move_to_end(cache_key)
                    self.cache_stats['l2']['hits'] += 1
                    self.cache_stats['total_hits'] += 1
                    self.logger.debug(f"L2 cache hit for {operation} (key: {cache_key[:8]})")
                    return cache_entry['data']
                else:
                    # Expired entry
                    del self.context_cache[cache_key]
                    self.cache_stats['l2']['evictions'] += 1
        
        self.cache_stats['l2']['misses'] += 1
        return None
    
    def _get_l3(self, cache_key: str, operation: str) -> Optional[Any]:
        """Get from L3 persistent cache"""
        if not self.connection_manager:
            self.cache_stats['l3']['misses'] += 1
            return None
        
        try:
            with self.connection_manager.get_connection() as conn:
                result = conn.execute("""
                    SELECT data FROM l3_persistent_cache 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, [cache_key]).fetchone()
                
                if result:
                    # Update access statistics
                    conn.execute("""
                        UPDATE l3_persistent_cache 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """, [cache_key])
                    
                    # Decompress and deserialize data
                    compressed_data = result[0]
                    data = pickle.loads(zlib.decompress(compressed_data))
                    
                    self.cache_stats['l3']['hits'] += 1
                    self.cache_stats['total_hits'] += 1
                    self.logger.debug(f"L3 cache hit for {operation} (key: {cache_key[:8]})")
                    return data
                    
        except Exception as e:
            self.logger.warning(f"L3 cache retrieval error for {cache_key[:8]}: {e}")
        
        self.cache_stats['l3']['misses'] += 1
        return None
    
    def put(self, operation: str, params: Dict[str, Any], data: Any, ttl_minutes: int = 5) -> None:
        """Enhanced put with intelligent multi-level caching"""
        cache_key = self._generate_cache_key(operation, params)
        
        # Always store in L1 for immediate access
        self._put_l1(cache_key, operation, data, ttl_minutes)
        
        # Store in L2 for context operations (longer TTL)
        if 'context' in operation:
            self._put_l2(cache_key, operation, data, max(ttl_minutes, 30))
        
        # Store in L3 for expensive operations or important results
        if self._should_persist_l3(operation, data, ttl_minutes):
            self._put_l3(cache_key, operation, data, max(ttl_minutes * 12, 1440))  # At least 24 hours
    
    def _put_l1(self, cache_key: str, operation: str, data: Any, ttl_minutes: int) -> None:
        """Store in L1 memory cache"""
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        
        cache_entry = {
            'data': data,
            'expires_at': expires_at,
            'created_at': datetime.utcnow(),
            'operation': operation
        }
        
        with self.memory_cache_lock:
            # Evict oldest entries if at capacity
            while len(self.memory_cache) >= self.max_memory_entries:
                oldest_key, _ = self.memory_cache.popitem(last=False)
                self.cache_stats['l1']['evictions'] += 1
                self.logger.debug(f"Evicted L1 cache entry: {oldest_key[:8]}")
            
            self.memory_cache[cache_key] = cache_entry
            self.logger.debug(f"L1 cached {operation} result (key: {cache_key[:8]}, TTL: {ttl_minutes}min)")
    
    def _put_l2(self, cache_key: str, operation: str, data: Any, ttl_minutes: int) -> None:
        """Store in L2 context cache"""
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        
        cache_entry = {
            'data': data,
            'expires_at': expires_at,
            'created_at': datetime.utcnow(),
            'operation': operation
        }
        
        with self.context_cache_lock:
            # Evict oldest entries if at capacity (L2 is larger than L1)
            max_l2_entries = self.max_memory_entries * 2
            while len(self.context_cache) >= max_l2_entries:
                oldest_key, _ = self.context_cache.popitem(last=False)
                self.cache_stats['l2']['evictions'] += 1
                self.logger.debug(f"Evicted L2 cache entry: {oldest_key[:8]}")
            
            self.context_cache[cache_key] = cache_entry
            self.logger.debug(f"L2 cached {operation} result (key: {cache_key[:8]}, TTL: {ttl_minutes}min)")
    
    def _put_l3(self, cache_key: str, operation: str, data: Any, ttl_minutes: int) -> None:
        """Store in L3 persistent cache"""
        if not self.connection_manager:
            return
        
        try:
            # Serialize and compress data
            compressed_data = zlib.compress(pickle.dumps(data))
            data_size = len(compressed_data)
            expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
            
            with self.connection_manager.get_connection() as conn:
                # Clean up expired entries periodically
                conn.execute("DELETE FROM l3_persistent_cache WHERE expires_at < CURRENT_TIMESTAMP")
                
                # Clean up oldest entries if at capacity
                count_result = conn.execute("SELECT COUNT(*) FROM l3_persistent_cache").fetchone()
                if count_result and count_result[0] >= self.max_l3_entries:
                    conn.execute("""
                        DELETE FROM l3_persistent_cache 
                        WHERE cache_key IN (
                            SELECT cache_key FROM l3_persistent_cache 
                            ORDER BY last_accessed ASC 
                            LIMIT 100
                        )
                    """)
                    self.cache_stats['l3']['evictions'] += 100
                
                # Insert new entry
                conn.execute("""
                    INSERT OR REPLACE INTO l3_persistent_cache 
                    (cache_key, operation, data, expires_at, data_size)
                    VALUES (?, ?, ?, ?, ?)
                """, [cache_key, operation, compressed_data, expires_at, data_size])
                
                self.logger.debug(f"L3 cached {operation} result (key: {cache_key[:8]}, size: {data_size} bytes, TTL: {ttl_minutes}min)")
                
        except Exception as e:
            self.logger.warning(f"Failed to store in L3 cache: {e}")
    
    def _should_persist_l3(self, operation: str, data: Any, ttl_minutes: int) -> bool:
        """Determine if data should be persisted in L3 cache"""
        # Persist expensive operations
        expensive_operations = ['benchmarking_analysis', 'system_context_analysis', 'knowledge_integration_query']
        if any(op in operation for op in expensive_operations):
            return True
        
        # Persist large results (heuristic: complex data structures)
        if isinstance(data, (list, dict)) and len(str(data)) > 1000:
            return True
        
        # Persist long TTL items
        if ttl_minutes > 30:
            return True
        
        return False
    
    def invalidate_pattern(self, operation_pattern: str) -> int:
        """Invalidate all cache entries matching operation pattern"""
        invalidated = 0
        
        with self.memory_cache_lock:
            keys_to_remove = []
            for cache_key, cache_entry in self.memory_cache.items():
                if operation_pattern in cache_entry['operation']:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
                
        self.cache_stats['invalidations'] += invalidated
        self.logger.info(f"Invalidated {invalidated} cache entries matching '{operation_pattern}'")
        return invalidated
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive multi-level cache performance statistics"""
        total_requests = self.cache_stats['total_hits'] + self.cache_stats['total_misses']
        overall_hit_rate = (self.cache_stats['total_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate per-level hit rates
        level_stats = {}
        for level in ['l1', 'l2', 'l3']:
            level_requests = self.cache_stats[level]['hits'] + self.cache_stats[level]['misses']
            level_hit_rate = (self.cache_stats[level]['hits'] / level_requests * 100) if level_requests > 0 else 0
            level_stats[level] = {
                **self.cache_stats[level],
                'hit_rate_percent': round(level_hit_rate, 2),
                'requests': level_requests
            }
        
        # Get L3 storage stats
        l3_storage_stats = self._get_l3_storage_stats()
        
        return {
            'overall': {
                'hit_rate_percent': round(overall_hit_rate, 2),
                'total_hits': self.cache_stats['total_hits'],
                'total_misses': self.cache_stats['total_misses'],
                'total_requests': total_requests
            },
            'levels': level_stats,
            'storage': {
                'l1_size': len(self.memory_cache),
                'l1_max_size': self.max_memory_entries,
                'l2_size': len(self.context_cache),
                'l2_max_size': self.max_memory_entries * 2,
                'l3_stats': l3_storage_stats
            }
        }
    
    def _get_l3_storage_stats(self) -> Dict[str, Any]:
        """Get L3 persistent cache storage statistics"""
        if not self.connection_manager:
            return {'status': 'unavailable'}
        
        try:
            with self.connection_manager.get_connection() as conn:
                stats_result = conn.execute("""
                    SELECT 
                        COUNT(*) as entry_count,
                        SUM(data_size) as total_size_bytes,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count
                    FROM l3_persistent_cache
                """).fetchone()
                
                if stats_result:
                    return {
                        'entry_count': stats_result[0] or 0,
                        'total_size_bytes': stats_result[1] or 0,
                        'total_size_mb': round((stats_result[1] or 0) / 1024 / 1024, 2),
                        'avg_access_count': round(stats_result[2] or 0, 2),
                        'expired_count': stats_result[3] or 0,
                        'max_entries': self.max_l3_entries
                    }
                    
        except Exception as e:
            self.logger.warning(f"Failed to get L3 storage stats: {e}")
        
        return {'status': 'error'}
    
    def clear_expired_entries(self) -> Dict[str, int]:
        """Clear expired entries from all cache levels and return cleanup statistics"""
        cleanup_stats = {
            'l1_cleared': 0,
            'l2_cleared': 0, 
            'l3_cleared': 0,
            'total_cleared': 0
        }
        
        current_time = datetime.utcnow()
        
        # Clear expired L1 memory cache entries
        with self.memory_cache_lock:
            expired_keys = []
            for cache_key, cache_entry in self.memory_cache.items():
                if cache_entry['expires_at'] <= current_time:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                cleanup_stats['l1_cleared'] += 1
                
        self.cache_stats['l1']['invalidations'] += cleanup_stats['l1_cleared']
        
        # Clear expired L2 context cache entries
        with self.context_cache_lock:
            expired_keys = []
            for cache_key, cache_entry in self.context_cache.items():
                if cache_entry['expires_at'] <= current_time:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.context_cache[key]
                cleanup_stats['l2_cleared'] += 1
                
        self.cache_stats['l2']['invalidations'] += cleanup_stats['l2_cleared']
        
        # Clear expired L3 persistent cache entries
        if self.connection_manager:
            try:
                with self.connection_manager.get_connection() as conn:
                    # Count expired entries before deletion
                    count_result = conn.execute("""
                        SELECT COUNT(*) FROM l3_persistent_cache 
                        WHERE expires_at <= CURRENT_TIMESTAMP
                    """).fetchone()
                    
                    if count_result and count_result[0] > 0:
                        cleanup_stats['l3_cleared'] = count_result[0]
                        
                        # Delete expired entries
                        conn.execute("""
                            DELETE FROM l3_persistent_cache 
                            WHERE expires_at <= CURRENT_TIMESTAMP
                        """)
                        
                        self.cache_stats['l3']['invalidations'] += cleanup_stats['l3_cleared']
                        
            except Exception as e:
                self.logger.warning(f"Failed to clear expired L3 cache entries: {e}")
        
        cleanup_stats['total_cleared'] = (
            cleanup_stats['l1_cleared'] + 
            cleanup_stats['l2_cleared'] + 
            cleanup_stats['l3_cleared']
        )
        
        self.logger.info(
            f"Cache cleanup completed: L1={cleanup_stats['l1_cleared']}, "
            f"L2={cleanup_stats['l2_cleared']}, L3={cleanup_stats['l3_cleared']}, "
            f"Total={cleanup_stats['total_cleared']} expired entries cleared"
        )
        
        return cleanup_stats


class DPIBSPerformanceOptimizer:
    """
    Main performance optimization layer for DPIBS database operations
    Provides sub-200ms context APIs and sub-100ms cached queries
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.connection_manager = DuckDBConnectionManager(self.config)
        self.cache_manager = DPIBSCacheManager()
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.metrics_lock = threading.RLock()
        
        # Connection pool optimization
        self._optimize_connection_pool()
        
        # Monitoring initialized
        
        self.logger.info("DPIBS Performance Optimizer with Phase 4 enhancements initialized")
    
    def _optimize_connection_pool(self) -> None:
        """Optimize connection pool for DPIBS workload patterns"""
        try:
            # Warm up connection pool
            with self.connection_manager.get_connection() as conn:
                # Pre-warm common indexes
                conn.execute("PRAGMA memory_limit='1GB'")
                conn.execute("PRAGMA threads=4")
                
            self.logger.info("Connection pool optimized for DPIBS workload")
        except Exception as e:
            self.logger.warning(f"Failed to optimize connection pool: {e}")
    
    def _track_performance(self, operation: str, duration_ms: float, cache_hit: bool = False, 
                          query_complexity: str = "simple") -> None:
        """Track performance metrics for monitoring and optimization"""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            query_complexity=query_complexity
        )
        
        with self.metrics_lock:
            self.performance_metrics.append(metric)
            # Keep only last 1000 metrics to prevent memory growth
            if len(self.performance_metrics) > 1000:
                self.performance_metrics.pop(0)
    
    def performance_monitor(self, operation: str, cache_ttl: int = 5):
        """Decorator for automatic performance monitoring and caching"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Try cache first (skip for self parameter)
                cache_params = {'args': args[1:], 'kwargs': kwargs} if args else {'kwargs': kwargs}
                cached_result = self.cache_manager.get(operation, cache_params)
                
                if cached_result is not None:
                    duration_ms = (time.time() - start_time) * 1000
                    self._track_performance(operation, duration_ms, cache_hit=True)
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Cache result
                self.cache_manager.put(operation, cache_params, result, cache_ttl)
                
                # Track performance  
                complexity = "complex" if duration_ms > 500 else "medium" if duration_ms > 200 else "simple"
                self._track_performance(operation, duration_ms, cache_hit=False, query_complexity=complexity)
                
                return result
            return wrapper
        return decorator
    
    def get_agent_context(self, agent_type: str, context_role: str, issue_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        High-performance agent context retrieval
        Target: <200ms response time, 95% cache hit rate
        """
        query = """
            SELECT id, context_data, relevance_score, performance_metadata, accessed_count
            FROM agent_contexts 
            WHERE agent_type = ? AND context_role = ?
        """
        params = [agent_type, context_role]
        
        if issue_number is not None:
            query += " AND (issue_number = ? OR issue_number IS NULL)"
            params.append(issue_number)
        
        query += " ORDER BY relevance_score DESC, accessed_count DESC LIMIT 10"
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
            
            # Update access statistics
            if result:
                context_ids = [row[0] for row in result]
                placeholders = ','.join(['?'] * len(context_ids))
                conn.execute(f"""
                    UPDATE agent_contexts 
                    SET accessed_count = accessed_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                """, context_ids)
        
        return [dict(zip(['id', 'context_data', 'relevance_score', 'performance_metadata', 'accessed_count'], row)) for row in result]
    
    def get_system_context(self, context_type: str, context_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        System context retrieval with dependency analysis
        Target: <500ms response time for complex queries
        """
        query = """
            SELECT sc.id, sc.context_name, sc.system_snapshot, sc.confidence_level, sc.version,
                   COUNT(cd.id) as dependency_count
            FROM system_contexts sc
            LEFT JOIN component_dependencies cd ON sc.context_name = cd.source_component
            WHERE sc.context_type = ?
        """
        params = [context_type]
        
        if context_name:
            query += " AND sc.context_name = ?"
            params.append(context_name)
        
        query += """ 
            GROUP BY sc.id, sc.context_name, sc.system_snapshot, sc.confidence_level, sc.version
            ORDER BY sc.confidence_level DESC, sc.version DESC
            LIMIT 20
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
        
        return [dict(zip(['id', 'context_name', 'system_snapshot', 'confidence_level', 'version', 'dependency_count'], row)) 
                for row in result]
    
    def get_benchmarking_results(self, issue_number: Optional[int] = None, analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Benchmarking results retrieval with performance metrics
        Target: <2min complete analysis, sub-second for cached results
        """
        query = """
            SELECT br.id, br.issue_number, br.analysis_type, br.compliance_score, br.grade,
                   br.evidence_collection, br.analysis_duration, br.created_at,
                   COUNT(ki.id) as integration_count
            FROM benchmarking_results br
            LEFT JOIN knowledge_integration ki ON br.issue_number::TEXT = ki.request_data->>'issue_number'
            WHERE 1=1
        """
        params = []
        
        if issue_number is not None:
            query += " AND br.issue_number = ?"
            params.append(issue_number)
        
        if analysis_type:
            query += " AND br.analysis_type = ?"
            params.append(analysis_type)
        
        query += """
            GROUP BY br.id, br.issue_number, br.analysis_type, br.compliance_score, br.grade,
                     br.evidence_collection, br.analysis_duration, br.created_at
            ORDER BY br.created_at DESC, br.compliance_score DESC
            LIMIT 50
        """
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
        
        return [dict(zip(['id', 'issue_number', 'analysis_type', 'compliance_score', 'grade', 
                         'evidence_collection', 'analysis_duration', 'created_at', 'integration_count'], row))
                for row in result]
    
    def query_knowledge_integration(self, integration_type: str, cached_only: bool = False) -> List[Dict[str, Any]]:
        """
        Knowledge integration queries with MCP compatibility
        Target: <100ms cached queries, <1000ms live queries
        """
        query = """
            SELECT id, integration_type, response_data, integration_status, response_time_ms,
                   cached, cache_key, created_at
            FROM knowledge_integration
            WHERE integration_type = ?
        """
        params = [integration_type]
        
        if cached_only:
            query += " AND cached = TRUE"
        
        query += " ORDER BY created_at DESC LIMIT 25"
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute(query, params).fetchall()
        
        return [dict(zip(['id', 'integration_type', 'response_data', 'integration_status', 
                         'response_time_ms', 'cached', 'cache_key', 'created_at'], row))
                for row in result]
    
    def store_agent_context(self, agent_type: str, context_role: str, context_data: str, 
                           relevance_score: float, issue_number: Optional[int] = None,
                           performance_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store agent context with automatic cache invalidation"""
        context_id = None
        context_hash = hashlib.md5(context_data.encode()).hexdigest()
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO agent_contexts (agent_type, context_role, issue_number, context_data, 
                                          relevance_score, context_hash, performance_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """, [agent_type, context_role, issue_number, context_data, relevance_score, 
                  context_hash, json.dumps(performance_metadata) if performance_metadata else None]).fetchone()
            
            if result:
                context_id = str(result[0])
        
        # Invalidate related cache entries
        self.cache_manager.invalidate_pattern("agent_context_retrieval")
        
        return context_id
    
    def store_benchmarking_result(self, issue_number: int, analysis_type: str, 
                                 specification_data: Dict[str, Any], implementation_data: Dict[str, Any],
                                 compliance_score: float, grade: str, evidence_collection: Dict[str, Any],
                                 analysis_duration_ms: Optional[int] = None) -> str:
        """Store benchmarking results with performance tracking"""
        result_id = None
        
        with self.connection_manager.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO benchmarking_results (issue_number, analysis_type, specification_data,
                                                implementation_data, compliance_score, grade, 
                                                evidence_collection, analysis_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """, [issue_number, analysis_type, json.dumps(specification_data), 
                  json.dumps(implementation_data), compliance_score, grade,
                  json.dumps(evidence_collection), analysis_duration_ms]).fetchone()
            
            if result:
                result_id = str(result[0])
        
        # Invalidate benchmarking cache
        self.cache_manager.invalidate_pattern("benchmarking_analysis")
        
        return result_id
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.metrics_lock:
            if not self.performance_metrics:
                return {"message": "No performance data available"}
            
            # Calculate metrics
            recent_metrics = self.performance_metrics[-100:] if len(self.performance_metrics) >= 100 else self.performance_metrics
            
            avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
            cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
            cache_hit_rate = (cache_hits / len(recent_metrics)) * 100
            
            # Performance by operation
            operations = {}
            for metric in recent_metrics:
                if metric.operation not in operations:
                    operations[metric.operation] = {'count': 0, 'total_duration': 0, 'cache_hits': 0}
                operations[metric.operation]['count'] += 1
                operations[metric.operation]['total_duration'] += metric.duration_ms
                if metric.cache_hit:
                    operations[metric.operation]['cache_hits'] += 1
            
            # Calculate averages
            for op in operations:
                operations[op]['avg_duration_ms'] = operations[op]['total_duration'] / operations[op]['count']
                operations[op]['cache_hit_rate'] = (operations[op]['cache_hits'] / operations[op]['count']) * 100
        
        return {
            'performance_summary': {
                'avg_response_time_ms': round(avg_duration, 2),
                'cache_hit_rate_percent': round(cache_hit_rate, 2),
                'total_operations': len(recent_metrics),
                'sub_200ms_operations': sum(1 for m in recent_metrics if m.duration_ms < 200),
                'sub_100ms_operations': sum(1 for m in recent_metrics if m.duration_ms < 100)
            },
            'operations_breakdown': {op: {
                'avg_duration_ms': round(data['avg_duration_ms'], 2),
                'cache_hit_rate_percent': round(data['cache_hit_rate'], 2),
                'operation_count': data['count']
            } for op, data in operations.items()},
            'cache_statistics': self.cache_manager.get_cache_stats(),
            'performance_targets': {
                'context_apis_target_ms': 200,
                'cached_queries_target_ms': 100,
                'benchmarking_target_minutes': 2,
                'target_cache_hit_rate_percent': 90
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for DPIBS performance layer"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            # Database connection health
            with self.connection_manager.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            health_status['components']['database'] = 'healthy'
        except Exception as e:
            health_status['components']['database'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Cache health
        cache_stats = self.cache_manager.get_cache_stats()
        hit_rate = cache_stats.get('overall', {}).get('hit_rate_percent', 0)
        if hit_rate > 70:
            health_status['components']['cache'] = 'healthy'
        elif hit_rate > 50:
            health_status['components']['cache'] = 'degraded'
        else:
            health_status['components']['cache'] = 'unhealthy'
        
        # Performance health
        perf_report = self.get_performance_report()
        if 'performance_summary' in perf_report:
            avg_response = perf_report['performance_summary']['avg_response_time_ms']
            if avg_response < 200:
                health_status['components']['performance'] = 'healthy'
            elif avg_response < 500:
                health_status['components']['performance'] = 'degraded'
            else:
                health_status['components']['performance'] = 'unhealthy'
        
        return health_status