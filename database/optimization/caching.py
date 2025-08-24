#!/usr/bin/env python3
"""
Database Caching Layer Implementation
Issue #138: Database Schema + Performance Optimization Layer

Implements intelligent multi-level caching for <100ms cached query performance
with cache invalidation, connection pooling, and performance monitoring.
"""

import os
import time
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Database connections
import duckdb
import sqlite3

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1 = "L1"  # In-memory, fastest access
    L2 = "L2"  # Local SQLite, fast access  
    L3 = "L3"  # DuckDB, persistent storage

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    cache_level: CacheLevel
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.expires_at
    
    def is_stale(self, staleness_threshold: int = 300) -> bool:
        """Check if cache entry is stale (in seconds)"""
        if self.last_accessed is None:
            return True
        return (datetime.now() - self.last_accessed).total_seconds() > staleness_threshold

class DatabaseCachingLayer:
    """
    Multi-level intelligent caching system for DPIBS database operations.
    Provides L1 (memory), L2 (SQLite), and L3 (DuckDB) caching with automatic
    invalidation and performance optimization.
    """
    
    def __init__(self, 
                 db_path: str = "/Users/cal/DEV/RIF/systems/context",
                 cache_db_path: str = "/Users/cal/DEV/RIF/systems/context/context_cache.db",
                 max_memory_cache_size: int = 1000,
                 default_ttl: int = 300,
                 enable_performance_monitoring: bool = True):
        
        self.db_path = db_path
        self.cache_db_path = cache_db_path
        self.max_memory_cache_size = max_memory_cache_size
        self.default_ttl = default_ttl
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # L1 Cache: In-memory dictionary
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_access_order: List[str] = []  # LRU tracking
        self.l1_lock = threading.RLock()
        
        # Performance metrics
        self.cache_metrics = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0, 
            "l3_hits": 0, "l3_misses": 0,
            "total_requests": 0,
            "cache_evictions": 0,
            "cache_invalidations": 0
        }
        
        # Connection pools
        self.duckdb_pool = []
        self.sqlite_pool = []
        self.pool_lock = threading.Lock()
        
        # Initialize caching infrastructure
        self._initialize_cache_infrastructure()
        self._initialize_connection_pools()
        
    def _initialize_cache_infrastructure(self):
        """Initialize caching database structures"""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        
        # Initialize L2 cache (SQLite)
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    cache_level TEXT NOT NULL,
                    cache_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    size_bytes INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Performance indexes for L2 cache
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_level_access ON cache_entries(cache_level, last_accessed DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key_level ON cache_entries(cache_key, cache_level)")
            
            # Performance monitoring table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_performance (
                    metric_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    cache_level TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    hit BOOLEAN NOT NULL,
                    data_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        # Initialize L3 cache (DuckDB) - Create table for caching layer use
        duckdb_path = os.path.join(self.db_path, "context_intelligence.duckdb")
        os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
        
        with duckdb.connect(duckdb_path) as conn:
            # Create separate cache table for caching layer (not conflicting with DPIBS context_cache)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_level TEXT NOT NULL,
                    cached_result TEXT NOT NULL,
                    data_size INTEGER NOT NULL,
                    expiry_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Performance indexes for L3 cache
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_cache_expires ON context_cache(expiry_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_cache_access ON context_cache(last_accessed DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_cache_level ON context_cache(cache_level, expiry_time)")
            
        logger.info("Cache infrastructure initialized (L2=SQLite, L3=DuckDB)")
    
    def _initialize_connection_pools(self, pool_size: int = 5):
        """Initialize database connection pools for performance"""
        try:
            # DuckDB connection pool
            for _ in range(pool_size):
                conn = duckdb.connect(os.path.join(self.db_path, "context_intelligence.duckdb"))
                self.duckdb_pool.append(conn)
            
            # SQLite connection pool  
            for _ in range(pool_size):
                conn = sqlite3.connect(self.cache_db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self.sqlite_pool.append(conn)
                
            logger.info(f"Connection pools initialized with {pool_size} connections each")
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
    
    def get_connection(self, db_type: str = "sqlite") -> Union[sqlite3.Connection, duckdb.DuckDBPyConnection]:
        """Get database connection from pool"""
        with self.pool_lock:
            if db_type == "duckdb" and self.duckdb_pool:
                return self.duckdb_pool.pop()
            elif db_type == "sqlite" and self.sqlite_pool:
                return self.sqlite_pool.pop()
            else:
                # Create new connection if pool empty
                if db_type == "duckdb":
                    return duckdb.connect(os.path.join(self.db_path, "context_intelligence.duckdb"))
                else:
                    conn = sqlite3.connect(self.cache_db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    return conn
    
    def return_connection(self, conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], db_type: str = "sqlite"):
        """Return connection to pool"""
        with self.pool_lock:
            if db_type == "duckdb" and len(self.duckdb_pool) < 10:
                self.duckdb_pool.append(conn)
            elif db_type == "sqlite" and len(self.sqlite_pool) < 10:
                self.sqlite_pool.append(conn)
            else:
                # Close connection if pool full
                conn.close()
    
    async def get_cached_data(self, cache_key: str, cache_levels: List[CacheLevel] = None) -> Optional[Any]:
        """
        Get data from cache with multi-level lookup.
        Returns data if found, None if not cached or expired.
        Target: <100ms for cached queries
        """
        start_time = time.time()
        
        if cache_levels is None:
            cache_levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        
        self.cache_metrics["total_requests"] += 1
        
        try:
            # L1 Cache lookup (fastest)
            if CacheLevel.L1 in cache_levels:
                l1_result = await self._get_l1_cache(cache_key)
                if l1_result is not None:
                    self.cache_metrics["l1_hits"] += 1
                    self._record_performance_metric("get", CacheLevel.L1, start_time, True, len(str(l1_result)))
                    return l1_result
                else:
                    self.cache_metrics["l1_misses"] += 1
            
            # L2 Cache lookup (fast)
            if CacheLevel.L2 in cache_levels:
                l2_result = await self._get_l2_cache(cache_key)
                if l2_result is not None:
                    self.cache_metrics["l2_hits"] += 1
                    # Promote to L1 for faster future access
                    await self._set_l1_cache(cache_key, l2_result, self.default_ttl)
                    self._record_performance_metric("get", CacheLevel.L2, start_time, True, len(str(l2_result)))
                    return l2_result
                else:
                    self.cache_metrics["l2_misses"] += 1
            
            # L3 Cache lookup (persistent)
            if CacheLevel.L3 in cache_levels:
                l3_result = await self._get_l3_cache(cache_key)
                if l3_result is not None:
                    self.cache_metrics["l3_hits"] += 1
                    # Promote to L2 and L1
                    await self._set_l2_cache(cache_key, l3_result, self.default_ttl)
                    await self._set_l1_cache(cache_key, l3_result, self.default_ttl)
                    self._record_performance_metric("get", CacheLevel.L3, start_time, True, len(str(l3_result)))
                    return l3_result
                else:
                    self.cache_metrics["l3_misses"] += 1
            
            # Cache miss
            self._record_performance_metric("get", CacheLevel.L1, start_time, False, 0)
            return None
            
        except Exception as e:
            logger.error(f"Cache get operation failed for key {cache_key}: {e}")
            return None
    
    async def set_cached_data(self, 
                            cache_key: str, 
                            data: Any, 
                            ttl: int = None,
                            cache_levels: List[CacheLevel] = None) -> bool:
        """
        Set data in cache across specified levels.
        Returns True if successfully cached.
        """
        start_time = time.time()
        
        if ttl is None:
            ttl = self.default_ttl
            
        if cache_levels is None:
            cache_levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        
        success = True
        data_size = len(str(data))
        
        try:
            # Set in each cache level
            for cache_level in cache_levels:
                if cache_level == CacheLevel.L1:
                    success &= await self._set_l1_cache(cache_key, data, ttl)
                elif cache_level == CacheLevel.L2:
                    success &= await self._set_l2_cache(cache_key, data, ttl)
                elif cache_level == CacheLevel.L3:
                    success &= await self._set_l3_cache(cache_key, data, ttl)
            
            if success:
                self._record_performance_metric("set", CacheLevel.L1, start_time, True, data_size)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set operation failed for key {cache_key}: {e}")
            return False
    
    async def invalidate_cache(self, 
                             cache_key: str = None,
                             pattern: str = None,
                             cache_levels: List[CacheLevel] = None) -> int:
        """
        Invalidate cache entries by key or pattern.
        Returns number of entries invalidated.
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        
        invalidated_count = 0
        self.cache_metrics["cache_invalidations"] += 1
        
        try:
            for cache_level in cache_levels:
                if cache_level == CacheLevel.L1:
                    invalidated_count += await self._invalidate_l1_cache(cache_key, pattern)
                elif cache_level == CacheLevel.L2:
                    invalidated_count += await self._invalidate_l2_cache(cache_key, pattern)
                elif cache_level == CacheLevel.L3:
                    invalidated_count += await self._invalidate_l3_cache(cache_key, pattern)
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0
    
    # L1 Cache Operations (In-Memory)
    
    async def _get_l1_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from L1 (memory) cache"""
        with self.l1_lock:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                if entry.is_expired():
                    # Remove expired entry
                    del self.l1_cache[cache_key]
                    if cache_key in self.l1_access_order:
                        self.l1_access_order.remove(cache_key)
                    return None
                
                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Update LRU order
                if cache_key in self.l1_access_order:
                    self.l1_access_order.remove(cache_key)
                self.l1_access_order.append(cache_key)
                
                return entry.value
            
            return None
    
    async def _set_l1_cache(self, cache_key: str, data: Any, ttl: int) -> bool:
        """Set data in L1 (memory) cache"""
        with self.l1_lock:
            # Check if we need to evict entries
            if len(self.l1_cache) >= self.max_memory_cache_size:
                await self._evict_l1_cache_entries()
            
            expires_at = datetime.now() + timedelta(seconds=ttl)
            entry = CacheEntry(
                key=cache_key,
                value=data,
                cache_level=CacheLevel.L1,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_accessed=datetime.now(),
                size_bytes=len(str(data)),
                metadata={}
            )
            
            self.l1_cache[cache_key] = entry
            self.l1_access_order.append(cache_key)
            
            return True
    
    async def _evict_l1_cache_entries(self, target_size: int = None):
        """Evict L1 cache entries using LRU policy"""
        if target_size is None:
            target_size = int(self.max_memory_cache_size * 0.8)  # Evict to 80% capacity
        
        entries_to_evict = len(self.l1_cache) - target_size
        if entries_to_evict <= 0:
            return
        
        # Evict least recently used entries
        for _ in range(entries_to_evict):
            if self.l1_access_order:
                lru_key = self.l1_access_order.pop(0)
                if lru_key in self.l1_cache:
                    del self.l1_cache[lru_key]
                    self.cache_metrics["cache_evictions"] += 1
    
    async def _invalidate_l1_cache(self, cache_key: str = None, pattern: str = None) -> int:
        """Invalidate L1 cache entries"""
        invalidated = 0
        with self.l1_lock:
            if cache_key:
                if cache_key in self.l1_cache:
                    del self.l1_cache[cache_key]
                    if cache_key in self.l1_access_order:
                        self.l1_access_order.remove(cache_key)
                    invalidated = 1
            elif pattern:
                keys_to_remove = [k for k in self.l1_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.l1_cache[key]
                    if key in self.l1_access_order:
                        self.l1_access_order.remove(key)
                invalidated = len(keys_to_remove)
        
        return invalidated
    
    # L2 Cache Operations (SQLite)
    
    async def _get_l2_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from L2 (SQLite) cache"""
        conn = self.get_connection("sqlite")
        try:
            cursor = conn.execute("""
                SELECT cache_value, expires_at FROM cache_entries 
                WHERE cache_key = ? AND cache_level = 'L2' 
                AND expires_at > CURRENT_TIMESTAMP
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                # Update access tracking
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ? AND cache_level = 'L2'
                """, (cache_key,))
                conn.commit()
                
                return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"L2 cache get failed for key {cache_key}: {e}")
            return None
        finally:
            self.return_connection(conn, "sqlite")
    
    async def _set_l2_cache(self, cache_key: str, data: Any, ttl: int) -> bool:
        """Set data in L2 (SQLite) cache"""
        conn = self.get_connection("sqlite")
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            serialized_data = json.dumps(data)
            
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (cache_key, cache_level, cache_value, expires_at, size_bytes)
                VALUES (?, 'L2', ?, ?, ?)
            """, (cache_key, serialized_data, expires_at, len(serialized_data)))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"L2 cache set failed for key {cache_key}: {e}")
            return False
        finally:
            self.return_connection(conn, "sqlite")
    
    async def _invalidate_l2_cache(self, cache_key: str = None, pattern: str = None) -> int:
        """Invalidate L2 cache entries"""
        conn = self.get_connection("sqlite")
        try:
            if cache_key:
                cursor = conn.execute("DELETE FROM cache_entries WHERE cache_key = ? AND cache_level = 'L2'", (cache_key,))
            elif pattern:
                cursor = conn.execute("DELETE FROM cache_entries WHERE cache_key LIKE ? AND cache_level = 'L2'", (f"%{pattern}%",))
            else:
                cursor = conn.execute("DELETE FROM cache_entries WHERE cache_level = 'L2'")
            
            conn.commit()
            return cursor.rowcount
            
        except Exception as e:
            logger.error(f"L2 cache invalidation failed: {e}")
            return 0
        finally:
            self.return_connection(conn, "sqlite")
    
    # L3 Cache Operations (DuckDB)
    
    async def _get_l3_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from L3 (DuckDB) cache"""
        conn = self.get_connection("duckdb")
        try:
            result = conn.execute("""
                SELECT cached_result FROM context_cache 
                WHERE cache_key = ? AND expiry_time > CURRENT_TIMESTAMP
            """, (cache_key,)).fetchone()
            
            if result:
                # Update access tracking
                conn.execute("""
                    UPDATE context_cache 
                    SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                
                return json.loads(result[0]) if isinstance(result[0], str) else result[0]
            
            return None
            
        except Exception as e:
            logger.error(f"L3 cache get failed for key {cache_key}: {e}")
            return None
        finally:
            self.return_connection(conn, "duckdb")
    
    async def _set_l3_cache(self, cache_key: str, data: Any, ttl: int) -> bool:
        """Set data in L3 (DuckDB) cache"""
        conn = self.get_connection("duckdb")
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            serialized_data = json.dumps(data)
            
            conn.execute("""
                INSERT OR REPLACE INTO context_cache 
                (cache_key, cache_level, cached_result, data_size, expiry_time)
                VALUES (?, 'L3', ?, ?, ?)
            """, (cache_key, serialized_data, len(serialized_data), expires_at))
            
            return True
            
        except Exception as e:
            logger.error(f"L3 cache set failed for key {cache_key}: {e}")
            return False
        finally:
            self.return_connection(conn, "duckdb")
    
    async def _invalidate_l3_cache(self, cache_key: str = None, pattern: str = None) -> int:
        """Invalidate L3 cache entries"""
        conn = self.get_connection("duckdb")
        try:
            if cache_key:
                result = conn.execute("DELETE FROM context_cache WHERE cache_key = ?", (cache_key,))
            elif pattern:
                result = conn.execute("DELETE FROM context_cache WHERE cache_key LIKE ?", (f"%{pattern}%",))
            else:
                result = conn.execute("DELETE FROM context_cache")
            
            return result.fetchone()[0] if result else 0
            
        except Exception as e:
            logger.error(f"L3 cache invalidation failed: {e}")
            return 0
        finally:
            self.return_connection(conn, "duckdb")
    
    # Performance monitoring
    
    def _record_performance_metric(self, operation: str, cache_level: CacheLevel, start_time: float, hit: bool, data_size: int):
        """Record performance metric for monitoring"""
        if not self.enable_performance_monitoring:
            return
        
        duration_ms = (time.time() - start_time) * 1000
        
        conn = self.get_connection("sqlite")
        try:
            conn.execute("""
                INSERT INTO cache_performance 
                (metric_id, operation_type, cache_level, duration_ms, hit, data_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"{operation}_{cache_level.value}_{int(time.time() * 1000)}",
                operation,
                cache_level.value,
                duration_ms,
                hit,
                data_size
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
        finally:
            self.return_connection(conn, "sqlite")
    
    async def get_performance_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive cache performance analytics"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        conn = self.get_connection("sqlite")
        try:
            # Get performance metrics
            cursor = conn.execute("""
                SELECT 
                    cache_level,
                    operation_type,
                    COUNT(*) as total_operations,
                    AVG(duration_ms) as avg_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    SUM(CASE WHEN hit THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN NOT hit THEN 1 ELSE 0 END) as misses,
                    AVG(data_size) as avg_data_size
                FROM cache_performance 
                WHERE created_at > ?
                GROUP BY cache_level, operation_type
                ORDER BY cache_level, operation_type
            """, (cutoff_time,))
            
            performance_data = []
            for row in cursor.fetchall():
                performance_data.append({
                    "cache_level": row[0],
                    "operation": row[1],
                    "total_operations": row[2],
                    "avg_duration_ms": round(row[3], 2) if row[3] else 0,
                    "max_duration_ms": round(row[4], 2) if row[4] else 0,
                    "min_duration_ms": round(row[5], 2) if row[5] else 0,
                    "hits": row[6],
                    "misses": row[7],
                    "hit_rate": row[6] / (row[6] + row[7]) if (row[6] + row[7]) > 0 else 0,
                    "avg_data_size_bytes": round(row[8], 0) if row[8] else 0
                })
            
            # Overall cache statistics
            total_requests = self.cache_metrics["total_requests"]
            total_hits = (self.cache_metrics["l1_hits"] + 
                         self.cache_metrics["l2_hits"] + 
                         self.cache_metrics["l3_hits"])
            
            return {
                "analysis_period_hours": hours_back,
                "overall_metrics": dict(self.cache_metrics),
                "overall_hit_rate": total_hits / total_requests if total_requests > 0 else 0,
                "performance_by_level": performance_data,
                "l1_cache_size": len(self.l1_cache),
                "connection_pools": {
                    "sqlite_pool_size": len(self.sqlite_pool),
                    "duckdb_pool_size": len(self.duckdb_pool)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e)}
        finally:
            self.return_connection(conn, "sqlite")
    
    async def cleanup_expired_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries across all levels"""
        cleanup_results = {"l1_cleaned": 0, "l2_cleaned": 0, "l3_cleaned": 0}
        
        # Clean L1 cache
        with self.l1_lock:
            expired_keys = [k for k, v in self.l1_cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.l1_cache[key]
                if key in self.l1_access_order:
                    self.l1_access_order.remove(key)
            cleanup_results["l1_cleaned"] = len(expired_keys)
        
        # Clean L2 cache
        conn = self.get_connection("sqlite")
        try:
            cursor = conn.execute("DELETE FROM cache_entries WHERE expires_at < CURRENT_TIMESTAMP")
            conn.commit()
            cleanup_results["l2_cleaned"] = cursor.rowcount
        except Exception as e:
            logger.error(f"L2 cache cleanup failed: {e}")
        finally:
            self.return_connection(conn, "sqlite")
        
        # Clean L3 cache
        conn = self.get_connection("duckdb")
        try:
            result = conn.execute("DELETE FROM context_cache WHERE expiry_time < CURRENT_TIMESTAMP")
            cleanup_results["l3_cleaned"] = result.fetchone()[0] if result else 0
        except Exception as e:
            logger.error(f"L3 cache cleanup failed: {e}")
        finally:
            self.return_connection(conn, "duckdb")
        
        return cleanup_results
    
    def close(self):
        """Clean up resources and close connections"""
        # Close all connections in pools
        with self.pool_lock:
            for conn in self.sqlite_pool:
                conn.close()
            for conn in self.duckdb_pool:
                conn.close()
            
            self.sqlite_pool.clear()
            self.duckdb_pool.clear()
        
        logger.info("Database caching layer closed")


# Factory function for easy instantiation
def create_database_caching_layer(**kwargs) -> DatabaseCachingLayer:
    """Create and configure database caching layer"""
    return DatabaseCachingLayer(**kwargs)

# Context manager for automatic cleanup
class DatabaseCachingContext:
    """Context manager for database caching layer with automatic cleanup"""
    
    def __init__(self, **kwargs):
        self.caching_layer = None
        self.kwargs = kwargs
    
    async def __aenter__(self) -> DatabaseCachingLayer:
        self.caching_layer = create_database_caching_layer(**self.kwargs)
        return self.caching_layer
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.caching_layer:
            self.caching_layer.close()

# CLI interface for testing and management
if __name__ == "__main__":
    import argparse
    
    async def test_caching_performance():
        """Test caching layer performance"""
        async with DatabaseCachingContext() as cache:
            print("Testing database caching layer performance...")
            
            # Test data
            test_data = {
                "agent_type": "rif-implementer",
                "context_items": ["item1", "item2", "item3"],
                "optimization_factors": {"relevance": 0.85, "freshness": 0.92}
            }
            
            # Test cache set operation
            start_time = time.time()
            success = await cache.set_cached_data("test_key", test_data, ttl=300)
            set_time = (time.time() - start_time) * 1000
            print(f"Cache set: {success}, Time: {set_time:.2f}ms")
            
            # Test cache get operation (should be cached)
            start_time = time.time()
            cached_data = await cache.get_cached_data("test_key")
            get_time = (time.time() - start_time) * 1000
            print(f"Cache get (hit): {cached_data is not None}, Time: {get_time:.2f}ms")
            
            # Test cache miss
            start_time = time.time()
            missing_data = await cache.get_cached_data("non_existent_key")
            miss_time = (time.time() - start_time) * 1000
            print(f"Cache get (miss): {missing_data is None}, Time: {miss_time:.2f}ms")
            
            # Performance analytics
            analytics = await cache.get_performance_analytics(1)
            print("\nPerformance Analytics:")
            print(f"Overall hit rate: {analytics.get('overall_hit_rate', 0):.2%}")
            print(f"L1 cache size: {analytics.get('l1_cache_size', 0)}")
    
    asyncio.run(test_caching_performance())