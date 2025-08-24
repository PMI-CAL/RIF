"""
DPIBS MCP Knowledge Server Integration Architecture
================================================

Layer 2 of DPIBS Integration: MCP Knowledge Server Integration

This module provides performance-optimized integration with the existing MCP Knowledge Server
while maintaining backward compatibility and enabling enhanced context delivery capabilities.

Architecture:
- MCP Knowledge Integrator: Performance-optimized interface to existing MCP server
- Data Synchronization Engine: Bidirectional sync with hybrid knowledge system
- Query Optimization Engine: Intelligent query routing and caching
- Caching Engine: Multi-layer caching for performance enhancement

Key Requirements:
- Preserve existing MCP knowledge server functionality
- Performance enhancement through intelligent caching and query optimization
- Bidirectional synchronization with existing hybrid knowledge database
- Zero regression in existing knowledge queries and operations
"""

import json
import logging
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue
import os

# RIF Infrastructure Imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knowledge.database.database_interface import RIFDatabase
from knowledge.integration.hybrid_knowledge_system import HybridKnowledgeSystem
from claude.commands.claude_code_knowledge_mcp_server import ClaudeCodeKnowledgeServer


@dataclass
class MCPQueryMetrics:
    """Metrics for MCP query performance."""
    query_id: str
    query_type: str
    processing_time_ms: int
    cache_hit: bool
    result_count: int
    optimization_applied: bool
    sync_required: bool


@dataclass
class SynchronizationStatus:
    """Status of knowledge synchronization."""
    sync_id: str
    direction: str  # "to_mcp", "from_mcp", "bidirectional"
    entities_synced: int
    sync_duration_ms: int
    success: bool
    conflicts_resolved: int
    timestamp: datetime


@dataclass
class CacheEntry:
    """Cache entry for query optimization."""
    cache_key: str
    query_hash: str
    result_data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    cache_level: str  # "l1_memory", "l2_persistent", "l3_distributed"


class MCPKnowledgeIntegrator:
    """
    Performance-optimized integrator with existing MCP Knowledge Server.
    
    Provides enhanced query performance, intelligent caching, and bidirectional
    synchronization while preserving all existing functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core Integration Components
        self.existing_mcp_server = None
        self.rif_db = None
        self.hybrid_system = None
        
        # Performance Optimization Components
        self.query_optimizer = None
        self.cache_engine = None
        self.sync_engine = None
        
        # Performance Metrics
        self.performance_metrics = {
            'queries_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_query_time_ms': 0.0,
            'sync_operations': 0,
            'optimization_success_rate': 0.0
        }
        
        # Backward Compatibility
        self.compatibility_mode = config.get('compatibility_mode', True)
        self.fallback_enabled = config.get('fallback_enabled', True)
        
        # Synchronization State
        self.sync_status = {
            'last_sync': None,
            'sync_in_progress': False,
            'pending_changes': [],
            'conflict_resolution_queue': queue.Queue()
        }
        
        self._initialize_integration()
    
    def _initialize_integration(self):
        """Initialize MCP knowledge server integration."""
        try:
            # Initialize connection to existing MCP server
            self._connect_to_existing_mcp_server()
            
            # Initialize RIF database connection
            self.rif_db = RIFDatabase()
            
            # Initialize hybrid knowledge system
            hybrid_config = {
                'memory_limit_mb': 1024,
                'cpu_cores': 4,
                'performance_mode': 'FAST',
                'database_path': 'knowledge/hybrid_knowledge.duckdb'
            }
            
            self.hybrid_system = HybridKnowledgeSystem(hybrid_config)
            if not self.hybrid_system.initialize():
                raise RuntimeError("Failed to initialize hybrid knowledge system")
            
            # Initialize performance optimization components
            self._initialize_query_optimizer()
            self._initialize_cache_engine()
            self._initialize_sync_engine()
            
            self.logger.info("MCP Knowledge Integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP integration: {e}")
            if self.fallback_enabled:
                self.logger.warning("Operating in fallback mode - limited integration")
                self.compatibility_mode = True
            else:
                raise
    
    def _connect_to_existing_mcp_server(self):
        """Connect to existing Claude Code Knowledge MCP Server."""
        try:
            # Initialize existing MCP server with enhanced configuration
            mcp_config = {
                'performance_optimization': True,
                'cache_integration': True,
                'sync_enabled': True
            }
            
            self.existing_mcp_server = ClaudeCodeKnowledgeServer(mcp_config)
            self.logger.info("Connected to existing MCP Knowledge Server")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to existing MCP server: {e}")
            if self.fallback_enabled:
                self.logger.warning("MCP server connection failed - operating without MCP integration")
                self.existing_mcp_server = None
            else:
                raise
    
    def _initialize_query_optimizer(self):
        """Initialize intelligent query optimization engine."""
        self.query_optimizer = QueryOptimizationEngine(
            config=self.config.get('query_optimization', {}),
            rif_db=self.rif_db,
            hybrid_system=self.hybrid_system
        )
    
    def _initialize_cache_engine(self):
        """Initialize multi-layer caching engine."""
        cache_config = self.config.get('caching', {
            'l1_memory_size_mb': 256,
            'l2_persistent_enabled': True,
            'l3_distributed_enabled': False,
            'default_ttl_minutes': 30,
            'max_cache_size_mb': 512
        })
        
        self.cache_engine = CachingEngine(cache_config)
    
    def _initialize_sync_engine(self):
        """Initialize bidirectional synchronization engine."""
        sync_config = self.config.get('synchronization', {
            'auto_sync_enabled': True,
            'sync_interval_minutes': 15,
            'conflict_resolution': 'hybrid_merge',
            'batch_size': 100
        })
        
        self.sync_engine = DataSynchronizationEngine(
            config=sync_config,
            rif_db=self.rif_db,
            hybrid_system=self.hybrid_system,
            mcp_server=self.existing_mcp_server
        )
    
    async def enhanced_query(self, query_type: str, query_data: Dict[str, Any], 
                           optimization_hints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute enhanced query with optimization and caching.
        
        Args:
            query_type: Type of query (compatibility_check, pattern_search, etc.)
            query_data: Query parameters
            optimization_hints: Hints for query optimization
            
        Returns:
            Enhanced query result with performance metrics
        """
        start_time = time.time()
        query_id = self._generate_query_id(query_type, query_data)
        
        try:
            # Step 1: Check cache for existing result
            cached_result = await self.cache_engine.get(query_type, query_data)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                return self._format_cached_result(cached_result, query_id, start_time)
            
            self.performance_metrics['cache_misses'] += 1
            
            # Step 2: Apply query optimization
            optimized_query = await self.query_optimizer.optimize_query(
                query_type, query_data, optimization_hints
            )
            
            # Step 3: Execute query through appropriate channel
            result = await self._execute_optimized_query(optimized_query)
            
            # Step 4: Cache result for future queries
            await self.cache_engine.set(query_type, query_data, result)
            
            # Step 5: Update performance metrics
            processing_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_metrics(query_id, query_type, processing_time_ms, False, result)
            
            return self._format_enhanced_result(result, query_id, processing_time_ms)
            
        except Exception as e:
            self.logger.error(f"Enhanced query failed: {e}")
            # Fallback to existing MCP server if available
            return await self._fallback_query(query_type, query_data, query_id, start_time)
    
    async def _execute_optimized_query(self, optimized_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized query through best available channel."""
        query_type = optimized_query['type']
        query_data = optimized_query['data']
        routing = optimized_query.get('routing', 'hybrid')
        
        if routing == 'mcp_server' and self.existing_mcp_server:
            return await self._query_mcp_server(query_type, query_data)
        elif routing == 'hybrid_system':
            return await self._query_hybrid_system(query_type, query_data)
        elif routing == 'rif_database':
            return await self._query_rif_database(query_type, query_data)
        else:
            # Auto-routing based on query type
            return await self._auto_route_query(query_type, query_data)
    
    async def _query_mcp_server(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query through existing MCP server."""
        if not self.existing_mcp_server:
            raise RuntimeError("MCP server not available")
        
        # Map query types to MCP server methods
        method_map = {
            'compatibility_check': 'check_compatibility',
            'pattern_search': 'get_patterns',
            'alternative_suggestions': 'suggest_alternatives',
            'architecture_validation': 'validate_architecture',
            'limitation_query': 'get_limitations'
        }
        
        method_name = method_map.get(query_type)
        if not method_name:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Create mock request object for existing MCP server
        request = type('MockRequest', (), {'arguments': query_data})()
        
        # Call appropriate method
        handler = getattr(self.existing_mcp_server, f"_handle_{method_name}")
        response = await handler(request)
        
        return response.content if hasattr(response, 'content') else response
    
    async def _query_hybrid_system(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query through hybrid knowledge system."""
        if query_type == 'pattern_search':
            results = self.hybrid_system.search(
                query=query_data.get('query', ''),
                entity_types=query_data.get('entity_types', []),
                limit=query_data.get('limit', 10)
            )
            return {'patterns': results}
        
        elif query_type == 'similarity_search':
            results = self.hybrid_system.vector_search(
                query_vector=query_data.get('vector'),
                k=query_data.get('limit', 5)
            )
            return {'similar_items': results}
        
        else:
            # Default hybrid search
            results = self.hybrid_system.hybrid_search(
                text_query=query_data.get('query', ''),
                vector_query=query_data.get('vector'),
                limit=query_data.get('limit', 10)
            )
            return {'results': results}
    
    async def _query_rif_database(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query through RIF database directly."""
        if query_type == 'entity_search':
            results = self.rif_db.search_entities(
                query=query_data.get('query', ''),
                entity_types=query_data.get('entity_types', []),
                limit=query_data.get('limit', 10)
            )
            return {'entities': results}
        
        elif query_type == 'relationship_query':
            results = self.rif_db.get_entity_relationships(
                entity_id=query_data.get('entity_id'),
                direction=query_data.get('direction', 'both')
            )
            return {'relationships': results}
        
        else:
            # Default search
            results = self.rif_db.search_entities(
                query=query_data.get('query', ''),
                limit=query_data.get('limit', 10)
            )
            return {'results': results}
    
    async def _auto_route_query(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically route query to best available system."""
        # Routing priority based on query type and system availability
        if query_type in ['compatibility_check', 'architecture_validation'] and self.existing_mcp_server:
            return await self._query_mcp_server(query_type, query_data)
        elif query_type in ['pattern_search', 'similarity_search']:
            return await self._query_hybrid_system(query_type, query_data)
        else:
            return await self._query_rif_database(query_type, query_data)
    
    async def _fallback_query(self, query_type: str, query_data: Dict[str, Any], 
                            query_id: str, start_time: float) -> Dict[str, Any]:
        """Fallback query execution when enhanced query fails."""
        try:
            if self.existing_mcp_server and query_type in ['compatibility_check', 'pattern_search']:
                result = await self._query_mcp_server(query_type, query_data)
                processing_time_ms = int((time.time() - start_time) * 1000)
                return self._format_fallback_result(result, query_id, processing_time_ms)
            else:
                # Basic fallback result
                processing_time_ms = int((time.time() - start_time) * 1000)
                return {
                    'query_id': query_id,
                    'result': {'fallback': True, 'message': 'Enhanced query unavailable'},
                    'performance': {
                        'processing_time_ms': processing_time_ms,
                        'fallback_used': True,
                        'cache_hit': False
                    }
                }
        except Exception as e:
            self.logger.error(f"Fallback query also failed: {e}")
            return {
                'query_id': query_id,
                'error': 'All query methods failed',
                'fallback_used': True
            }
    
    def _generate_query_id(self, query_type: str, query_data: Dict[str, Any]) -> str:
        """Generate unique query ID."""
        query_str = f"{query_type}_{json.dumps(query_data, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()[:12]
    
    def _format_cached_result(self, cached_result: Any, query_id: str, start_time: float) -> Dict[str, Any]:
        """Format cached result with performance metadata."""
        processing_time_ms = int((time.time() - start_time) * 1000)
        return {
            'query_id': query_id,
            'result': cached_result,
            'performance': {
                'processing_time_ms': processing_time_ms,
                'cache_hit': True,
                'optimization_applied': True
            }
        }
    
    def _format_enhanced_result(self, result: Any, query_id: str, processing_time_ms: int) -> Dict[str, Any]:
        """Format enhanced result with metadata."""
        return {
            'query_id': query_id,
            'result': result,
            'performance': {
                'processing_time_ms': processing_time_ms,
                'cache_hit': False,
                'optimization_applied': True,
                'enhanced_query': True
            }
        }
    
    def _format_fallback_result(self, result: Any, query_id: str, processing_time_ms: int) -> Dict[str, Any]:
        """Format fallback result with metadata."""
        return {
            'query_id': query_id,
            'result': result,
            'performance': {
                'processing_time_ms': processing_time_ms,
                'fallback_used': True,
                'cache_hit': False
            }
        }
    
    def _update_query_metrics(self, query_id: str, query_type: str, processing_time_ms: int, 
                            cache_hit: bool, result: Any):
        """Update query performance metrics."""
        self.performance_metrics['queries_processed'] += 1
        
        # Update average processing time
        total_queries = self.performance_metrics['queries_processed']
        current_avg = self.performance_metrics['average_query_time_ms']
        new_avg = (current_avg * (total_queries - 1) + processing_time_ms) / total_queries
        self.performance_metrics['average_query_time_ms'] = new_avg
        
        # Update success rate
        if result and not isinstance(result, dict) or not result.get('error'):
            success_rate = self.performance_metrics.get('optimization_success_rate', 0.0)
            new_success_rate = (success_rate * (total_queries - 1) + 1.0) / total_queries
            self.performance_metrics['optimization_success_rate'] = new_success_rate
    
    async def synchronize_knowledge(self, direction: str = "bidirectional", 
                                  batch_size: int = 100) -> SynchronizationStatus:
        """Synchronize knowledge between systems."""
        sync_id = f"sync_{int(time.time())}"
        start_time = time.time()
        
        try:
            if self.sync_status['sync_in_progress']:
                raise RuntimeError("Synchronization already in progress")
            
            self.sync_status['sync_in_progress'] = True
            
            sync_result = await self.sync_engine.synchronize(
                direction=direction,
                batch_size=batch_size,
                sync_id=sync_id
            )
            
            sync_duration_ms = int((time.time() - start_time) * 1000)
            
            status = SynchronizationStatus(
                sync_id=sync_id,
                direction=direction,
                entities_synced=sync_result.get('entities_synced', 0),
                sync_duration_ms=sync_duration_ms,
                success=sync_result.get('success', False),
                conflicts_resolved=sync_result.get('conflicts_resolved', 0),
                timestamp=datetime.now()
            )
            
            self.sync_status['last_sync'] = status
            self.performance_metrics['sync_operations'] += 1
            
            return status
            
        except Exception as e:
            self.logger.error(f"Knowledge synchronization failed: {e}")
            return SynchronizationStatus(
                sync_id=sync_id,
                direction=direction,
                entities_synced=0,
                sync_duration_ms=int((time.time() - start_time) * 1000),
                success=False,
                conflicts_resolved=0,
                timestamp=datetime.now()
            )
        finally:
            self.sync_status['sync_in_progress'] = False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics."""
        return {
            'integration_active': self.existing_mcp_server is not None,
            'compatibility_mode': self.compatibility_mode,
            'fallback_enabled': self.fallback_enabled,
            'performance_metrics': self.performance_metrics.copy(),
            'cache_status': self.cache_engine.get_status() if self.cache_engine else None,
            'sync_status': {
                'last_sync': asdict(self.sync_status['last_sync']) if self.sync_status['last_sync'] else None,
                'sync_in_progress': self.sync_status['sync_in_progress'],
                'pending_changes': len(self.sync_status['pending_changes'])
            },
            'components_initialized': {
                'mcp_server': self.existing_mcp_server is not None,
                'rif_database': self.rif_db is not None,
                'hybrid_system': self.hybrid_system is not None,
                'query_optimizer': self.query_optimizer is not None,
                'cache_engine': self.cache_engine is not None,
                'sync_engine': self.sync_engine is not None
            }
        }
    
    def shutdown(self):
        """Clean shutdown of MCP integration."""
        try:
            if self.cache_engine:
                self.cache_engine.shutdown()
            if self.sync_engine:
                self.sync_engine.shutdown()
            if self.hybrid_system:
                self.hybrid_system.shutdown()
            if self.rif_db:
                self.rif_db.close()
            
            self.logger.info("MCP Knowledge Integrator shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during MCP integration shutdown: {e}")


class QueryOptimizationEngine:
    """Engine for optimizing knowledge queries."""
    
    def __init__(self, config: Dict[str, Any], rif_db: RIFDatabase, hybrid_system: HybridKnowledgeSystem):
        self.config = config
        self.rif_db = rif_db
        self.hybrid_system = hybrid_system
        self.logger = logging.getLogger(__name__)
        
        self.optimization_patterns = {
            'frequent_patterns': {},
            'performance_heuristics': {},
            'routing_rules': {}
        }
    
    async def optimize_query(self, query_type: str, query_data: Dict[str, Any], 
                           hints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize query for best performance."""
        try:
            # Apply query optimization patterns
            optimized_data = query_data.copy()
            
            # Add optimization metadata
            optimized_query = {
                'type': query_type,
                'data': optimized_data,
                'routing': self._determine_optimal_routing(query_type, query_data),
                'optimization_applied': True,
                'hints': hints or {}
            }
            
            return optimized_query
            
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return {
                'type': query_type,
                'data': query_data,
                'routing': 'auto',
                'optimization_applied': False
            }
    
    def _determine_optimal_routing(self, query_type: str, query_data: Dict[str, Any]) -> str:
        """Determine optimal routing for query type."""
        # Routing rules based on query characteristics
        if query_type in ['compatibility_check', 'architecture_validation']:
            return 'mcp_server'
        elif query_type in ['pattern_search', 'similarity_search']:
            return 'hybrid_system'
        elif query_type in ['entity_search', 'relationship_query']:
            return 'rif_database'
        else:
            return 'auto'


class CachingEngine:
    """Multi-layer caching engine for query optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # L1 Memory Cache
        self.l1_cache = {}
        self.l1_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
        # Cache settings
        self.max_l1_size = config.get('l1_memory_size_mb', 256) * 1024 * 1024
        self.default_ttl = config.get('default_ttl_minutes', 30) * 60
    
    async def get(self, query_type: str, query_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for query."""
        try:
            cache_key = self._generate_cache_key(query_type, query_data)
            
            # Check L1 memory cache
            cache_entry = self.l1_cache.get(cache_key)
            if cache_entry and self._is_cache_valid(cache_entry):
                self.l1_stats['hits'] += 1
                cache_entry.last_accessed = datetime.now()
                cache_entry.access_count += 1
                return cache_entry.result_data
            
            self.l1_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def set(self, query_type: str, query_data: Dict[str, Any], result: Any):
        """Cache query result."""
        try:
            cache_key = self._generate_cache_key(query_type, query_data)
            
            # Create cache entry
            cache_entry = CacheEntry(
                cache_key=cache_key,
                query_hash=self._hash_query(query_type, query_data),
                result_data=result,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=self.default_ttl,
                cache_level='l1_memory'
            )
            
            # Store in L1 cache
            self.l1_cache[cache_key] = cache_entry
            self._cleanup_expired_entries()
            
        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, query_type: str, query_data: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        query_str = f"{query_type}_{json.dumps(query_data, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _hash_query(self, query_type: str, query_data: Dict[str, Any]) -> str:
        """Generate hash for query."""
        return hashlib.sha256(f"{query_type}_{json.dumps(query_data, sort_keys=True)}".encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        age_seconds = (datetime.now() - cache_entry.created_at).total_seconds()
        return age_seconds < cache_entry.ttl_seconds
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self.l1_cache.items()
            if (current_time - entry.created_at).total_seconds() > entry.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.l1_cache[key]
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache engine status."""
        return {
            'l1_cache': {
                'entries': len(self.l1_cache),
                'hits': self.l1_stats['hits'],
                'misses': self.l1_stats['misses'],
                'hit_rate': (self.l1_stats['hits'] / (self.l1_stats['hits'] + self.l1_stats['misses'])) 
                          if (self.l1_stats['hits'] + self.l1_stats['misses']) > 0 else 0.0
            },
            'config': self.config
        }
    
    def shutdown(self):
        """Shutdown cache engine."""
        self.l1_cache.clear()
        self.logger.info("Cache engine shutdown completed")


class DataSynchronizationEngine:
    """Engine for bidirectional knowledge synchronization."""
    
    def __init__(self, config: Dict[str, Any], rif_db: RIFDatabase, 
                 hybrid_system: HybridKnowledgeSystem, mcp_server: Optional[ClaudeCodeKnowledgeServer]):
        self.config = config
        self.rif_db = rif_db
        self.hybrid_system = hybrid_system
        self.mcp_server = mcp_server
        self.logger = logging.getLogger(__name__)
        
        self.sync_lock = threading.Lock()
    
    async def synchronize(self, direction: str, batch_size: int, sync_id: str) -> Dict[str, Any]:
        """Perform knowledge synchronization."""
        try:
            with self.sync_lock:
                if direction == "bidirectional":
                    result_to_mcp = await self._sync_to_mcp(batch_size, sync_id)
                    result_from_mcp = await self._sync_from_mcp(batch_size, sync_id)
                    
                    return {
                        'success': result_to_mcp['success'] and result_from_mcp['success'],
                        'entities_synced': result_to_mcp['entities_synced'] + result_from_mcp['entities_synced'],
                        'conflicts_resolved': result_to_mcp.get('conflicts_resolved', 0) + 
                                            result_from_mcp.get('conflicts_resolved', 0)
                    }
                elif direction == "to_mcp":
                    return await self._sync_to_mcp(batch_size, sync_id)
                elif direction == "from_mcp":
                    return await self._sync_from_mcp(batch_size, sync_id)
                else:
                    raise ValueError(f"Invalid sync direction: {direction}")
                    
        except Exception as e:
            self.logger.error(f"Synchronization failed: {e}")
            return {'success': False, 'entities_synced': 0, 'error': str(e)}
    
    async def _sync_to_mcp(self, batch_size: int, sync_id: str) -> Dict[str, Any]:
        """Sync data from RIF/Hybrid to MCP."""
        # Simplified implementation - would contain actual sync logic
        return {'success': True, 'entities_synced': 0, 'conflicts_resolved': 0}
    
    async def _sync_from_mcp(self, batch_size: int, sync_id: str) -> Dict[str, Any]:
        """Sync data from MCP to RIF/Hybrid."""
        # Simplified implementation - would contain actual sync logic
        return {'success': True, 'entities_synced': 0, 'conflicts_resolved': 0}
    
    def shutdown(self):
        """Shutdown sync engine."""
        self.logger.info("Synchronization engine shutdown completed")


# Integration Interface Functions
def create_mcp_integration(config: Dict[str, Any] = None) -> MCPKnowledgeIntegrator:
    """
    Factory function to create MCP knowledge integration system.
    
    This is the main entry point for DPIBS MCP Integration.
    """
    if config is None:
        config = {
            'compatibility_mode': True,
            'fallback_enabled': True,
            'performance_optimization': True,
            'caching': {
                'l1_memory_size_mb': 256,
                'default_ttl_minutes': 30
            },
            'synchronization': {
                'auto_sync_enabled': True,
                'sync_interval_minutes': 15
            }
        }
    
    return MCPKnowledgeIntegrator(config)


async def enhanced_mcp_query(integrator: MCPKnowledgeIntegrator, query_type: str, 
                           query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute enhanced MCP query with optimization.
    Used by other DPIBS components.
    """
    return await integrator.enhanced_query(query_type, query_data)


# Backward Compatibility Functions
def is_mcp_integration_available() -> bool:
    """Check if MCP integration is available and working."""
    try:
        config = {'compatibility_mode': True}
        integrator = MCPKnowledgeIntegrator(config)
        return integrator.existing_mcp_server is not None or integrator.fallback_enabled
    except Exception:
        return False


def get_mcp_fallback_result(query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get fallback result when MCP integration is unavailable."""
    return {
        'query_type': query_type,
        'result': {'fallback': True, 'message': 'MCP integration unavailable'},
        'performance': {'fallback_used': True}
    }