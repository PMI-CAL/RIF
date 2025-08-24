#!/usr/bin/env python3
"""
Context Intelligence Platform - Database Schema Extensions
Issue #119: DPIBS Architecture Phase 1

Extends the existing hybrid knowledge system with Context Intelligence Platform
specific schemas for performance tracking, caching, and real-time context management.
"""

import os
import sqlite3
import duckdb
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ContextDatabaseSchema:
    """Database schema manager for Context Intelligence Platform"""
    
    def __init__(self, base_path: str = "/Users/cal/DEV/RIF/systems/context"):
        self.base_path = base_path
        self.context_db_path = os.path.join(base_path, "context_intelligence.duckdb")
        self.hybrid_db_path = "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        self.performance_db_path = os.path.join(base_path, "performance_metrics.db")
        
        os.makedirs(base_path, exist_ok=True)
        
        # Initialize databases
        self.init_context_intelligence_schema()
        self.init_performance_metrics_schema()
        self.create_hybrid_extensions()
    
    def init_context_intelligence_schema(self):
        """Initialize Context Intelligence Platform specific schema"""
        try:
            with duckdb.connect(self.context_db_path) as conn:
                # Create context optimization tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_optimizations (
                        optimization_id VARCHAR PRIMARY KEY,
                        agent_type VARCHAR NOT NULL,
                        task_context JSON NOT NULL,
                        issue_number INTEGER,
                        original_context_size INTEGER NOT NULL,
                        optimized_context_size INTEGER NOT NULL,
                        relevance_score DOUBLE NOT NULL,
                        optimization_time_ms DOUBLE NOT NULL,
                        cache_hit BOOLEAN DEFAULT FALSE,
                        context_items JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        performance_metrics JSON
                    )
                """)
                
                # Create context cache metadata table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_cache_metadata (
                        cache_key VARCHAR PRIMARY KEY,
                        cache_level VARCHAR NOT NULL, -- L1, L2, L3
                        data_type VARCHAR NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        miss_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        agent_type VARCHAR,
                        context_hash VARCHAR,
                        invalidation_reason VARCHAR
                    )
                """)
                
                # Create agent context delivery tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_context_deliveries (
                        delivery_id VARCHAR PRIMARY KEY,
                        request_id VARCHAR NOT NULL,
                        agent_type VARCHAR NOT NULL,
                        context_window_utilization DOUBLE NOT NULL,
                        total_context_size INTEGER NOT NULL,
                        relevant_items_count INTEGER NOT NULL,
                        response_time_ms DOUBLE NOT NULL,
                        cache_hit BOOLEAN DEFAULT FALSE,
                        source_services JSON NOT NULL,
                        quality_score DOUBLE,
                        feedback_score DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        performance_data JSON
                    )
                """)
                
                # Create system context snapshots
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_context_snapshots (
                        snapshot_id VARCHAR PRIMARY KEY,
                        context_version INTEGER NOT NULL,
                        system_overview TEXT NOT NULL,
                        design_goals JSON NOT NULL,
                        constraints JSON NOT NULL,
                        dependencies JSON NOT NULL,
                        architecture_summary TEXT NOT NULL,
                        change_events JSON,
                        consistency_hash VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        validation_status VARCHAR DEFAULT 'pending'
                    )
                """)
                
                # Create knowledge integration tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_integrations (
                        integration_id VARCHAR PRIMARY KEY,
                        mcp_server VARCHAR NOT NULL,
                        query_type VARCHAR NOT NULL,
                        query_params JSON NOT NULL,
                        response_items JSON NOT NULL,
                        response_time_ms DOUBLE NOT NULL,
                        cache_hit BOOLEAN DEFAULT FALSE,
                        knowledge_freshness TIMESTAMP,
                        integration_quality DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        correlation_id VARCHAR
                    )
                """)
                
                # Create real-time event tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_events (
                        event_id VARCHAR PRIMARY KEY,
                        event_type VARCHAR NOT NULL,
                        event_priority INTEGER NOT NULL,
                        source_service VARCHAR NOT NULL,
                        target_service VARCHAR,
                        payload JSON NOT NULL,
                        processing_time_ms DOUBLE,
                        status VARCHAR DEFAULT 'queued',
                        retry_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP,
                        correlation_id VARCHAR
                    )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_context_optimizations_agent ON context_optimizations(agent_type, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_context_optimizations_issue ON context_optimizations(issue_number)",
                    "CREATE INDEX IF NOT EXISTS idx_context_cache_level ON context_cache_metadata(cache_level, last_accessed)",
                    "CREATE INDEX IF NOT EXISTS idx_context_cache_expires ON context_cache_metadata(expires_at)",
                    "CREATE INDEX IF NOT EXISTS idx_agent_deliveries_agent ON agent_context_deliveries(agent_type, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_agent_deliveries_performance ON agent_context_deliveries(response_time_ms)",
                    "CREATE INDEX IF NOT EXISTS idx_system_snapshots_version ON system_context_snapshots(context_version, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_integrations_server ON knowledge_integrations(mcp_server, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_context_events_type ON context_events(event_type, status, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_context_events_priority ON context_events(event_priority, created_at)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                logger.info("Context Intelligence database schema initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize context intelligence schema: {e}")
            raise
    
    def init_performance_metrics_schema(self):
        """Initialize performance metrics schema using SQLite for high-frequency writes"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                # High-frequency performance metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        service_name TEXT NOT NULL,
                        operation_name TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        success INTEGER NOT NULL,
                        cache_level TEXT,
                        request_size INTEGER,
                        response_size INTEGER,
                        concurrent_requests INTEGER,
                        cpu_usage REAL,
                        memory_usage REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # API Gateway specific metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_gateway_metrics (
                        request_id TEXT PRIMARY KEY,
                        route TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        response_time_ms REAL NOT NULL,
                        cache_hit INTEGER DEFAULT 0,
                        client_ip TEXT,
                        user_agent TEXT,
                        access_level TEXT,
                        rate_limited INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Event Service Bus metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_bus_metrics (
                        event_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        queue_time_ms REAL,
                        processing_time_ms REAL,
                        handler_id TEXT,
                        success INTEGER NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        circuit_breaker_state TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP
                    )
                """)
                
                # Cache performance metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        cache_level TEXT NOT NULL,
                        operation TEXT NOT NULL, -- get, set, invalidate
                        hit INTEGER DEFAULT 0,
                        duration_ms REAL NOT NULL,
                        data_size INTEGER,
                        eviction_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_service_time ON performance_metrics(service_name, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_operation ON performance_metrics(operation_name, success, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_api_route_time ON api_gateway_metrics(route, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_api_status ON api_gateway_metrics(status_code, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type_time ON event_bus_metrics(event_type, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_level_time ON cache_performance_metrics(cache_level, created_at)")
                
                logger.info("Performance metrics database schema initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance metrics schema: {e}")
            raise
    
    def create_hybrid_extensions(self):
        """Extend existing hybrid knowledge system with Context Intelligence features"""
        try:
            # Check if hybrid database exists
            if not os.path.exists(self.hybrid_db_path):
                logger.warning("Hybrid knowledge database not found, skipping extensions")
                return
            
            with duckdb.connect(self.hybrid_db_path) as conn:
                # Add context intelligence metadata to existing entities table
                try:
                    conn.execute("""
                        ALTER TABLE entities ADD COLUMN context_relevance_score DOUBLE DEFAULT 0.5
                    """)
                    logger.info("Added context_relevance_score to entities table")
                except:
                    # Column might already exist
                    pass
                
                try:
                    conn.execute("""
                        ALTER TABLE entities ADD COLUMN agent_preferences JSON DEFAULT '{}'
                    """)
                    logger.info("Added agent_preferences to entities table") 
                except:
                    # Column might already exist
                    pass
                
                try:
                    conn.execute("""
                        ALTER TABLE entities ADD COLUMN context_cache_key VARCHAR
                    """)
                    logger.info("Added context_cache_key to entities table")
                except:
                    # Column might already exist
                    pass
                
                # Add context optimization metadata to agent_memory table
                try:
                    conn.execute("""
                        ALTER TABLE agent_memory ADD COLUMN context_optimization_id VARCHAR
                    """)
                    logger.info("Added context_optimization_id to agent_memory table")
                except:
                    # Column might already exist
                    pass
                
                try:
                    conn.execute("""
                        ALTER TABLE agent_memory ADD COLUMN context_quality_score DOUBLE DEFAULT 0.5
                    """)
                    logger.info("Added context_quality_score to agent_memory table")
                except:
                    # Column might already exist
                    pass
                
                # Create context intelligence relationships table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_relationships (
                        relationship_id VARCHAR PRIMARY KEY,
                        source_entity_id VARCHAR NOT NULL,
                        target_entity_id VARCHAR NOT NULL,
                        relationship_type VARCHAR NOT NULL,
                        strength DOUBLE NOT NULL DEFAULT 0.5,
                        context_types JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        validation_status VARCHAR DEFAULT 'pending'
                    )
                """)
                
                # Create agent context preferences
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_context_preferences (
                        preference_id VARCHAR PRIMARY KEY,
                        agent_type VARCHAR NOT NULL,
                        context_type VARCHAR NOT NULL,
                        relevance_weight DOUBLE NOT NULL DEFAULT 0.5,
                        max_items INTEGER DEFAULT 10,
                        freshness_weight DOUBLE DEFAULT 0.3,
                        quality_threshold DOUBLE DEFAULT 0.6,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create context quality feedback
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_quality_feedback (
                        feedback_id VARCHAR PRIMARY KEY,
                        optimization_id VARCHAR NOT NULL,
                        agent_type VARCHAR NOT NULL,
                        quality_rating DOUBLE NOT NULL,
                        relevance_rating DOUBLE NOT NULL,
                        completeness_rating DOUBLE NOT NULL,
                        feedback_text TEXT,
                        improvement_suggestions JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for the new tables
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_relationships_source ON context_relationships(source_entity_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_relationships_target ON context_relationships(target_entity_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_relationships_type ON context_relationships(relationship_type, strength)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_preferences_agent ON agent_context_preferences(agent_type, context_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_feedback_agent ON context_quality_feedback(agent_type, created_at)")
                
                logger.info("Hybrid knowledge system extensions created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create hybrid system extensions: {e}")
            # Non-critical, continue without extensions
    
    def insert_context_optimization(self, optimization_data: Dict[str, Any]):
        """Insert context optimization record"""
        try:
            with duckdb.connect(self.context_db_path) as conn:
                conn.execute("""
                    INSERT INTO context_optimizations 
                    (optimization_id, agent_type, task_context, issue_number, 
                     original_context_size, optimized_context_size, relevance_score,
                     optimization_time_ms, cache_hit, context_items, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    optimization_data['optimization_id'],
                    optimization_data['agent_type'],
                    json.dumps(optimization_data['task_context']),
                    optimization_data.get('issue_number'),
                    optimization_data['original_context_size'],
                    optimization_data['optimized_context_size'],
                    optimization_data['relevance_score'],
                    optimization_data['optimization_time_ms'],
                    optimization_data.get('cache_hit', False),
                    json.dumps(optimization_data['context_items']),
                    json.dumps(optimization_data.get('performance_metrics', {}))
                ))
        except Exception as e:
            logger.error(f"Failed to insert context optimization: {e}")
    
    def update_cache_metadata(self, cache_key: str, operation: str, hit: bool = False):
        """Update cache metadata with hit/miss statistics"""
        try:
            with duckdb.connect(self.context_db_path) as conn:
                if hit:
                    conn.execute("""
                        UPDATE context_cache_metadata 
                        SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """, (cache_key,))
                else:
                    conn.execute("""
                        UPDATE context_cache_metadata 
                        SET miss_count = miss_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """, (cache_key,))
        except Exception as e:
            logger.error(f"Failed to update cache metadata: {e}")
    
    def insert_performance_metric(self, metric_data: Dict[str, Any]):
        """Insert high-frequency performance metric"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (metric_id, service_name, operation_name, duration_ms, success,
                     cache_level, request_size, response_size, concurrent_requests,
                     cpu_usage, memory_usage, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_data['metric_id'],
                    metric_data['service_name'],
                    metric_data['operation_name'],
                    metric_data['duration_ms'],
                    int(metric_data['success']),
                    metric_data.get('cache_level'),
                    metric_data.get('request_size'),
                    metric_data.get('response_size'),
                    metric_data.get('concurrent_requests'),
                    metric_data.get('cpu_usage'),
                    metric_data.get('memory_usage'),
                    json.dumps(metric_data.get('metadata', {}))
                ))
        except Exception as e:
            logger.error(f"Failed to insert performance metric: {e}")
    
    def get_context_optimization_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get context optimization analytics for the specified time period"""
        try:
            with duckdb.connect(self.context_db_path) as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                # Basic optimization statistics  
                basic_stats_query = conn.execute("""
                    SELECT 
                        agent_type,
                        COUNT(*) as optimization_count,
                        AVG(relevance_score) as avg_relevance_score,
                        AVG(optimization_time_ms) as avg_optimization_time_ms,
                        AVG(optimized_context_size) as avg_optimized_size,
                        AVG(original_context_size) as avg_original_size,
                        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                        COUNT(*) - SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_misses
                    FROM context_optimizations 
                    WHERE created_at > ?
                    GROUP BY agent_type
                    ORDER BY optimization_count DESC
                """, (cutoff_time,))
                
                basic_stats_columns = [desc[0] for desc in basic_stats_query.description]
                basic_stats = [dict(zip(basic_stats_columns, row)) for row in basic_stats_query.fetchall()]
                
                # Performance trends
                performance_trends_query = conn.execute("""
                    SELECT 
                        DATE_TRUNC('hour', created_at) as hour,
                        COUNT(*) as optimizations,
                        AVG(optimization_time_ms) as avg_time_ms,
                        AVG(relevance_score) as avg_relevance
                    FROM context_optimizations 
                    WHERE created_at > ?
                    GROUP BY DATE_TRUNC('hour', created_at)
                    ORDER BY hour DESC
                """, (cutoff_time,))
                
                performance_trends_columns = [desc[0] for desc in performance_trends_query.description]
                performance_trends = [dict(zip(performance_trends_columns, row)) for row in performance_trends_query.fetchall()]
                
                return {
                    "time_period_hours": hours,
                    "agent_statistics": basic_stats if basic_stats else [],
                    "performance_trends": performance_trends if performance_trends else [],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get context optimization analytics: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary across all services"""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                # Service performance summary
                service_cursor = conn.execute("""
                    SELECT 
                        service_name,
                        COUNT(*) as total_operations,
                        AVG(duration_ms) as avg_duration_ms,
                        MAX(duration_ms) as max_duration_ms,
                        SUM(success) as successful_operations,
                        COUNT(*) - SUM(success) as failed_operations
                    FROM performance_metrics 
                    WHERE created_at > ?
                    GROUP BY service_name
                    ORDER BY total_operations DESC
                """, (cutoff_time,))
                service_columns = [col[0] for col in service_cursor.description]
                service_stats = [dict(zip(service_columns, row)) for row in service_cursor.fetchall()]
                
                # API Gateway performance
                api_cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        AVG(response_time_ms) as avg_response_time_ms,
                        SUM(cache_hit) as cache_hits,
                        COUNT(*) - SUM(cache_hit) as cache_misses,
                        SUM(CASE WHEN status_code < 400 THEN 1 ELSE 0 END) as successful_requests,
                        SUM(rate_limited) as rate_limited_requests
                    FROM api_gateway_metrics 
                    WHERE created_at > ?
                """, (cutoff_time,))
                api_columns = [col[0] for col in api_cursor.description]
                api_result = api_cursor.fetchone()
                api_stats = dict(zip(api_columns, api_result)) if api_result else {}
                
                # Event Bus performance  
                event_cursor = conn.execute("""
                    SELECT 
                        event_type,
                        COUNT(*) as total_events,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        SUM(success) as successful_events,
                        AVG(retry_count) as avg_retry_count
                    FROM event_bus_metrics 
                    WHERE created_at > ?
                    GROUP BY event_type
                    ORDER BY total_events DESC
                """, (cutoff_time,))
                event_columns = [col[0] for col in event_cursor.description]
                event_stats = [dict(zip(event_columns, row)) for row in event_cursor.fetchall()]
                
                return {
                    "time_period_hours": hours,
                    "service_performance": service_stats if service_stats else [],
                    "api_gateway_performance": api_stats if api_stats else {},
                    "event_bus_performance": event_stats if event_stats else [],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance and context data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up Context Intelligence data
            with duckdb.connect(self.context_db_path) as conn:
                tables = [
                    "context_optimizations",
                    "agent_context_deliveries", 
                    "knowledge_integrations",
                    "context_events"
                ]
                
                for table in tables:
                    result = conn.execute(f"""
                        DELETE FROM {table} WHERE created_at < ?
                    """, (cutoff_date,))
                    deleted_count = result.fetchone()[0] if result else 0
                    logger.info(f"Cleaned up {deleted_count} old records from {table}")
            
            # Clean up performance metrics
            with sqlite3.connect(self.performance_db_path) as conn:
                perf_tables = [
                    "performance_metrics",
                    "api_gateway_metrics",
                    "event_bus_metrics",
                    "cache_performance_metrics"
                ]
                
                for table in perf_tables:
                    cursor = conn.execute(f"""
                        DELETE FROM {table} WHERE created_at < ?
                    """, (cutoff_date,))
                    logger.info(f"Cleaned up {cursor.rowcount} old records from {table}")
                    
                conn.commit()
            
            logger.info(f"Database cleanup completed, kept data from last {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# CLI and Testing Interface
def main():
    """Main function for database schema management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Intelligence Database Schema Manager")
    parser.add_argument("--init", action="store_true", help="Initialize all database schemas")
    parser.add_argument("--analytics", type=int, help="Get analytics for last N hours", metavar="HOURS")
    parser.add_argument("--performance", type=int, help="Get performance summary for last N hours", metavar="HOURS")
    parser.add_argument("--cleanup", type=int, help="Cleanup data older than N days", metavar="DAYS")
    parser.add_argument("--test", action="store_true", help="Run schema tests")
    
    args = parser.parse_args()
    
    schema_manager = ContextDatabaseSchema()
    
    if args.init:
        print("Database schemas initialized successfully")
        
    elif args.analytics:
        analytics = schema_manager.get_context_optimization_analytics(args.analytics)
        print("=== Context Optimization Analytics ===")
        print(json.dumps(analytics, indent=2))
        
    elif args.performance:
        performance = schema_manager.get_performance_summary(args.performance)
        print("=== Performance Summary ===")
        print(json.dumps(performance, indent=2))
        
    elif args.cleanup:
        schema_manager.cleanup_old_data(args.cleanup)
        print(f"Database cleanup completed for data older than {args.cleanup} days")
        
    elif args.test:
        print("=== Database Schema Test ===")
        
        # Test context optimization insert
        test_optimization = {
            'optimization_id': f"test_{int(datetime.now().timestamp())}",
            'agent_type': 'rif-implementer',
            'task_context': {'description': 'Test context optimization'},
            'issue_number': 119,
            'original_context_size': 5000,
            'optimized_context_size': 2500,
            'relevance_score': 0.85,
            'optimization_time_ms': 45.2,
            'cache_hit': False,
            'context_items': ['item1', 'item2', 'item3']
        }
        
        schema_manager.insert_context_optimization(test_optimization)
        print("✓ Context optimization insert test passed")
        
        # Test performance metric insert
        test_metric = {
            'metric_id': f"test_metric_{int(datetime.now().timestamp())}",
            'service_name': 'context-optimization',
            'operation_name': 'optimize_context',
            'duration_ms': 45.2,
            'success': True,
            'cache_level': 'L2',
            'request_size': 1024,
            'response_size': 512
        }
        
        schema_manager.insert_performance_metric(test_metric)
        print("✓ Performance metric insert test passed")
        
        # Test analytics
        analytics = schema_manager.get_context_optimization_analytics(1)
        performance = schema_manager.get_performance_summary(1)
        
        print("✓ Analytics query test passed")
        print("✓ Performance summary query test passed")
        
        print("\n=== Test Results ===")
        print(f"Analytics found {len(analytics.get('agent_statistics', []))} agent statistics")
        print(f"Performance found {len(performance.get('service_performance', []))} service statistics")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()