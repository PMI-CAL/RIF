-- DPIBS Performance Indexing Strategy
-- Issue #138: Database Schema + Performance Optimization Layer
-- Strategic indexing for <100ms cached query performance

-- Context Performance Indexes
-- Optimized for frequent agent context queries with sub-100ms response times

-- Primary agent context lookup (most frequent query)
DROP INDEX IF EXISTS idx_agent_contexts_hot_lookup;
CREATE INDEX idx_agent_contexts_hot_lookup 
    ON agent_contexts(agent_id, relevance_score DESC, created_at DESC)
    WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP;

-- Context type filtering with performance focus
DROP INDEX IF EXISTS idx_agent_contexts_type_performance;
CREATE INDEX idx_agent_contexts_type_performance 
    ON agent_contexts(context_type, relevance_score DESC, size_estimate)
    INCLUDE (context_data, performance_metrics);

-- Cache key optimization for instant lookups
DROP INDEX IF EXISTS idx_agent_contexts_cache_instant;
CREATE INDEX idx_agent_contexts_cache_instant 
    ON agent_contexts(cache_key)
    WHERE cache_key IS NOT NULL
    INCLUDE (context_data, relevance_score, created_at);

-- Time-based context queries (recent context prioritization)
DROP INDEX IF EXISTS idx_agent_contexts_temporal;
CREATE INDEX idx_agent_contexts_temporal 
    ON agent_contexts(created_at DESC, agent_id, relevance_score DESC)
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';

-- Context Cache Performance Indexes
-- Optimized for high-frequency cache operations

-- Primary cache lookup with expiry filtering
DROP INDEX IF EXISTS idx_context_cache_primary;
CREATE INDEX idx_context_cache_primary 
    ON context_cache(cache_key, expiry_time)
    INCLUDE (cached_result, hit_count, last_accessed);

-- Cache efficiency analysis
DROP INDEX IF EXISTS idx_context_cache_efficiency;
CREATE INDEX idx_context_cache_efficiency 
    ON context_cache(cache_level, hit_count DESC, data_size)
    WHERE expiry_time > CURRENT_TIMESTAMP;

-- Agent-specific cache performance
DROP INDEX IF EXISTS idx_context_cache_agent_perf;
CREATE INDEX idx_context_cache_agent_perf 
    ON context_cache(agent_type, last_accessed DESC, hit_count DESC)
    WHERE agent_type IS NOT NULL;

-- Cache cleanup optimization
DROP INDEX IF EXISTS idx_context_cache_cleanup;
CREATE INDEX idx_context_cache_cleanup 
    ON context_cache(expiry_time, created_at)
    WHERE expiry_time < CURRENT_TIMESTAMP;

-- Context Delivery Performance Indexes
-- Optimized for delivery tracking and performance analysis

-- Agent delivery performance tracking
DROP INDEX IF EXISTS idx_context_deliveries_agent_perf;
CREATE INDEX idx_context_deliveries_agent_perf 
    ON context_deliveries(agent_type, response_time_ms, created_at DESC)
    INCLUDE (cache_hit, quality_score);

-- SLA compliance monitoring (<200ms target)
DROP INDEX IF EXISTS idx_context_deliveries_sla;
CREATE INDEX idx_context_deliveries_sla 
    ON context_deliveries(response_time_ms, created_at DESC)
    WHERE response_time_ms IS NOT NULL
    INCLUDE (agent_type, cache_hit);

-- Quality score analysis
DROP INDEX IF EXISTS idx_context_deliveries_quality;
CREATE INDEX idx_context_deliveries_quality 
    ON context_deliveries(quality_score DESC, feedback_score DESC, created_at DESC)
    WHERE quality_score IS NOT NULL;

-- Issue-based delivery tracking  
DROP INDEX IF EXISTS idx_context_deliveries_issue;
CREATE INDEX idx_context_deliveries_issue 
    ON context_deliveries(issue_number, created_at DESC, response_time_ms)
    WHERE issue_number IS NOT NULL;

-- System Context Performance Indexes
-- Optimized for system understanding queries

-- Version-based system context lookup
DROP INDEX IF EXISTS idx_system_contexts_version_perf;
CREATE INDEX idx_system_contexts_version_perf 
    ON system_contexts(context_version DESC, validation_status, created_at DESC)
    INCLUDE (system_overview, architecture_summary);

-- System context validation status
DROP INDEX IF EXISTS idx_system_contexts_validation;
CREATE INDEX idx_system_contexts_validation 
    ON system_contexts(validation_status, updated_at DESC)
    WHERE validation_status IN ('validated', 'pending')
    INCLUDE (context_version, consistency_hash);

-- Dependency relationship performance indexes
DROP INDEX IF EXISTS idx_component_deps_source_perf;
CREATE INDEX idx_component_deps_source_perf 
    ON component_dependencies(component_name, strength_score DESC, impact_level)
    INCLUDE (depends_on, dependency_type, confidence_level);

-- Dependency target lookup
DROP INDEX IF EXISTS idx_component_deps_target_perf;
CREATE INDEX idx_component_deps_target_perf 
    ON component_dependencies(depends_on, strength_score DESC)
    INCLUDE (component_name, dependency_type, impact_level);

-- High-impact dependency analysis
DROP INDEX IF EXISTS idx_component_deps_impact;
CREATE INDEX idx_component_deps_impact 
    ON component_dependencies(impact_level, strength_score DESC, confidence_level DESC)
    WHERE impact_level IN ('high', 'critical');

-- Benchmarking Performance Indexes
-- Optimized for benchmarking and grading queries

-- Issue-based benchmarking lookup
DROP INDEX IF EXISTS idx_benchmarking_issue_perf;
CREATE INDEX idx_benchmarking_issue_perf 
    ON benchmarking_results(issue_id, created_at DESC, grade_score DESC)
    INCLUDE (grade_level, specification_title);

-- Grade performance analysis
DROP INDEX IF EXISTS idx_benchmarking_grade_perf;
CREATE INDEX idx_benchmarking_grade_perf 
    ON benchmarking_results(grade_score DESC, grade_percentage DESC, created_at DESC)
    INCLUDE (benchmark_category, complexity_assessment);

-- Category-based benchmarking analysis
DROP INDEX IF EXISTS idx_benchmarking_category_perf;
CREATE INDEX idx_benchmarking_category_perf 
    ON benchmarking_results(benchmark_category, grade_level, created_at DESC)
    INCLUDE (grade_score, confidence_level);

-- Benchmarking validation workflow
DROP INDEX IF EXISTS idx_benchmarking_validation;
CREATE INDEX idx_benchmarking_validation 
    ON benchmarking_results(validation_status, updated_at DESC)
    WHERE validation_status IN ('pending', 'needs_review')
    INCLUDE (issue_id, grade_score);

-- Criterion scores performance
DROP INDEX IF EXISTS idx_criterion_scores_perf;
CREATE INDEX idx_criterion_scores_perf 
    ON criterion_scores(benchmark_result_id, weighted_score DESC)
    INCLUDE (criteria_id, achievement_level);

-- Knowledge Integration Performance Indexes
-- Optimized for MCP knowledge server integration

-- Knowledge sync performance
DROP INDEX IF EXISTS idx_knowledge_integration_sync;
CREATE INDEX idx_knowledge_integration_sync 
    ON knowledge_integration(sync_status, last_sync DESC, integration_type)
    INCLUDE (mcp_compatibility, integration_quality);

-- MCP compatibility tracking
DROP INDEX IF EXISTS idx_knowledge_integration_mcp;
CREATE INDEX idx_knowledge_integration_mcp 
    ON knowledge_integration(mcp_compatibility, mcp_server_validated, created_at DESC)
    WHERE mcp_compatibility = TRUE
    INCLUDE (integration_type, sync_status);

-- Knowledge access patterns
DROP INDEX IF EXISTS idx_knowledge_integration_access;
CREATE INDEX idx_knowledge_integration_access 
    ON knowledge_integration(access_frequency DESC, integration_quality DESC)
    INCLUDE (integration_type, knowledge_tags);

-- Evidence Collection Performance Indexes
-- Optimized for evidence item queries

-- Benchmark evidence lookup
DROP INDEX IF EXISTS idx_evidence_items_benchmark;
CREATE INDEX idx_evidence_items_benchmark 
    ON evidence_items(benchmark_result_id, evidence_type, collection_timestamp DESC)
    INCLUDE (reliability_score, validation_status);

-- Evidence validation workflow
DROP INDEX IF EXISTS idx_evidence_items_validation;
CREATE INDEX idx_evidence_items_validation 
    ON evidence_items(validation_status, reliability_score DESC)
    WHERE validation_status IN ('collected', 'validated')
    INCLUDE (evidence_type, collection_timestamp);

-- High-reliability evidence
DROP INDEX IF EXISTS idx_evidence_items_reliability;
CREATE INDEX idx_evidence_items_reliability 
    ON evidence_items(reliability_score DESC, collection_timestamp DESC)
    WHERE reliability_score >= 0.8
    INCLUDE (evidence_type, collection_method);

-- Composite Performance Indexes
-- Multi-table query optimization

-- Agent performance composite
DROP INDEX IF EXISTS idx_agent_context_delivery_composite;
CREATE INDEX idx_agent_context_delivery_composite 
    ON context_deliveries(agent_type, created_at DESC, response_time_ms, cache_hit)
    INCLUDE (quality_score, total_context_size, relevant_items_count);

-- System analysis composite  
DROP INDEX IF EXISTS idx_system_dependency_composite;
CREATE INDEX idx_system_dependency_composite 
    ON component_dependencies(component_name, dependency_type, strength_score DESC)
    INCLUDE (depends_on, impact_level, last_verified);

-- Benchmarking analysis composite
DROP INDEX IF EXISTS idx_benchmarking_analysis_composite;
CREATE INDEX idx_benchmarking_analysis_composite 
    ON benchmarking_results(benchmark_category, created_at DESC, grade_score DESC)
    INCLUDE (issue_id, complexity_assessment, validation_status);

-- Partial Indexes for Hot Data
-- Focus on most frequently accessed data

-- Recent context deliveries (last 7 days)
DROP INDEX IF EXISTS idx_recent_context_deliveries;
CREATE INDEX idx_recent_context_deliveries 
    ON context_deliveries(agent_type, response_time_ms, cache_hit)
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
    INCLUDE (quality_score, created_at);

-- Active context cache entries
DROP INDEX IF EXISTS idx_active_context_cache;
CREATE INDEX idx_active_context_cache 
    ON context_cache(agent_type, hit_count DESC)
    WHERE expiry_time > CURRENT_TIMESTAMP
    INCLUDE (cache_key, last_accessed, data_size);

-- Recent benchmarking results (last 30 days)
DROP INDEX IF EXISTS idx_recent_benchmarking;
CREATE INDEX idx_recent_benchmarking 
    ON benchmarking_results(issue_id, grade_score DESC)
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
    INCLUDE (benchmark_category, grade_level);

-- High-strength dependencies (optimization focus)
DROP INDEX IF EXISTS idx_strong_dependencies;
CREATE INDEX idx_strong_dependencies 
    ON component_dependencies(component_name, depends_on)
    WHERE strength_score >= 0.7
    INCLUDE (dependency_type, impact_level, strength_score);

-- Expression Indexes for Calculated Fields
-- Optimize computed queries

-- Context window utilization efficiency
DROP INDEX IF EXISTS idx_context_utilization_efficiency;
CREATE INDEX idx_context_utilization_efficiency 
    ON context_deliveries(
        (CASE WHEN context_window_utilization BETWEEN 0.7 AND 0.9 
         THEN 'optimal' 
         WHEN context_window_utilization < 0.7 
         THEN 'underutilized'
         ELSE 'overutilized' END),
        response_time_ms
    );

-- Performance rating expression
DROP INDEX IF EXISTS idx_delivery_performance_rating;
CREATE INDEX idx_delivery_performance_rating 
    ON context_deliveries(
        (CASE 
         WHEN response_time_ms < 100 THEN 'excellent'
         WHEN response_time_ms < 200 THEN 'good' 
         WHEN response_time_ms < 500 THEN 'acceptable'
         ELSE 'poor' END),
        agent_type,
        created_at DESC
    );

-- GIN Indexes for JSONB Fields
-- Optimize JSON field queries

-- Context data search optimization
DROP INDEX IF EXISTS idx_agent_contexts_data_gin;
CREATE INDEX idx_agent_contexts_data_gin 
    ON agent_contexts USING GIN(context_data);

-- Performance metrics search
DROP INDEX IF EXISTS idx_context_deliveries_perf_gin;
CREATE INDEX idx_context_deliveries_perf_gin 
    ON context_deliveries USING GIN(performance_breakdown);

-- Evidence data search optimization
DROP INDEX IF EXISTS idx_evidence_data_gin;
CREATE INDEX idx_evidence_data_gin 
    ON evidence_items USING GIN(evidence_data);

-- Knowledge data search optimization
DROP INDEX IF EXISTS idx_knowledge_data_gin;
CREATE INDEX idx_knowledge_data_gin 
    ON knowledge_integration USING GIN(knowledge_data);

-- Index Usage Monitoring Views
-- Track index effectiveness

CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    CASE WHEN idx_scan > 0 
         THEN round(idx_tup_fetch::DECIMAL / idx_scan, 2)
         ELSE 0 END as avg_tuples_per_scan
FROM pg_stat_user_indexes 
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

CREATE OR REPLACE VIEW table_performance_stats AS
SELECT 
    schemaname,
    relname as tablename,
    seq_scan as sequential_scans,
    seq_tup_read as seq_tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as index_tuples_fetched,
    CASE WHEN (seq_scan + idx_scan) > 0
         THEN round((idx_scan::DECIMAL / (seq_scan + idx_scan)) * 100, 2)
         ELSE 0 END as index_usage_percentage
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY seq_scan DESC;

-- Performance optimization functions

CREATE OR REPLACE FUNCTION analyze_index_performance() 
RETURNS TABLE(
    recommendation_type VARCHAR,
    table_name VARCHAR, 
    index_name VARCHAR,
    issue_description TEXT,
    suggested_action TEXT
) AS $$
BEGIN
    -- Identify unused indexes
    RETURN QUERY
    SELECT 
        'unused_index'::VARCHAR,
        sui.tablename::VARCHAR,
        sui.indexname::VARCHAR,
        'Index has very low usage'::TEXT,
        CASE WHEN sui.idx_scan < 10 
             THEN 'Consider dropping this index'
             ELSE 'Monitor index usage patterns' END::TEXT
    FROM pg_stat_user_indexes sui
    WHERE sui.idx_scan < 100
    AND sui.indexname NOT LIKE '%_pkey'
    ORDER BY sui.idx_scan;
    
    -- Identify tables with high sequential scan ratio
    RETURN QUERY  
    SELECT 
        'high_seq_scan'::VARCHAR,
        sut.relname::VARCHAR,
        ''::VARCHAR,
        format('Sequential scans: %s, Index scans: %s', sut.seq_scan, sut.idx_scan)::TEXT,
        'Consider adding indexes for frequent query patterns'::TEXT
    FROM pg_stat_user_tables sut
    WHERE sut.seq_scan > sut.idx_scan * 2
    AND sut.seq_scan > 1000
    ORDER BY sut.seq_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- Index maintenance function
CREATE OR REPLACE FUNCTION maintain_performance_indexes() 
RETURNS JSONB AS $$
DECLARE
    maintenance_start TIMESTAMP;
    maintenance_results JSONB := '{}'::JSONB;
    reindex_count INTEGER := 0;
BEGIN
    maintenance_start := CURRENT_TIMESTAMP;
    
    -- Reindex critical performance tables
    REINDEX TABLE agent_contexts;
    REINDEX TABLE context_cache;  
    REINDEX TABLE context_deliveries;
    reindex_count := reindex_count + 3;
    
    -- Update table statistics
    ANALYZE agent_contexts;
    ANALYZE context_cache;
    ANALYZE context_deliveries;
    ANALYZE benchmarking_results;
    ANALYZE component_dependencies;
    
    maintenance_results := jsonb_build_object(
        'maintenance_timestamp', maintenance_start,
        'duration_ms', EXTRACT(MILLISECONDS FROM CURRENT_TIMESTAMP - maintenance_start),
        'reindexed_tables', reindex_count,
        'analyzed_tables', 5,
        'status', 'completed'
    );
    
    RETURN maintenance_results;
END;
$$ LANGUAGE plpgsql;

-- Query performance monitoring
CREATE OR REPLACE FUNCTION monitor_query_performance(
    p_threshold_ms INTEGER DEFAULT 1000
) 
RETURNS TABLE(
    query_type VARCHAR,
    avg_duration_ms DECIMAL,
    max_duration_ms DECIMAL,
    call_count BIGINT,
    total_time_ms DECIMAL
) AS $$
BEGIN
    -- This would integrate with pg_stat_statements in production
    -- Placeholder for query performance monitoring
    
    RETURN QUERY
    SELECT 
        'context_optimization'::VARCHAR,
        150.5::DECIMAL,
        850.2::DECIMAL, 
        1250::BIGINT,
        188125.0::DECIMAL
    UNION ALL
    SELECT 
        'benchmarking_analysis'::VARCHAR,
        75.3::DECIMAL,
        420.1::DECIMAL,
        890::BIGINT,
        67017.0::DECIMAL;
END;
$$ LANGUAGE plpgsql;