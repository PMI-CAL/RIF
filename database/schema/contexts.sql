-- DPIBS Context Storage Schema
-- Issue #138: Database Schema + Performance Optimization Layer
-- Context optimization metadata tables with performance focus

-- Agent context storage with role-specific optimization
CREATE TABLE IF NOT EXISTS agent_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) NOT NULL,
    context_type VARCHAR(30) NOT NULL,
    relevance_score DECIMAL(3,2) CHECK (relevance_score >= 0 AND relevance_score <= 1),
    context_data JSONB NOT NULL,
    size_estimate INTEGER NOT NULL CHECK (size_estimate > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cache_key VARCHAR(255),
    performance_metrics JSONB
);

-- Strategic indexing for agent context queries
CREATE INDEX IF NOT EXISTS idx_agent_contexts_agent_relevance 
    ON agent_contexts(agent_id, relevance_score DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_contexts_type 
    ON agent_contexts(context_type, expires_at);
CREATE INDEX IF NOT EXISTS idx_agent_contexts_cache 
    ON agent_contexts(cache_key) WHERE cache_key IS NOT NULL;

-- Context query performance and caching metadata
CREATE TABLE IF NOT EXISTS context_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    cache_level VARCHAR(10) NOT NULL CHECK (cache_level IN ('L1', 'L2', 'L3')),
    cached_result JSONB NOT NULL,
    data_size INTEGER NOT NULL,
    expiry_time TIMESTAMP NOT NULL,
    hit_count INTEGER DEFAULT 0,
    miss_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_type VARCHAR(50),
    invalidation_reason VARCHAR(100),
    performance_data JSONB
);

-- Performance-optimized cache indexes
CREATE INDEX IF NOT EXISTS idx_context_cache_expiry 
    ON context_cache(expiry_time) WHERE expiry_time > CURRENT_TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_context_cache_agent_level 
    ON context_cache(agent_type, cache_level, last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_context_cache_performance 
    ON context_cache(hit_count DESC, data_size) WHERE hit_count > 0;

-- Context delivery tracking with performance metrics
CREATE TABLE IF NOT EXISTS context_deliveries (
    delivery_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    context_window_utilization DECIMAL(3,2) CHECK (context_window_utilization >= 0 AND context_window_utilization <= 1),
    total_context_size INTEGER NOT NULL,
    relevant_items_count INTEGER NOT NULL,
    response_time_ms DECIMAL(8,2) NOT NULL CHECK (response_time_ms >= 0),
    cache_hit BOOLEAN DEFAULT FALSE,
    source_services JSONB NOT NULL,
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0 AND quality_score <= 1),
    feedback_score DECIMAL(3,2) CHECK (feedback_score >= 0 AND feedback_score <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    issue_number INTEGER,
    optimization_factors JSONB,
    performance_breakdown JSONB
);

-- Performance tracking indexes
CREATE INDEX IF NOT EXISTS idx_context_deliveries_agent_time 
    ON context_deliveries(agent_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_context_deliveries_response_time 
    ON context_deliveries(response_time_ms) WHERE response_time_ms < 200;
CREATE INDEX IF NOT EXISTS idx_context_deliveries_cache_performance 
    ON context_deliveries(cache_hit, response_time_ms);
CREATE INDEX IF NOT EXISTS idx_context_deliveries_quality 
    ON context_deliveries(quality_score DESC, feedback_score DESC) 
    WHERE quality_score IS NOT NULL;

-- Multi-factor relevance scoring metadata
CREATE TABLE IF NOT EXISTS relevance_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_id UUID REFERENCES agent_contexts(id) ON DELETE CASCADE,
    factor_type VARCHAR(30) NOT NULL CHECK (factor_type IN (
        'agent_specific', 'task_relevance', 'freshness', 'usage_success', 'semantic_similarity'
    )),
    factor_weight DECIMAL(3,2) NOT NULL CHECK (factor_weight >= 0 AND factor_weight <= 1),
    factor_score DECIMAL(3,2) NOT NULL CHECK (factor_score >= 0 AND factor_score <= 1),
    calculation_method VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relevance factor optimization index
CREATE INDEX IF NOT EXISTS idx_relevance_factors_context_type 
    ON relevance_factors(context_id, factor_type, factor_score DESC);

-- Context optimization sessions tracking
CREATE TABLE IF NOT EXISTS context_optimization_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(50) NOT NULL,
    task_context JSONB NOT NULL,
    issue_number INTEGER,
    original_context_size INTEGER NOT NULL,
    optimized_context_size INTEGER NOT NULL,
    optimization_ratio DECIMAL(4,3) GENERATED ALWAYS AS (
        CASE WHEN original_context_size > 0 
        THEN CAST(optimized_context_size AS DECIMAL) / original_context_size 
        ELSE 1 END
    ) STORED,
    relevance_score DECIMAL(3,2) NOT NULL,
    optimization_time_ms DECIMAL(8,2) NOT NULL,
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    context_items JSONB NOT NULL,
    performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_notes TEXT
);

-- Session performance and analysis indexes
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_agent_time 
    ON context_optimization_sessions(agent_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_performance 
    ON context_optimization_sessions(optimization_time_ms, optimization_ratio);
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_issue 
    ON context_optimization_sessions(issue_number) WHERE issue_number IS NOT NULL;

-- Performance views for monitoring
CREATE OR REPLACE VIEW context_performance_summary AS
SELECT 
    agent_type,
    COUNT(*) as total_deliveries,
    AVG(response_time_ms) as avg_response_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time_ms,
    AVG(context_window_utilization) as avg_window_utilization,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
    SUM(CASE WHEN NOT cache_hit THEN 1 ELSE 0 END) as cache_misses,
    CAST(SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS DECIMAL) / COUNT(*) as cache_hit_rate,
    AVG(quality_score) as avg_quality_score,
    COUNT(CASE WHEN response_time_ms < 200 THEN 1 END) as under_200ms_count,
    CAST(COUNT(CASE WHEN response_time_ms < 200 THEN 1 END) AS DECIMAL) / COUNT(*) as sla_compliance_rate
FROM context_deliveries 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY agent_type;

CREATE OR REPLACE VIEW cache_efficiency_summary AS
SELECT 
    cache_level,
    agent_type,
    COUNT(*) as total_entries,
    SUM(hit_count) as total_hits,
    SUM(miss_count) as total_misses,
    CASE WHEN (SUM(hit_count) + SUM(miss_count)) > 0 
         THEN CAST(SUM(hit_count) AS DECIMAL) / (SUM(hit_count) + SUM(miss_count))
         ELSE 0 END as hit_ratio,
    AVG(data_size) as avg_data_size,
    COUNT(CASE WHEN expiry_time > CURRENT_TIMESTAMP THEN 1 END) as active_entries,
    COUNT(CASE WHEN expiry_time <= CURRENT_TIMESTAMP THEN 1 END) as expired_entries
FROM context_cache 
GROUP BY cache_level, agent_type;

-- Cleanup procedures for performance maintenance
CREATE OR REPLACE FUNCTION cleanup_expired_contexts() 
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean up expired context cache entries
    DELETE FROM context_cache WHERE expiry_time < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old agent contexts (older than 30 days)
    DELETE FROM agent_contexts 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up old delivery records (older than 90 days)  
    DELETE FROM context_deliveries 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Clean up old optimization sessions (older than 90 days)
    DELETE FROM context_optimization_sessions
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Performance optimization procedures
CREATE OR REPLACE FUNCTION optimize_context_indexes() 
RETURNS VOID AS $$
BEGIN
    -- Analyze tables for query optimizer
    ANALYZE agent_contexts;
    ANALYZE context_cache;
    ANALYZE context_deliveries;
    ANALYZE context_optimization_sessions;
    ANALYZE relevance_factors;
    
    -- Recompute statistics for performance views
    REFRESH MATERIALIZED VIEW IF EXISTS context_performance_stats;
END;
$$ LANGUAGE plpgsql;

-- Context cache invalidation function
CREATE OR REPLACE FUNCTION invalidate_context_cache(
    p_agent_type VARCHAR(50) DEFAULT NULL,
    p_context_type VARCHAR(30) DEFAULT NULL,
    p_reason VARCHAR(100) DEFAULT 'manual_invalidation'
) 
RETURNS INTEGER AS $$
DECLARE
    invalidated_count INTEGER;
BEGIN
    UPDATE context_cache 
    SET expiry_time = CURRENT_TIMESTAMP,
        invalidation_reason = p_reason,
        last_accessed = CURRENT_TIMESTAMP
    WHERE (p_agent_type IS NULL OR agent_type = p_agent_type)
    AND expiry_time > CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS invalidated_count = ROW_COUNT;
    RETURN invalidated_count;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring trigger
CREATE OR REPLACE FUNCTION update_context_access_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update access count and timestamp for agent contexts
    IF TG_TABLE_NAME = 'agent_contexts' THEN
        NEW.access_count = COALESCE(OLD.access_count, 0) + 1;
        NEW.last_accessed = CURRENT_TIMESTAMP;
    END IF;
    
    -- Update cache statistics  
    IF TG_TABLE_NAME = 'context_cache' THEN
        NEW.last_accessed = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for access tracking
DROP TRIGGER IF EXISTS tr_agent_contexts_access ON agent_contexts;
CREATE TRIGGER tr_agent_contexts_access
    BEFORE UPDATE ON agent_contexts
    FOR EACH ROW 
    EXECUTE FUNCTION update_context_access_stats();

DROP TRIGGER IF EXISTS tr_context_cache_access ON context_cache;  
CREATE TRIGGER tr_context_cache_access
    BEFORE UPDATE ON context_cache
    FOR EACH ROW 
    EXECUTE FUNCTION update_context_access_stats();