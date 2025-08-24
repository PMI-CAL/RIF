-- DPIBS System Understanding Schema  
-- Issue #138: Database Schema + Performance Optimization Layer
-- Live system context and dependency relationship tracking

-- Live system context with versioning and change tracking
CREATE TABLE IF NOT EXISTS system_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_version INTEGER NOT NULL,
    system_overview TEXT NOT NULL,
    purpose TEXT NOT NULL,
    design_goals JSONB NOT NULL,
    constraints JSONB NOT NULL,
    dependencies JSONB NOT NULL,
    architecture_summary TEXT NOT NULL,
    change_events JSONB,
    consistency_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_status VARCHAR(20) DEFAULT 'pending' CHECK (
        validation_status IN ('pending', 'validated', 'stale', 'invalid')
    ),
    performance_metrics JSONB,
    source_analysis JSONB
);

-- System context versioning and performance indexes
CREATE INDEX IF NOT EXISTS idx_system_contexts_version 
    ON system_contexts(context_version DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_contexts_hash 
    ON system_contexts(consistency_hash, validation_status);
CREATE INDEX IF NOT EXISTS idx_system_contexts_expires 
    ON system_contexts(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_system_contexts_status 
    ON system_contexts(validation_status, updated_at DESC);

-- Component dependencies and relationships with strength scoring
CREATE TABLE IF NOT EXISTS component_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_name VARCHAR(255) NOT NULL,
    depends_on VARCHAR(255) NOT NULL,
    dependency_type VARCHAR(50) NOT NULL CHECK (dependency_type IN (
        'import', 'module', 'function', 'class', 'api', 'database', 'service', 'configuration'
    )),
    strength_score DECIMAL(3,2) NOT NULL DEFAULT 0.5 CHECK (
        strength_score >= 0 AND strength_score <= 1
    ),
    confidence_level DECIMAL(3,2) DEFAULT 0.8 CHECK (
        confidence_level >= 0 AND confidence_level <= 1
    ),
    impact_level VARCHAR(20) DEFAULT 'medium' CHECK (
        impact_level IN ('low', 'medium', 'high', 'critical')
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verification_method VARCHAR(50),
    metadata JSONB,
    source_files TEXT[],
    change_frequency INTEGER DEFAULT 0
);

-- High-performance dependency relationship indexes
CREATE INDEX IF NOT EXISTS idx_component_dependencies_component 
    ON component_dependencies(component_name, strength_score DESC);
CREATE INDEX IF NOT EXISTS idx_component_dependencies_depends 
    ON component_dependencies(depends_on, impact_level, strength_score DESC);
CREATE INDEX IF NOT EXISTS idx_component_dependencies_type_strength 
    ON component_dependencies(dependency_type, strength_score DESC, confidence_level DESC);
CREATE INDEX IF NOT EXISTS idx_component_dependencies_verification 
    ON component_dependencies(last_verified DESC) WHERE verification_method IS NOT NULL;

-- Dependency graph paths for transitive dependency analysis
CREATE TABLE IF NOT EXISTS dependency_paths (
    path_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_component VARCHAR(255) NOT NULL,
    target_component VARCHAR(255) NOT NULL,
    path_length INTEGER NOT NULL CHECK (path_length > 0),
    path_components TEXT[] NOT NULL,
    path_strength DECIMAL(3,2) NOT NULL CHECK (path_strength >= 0 AND path_strength <= 1),
    is_circular BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    computation_time_ms DECIMAL(8,2)
);

-- Dependency path performance indexes
CREATE INDEX IF NOT EXISTS idx_dependency_paths_source_target 
    ON dependency_paths(source_component, target_component, path_length);
CREATE INDEX IF NOT EXISTS idx_dependency_paths_circular 
    ON dependency_paths(is_circular, path_strength DESC) WHERE is_circular = TRUE;
CREATE INDEX IF NOT EXISTS idx_dependency_paths_strength 
    ON dependency_paths(path_strength DESC, path_length);

-- System architecture snapshots for change tracking
CREATE TABLE IF NOT EXISTS architecture_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    system_context_id UUID REFERENCES system_contexts(id),
    component_count INTEGER NOT NULL,
    dependency_count INTEGER NOT NULL,
    complexity_metrics JSONB NOT NULL,
    architectural_patterns JSONB,
    quality_indicators JSONB,
    change_summary JSONB,
    performance_baseline JSONB,
    created_by VARCHAR(100),
    validation_results JSONB
);

-- Architecture snapshot tracking indexes
CREATE INDEX IF NOT EXISTS idx_architecture_snapshots_timestamp 
    ON architecture_snapshots(snapshot_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_architecture_snapshots_context 
    ON architecture_snapshots(system_context_id) WHERE system_context_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_architecture_snapshots_complexity 
    ON architecture_snapshots((complexity_metrics->>'overall_score')::DECIMAL DESC);

-- Live system monitoring events
CREATE TABLE IF NOT EXISTS system_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'dependency_added', 'dependency_removed', 'component_added', 'component_removed',
        'architecture_change', 'performance_change', 'complexity_change', 'validation_update'
    )),
    event_priority INTEGER NOT NULL DEFAULT 5 CHECK (event_priority BETWEEN 1 AND 10),
    source_component VARCHAR(255),
    target_component VARCHAR(255),
    event_data JSONB NOT NULL,
    impact_assessment JSONB,
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (
        processing_status IN ('pending', 'processing', 'completed', 'failed', 'ignored')
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    processing_time_ms DECIMAL(8,2),
    correlation_id UUID,
    automated_response BOOLEAN DEFAULT FALSE
);

-- System event processing indexes  
CREATE INDEX IF NOT EXISTS idx_system_events_type_priority 
    ON system_events(event_type, event_priority DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_status_time 
    ON system_events(processing_status, created_at) 
    WHERE processing_status IN ('pending', 'processing');
CREATE INDEX IF NOT EXISTS idx_system_events_component 
    ON system_events(source_component, target_component) 
    WHERE source_component IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_system_events_correlation 
    ON system_events(correlation_id) WHERE correlation_id IS NOT NULL;

-- Performance views for system understanding
CREATE OR REPLACE VIEW system_dependency_summary AS
SELECT 
    component_name,
    COUNT(*) as total_dependencies,
    COUNT(CASE WHEN dependency_type = 'import' THEN 1 END) as import_dependencies,
    COUNT(CASE WHEN dependency_type = 'api' THEN 1 END) as api_dependencies,
    COUNT(CASE WHEN dependency_type = 'database' THEN 1 END) as database_dependencies,
    AVG(strength_score) as avg_dependency_strength,
    MAX(strength_score) as max_dependency_strength,
    COUNT(CASE WHEN impact_level = 'critical' THEN 1 END) as critical_dependencies,
    COUNT(CASE WHEN impact_level = 'high' THEN 1 END) as high_impact_dependencies,
    AVG(confidence_level) as avg_confidence,
    MAX(last_verified) as last_dependency_verification
FROM component_dependencies 
GROUP BY component_name;

CREATE OR REPLACE VIEW dependency_complexity_analysis AS
SELECT 
    source_component,
    target_component,
    COUNT(*) as path_count,
    MIN(path_length) as shortest_path,
    MAX(path_length) as longest_path,
    AVG(path_length) as avg_path_length,
    MAX(path_strength) as strongest_path,
    COUNT(CASE WHEN is_circular THEN 1 END) as circular_paths,
    AVG(computation_time_ms) as avg_computation_time
FROM dependency_paths 
GROUP BY source_component, target_component;

CREATE OR REPLACE VIEW system_context_history AS
SELECT 
    context_version,
    system_overview,
    validation_status,
    created_at,
    updated_at,
    consistency_hash,
    LAG(consistency_hash) OVER (ORDER BY context_version) as previous_hash,
    CASE WHEN LAG(consistency_hash) OVER (ORDER BY context_version) != consistency_hash 
         THEN TRUE ELSE FALSE END as has_changes,
    (performance_metrics->>'analysis_time_ms')::DECIMAL as analysis_time_ms
FROM system_contexts 
ORDER BY context_version DESC;

-- System performance optimization functions
CREATE OR REPLACE FUNCTION analyze_dependency_graph(
    p_max_depth INTEGER DEFAULT 10,
    p_min_strength DECIMAL DEFAULT 0.1
) 
RETURNS TABLE(
    analysis_summary JSONB,
    performance_metrics JSONB,
    recommendations TEXT[]
) AS $$
DECLARE
    start_time TIMESTAMP;
    computation_time DECIMAL;
    total_components INTEGER;
    total_dependencies INTEGER;
    circular_count INTEGER;
    max_depth_reached INTEGER;
    recommendations TEXT[] := ARRAY[]::TEXT[];
BEGIN
    start_time := CURRENT_TIMESTAMP;
    
    -- Get basic counts
    SELECT COUNT(DISTINCT component_name) INTO total_components 
    FROM component_dependencies;
    
    SELECT COUNT(*) INTO total_dependencies 
    FROM component_dependencies 
    WHERE strength_score >= p_min_strength;
    
    -- Count circular dependencies
    SELECT COUNT(*) INTO circular_count 
    FROM dependency_paths 
    WHERE is_circular = TRUE;
    
    -- Find maximum dependency depth
    SELECT COALESCE(MAX(path_length), 0) INTO max_depth_reached 
    FROM dependency_paths;
    
    -- Generate recommendations
    IF circular_count > 0 THEN
        recommendations := array_append(recommendations, 
            format('Found %s circular dependencies - consider refactoring', circular_count));
    END IF;
    
    IF max_depth_reached > p_max_depth THEN
        recommendations := array_append(recommendations,
            format('Dependency depth %s exceeds recommended max %s', max_depth_reached, p_max_depth));
    END IF;
    
    computation_time := EXTRACT(MILLISECONDS FROM CURRENT_TIMESTAMP - start_time);
    
    RETURN QUERY SELECT 
        jsonb_build_object(
            'total_components', total_components,
            'total_dependencies', total_dependencies,
            'circular_dependencies', circular_count,
            'max_dependency_depth', max_depth_reached,
            'analysis_timestamp', CURRENT_TIMESTAMP
        ) as analysis_summary,
        jsonb_build_object(
            'computation_time_ms', computation_time,
            'components_per_second', CASE WHEN computation_time > 0 
                                         THEN total_components / (computation_time / 1000) 
                                         ELSE 0 END,
            'dependencies_analyzed', total_dependencies
        ) as performance_metrics,
        recommendations;
END;
$$ LANGUAGE plpgsql;

-- System context validation function
CREATE OR REPLACE FUNCTION validate_system_context(
    p_context_id UUID
) 
RETURNS JSONB AS $$
DECLARE
    context_record system_contexts%ROWTYPE;
    validation_results JSONB := '{}'::JSONB;
    dependency_count INTEGER;
    consistency_issues INTEGER := 0;
BEGIN
    -- Get context record
    SELECT * INTO context_record FROM system_contexts WHERE id = p_context_id;
    
    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'Context not found', 'context_id', p_context_id);
    END IF;
    
    -- Validate dependency data consistency
    SELECT COUNT(*) INTO dependency_count 
    FROM component_dependencies 
    WHERE last_verified > context_record.created_at - INTERVAL '1 hour';
    
    -- Check for stale dependencies
    SELECT COUNT(*) INTO consistency_issues
    FROM component_dependencies 
    WHERE last_verified < CURRENT_TIMESTAMP - INTERVAL '24 hours';
    
    -- Build validation results
    validation_results := jsonb_build_object(
        'context_id', p_context_id,
        'validation_timestamp', CURRENT_TIMESTAMP,
        'context_age_hours', EXTRACT(HOURS FROM CURRENT_TIMESTAMP - context_record.created_at),
        'recent_dependency_updates', dependency_count,
        'stale_dependencies', consistency_issues,
        'validation_status', CASE 
            WHEN consistency_issues = 0 AND dependency_count > 0 THEN 'validated'
            WHEN consistency_issues > 0 THEN 'stale'
            ELSE 'needs_refresh'
        END
    );
    
    -- Update context validation status
    UPDATE system_contexts 
    SET validation_status = (validation_results->>'validation_status')::VARCHAR,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_context_id;
    
    RETURN validation_results;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for system data
CREATE OR REPLACE FUNCTION cleanup_system_data(
    p_days_to_keep INTEGER DEFAULT 90
) 
RETURNS JSONB AS $$
DECLARE
    cutoff_date TIMESTAMP;
    deleted_contexts INTEGER;
    deleted_snapshots INTEGER;
    deleted_events INTEGER;
    cleanup_results JSONB;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (p_days_to_keep || ' days')::INTERVAL;
    
    -- Clean up old system contexts (keep latest version of each)
    WITH latest_contexts AS (
        SELECT MAX(context_version) as max_version
        FROM system_contexts
        WHERE validation_status != 'invalid'
    )
    DELETE FROM system_contexts 
    WHERE created_at < cutoff_date 
    AND context_version NOT IN (SELECT max_version FROM latest_contexts);
    GET DIAGNOSTICS deleted_contexts = ROW_COUNT;
    
    -- Clean up old architecture snapshots
    DELETE FROM architecture_snapshots 
    WHERE snapshot_timestamp < cutoff_date;
    GET DIAGNOSTICS deleted_snapshots = ROW_COUNT;
    
    -- Clean up processed system events
    DELETE FROM system_events 
    WHERE created_at < cutoff_date 
    AND processing_status = 'completed';
    GET DIAGNOSTICS deleted_events = ROW_COUNT;
    
    cleanup_results := jsonb_build_object(
        'cleanup_timestamp', CURRENT_TIMESTAMP,
        'cutoff_date', cutoff_date,
        'deleted_contexts', deleted_contexts,
        'deleted_snapshots', deleted_snapshots,
        'deleted_events', deleted_events
    );
    
    RETURN cleanup_results;
END;
$$ LANGUAGE plpgsql;