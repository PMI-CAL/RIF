-- DPIBS Benchmarking and Grading Schema
-- Issue #138: Database Schema + Performance Optimization Layer  
-- Design specification benchmarking results storage with analysis tracking

-- Design specification benchmarking results with evidence collection
CREATE TABLE IF NOT EXISTS benchmarking_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id INTEGER NOT NULL,
    specification_title VARCHAR(500) NOT NULL,
    specification_data JSONB NOT NULL,
    analysis_results JSONB NOT NULL,
    grade_score DECIMAL(5,2) NOT NULL CHECK (grade_score >= 0 AND grade_score <= 100),
    max_possible_score DECIMAL(5,2) DEFAULT 100,
    grade_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN max_possible_score > 0 
        THEN (grade_score / max_possible_score) * 100 
        ELSE 0 END
    ) STORED,
    grade_level VARCHAR(2) GENERATED ALWAYS AS (
        CASE 
            WHEN grade_percentage >= 97 THEN 'A+'
            WHEN grade_percentage >= 93 THEN 'A'
            WHEN grade_percentage >= 90 THEN 'A-'
            WHEN grade_percentage >= 87 THEN 'B+'
            WHEN grade_percentage >= 83 THEN 'B'
            WHEN grade_percentage >= 80 THEN 'B-'
            WHEN grade_percentage >= 77 THEN 'C+'
            WHEN grade_percentage >= 73 THEN 'C'
            WHEN grade_percentage >= 70 THEN 'C-'
            WHEN grade_percentage >= 65 THEN 'D'
            ELSE 'F'
        END
    ) STORED,
    evidence_collection JSONB NOT NULL,
    confidence_level DECIMAL(3,2) DEFAULT 0.8 CHECK (confidence_level >= 0 AND confidence_level <= 1),
    benchmark_category VARCHAR(100) NOT NULL,
    complexity_assessment VARCHAR(20) CHECK (complexity_assessment IN ('low', 'medium', 'high', 'very-high')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_duration_ms DECIMAL(10,2),
    reviewer_notes TEXT,
    validation_status VARCHAR(20) DEFAULT 'pending' CHECK (
        validation_status IN ('pending', 'validated', 'needs_review', 'disputed', 'archived')
    ),
    metadata JSONB
);

-- High-performance benchmarking indexes
CREATE INDEX IF NOT EXISTS idx_benchmarking_results_issue 
    ON benchmarking_results(issue_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmarking_results_grade 
    ON benchmarking_results(grade_score DESC, grade_percentage DESC);
CREATE INDEX IF NOT EXISTS idx_benchmarking_results_category_grade 
    ON benchmarking_results(benchmark_category, grade_level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmarking_results_complexity 
    ON benchmarking_results(complexity_assessment, grade_score DESC) 
    WHERE complexity_assessment IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_benchmarking_results_validation 
    ON benchmarking_results(validation_status, updated_at DESC);

-- Benchmarking criteria and scoring rubrics
CREATE TABLE IF NOT EXISTS benchmarking_criteria (
    criteria_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    criteria_name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    weight DECIMAL(3,2) NOT NULL DEFAULT 1.0 CHECK (weight > 0 AND weight <= 10),
    scoring_method VARCHAR(50) NOT NULL CHECK (scoring_method IN (
        'binary', 'scale', 'percentage', 'rubric', 'automated', 'manual'
    )),
    scoring_rubric JSONB,
    automation_script TEXT,
    minimum_threshold DECIMAL(5,2),
    target_threshold DECIMAL(5,2),
    excellence_threshold DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1
);

-- Benchmarking criteria performance indexes
CREATE INDEX IF NOT EXISTS idx_benchmarking_criteria_category_weight 
    ON benchmarking_criteria(category, weight DESC, active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_benchmarking_criteria_method 
    ON benchmarking_criteria(scoring_method, active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_benchmarking_criteria_version 
    ON benchmarking_criteria(criteria_name, version DESC);

-- Individual criterion scores for detailed analysis
CREATE TABLE IF NOT EXISTS criterion_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_result_id UUID REFERENCES benchmarking_results(id) ON DELETE CASCADE,
    criteria_id UUID REFERENCES benchmarking_criteria(criteria_id) ON DELETE RESTRICT,
    raw_score DECIMAL(8,2) NOT NULL,
    weighted_score DECIMAL(8,2) NOT NULL,
    max_possible_score DECIMAL(8,2) NOT NULL,
    score_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN max_possible_score > 0 
        THEN (raw_score / max_possible_score) * 100 
        ELSE 0 END
    ) STORED,
    achievement_level VARCHAR(20) GENERATED ALWAYS AS (
        CASE 
            WHEN score_percentage >= 90 THEN 'excellent'
            WHEN score_percentage >= 80 THEN 'good'
            WHEN score_percentage >= 70 THEN 'acceptable'
            WHEN score_percentage >= 60 THEN 'needs_improvement'
            ELSE 'inadequate'
        END
    ) STORED,
    evidence_data JSONB,
    scoring_rationale TEXT,
    automated_score BOOLEAN DEFAULT FALSE,
    reviewer_adjustments JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criterion scores analysis indexes
CREATE INDEX IF NOT EXISTS idx_criterion_scores_benchmark 
    ON criterion_scores(benchmark_result_id, weighted_score DESC);
CREATE INDEX IF NOT EXISTS idx_criterion_scores_criteria 
    ON criterion_scores(criteria_id, score_percentage DESC);
CREATE INDEX IF NOT EXISTS idx_criterion_scores_achievement 
    ON criterion_scores(achievement_level, automated_score);

-- Knowledge integration with MCP compatibility tracking
CREATE TABLE IF NOT EXISTS knowledge_integration (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_type VARCHAR(50) NOT NULL CHECK (integration_type IN (
        'pattern_learning', 'decision_capture', 'best_practices', 'failure_analysis',
        'performance_baseline', 'architectural_insight', 'quality_metric'
    )),
    source_benchmark_id UUID REFERENCES benchmarking_results(id) ON DELETE SET NULL,
    knowledge_data JSONB NOT NULL,
    integration_quality DECIMAL(3,2) DEFAULT 0.8 CHECK (integration_quality >= 0 AND integration_quality <= 1),
    mcp_compatibility BOOLEAN DEFAULT TRUE,
    mcp_server_validated VARCHAR(100),
    sync_status VARCHAR(30) DEFAULT 'pending' CHECK (
        sync_status IN ('pending', 'syncing', 'synced', 'failed', 'conflict', 'archived')
    ),
    sync_attempts INTEGER DEFAULT 0,
    last_sync TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    correlation_id UUID,
    retention_policy VARCHAR(50) DEFAULT 'standard',
    access_frequency INTEGER DEFAULT 0,
    knowledge_tags TEXT[]
);

-- Knowledge integration performance indexes  
CREATE INDEX IF NOT EXISTS idx_knowledge_integration_type_sync 
    ON knowledge_integration(integration_type, sync_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_integration_mcp 
    ON knowledge_integration(mcp_compatibility, mcp_server_validated) 
    WHERE mcp_compatibility = TRUE;
CREATE INDEX IF NOT EXISTS idx_knowledge_integration_benchmark 
    ON knowledge_integration(source_benchmark_id) WHERE source_benchmark_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_integration_tags 
    ON knowledge_integration USING GIN(knowledge_tags) WHERE knowledge_tags IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_integration_quality 
    ON knowledge_integration(integration_quality DESC, access_frequency DESC);

-- Evidence collection detailed tracking
CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_result_id UUID REFERENCES benchmarking_results(id) ON DELETE CASCADE,
    evidence_type VARCHAR(50) NOT NULL CHECK (evidence_type IN (
        'code_analysis', 'test_results', 'performance_metrics', 'documentation_review',
        'architectural_compliance', 'security_scan', 'dependency_analysis', 'manual_review'
    )),
    evidence_data JSONB NOT NULL,
    collection_method VARCHAR(50) NOT NULL,
    reliability_score DECIMAL(3,2) DEFAULT 0.8 CHECK (reliability_score >= 0 AND reliability_score <= 1),
    collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_status VARCHAR(20) DEFAULT 'collected' CHECK (
        validation_status IN ('collected', 'validated', 'disputed', 'excluded', 'archived')
    ),
    validator VARCHAR(100),
    validation_notes TEXT,
    file_references TEXT[],
    size_bytes INTEGER DEFAULT 0,
    checksum VARCHAR(64)
);

-- Evidence collection performance indexes
CREATE INDEX IF NOT EXISTS idx_evidence_items_benchmark_type 
    ON evidence_items(benchmark_result_id, evidence_type, collection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_items_reliability 
    ON evidence_items(reliability_score DESC, validation_status);
CREATE INDEX IF NOT EXISTS idx_evidence_items_method 
    ON evidence_items(collection_method, collection_timestamp DESC);

-- Comparative benchmarking for trend analysis
CREATE TABLE IF NOT EXISTS benchmark_comparisons (
    comparison_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_benchmark_id UUID REFERENCES benchmarking_results(id) ON DELETE CASCADE,
    current_benchmark_id UUID REFERENCES benchmarking_results(id) ON DELETE CASCADE,
    comparison_category VARCHAR(100) NOT NULL,
    score_improvement DECIMAL(8,2) GENERATED ALWAYS AS (
        (SELECT br2.grade_score FROM benchmarking_results br2 WHERE br2.id = current_benchmark_id) -
        (SELECT br1.grade_score FROM benchmarking_results br1 WHERE br1.id = baseline_benchmark_id)
    ) STORED,
    percentage_change DECIMAL(5,2),
    improvement_areas JSONB,
    regression_areas JSONB,
    analysis_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_duration_ms DECIMAL(8,2),
    statistical_significance DECIMAL(4,3)
);

-- Benchmark comparison analysis indexes
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_improvement 
    ON benchmark_comparisons(score_improvement DESC, percentage_change DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_category 
    ON benchmark_comparisons(comparison_category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_baseline 
    ON benchmark_comparisons(baseline_benchmark_id, current_benchmark_id);

-- Performance views for benchmarking analysis
CREATE OR REPLACE VIEW benchmarking_performance_summary AS
SELECT 
    benchmark_category,
    COUNT(*) as total_benchmarks,
    AVG(grade_score) as avg_grade_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY grade_score) as median_grade_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY grade_score) as p95_grade_score,
    MIN(grade_score) as min_grade_score,
    MAX(grade_score) as max_grade_score,
    COUNT(CASE WHEN grade_level IN ('A+', 'A', 'A-') THEN 1 END) as a_grades,
    COUNT(CASE WHEN grade_level IN ('B+', 'B', 'B-') THEN 1 END) as b_grades,
    COUNT(CASE WHEN grade_level IN ('C+', 'C', 'C-') THEN 1 END) as c_grades,
    COUNT(CASE WHEN grade_level = 'F' THEN 1 END) as failing_grades,
    AVG(confidence_level) as avg_confidence,
    AVG(analysis_duration_ms) as avg_analysis_time_ms
FROM benchmarking_results 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY benchmark_category;

CREATE OR REPLACE VIEW knowledge_integration_efficiency AS
SELECT 
    integration_type,
    mcp_server_validated,
    COUNT(*) as total_integrations,
    COUNT(CASE WHEN sync_status = 'synced' THEN 1 END) as successful_syncs,
    COUNT(CASE WHEN sync_status = 'failed' THEN 1 END) as failed_syncs,
    CAST(COUNT(CASE WHEN sync_status = 'synced' THEN 1 END) AS DECIMAL) / COUNT(*) as sync_success_rate,
    AVG(integration_quality) as avg_integration_quality,
    AVG(access_frequency) as avg_access_frequency,
    MAX(last_sync) as latest_sync
FROM knowledge_integration 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY integration_type, mcp_server_validated;

-- Benchmarking analysis and optimization functions
CREATE OR REPLACE FUNCTION analyze_benchmarking_trends(
    p_category VARCHAR(100) DEFAULT NULL,
    p_days_back INTEGER DEFAULT 30
) 
RETURNS TABLE(
    trend_analysis JSONB,
    performance_insights JSONB,
    recommendations TEXT[]
) AS $$
DECLARE
    analysis_period INTERVAL;
    trend_data JSONB;
    performance_data JSONB;
    recommendations_list TEXT[] := ARRAY[]::TEXT[];
    avg_improvement DECIMAL;
    consistency_score DECIMAL;
BEGIN
    analysis_period := (p_days_back || ' days')::INTERVAL;
    
    -- Calculate trend metrics
    WITH trend_stats AS (
        SELECT 
            benchmark_category,
            COUNT(*) as benchmark_count,
            AVG(grade_score) as avg_score,
            STDDEV(grade_score) as score_stddev,
            MIN(created_at) as earliest_benchmark,
            MAX(created_at) as latest_benchmark
        FROM benchmarking_results 
        WHERE (p_category IS NULL OR benchmark_category = p_category)
        AND created_at > CURRENT_TIMESTAMP - analysis_period
        GROUP BY benchmark_category
    )
    SELECT jsonb_agg(
        jsonb_build_object(
            'category', benchmark_category,
            'count', benchmark_count,
            'avg_score', round(avg_score, 2),
            'consistency', CASE WHEN score_stddev > 0 THEN round(100 - (score_stddev * 10), 2) ELSE 100 END,
            'time_span_days', EXTRACT(DAYS FROM latest_benchmark - earliest_benchmark)
        )
    ) INTO trend_data
    FROM trend_stats;
    
    -- Calculate performance insights
    SELECT jsonb_build_object(
        'total_benchmarks_analyzed', COUNT(*),
        'avg_grade_score', round(AVG(grade_score), 2),
        'grade_distribution', jsonb_build_object(
            'excellent', COUNT(CASE WHEN grade_percentage >= 90 THEN 1 END),
            'good', COUNT(CASE WHEN grade_percentage >= 80 AND grade_percentage < 90 THEN 1 END),
            'needs_improvement', COUNT(CASE WHEN grade_percentage < 80 THEN 1 END)
        ),
        'avg_analysis_time_ms', round(AVG(analysis_duration_ms), 2),
        'confidence_level', round(AVG(confidence_level), 3)
    ) INTO performance_data
    FROM benchmarking_results 
    WHERE (p_category IS NULL OR benchmark_category = p_category)
    AND created_at > CURRENT_TIMESTAMP - analysis_period;
    
    -- Generate recommendations
    SELECT AVG(score_improvement) INTO avg_improvement
    FROM benchmark_comparisons 
    WHERE created_at > CURRENT_TIMESTAMP - analysis_period;
    
    IF avg_improvement < 0 THEN
        recommendations_list := array_append(recommendations_list,
            'Benchmarking scores showing decline trend - review recent changes');
    END IF;
    
    IF (performance_data->>'avg_analysis_time_ms')::DECIMAL > 5000 THEN
        recommendations_list := array_append(recommendations_list,
            'Analysis time exceeds 5 seconds - consider performance optimization');
    END IF;
    
    RETURN QUERY SELECT trend_data, performance_data, recommendations_list;
END;
$$ LANGUAGE plpgsql;

-- Knowledge integration sync function
CREATE OR REPLACE FUNCTION sync_knowledge_integration(
    p_integration_id UUID DEFAULT NULL,
    p_integration_type VARCHAR(50) DEFAULT NULL
) 
RETURNS JSONB AS $$
DECLARE
    sync_results JSONB := '{}'::JSONB;
    processed_count INTEGER := 0;
    success_count INTEGER := 0;
    error_count INTEGER := 0;
    integration_record knowledge_integration%ROWTYPE;
BEGIN
    -- Process specific integration or all pending ones
    FOR integration_record IN 
        SELECT * FROM knowledge_integration 
        WHERE (p_integration_id IS NULL OR id = p_integration_id)
        AND (p_integration_type IS NULL OR integration_type = p_integration_type)
        AND sync_status IN ('pending', 'failed')
        AND sync_attempts < 3
        ORDER BY created_at
    LOOP
        BEGIN
            processed_count := processed_count + 1;
            
            -- Update sync attempt
            UPDATE knowledge_integration 
            SET sync_attempts = sync_attempts + 1,
                sync_status = 'syncing',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = integration_record.id;
            
            -- Simulate MCP sync process (would be actual MCP call in production)
            -- This is a placeholder for the actual MCP knowledge server integration
            
            -- Mark as successful
            UPDATE knowledge_integration 
            SET sync_status = 'synced',
                last_sync = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = integration_record.id;
            
            success_count := success_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Mark as failed
            UPDATE knowledge_integration 
            SET sync_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = integration_record.id;
            
            error_count := error_count + 1;
        END;
    END LOOP;
    
    sync_results := jsonb_build_object(
        'sync_timestamp', CURRENT_TIMESTAMP,
        'processed_integrations', processed_count,
        'successful_syncs', success_count,
        'failed_syncs', error_count,
        'success_rate', CASE WHEN processed_count > 0 
                           THEN CAST(success_count AS DECIMAL) / processed_count 
                           ELSE 0 END
    );
    
    RETURN sync_results;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for benchmarking data
CREATE OR REPLACE FUNCTION cleanup_benchmarking_data(
    p_days_to_keep INTEGER DEFAULT 180
) 
RETURNS JSONB AS $$
DECLARE
    cutoff_date TIMESTAMP;
    deleted_results INTEGER;
    deleted_evidence INTEGER;
    deleted_comparisons INTEGER;
    archived_count INTEGER;
    cleanup_results JSONB;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (p_days_to_keep || ' days')::INTERVAL;
    
    -- Archive old benchmarking results instead of deleting (for historical analysis)
    UPDATE benchmarking_results 
    SET validation_status = 'archived',
        updated_at = CURRENT_TIMESTAMP
    WHERE created_at < cutoff_date 
    AND validation_status NOT IN ('archived', 'disputed');
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    -- Delete very old evidence items (keep archived benchmark results)
    DELETE FROM evidence_items 
    WHERE collection_timestamp < CURRENT_TIMESTAMP - INTERVAL '1 year';
    GET DIAGNOSTICS deleted_evidence = ROW_COUNT;
    
    -- Clean up old comparison analysis
    DELETE FROM benchmark_comparisons 
    WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_comparisons = ROW_COUNT;
    
    cleanup_results := jsonb_build_object(
        'cleanup_timestamp', CURRENT_TIMESTAMP,
        'cutoff_date', cutoff_date,
        'archived_results', archived_count,
        'deleted_evidence_items', deleted_evidence,
        'deleted_comparisons', deleted_comparisons
    );
    
    RETURN cleanup_results;
END;
$$ LANGUAGE plpgsql;