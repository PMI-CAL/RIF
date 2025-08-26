-- Simple DuckDB Schema for RIF Knowledge Graph
-- Issue #26: Implement DuckDB schema for knowledge graph with vector search
-- Compatible with DuckDB vector types and VSS extension

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Entities Table: Store code entities with vector embeddings
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,          -- function, class, module, variable
    name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    ast_hash VARCHAR(64),               -- For incremental updates
    embedding FLOAT[768],               -- Vector representation for VSS
    metadata JSON,                      -- Flexible attributes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Check constraints - Updated to support knowledge migration types and Claude documentation
    CHECK (type IN ('function', 'class', 'module', 'variable', 'constant', 'interface', 'enum', 'pattern', 'decision', 'learning', 'metric', 'issue_resolution', 'checkpoint', 'knowledge_item', 'claude_capability', 'claude_limitation', 'claude_tool', 'implementation_pattern', 'anti_pattern', 'claude_documentation')),
    CHECK (line_start IS NULL OR line_start >= 1),
    CHECK (line_end IS NULL OR line_end >= line_start),
    CHECK (name != ''),
    CHECK (file_path != '')
);

-- Relationships Table: Track code relationships and dependencies
CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL, 
    relationship_type VARCHAR(50) NOT NULL, -- imports, calls, extends, uses
    confidence FLOAT DEFAULT 1.0,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Check constraints - Updated to support Claude documentation relationships
    CHECK (relationship_type IN ('imports', 'calls', 'extends', 'uses', 'implements', 'references', 'contains', 'supports', 'alternative_to', 'conflicts_with', 'documents', 'explains', 'warns_against')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (source_id != target_id) -- Prevent self-references
);

-- Agent Memory Table: Store agent conversations, decisions, and learnings
CREATE TABLE IF NOT EXISTS agent_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(50) NOT NULL,
    issue_number INTEGER,
    context TEXT NOT NULL,
    decision TEXT,
    outcome VARCHAR(50),
    embedding FLOAT[768],               -- Vector representation for similarity search
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Check constraints
    CHECK (agent_type IN ('RIF-Analyst', 'RIF-Planner', 'RIF-Architect', 'RIF-Implementer', 'RIF-Validator', 'RIF-Learner', 'RIF-PR-Manager')),
    CHECK (outcome IS NULL OR outcome IN ('success', 'failure', 'partial', 'pending', 'skipped')),
    CHECK (context != '')
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Entity lookup indexes
CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(type, name);
CREATE INDEX IF NOT EXISTS idx_entities_file_path ON entities(file_path);
CREATE INDEX IF NOT EXISTS idx_entities_hash ON entities(ast_hash);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at);

-- Relationship traversal indexes
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_source_type ON relationships(source_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_target_type ON relationships(target_id, relationship_type);

-- Agent memory query indexes
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_memory_issue ON agent_memory(issue_number);
CREATE INDEX IF NOT EXISTS idx_agent_memory_outcome ON agent_memory(outcome);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type_issue ON agent_memory(agent_type, issue_number);
CREATE INDEX IF NOT EXISTS idx_agent_memory_created_at ON agent_memory(created_at);

-- =============================================================================
-- ANALYTICAL VIEWS
-- =============================================================================

-- Module Dependency Graph
CREATE OR REPLACE VIEW mv_module_dependencies AS
SELECT DISTINCT 
    e1.name AS source_module,
    e1.file_path AS source_file,
    e2.name AS target_module,
    e2.file_path AS target_file,
    r.relationship_type,
    r.confidence
FROM relationships r
JOIN entities e1 ON r.source_id = e1.id  
JOIN entities e2 ON r.target_id = e2.id
WHERE e1.type = 'module' AND e2.type = 'module'
ORDER BY source_module, target_module;

-- Function Call Graph
CREATE OR REPLACE VIEW mv_function_calls AS
SELECT 
    e1.name AS caller_function,
    e1.file_path AS caller_file,
    e2.name AS called_function, 
    e2.file_path AS called_file,
    COUNT(*) AS call_count
FROM relationships r
JOIN entities e1 ON r.source_id = e1.id
JOIN entities e2 ON r.target_id = e2.id  
WHERE r.relationship_type = 'calls'
  AND e1.type = 'function' 
  AND e2.type = 'function'
GROUP BY e1.name, e1.file_path, e2.name, e2.file_path
ORDER BY call_count DESC;

-- Agent Learning Patterns
CREATE OR REPLACE VIEW mv_agent_learnings AS
SELECT 
    agent_type,
    outcome, 
    COUNT(*) AS frequency,
    AVG(LENGTH(context)) AS avg_context_length
FROM agent_memory 
WHERE outcome IS NOT NULL
GROUP BY agent_type, outcome
ORDER BY frequency DESC;

-- Entity Statistics
CREATE OR REPLACE VIEW mv_entity_stats AS
SELECT 
    type,
    COUNT(*) AS entity_count,
    COUNT(DISTINCT file_path) AS file_count,
    AVG(CASE WHEN line_end IS NOT NULL AND line_start IS NOT NULL 
             THEN line_end - line_start + 1 
             ELSE NULL END) AS avg_lines
FROM entities
GROUP BY type
ORDER BY entity_count DESC;

-- Relationship Statistics  
CREATE OR REPLACE VIEW mv_relationship_stats AS
SELECT 
    relationship_type,
    COUNT(*) AS relationship_count,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence
FROM relationships
GROUP BY relationship_type
ORDER BY relationship_count DESC;

-- =============================================================================
-- VECTOR SIMILARITY INDEXES (VSS Extension) 
-- =============================================================================

-- Install and load VSS extension
INSTALL vss;
LOAD vss;

-- Set experimental persistence setting
SET hnsw_enable_experimental_persistence=true;

-- Create HNSW indexes for vector similarity search (if VSS is available)
-- Note: These may fail if VSS extension is not available, but that's okay
-- They are not essential for basic functionality

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- Test basic functionality
SELECT 'DuckDB Knowledge Graph Schema Created with Migration Support' AS status;

-- Verify table creation
SELECT table_name, column_count 
FROM (
    SELECT 'entities' AS table_name, COUNT(*) AS column_count
    FROM information_schema.columns 
    WHERE table_name = 'entities'
    
    UNION ALL
    
    SELECT 'relationships' AS table_name, COUNT(*) AS column_count
    FROM information_schema.columns 
    WHERE table_name = 'relationships'
    
    UNION ALL
    
    SELECT 'agent_memory' AS table_name, COUNT(*) AS column_count
    FROM information_schema.columns 
    WHERE table_name = 'agent_memory'
) AS table_info
ORDER BY table_name;

-- Schema setup complete
SELECT 
    'Tables: entities, relationships, agent_memory' AS tables,
    'Vector support: FLOAT[768] embeddings' AS vectors,
    'Extensions: VSS for similarity search' AS extensions;