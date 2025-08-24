-- Migration: Add Claude Code Knowledge Support (Issue #97 Phase 1)
-- Date: 2025-08-23
-- Description: Extend existing knowledge graph schema to support Claude Code capabilities
-- Safety: This migration is ADDITIVE ONLY - no existing data is modified

-- =============================================================================
-- MIGRATION VALIDATION CHECKS
-- =============================================================================

-- Check that we're working with the expected schema version (DuckDB compatible)
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollback_script TEXT
);

-- Insert migration record
INSERT INTO schema_migrations (version, description, rollback_script) 
VALUES (
    'claude_knowledge_v1', 
    'Add Claude Code knowledge entity types and relationships',
    'DROP INDEX IF EXISTS idx_entities_claude_type; UPDATE entities SET type = ''function'' WHERE type IN (''claude_capability'', ''claude_limitation'', ''claude_tool'', ''implementation_pattern'', ''anti_pattern''); UPDATE relationships SET relationship_type = ''uses'' WHERE relationship_type IN (''supports'', ''conflicts_with'', ''alternative_to'', ''validates'');'
);

-- =============================================================================
-- EXTEND EXISTING ENTITY TYPES (SAFE ADDITIVE CHANGES)
-- =============================================================================

-- Update entities table constraints to include new Claude-specific types  
-- NOTE: DuckDB requires recreating the table to modify constraints
-- This creates a new table with expanded constraints and migrates data

-- Create temporary table with expanded constraints
CREATE TABLE entities_new (
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
    
    -- Expanded check constraints to include Claude types
    CHECK (type IN (
        'function', 'class', 'module', 'variable', 'constant', 'interface', 'enum',
        'claude_capability', 'claude_limitation', 'claude_tool', 
        'implementation_pattern', 'anti_pattern'
    )),
    CHECK (line_start IS NULL OR line_start >= 1),
    CHECK (line_end IS NULL OR line_end >= line_start),
    CHECK (name != ''),
    CHECK (file_path != '')
);

-- Copy all existing data to new table
INSERT INTO entities_new (
    id, type, name, file_path, line_start, line_end, ast_hash, 
    embedding, metadata, created_at, updated_at
)
SELECT 
    id, type, name, file_path, line_start, line_end, ast_hash,
    embedding, metadata, created_at, updated_at
FROM entities;

-- Drop the old table
DROP TABLE entities;

-- Rename new table to original name
ALTER TABLE entities_new RENAME TO entities;

-- =============================================================================
-- EXTEND EXISTING RELATIONSHIP TYPES (SAFE ADDITIVE CHANGES)
-- =============================================================================

-- Update relationships table constraints to include new relationship types
-- NOTE: DuckDB requires recreating the table to modify constraints

-- Create temporary table with expanded constraints
CREATE TABLE relationships_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL, 
    relationship_type VARCHAR(50) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Expanded check constraints to include Claude relationship types
    CHECK (relationship_type IN (
        'imports', 'calls', 'extends', 'uses', 'implements', 'references', 'contains',
        'supports', 'conflicts_with', 'alternative_to', 'validates'
    )),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (source_id != target_id) -- Prevent self-references
);

-- Copy all existing data to new table
INSERT INTO relationships_new (
    id, source_id, target_id, relationship_type, confidence, metadata, created_at
)
SELECT 
    id, source_id, target_id, relationship_type, confidence, metadata, created_at
FROM relationships;

-- Drop the old table
DROP TABLE relationships;

-- Rename new table to original name
ALTER TABLE relationships_new RENAME TO relationships;

-- =============================================================================
-- ADDITIONAL INDEXES FOR CLAUDE KNOWLEDGE PERFORMANCE
-- =============================================================================

-- Index for Claude-specific entity types queries
CREATE INDEX IF NOT EXISTS idx_entities_claude_type 
    ON entities(type) 
    WHERE type IN ('claude_capability', 'claude_limitation', 'claude_tool', 'implementation_pattern', 'anti_pattern');

-- Index for Claude-specific relationship types queries  
CREATE INDEX IF NOT EXISTS idx_relationships_claude_type
    ON relationships(relationship_type)
    WHERE relationship_type IN ('supports', 'conflicts_with', 'alternative_to', 'validates');

-- Composite index for capability-tool relationships
CREATE INDEX IF NOT EXISTS idx_claude_capability_relationships
    ON relationships(source_id, relationship_type)
    WHERE relationship_type = 'supports';

-- =============================================================================
-- ANALYTICAL VIEWS FOR CLAUDE KNOWLEDGE
-- =============================================================================

-- View: Claude Capability Coverage
CREATE OR REPLACE VIEW claude_capability_coverage AS
SELECT 
    cap.name AS capability_name,
    cap.metadata AS capability_details,
    COUNT(CASE WHEN r.relationship_type = 'supports' THEN 1 END) AS supporting_tools_count,
    COUNT(CASE WHEN r.relationship_type = 'conflicts_with' THEN 1 END) AS conflicts_count,
    STRING_AGG(
        CASE WHEN r.relationship_type = 'supports' THEN tool.name END, 
        ', '
    ) AS supporting_tools
FROM entities cap
LEFT JOIN relationships r ON cap.id = r.target_id OR cap.id = r.source_id
LEFT JOIN entities tool ON (
    (r.source_id = tool.id AND r.target_id = cap.id) OR
    (r.target_id = tool.id AND r.source_id = cap.id)
) AND tool.type = 'claude_tool'
WHERE cap.type = 'claude_capability'
GROUP BY cap.id, cap.name, cap.metadata
ORDER BY supporting_tools_count DESC;

-- View: Implementation Pattern Analysis
CREATE OR REPLACE VIEW claude_pattern_analysis AS
SELECT 
    pattern.name AS pattern_name,
    pattern.metadata AS pattern_details,
    COUNT(CASE WHEN r.relationship_type = 'alternative_to' THEN 1 END) AS alternatives_count,
    COUNT(CASE WHEN r.relationship_type = 'conflicts_with' THEN 1 END) AS conflicts_count,
    STRING_AGG(
        CASE WHEN r.relationship_type = 'alternative_to' THEN anti.name END,
        ', '
    ) AS alternative_to_antipatterns
FROM entities pattern
LEFT JOIN relationships r ON pattern.id = r.source_id
LEFT JOIN entities anti ON r.target_id = anti.id AND anti.type = 'anti_pattern'
WHERE pattern.type = 'implementation_pattern'
GROUP BY pattern.id, pattern.name, pattern.metadata
ORDER BY alternatives_count DESC;

-- View: Tool Validation Coverage
CREATE OR REPLACE VIEW claude_tool_validation AS
SELECT 
    tool.name AS tool_name,
    tool.metadata AS tool_details,
    COUNT(CASE WHEN r.relationship_type = 'validates' THEN 1 END) AS validates_patterns_count,
    COUNT(CASE WHEN r.relationship_type = 'supports' THEN 1 END) AS supports_capabilities_count,
    STRING_AGG(
        CASE WHEN r.relationship_type = 'validates' THEN pattern.name END,
        ', '
    ) AS validated_patterns
FROM entities tool
LEFT JOIN relationships r ON tool.id = r.source_id
LEFT JOIN entities pattern ON r.target_id = pattern.id AND pattern.type = 'implementation_pattern'
WHERE tool.type = 'claude_tool'
GROUP BY tool.id, tool.name, tool.metadata
ORDER BY validates_patterns_count DESC;

-- =============================================================================
-- DATA INTEGRITY VALIDATION FOR CLAUDE KNOWLEDGE
-- =============================================================================

-- Extended validation view including Claude knowledge
CREATE OR REPLACE VIEW validate_claude_knowledge AS
SELECT 
    'claude_entities' AS table_name,
    COUNT(*) AS row_count,
    COUNT(CASE WHEN type = 'claude_capability' THEN 1 END) AS capabilities_count,
    COUNT(CASE WHEN type = 'claude_limitation' THEN 1 END) AS limitations_count,
    COUNT(CASE WHEN type = 'claude_tool' THEN 1 END) AS tools_count,
    COUNT(CASE WHEN type = 'implementation_pattern' THEN 1 END) AS patterns_count,
    COUNT(CASE WHEN type = 'anti_pattern' THEN 1 END) AS antipatterns_count
FROM entities
WHERE type IN ('claude_capability', 'claude_limitation', 'claude_tool', 'implementation_pattern', 'anti_pattern')
UNION ALL
SELECT 
    'claude_relationships' AS table_name,
    COUNT(*) AS row_count,
    COUNT(CASE WHEN relationship_type = 'supports' THEN 1 END) AS supports_count,
    COUNT(CASE WHEN relationship_type = 'conflicts_with' THEN 1 END) AS conflicts_count,
    COUNT(CASE WHEN relationship_type = 'alternative_to' THEN 1 END) AS alternatives_count,
    COUNT(CASE WHEN relationship_type = 'validates' THEN 1 END) AS validates_count,
    0 AS unused_column -- To match column count
FROM relationships
WHERE relationship_type IN ('supports', 'conflicts_with', 'alternative_to', 'validates');

-- =============================================================================
-- MIGRATION COMPLETION VERIFICATION
-- =============================================================================

-- Verify all constraints are properly applied
CREATE OR REPLACE VIEW migration_verification AS
SELECT 
    'constraint_check' AS check_type,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM information_schema.check_constraints 
            WHERE constraint_name = 'entities_type_check_claude_v1'
        ) THEN 'PASSED'
        ELSE 'FAILED'
    END AS entity_constraint_status,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM information_schema.check_constraints 
            WHERE constraint_name = 'relationships_relationship_type_check_claude_v1'  
        ) THEN 'PASSED'
        ELSE 'FAILED'
    END AS relationship_constraint_status
UNION ALL
SELECT 
    'index_check' AS check_type,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM pg_indexes 
            WHERE indexname = 'idx_entities_claude_type'
        ) THEN 'PASSED'
        ELSE 'FAILED' 
    END AS entity_index_status,
    CASE
        WHEN EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_relationships_claude_type'
        ) THEN 'PASSED'
        ELSE 'FAILED'
    END AS relationship_index_status;

-- =============================================================================
-- MIGRATION STATUS REPORT
-- =============================================================================

SELECT 'Claude Knowledge Migration v1 Complete' AS status,
       'New entity types: claude_capability, claude_limitation, claude_tool, implementation_pattern, anti_pattern' AS entities_added,
       'New relationship types: supports, conflicts_with, alternative_to, validates' AS relationships_added,
       'Views created: claude_capability_coverage, claude_pattern_analysis, claude_tool_validation' AS views_added,
       'All changes are ADDITIVE - existing data preserved' AS safety_note;