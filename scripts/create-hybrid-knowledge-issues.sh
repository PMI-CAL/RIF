#!/bin/bash

# Script to create all GitHub issues for the Hybrid Knowledge System implementation
# Run this script in the RIF repository root

echo "Creating GitHub Issues for Hybrid Knowledge System Implementation..."
echo "================================================================="

# Create the epic first
echo "Creating Epic issue..."
EPIC_NUMBER=$(gh issue create \
  --title "Epic: Replace LightRAG with Hybrid Graph-Based Knowledge System" \
  --body "# Epic: Hybrid Graph-Based Knowledge System Implementation

## Overview
Complete replacement of the current LightRAG implementation with a hybrid graph-based knowledge system that provides real-time code understanding, relationship mapping, and intelligent context retrieval for RIF agents.

## Goals
- Transform every file and code change into a queryable knowledge graph
- Provide real-time AST-based code understanding
- Enable agents to understand code relationships and dependencies
- Optimize context delivery to fit within agent windows
- Run efficiently on resource-constrained hardware (8GB MacBook Air)

## Technology Stack
- **Tree-sitter**: Fast AST parsing and code structure analysis
- **DuckDB + VSS**: Embedded database with vector similarity search
- **Graph Relationships**: Built on DuckDB using recursive CTEs
- **Local Embeddings**: Sentence-transformers for offline operation
- **File Monitoring**: Watchdog with priority queue processing

## Implementation Phases
1. **Foundation** (Issues #1-3): Decouple from LightRAG, setup infrastructure
2. **Core Implementation** (Issues #4-8): Build parsing, extraction, relationships
3. **Query Interface** (Issues #9-11): Create query planner and context optimizer
4. **Integration** (Issues #12-14): Compatibility layer and shadow mode
5. **Migration** (Issue #15): Phased cutover from LightRAG

## Success Criteria
- Query latency P95 < 100ms
- Memory usage < 2GB for typical projects
- Real-time file change processing < 100ms
- Agent task completion improvement > 20%
- Zero downtime migration

## Documentation
- PRD: architecture/hybrid-knowledge-system-prd.md
- Issues: architecture/hybrid-knowledge-github-issues.md

## Tracking
This epic tracks the overall implementation. Individual issues will be created for each component." \
  --label "epic,state:new,complexity:very-high" \
  --repo PMI-CAL/RIF | grep -oE '[0-9]+' | head -1)

echo "Created Epic issue #$EPIC_NUMBER"
echo ""

# Function to create an issue and extract its number
create_issue() {
    local title="$1"
    local body="$2"
    local labels="$3"
    
    echo "Creating: $title"
    issue_num=$(gh issue create \
        --title "$title" \
        --body "$body" \
        --label "$labels" \
        --repo PMI-CAL/RIF | grep -oE '[0-9]+' | head -1)
    echo "  Created issue #$issue_num"
    echo ""
    return $issue_num
}

# Foundation Phase Issues

echo "Phase 1: Foundation Issues"
echo "--------------------------"

# Issue 1
create_issue "Decouple RIF agents from LightRAG implementation" \
"## Description
Create an abstraction layer between RIF agents and the knowledge system to enable hot-swapping between LightRAG and the new hybrid system.

## Technical Requirements
- Define a KnowledgeInterface abstract base class
- Implement interface methods for all current LightRAG operations
- Update all agents to use interface instead of direct LightRAG calls
- Create mock implementation for testing

## Implementation Steps
1. Analyze current LightRAG usage in agents
2. Define minimal interface covering all operations
3. Create LightRAGAdapter implementing interface
4. Update agents to use dependency injection
5. Add configuration for knowledge provider selection

## Acceptance Criteria
- [ ] KnowledgeInterface abstract class defined with all methods
- [ ] LightRAGAdapter wraps existing LightRAG with interface
- [ ] All 5 RIF agents updated to use interface
- [ ] Tests pass with mock knowledge provider
- [ ] No performance regression

## Files to Modify
- claude/agents/rif-analyst.md
- claude/agents/rif-architect.md  
- claude/agents/rif-implementer.md
- claude/agents/rif-validator.md
- claude/agents/rif-learner.md
- NEW: knowledge/interface.py
- NEW: knowledge/adapters/lightrag_adapter.py

## Testing Strategy
- Unit tests for interface and adapter
- Integration tests with mock provider
- Regression tests for agent operations

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 4 hours
## Dependencies: None" \
"state:new,type:refactoring,component:knowledge"

# Issue 2
create_issue "Set up DuckDB as embedded database with vector search" \
"## Description
Install and configure DuckDB with the VSS (Vector Similarity Search) extension for hybrid storage combining relational, vector, and graph capabilities.

## Technical Requirements
- Install duckdb Python package (>=0.9.0)
- Install and load VSS extension
- Configure memory limits for constrained hardware
- Set up connection pooling
- Create initialization script

## Implementation Steps
1. Add duckdb to requirements.txt
2. Create DuckDB initialization module
3. Load VSS extension at startup
4. Configure memory_limit=500MB
5. Set up connection pool (max 4 connections)
6. Create database in knowledge/duckdb/ directory
7. Add backup and recovery procedures

## Acceptance Criteria
- [ ] DuckDB installed and importable
- [ ] VSS extension loaded successfully
- [ ] Can create vector indexes
- [ ] Memory usage stays under 500MB
- [ ] Connection pooling works with concurrent access
- [ ] Database persists between restarts

## Files to Create
- knowledge/duckdb/__init__.py
- knowledge/duckdb/connection.py
- knowledge/duckdb/init.sql
- knowledge/duckdb/config.yaml

## Testing Strategy
- Test vector similarity search
- Test concurrent connections
- Verify memory limits
- Test persistence

## Epic
Related to #$EPIC_NUMBER

## Complexity: Low
## Estimated time: 2 hours
## Dependencies: Issue #1" \
"state:new,type:infrastructure,component:database"

# Issue 3
create_issue "Create tree-sitter parsing infrastructure" \
"## Description
Build a parser manager for multiple programming languages using tree-sitter with AST caching and incremental parsing support.

## Technical Requirements
- Install tree-sitter Python bindings
- Download and compile language grammars
- Implement parser pool with language detection
- Add AST caching with LRU eviction
- Support incremental parsing for edits

## Implementation Steps
1. Install tree-sitter and tree-sitter-languages packages
2. Create ParserManager class
3. Load grammars for: JavaScript, TypeScript, Python, Go, Rust
4. Implement language detection from file extensions
5. Add thread-safe parser pool (1 parser per language)
6. Create LRU cache for parsed ASTs (max 100 files)
7. Implement incremental parsing for file edits

## Acceptance Criteria
- [ ] Tree-sitter installed with Python bindings
- [ ] Can parse JS, TS, Python, Go files
- [ ] Language auto-detected from extension
- [ ] AST cache with 100-file LRU eviction
- [ ] Incremental parsing reduces time by >50%
- [ ] Thread-safe for concurrent parsing
- [ ] Memory usage <200MB with cache full

## Files to Create
- knowledge/parsing/__init__.py
- knowledge/parsing/parser_manager.py
- knowledge/parsing/ast_cache.py
- knowledge/parsing/languages.yaml
- knowledge/parsing/tree_queries/

## Testing Strategy
- Parse sample files in each language
- Test incremental parsing with edits
- Verify cache eviction
- Test concurrent parsing
- Measure parsing performance

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium  
## Estimated time: 6 hours
## Dependencies: None" \
"state:new,type:feature,component:parser"

echo "Phase 2: Core Implementation Issues"
echo "------------------------------------"

# Issue 4
create_issue "Implement DuckDB schema for knowledge graph" \
"## Description
Create the database schema for storing code entities, relationships, embeddings, and agent memory in DuckDB.

## Technical Requirements
- Design normalized schema for graph storage
- Add vector columns for embeddings
- Create indexes for query performance
- Implement materialized views for common queries
- Add migration system for schema updates

## Implementation Steps
1. Create schema.sql with all table definitions
2. Add entities table with AST hash for change detection
3. Add relationships table with typed relationships
4. Add agent_memory table for storing decisions
5. Create indexes on frequently queried columns
6. Add materialized views for dependency graphs
7. Implement migration system using version tracking

## Schema Tables
\`\`\`sql
entities: id, type, name, file_path, line_start, line_end, ast_hash, embedding[768], metadata, timestamps
relationships: id, source_id, target_id, type, confidence, metadata, created_at
agent_memory: id, agent_type, issue_number, context, decision, outcome, embedding[768], metadata, created_at
schema_versions: version, applied_at, migration_file
\`\`\`

## Acceptance Criteria
- [ ] All tables created successfully
- [ ] Foreign key constraints enforced
- [ ] Indexes improve query performance >2x
- [ ] Materialized views update correctly
- [ ] Migration system tracks versions
- [ ] Can rollback migrations

## Files to Create
- knowledge/duckdb/schema.sql
- knowledge/duckdb/migrations/
- knowledge/duckdb/migrate.py
- knowledge/duckdb/views.sql

## Testing Strategy
- Test all CRUD operations
- Verify constraint enforcement
- Benchmark query performance
- Test migration and rollback

## Epic
Related to #$EPIC_NUMBER

## Complexity: Low
## Estimated time: 3 hours  
## Dependencies: Issue #2" \
"state:new,type:feature,component:database"

# Issue 5
create_issue "Implement real-time file monitoring with watchdog" \
"## Description
Create a file monitoring system that detects changes in real-time, with debouncing and a priority queue for processing.

## Technical Requirements
- Use watchdog library for cross-platform monitoring
- Implement debouncing to batch rapid changes
- Priority queue for processing order
- Handle file moves, renames, and deletes
- Respect .gitignore patterns

## Implementation Steps
1. Install watchdog package
2. Create FileMonitor class with event handlers
3. Implement 500ms debounce window
4. Add priority queue with 3 levels:
   - P1: Currently edited files
   - P2: Imported/dependent files  
   - P3: Other project files
5. Parse .gitignore for exclusion patterns
6. Handle special events (moves, renames)
7. Add graceful shutdown

## Acceptance Criteria
- [ ] Detects file changes within 100ms
- [ ] Debouncing prevents duplicate processing
- [ ] Priority queue processes important files first
- [ ] Respects .gitignore patterns
- [ ] Handles 1000+ file changes gracefully
- [ ] Clean shutdown without data loss

## Files to Create
- knowledge/monitoring/__init__.py
- knowledge/monitoring/file_watcher.py
- knowledge/monitoring/event_queue.py
- knowledge/monitoring/priority_processor.py

## Testing Strategy
- Test with rapid file changes
- Verify priority ordering
- Test .gitignore compliance
- Benchmark with large changes

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 4 hours
## Dependencies: Issue #3" \
"state:new,type:feature,component:monitoring"

# Issue 6
create_issue "Extract code entities from AST" \
"## Description
Parse AST to extract code entities (functions, classes, modules) with metadata and store them in the database.

## Technical Requirements
- Extract entities from tree-sitter AST
- Capture entity metadata (parameters, return types, docstrings)
- Generate unique IDs for entities
- Handle nested entities (inner classes, closures)
- Support multiple languages

## Implementation Steps
1. Create EntityExtractor base class
2. Implement language-specific extractors:
   - JavaScriptExtractor (functions, classes, modules)
   - PythonExtractor (functions, classes, modules, methods)
   - GoExtractor (functions, types, interfaces, methods)
3. Extract metadata:
   - Parameters and types
   - Return types
   - Docstrings/comments
   - Decorators/annotations
4. Generate deterministic entity IDs
5. Handle nested and anonymous entities
6. Store entities in DuckDB

## Acceptance Criteria
- [ ] Extracts all major entity types
- [ ] Captures complete metadata
- [ ] Handles nested entities correctly
- [ ] Supports JS, Python, Go
- [ ] Stores in database successfully
- [ ] Processing speed >100 files/minute

## Files to Create
- knowledge/extraction/__init__.py
- knowledge/extraction/entity_extractor.py
- knowledge/extraction/javascript_extractor.py
- knowledge/extraction/python_extractor.py
- knowledge/extraction/go_extractor.py

## Testing Strategy
- Test with sample codebases
- Verify entity completeness
- Test edge cases (anonymous, nested)
- Benchmark extraction speed

## Epic
Related to #$EPIC_NUMBER

## Complexity: High
## Estimated time: 8 hours
## Dependencies: Issues #3, #4" \
"state:new,type:feature,component:extraction"

# Issue 7
create_issue "Detect and store code relationships" \
"## Description
Analyze AST and extracted entities to detect relationships between code elements (imports, calls, inheritance).

## Technical Requirements
- Detect static relationships (imports, exports, extends)
- Detect dynamic relationships (calls, uses, instantiates)
- Resolve cross-file references
- Calculate relationship confidence scores
- Handle indirect relationships

## Implementation Steps
1. Create RelationshipDetector base class
2. Implement relationship types:
   - imports/exports
   - function calls
   - class inheritance
   - interface implementation
   - type usage
   - variable references
3. Add cross-file reference resolution
4. Calculate confidence based on:
   - Direct vs indirect reference
   - Static vs dynamic analysis
   - Name matching accuracy
5. Store relationships with metadata
6. Build dependency graph

## Acceptance Criteria
- [ ] Detects all major relationship types
- [ ] Resolves cross-file references
- [ ] Confidence scores are meaningful
- [ ] Handles circular dependencies
- [ ] Graph queries work correctly
- [ ] Processing maintains <100ms latency

## Files to Create
- knowledge/relationships/__init__.py
- knowledge/relationships/detector.py
- knowledge/relationships/resolver.py
- knowledge/relationships/graph_builder.py

## Testing Strategy
- Test with known relationships
- Verify cross-file resolution
- Test circular dependencies
- Validate confidence scores

## Epic
Related to #$EPIC_NUMBER

## Complexity: High
## Estimated time: 8 hours
## Dependencies: Issue #6" \
"state:new,type:feature,component:relationships"

# Issue 8
create_issue "Generate and store vector embeddings" \
"## Description
Generate vector embeddings for code entities using a local model and store them in DuckDB's VSS index.

## Technical Requirements
- Use local embedding model (no API calls)
- Generate meaningful code embeddings
- Store in DuckDB VSS index
- Support batch generation
- Cache embeddings for unchanged code

## Implementation Steps
1. Install sentence-transformers package
2. Load CodeBERT or similar model
3. Create EmbeddingGenerator class
4. Implement code preprocessing:
   - Remove comments for similarity
   - Normalize naming conventions
   - Extract semantic features
5. Generate embeddings in batches of 32
6. Store in DuckDB with VSS index
7. Implement embedding cache

## Acceptance Criteria
- [ ] Local model loads successfully
- [ ] Embeddings capture code semantics
- [ ] Batch processing reduces time >50%
- [ ] VSS queries return similar code
- [ ] Cache prevents redundant generation
- [ ] Memory usage <500MB

## Files to Create
- knowledge/embeddings/__init__.py
- knowledge/embeddings/generator.py
- knowledge/embeddings/models.py
- knowledge/embeddings/preprocessor.py

## Testing Strategy
- Test embedding quality with known similar code
- Verify batch processing
- Test cache effectiveness
- Measure memory usage

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 5 hours
## Dependencies: Issue #6" \
"state:new,type:feature,component:embeddings"

echo "Phase 3: Query Interface Issues"
echo "--------------------------------"

# Issue 9
create_issue "Create query planner for hybrid searches" \
"## Description
Build a query planning engine that parses natural language and structured queries, then plans optimal execution strategy.

## Technical Requirements
- Parse natural language queries
- Support structured query format
- Plan vector vs graph vs hybrid strategy
- Optimize query execution order
- Handle complex multi-step queries

## Implementation Steps
1. Create QueryPlanner class
2. Implement natural language parser:
   - Extract intent and entities
   - Identify query type
   - Extract filters and constraints
3. Build execution strategies:
   - Vector-only for similarity
   - Graph-only for relationships
   - Hybrid for complex queries
4. Add query optimization:
   - Predicate pushdown
   - Join ordering
   - Result size estimation
5. Implement query templates
6. Add explain plan feature

## Query Examples
- 'Find functions similar to processData'
- 'What calls the authentication module?'
- 'Show dependencies of user service'
- 'Find error handling patterns'

## Acceptance Criteria
- [ ] Parses natural language correctly
- [ ] Plans appropriate strategy
- [ ] Optimized plans execute >2x faster
- [ ] Handles complex queries
- [ ] Explain plan shows strategy
- [ ] Query latency <100ms

## Files to Create
- knowledge/query/__init__.py
- knowledge/query/planner.py
- knowledge/query/parser.py
- knowledge/query/optimizer.py
- knowledge/query/templates.yaml

## Testing Strategy
- Test various query types
- Verify optimization improvements
- Test edge cases
- Benchmark performance

## Epic
Related to #$EPIC_NUMBER

## Complexity: High
## Estimated time: 8 hours
## Dependencies: Issues #7, #8" \
"state:new,type:feature,component:query"

# Issue 10
create_issue "Optimize context for agent consumption" \
"## Description
Create a system to rank and prune query results to fit within agent context windows while maximizing relevance.

## Technical Requirements
- Implement relevance scoring algorithm
- Prune results to fit context window
- Preserve essential context
- Support different agent context sizes
- Add result summarization

## Implementation Steps
1. Create ContextOptimizer class
2. Implement scoring algorithm:
   - Direct relevance (40%): exact matches
   - Semantic relevance (30%): embedding similarity
   - Structural relevance (20%): graph distance
   - Temporal relevance (10%): recency
3. Add pruning strategies:
   - Keep highest scored items
   - Include key dependencies
   - Preserve context connectivity
4. Implement context budget:
   - 50% for direct results
   - 25% for context
   - 25% reserve
5. Add summarization for overflow
6. Support different window sizes

## Acceptance Criteria
- [ ] Relevance scoring is accurate
- [ ] Results fit context window
- [ ] Essential context preserved
- [ ] Quality metrics improve
- [ ] Supports multiple window sizes
- [ ] Graceful degradation

## Files to Create
- knowledge/context/__init__.py
- knowledge/context/optimizer.py
- knowledge/context/scorer.py
- knowledge/context/pruner.py

## Testing Strategy
- Test with various result sizes
- Verify scoring accuracy
- Test pruning effectiveness
- Measure context quality

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 5 hours
## Dependencies: Issue #9" \
"state:new,type:feature,component:context"

# Issue 11
create_issue "Store and query agent conversations" \
"## Description
Build a system to store agent conversations, decisions, and learnings for future reference and pattern detection.

## Technical Requirements
- Store complete agent conversations
- Extract key decisions and outcomes
- Index for searchability
- Detect error patterns
- Learn from successes and failures

## Implementation Steps
1. Create AgentMemory class
2. Implement conversation storage:
   - Full context and responses
   - Timestamp and issue context
   - Agent type and configuration
3. Extract key information:
   - Decisions made
   - Actions taken
   - Errors encountered
   - Solutions found
4. Add pattern detection:
   - Common error patterns
   - Successful strategies
   - Anti-patterns
5. Create searchable index
6. Build learning extraction

## Acceptance Criteria
- [ ] Stores all agent outputs
- [ ] Extracts decisions accurately
- [ ] Searchable by patterns
- [ ] Detects recurring errors
- [ ] Learning improves agent performance
- [ ] Query latency <50ms

## Files to Create
- knowledge/memory/__init__.py
- knowledge/memory/agent_memory.py
- knowledge/memory/pattern_detector.py
- knowledge/memory/learning_extractor.py

## Testing Strategy
- Test storage completeness
- Verify pattern detection
- Test search functionality
- Measure learning effectiveness

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 6 hours
## Dependencies: Issue #4" \
"state:new,type:feature,component:memory"

echo "Phase 4: Integration Issues"
echo "---------------------------"

# Issue 12
create_issue "Create LightRAG compatibility interface" \
"## Description
Implement an adapter layer that allows the new system to respond to LightRAG queries during migration.

## Technical Requirements
- Translate LightRAG queries to new format
- Convert responses to LightRAG format
- Maintain performance parity
- Support all LightRAG operations
- Enable gradual migration

## Implementation Steps
1. Create CompatibilityAdapter class
2. Map LightRAG operations:
   - store_pattern → entity + embedding
   - retrieve_similar → vector search
   - get_decisions → graph query
   - update_feedback → memory update
3. Implement response translation:
   - Convert graph results to patterns
   - Format embeddings as expected
   - Maintain metadata structure
4. Add performance optimization
5. Create migration flags
6. Add compatibility tests

## Acceptance Criteria
- [ ] All LightRAG queries work
- [ ] Response format matches
- [ ] No performance degradation
- [ ] Agents work unchanged
- [ ] Migration flags control behavior
- [ ] 100% compatibility test pass

## Files to Create
- knowledge/compatibility/__init__.py
- knowledge/compatibility/adapter.py
- knowledge/compatibility/translator.py
- knowledge/compatibility/tests.py

## Testing Strategy
- Test all LightRAG operations
- Verify response formats
- Compare performance
- Test with real agents

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 5 hours
## Dependencies: Issue #10" \
"state:new,type:compatibility,component:migration"

# Issue 13
create_issue "Run new system in parallel for testing" \
"## Description
Configure the system to run in shadow mode, processing everything in parallel with LightRAG without affecting agents.

## Technical Requirements
- Parallel processing of all operations
- Compare results between systems
- Log differences for analysis
- No impact on agent operations
- Performance monitoring

## Implementation Steps
1. Create ShadowMode controller
2. Implement parallel execution:
   - Intercept all queries
   - Run on both systems
   - Compare results
3. Add difference logging:
   - Query time comparison
   - Result set differences
   - Quality metrics
4. Create analysis dashboard
5. Add toggle mechanism
6. Implement gradual rollout

## Acceptance Criteria
- [ ] Runs without affecting agents
- [ ] Captures all operations
- [ ] Logs meaningful differences
- [ ] Performance metrics collected
- [ ] Can toggle on/off easily
- [ ] No memory leaks

## Files to Create
- knowledge/shadow/__init__.py
- knowledge/shadow/controller.py
- knowledge/shadow/comparator.py
- knowledge/shadow/metrics.py

## Testing Strategy
- Verify no agent impact
- Test comparison accuracy
- Check performance overhead
- Validate metrics

## Epic
Related to #$EPIC_NUMBER

## Complexity: Low
## Estimated time: 4 hours
## Dependencies: Issue #12" \
"state:new,type:testing,component:migration"

# Issue 14
create_issue "Implement system monitoring and metrics" \
"## Description
Add comprehensive monitoring for system health, performance metrics, and usage patterns.

## Technical Requirements
- Track memory usage
- Monitor query latency
- Measure indexing speed
- Count cache hits/misses
- Alert on anomalies

## Implementation Steps
1. Create MetricsCollector class
2. Add performance counters:
   - Query count and latency
   - Indexing rate
   - Cache hit ratio
   - Memory usage
   - CPU utilization
3. Implement time-series storage
4. Add alerting thresholds:
   - Memory >1.8GB
   - Query latency >500ms
   - Indexing queue >1000
5. Create metrics dashboard
6. Add export capability

## Metrics to Track
- Query latency (p50, p95, p99)
- Queries per second
- Index updates per minute
- Memory usage by component
- Cache effectiveness
- Error rates

## Acceptance Criteria
- [ ] All metrics collected
- [ ] Dashboard shows real-time data
- [ ] Alerts trigger correctly
- [ ] Historical data retained
- [ ] Export works
- [ ] Low overhead (<2%)

## Files to Create
- knowledge/metrics/__init__.py
- knowledge/metrics/collector.py
- knowledge/metrics/dashboard.py
- knowledge/metrics/alerts.py

## Testing Strategy
- Verify metric accuracy
- Test alert triggers
- Check overhead
- Validate dashboard

## Epic
Related to #$EPIC_NUMBER

## Complexity: Low
## Estimated time: 4 hours
## Dependencies: Issue #13" \
"state:new,type:monitoring,component:metrics"

echo "Phase 5: Migration Issues"
echo "-------------------------"

# Issue 15
create_issue "Migrate from LightRAG to new system" \
"## Description
Execute the phased migration plan to transition from LightRAG to the new hybrid knowledge system.

## Technical Requirements
- Phase 1: Read migration
- Phase 2: Write migration
- Phase 3: Full cutover
- Rollback capability at each phase
- Zero downtime migration

## Implementation Steps
1. Week 1 - Read Migration:
   - Route read queries to new system
   - Keep writes to LightRAG
   - Monitor and compare
2. Week 2 - Write Migration:
   - Dual-write to both systems
   - Verify consistency
   - Test rollback
3. Week 3 - Cutover:
   - Disable LightRAG queries
   - Archive LightRAG data
   - Update configuration
4. Week 4 - Cleanup:
   - Remove LightRAG code
   - Delete compatibility layer
   - Update documentation

## Rollback Procedures
- Phase 1: Route reads back to LightRAG
- Phase 2: Stop dual-writes
- Phase 3: Restore from archive
- Phase 4: Revert git commits

## Acceptance Criteria
- [ ] Each phase completes successfully
- [ ] No agent disruption
- [ ] Rollback tested at each phase
- [ ] Performance improves
- [ ] All data migrated
- [ ] Documentation updated

## Files to Modify
- config/knowledge_provider.yaml
- All agent configurations
- knowledge/lightrag/ (remove)
- requirements.txt

## Testing Strategy
- Test each phase independently
- Verify rollback procedures
- Load test during migration
- Validate data integrity

## Epic
Related to #$EPIC_NUMBER

## Complexity: Medium
## Estimated time: 8 hours (spread over 4 weeks)
## Dependencies: Issue #14" \
"state:new,type:migration,component:cutover"

echo ""
echo "================================================================="
echo "All issues created successfully!"
echo ""
echo "Summary:"
echo "- Epic Issue: #$EPIC_NUMBER"
echo "- Total Implementation Issues: 15"
echo "- Total Estimated Time: ~80 hours"
echo ""
echo "Next Steps:"
echo "1. RIF agents will automatically pick up issues with 'state:new' label"
echo "2. Issues will progress through the workflow automatically"
echo "3. Monitor progress through the epic issue #$EPIC_NUMBER"
echo ""
echo "To view all issues:"
echo "  gh issue list --label 'epic'"
echo "  gh issue list --label 'state:new'"