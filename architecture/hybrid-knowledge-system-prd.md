# Product Requirements Document: Hybrid Graph-Based Knowledge System for RIF

## Executive Summary

This document defines the complete replacement of the current LightRAG implementation with a hybrid graph-based knowledge system that provides real-time code understanding, relationship mapping, and intelligent context retrieval for RIF agents. The new system will transform every file and code change into a queryable knowledge graph while maintaining performance on resource-constrained hardware.

### Vision Statement
Create a local, free, and performant knowledge system that enables RIF agents to understand code structure, relationships, and evolution through a combination of AST analysis, vector embeddings, and graph relationships - providing precisely relevant context on-demand without overwhelming agent context windows.

## 1. Core Requirements and Constraints

### 1.1 Hardware Constraints
- **Target Hardware**: MacBook Air (2020-2022 models)
  - RAM: 8GB unified memory
  - CPU: Apple M1 or Intel dual-core i5
  - Storage: 256GB SSD with ~50GB available
  - Expected memory budget: 2GB maximum for knowledge system
  - CPU utilization: <30% during indexing, <5% idle

### 1.2 Functional Requirements

#### 1.2.1 Real-time Code Understanding
- **File Monitoring**: Track all file changes within 100ms
- **Incremental Updates**: Parse only changed portions of code
- **Language Support**: JavaScript/TypeScript, Python, Go, Rust, Java, C/C++
- **Relationship Tracking**: Imports, exports, calls, inheritance, usage
- **Temporal History**: Track how code evolves over time

#### 1.2.2 Knowledge Graph Capabilities
- **Entity Extraction**: Classes, functions, variables, modules, packages
- **Relationship Types**:
  - Static: imports, exports, extends, implements
  - Dynamic: calls, uses, modifies, depends_on
  - Semantic: similar_to, alternative_for, refactored_from
- **Query Patterns**:
  - Impact analysis: "What breaks if I change X?"
  - Similarity search: "Find code similar to this pattern"
  - Dependency tracking: "What does X depend on?"
  - Architecture understanding: "Show module relationships"

#### 1.2.3 Agent Integration
- **Context Optimization**: Provide only relevant information
- **Query Interface**: Natural language to graph queries
- **Memory Management**: Store agent conversations and decisions
- **Error Learning**: Track and learn from failures
- **Performance Baseline**: <100ms query response time

### 1.3 Non-Functional Requirements
- **Local Operation**: No external API calls or services
- **Free Software**: Only open-source components
- **Data Persistence**: Survive system restarts
- **Concurrent Access**: Multiple agents querying simultaneously
- **Graceful Degradation**: Partial functionality if components fail

## 2. Architecture Design Decisions

### 2.1 Component Selection Analysis

#### 2.1.1 Code Parsing Layer: Tree-sitter (Primary) + LSP (Secondary)

**Tree-sitter as Primary Parser**
- **Rationale**: 
  - 10-100x faster than LSP (microseconds vs milliseconds)
  - Consistent memory usage (~50MB per language)
  - Works offline, no language server required
  - Incremental parsing for real-time updates
  - Error recovery (parses incomplete code)
- **Implementation**: 
  - Use tree-sitter Python bindings
  - Pre-load grammars for supported languages
  - Cache parsed ASTs in memory with LRU eviction

**LSP as Semantic Enhancement**
- **When to Use**:
  - Type information needed (TypeScript, Java)
  - Cross-file symbol resolution
  - Refactoring impact analysis
- **Implementation**:
  - Lazy initialization only when needed
  - Query results cached in vector DB
  - Graceful fallback if LSP unavailable

#### 2.1.2 Vector Database: DuckDB with VSS Extension

**Why DuckDB over Alternatives**:
- **Memory Efficiency**: Column-store with aggressive compression (10-100x smaller than row stores)
- **Query Performance**: Vectorized execution, parallel processing
- **Embedded Mode**: No separate server process
- **ACID Compliance**: Transactional consistency for concurrent access
- **VSS Extension**: Native vector similarity search without external dependencies

**Comparison Matrix**:
| Database | Memory | Query Speed | Setup | Features | Stability |
|----------|--------|-------------|-------|----------|-----------|
| DuckDB+VSS | 200MB | <10ms | Simple | SQL+Vector | Excellent |
| Qdrant | 500MB | <20ms | Server | Vector only | Good |
| ChromaDB | 400MB | <50ms | Simple | Vector only | Fair |
| Redis | 1GB+ | <5ms | Server | Multi-model | Excellent |
| FAISS | 300MB | <10ms | Complex | Vector only | Good |

#### 2.1.3 Graph Layer: Embedded Graph on DuckDB

**Design Decision**: Build graph relationships directly in DuckDB rather than separate graph database.

**Rationale**:
- Single data store reduces complexity
- SQL recursive CTEs handle graph traversal
- JSON columns store flexible relationships
- Materialized views cache common queries
- Memory footprint stays under 500MB

**Schema Design**:
```sql
-- Entities table
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    type VARCHAR(50), -- function, class, module, etc.
    name VARCHAR(255),
    file_path VARCHAR(500),
    line_start INTEGER,
    line_end INTEGER,
    ast_hash VARCHAR(64), -- For change detection
    embedding FLOAT[768], -- Vector representation
    metadata JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Relationships table
CREATE TABLE relationships (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES entities(id),
    target_id UUID REFERENCES entities(id),
    relationship_type VARCHAR(50),
    confidence FLOAT,
    metadata JSON,
    created_at TIMESTAMP
);

-- Agent memory table
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY,
    agent_type VARCHAR(50),
    issue_number INTEGER,
    context TEXT,
    decision TEXT,
    outcome VARCHAR(50),
    embedding FLOAT[768],
    metadata JSON,
    created_at TIMESTAMP
);
```

### 2.2 Real-time Monitoring Architecture

#### 2.2.1 File Watcher Strategy
- **Technology**: Python watchdog library with debouncing
- **Update Queue**: Thread-safe queue with priority levels
  - Priority 1: Currently edited files
  - Priority 2: Imported/dependent files
  - Priority 3: Other project files
- **Batch Processing**: Accumulate changes for 500ms before processing
- **Incremental Parsing**: Only reparse changed sections using tree-sitter

#### 2.2.2 Indexing Pipeline
```
File Change → Debounce → Parse AST → Extract Entities → 
Generate Embeddings → Update Graph → Invalidate Caches → 
Notify Agents
```

**Performance Optimizations**:
- Parallel processing with thread pool (4 workers max)
- Bloom filters for duplicate detection
- Memory-mapped files for large codebases
- Lazy embedding generation (on first query)

### 2.3 Query Interface Design

#### 2.3.1 Query Types and Patterns

**Structured Queries** (for precise needs):
```python
# Find all functions that call 'processData'
query = {
    "type": "relationship",
    "source": {"name": "processData"},
    "relationship": "called_by",
    "depth": 2
}
```

**Natural Language Queries** (for agents):
```python
# "What functions handle user authentication?"
# Converted to: semantic search + filter by keywords + graph traversal
```

**Hybrid Queries** (combining approaches):
```python
# "Find similar error handling patterns to this code"
# Combines: embedding similarity + AST pattern matching + usage context
```

#### 2.3.2 Context Window Optimization

**Relevance Scoring Algorithm**:
1. **Direct Relevance** (weight: 0.4): Exact matches, direct relationships
2. **Semantic Relevance** (weight: 0.3): Embedding similarity
3. **Structural Relevance** (weight: 0.2): Graph distance, shared dependencies
4. **Temporal Relevance** (weight: 0.1): Recent changes, edit frequency

**Context Pruning Strategy**:
- Start with high-relevance items
- Expand to include dependencies until 50% of context window
- Add semantic similar items until 75% of context window
- Reserve 25% for agent's working memory

## 3. Implementation Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│                    RIF Agents Layer                      │
│  (Analyst, Architect, Implementer, Validator, Learner)   │
└────────────────────┬────────────────────────────────────┘
                     │ Query API
┌────────────────────▼────────────────────────────────────┐
│              Knowledge Query Interface                   │
│  • Natural Language Processing                          │
│  • Query Planning & Optimization                        │
│  • Result Ranking & Filtering                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Hybrid Knowledge Store                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  AST Cache   │ │ Vector Index │ │ Graph Store  │   │
│  │(Tree-sitter) │ │  (DuckDB VSS)│ │  (DuckDB)    │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Code Analysis Pipeline                      │
│  • File Monitoring (watchdog)                           │
│  • AST Parsing (tree-sitter)                           │
│  • Entity Extraction                                    │
│  • Relationship Detection                               │
│  • Embedding Generation                                 │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

#### 3.2.1 Indexing Flow
1. **File Detection**: Watchdog detects file change
2. **Parsing**: Tree-sitter generates AST
3. **Extraction**: Extract entities and relationships from AST
4. **Enrichment**: 
   - Generate embeddings for semantic search
   - Detect patterns and anti-patterns
   - Link to existing knowledge
5. **Storage**: 
   - Entities → DuckDB entities table
   - Relationships → DuckDB relationships table
   - Embeddings → DuckDB VSS index
6. **Notification**: Alert active agents of relevant changes

#### 3.2.2 Query Flow
1. **Query Receipt**: Agent sends natural language or structured query
2. **Query Planning**: 
   - Parse intent and extract keywords
   - Determine query strategy (vector, graph, hybrid)
   - Estimate result size
3. **Execution**:
   - Vector search for semantic similarity
   - Graph traversal for relationships
   - SQL queries for structured data
4. **Result Synthesis**:
   - Merge results from different sources
   - Apply relevance scoring
   - Prune to fit context window
5. **Response**: Return formatted results to agent

### 3.3 Performance Optimizations

#### 3.3.1 Memory Management
- **Total Budget**: 2GB maximum
  - DuckDB: 500MB (configurable via memory_limit)
  - Tree-sitter: 200MB (cached ASTs with LRU)
  - Python process: 300MB
  - File cache: 500MB (memory-mapped files)
  - Buffer: 500MB for spikes

- **Memory Pressure Handling**:
  1. Evict LRU AST cache entries
  2. Flush DuckDB buffers to disk
  3. Reduce file cache size
  4. Pause indexing if critical

#### 3.3.2 Query Optimization
- **Materialized Views** for common queries:
  - Module dependency graph
  - Function call graph
  - File import/export map
- **Query Result Caching**:
  - 1-hour TTL for structural queries
  - 5-minute TTL for semantic searches
  - Invalidate on relevant file changes
- **Parallel Query Execution**:
  - Vector search and graph traversal in parallel
  - Merge results using scoring algorithm

## 4. Migration Strategy

### 4.1 Phase 1: Parallel Installation (Week 1)
- Install new system alongside LightRAG
- Shadow mode: Index in background without agent queries
- Validate indexing accuracy and performance
- No agent behavior changes

### 4.2 Phase 2: Read Migration (Week 2)
- Route read queries to new system
- Keep LightRAG for writes
- A/B test query results
- Monitor performance metrics

### 4.3 Phase 3: Write Migration (Week 3)
- Migrate write operations to new system
- Dual-write to both systems for rollback capability
- Verify data consistency

### 4.4 Phase 4: Cutover (Week 4)
- Disable LightRAG queries
- Archive LightRAG data
- Remove LightRAG dependencies
- Full system validation

### 4.5 Rollback Plan
Each phase has independent rollback:
- Phase 1: Simply uninstall new components
- Phase 2: Route reads back to LightRAG
- Phase 3: Stop dual-writes, continue with LightRAG
- Phase 4: Restore LightRAG from archive

## 5. Risk Analysis and Mitigation

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Memory exhaustion on large repos | Medium | High | Implement streaming parser, partial indexing |
| Query performance degradation | Low | High | Query optimizer, materialized views |
| Tree-sitter language support gaps | Medium | Medium | Fallback to regex patterns, LSP |
| DuckDB corruption | Low | High | Regular backups, WAL mode |
| File system race conditions | Medium | Low | Proper locking, transaction isolation |

### 5.2 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Agent confusion during migration | Medium | Medium | Compatibility layer, gradual rollout |
| Context window overflow | Low | Medium | Aggressive pruning, summarization |
| Embedding model compatibility | Low | Low | Multiple model support, fallbacks |
| Query language learning curve | High | Low | Query templates, examples |

## 6. GitHub Issues for Implementation

### 6.1 Foundation Phase (Prerequisites)

#### Issue #1: Remove LightRAG Dependencies
**Title**: Decouple RIF agents from LightRAG implementation
**Description**: Create abstraction layer between agents and knowledge system to enable hot-swapping
**Complexity**: Medium
**Dependencies**: None
**Acceptance Criteria**:
- [ ] Knowledge interface defined
- [ ] All agents use interface, not direct LightRAG
- [ ] Tests pass with mock knowledge provider

#### Issue #2: Install DuckDB with VSS Extension
**Title**: Set up DuckDB as embedded database with vector search
**Description**: Install and configure DuckDB with VSS extension for hybrid storage
**Complexity**: Low
**Dependencies**: #1
**Acceptance Criteria**:
- [ ] DuckDB Python package installed
- [ ] VSS extension loaded and tested
- [ ] Basic CRUD operations working
- [ ] Memory limits configured

#### Issue #3: Implement Tree-sitter Parser Manager
**Title**: Create tree-sitter parsing infrastructure
**Description**: Build parser manager for multiple languages with caching
**Complexity**: Medium
**Dependencies**: None
**Acceptance Criteria**:
- [ ] Tree-sitter Python bindings installed
- [ ] Language grammars for JS, Python, Go loaded
- [ ] Parse sample files successfully
- [ ] AST cache with LRU eviction

### 6.2 Core Implementation Phase

#### Issue #4: Design and Create Database Schema
**Title**: Implement DuckDB schema for knowledge graph
**Description**: Create tables for entities, relationships, and agent memory
**Complexity**: Low
**Dependencies**: #2
**Acceptance Criteria**:
- [ ] Schema created and migrated
- [ ] Indexes optimized for query patterns
- [ ] Test data inserted successfully

#### Issue #5: Build File Monitoring System
**Title**: Implement real-time file monitoring with watchdog
**Description**: Create file watcher with debouncing and priority queue
**Complexity**: Medium
**Dependencies**: #3
**Acceptance Criteria**:
- [ ] Watchdog monitoring project files
- [ ] Changes debounced correctly
- [ ] Priority queue working
- [ ] Memory usage acceptable

#### Issue #6: Create Entity Extraction Pipeline
**Title**: Extract code entities from AST
**Description**: Parse AST to extract functions, classes, modules with metadata
**Complexity**: High
**Dependencies**: #3, #4
**Acceptance Criteria**:
- [ ] Extract entities from JavaScript
- [ ] Extract entities from Python
- [ ] Store in DuckDB with relationships
- [ ] Handle errors gracefully

#### Issue #7: Implement Relationship Detection
**Title**: Detect and store code relationships
**Description**: Analyze AST for imports, calls, inheritance relationships
**Complexity**: High
**Dependencies**: #6
**Acceptance Criteria**:
- [ ] Detect import relationships
- [ ] Detect function calls
- [ ] Detect class inheritance
- [ ] Store in graph structure

#### Issue #8: Add Embedding Generation
**Title**: Generate and store vector embeddings
**Description**: Create embeddings for code entities using local model
**Complexity**: Medium
**Dependencies**: #6
**Acceptance Criteria**:
- [ ] Local embedding model loaded
- [ ] Embeddings generated for entities
- [ ] Stored in DuckDB VSS index
- [ ] Query by similarity working

### 6.3 Query Interface Phase

#### Issue #9: Build Query Planning Engine
**Title**: Create query planner for hybrid searches
**Description**: Parse queries and plan execution strategy
**Complexity**: High
**Dependencies**: #7, #8
**Acceptance Criteria**:
- [ ] Parse natural language queries
- [ ] Plan vector vs graph strategy
- [ ] Optimize query execution
- [ ] Return results efficiently

#### Issue #10: Implement Context Window Optimizer
**Title**: Optimize context for agent consumption
**Description**: Rank and prune results to fit context window
**Complexity**: Medium
**Dependencies**: #9
**Acceptance Criteria**:
- [ ] Relevance scoring implemented
- [ ] Context pruning working
- [ ] Results fit context window
- [ ] Quality metrics acceptable

#### Issue #11: Create Agent Memory System
**Title**: Store and query agent conversations
**Description**: Track agent decisions, errors, and learnings
**Complexity**: Medium
**Dependencies**: #4
**Acceptance Criteria**:
- [ ] Agent outputs stored
- [ ] Searchable by error patterns
- [ ] Learnings extracted
- [ ] Memory queryable

### 6.4 Integration Phase

#### Issue #12: Build Compatibility Layer
**Title**: Create LightRAG compatibility interface
**Description**: Implement adapter for gradual migration
**Complexity**: Medium
**Dependencies**: #10
**Acceptance Criteria**:
- [ ] Old queries still work
- [ ] Results format compatible
- [ ] Performance acceptable
- [ ] No agent changes needed

#### Issue #13: Implement Shadow Mode
**Title**: Run new system in parallel for testing
**Description**: Index and query without affecting agents
**Complexity**: Low
**Dependencies**: #12
**Acceptance Criteria**:
- [ ] Parallel indexing working
- [ ] Results compared and logged
- [ ] Performance monitored
- [ ] No agent impact

#### Issue #14: Add Monitoring and Metrics
**Title**: Implement system monitoring
**Description**: Track performance, memory, and query metrics
**Complexity**: Low
**Dependencies**: #13
**Acceptance Criteria**:
- [ ] Memory usage tracked
- [ ] Query latency measured
- [ ] Indexing speed monitored
- [ ] Alerts configured

### 6.5 Migration Phase

#### Issue #15: Execute Gradual Cutover
**Title**: Migrate from LightRAG to new system
**Description**: Execute phased migration plan
**Complexity**: Medium
**Dependencies**: #14
**Acceptance Criteria**:
- [ ] Read traffic migrated
- [ ] Write traffic migrated
- [ ] LightRAG disabled
- [ ] Rollback tested

## 7. Success Metrics

### 7.1 Performance Metrics
- **Query Latency**: P95 < 100ms, P99 < 500ms
- **Indexing Speed**: >1000 files/minute
- **Memory Usage**: <2GB for 50k file project
- **CPU Usage**: <30% during indexing, <5% idle

### 7.2 Quality Metrics
- **Query Accuracy**: >90% relevant results in top 10
- **Context Efficiency**: <50% of context window used on average
- **Agent Success Rate**: >20% improvement in task completion
- **Error Recovery**: >80% of failures auto-recovered

### 7.3 Adoption Metrics
- **Agent Utilization**: All agents using new system
- **Query Volume**: >1000 queries/day
- **Knowledge Growth**: >100 new entities/day
- **User Satisfaction**: Improved development velocity

## 8. Edge Cases and Considerations

### 8.1 Large Codebases (>100k files)
- **Strategy**: Selective indexing based on .gitignore and usage patterns
- **Implementation**: 
  - Index only active working set
  - Background index remaining files
  - Use file popularity for cache priority

### 8.2 Mixed Language Projects
- **Strategy**: Language-specific parsers with unified representation
- **Implementation**:
  - Detect language from file extension
  - Load appropriate tree-sitter grammar
  - Normalize entity types across languages

### 8.3 Rapid Concurrent Edits
- **Strategy**: Event coalescing and batch processing
- **Implementation**:
  - 500ms debounce window
  - Merge related changes
  - Priority queue for active files

### 8.4 Network File Systems
- **Strategy**: Local caching with periodic sync
- **Implementation**:
  - Cache parsed ASTs locally
  - Sync on file modification time change
  - Handle stale cache gracefully

### 8.5 Symbolic Links and Unusual Structures
- **Strategy**: Canonical path resolution
- **Implementation**:
  - Resolve all symlinks to real paths
  - Detect and prevent circular references
  - Handle missing link targets

## 9. Future Enhancements

### 9.1 Phase 2 Features (Post-MVP)
- **Multi-repository Knowledge**: Federated search across projects
- **AI-Powered Refactoring**: Suggest improvements based on patterns
- **Collaborative Memory**: Share learnings across team members
- **Visual Architecture Explorer**: Interactive dependency graphs

### 9.2 Advanced Capabilities
- **Predictive Assistance**: Anticipate developer needs
- **Automated Documentation**: Generate from code understanding
- **Security Vulnerability Detection**: Pattern-based security analysis
- **Performance Optimization**: Identify bottlenecks from patterns

## 10. Conclusion

This hybrid graph-based knowledge system represents a fundamental evolution in how RIF agents understand and work with code. By combining the speed of tree-sitter, the intelligence of vector embeddings, and the relationships of graph structures, we create a system that provides precisely relevant context while maintaining performance on constrained hardware.

The phased implementation approach ensures smooth migration from LightRAG while maintaining system stability. Each GitHub issue is designed to be implementable within a single Claude Code context window, enabling the RIF system itself to execute this upgrade.

Success will be measured not just in technical metrics, but in the fundamental improvement in how RIF agents complete development tasks - with better understanding, fewer errors, and more intelligent decision-making.