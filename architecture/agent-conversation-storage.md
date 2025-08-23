# Agent Conversation Storage Architecture

## Overview

The agent conversation storage system implements an event sourcing pattern combined with vector embeddings to capture, store, and query all agent interactions, decisions, and learnings in a searchable knowledge base.

## Design Pattern: Event Sourcing with Vector Storage

### Core Components

1. **Conversation Capture Hooks** - Automatic capture of agent interactions
2. **Event Stream Processor** - Real-time event processing and storage
3. **Conversation Parser** - Structure extraction from agent interactions
4. **Vector Embedding Generator** - Semantic embeddings for search
5. **Hybrid Storage Layer** - DuckDB with vector search capabilities
6. **Query Interface** - Natural language and structured conversation queries
7. **Analytics Engine** - Pattern detection and learning extraction

### System Architecture

```
RIF Agents → Conversation Capture Hooks → Event Stream Processor
                                       ↓
                           Conversation Parser & Metadata Extractor
                                       ↓
                           Vector Embedding Generator
                                       ↓
                  Hybrid Storage Layer (DuckDB + Vector Index)
                                       ↓
          Query Interface (Natural Language + Structured Filters)
                                       ↓
                Analytics Layer (Pattern Detection + Learning)
```

## Database Schema Design

### Core Tables

#### Conversation Events
```sql
CREATE TABLE conversation_events (
    event_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    issue_number INTEGER,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- start, tool_use, decision, error, completion
    event_data JSON NOT NULL,
    parent_event_id UUID REFERENCES conversation_events(event_id),
    embedding FLOAT[768],
    context_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Decision Points
```sql
CREATE TABLE agent_decisions (
    decision_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversation_events(conversation_id),
    agent_type VARCHAR(50) NOT NULL,
    decision_point TEXT NOT NULL,
    options_considered JSON NOT NULL,
    chosen_option TEXT NOT NULL,
    rationale TEXT,
    confidence_score FLOAT DEFAULT 0.5,
    outcome VARCHAR(50),  -- success, failure, partial, unknown
    outcome_timestamp TIMESTAMP,
    learning_value FLOAT DEFAULT 0.0,
    embedding FLOAT[768],
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Error Patterns
```sql
CREATE TABLE conversation_errors (
    error_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversation_events(conversation_id),
    agent_type VARCHAR(50) NOT NULL,
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    error_context JSON,
    resolution_attempted TEXT,
    resolution_success BOOLEAN,
    pattern_signature VARCHAR(128),  -- For grouping similar errors
    embedding FLOAT[768],
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Integration Strategy

### Automatic Hook Integration
The system integrates with agents through automatic hooks that require no code changes:

1. **Tool Usage Hooks** - Capture at tool invocation and completion
2. **Decision Point Hooks** - Manual capture at key decision moments
3. **Error Boundary Hooks** - Automatic capture of all exceptions and failures
4. **Session Lifecycle Hooks** - Conversation start/end tracking

### File Structure
```
lightrag/
├── conversations/
│   ├── __init__.py
│   ├── capture_engine.py           # Main conversation capture
│   ├── storage_backend.py          # DuckDB storage implementation
│   ├── query_engine.py             # Search and analysis interface
│   ├── pattern_detector.py         # Error and decision pattern analysis
│   ├── threading_manager.py        # Conversation threading and context
│   └── analytics_dashboard.py      # Conversation analytics and metrics
├── agents/
│   └── agent_integration.py        # Modified with conversation hooks
└── core/
    └── conversation_hooks.py       # Integration hooks for all agents
```

## Query Capabilities

### Natural Language Search
- Semantic search across all conversation content using vector embeddings
- Context-aware results with relevance scoring
- Cross-conversation pattern recognition

### Structured Queries
- Filter by agent type, time range, error patterns
- Decision outcome analysis and correlation
- Performance metrics and trend analysis

### Pattern Detection
- Automatic identification of recurring issues
- Decision pattern analysis with outcome correlation
- Learning opportunity identification

## Quality Requirements

### Performance Targets
- **Capture Latency**: <10ms overhead per agent interaction
- **Query Response**: <2s for semantic searches across 100k+ conversations
- **Storage Efficiency**: <1MB per hour of agent conversation
- **Concurrent Access**: Support 10+ agents capturing simultaneously

### Reliability Requirements
- **Data Durability**: 99.9% conversation capture success rate
- **Query Accuracy**: >90% relevant results for natural language queries
- **Pattern Detection**: >85% accuracy in identifying recurring issues

## Dependencies
- **Blocking**: Issue #34 (Context optimizer) for vector embedding infrastructure
- **Integrates with**: All RIF agents for conversation capture
- **Enables**: Enhanced learning and pattern recognition across the system

## Implementation Benefits

1. **Complete Audit Trail** - Full history of all agent interactions and decisions
2. **Pattern Learning** - Automatic detection of successful and failed approaches
3. **Error Analysis** - Systematic tracking and correlation of error patterns
4. **Decision Optimization** - Historical analysis of decision outcomes
5. **Continuous Improvement** - Data-driven agent enhancement over time