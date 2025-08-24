# Claude Code Knowledge MCP Server

A Model Context Protocol (MCP) server providing Claude Code capability knowledge and implementation guidance through the RIF knowledge graph system.

## Overview

This MCP server acts as a lightweight query interface over the existing RIF knowledge graph, specifically designed to provide Claude Code users with:

- **Compatibility checking** - Validate approaches against Claude Code capabilities
- **Pattern recommendations** - Suggest correct implementation patterns  
- **Alternative solutions** - Find compatible alternatives to problematic approaches
- **Architecture validation** - Check system designs against Claude Code constraints
- **Limitation queries** - Get specific limitations with workarounds

## Architecture

```
Claude Code → MCP Client → Claude Code Knowledge Server → RIF Knowledge Graph (DuckDB)
                                                     → Vector Search Engine
                                                     → Cached Query Results
```

### Key Components

- **MCP Server Core** (`server.py`) - JSON-RPC 2.0 compliant server
- **Query Engine** (`query_engine.py`) - Optimized knowledge graph queries
- **Safety Module** (`safety.py`) - Input validation, output sanitization, graceful degradation
- **Configuration** (`config.py`) - Centralized configuration management

## Installation

### Prerequisites

- Python 3.8+
- RIF knowledge graph with Phase 1 Claude Code entities
- DuckDB 0.9.0+
- NumPy 1.21.0+

### Setup

1. **Ensure Phase 1 is complete**:
   ```bash
   # Verify Claude Code entities exist
   python3 -c "from knowledge.database.database_interface import RIFDatabase; db = RIFDatabase(); print(len(db.search_entities(entity_types=['claude_capability'], limit=5)))"
   ```

2. **Install dependencies**:
   ```bash
   pip install duckdb numpy
   ```

3. **Configure server**:
   ```bash
   cd mcp/claude-code-knowledge
   python3 config.py --create-default
   ```

4. **Test installation**:
   ```bash
   python3 server.py --debug
   ```

## Configuration

### Configuration File (`config.json`)

```json
{
  "log_level": "INFO",
  "cache_size": 100,
  "cache_ttl": 300,
  "timeout_seconds": 30,
  "target_response_time_ms": 200,
  "enable_caching": true,
  "database_path": "knowledge/hybrid_knowledge.duckdb"
}
```

### Environment Variables

- `MCP_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `MCP_CACHE_SIZE` - Maximum cached queries (default: 100)
- `MCP_ENABLE_CACHING` - Enable/disable caching (default: true)
- `RIF_ROOT` - Path to RIF root directory

## MCP Tools

### 1. check_compatibility

Validates proposed solutions against Claude Code capabilities and limitations.

**Parameters:**
- `issue_description` (required) - Description of the issue or requirement
- `approach` (optional) - Proposed solution approach

**Response:**
```json
{
  "compatible": true,
  "confidence": 0.85,
  "concepts_analyzed": 3,
  "issues": [],
  "recommendations": [],
  "execution_time_ms": 45.2
}
```

**Example Usage:**
```python
result = await server._check_compatibility({
    "issue_description": "Need to process files in parallel",
    "approach": "Use Task() to launch multiple file processors"
})
```

### 2. recommend_pattern

Returns implementation patterns for specific technology and task type.

**Parameters:**
- `technology` (required) - Technology stack (e.g., "Python", "JavaScript")
- `task_type` (required) - Task type (e.g., "file_processing", "api_integration")
- `limit` (optional) - Maximum patterns to return (default: 5, max: 10)

**Response:**
```json
{
  "patterns": [
    {
      "pattern_id": "pat_123",
      "name": "Direct Tool Usage Pattern",
      "description": "Use built-in tools directly without orchestration",
      "technology": "Python",
      "task_type": "file_processing",
      "code_example": "Read(file_path='/path/to/file')",
      "confidence": 0.92,
      "supporting_tools": ["Read", "Write", "Edit"],
      "usage_count": 45
    }
  ],
  "total_found": 1
}
```

### 3. find_alternatives

Proposes compatible solutions when incompatible approaches are detected.

**Parameters:**
- `problematic_approach` (required) - The problematic approach needing alternatives

**Response:**
```json
{
  "alternatives": [
    {
      "id": "alt_456", 
      "name": "Direct Tool Integration",
      "description": "Use Claude Code tools directly instead of orchestration",
      "confidence": 0.88,
      "technology": "general"
    }
  ],
  "total_found": 1
}
```

### 4. validate_architecture

Reviews system design against Claude Code architectural constraints.

**Parameters:**
- `system_design` (required) - Description of the proposed system architecture

**Response:**
```json
{
  "valid": false,
  "confidence": 0.65,
  "components_analyzed": 4,
  "issues_found": [
    {
      "issue": "Orchestrator patterns not supported",
      "recommendation": "Use direct tool calls instead",
      "severity": "high"
    }
  ],
  "recommendations": [
    "Use direct tool usage patterns",
    "Leverage MCP servers for complex integrations"
  ]
}
```

### 5. query_limitations

Returns known limitations for specific capability areas.

**Parameters:**
- `capability_area` (required) - Area to query (e.g., "orchestration", "networking")
- `severity` (optional) - Filter by severity ("low", "medium", "high")

**Response:**
```json
{
  "limitations": [
    {
      "limitation_id": "lim_789",
      "name": "No Task Orchestration",
      "category": "orchestration", 
      "description": "Cannot use Task() for orchestration",
      "severity": "high",
      "workarounds": ["Use direct tool calls", "Delegate to subagents"],
      "alternatives": []
    }
  ],
  "capability_area": "orchestration",
  "total_found": 1
}
```

## Performance

### Targets
- **Response Time**: <200ms per query
- **Cache Hit Rate**: >70%
- **Concurrent Requests**: Up to 10
- **Memory Usage**: <256MB

### Optimization Features
- Query result caching with TTL
- Connection pooling
- Vector search indexing
- Graceful degradation

## Safety Features

### Input Validation
- Required field validation
- Type checking and sanitization
- Dangerous content detection
- Size limits (10KB max input)

### Output Sanitization  
- Sensitive data filtering
- Response size limits
- Error message sanitization

### Rate Limiting
- 60 requests per minute per client
- Burst capacity of 10 requests
- Token bucket algorithm

### Graceful Degradation
Fallback responses when knowledge graph unavailable:
- Basic compatibility warnings
- Standard implementation patterns
- General best practices

## Integration with Claude Code

### MCP Client Setup

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "claude-code-knowledge": {
      "command": "python3",
      "args": ["-m", "mcp.claude-code-knowledge.server"],
      "env": {
        "PYTHONPATH": "/path/to/rif"
      }
    }
  }
}
```

### Usage in Claude Code

The server provides tools that help Claude Code:

1. **Before implementation** - Check compatibility of proposed approaches
2. **During planning** - Get recommended patterns for specific technologies
3. **When stuck** - Find alternatives to problematic approaches
4. **Architecture review** - Validate system designs
5. **Learning** - Understand specific limitations and workarounds

## Development

### Running Tests

```bash
# Unit tests
cd mcp/claude-code-knowledge
python3 -m pytest tests/test_mcp_tools.py -v

# Integration tests (requires real database)
python3 -m pytest tests/test_integration.py -v

# Performance benchmarks
python3 -m pytest tests/test_performance.py -v
```

### Development Setup

```bash
# Install development dependencies
pip install pytest pytest-asyncio psutil

# Enable debug logging
export MCP_LOG_LEVEL=DEBUG
export MCP_ENABLE_DEBUG=true

# Run server in development mode
python3 server.py --debug --config config/dev.json
```

### Adding New Tools

1. Add tool method to `ClaudeCodeKnowledgeServer` class
2. Register tool in `_register_tools()` method
3. Add query methods to `ClaudeKnowledgeQueryEngine`
4. Add input validation to `InputValidator`
5. Write unit and integration tests
6. Update documentation

## Troubleshooting

### Common Issues

**Server won't start**:
```bash
# Check RIF database connection
python3 -c "from knowledge.database.database_interface import RIFDatabase; RIFDatabase()"

# Verify Claude entities exist
python3 knowledge/schema/seed_claude_knowledge.py --verify
```

**Slow responses**:
```bash
# Check cache hit rate
curl http://localhost:8080/metrics

# Enable query logging
export MCP_ENABLE_QUERY_LOGGING=true
```

**No results found**:
```bash
# Verify Phase 1 seeding completed
python3 -c "from knowledge.database.database_interface import RIFDatabase; db = RIFDatabase(); print(db.search_entities(entity_types=['claude_capability'], limit=1))"
```

### Health Monitoring

```bash
# Health check endpoint
curl http://localhost:8080/health

# Performance metrics
curl http://localhost:8080/metrics
```

### Logging

Logs are written to stdout/stderr with configurable levels:
- **DEBUG** - Detailed query execution
- **INFO** - Request/response summaries  
- **WARNING** - Performance issues, fallbacks
- **ERROR** - Failures, exceptions

## Security Considerations

- Input validation prevents injection attacks
- Output sanitization removes sensitive data
- Rate limiting prevents resource exhaustion
- Read-only access to knowledge graph
- No persistent state or user data storage

## Roadmap

### Phase 3 (Future)
- WebSocket support for real-time queries
- Custom pattern learning from usage
- Integration with Claude Code hooks
- Performance analytics dashboard
- Multi-language pattern support

## Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review server logs for error details  
3. Test with minimal examples
4. Check RIF knowledge graph integrity
5. File issues with reproduction steps

## License

Part of the RIF (Reactive Intelligence Framework) project.