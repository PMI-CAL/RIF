# Claude Code Knowledge MCP Server - User Guide

## Overview

The Claude Code Knowledge MCP Server provides Claude Code with access to the RIF (Reactive Intelligence Framework) knowledge graph for intelligent compatibility checks, pattern recommendations, and architectural guidance. This server implements the Model Context Protocol (MCP) to seamlessly integrate with Claude Code.

## Features

### 🔍 **Compatibility Analysis**
- Check if proposed approaches are compatible with Claude Code capabilities
- Get confidence scores and detailed recommendations
- Identify potential conflicts before implementation

### 🎯 **Pattern Recommendations** 
- Get proven implementation patterns from the RIF knowledge base
- Technology-specific recommendations (Python, JavaScript, etc.)
- Complexity-aware suggestions

### 🔀 **Alternative Solutions**
- Find alternatives when approaches are problematic
- Constraint-aware replacements
- Multiple fallback strategies

### 🏗️ **Architecture Validation**
- Validate system architecture against Claude Code capabilities
- Component-level analysis
- Design alignment verification

### ⚠️ **Limitation Insights**
- Query specific capability limitations
- Severity-filtered results
- Workaround suggestions

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the MCP server directory
cd /path/to/rif/mcp/claude-code-knowledge

# Verify prerequisites
./start.sh health
```

### 2. Start the Server

```bash
# Start with default settings
./start.sh start

# Start with debug logging
MCP_LOG_LEVEL=DEBUG ./start.sh start

# Check server status
./start.sh status
```

### 3. Configure Claude Code

Add the server to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "claude-code-knowledge": {
      "command": "python3",
      "args": ["/path/to/rif/mcp/claude-code-knowledge/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/rif/mcp/claude-code-knowledge:/path/to/rif"
      }
    }
  }
}
```

## Tools Reference

### check_compatibility

Check if a proposed approach is compatible with Claude Code capabilities.

**Parameters:**
- `issue_description` (required): Description of the issue or task
- `approach` (required): Proposed solution approach

**Example:**
```json
{
  "issue_description": "Need to process multiple files efficiently",
  "approach": "Using direct Read and Write tools with parallel processing"
}
```

**Response:**
```json
{
  "compatible": true,
  "confidence": 0.85,
  "summary": "Approach is compatible with Claude Code capabilities",
  "issues": [],
  "recommendations": [
    "Consider using batch operations for better performance",
    "Implement error handling for file access failures"
  ]
}
```

### recommend_pattern

Get implementation pattern recommendations based on task requirements.

**Parameters:**
- `task_description` (required): Description of the task
- `technology` (required): Technology stack (Python, JavaScript, etc.)
- `complexity` (required): One of: low, medium, high, very-high
- `limit` (optional): Maximum patterns to return (default: 5)

**Example:**
```json
{
  "task_description": "Build a file processing pipeline with error handling",
  "technology": "Python",
  "complexity": "medium",
  "limit": 3
}
```

**Response:**
```json
{
  "patterns": [
    {
      "name": "Sequential File Processor",
      "description": "Process files one by one with comprehensive error handling",
      "code_example": "# Example implementation...",
      "confidence": 0.92,
      "usage_count": 15
    }
  ]
}
```

### find_alternatives

Find alternative approaches when the proposed solution has issues.

**Parameters:**
- `problematic_approach` (required): The problematic approach
- `context` (required): Context or requirements
- `constraint_type` (optional): technical, performance, security, or compatibility

**Example:**
```json
{
  "problematic_approach": "Direct file manipulation without error handling",
  "context": "Need to process user-uploaded files safely",
  "constraint_type": "security"
}
```

### validate_architecture

Validate architectural components and design decisions.

**Parameters:**
- `architecture_description` (required): System architecture description
- `components` (required): List of architectural components
- `technology_stack` (optional): Technology stack being used

**Example:**
```json
{
  "architecture_description": "Microservices architecture with API gateway",
  "components": ["api-gateway", "user-service", "file-processor", "database"],
  "technology_stack": "Python Flask with PostgreSQL"
}
```

### query_limitations

Query specific limitations for capability areas.

**Parameters:**
- `capability_area` (required): Area to query (e.g., 'file_operations', 'network_access')
- `severity_filter` (optional): low, medium, high, or critical
- `include_workarounds` (optional): Whether to include workarounds (default: true)

**Example:**
```json
{
  "capability_area": "file_operations",
  "severity_filter": "high",
  "include_workarounds": true
}
```

## Configuration

### Environment Variables

- `MCP_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MCP_CACHE_ENABLED`: Enable caching (true/false)
- `MCP_DEBUG_MODE`: Enable debug mode (true/false)

### Configuration File

The server uses `/mcp/claude-code-knowledge/mcp.json` for configuration:

```json
{
  "cache_size": 100,
  "cache_ttl": 300,
  "timeout_seconds": 30,
  "max_request_size_mb": 1,
  "target_response_time_ms": 200
}
```

## Performance

### Target Metrics
- **Response Time**: <200ms average
- **Throughput**: 10+ concurrent requests
- **Availability**: 99.9% uptime
- **Cache Hit Rate**: >70%

### Performance Tuning

1. **Increase Cache Size**: Adjust `cache_size` in configuration
2. **Optimize Database**: Ensure RIF knowledge graph is properly indexed
3. **Memory Allocation**: Increase available memory for better caching
4. **Concurrent Limits**: Adjust `max_concurrent_requests` based on system capacity

## Troubleshooting

### Common Issues

#### Server Won't Start
```bash
# Check health
./start.sh health

# Check logs
./start.sh logs error

# Verify prerequisites
python3 -c "import asyncio, json, logging"
```

#### Slow Response Times
```bash
# Enable debug logging
MCP_LOG_LEVEL=DEBUG ./start.sh restart

# Check system resources
top -p $(cat .server.pid)

# Review cache hit rates in logs
./start.sh logs | grep "cache"
```

#### Connection Errors
- Verify PYTHONPATH includes both MCP server and RIF root directories
- Check that RIF knowledge database is accessible
- Ensure no port conflicts with other services

### Error Codes

- `400`: Invalid request parameters
- `404`: Requested resource not found
- `500`: Internal server error
- `503`: Service temporarily unavailable

## Advanced Usage

### Custom Patterns

The server learns from the RIF knowledge graph. To add custom patterns:

1. Add patterns to RIF knowledge base
2. Restart the server to reload patterns
3. Verify patterns are available via `recommend_pattern`

### Monitoring

Monitor server health and performance:

```bash
# Continuous monitoring
watch -n 5 './start.sh status'

# Log analysis
tail -f logs/server.log | grep "performance"

# Generate performance reports
python3 performance_benchmark.py
```

### Integration with CI/CD

```yaml
# Example GitHub Action
- name: Start MCP Server
  run: |
    cd mcp/claude-code-knowledge
    ./start.sh start
    sleep 5
    ./start.sh status

- name: Run Compatibility Checks
  run: |
    # Your compatibility check script here
    python3 check_compatibility.py

- name: Stop MCP Server
  run: |
    cd mcp/claude-code-knowledge
    ./start.sh stop
```

## API Reference

### Request Format

All requests follow JSON-RPC 2.0 specification:

```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "tool_name",
  "params": {
    "parameter1": "value1",
    "parameter2": "value2"
  }
}
```

### Response Format

Successful responses:
```json
{
  "jsonrpc": "2.0", 
  "id": "unique-request-id",
  "result": {
    "tool_specific_data": "..."
  }
}
```

Error responses:
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id", 
  "error": {
    "code": -1000,
    "message": "Error description",
    "data": {}
  }
}
```

## Security Considerations

### Input Validation
- All inputs are validated for type, length, and content
- Dangerous content is automatically filtered
- Request size limits prevent memory exhaustion

### Output Sanitization
- All outputs are sanitized to prevent XSS
- Sensitive information is filtered from responses
- Error messages don't expose system internals

### Access Control
- Server runs with minimal privileges
- No direct file system access beyond designated directories
- Network access is restricted to essential operations

## Support

### Debugging

1. **Enable Debug Mode**: Set `MCP_DEBUG_MODE=true`
2. **Check Logs**: Use `./start.sh logs` to view detailed logs
3. **Run Health Checks**: Execute `./start.sh health`
4. **Performance Testing**: Run `python3 performance_benchmark.py`

### Getting Help

- **Documentation**: See README.md for technical details
- **Issue Tracking**: Report issues through the RIF GitHub repository  
- **Performance Issues**: Run benchmarks and include results with reports
- **Configuration Problems**: Include configuration files and error logs

### Version Information

- **MCP Server Version**: 1.0.0
- **MCP Protocol Version**: 1.0+
- **Python Requirements**: 3.8+
- **RIF Compatibility**: Latest version

---

## Examples

### Example 1: File Processing Compatibility Check

```python
import asyncio
from server import ClaudeCodeKnowledgeServer, MCPRequest

async def check_file_processing():
    server = ClaudeCodeKnowledgeServer()
    await server.initialize()
    
    request = MCPRequest(
        id="example-1",
        method="check_compatibility",
        params={
            "issue_description": "Process 1000 CSV files and generate reports",
            "approach": "Use Read tool to load files, pandas for processing, Write tool for outputs"
        }
    )
    
    response = await server.handle_request(request.__dict__)
    print(f"Compatible: {response['result']['compatible']}")
    print(f"Recommendations: {response['result']['recommendations']}")
    
    await server.shutdown()

asyncio.run(check_file_processing())
```

### Example 2: Pattern Recommendation

```python
async def get_patterns():
    server = ClaudeCodeKnowledgeServer()
    await server.initialize()
    
    request = MCPRequest(
        id="example-2",
        method="recommend_pattern",
        params={
            "task_description": "Web scraping with rate limiting and error handling",
            "technology": "Python",
            "complexity": "high"
        }
    )
    
    response = await server.handle_request(request.__dict__)
    
    for pattern in response['result']['patterns']:
        print(f"Pattern: {pattern['name']}")
        print(f"Confidence: {pattern['confidence']}")
        print(f"Code: {pattern['code_example'][:100]}...")
    
    await server.shutdown()

asyncio.run(get_patterns())
```

This guide provides comprehensive information for users to effectively utilize the Claude Code Knowledge MCP Server for their development needs.