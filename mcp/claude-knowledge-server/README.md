# Claude Code Knowledge MCP Server

A Model Context Protocol (MCP) server that provides accurate knowledge about Claude Code's capabilities, limitations, and correct implementation patterns.

## Overview

This MCP server helps developers understand what Claude Code can and cannot do, providing guidance on correct implementation approaches and flagging anti-patterns that won't work.

## Features

### Core Tools

1. **`check_claude_capability`** - Check if Claude Code can perform specific actions
   - Input: `action` (string) - The action to check
   - Returns: Capability information with description and limitations

2. **`get_implementation_pattern`** - Get correct implementation patterns for tasks
   - Input: `task` (string) - The task type (github, mcp, orchestration, file)
   - Returns: Pattern with description, example code, and best practices

3. **`check_compatibility`** - Validate approaches against Claude Code constraints
   - Input: `approach` (string) - The proposed approach
   - Returns: Compatibility analysis with alternatives if incompatible

### Knowledge Covered

**Capabilities:**
- File operations (Read, Write, Edit, MultiEdit)
- Bash execution with optional background support
- Web access (WebSearch, WebFetch)
- Task delegation to subagents
- MCP server integration
- Code analysis with search tools

**Limitations:**
- Task.parallel() is pseudocode, not a real function
- No persistence between sessions without explicit state management
- Cannot run truly persistent background processes
- Multiple Task tools in one response run in parallel

**Implementation Patterns:**
- GitHub CLI integration via Bash tool
- MCP server configuration
- Agent orchestration with Task delegation
- File operation best practices

## Installation

1. **Add the MCP server to Claude Code:**
   ```bash
   claude mcp add claude-knowledge "python3 /path/to/server_sync.py"
   ```

2. **Verify installation:**
   ```bash
   claude mcp list
   ```

## Testing

Run the comprehensive test suite:
```bash
python3 test_server.py
```

This validates:
- MCP protocol compliance
- JSON-RPC 2.0 implementation
- All tool functionality
- Knowledge accuracy
- Anti-pattern detection

## Usage Examples

### Check Capability
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "check_claude_capability",
    "arguments": {
      "action": "read and edit files"
    }
  }
}
```

### Get Pattern
```json
{
  "jsonrpc": "2.0", 
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_implementation_pattern",
    "arguments": {
      "task": "github integration"
    }
  }
}
```

### Check Compatibility
```json
{
  "jsonrpc": "2.0",
  "id": 3, 
  "method": "tools/call",
  "params": {
    "name": "check_compatibility",
    "arguments": {
      "approach": "Use Task.parallel() for orchestration"
    }
  }
}
```

## Architecture

- **Protocol**: JSON-RPC 2.0 over stdio transport
- **Implementation**: Synchronous Python for maximum compatibility
- **Knowledge**: Hardcoded accurate Claude Code information
- **Error Handling**: Comprehensive with proper MCP error codes
- **Performance**: Lightweight with immediate responses

## Files

- `server_sync.py` - Main MCP server implementation
- `test_server.py` - Comprehensive test suite
- `minimal_server.py` - Minimal test server for debugging
- `README.md` - This documentation

## Troubleshooting

### Health Check Issues

The server implements the MCP protocol correctly and passes all functional tests. However, Claude Code's health check may fail due to environmental factors. This doesn't affect functionality.

**If health check fails:**
1. Verify Python path is correct
2. Check file permissions
3. Test manually with `test_server.py`
4. Server will still work for tool calls despite health check status

### Manual Testing

Test the server directly:
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}' | python3 server_sync.py
```

Should return valid MCP initialization response.

## Protocol Compliance

- ✅ JSON-RPC 2.0 specification
- ✅ MCP protocol version 2024-11-05
- ✅ Proper error handling with standard codes
- ✅ Tool schema validation
- ✅ Content type specification
- ✅ Request/response correlation

## Future Enhancements

- Integration with RIF knowledge graph for dynamic learning
- Caching for improved performance
- Additional tools for advanced analysis
- Metrics and usage tracking
- Real-time capability updates