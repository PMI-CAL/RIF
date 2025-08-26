# Claude Code MCP Server Integration

## What are MCP Servers?

Model Context Protocol (MCP) servers are the **primary integration mechanism** for Claude Code to connect with external tools, databases, and APIs. They provide real-time access to various services and data sources.

## Key Facts

- **Open-source standard** for AI-tool integrations
- Enable Claude Code to connect with external tools, databases, and APIs
- Provide **real-time access** to various services and data sources
- **User responsible** for verifying server trust and security

## Installation Methods

### 1. Local stdio servers
Run as local processes that communicate via standard input/output

### 2. Remote SSE servers  
Server-sent events for streaming real-time data

### 3. Remote HTTP servers
Standard HTTP request/response patterns

## Configuration Scopes

### Local Configuration
- **Personal**, project-specific servers
- Configured per-project in `.claude/settings.json`
- Not shared with other projects or users

### Project Configuration
- **Team-shared** configurations
- Version controlled with the project
- All team members get same MCP servers

### User Configuration
- **Cross-project** accessible servers
- Available across all projects for a user
- Personal productivity tools and integrations

## Notable Capabilities

### Authentication
- **OAuth 2.0 authentication support**
- Secure credential management
- Token refresh and validation

### Configuration
- **Environment variable expansion**
- Dynamic configuration based on environment
- Flexible parameter passing

### Ecosystem
- **Hundreds of available integrations**
- GitHub, databases, APIs, monitoring tools
- Community-contributed servers

## RIF Integration Patterns

### GitHub Integration via MCP
Instead of assuming independent agents can post to GitHub:

```json
{
  "mcpServers": {
    "github": {
      "command": "gh-mcp-server",
      "args": ["--token", "${GITHUB_TOKEN}"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Database Integration via MCP
For accessing the RIF knowledge database:

```json
{
  "mcpServers": {
    "rif-knowledge": {
      "command": "rif-knowledge-mcp-server",
      "args": ["--database", "./knowledge/hybrid_knowledge.duckdb"],
      "cwd": "${PROJECT_ROOT}"
    }
  }
}
```

### API Integration via MCP
For external service integrations:

```json
{
  "mcpServers": {
    "monitoring": {
      "command": "monitoring-mcp-server",
      "args": ["--endpoint", "${MONITORING_ENDPOINT}"],
      "env": {
        "API_KEY": "${MONITORING_API_KEY}"
      }
    }
  }
}
```

## Security Considerations

### Trust Model
- **User responsible** for verifying server trust
- MCP servers run with user permissions
- Review server code before installation

### Credential Management  
- Use environment variables for secrets
- Never hardcode credentials in configuration
- Implement proper token rotation

### Network Security
- Validate SSL certificates for remote servers
- Use secure communication channels
- Monitor server network activity

## Common MCP Server Types for RIF

### 1. GitHub MCP Server
- **Purpose**: Direct GitHub API integration
- **Capabilities**: Issue management, PR creation, repository access
- **Installation**: via npm or pip package

### 2. Database MCP Server
- **Purpose**: Knowledge base access
- **Capabilities**: Query patterns, decisions, metrics
- **Installation**: Custom RIF knowledge server

### 3. File System MCP Server
- **Purpose**: Advanced file operations
- **Capabilities**: Monitoring, batch operations, search
- **Installation**: Standard filesystem MCP server

### 4. Web API MCP Server
- **Purpose**: External service integration
- **Capabilities**: REST API calls, webhook handling
- **Installation**: Generic HTTP MCP server

## Implementation Benefits

### Replaces Complex Agent Architecture
Instead of:
```
Independent Agent → GitHub API
```

Use:
```
Claude Code → GitHub MCP Server → GitHub API
```

### Simplifies Integration
Instead of:
```
Complex automation assumptions
```

Use:
```
MCP server providing real-time API access
```

### Enables Real Functionality
Instead of:
```
Assumed Task() tool orchestration
```

Use:
```
MCP servers for actual external integration
```

## Configuration Examples

### Minimal RIF MCP Configuration
```json
{
  "mcpServers": {
    "github": {
      "command": "gh-mcp-server"
    },
    "rif-knowledge": {
      "command": "python",
      "args": ["./mcp/rif-knowledge-server/rif_knowledge_server.py"],
      "cwd": "${PROJECT_ROOT}"
    }
  }
}
```

### Production RIF MCP Configuration
```json
{
  "mcpServers": {
    "github": {
      "command": "gh-mcp-server",
      "args": ["--token", "${GITHUB_TOKEN}"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "GITHUB_OWNER": "${GITHUB_OWNER}",
        "GITHUB_REPO": "${GITHUB_REPO}"
      }
    },
    "rif-knowledge": {
      "command": "python", 
      "args": ["./mcp/rif-knowledge-server/rif_knowledge_server.py"],
      "cwd": "${PROJECT_ROOT}",
      "env": {
        "DATABASE_PATH": "./knowledge/hybrid_knowledge.duckdb",
        "DEBUG": "false"
      }
    },
    "monitoring": {
      "command": "monitoring-mcp-server",
      "args": ["--config", "./config/monitoring.json"],
      "env": {
        "MONITORING_ENDPOINT": "${MONITORING_ENDPOINT}",
        "API_KEY": "${MONITORING_API_KEY}"
      }
    }
  }
}
```

## Testing MCP Integration

### Verify Server Connection
```bash
# Test MCP server directly
echo '{"method": "initialize", "id": 1}' | python ./mcp/rif-knowledge-server/rif_knowledge_server.py
```

### Test via Claude Code
1. Configure MCP server in `.claude/settings.json`
2. Start Claude Code session
3. Verify server appears in available tools
4. Test server functionality

## Common Issues and Solutions

### Server Not Loading
- Check command path and arguments
- Verify environment variables are set
- Check server logs for error messages
- Validate JSON configuration syntax

### Permission Errors
- Ensure script has execute permissions
- Check working directory exists and is accessible
- Verify user has required system permissions

### Authentication Failures
- Validate environment variables contain correct values
- Check token expiration and refresh needs
- Verify API endpoints and credentials

## Migration from Task-Based Architecture

### Before (RIF Assumption)
```python
# This doesn't work - Task tool doesn't exist
Task(
    description="Get GitHub issues",
    subagent_type="github-agent"
)
```

### After (MCP Reality)  
```python
# This works - MCP server provides real GitHub integration
# Claude Code can now directly access GitHub via MCP server
# Use GitHub MCP server tools to list issues, create PRs, etc.
```

## Best Practices

### Performance
- Use local MCP servers when possible for better performance
- Implement caching in custom MCP servers
- Monitor server resource usage

### Reliability
- Implement proper error handling in custom servers
- Use connection pooling for database servers
- Set appropriate timeouts

### Maintainability
- Document custom MCP server APIs
- Version control MCP server configurations
- Test MCP servers independently of Claude Code

---

*MCP servers are the real solution for Claude Code external integration, replacing the fictional Task-based orchestration architecture.*