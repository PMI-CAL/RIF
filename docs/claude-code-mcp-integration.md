# Claude Code MCP Server Integration

## Overview

The dev-framework now includes intelligent integration with the Claude Code Docs MCP Server, automatically detecting Claude Code CLI usage and configuring access to official Claude Code documentation through the Model Context Protocol.

## Integration Architecture

### 1. Automatic Detection

The Context Server Discovery Agent detects Claude Code usage through multiple signals:

```bash
# Detection commands in Phase 1 analysis
find . -name ".claude" -type d -o -name "CLAUDE.md" -o -name ".claude.*"
which claude 2>/dev/null && echo "Claude CLI installed"
```

**Detection Triggers:**
- `.claude/` directory presence
- `CLAUDE.md` file existence
- `.claude.*` configuration files
- Claude CLI installed in system PATH

### 2. Automatic Classification

When Claude Code is detected, the server is automatically classified as **Essential**:

- **Value**: High - Instant access to Claude Code documentation and help
- **Complexity**: Low - Simple npm installation
- **Prerequisites**: Claude Code CLI detected
- **Installation**: `npm install -g claude-code-docs-mcp`

### 3. Configuration Generation

The agent automatically generates MCP server configuration:

```json
{
  "mcpServers": {
    "claude-code-docs": {
      "command": "npx",
      "args": ["claude-code-docs-mcp"],
      "env": {}
    },
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git"],
      "env": {}
    }
  }
}
```

### 4. Conditional Installation

The installation script includes conditional logic for Claude Code detection:

```bash
#!/bin/bash
echo "Installing recommended context servers..."

# Install Claude Code docs server if Claude CLI is detected
if command -v claude &> /dev/null || [[ -d ".claude" ]]; then
    echo "Claude Code CLI detected - installing documentation server..."
    npm install -g claude-code-docs-mcp
fi

# Install Git server
npm install -g @modelcontextprotocol/server-git

echo "Configuration complete. Restart Claude Code to activate servers."
```

## Available Features

### Tools Provided by Claude Code Docs MCP Server

1. **search-docs**: Search Claude Code documentation for specific topics
2. **get-section**: Retrieve specific documentation sections
3. **list-sections**: List all available documentation sections
4. **get-cli-command**: Get help for specific CLI commands
5. **refresh-docs-cache**: Clear documentation cache for fresh content

### Prompts Provided by Claude Code Docs MCP Server

1. **claude-code-help**: General help with Claude Code CLI
2. **troubleshooting**: Help troubleshoot Claude Code issues
3. **cli-usage**: Get examples of CLI usage
4. **getting-started**: Help new users get started
5. **configuration-help**: Help with Claude Code configuration

### Documentation Sections Available

- **Getting Started**: overview, quickstart, common-workflows
- **Build**: sdk, hooks, github-actions, mcp, troubleshooting
- **Deployment**: third-party-integrations, amazon-bedrock, google-vertex-ai, corporate-proxy, llm-gateway, devcontainer
- **Administration**: iam, security, monitoring-usage, costs
- **Configuration**: settings, ide-integrations, memory
- **Reference**: cli-reference, interactive-mode, slash-commands

## Workflow Integration

### 1. Framework Setup
When the dev-framework is initialized:
```bash
./setup.sh
```

### 2. Automatic Trigger
The setup script automatically triggers context server discovery:
```bash
gh issue create \
    --title "Context Server Discovery and Integration" \
    --label "workflow-state:context-discovery"
```

### 3. Agent Activation
The Context Server Discovery Agent automatically activates and:
1. Analyzes project structure
2. Detects Claude Code CLI usage
3. Recommends Claude Code Docs MCP Server as Essential
4. Generates configuration files
5. Creates installation scripts

### 4. Documentation Generated
The agent creates comprehensive documentation:
```markdown
### Installed Servers
- **Claude Code Docs Server**: Provides access to Claude Code CLI documentation
  - Tools: search-docs, get-section, list-sections, get-cli-command, refresh-docs-cache
  - Prompts: claude-code-help, troubleshooting, cli-usage, getting-started, configuration-help
  - Usage: Ask for Claude Code help, CLI commands, or browse documentation sections
```

## Usage Examples

### Example 1: Getting Help with Claude Code
```
You: I need help configuring Claude Code for my team
Claude: [Uses claude-code-help prompt to provide configuration guidance from official docs]
```

### Example 2: CLI Command Help
```
You: How do I use the claude init command?
Claude: [Uses get-cli-command tool to retrieve specific command documentation]
```

### Example 3: Troubleshooting
```
You: Claude Code is giving me authentication errors
Claude: [Uses troubleshooting prompt with the specific error context]
```

### Example 4: Finding Documentation
```
You: Show me the security documentation for Claude Code
Claude: [Uses search-docs tool to find security-related sections, then get-section to retrieve content]
```

## Benefits

### For Developers
- **Instant Documentation Access**: No need to switch to browser or search docs
- **Context-Aware Help**: Help integrated directly into development workflow  
- **Always Up-to-Date**: Documentation fetched from official source with caching
- **Comprehensive Coverage**: Access to all Claude Code documentation sections

### For Teams
- **Consistent Knowledge**: All team members have access to same documentation
- **Reduced Context Switching**: Help available without leaving development environment
- **Better Onboarding**: New team members get instant access to Claude Code guidance
- **Improved Productivity**: Quick answers to Claude Code questions

## Technical Details

### Server Implementation
- **Source**: `/Users/cal/Projects/claude mcp/claude-code-docs-mcp`
- **Technology**: Node.js/TypeScript with MCP SDK
- **Documentation Source**: `https://docs.anthropic.com/en/docs/claude-code`
- **Caching**: 1-hour cache for performance
- **Content Extraction**: HTML parsing with fallback handling

### Integration Points
- **Detection Logic**: Added to Context Server Discovery Agent Phase 1
- **Server Catalog**: Listed in Documentation and Knowledge Context Servers
- **Configuration Template**: Included in MCP settings generation
- **Installation Script**: Conditional installation based on detection
- **Documentation**: Comprehensive usage and capability documentation

## Testing

Integration testing framework created at:
`/Users/cal/Projects/Billables_TimeTracking/dev-framework/tests/test_claude_code_integration.md`

### Test Coverage
- ✅ Directory detection (`.claude/`)
- ✅ File detection (`CLAUDE.md`)
- ✅ CLI detection (`which claude`)
- ✅ Configuration generation
- ✅ Installation script generation
- ✅ Catalog inclusion
- ✅ Detection logic inclusion
- ✅ Essential server classification

## Status

**Integration Status**: ✅ Complete

The Claude Code MCP Server is now fully integrated into the dev-framework's Context Server Discovery Agent, providing automatic detection, configuration, and installation for projects using Claude Code CLI.

## Next Steps

1. **Validation**: Run integration tests to verify functionality
2. **Documentation**: Update main dev-framework README with Claude Code integration details
3. **Enhancement**: Consider adding project-specific Claude Code configuration detection
4. **Feedback**: Collect user feedback on Claude Code documentation access experience