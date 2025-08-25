# Claude Code Official Documentation

This directory contains comprehensive documentation about Claude Code's actual capabilities, architecture, and integration patterns based on official Anthropic documentation and extensive research.

## Documentation Files

### Core Documentation
- **[capabilities.md](capabilities.md)** - Comprehensive analysis of Claude Code's actual capabilities vs. common misunderstandings
- **[research-findings.json](research-findings.json)** - Structured research data with compatibility analysis and corrected implementation patterns

### Integration Systems  
- **[mcp-servers.md](mcp-servers.md)** - Complete guide to MCP (Model Context Protocol) servers for external integrations
- **[subagents.md](subagents.md)** - Real subagent capabilities and how to use them effectively
- **[hooks-system.md](hooks-system.md)** - Event-triggered automation system configuration and best practices

## Key Findings Summary

### What Claude Code Actually Is
- **Single AI assistant** with specialized tools and capabilities
- **Terminal-based coding tool** that integrates with existing developer workflows
- **NOT an orchestration platform** or multi-agent system

### Real Integration Methods
1. **MCP Servers** - Primary external integration mechanism
2. **Subagents** - Contextual specialists within the same session  
3. **Hooks** - Event-triggered automation scripts
4. **Direct Tools** - File operations, command execution, code analysis

### Critical Architecture Corrections

#### Task-Based Orchestration → Subagent Delegation
```python
# WRONG (Common Assumption):
Task(description="Analyze issue", subagent_type="analyst")

# CORRECT (Claude Code Reality):
# Use /agents command to create subagents
# Use explicit invocation: /agent rif-analyst Analyze issue #123
```

#### Independent Agent Processes → Single Session Specialists  
```python
# WRONG (Common Assumption):
Multiple independent agents posting to GitHub autonomously

# CORRECT (Claude Code Reality):
Single Claude Code session with:
- Subagent specialization via custom prompts
- GitHub integration via MCP servers
- File-based coordination between subagents
```

#### Background Automation → Event-Triggered Hooks
```python
# WRONG (Common Assumption):
Continuous background monitoring and interval-based automation

# CORRECT (Claude Code Reality):  
Event-triggered hooks in .claude/settings.json:
- SessionStart, PostToolUse, UserPromptSubmit, etc.
- Command execution on specific tool events
- No continuous background processes
```

## Compatibility Matrix

| Feature | Status | Implementation |
|---------|--------|----------------|
| GitHub Issue Processing | ✅ Compatible | MCP server integration |
| Code Analysis | ✅ Compatible | Built-in capabilities |
| File Operations | ✅ Compatible | Native tools |
| Subagent Specialization | ✅ Compatible | Real feature |
| Task-based Orchestration | ❌ Incompatible | No Task tool exists |
| Independent Agent Processes | ❌ Incompatible | Subagents not independent |
| Background Scheduling | ❌ Incompatible | Event-triggered only |
| Workflow Automation | 🔄 Needs Modification | Simplify to hooks |
| Agent Coordination | 🔄 Needs Modification | File/GitHub-based |

## Migration Guidance

### For RIF System
1. **Replace Task() calls** with subagent creation and invocation
2. **Implement GitHub MCP server** instead of assuming agent autonomy
3. **Simplify automation** to use hooks instead of background processes  
4. **Update agent coordination** to use files/GitHub instead of direct communication

### For Other Systems
1. **Audit orchestration assumptions** - verify Task tool actually exists
2. **Review agent architecture** - ensure understanding of subagent limitations
3. **Validate automation patterns** - confirm hook capabilities vs. requirements
4. **Test MCP integration** - verify external integration patterns work

## Evidence Sources

All documentation is based on official Anthropic sources:
- https://docs.anthropic.com/en/docs/claude-code
- https://docs.anthropic.com/en/docs/claude-code/mcp  
- https://docs.anthropic.com/en/docs/claude-code/subagents
- https://docs.anthropic.com/en/docs/claude-code/hooks
- https://docs.anthropic.com/en/docs/claude-code/getting-started

## Usage in RIF Knowledge Base

This documentation is integrated into the RIF knowledge base via the MCP server at:
`/Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py`

Query examples:
- `get_claude_documentation(topic="capabilities")` 
- `get_claude_documentation(topic="limitations")`
- `get_claude_documentation(topic="MCP servers")`
- `get_claude_documentation(topic="subagents")`
- `check_compatibility(approach="Task-based orchestration")`

## Contributing

When updating Claude Code documentation:
1. Verify against official Anthropic documentation
2. Update compatibility matrices
3. Test patterns with actual Claude Code sessions
4. Update integration examples
5. Validate with RIF knowledge base queries

---

*Last updated: 2025-08-25*  
*Research phase: Complete*  
*Integration status: Active*