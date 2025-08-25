# Claude Code Subagents: Real Capabilities

## What are Subagents?

Subagents are **specialized AI assistants within Claude Code** that operate in separate context windows with customized system prompts. They are **not independent processes** but rather contextual specialists that help preserve the main conversation context.

## Key Facts

- **Contextual specialists** within the same Claude Code session
- Operate in **separate context windows** 
- Have **customized system prompts** for domain expertise
- Can be configured with **specific tool access restrictions**
- Support both **automatic delegation** and **explicit invocation**

## Subagent Types

### Project-Level Subagents
- **Highest priority** when Claude Code chooses which subagent to use
- Specific to the current project/repository
- Configured in project's `.claude/` directory
- Shared with team via version control

### User-Level Subagents  
- **Available across all projects** for a user
- Personal productivity and specialization agents
- Configured in user's global Claude Code settings
- Not shared with team members

## Creation Process

### Using the `/agents` Command
1. Use `/agents` command in Claude Code to open subagent interface
2. Choose between project-level or user-level subagent
3. Define required properties:
   - **Unique name** (identifier for the subagent)
   - **Description** (what the subagent specializes in)
   - **Tool restrictions** (which tools it can/cannot use)
   - **System prompt** (specialized instructions and expertise)

### Configuration Example
```json
{
  "subagents": {
    "rif-analyst": {
      "name": "RIF Requirements Analyst",
      "description": "Specialized in analyzing GitHub issues and extracting requirements",
      "systemPrompt": "You are a requirements analyst specialized in the RIF framework...",
      "toolRestrictions": {
        "allowed": ["Read", "Grep", "Glob"],
        "denied": ["Write", "Edit", "Bash"]
      }
    },
    "rif-implementer": {
      "name": "RIF Code Implementer", 
      "description": "Specialized in implementing RIF features and fixes",
      "systemPrompt": "You are a code implementer specialized in the RIF framework...",
      "toolRestrictions": {
        "allowed": ["Read", "Write", "Edit", "MultiEdit", "Bash"],
        "denied": ["WebSearch"]
      }
    }
  }
}
```

## Performance Characteristics

### Benefits
- **Help preserve main conversation context** by handling specialized tasks separately
- **Domain expertise** through customized system prompts
- **Tool restrictions** prevent inappropriate tool usage
- **Automatic delegation** for matching request types

### Limitations  
- **May add slight latency** when gathering context for delegation
- **Not independent processes** - still part of Claude Code session
- **No persistent memory** beyond the conversation context
- **Cannot run in parallel** with each other or main Claude

## RIF Integration Patterns

### Correct Subagent Usage
Instead of assuming independent agent processes:

```python
# WRONG (RIF assumption):
Task(description="Analyze issue #123", subagent_type="rif-analyst")

# CORRECT (Claude Code reality):
# Create subagent via /agents command, then:
# Claude Code automatically delegates or user explicitly invokes
# "/agent rif-analyst Please analyze issue #123"
```

### Subagent Specialization Examples

#### RIF-Analyst Subagent
```json
{
  "name": "rif-analyst",
  "description": "Analyzes GitHub issues and extracts requirements",
  "systemPrompt": "You are RIF-Analyst, specialized in analyzing GitHub issues for the RIF framework. You extract requirements, identify patterns, assess complexity, and determine dependencies. Always provide structured analysis output.",
  "toolRestrictions": {
    "allowed": ["Read", "Grep", "Glob", "WebFetch"],
    "denied": ["Write", "Edit", "MultiEdit", "Bash"]
  }
}
```

#### RIF-Implementer Subagent
```json
{
  "name": "rif-implementer", 
  "description": "Implements RIF features and creates code",
  "systemPrompt": "You are RIF-Implementer, specialized in implementing features for the RIF framework. You write code, create files, run tests, and ensure implementation quality. Always follow RIF coding standards.",
  "toolRestrictions": {
    "allowed": ["Read", "Write", "Edit", "MultiEdit", "Bash", "Grep", "Glob"],
    "denied": ["WebSearch"]
  }
}
```

#### RIF-Validator Subagent
```json
{
  "name": "rif-validator",
  "description": "Validates RIF implementations and runs quality checks", 
  "systemPrompt": "You are RIF-Validator, specialized in validating implementations for the RIF framework. You run tests, check code quality, validate functionality, and ensure standards compliance.",
  "toolRestrictions": {
    "allowed": ["Read", "Bash", "Grep", "Glob"],
    "denied": ["Write", "Edit", "MultiEdit"]
  }
}
```

## Delegation Patterns

### Automatic Delegation
Claude Code can automatically choose appropriate subagents based on:
- **Request content** matching subagent descriptions
- **Tool requirements** for the requested task
- **Domain expertise** indicated in system prompts

### Explicit Invocation
Users can explicitly invoke subagents:
```
/agent rif-analyst Analyze the requirements in issue #456
/agent rif-implementer Create the user authentication system
/agent rif-validator Run tests and validate the implementation
```

## Coordination Between Subagents

### File-Based Coordination
Since subagents cannot communicate directly:
```json
{
  "coordination_pattern": "file_based",
  "workflow": [
    "RIF-Analyst writes analysis to ./analysis/issue-123.json",
    "RIF-Implementer reads analysis file for requirements",  
    "RIF-Implementer writes implementation status to ./status/issue-123.json",
    "RIF-Validator reads status file to know what to test"
  ]
}
```

### GitHub Issue Coordination
```json
{
  "coordination_pattern": "github_based",
  "workflow": [
    "RIF-Analyst posts analysis as GitHub issue comment",
    "RIF-Implementer reads issue comments for context",
    "RIF-Implementer posts implementation updates to issue", 
    "RIF-Validator reads issue history for validation context"
  ]
}
```

## Migration from Task Architecture

### Before (RIF Assumption)
```python
# This doesn't work - no Task tool exists
Task(description="RIF-Analyst for issue #3", subagent_type="general-purpose", prompt="...")
Task(description="RIF-Implementer for issue #2", subagent_type="general-purpose", prompt="...")
```

### After (Subagent Reality)
1. **Create subagents** using `/agents` command with proper configurations
2. **Use explicit invocation**: `/agent rif-analyst Analyze issue #3`
3. **Allow automatic delegation**: Claude Code chooses appropriate subagent
4. **Coordinate via files/GitHub**: Subagents share context through persistent storage

## Limitations vs RIF Assumptions

### What Subagents CANNOT Do
- **Run independently** from Claude Code session
- **Post to GitHub autonomously** (Claude Code handles GitHub interaction)
- **Communicate directly** with each other
- **Maintain persistent state** beyond conversation
- **Run in parallel** with each other
- **Be orchestrated externally** by other systems

### What Subagents CAN Do
- **Specialize in domain areas** through custom prompts
- **Use restricted tool sets** for focused capabilities  
- **Preserve main context** by handling specialized tasks separately
- **Be invoked automatically** based on request matching
- **Access MCP servers** if configured in their tool restrictions
- **Coordinate via files** and external storage

## Best Practices

### Subagent Design
- **Single responsibility** - each subagent has one clear specialization
- **Clear boundaries** - distinct domains with minimal overlap
- **Tool restrictions** - only allow necessary tools for the subagent's role
- **Descriptive prompts** - detailed system prompts explaining expertise

### Workflow Design  
- **Sequential delegation** - complete one subagent's work before moving to next
- **File-based handoffs** - use structured files for passing context
- **Status tracking** - maintain clear progress indicators
- **Error handling** - graceful fallback when subagents cannot complete tasks

### Integration with RIF
- **Replace Task calls** with subagent creation and invocation
- **Use MCP servers** for external integrations instead of assuming agent autonomy
- **Leverage hooks** for automation instead of assuming background processes
- **Coordinate through GitHub** instead of assuming direct agent communication

---

*Subagents are the real specialization mechanism in Claude Code, replacing the fictional Task-based agent orchestration.*