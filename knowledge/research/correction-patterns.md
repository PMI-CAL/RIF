# RIF Correction Patterns: From Assumptions to Reality

## Overview

This document provides specific patterns for correcting RIF's implementation to align with Claude Code's actual capabilities. Each pattern shows the wrong approach, the correct approach, and migration steps.

## Pattern 1: Task Orchestration → Subagent Delegation

### ❌ Current RIF Pattern (Broken)
```python
# This doesn't work - Task tool doesn't exist
Task(
    description="RIF-Analyst: Analyze issue #5 requirements",
    subagent_type="general-purpose", 
    prompt="You are RIF-Analyst. Analyze GitHub issue #5..."
)
Task(
    description="RIF-Implementer: Implement issue #3 fix",
    subagent_type="general-purpose",
    prompt="You are RIF-Implementer. Implement the fix..."
)
```

### ✅ Corrected Pattern (Works)
```python
# Claude Code session with subagent delegation
# Step 1: Create specialized subagents (one-time setup)
# /agents command in Claude Code to create:
# - RIF-Analyst subagent with custom prompt
# - RIF-Implementer subagent with custom prompt

# Step 2: Delegate to subagents within conversation
# "Use the RIF-Analyst to analyze issue #5 requirements"
# "Use the RIF-Implementer to implement the fix for issue #3"

# Step 3: MCP integration for GitHub operations
# GitHub MCP server handles issue updates, comments, etc.
```

### Migration Steps
1. **Remove** all Task() tool calls
2. **Create** subagents using Claude Code `/agents` command
3. **Update** CLAUDE.md to use delegation language
4. **Implement** GitHub MCP server for issue operations

## Pattern 2: Independent Agent Processes → Contextual Specialists

### ❌ Current RIF Pattern (Broken)
```yaml
# agents.yaml - assumes independent processes
agents:
  rif-analyst:
    trigger: ["state:new", "state:analyzing"]
    parallel: true
    tasks: 4
    memory: persistent
    communication: inter-agent
```

### ✅ Corrected Pattern (Works)
```json
// Claude Code subagent configuration
{
  "name": "RIF-Analyst",
  "description": "Requirements analysis and pattern recognition specialist",
  "tools": ["Read", "Grep", "Bash"],
  "systemPrompt": "You are RIF-Analyst. You specialize in analyzing GitHub issues, identifying patterns from the knowledge base, and providing complexity assessments. Always search for similar past issues and apply learned patterns."
}
```

### Migration Steps
1. **Remove** agent configuration YAML files
2. **Create** subagents via Claude Code interface
3. **Convert** agent instructions to subagent system prompts
4. **Update** workflow to use conversation delegation

## Pattern 3: Automatic Scheduling → Event-Driven Hooks

### ❌ Current RIF Pattern (Broken)
```json
// Assumes background scheduling capability
{
  "automations": {
    "issue_detection": {
      "enabled": true,
      "interval": "5m",
      "command": "gh issue list --state open",
      "trigger": "rif-analyst"
    }
  }
}
```

### ✅ Corrected Pattern (Works)
```json
// Event-driven hooks in .claude/settings.json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*orchestrate.*|.*rif.*|.*issues.*",
        "hooks": [
          {
            "type": "command",
            "command": "gh issue list --state open --json number,title,labels | jq -r '.[] | \"Issue #\\(.number): \\(.title)\"'",
            "output": "context"
          }
        ]
      }
    ]
  }
}
```

### Migration Steps
1. **Remove** automation configurations with intervals
2. **Convert** to event-triggered hooks
3. **Update** triggers to user actions, not time-based
4. **Test** hook functionality in Claude Code

## Pattern 4: Inter-Agent Communication → GitHub Integration

### ❌ Current RIF Pattern (Broken)
```python
# Assumes agents can communicate and post independently
def rif_analyst_complete():
    # Agent posts to GitHub issue
    github.post_comment(issue, "Analysis complete")
    # Agent notifies other agents
    message_queue.send("rif-planner", analysis_results)
```

### ✅ Corrected Pattern (Works)
```python
# Claude Code handles GitHub integration via MCP
# Subagent completes analysis and provides results
# Claude Code posts to GitHub using MCP server

# 1. Subagent provides results in conversation
# 2. Claude Code processes results
# 3. GitHub MCP server posts comment
# 4. Next subagent is delegated within same session
```

### Migration Steps
1. **Remove** agent-to-agent communication code
2. **Implement** GitHub MCP server integration
3. **Update** workflow to single-session conversation
4. **Configure** MCP server for GitHub operations

## Pattern 5: Complex State Machine → Conversation Flow

### ❌ Current RIF Pattern (Broken)
```yaml
# Complex state transitions between independent agents
workflow:
  states:
    - name: "analyzing"
      agent: "rif-analyst"
      transitions:
        success: "planning"
        failure: "error"
    - name: "planning"  
      agent: "rif-planner"
      transitions:
        success: "implementing"
        complexity_high: "architecting"
```

### ✅ Corrected Pattern (Works)
```markdown
# Conversation-based workflow within Claude Code session

## RIF Workflow Process

1. **Analysis Phase**
   - Use RIF-Analyst subagent to analyze issue
   - Update GitHub issue with analysis via MCP
   - Determine next steps based on complexity

2. **Planning/Architecture Phase** (if needed)
   - Use RIF-Planner for medium complexity
   - Use RIF-Architect for high complexity
   - Update GitHub with plan via MCP

3. **Implementation Phase**
   - Use RIF-Implementer subagent
   - Create PR via GitHub MCP
   - Update issue status via MCP

4. **Validation Phase**
   - Use RIF-Validator subagent
   - Run tests and quality checks
   - Update PR with results via MCP
```

### Migration Steps
1. **Remove** state machine YAML configurations
2. **Convert** to conversation-based workflow documentation
3. **Update** CLAUDE.md with new workflow
4. **Train** on subagent delegation patterns

## Pattern 6: Persistent Agent Memory → Knowledge Base Integration

### ❌ Current RIF Pattern (Broken)
```python
# Assumes agents have persistent memory across sessions
class RifAnalyst:
    def __init__(self):
        self.memory = PersistentMemory()
        self.learned_patterns = self.memory.load("patterns")
    
    def analyze(self, issue):
        # Uses cross-session memory
        similar = self.memory.find_similar(issue)
```

### ✅ Corrected Pattern (Works)
```python
# Knowledge base integration via MCP server + file operations

# 1. Create Knowledge MCP server for pattern queries
# 2. Use file operations for knowledge storage
# 3. Hooks for automatic knowledge updates

# Example subagent prompt:
"""
You are RIF-Analyst. Before analyzing any issue:

1. Search for similar patterns using:
   - Read knowledge/patterns/*.json files
   - Query knowledge MCP server if available
   - Use Grep to find related past issues

2. Apply learned patterns to current analysis

3. Update knowledge base with new findings:
   - Write new patterns to knowledge/patterns/
   - Update metrics in knowledge/metrics/
"""
```

### Migration Steps
1. **Remove** persistent agent memory code
2. **Create** knowledge base MCP server
3. **Update** subagent prompts to include knowledge queries
4. **Implement** hooks for knowledge updates

## Pattern 7: Parallel Agent Execution → Sequential Delegation

### ❌ Current RIF Pattern (Broken)
```python
# Assumes multiple agents can run in parallel
async def orchestrate_parallel():
    tasks = [
        launch_agent("rif-analyst", issue_1),
        launch_agent("rif-implementer", issue_2),
        launch_agent("rif-validator", issue_3)
    ]
    await asyncio.gather(*tasks)
```

### ✅ Corrected Pattern (Works)
```python
# Sequential delegation within conversation
def orchestrate_sequential():
    # Single Claude Code session handles all work
    
    # For multiple issues, process sequentially:
    # 1. "Use RIF-Analyst to analyze issue #1"
    # 2. When complete: "Use RIF-Implementer to work on issue #2"  
    # 3. When complete: "Use RIF-Validator to validate issue #3"
    
    # For single complex issue, delegate through phases:
    # 1. Analysis → Planning → Implementation → Validation
    # Each phase uses appropriate subagent
```

### Migration Steps
1. **Remove** parallel execution assumptions
2. **Update** documentation to sequential workflow
3. **Redesign** orchestration for single session
4. **Focus** on efficient subagent delegation

## Pattern 8: Direct API Calls → MCP Server Integration

### ❌ Current RIF Pattern (Broken)
```python
# Direct API calls assumed from agents
import requests

def update_github_issue(issue_number, data):
    response = requests.patch(
        f"https://api.github.com/repos/owner/repo/issues/{issue_number}",
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json=data
    )
```

### ✅ Corrected Pattern (Works)
```bash
# MCP server integration
# 1. Install GitHub MCP server
claude mcp add github --env GITHUB_TOKEN=$GITHUB_TOKEN

# 2. Use MCP server for GitHub operations
# Claude Code automatically uses MCP for GitHub operations
# No direct API code needed in RIF
```

### Migration Steps
1. **Remove** direct API calling code
2. **Install** appropriate MCP servers
3. **Configure** authentication for MCP servers
4. **Update** code to rely on Claude Code's MCP integration

## Pattern 9: Complex Configuration → Simplified Settings

### ❌ Current RIF Pattern (Over-Complex)
```yaml
# Multiple complex configuration files
# config/rif-workflow.yaml
# config/multi-agent.yaml  
# config/framework-variables.yaml
# claude/agents/*.md
# etc.
```

### ✅ Corrected Pattern (Simplified)
```json
// Single .claude/settings.json with essentials
{
  "hooks": {
    // Event-triggered automation only
  },
  "subagents": {
    // Subagent configurations
  },
  "mcp_servers": {
    // MCP server configurations
  }
}
```

### Migration Steps
1. **Consolidate** multiple config files
2. **Remove** unused configuration options
3. **Focus** on hooks and MCP servers
4. **Simplify** subagent definitions

## Pattern 10: Agent Error Recovery → Hook-Based Recovery

### ❌ Current RIF Pattern (Broken)
```python
# Agent-level error recovery
class RifAgent:
    def execute(self, task):
        try:
            result = self.perform_task(task)
        except Exception as e:
            self.recover_from_error(e)
            self.retry_task(task)
```

### ✅ Corrected Pattern (Works)
```json
// Hook-based error recovery
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if [ $? -ne 0 ]; then echo 'Error detected, logging...'; echo '{\"error\": \"$?\", \"command\": \"$1\"}' >> knowledge/errors.jsonl; fi",
            "output": "silent"
          }
        ]
      }
    ]
  }
}
```

### Migration Steps
1. **Remove** agent-level error handling
2. **Implement** hook-based error recovery
3. **Use** GitHub MCP for error reporting
4. **Simplify** to tool-level error handling

## Summary of Critical Changes

### Must Change Immediately
1. **Remove all Task() tool usage** - Replace with subagent delegation
2. **Update CLAUDE.md** - Remove orchestration assumptions
3. **Implement GitHub MCP** - For issue management
4. **Simplify automation** - Event-driven hooks only

### Change Soon
1. **Consolidate configurations** - Single settings.json
2. **Redesign workflows** - Conversation-based
3. **Create proper subagents** - Via Claude Code interface
4. **Remove state machine** - Sequential delegation

### Plan for Later
1. **Optimize performance** - Single-session efficiency
2. **Enhance MCP integration** - Full server stack
3. **Improve error handling** - Hook-based recovery
4. **Update documentation** - Complete accuracy

---

*These patterns provide the roadmap for migrating RIF from its incorrect assumptions to Claude Code's actual capabilities.*