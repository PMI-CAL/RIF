# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RIF - Reactive Intelligence Framework

An automatic intelligent development system that orchestrates specialized agents through GitHub issues with zero manual intervention.

## 🚨 CRITICAL: How RIF Orchestration Actually Works

**IMPORTANT**: The RIF Orchestrator is NOT a Task to launch. Claude Code IS the orchestrator.

### Correct Orchestration Pattern
When asked to "orchestrate" or "launch RIF orchestrator":
1. **Claude Code directly** checks GitHub issues (not via Task)
2. **Claude Code directly** analyzes states and determines which agents need launching
3. **Claude Code launches MULTIPLE Task agents in ONE response** for parallel execution

### ❌ WRONG: Claude Doing Work Directly
```bash
# This is INCORRECT - Claude doing implementation work:
Edit(file_path="/path/to/file.js", old_string="...", new_string="...")
# Or Claude analyzing requirements without launching agents
```

### ✅ CORRECT: Claude Launching Agents
```python
# When orchestrating, Claude Code should execute (in a single response):
Task(
    description="RIF-Analyst for issue #3",
    subagent_type="general-purpose",
    prompt="You are RIF-Analyst. Analyze issue #3 about LightRAG usage. [Include full rif-analyst.md instructions]"
)
Task(
    description="RIF-Implementer for issue #2", 
    subagent_type="general-purpose",
    prompt="You are RIF-Implementer. Implement fix for issue #2. [Include full rif-implementer.md instructions]"
)
# These Tasks run IN PARALLEL because they're in the same response
```

### Real-World Orchestration Examples

#### Example 1: New Issue Detected
```python
# User: "Orchestrate RIF to handle the open issues"
# Claude Code should respond with:

Task(
    description="RIF-Analyst: Analyze issue #5 requirements",
    subagent_type="general-purpose", 
    prompt="You are RIF-Analyst. Analyze GitHub issue #5 titled 'Add user authentication'. Extract requirements, identify patterns from knowledge base, assess complexity. Follow all instructions in claude/agents/rif-analyst.md."
)
Task(
    description="RIF-Implementer: Implement issue #3 fix",
    subagent_type="general-purpose",
    prompt="You are RIF-Implementer. Implement the fix for GitHub issue #3 'Database connection pooling'. Use checkpoints for progress tracking. Follow all instructions in claude/agents/rif-implementer.md."
)
```

#### Example 2: Multiple States Need Processing  
```python
# When multiple issues are in different states:

Task(
    description="RIF-Validator: Validate issue #1 implementation", 
    subagent_type="general-purpose",
    prompt="You are RIF-Validator. Validate the implementation for GitHub issue #1. Run tests, check quality gates, ensure standards compliance. Follow all instructions in claude/agents/rif-validator.md."
)
Task(
    description="RIF-Planner: Plan architecture for issue #4",
    subagent_type="general-purpose", 
    prompt="You are RIF-Planner. Create detailed plan for GitHub issue #4 'Microservices migration'. Assess complexity, create workflow. Follow all instructions in claude/agents/rif-planner.md."
)
Task(
    description="RIF-Learner: Update knowledge from completed issue #2",
    subagent_type="general-purpose",
    prompt="You are RIF-Learner. Extract learnings from completed GitHub issue #2. Update knowledge base with patterns, decisions, metrics. Follow all instructions in claude/agents/rif-learner.md."
)
```

#### Example 3: Single Issue Full Workflow
```python
# For a complex issue requiring multiple agents in sequence:

Task(
    description="RIF-Analyst: Deep analysis of complex issue #7",
    subagent_type="general-purpose",
    prompt="You are RIF-Analyst. Perform deep analysis of GitHub issue #7 'Real-time data processing pipeline'. This is a high-complexity task requiring thorough requirements analysis. Follow all instructions in claude/agents/rif-analyst.md."
)
Task(
    description="RIF-Architect: Design system for issue #7", 
    subagent_type="general-purpose",
    prompt="You are RIF-Architect. Design the system architecture for GitHub issue #7 based on RIF-Analyst findings. Create detailed technical design. Follow all instructions in claude/agents/rif-architect.md."
)
```

### What "Task.parallel()" Means
Documentation references to `Task.parallel()` are **pseudocode** meaning:
"Launch multiple Task tool invocations in a single Claude response"

It is NOT a real function - it represents the pattern of parallel Task execution.

### Key Rules for Orchestration

1. **NEVER do implementation work directly** - Always delegate to agents
2. **Launch multiple Tasks in ONE response** for parallel execution  
3. **Include full agent instructions** in the Task prompt
4. **Match agents to issue states** (analyst for state:new, implementer for state:implementing, etc.)
5. **Let agents handle GitHub interactions** (posting comments, changing labels)
6. **Trust the agent specialization** - Don't micromanage their work

### Quick Reference: Agent Launching Template

```python
# Step 1: Check GitHub issues (Claude Code does this directly)
# Step 2: Identify which agents need to run
# Step 3: Launch agents in parallel using this pattern:

Task(
    description="[AGENT_NAME]: [BRIEF_TASK_DESCRIPTION]",
    subagent_type="general-purpose",
    prompt="You are [AGENT_NAME]. [SPECIFIC_TASK_DETAILS]. Follow all instructions in claude/agents/[agent-file].md."
)
Task(
    description="[ANOTHER_AGENT]: [ANOTHER_TASK_DESCRIPTION]", 
    subagent_type="general-purpose",
    prompt="You are [ANOTHER_AGENT]. [ANOTHER_TASK_DETAILS]. Follow all instructions in claude/agents/[agent-file].md."
)
# Add more Tasks as needed - they all run in parallel
```

### Common Orchestration Scenarios

| Scenario | Claude Action | Agent Tasks |
|----------|---------------|-------------|
| New issue created | Check issue state | Launch RIF-Analyst |
| Multiple issues in different states | Identify all states | Launch appropriate agent for each |
| Complex issue needs planning | Assess complexity | Launch RIF-Analyst + RIF-Planner |
| Implementation ready | Check state:implementing | Launch RIF-Implementer |
| Code needs validation | Check state:validating | Launch RIF-Validator |
| Learning phase | Check completed issues | Launch RIF-Learner |

### Troubleshooting Orchestration

**Problem**: Claude is doing work directly instead of launching agents
**Solution**: Always use Task() tool to delegate work to specialized agents

**Problem**: Agents are not running in parallel  
**Solution**: Launch all Tasks in a single Claude response, not separate responses

**Problem**: Agent instructions are incomplete
**Solution**: Always include "Follow all instructions in claude/agents/[agent-file].md" in prompts

## Architecture Overview

RIF operates as a **completely automatic** framework:
- GitHub issues trigger all workflows - no manual commands needed
- Specialized agents activate based on state labels
- LightRAG knowledge base learns from every interaction
- State machine manages workflow progression
- Quality gates enforce standards automatically

## Core Components

### Directory Structure
```
/
├── claude/
│   ├── agents/           # Agent definitions (rif-*.md, specialized agents)
│   ├── commands/         # Development commands (technology-agnostic templates)
│   ├── rules/           # Code quality and GitHub workflow rules
│   └── docs/            # Framework documentation
├── config/
│   ├── rif-workflow.yaml      # State machine configuration
│   ├── multi-agent.yaml       # Parallel agent execution settings
│   └── framework-variables.yaml # Universal variable definitions
├── knowledge/           # LightRAG knowledge base
│   ├── patterns/       # Successful code patterns
│   ├── decisions/      # Architectural decisions
│   ├── issues/         # Issue resolutions
│   ├── metrics/        # Performance data
│   ├── learning/       # Continuous improvements
│   └── checkpoints/    # Recovery points
├── .claude/
│   └── settings.json   # Claude Code hooks configuration
├── setup.sh            # Technology detection and setup
└── rif-init.sh        # RIF initialization script
```

## Development Commands

RIF uses technology-agnostic command placeholders that adapt to your stack:

### Setup and Installation
```bash
# Initialize RIF for a project
./rif-init.sh

# Setup framework with technology detection
./setup.sh <project_directory>

# Check prerequisites (git, gh, jq)
./rif-init.sh
```

### Testing and Quality
The framework automatically detects your technology stack and uses appropriate commands:
- **JavaScript/Node.js**: npm test, npm run lint, npm run build
- **Python**: pytest, flake8/black, python setup.py build
- **Java**: mvn test, mvn checkstyle:check, mvn package
- **Go**: go test ./..., golangci-lint run, go build
- **Rust**: cargo test, cargo clippy, cargo build --release

To run tests in your project, check for:
1. package.json scripts (Node.js)
2. Makefile targets
3. pytest.ini or setup.cfg (Python)
4. pom.xml or build.gradle (Java)

## Agent System

### Core RIF Agents (Automatic Activation)

| Agent | Trigger Label | Purpose |
|-------|--------------|---------|
| RIF-Analyst | `state:new` | Requirements analysis, pattern recognition |
| RIF-Planner | `state:planning` | Strategic planning, workflow configuration |
| RIF-Architect | `state:architecting` | System design, dependency graphs |
| RIF-Implementer | `state:implementing` | Code implementation, checkpoint tracking |
| RIF-Validator | `state:validating` | Testing, quality gate enforcement |
| RIF-Learner | `state:learning` | Knowledge base updates |

### Workflow State Machine

```
[Issue Created] → state:new → RIF-Analyst
    ↓
state:analyzing → Pattern Recognition
    ↓
state:planning → RIF-Planner (complexity assessment)
    ↓
state:architecting → RIF-Architect (if complex)
    ↓
state:implementing → RIF-Implementer
    ↓
state:validating → RIF-Validator
    ↓
state:learning → RIF-Learner
    ↓
state:complete → Issue Closed
```

## Claude Code Integration

### Required Hooks Configuration

Add to `.claude/settings.json` for automatic RIF activation:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "gh issue list --state open --label 'state:*' --json number,title,labels,body | head -10 > /tmp/rif-context.json && echo 'RIF: Found '$(jq length /tmp/rif-context.json)' active issues'",
        "output": "context"
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": ".*issue.*|.*fix.*|.*implement.*|.*analyze.*",
        "hooks": [
          {
            "type": "command",
            "command": "if [ -f ./knowledge/latest-patterns.json ]; then cat ./knowledge/latest-patterns.json; fi",
            "output": "context"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"timestamp\": \"'$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")'\", \"action\": \"code_modified\"}' >> ./knowledge/events.jsonl"
          }
        ]
      }
    ]
  }
}
```

## GitHub Workflow

### Creating Work Items

All work starts with GitHub issues:

```bash
# Create issue to trigger RIF
gh issue create --title "Implement feature X" --body "Description"

# RIF automatically:
# 1. Detects new issue
# 2. Applies state:new label
# 3. Triggers RIF-Analyst
# 4. Progresses through workflow
# 5. Posts results as comments
# 6. Creates PR if needed
# 7. Closes issue when complete
```

### Label System

RIF manages these labels automatically:
- `state:*` - Workflow state (new, analyzing, planning, etc.)
- `complexity:*` - Task complexity (low, medium, high, very-high)
- `agent:*` - Active agent tracking
- `priority:*` - Urgency level
- `learned:*` - New patterns discovered

## Quality Gates

Automatic enforcement without configuration:

| Gate | Threshold | Failure Action |
|------|-----------|----------------|
| Test Coverage | >80% | Return to implementer |
| Security | No critical vulnerabilities | Block and alert |
| Performance | Meet baseline | Trigger optimization |
| Documentation | Complete | Generate automatically |

## Knowledge Base

### Automatic Learning System

RIF learns from every interaction:
- **Successful patterns** stored in `knowledge/patterns/`
- **Design decisions** tracked in `knowledge/decisions/`
- **Issue resolutions** saved in `knowledge/issues/`
- **Performance metrics** logged in `knowledge/metrics/`

### Pattern Application

When analyzing new issues, RIF:
1. Searches for similar past issues
2. Applies successful patterns
3. Avoids known failures
4. Documents new learnings

## Complexity Assessment

RIF automatically calibrates planning depth:

| Complexity | LOC | Files | Planning Depth | Agents |
|------------|-----|-------|----------------|---------|
| Low | <50 | 1 | Shallow | 2-3 |
| Medium | <500 | 2-5 | Standard | 3-4 |
| High | <2000 | 6-20 | Deep | 4-5 |
| Very High | >2000 | 20+ | Recursive | 5+ with sub-planning |

## Parallel Execution

Multi-agent configuration (`config/multi-agent.yaml`):
- Maximum 4 parallel subagents
- Resource limits per agent
- Communication bus for coordination
- Automatic retry on failure

## Recovery System

### Checkpoint Management

Checkpoints created automatically:
- After each agent completion
- Before risky operations
- On test failures
- Every 30 minutes

### Automatic Recovery

On failure, RIF:
1. Identifies last stable checkpoint
2. Restores to that state
3. Adjusts approach based on failure
4. Retries with modifications
5. Updates knowledge base

## Quick Start

1. **Initialize RIF**: Run `./rif-init.sh` in your project
2. **Configure Hooks**: Add settings to `.claude/settings.json`
3. **Create Issue**: `gh issue create --title "Your task"`
4. **Watch It Work**: RIF handles everything automatically

## Important Notes

- **NO MANUAL COMMANDS**: Everything triggers automatically via GitHub
- **TECHNOLOGY AGNOSTIC**: Works with any programming language
- **SELF-IMPROVING**: Gets smarter with each task
- **PRODUCTION READY**: Enterprise-grade quality gates
- **ZERO CONFIGURATION**: Works out of the box

## Troubleshooting

If RIF isn't responding:
1. Check GitHub labels are correct (`state:*`)
2. Verify hooks in `.claude/settings.json`
3. Ensure `gh` CLI is authenticated
4. Check `knowledge/` directory is accessible
5. Look for error comments in GitHub issue

The system will automatically detect stuck states and attempt self-recovery after timeout.