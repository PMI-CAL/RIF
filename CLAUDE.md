# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RIF - Reactive Intelligence Framework

An automatic intelligent development system that orchestrates specialized agents through GitHub issues with zero manual intervention.

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