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

## 🚨 CRITICAL: PR Priority Rule

**ABSOLUTE PRIORITY**: Open Pull Requests BLOCK ALL OTHER ORCHESTRATION

### Why PRs Are Highest Priority
- **Work-in-Progress**: PRs represent work already started that needs completion
- **Production Pipeline**: PRs are closest to delivering value to users
- **Resource Efficiency**: Completing existing work before starting new work maximizes throughput
- **Quality Gates**: PRs require validation before new development should begin

### PR Detection and Handling
Before ANY orchestration, check for open PRs:
```bash
# Always check for open PRs first
gh pr list --state open
```

### Orchestration Decision Matrix

| Open PRs | Other Issues | Orchestration Decision |
|----------|-------------|------------------------|
| ✅ Yes | Any | Handle PRs ONLY - BLOCK all other work |
| ❌ No | Multiple | Proceed with normal orchestration |
| ❌ No | None | No work needed |

### PR Orchestration Examples

#### ❌ WRONG: Starting New Work with Open PRs
```python
# This is INCORRECT when PRs exist:
gh pr list --state open  # Shows: PR #45 "Fix authentication bug"

# But then orchestrator starts new work anyway:
Task(
    description="RIF-Analyst: Analyze new issue #50",
    prompt="..."
)  # Should NOT happen - PR #45 needs attention first
```

#### ✅ CORRECT: PR-First Orchestration
```python
# Check for PRs first
open_prs = check_open_prs()

if open_prs:
    # Handle PRs ONLY - block all other work
    for pr in open_prs:
        Task(
            description=f"RIF-Validator: Review and validate PR #{pr.number}",
            subagent_type="general-purpose",
            prompt=f"You are RIF-Validator. Review PR #{pr.number} '{pr.title}'. Run tests, check quality gates, validate implementation. If ready, approve and merge. If issues found, provide specific feedback. Follow all instructions in claude/agents/rif-validator.md."
        )
else:
    # No PRs - proceed with normal orchestration
    proceed_with_issue_orchestration()
```

### Key PR Priority Rules
1. **ALWAYS check for PRs before any other orchestration**
2. **NEVER start new development work while PRs are open**
3. **Launch PR validation agents FIRST**
4. **Complete PR workflows before issue orchestration**
5. **PRs override ALL other priority considerations**

## 🧠 ORCHESTRATION INTELLIGENCE FRAMEWORK

**CRITICAL**: Before launching any agents, the orchestrator MUST perform intelligent dependency analysis.

### Dependency Analysis Process (MANDATORY)

**Step 1: Issue Dependency Mapping**
```python
# BEFORE launching agents, analyze:
# 1. Sequential Dependencies: Research → Architecture → Implementation → Validation
# 2. Foundational Dependencies: Core systems before dependent features  
# 3. Blocking Dependencies: Critical infrastructure before all other work
# 4. Integration Dependencies: APIs before integrations that use them
```

**Step 2: Critical Path Identification**
- **Blocking Issues**: Must complete before ANY other work (e.g., agent context reading failures)
- **Foundation Issues**: Core systems that other issues depend on
- **Sequential Phases**: Research/Architecture must complete before Implementation
- **Integration Issues**: Cannot start until prerequisite APIs/schemas exist

**Step 3: Intelligent Launch Decision**
```python
# DECISION FRAMEWORK:
if blocking_issues_exist:
    launch_agents_for_blocking_issues_ONLY()
elif foundation_incomplete and has_dependent_issues:
    launch_agents_for_foundation_issues_ONLY()  
elif research_phase_incomplete:
    launch_agents_for_research_issues_ONLY()
else:
    launch_parallel_agents_for_ready_issues()
```

### Orchestration Decision Examples

#### ❌ WRONG: Naive Parallel Launching
```python
# This ignores dependencies and wastes agent resources:
Task("Implement feature X", ...)  # Depends on incomplete research
Task("Validate feature Y", ...)   # Nothing to validate yet  
Task("Integrate system Z", ...)   # APIs don't exist yet
```

#### ✅ CORRECT: Dependency-Aware Orchestration
```python  
# First: Handle blocking issues
if critical_infrastructure_broken:
    Task("Fix agent context reading", ...)
    # STOP - no other work until this completes

# Then: Foundation before dependent work
elif core_apis_incomplete:
    Task("Implement core API framework", ...)
    Task("Create database schema", ...)
    # STOP - dependent work waits

# Then: Research before implementation  
elif research_incomplete:
    Task("Research live context architecture", ...)
    Task("Research dependency tracking", ...)
    # STOP - implementation waits

# Finally: Parallel work on ready issues
else:
    Task("Implement validated feature A", ...)
    Task("Implement validated feature B", ...)
```

### Key Rules for Intelligent Orchestration

1. **PR PRIORITY ABSOLUTE** - Always handle open PRs before any other work
2. **ANALYZE DEPENDENCIES FIRST** - Never launch agents without dependency analysis
3. **RESPECT SEQUENTIAL PHASES** - Research → Architecture → Implementation → Validation
4. **PRIORITIZE BLOCKING ISSUES** - Critical infrastructure before all other work
5. **ONE PHASE AT A TIME** - Don't start implementation until research/architecture complete
6. **DEEP RESEARCH REQUIRED** - Research must analyze HOW things work, not just WHAT they do
7. **NEVER do implementation work directly** - Always delegate to agents
8. **Launch multiple Tasks in ONE response** for parallel execution (when appropriate) - NEVER launch agents in separate responses
9. **Include full agent instructions** in the Task prompt
10. **Match agents to issue states** (analyst for state:new, implementer for state:implementing, etc.)
11. **Let agents handle GitHub interactions** (posting comments, changing labels)
12. **Trust the agent specialization** - Don't micromanage their work

## 🚨 CRITICAL: Enhanced Orchestration Intelligence (Issue #228)

**ISSUE #228 INTEGRATION**: Following the critical orchestration failure where Issue #225 declared "THIS ISSUE BLOCKS ALL OTHERS" but was ignored, RIF now includes enhanced blocking detection that prevents such failures.

### Enhanced Blocking Detection System

The enhanced orchestration intelligence includes sophisticated blocking detection that:
- **Scans issue bodies and comments** for explicit blocking declarations
- **Halts all orchestration** when blocking issues are detected
- **Prevents false positives** by requiring exact blocking phrases
- **Provides detailed analysis** of why orchestration was blocked

### Supported Blocking Declarations

The system detects these exact phrases (case-insensitive):
- "**THIS ISSUE BLOCKS ALL OTHERS**"
- "**THIS ISSUE BLOCKS ALL OTHER WORK**"
- "**BLOCKS ALL OTHER WORK**"
- "**BLOCKS ALL OTHERS**"
- "**STOP ALL WORK**"
- "**MUST COMPLETE BEFORE ALL**"
- "**MUST COMPLETE BEFORE ALL OTHER WORK**"
- "**MUST COMPLETE BEFORE ALL OTHERS**"

**Note**: Generic terms like "critical", "urgent", or "blocking" alone do NOT trigger blocking detection to prevent false positives.

### Orchestration Intelligence Integration

Enhanced orchestration includes three levels of intelligence:

#### 1. Pre-Flight Blocking Detection
```python
from claude.commands.orchestration_intelligence_integration import validate_orchestration_request

# MANDATORY: Check for blocking issues before any orchestration
should_block, message = validate_orchestration_request([225, 226, 227])

if should_block:
    print(f"🚨 ORCHESTRATION BLOCKED: {message}")
    return  # Do not proceed with orchestration
```

#### 2. Intelligent Orchestration Decision
```python
from claude.commands.orchestration_intelligence_integration import make_intelligent_orchestration_decision

# Generate intelligent orchestration plan with blocking detection
decision = make_intelligent_orchestration_decision([225, 226, 227])

if decision.enforcement_action == "HALT_ALL_ORCHESTRATION":
    print(f"🚨 BLOCKING ISSUES DETECTED: {decision.blocking_issues}")
    print(f"🚫 BLOCKED ISSUES: {decision.blocked_issues}")
    print(f"✅ ALLOWED WORK: Only {decision.allowed_issues}")
    
    # Launch only blocking issue tasks
    for task_code in decision.task_launch_codes:
        exec(task_code)  # Execute the Task() commands for blocking issues only
    return

# Normal orchestration for non-blocking scenarios
for task_code in decision.task_launch_codes:
    exec(task_code)
```

#### 3. Pre-Flight Validation Command
```bash
# Standalone validation tool for blocking detection
python claude/commands/pre_flight_blocking_validator.py --issues 225,226,227

# Output:
# 🚨 RESULT: HALT ORCHESTRATION - 1 blocking issues detected
# or
# ✅ RESULT: ALLOW ORCHESTRATION - No blocking issues detected
```

## 🚨 CRITICAL: Enhanced Orchestration Intelligence (Issue #228)

**ISSUE #228 INTEGRATION**: Following the critical orchestration failure where Issue #225 declared "THIS ISSUE BLOCKS ALL OTHERS" but was ignored, RIF now includes enhanced blocking detection that prevents such failures.

### Enhanced Blocking Detection System

The enhanced orchestration intelligence includes sophisticated blocking detection that:
- **Scans issue bodies and comments** for explicit blocking declarations
- **Halts all orchestration** when blocking issues are detected
- **Prevents false positives** by requiring exact blocking phrases
- **Provides detailed analysis** of why orchestration was blocked

### Supported Blocking Declarations

The system detects these exact phrases (case-insensitive):
- "**THIS ISSUE BLOCKS ALL OTHERS**"
- "**THIS ISSUE BLOCKS ALL OTHER WORK**"
- "**BLOCKS ALL OTHER WORK**"
- "**BLOCKS ALL OTHERS**"
- "**STOP ALL WORK**"
- "**MUST COMPLETE BEFORE ALL**"
- "**MUST COMPLETE BEFORE ALL OTHER WORK**"
- "**MUST COMPLETE BEFORE ALL OTHERS**"

**Note**: Generic terms like "critical", "urgent", or "blocking" alone do NOT trigger blocking detection to prevent false positives.

### Orchestration Intelligence Integration

Enhanced orchestration includes three levels of intelligence:

#### 1. Pre-Flight Blocking Detection
```python
from claude.commands.orchestration_intelligence_integration import validate_orchestration_request

# MANDATORY: Check for blocking issues before any orchestration
should_block, message = validate_orchestration_request([225, 226, 227])

if should_block:
    print(f"🚨 ORCHESTRATION BLOCKED: {message}")
    return  # Do not proceed with orchestration
```

#### 2. Intelligent Orchestration Decision
```python
from claude.commands.orchestration_intelligence_integration import make_intelligent_orchestration_decision

# Generate intelligent orchestration plan with blocking detection
decision = make_intelligent_orchestration_decision([225, 226, 227])

if decision.enforcement_action == "HALT_ALL_ORCHESTRATION":
    print(f"🚨 BLOCKING ISSUES DETECTED: {decision.blocking_issues}")
    print(f"🚫 BLOCKED ISSUES: {decision.blocked_issues}")
    print(f"✅ ALLOWED WORK: Only {decision.allowed_issues}")
    
    # Launch only blocking issue tasks
    for task_code in decision.task_launch_codes:
        exec(task_code)  # Execute the Task() commands for blocking issues only
    return

# Normal orchestration for non-blocking scenarios
for task_code in decision.task_launch_codes:
    exec(task_code)
```

#### 3. Pre-Flight Validation Command
```bash
# Standalone validation tool for blocking detection
python claude/commands/pre_flight_blocking_validator.py --issues 225,226,227

# Output:
# 🚨 RESULT: HALT ORCHESTRATION - 1 blocking issues detected
# or
# ✅ RESULT: ALLOW ORCHESTRATION - No blocking issues detected
```
### Orchestration Failure Prevention

The enhanced system prevents these critical failures:
1. **Ignoring explicit blocking declarations** (like Issue #225)
2. **Proceeding with parallel work** when blocking issues exist
3. **False positive blocking** from generic urgent language
4. **Missed blocking declarations** in issue comments

### Integration Architecture

```
GitHub Issues → Enhanced Blocking Detection → Orchestration Decision
                        ↓
                Pre-Flight Validation
                        ↓  
                Intelligent Analysis
                        ↓
                Task Launch Codes (blocking-aware)
```

### Enhanced Orchestration Example

**Scenario**: Issues #225 (with "THIS ISSUE BLOCKS ALL OTHERS"), #226, and #227

**Before Issue #228 Fix**:
```python
# ❌ WRONG: Would launch all tasks in parallel
Task("Work on issue #225", ...)
Task("Work on issue #226", ...)  # Should be blocked!
Task("Work on issue #227", ...)  # Should be blocked!
```

**After Issue #228 Fix**:
```python
from claude.commands.orchestration_intelligence_integration import make_intelligent_orchestration_decision

decision = make_intelligent_orchestration_decision([225, 226, 227])

# System detects Issue #225 blocks others:
# decision.enforcement_action == "HALT_ALL_ORCHESTRATION"
# decision.blocking_issues == [225]
# decision.blocked_issues == [226, 227]

# Only launches task for blocking issue:
Task(
    description="Resolve BLOCKING issue #225 (THIS ISSUE BLOCKS ALL OTHERS)",
    subagent_type="general-purpose",
    prompt="You are RIF-Implementer. Resolve BLOCKING issue #225 immediately. This issue blocks all other work. Follow all instructions in claude/agents/rif-implementer.md."
)
# Issues #226 and #227 are automatically blocked until #225 completes
```

### Quick Reference: Intelligent Agent Launching Template (ENHANCED for Issue #223 & #228)

```python
# Step 1: Check GitHub issues (Claude Code does this directly)
github_issues = get_current_github_issues()  # Claude Code handles this

# Step 2: MANDATORY BLOCKING DETECTION (CRITICAL FIX for Issue #228)
from claude.commands.orchestration_intelligence_integration import make_intelligent_orchestration_decision

# CRITICAL: Enhanced blocking detection prevents orchestration failures
orchestration_decision = make_intelligent_orchestration_decision(github_issues)

# Handle blocking issues FIRST (Issue #228 fix)
if orchestration_decision.enforcement_action == "HALT_ALL_ORCHESTRATION":
    print("🚨 BLOCKING ISSUES DETECTED - HALTING ALL OTHER ORCHESTRATION")
    print(f"   Blocking Issues: {orchestration_decision.blocking_issues}")
    print(f"   Blocked Issues: {orchestration_decision.blocked_issues}")
    print("   ACTION: Complete blocking issues before proceeding with others")
    
    # Execute ONLY blocking issue tasks
    for task_code in orchestration_decision.task_launch_codes:
        # Task codes are ready-to-execute strings
        exec(task_code)
    return  # Do not proceed with other issues

# Step 3: PHASE DEPENDENCY ENFORCEMENT (Issue #223)
from claude.commands.simple_phase_dependency_enforcer import enforce_orchestration_phase_dependencies

# Define proposed agent launches (only for non-blocking issues)
proposed_agent_launches = [
    {
        "description": "[AGENT_NAME]: [BRIEF_TASK_DESCRIPTION]",
        "prompt": "You are [AGENT_NAME]. [SPECIFIC_TASK_DETAILS]. Follow all instructions in claude/agents/[agent-file].md.",
        "subagent_type": "general-purpose"
    }
    # Add more proposed launches...
]

# ENFORCE PHASE DEPENDENCIES
enforcement_result = enforce_orchestration_phase_dependencies(github_issues, proposed_agent_launches)

# Step 4: Execute based on combined enforcement decision
if not enforcement_result.is_execution_allowed:
    print("❌ PHASE DEPENDENCY VIOLATIONS - EXECUTION BLOCKED")
    print(f"Decision: {enforcement_result.decision_type.value}")
    print("Violations:")
    for violation in enforcement_result.violations:
        print(f"  - {violation}")
    print("Required Actions:")  
    for action in enforcement_result.remediation_actions:
        print(f"  → {action}")
    
    # Execute prerequisite tasks instead of blocked tasks
    print("\n🔄 EXECUTING PREREQUISITE TASKS:")
    for task in enforcement_result.prerequisite_tasks:
        Task(
            description=task["description"],
            subagent_type=task["subagent_type"], 
            prompt=task["prompt"]
        )
        
else:
    print("✅ ALL VALIDATIONS PASSED - PROCEEDING WITH EXECUTION")
    print(f"Phase Decision: {enforcement_result.decision_type.value}")
    print(f"Blocking Decision: {orchestration_decision.enforcement_action}")
    print(f"Confidence: {enforcement_result.confidence_score:.2f}")
    
    # Execute validated tasks in parallel
    for task in enforcement_result.allowed_tasks:
        Task(
            description=task["description"],
            subagent_type=task["subagent_type"],
            prompt=task["prompt"]
        )

# Enhanced Orchestration Status:
# Blocking Detection: ✅ ACTIVE (Issue #228)
# Phase Dependency Enforcement: ✅ ACTIVE (Issue #223)
# Sequential Phase Validation: ✅ ENFORCED
# False Positive Prevention: ✅ IMPLEMENTED
        
else:
    print("✅ ALL VALIDATIONS PASSED - PROCEEDING WITH EXECUTION")
    print(f"Phase Decision: {enforcement_result.decision_type.value}")
    print(f"Blocking Decision: {orchestration_decision.enforcement_action}")
    print(f"Confidence: {enforcement_result.confidence_score:.2f}")
    
    # Execute validated tasks in parallel
    for task in enforcement_result.allowed_tasks:
        Task(
            description=task["description"],
            subagent_type=task["subagent_type"],
            prompt=task["prompt"]
        )

# Enhanced Orchestration Status:
# Blocking Detection: ✅ ACTIVE (Issue #228)
# Phase Dependency Enforcement: ✅ ACTIVE (Issue #223)
# Sequential Phase Validation: ✅ ENFORCED
# False Positive Prevention: ✅ IMPLEMENTED
```

### Intelligent Orchestration Scenarios

| Scenario | Dependency Analysis | Orchestration Decision | Agent Tasks |
|----------|--------------------|-----------------------|-------------|
| **Critical infrastructure broken** | Blocking issue affects all agents | Launch blocking agents ONLY | Fix context reading, repair core systems |
| **Foundation APIs missing** | Implementation depends on APIs | Launch foundation agents ONLY | Core API framework, database schema |
| **Research phase incomplete** | Implementation depends on research | Launch research agents ONLY | Architecture research, methodology analysis |
| **Multiple ready issues** | No blocking dependencies | Launch parallel agents | Multiple implementers/validators in parallel |
| **Mixed states with dependencies** | Some dependent, some ready | Prioritized sequential launch | Foundation first, then dependent work |
| **New issue in dependent chain** | Check position in dependency chain | Wait or immediate based on prerequisites | RIF-Analyst only if no blocking dependencies |

### DPIBS Orchestration Example

**Current State**: 25+ DPIBS issues in various states
**Problem**: Implementation/validation issues launched before research/architecture complete

**Correct Orchestration**:
1. **Phase 1** (Research): Issues #133-136 - Launch research agents ONLY
2. **Phase 2** (Architecture): Issues #117,#120-122 - Launch after research complete  
3. **Phase 3** (Implementation): Issues #123,#127-128,#137-138 - Launch after architecture complete
4. **Phase 4** (Validation): Issues #129-132 - Launch after implementation complete

### Troubleshooting Intelligent Orchestration

**Problem**: Claude is doing work directly instead of launching agents
**Solution**: Always use Task() tool to delegate work to specialized agents

**Problem**: Agents are not running in parallel  
**Solution**: Launch all Tasks in a single Claude response, not separate responses

### 🔄 CRITICAL: Parallel Execution Requirements

**MANDATORY**: ALL Task() invocations for parallel work MUST be in the SAME message/response.

#### ❌ WRONG: Sequential Agent Launching
```python
# This is INCORRECT - agents launch sequentially:
Task(
    description="RIF-Analyst: Analyze issue #5",
    prompt="You are RIF-Analyst..."
)
# ANY TEXT OR ANALYSIS HERE BREAKS PARALLEL EXECUTION
# This creates a separate response:
Task(
    description="RIF-Implementer: Implement issue #3", 
    prompt="You are RIF-Implementer..."
)
```

#### ✅ CORRECT: True Parallel Agent Launching
```python
# This is CORRECT - all agents launch in parallel:
Task(
    description="RIF-Analyst: Analyze issue #5",
    subagent_type="general-purpose",
    prompt="You are RIF-Analyst. Analyze GitHub issue #5. Follow all instructions in claude/agents/rif-analyst.md."
)
Task(
    description="RIF-Implementer: Implement issue #3",
    subagent_type="general-purpose", 
    prompt="You are RIF-Implementer. Implement fix for GitHub issue #3. Follow all instructions in claude/agents/rif-implementer.md."
)
Task(
    description="RIF-Validator: Validate issue #1",
    subagent_type="general-purpose",
    prompt="You are RIF-Validator. Validate implementation for GitHub issue #1. Follow all instructions in claude/agents/rif-validator.md."
)
# All three agents start working simultaneously
```

#### Parallel vs Sequential Decision Matrix

| Scenario | Dependencies | Launch Pattern | Justification |
|----------|-------------|---------------|--------------|
| Multiple ready issues | None | ✅ Parallel (same response) | Maximum efficiency |
| Research + Implementation | Research must complete first | ❌ Sequential (research only) | Dependency constraint |
| Validation of complete work | Implementation done | ✅ Parallel validation | Independent validation tasks |
| Foundation + Dependent work | Foundation must complete | ❌ Sequential (foundation only) | Architectural dependency |
| Mixed states, no dependencies | None | ✅ Parallel (same response) | Optimal resource utilization |

#### Key Parallel Execution Rules
1. **Same Response Rule**: All parallel Tasks MUST be in the same Claude response
2. **No Interruption Rule**: No text, analysis, or explanations between parallel Task() calls
3. **Dependency Check Rule**: Only launch in parallel if no sequential dependencies exist
4. **Resource Rule**: Maximum 4 parallel agents to avoid resource exhaustion
5. **State Verification Rule**: Ensure all parallel work can proceed independently

**Problem**: Implementation agents launched for issues with incomplete prerequisites
**Solution**: Perform mandatory dependency analysis - check research/architecture completion first

**Problem**: Validation agents launched when nothing is ready to validate
**Solution**: Apply sequential phase logic - implementation must complete before validation

**Problem**: Multiple agents working on dependent issues causing conflicts
**Solution**: Use foundation-first orchestration - core systems before dependent features

**Problem**: Critical infrastructure issues ignored while other work proceeds  
**Solution**: Identify blocking issues first - launch blocking agents ONLY until resolved

**Problem**: Agent instructions are incomplete
**Solution**: Always include "Follow all instructions in claude/agents/[agent-file].md" in prompts

**Problem**: Orchestration decisions appear arbitrary or inefficient
**Solution**: Document dependency analysis reasoning before launching agents

## 🔬 CRITICAL: Deep Research Requirements

**MANDATORY**: When research tasks involve analyzing other systems, methodologies, or implementations, the research MUST focus on **HOW** they work, not just **WHAT** they do.

### Research Quality Requirements

#### ❌ INSUFFICIENT: Surface-Level Research
```markdown
# This type of research is NOT sufficient:
"BMAD-METHOD is a development methodology that uses branches and automated deployment.
It has the following features:
- Branch-based development
- Automated testing
- Deployment automation
- Issue tracking integration"
```

#### ✅ REQUIRED: Deep Implementation Analysis
```markdown
# This type of research IS required:
"BMAD-METHOD Implementation Analysis:

**HOW Branch Management Works:**
- Uses specific git hooks to enforce branch naming: `feature/ISSUE-123-description`
- Automated scripts scan commit messages for issue references
- Branch protection rules prevent direct pushes to main
- Specific merge strategy: squash merges with automated commit message formatting

**HOW Automated Testing Integration Functions:**
- GitHub Actions workflow triggers on PR creation with specific job matrix
- Test execution follows this sequence: lint → unit → integration → e2e
- Failure at any stage blocks subsequent stages using job dependencies
- Test results are parsed and posted as PR comments using specific API calls

**HOW Deployment Automation Mechanism Works:**
- Uses deployment branches (staging, production) that trigger different workflows
- Environment promotion happens through specific git tag patterns: v1.2.3-staging
- Rollback mechanism uses git revert with automated environment sync
- Configuration management through environment-specific config files

**HOW Issue Integration Functions:**
- Webhook integration between GitHub Issues and deployment system
- Issue state automatically updates based on deployment success/failure
- Specific label management rules: state:deploying → state:deployed
- Automated issue closure on successful production deployment

**Specific Implementation Details:**
- Configuration files: .bmad/config.yaml, .github/workflows/bmad.yml
- Required repository structure: /deploy, /config, /tests
- Dependency on specific tools: gh CLI, jq, deployment scripts
- Integration points: GitHub API, deployment targets, monitoring systems"
```

### Research Agent Enhancement

When launching research agents, prompts MUST include deep analysis requirements:

```python
# Enhanced research agent prompt:
Task(
    description="RIF-Researcher: Deep analysis of BMAD-METHOD implementation",
    subagent_type="general-purpose",
    prompt="You are RIF-Researcher. Conduct DEEP analysis of BMAD-METHOD focusing on HOW it works, not what it does. Your research must include: 1) Specific implementation mechanisms and code examples, 2) Actual workflows and automation scripts, 3) Integration patterns and API usage, 4) Configuration details and file structures, 5) Decision logic and algorithms used. Provide implementation-ready insights that show exactly HOW to build similar functionality. Follow all instructions in claude/agents/rif-researcher.md."
)
```

### Deep Research Quality Gates

**Research is INSUFFICIENT if it only provides:**
- High-level feature descriptions
- Marketing or documentation summaries
- Conceptual overviews without implementation details
- General benefits without specific mechanisms

**Research is SUFFICIENT when it includes:**
- Specific code patterns and implementation examples
- Detailed workflow sequences with actual steps
- Configuration files, scripts, and automation details
- Integration patterns with exact API calls or tool usage
- Decision trees and algorithmic logic
- File structures and architectural components
- Command sequences and tool invocations

### Research Application Requirements

Deep research must enable the implementer to:
1. **Replicate the approach** with specific implementation guidance
2. **Understand the mechanisms** behind the functionality
3. **Adapt the patterns** to the current system architecture
4. **Avoid implementation pitfalls** through detailed analysis
5. **Make informed decisions** based on how systems actually work

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