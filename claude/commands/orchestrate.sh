#!/bin/bash
# RIF Orchestration Pattern Documentation
# CRITICAL: This documents how Claude Code should orchestrate - NOT a script to run via Task

cat << 'EOF'
=================================================================
RIF ORCHESTRATION PATTERN - HOW CLAUDE CODE SHOULD ORCHESTRATE
=================================================================

🚨 CRITICAL UNDERSTANDING:
- The RIF Orchestrator is NOT a Task to launch
- Claude Code IS the orchestrator
- Orchestration means launching MULTIPLE Tasks in ONE response

=================================================================
CORRECT ORCHESTRATION WORKFLOW
=================================================================

When user says "launch RIF orchestrator" or "orchestrate":

1. CLAUDE CODE DIRECTLY checks issues (not via Task):
   gh issue list --state open --label 'state:*' --json number,title,labels,body

2. CLAUDE CODE DIRECTLY analyzes and determines agents needed:
   - state:new → RIF-Analyst
   - state:planning → RIF-Planner  
   - state:architecting → RIF-Architect
   - state:implementing → RIF-Implementer
   - state:validating → RIF-Validator

3. CLAUDE CODE launches MULTIPLE Tasks IN ONE RESPONSE:

=================================================================
EXAMPLE: PARALLEL TASK EXECUTION (CORRECT PATTERN)
=================================================================

# In a SINGLE Claude response, launch multiple Tasks:

Task(
    description="RIF-Analyst for issue #3",
    subagent_type="general-purpose",
    prompt="""You are RIF-Analyst, responsible for requirements analysis.
    
    Your issue: #3 - Analyze LightRAG usage
    
    Instructions:
    1. Read issue #3 from GitHub
    2. Analyze codebase for LightRAG implementation
    3. Search knowledge base for patterns
    4. Generate comprehensive analysis
    5. Post findings as GitHub comment
    6. Update issue labels to state:planning when complete
    
    [Include full contents of rif-analyst.md here]"""
)

Task(
    description="RIF-Implementer for issue #2", 
    subagent_type="general-purpose",
    prompt="""You are RIF-Implementer, responsible for implementation.
    
    Your issue: #2 - Fix agent delegation
    
    Instructions:
    1. Read issue #2 and #4 for context
    2. Implement the orchestration fix
    3. Create checkpoints for progress
    4. Write tests for changes
    5. Update documentation
    6. Create PR when complete
    7. Update labels to state:validating
    
    [Include full contents of rif-implementer.md here]"""
)

Task(
    description="RIF-Validator for issue #1",
    subagent_type="general-purpose", 
    prompt="""You are RIF-Validator, responsible for validation.
    
    Your issue: #1 - Validate completed work
    
    Instructions:
    1. Run all tests
    2. Check quality gates
    3. Verify documentation
    4. Post validation report
    5. Update labels appropriately
    
    [Include full contents of rif-validator.md here]"""
)

=================================================================
WHY THIS WORKS
=================================================================

✅ Multiple Tasks in ONE response = PARALLEL execution
✅ Each Task is a specialized agent with full instructions
✅ Tasks work independently on different issues
✅ True parallel processing, not sequential

=================================================================
COMMON MISTAKES TO AVOID
=================================================================

❌ DON'T launch "RIF-Orchestrator" as a Task
❌ DON'T use Task to check issues 
❌ DON'T execute work sequentially
❌ DON'T treat "Task.parallel()" as a real function

=================================================================
REMEMBER
=================================================================

"Task.parallel()" in documentation is PSEUDOCODE meaning:
"Launch multiple Task invocations in a single Claude response"

The orchestrator is YOU (Claude Code) launching parallel Tasks!
EOF