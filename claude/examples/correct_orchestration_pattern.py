#!/usr/bin/env python3
"""
RIF Correct Orchestration Pattern - Examples for Claude Code

This file demonstrates the CORRECT way to implement RIF orchestration,
where Claude Code IS the orchestrator and uses utilities to launch Task agents.

CRITICAL: This shows the pattern-compliant approach that RIF-Validator approved.
"""

import sys
sys.path.append('/Users/cal/DEV/RIF/claude/commands')
from orchestration_utilities import (
    ContextAnalyzer, StateValidator, OrchestrationHelper, GitHubStateManager
)

def example_orchestration_for_single_issue():
    """
    Example: How Claude Code should orchestrate a single issue
    
    This is what Claude Code should do when asked to "orchestrate issue #52"
    """
    print("=== CORRECT ORCHESTRATION PATTERN: Single Issue ===")
    
    # Step 1: Claude Code uses utilities to analyze the issue
    helper = OrchestrationHelper()
    recommendation = helper.recommend_orchestration_action(52)
    
    print(f"Analysis complete for issue #{recommendation.get('issue', 'unknown')}")
    print(f"Current State: {recommendation.get('current_state', 'none')}")
    print(f"Recommended Agent: {recommendation.get('recommended_agent', 'none')}")
    
    # Step 2: Claude Code executes the Task launch code
    if recommendation['action'] == 'launch_agent':
        task_code = recommendation['task_launch_code']
        print("\n🚀 Claude Code should now execute this Task() call:")
        print("=" * 60)
        print(task_code)
        print("=" * 60)
        
        print("\n✅ CORRECT: Claude Code launches agent via Task() tool")
        print("❌ WRONG: Creating DynamicOrchestrator() instance")
    else:
        print(f"No action needed: {recommendation.get('reason', 'Unknown reason')}")


def example_orchestration_for_multiple_issues():
    """
    Example: How Claude Code should orchestrate multiple issues in parallel
    
    This is what Claude Code should do when asked to "orchestrate all open issues"
    """
    print("\n=== CORRECT ORCHESTRATION PATTERN: Multiple Issues ===")
    
    # Step 1: Get active issues
    github_manager = GitHubStateManager()
    active_issues = github_manager.get_active_issues()
    
    if not active_issues:
        print("No active issues found")
        return
    
    issue_numbers = [issue['number'] for issue in active_issues[:3]]  # Limit for example
    print(f"Found {len(issue_numbers)} active issues: {issue_numbers}")
    
    # Step 2: Generate orchestration plan
    helper = OrchestrationHelper()
    plan = helper.generate_orchestration_plan(issue_numbers)
    
    print(f"\nOrchestration Plan Generated:")
    print(f"- Total Issues: {plan['total_issues']}")
    print(f"- Parallel Tasks: {len(plan['parallel_tasks'])}")
    
    # Step 3: Show Task launch codes for parallel execution
    print("\n🚀 Claude Code should execute ALL these Tasks in ONE response for parallel execution:")
    print("=" * 80)
    
    for i, task_code in enumerate(plan['task_launch_codes'], 1):
        print(f"# Task {i}:")
        print(task_code)
        print()
    
    print("=" * 80)
    print("✅ CORRECT: All Task() calls in single Claude response = parallel execution")
    print("❌ WRONG: Creating orchestrator to manage agents")


def example_state_progression():
    """
    Example: How states should progress through the RIF workflow
    """
    print("\n=== CORRECT STATE PROGRESSION PATTERN ===")
    
    validator = StateValidator()
    
    # Show the correct workflow
    states = ['new', 'analyzing', 'planning', 'implementing', 'validating', 'learning', 'complete']
    
    print("RIF Workflow State Machine:")
    for i, state in enumerate(states):
        agent = validator.get_required_agent(state)
        if agent:
            print(f"{i+1}. state:{state} → {agent}")
        else:
            print(f"{i+1}. state:{state} → (no agent needed)")
        
        if i < len(states) - 1:
            next_states = validator.VALID_TRANSITIONS.get(state, [])
            if next_states:
                print(f"   └─ Can transition to: {next_states}")
    
    print("\n✅ CORRECT: Claude Code checks states and launches appropriate agents")
    print("❌ WRONG: Orchestrator class managing state transitions")


def example_task_launch_code_formats():
    """
    Example: Show different Task() launch code formats for different agents
    """
    print("\n=== CORRECT TASK LAUNCH CODE FORMATS ===")
    
    helper = OrchestrationHelper()
    
    # Mock context for examples
    from orchestration_utilities import IssueContext
    
    sample_context = IssueContext(
        number=52,
        title="Implement DynamicOrchestrator class",
        body="Need to implement the orchestrator system",
        labels=['state:implementing', 'complexity:high'],
        state='open',
        complexity='high',
        priority='high',
        agent_history=['RIF-Analyst', 'RIF-Planner'],
        created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T01:00:00Z',
        comments_count=5
    )
    
    agents = ['RIF-Analyst', 'RIF-Implementer', 'RIF-Validator', 'RIF-Learner']
    
    print("Task Launch Code Examples for Different Agents:")
    print("=" * 60)
    
    for agent in agents:
        print(f"\n# {agent}:")
        task_code = helper.generate_task_launch_code(sample_context, agent)
        print(task_code)
    
    print("=" * 60)
    print("✅ CORRECT: Each Task() call creates a specialized agent")
    print("❌ WRONG: Orchestrator.assign_agent() or similar methods")


def example_github_integration():
    """
    Example: How Claude Code should interact with GitHub
    """
    print("\n=== CORRECT GITHUB INTEGRATION PATTERN ===")
    
    github_manager = GitHubStateManager()
    
    print("GitHub Operations Claude Code Should Perform:")
    print("1. Check issue states directly (not via orchestrator)")
    print("2. Update states after agent completion")
    print("3. Add tracking labels for agent execution")
    
    # Example state update
    print("\nExample: Update issue state after RIF-Implementer completes")
    success = github_manager.update_issue_state(
        52, 
        'validating', 
        "Implementation complete. Ready for RIF-Validator."
    )
    
    if success:
        print("✅ State updated to: state:validating")
        print("✅ Comment added explaining transition")
    
    print("\n✅ CORRECT: Claude Code directly manages GitHub states")
    print("❌ WRONG: Orchestrator managing GitHub interactions")


def demonstrate_anti_patterns():
    """
    Show what NOT to do - the patterns that RIF-Validator rejected
    """
    print("\n=== ANTI-PATTERNS - WHAT NOT TO DO ===")
    print("🚨 These patterns violate RIF principles and will be rejected:")
    
    print("\n❌ WRONG PATTERN 1: Creating Orchestrator Class")
    print("```python")
    print("# This violates the fundamental RIF principle:")
    print("orchestrator = DynamicOrchestrator()")
    print("orchestrator.process_issue(52)")
    print("```")
    print("PROBLEM: Claude Code IS the orchestrator, not a separate class")
    
    print("\n❌ WRONG PATTERN 2: Agent Management by Class")
    print("```python")
    print("# This is incorrect agent management:")
    print("orchestrator.assign_agent('RIF-Implementer', issue_52)")
    print("orchestrator.execute_workflow()")
    print("```")
    print("PROBLEM: Claude Code should launch agents via Task() tool")
    
    print("\n❌ WRONG PATTERN 3: Complex State Management Classes")
    print("```python")
    print("# This over-complicates the simple state progression:")
    print("state_manager = ComplexStateManager()")
    print("workflow = WorkflowEngine()")
    print("state_manager.transition_with_validation(issue, new_state)")
    print("```")
    print("PROBLEM: Simple state labels in GitHub are sufficient")
    
    print("\n❌ WRONG PATTERN 4: Orchestrator-to-Orchestrator Communication")
    print("```python")
    print("# This creates unnecessary complexity:")
    print("main_orchestrator.delegate_to_sub_orchestrator(sub_orchestrator)")
    print("```")
    print("PROBLEM: Claude Code handles everything directly")
    
    print("\n✅ CORRECT APPROACH: Use utilities to support Claude Code")
    print("```python")
    print("# This is the correct pattern:")
    print("helper = OrchestrationHelper()")
    print("recommendation = helper.recommend_orchestration_action(52)")
    print("# Claude Code then executes: Task(...)")
    print("```")


def main():
    """
    Main demonstration of correct RIF orchestration patterns
    """
    print("🚀 RIF Correct Orchestration Pattern Examples")
    print("=" * 80)
    print("These examples show how Claude Code should orchestrate RIF agents")
    print("following the pattern-compliant approach that RIF-Validator approved.")
    print("=" * 80)
    
    try:
        example_orchestration_for_single_issue()
        example_orchestration_for_multiple_issues() 
        example_state_progression()
        example_task_launch_code_formats()
        example_github_integration()
        demonstrate_anti_patterns()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("This is expected when utilities can't connect to GitHub")
        print("The pattern examples above show the correct approach.")
    
    print("\n🎉 SUMMARY: Pattern-Compliant RIF Orchestration")
    print("=" * 50)
    print("✅ Claude Code IS the orchestrator")
    print("✅ Utilities support orchestration decisions")
    print("✅ Task() tool launches specialized agents")
    print("✅ Agents run in parallel when launched together")
    print("✅ GitHub states tracked with simple labels")
    print("✅ No complex orchestrator classes needed")
    print("\n🚨 This approach was validated and approved by RIF-Validator!")


if __name__ == "__main__":
    main()