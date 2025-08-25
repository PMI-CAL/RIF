#!/usr/bin/env python3
"""
Orchestration Phase Dependency Enforcer

CRITICAL FIX FOR ISSUE #223: RIF Orchestration Error: Not Following Phase Dependencies

This module implements the mandatory phase dependency enforcement that was missing from 
the RIF orchestration system. It integrates with the existing orchestration framework
to ensure that phase dependencies are strictly enforced before any agent launching.

Integration Points:
- Enhanced orchestration intelligence
- Phase dependency validator 
- GitHub state management
- CLAUDE.md orchestration template

Issue: #223 - RIF Orchestration Error: Not Following Phase Dependencies
"""

import json
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

# Import existing phase dependency components
try:
    from .phase_dependency_validator import (
        PhaseDependencyValidator, 
        PhaseValidationResult,
        PhaseType,
        PhaseDependencyViolation
    )
except ImportError:
    # Handle import for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from phase_dependency_validator import (
        PhaseDependencyValidator, 
        PhaseValidationResult,
        PhaseType,
        PhaseDependencyViolation
    )


class OrchestrationDecisionType(Enum):
    """Types of orchestration decisions"""
    ALLOW_PARALLEL_EXECUTION = "allow_parallel_execution"
    BLOCK_PHASE_VIOLATIONS = "block_phase_violations"
    REDIRECT_TO_PREREQUISITES = "redirect_to_prerequisites"
    BLOCK_CRITICAL_VIOLATIONS = "block_critical_violations"
    ALLOW_WITH_WARNINGS = "allow_with_warnings"


@dataclass
class OrchestrationEnforcementResult:
    """Result of orchestration phase dependency enforcement"""
    decision_type: OrchestrationDecisionType
    is_execution_allowed: bool
    phase_validation_result: PhaseValidationResult
    allowed_tasks: List[Dict[str, Any]]
    blocked_tasks: List[Dict[str, Any]]
    prerequisite_tasks: List[Dict[str, Any]]
    violations: List[str]
    warnings: List[str]
    remediation_actions: List[str]
    confidence_score: float
    execution_rationale: str
    github_notifications: List[Dict[str, Any]]
    enforcement_timestamp: str


@dataclass 
class AgentLaunchRequest:
    """Standardized agent launch request"""
    agent_type: str
    description: str
    prompt: str
    target_issues: List[int]
    subagent_type: str = "general-purpose"
    priority: str = "normal"


class OrchestrationPhaseDependencyEnforcer:
    """
    Main enforcement system for phase dependencies in RIF orchestration.
    
    This is the CRITICAL FIX for issue #223. This class integrates with the existing
    orchestration system to ensure phase dependencies are enforced BEFORE agent launching.
    
    Key Functions:
    1. Validate phase dependencies before any orchestration decision
    2. Block execution when critical phase violations exist
    3. Redirect to prerequisite phases when dependencies are missing
    4. Generate corrective actions and warnings
    5. Integrate with GitHub for notification and state management
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.phase_validator = PhaseDependencyValidator(knowledge_base_path)
        self.enforcement_history = []
        
        # Set up paths
        self.knowledge_path = Path(self.knowledge_base_path)
        self.enforcement_log_path = self.knowledge_path / "enforcement_logs"
        self.enforcement_log_path.mkdir(exist_ok=True)
        
    def enforce_phase_dependencies(
        self,
        github_issues: List[Dict[str, Any]], 
        proposed_agent_launches: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationEnforcementResult:
        """
        MAIN ENFORCEMENT FUNCTION - This is the critical fix for issue #223.
        
        This function MUST be called before any agent launching in the orchestration system.
        It validates phase dependencies and blocks execution if violations exist.
        """
        
        print("🔍 ENFORCING PHASE DEPENDENCIES (Issue #223 Fix)")
        
        # Step 1: Validate phase dependencies using existing validator
        print("   Step 1: Validating phase dependencies...")
        phase_validation_result = self.phase_validator.validate_phase_dependencies(
            github_issues, proposed_agent_launches
        )
        
        # Step 2: Analyze enforcement decision
        print("   Step 2: Analyzing enforcement decision...")
        decision_type, execution_rationale = self._determine_enforcement_decision(
            phase_validation_result, github_issues, proposed_agent_launches
        )
        
        # Step 3: Categorize tasks based on phase validation
        print("   Step 3: Categorizing tasks...")
        allowed_tasks, blocked_tasks, prerequisite_tasks = self._categorize_tasks_by_phase_validation(
            proposed_agent_launches, phase_validation_result, github_issues
        )
        
        # Step 4: Generate violations, warnings, and remediation actions
        print("   Step 4: Generating guidance...")
        violations, warnings, remediation_actions = self._generate_enforcement_guidance(
            phase_validation_result, blocked_tasks, prerequisite_tasks
        )
        
        # Step 5: Calculate confidence score
        confidence_score = self._calculate_enforcement_confidence(
            phase_validation_result, allowed_tasks, blocked_tasks
        )
        
        # Step 6: Generate GitHub notifications if needed
        github_notifications = self._generate_github_notifications(
            violations, github_issues, blocked_tasks
        )
        
        # Step 7: Create enforcement result
        result = OrchestrationEnforcementResult(
            decision_type=decision_type,
            is_execution_allowed=decision_type in [
                OrchestrationDecisionType.ALLOW_PARALLEL_EXECUTION,
                OrchestrationDecisionType.ALLOW_WITH_WARNINGS
            ],
            phase_validation_result=phase_validation_result,
            allowed_tasks=allowed_tasks,
            blocked_tasks=blocked_tasks,
            prerequisite_tasks=prerequisite_tasks,
            violations=violations,
            warnings=warnings,
            remediation_actions=remediation_actions,
            confidence_score=confidence_score,
            execution_rationale=execution_rationale,
            github_notifications=github_notifications,
            enforcement_timestamp=datetime.utcnow().isoformat()
        )
        
        # Step 8: Store enforcement result
        self._store_enforcement_result(result)
        
        print(f"   ✅ Enforcement complete: {decision_type.value}")
        print(f"   📊 Execution allowed: {result.is_execution_allowed}")
        print(f"   📈 Confidence: {confidence_score:.2f}")
        print(f"   ⚠️  Violations: {len(violations)}")
        
        return result
        
    def generate_orchestration_template(
        self, 
        enforcement_result: OrchestrationEnforcementResult
    ) -> str:
        """
        Generate orchestration template with phase dependency enforcement integrated.
        
        This replaces the standard orchestration template to ensure phase dependencies
        are properly handled.
        """
        
        template_parts = []
        
        # Header with enforcement context
        template_parts.append(f'''# PHASE DEPENDENCY ENFORCEMENT ACTIVE (Issue #223 Fix)
# Decision: {enforcement_result.decision_type.value}
# Execution Allowed: {'✅ YES' if enforcement_result.is_execution_allowed else '❌ NO'}
# Phase Validation: {'✅ PASSED' if enforcement_result.phase_validation_result.is_valid else '❌ FAILED'}
# Confidence: {enforcement_result.confidence_score:.2f}
# Violations: {len(enforcement_result.violations)}
# Generated: {enforcement_result.enforcement_timestamp}
# 
# Rationale: {enforcement_result.execution_rationale}
''')
        
        # Phase violations section (if any)
        if enforcement_result.violations:
            template_parts.append('''
# ❌ PHASE DEPENDENCY VIOLATIONS DETECTED
# Execution BLOCKED until violations resolved:
#''')
            for i, violation in enumerate(enforcement_result.violations):
                template_parts.append(f'#   {i+1}. {violation}')
                
            template_parts.append('#')
            
        # Remediation actions section
        if enforcement_result.remediation_actions:
            template_parts.append('''# 🔧 REQUIRED REMEDIATION ACTIONS:
#''')
            for i, action in enumerate(enforcement_result.remediation_actions):
                template_parts.append(f'#   → {i+1}. {action}')
                
            template_parts.append('#')
            
        # Prerequisite tasks section (if blocking execution)
        if enforcement_result.prerequisite_tasks and not enforcement_result.is_execution_allowed:
            template_parts.append('''
# 🔄 PREREQUISITE TASKS (Execute these FIRST):
# Phase dependencies require these tasks before proceeding with blocked tasks:
''')
            for task in enforcement_result.prerequisite_tasks:
                template_parts.append(f'''
Task(
    description="{task['description']}",
    subagent_type="{task['subagent_type']}",
    prompt="{task['prompt']}"
)''')
                
        # Allowed tasks section (if execution is allowed)
        elif enforcement_result.is_execution_allowed and enforcement_result.allowed_tasks:
            template_parts.append('''
# ✅ PHASE DEPENDENCIES VALIDATED - PROCEEDING WITH EXECUTION
# The following tasks have been validated and are approved for parallel execution:
''')
            for task in enforcement_result.allowed_tasks:
                template_parts.append(f'''
Task(
    description="{task['description']}",
    subagent_type="{task['subagent_type']}",
    prompt="{task['prompt']}"
)''')
                
        # Blocked tasks summary
        if enforcement_result.blocked_tasks:
            template_parts.append(f'''
# 🚫 BLOCKED TASKS ({len(enforcement_result.blocked_tasks)} tasks blocked due to phase dependency violations):
#''')
            for task in enforcement_result.blocked_tasks:
                template_parts.append(f'#   - {task["description"]}')
                
        # Warnings section
        if enforcement_result.warnings:
            template_parts.append('''
# ⚠️  WARNINGS:
#''')
            for warning in enforcement_result.warnings:
                template_parts.append(f'#   • {warning}')
                
        # Footer
        template_parts.append(f'''
# 
# Phase Dependency Enforcement: ✅ ACTIVE
# Issue #223 Fix: ✅ IMPLEMENTED  
# Sequential Phase Validation: ✅ ENFORCED
# Resource Waste Prevention: ✅ ENABLED
# Execution Decision: {enforcement_result.decision_type.value.replace('_', ' ').title()}
''')
        
        return '\n'.join(template_parts)
        
    def _determine_enforcement_decision(
        self,
        phase_validation: PhaseValidationResult,
        github_issues: List[Dict[str, Any]],
        proposed_launches: List[Dict[str, Any]]
    ) -> Tuple[OrchestrationDecisionType, str]:
        """Determine the enforcement decision based on phase validation"""
        
        if not phase_validation.is_valid:
            # Check severity of violations
            critical_violations = [
                v for v in phase_validation.violations
                if v.severity == "critical"
            ]
            
            if critical_violations:
                return (
                    OrchestrationDecisionType.BLOCK_CRITICAL_VIOLATIONS,
                    f"Blocking execution due to {len(critical_violations)} critical phase dependency violations. "
                    f"Sequential phase completion must be enforced."
                )
            else:
                return (
                    OrchestrationDecisionType.BLOCK_PHASE_VIOLATIONS,
                    f"Blocking execution due to {len(phase_validation.violations)} phase dependency violations. "
                    f"Prerequisite phases must complete before proceeding."
                )
        
        # Check if we should redirect to prerequisites
        if self._should_redirect_to_prerequisites(phase_validation, github_issues):
            return (
                OrchestrationDecisionType.REDIRECT_TO_PREREQUISITES,
                "Redirecting to prerequisite phases to ensure proper sequential completion."
            )
            
        # Check for warnings that might require conditional approval
        if phase_validation.warnings:
            return (
                OrchestrationDecisionType.ALLOW_WITH_WARNINGS,
                f"Allowing execution with {len(phase_validation.warnings)} warnings. Monitor for phase completion."
            )
            
        # Default: allow parallel execution
        return (
            OrchestrationDecisionType.ALLOW_PARALLEL_EXECUTION,
            "Phase dependencies validated. Parallel execution approved."
        )
        
    def _categorize_tasks_by_phase_validation(
        self,
        proposed_launches: List[Dict[str, Any]],
        phase_validation: PhaseValidationResult,
        github_issues: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Categorize tasks into allowed, blocked, and prerequisite lists"""
        
        allowed_tasks = []
        blocked_tasks = []
        prerequisite_tasks = []
        
        # If validation passed, all tasks are allowed
        if phase_validation.is_valid:
            return proposed_launches, [], []
            
        # Extract issue numbers mentioned in violations
        violating_issue_numbers = set()
        for violation in phase_validation.violations:
            violating_issue_numbers.update(violation.issue_numbers)
            
        for task in proposed_launches:
            # Extract issue numbers from task description and prompt
            task_text = f"{task.get('description', '')} {task.get('prompt', '')}"
            task_issue_numbers = self._extract_issue_numbers(task_text)
            
            # Check if this task targets any violating issues
            if any(issue_num in violating_issue_numbers for issue_num in task_issue_numbers):
                blocked_tasks.append(task)
                
                # Generate prerequisite tasks for blocked issues
                prerequisite_task = self._generate_prerequisite_task(
                    task, task_issue_numbers, phase_validation.violations
                )
                if prerequisite_task:
                    prerequisite_tasks.append(prerequisite_task)
            else:
                allowed_tasks.append(task)
                
        return allowed_tasks, blocked_tasks, prerequisite_tasks
        
    def _generate_prerequisite_task(
        self,
        blocked_task: Dict[str, Any],
        issue_numbers: List[int],
        violations: List[PhaseDependencyViolation]
    ) -> Optional[Dict[str, Any]]:
        """Generate prerequisite task for blocked task"""
        
        # Find relevant violations for this task's issues
        relevant_violations = [
            v for v in violations
            if any(issue in v.issue_numbers for issue in issue_numbers)
        ]
        
        if not relevant_violations:
            return None
            
        violation = relevant_violations[0]  # Use first violation for simplicity
        
        # Determine what type of prerequisite agent is needed
        missing_phases = violation.missing_prerequisite_phases
        
        if PhaseType.ANALYSIS in missing_phases or PhaseType.RESEARCH in missing_phases:
            agent_type = "RIF-Analyst"
            task_description = f"Analysis and research for issues {issue_numbers}"
            task_prompt = f"You are RIF-Analyst. Analyze and research requirements for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete analysis phase before implementation. Follow all instructions in claude/agents/rif-analyst.md."
        elif PhaseType.PLANNING in missing_phases:
            agent_type = "RIF-Planner" 
            task_description = f"Planning for issues {issue_numbers}"
            task_prompt = f"You are RIF-Planner. Create execution plan for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete planning phase before implementation. Follow all instructions in claude/agents/rif-planner.md."
        elif PhaseType.ARCHITECTURE in missing_phases:
            agent_type = "RIF-Architect"
            task_description = f"Architecture design for issues {issue_numbers}"
            task_prompt = f"You are RIF-Architect. Design architecture for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete architecture phase before implementation. Follow all instructions in claude/agents/rif-architect.md."
        else:
            # Default to analyst for any missing prerequisite
            agent_type = "RIF-Analyst"
            task_description = f"Prerequisite analysis for issues {issue_numbers}"
            task_prompt = f"You are RIF-Analyst. Complete prerequisite analysis for issues {', '.join([f'#{i}' for i in issue_numbers])}. Follow all instructions in claude/agents/rif-analyst.md."
            
        return {
            "description": f"{agent_type}: {task_description}",
            "prompt": task_prompt,
            "subagent_type": "general-purpose"
        }
        
    def _should_redirect_to_prerequisites(
        self,
        phase_validation: PhaseValidationResult,
        github_issues: List[Dict[str, Any]]
    ) -> bool:
        """Determine if we should redirect to prerequisite phases instead of blocking"""
        
        # If validation passed, no need to redirect
        if phase_validation.is_valid:
            return False
            
        # Check if there are clear prerequisite phases that can be launched
        has_clear_prerequisites = False
        
        for violation in phase_validation.violations:
            if violation.missing_prerequisite_phases:
                # Check if the prerequisite phases are actionable (not already in progress)
                for issue_num in violation.issue_numbers:
                    issue = next((i for i in github_issues if i.get('number') == issue_num), None)
                    if issue:
                        state = self._extract_issue_state(issue)
                        if state in ["state:new", "state:analyzing"]:
                            has_clear_prerequisites = True
                            break
                            
        return has_clear_prerequisites
        
    def _generate_enforcement_guidance(
        self,
        phase_validation: PhaseValidationResult,
        blocked_tasks: List[Dict[str, Any]],
        prerequisite_tasks: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate violations, warnings, and remediation actions"""
        
        violations = []
        warnings = []
        remediation_actions = []
        
        # Extract violations from phase validation
        for violation in phase_validation.violations:
            violations.append(violation.description)
            remediation_actions.extend(violation.remediation_actions)
            
        # Add warnings from phase validation
        warnings.extend(phase_validation.warnings)
        
        # Add task-specific guidance
        if blocked_tasks:
            violations.append(f"{len(blocked_tasks)} agent launches blocked due to phase dependency violations")
            remediation_actions.append(f"Complete prerequisite phases for blocked issues before retrying")
            
        if prerequisite_tasks:
            remediation_actions.append(f"Execute {len(prerequisite_tasks)} prerequisite tasks to resolve phase dependencies")
            
        # Add general remediation guidance
        if violations:
            remediation_actions.extend([
                "Review issue states and labels to confirm phase completion",
                "Ensure sequential phase progression: Research → Planning → Architecture → Implementation → Validation", 
                "Use phase-specific agents (RIF-Analyst, RIF-Planner, RIF-Architect) for prerequisite work",
                "Update issue labels and comments to reflect phase completion evidence"
            ])
            
        return violations, warnings, list(set(remediation_actions))  # Remove duplicates
        
    def _calculate_enforcement_confidence(
        self,
        phase_validation: PhaseValidationResult,
        allowed_tasks: List[Dict[str, Any]],
        blocked_tasks: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in enforcement decision"""
        
        # Start with phase validation confidence
        confidence = phase_validation.confidence_score
        
        # Adjust based on task categorization clarity
        total_tasks = len(allowed_tasks) + len(blocked_tasks)
        if total_tasks > 0:
            clear_categorization_ratio = abs(len(allowed_tasks) - len(blocked_tasks)) / total_tasks
            confidence += clear_categorization_ratio * 0.1
            
        # Reduce confidence for complex scenarios
        if len(phase_validation.violations) > 3:
            confidence -= 0.1
            
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))
        
    def _generate_github_notifications(
        self,
        violations: List[str],
        github_issues: List[Dict[str, Any]],
        blocked_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate GitHub notifications for phase dependency violations"""
        
        notifications = []
        
        if not violations:
            return notifications
            
        # Extract issue numbers from blocked tasks
        affected_issue_numbers = set()
        for task in blocked_tasks:
            task_text = f"{task.get('description', '')} {task.get('prompt', '')}"
            issue_numbers = self._extract_issue_numbers(task_text)
            affected_issue_numbers.update(issue_numbers)
            
        # Generate notification for each affected issue
        for issue_num in affected_issue_numbers:
            notifications.append({
                "action_type": "phase_dependency_violation",
                "issue_number": issue_num,
                "timestamp": datetime.utcnow().isoformat(),
                "violations": violations,
                "message": f"Phase dependency violations detected for issue #{issue_num}. Agent launching blocked until prerequisites complete.",
                "suggested_labels": ["phase-dependency-blocked", "prerequisite-required"],
                "priority": "high"
            })
            
        return notifications
        
    def _extract_issue_numbers(self, text: str) -> List[int]:
        """Extract issue numbers from text"""
        matches = re.findall(r"#(\d+)", text)
        return [int(match) for match in matches]
        
    def _extract_issue_state(self, issue: Dict[str, Any]) -> str:
        """Extract state from issue labels"""
        labels = [label.get("name", "") for label in issue.get("labels", [])]
        state_labels = [label for label in labels if label.startswith("state:")]
        return state_labels[0] if state_labels else "state:unknown"
        
    def _store_enforcement_result(self, result: OrchestrationEnforcementResult):
        """Store enforcement result in knowledge base"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"orchestration_phase_enforcement_{timestamp}.json"
            filepath = self.enforcement_log_path / filename
            
            # Convert to JSON-serializable format
            result_data = {
                "decision_type": result.decision_type.value,
                "is_execution_allowed": result.is_execution_allowed,
                "phase_validation_result": {
                    "is_valid": result.phase_validation_result.is_valid,
                    "violations": [
                        {
                            "violation_type": v.violation_type,
                            "issue_numbers": v.issue_numbers,
                            "attempted_phase": v.attempted_phase.value,
                            "missing_prerequisite_phases": [p.value for p in v.missing_prerequisite_phases],
                            "severity": v.severity,
                            "description": v.description,
                            "remediation_actions": v.remediation_actions
                        }
                        for v in result.phase_validation_result.violations
                    ],
                    "warnings": result.phase_validation_result.warnings,
                    "confidence_score": result.phase_validation_result.confidence_score
                },
                "allowed_tasks": result.allowed_tasks,
                "blocked_tasks": result.blocked_tasks,
                "prerequisite_tasks": result.prerequisite_tasks,
                "violations": result.violations,
                "warnings": result.warnings,
                "remediation_actions": result.remediation_actions,
                "confidence_score": result.confidence_score,
                "execution_rationale": result.execution_rationale,
                "github_notifications": result.github_notifications,
                "enforcement_timestamp": result.enforcement_timestamp,
                "issue_reference": 223
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
                
            self.enforcement_history.append(result)
            
            print(f"   📝 Enforcement result stored: {filename}")
            
        except Exception as e:
            print(f"Warning: Could not store enforcement result: {e}")


# Main integration functions for CLAUDE.md orchestration template

def enforce_orchestration_phase_dependencies(
    github_issues: List[Dict[str, Any]], 
    proposed_agent_launches: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> OrchestrationEnforcementResult:
    """
    MAIN INTEGRATION FUNCTION for CLAUDE.md orchestration template.
    
    This function MUST be called in orchestration workflows to enforce phase dependencies.
    It replaces the basic dependency analysis with comprehensive phase validation.
    
    Usage in CLAUDE.md template:
    ```python
    # MANDATORY: Enforce phase dependencies before any agent launching
    from claude.commands.orchestration_phase_dependency_enforcer import enforce_orchestration_phase_dependencies
    
    enforcement_result = enforce_orchestration_phase_dependencies(github_issues, proposed_agent_launches)
    
    if not enforcement_result.is_execution_allowed:
        print("❌ PHASE DEPENDENCY VIOLATIONS - EXECUTION BLOCKED")
        for violation in enforcement_result.violations:
            print(f"  - {violation}")
        # Execute prerequisite tasks instead
        for task in enforcement_result.prerequisite_tasks:
            Task(description=task["description"], prompt=task["prompt"], subagent_type=task["subagent_type"])
    else:
        # Execute allowed tasks
        for task in enforcement_result.allowed_tasks:
            Task(description=task["description"], prompt=task["prompt"], subagent_type=task["subagent_type"])
    ```
    """
    enforcer = OrchestrationPhaseDependencyEnforcer()
    return enforcer.enforce_phase_dependencies(github_issues, proposed_agent_launches, context)


def generate_phase_validated_orchestration_template(
    github_issues: List[Dict[str, Any]], 
    proposed_agent_launches: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate complete orchestration template with phase dependency enforcement.
    
    This provides a ready-to-use orchestration template that includes all the
    phase dependency validation and enforcement logic.
    """
    enforcer = OrchestrationPhaseDependencyEnforcer()
    enforcement_result = enforcer.enforce_phase_dependencies(github_issues, proposed_agent_launches, context)
    return enforcer.generate_orchestration_template(enforcement_result)


# Testing function
def test_phase_dependency_enforcement():
    """Test the phase dependency enforcement system"""
    
    # Mock GitHub issues with phase dependency scenarios
    test_issues = [
        {
            "number": 1,
            "title": "Research user authentication patterns",
            "labels": [{"name": "state:new"}],
            "body": "Research authentication approaches"
        },
        {
            "number": 2, 
            "title": "Implement user authentication system",
            "labels": [{"name": "state:implementing"}],
            "body": "Implement user authentication"
        }
    ]
    
    # Mock agent launches that should be blocked due to missing research
    test_launches = [
        {
            "description": "RIF-Implementer: User authentication implementation",
            "prompt": "You are RIF-Implementer. Implement user authentication for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    print("🧪 Testing Phase Dependency Enforcement...")
    
    # Test enforcement
    enforcement_result = enforce_orchestration_phase_dependencies(test_issues, test_launches)
    
    print(f"   Decision: {enforcement_result.decision_type.value}")
    print(f"   Execution allowed: {enforcement_result.is_execution_allowed}")
    print(f"   Violations: {len(enforcement_result.violations)}")
    print(f"   Blocked tasks: {len(enforcement_result.blocked_tasks)}")
    print(f"   Prerequisite tasks: {len(enforcement_result.prerequisite_tasks)}")
    print(f"   Confidence: {enforcement_result.confidence_score:.2f}")
    
    # Test template generation
    template = generate_phase_validated_orchestration_template(test_issues, test_launches)
    print(f"   Generated template length: {len(template)} characters")
    
    return enforcement_result


if __name__ == "__main__":
    test_phase_dependency_enforcement()