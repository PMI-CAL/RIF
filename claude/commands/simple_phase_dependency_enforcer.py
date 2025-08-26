#!/usr/bin/env python3
"""
Simple Phase Dependency Enforcer

CRITICAL FIX FOR ISSUE #223: RIF Orchestration Error: Not Following Phase Dependencies

This is a streamlined implementation that fixes the core orchestration phase dependency
enforcement issue without relying on complex validator frameworks that may not exist.

The issue: RIF orchestration was ignoring phase dependencies and launching implementation
agents before research/architecture phases were complete.

The fix: This enforcer validates GitHub issue states and blocks agent launches that
violate sequential phase dependencies.

Integration: Replaces complex orchestration intelligence with simple, effective validation.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum


class PhaseType(Enum):
    """RIF workflow phases"""
    RESEARCH = "research"
    ANALYSIS = "analysis"  
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    LEARNING = "learning"


class ViolationType(Enum):
    """Types of phase dependency violations"""
    IMPLEMENTATION_WITHOUT_ANALYSIS = "implementation_without_analysis"
    VALIDATION_WITHOUT_IMPLEMENTATION = "validation_without_implementation"
    ARCHITECTURE_WITHOUT_PLANNING = "architecture_without_planning"
    SEQUENTIAL_PHASE_SKIP = "sequential_phase_skip"


@dataclass
class PhaseViolation:
    """Represents a phase dependency violation"""
    violation_type: ViolationType
    issue_numbers: List[int]
    attempted_phase: PhaseType
    missing_prerequisite: PhaseType
    severity: str
    description: str
    remediation_action: str


@dataclass  
class OrchestrationEnforcementResult:
    """Result of phase dependency enforcement"""
    is_execution_allowed: bool
    violations: List[PhaseViolation]
    allowed_tasks: List[Dict[str, Any]]
    blocked_tasks: List[Dict[str, Any]] 
    prerequisite_tasks: List[Dict[str, Any]]
    execution_rationale: str
    confidence_score: float
    enforcement_timestamp: str


class SimplePhaseEnforcer:
    """
    Simple but effective phase dependency enforcement system.
    
    This fixes issue #223 by implementing the core logic needed to enforce
    phase dependencies in RIF orchestration without complex dependencies.
    
    Key Rules Enforced:
    1. Research/Analysis must complete before Architecture  
    2. Architecture must complete before Implementation
    3. Implementation must complete before Validation
    4. No jumping phases (e.g., state:new -> state:implementing)
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.knowledge_path = Path(self.knowledge_base_path)
        self.enforcement_log_path = self.knowledge_path / "enforcement_logs"
        self.enforcement_log_path.mkdir(exist_ok=True)
        
        # Define phase progression rules
        self.phase_prerequisites = {
            PhaseType.ARCHITECTURE: [PhaseType.ANALYSIS, PhaseType.PLANNING],
            PhaseType.IMPLEMENTATION: [PhaseType.ANALYSIS, PhaseType.PLANNING, PhaseType.ARCHITECTURE],  
            PhaseType.VALIDATION: [PhaseType.IMPLEMENTATION],
            PhaseType.LEARNING: [PhaseType.VALIDATION]
        }
        
        # Map GitHub states to phases
        self.state_to_phase = {
            "state:new": PhaseType.ANALYSIS,
            "state:analyzing": PhaseType.ANALYSIS,
            "state:planning": PhaseType.PLANNING,
            "state:architecting": PhaseType.ARCHITECTURE,
            "state:implementing": PhaseType.IMPLEMENTATION,
            "state:validating": PhaseType.VALIDATION,
            "state:learning": PhaseType.LEARNING,
            "state:complete": PhaseType.LEARNING
        }
        
    def enforce_phase_dependencies(
        self,
        github_issues: List[Dict[str, Any]],
        proposed_agent_launches: List[Dict[str, Any]]
    ) -> OrchestrationEnforcementResult:
        """
        MAIN ENFORCEMENT FUNCTION - Critical fix for issue #223
        
        Validates that proposed agent launches don't violate phase dependencies
        and blocks execution if violations are detected.
        """
        
        print("🔍 SIMPLE PHASE DEPENDENCY ENFORCEMENT (Issue #223 Fix)")
        
        # Step 1: Analyze current issue states
        issue_phases = self._analyze_issue_phases(github_issues)
        
        # Step 2: Validate proposed agent launches
        violations = self._validate_agent_launches(proposed_agent_launches, issue_phases)
        
        # Step 3: Categorize tasks based on violations
        allowed_tasks, blocked_tasks, prerequisite_tasks = self._categorize_tasks(
            proposed_agent_launches, violations
        )
        
        # Step 4: Determine execution decision
        is_execution_allowed = len(violations) == 0
        
        # Step 5: Generate rationale and confidence
        execution_rationale, confidence_score = self._generate_enforcement_decision(
            violations, allowed_tasks, blocked_tasks
        )
        
        # Step 6: Create result
        result = OrchestrationEnforcementResult(
            is_execution_allowed=is_execution_allowed,
            violations=violations,
            allowed_tasks=allowed_tasks,
            blocked_tasks=blocked_tasks,
            prerequisite_tasks=prerequisite_tasks,
            execution_rationale=execution_rationale,
            confidence_score=confidence_score,
            enforcement_timestamp=datetime.utcnow().isoformat()
        )
        
        # Step 7: Log result
        self._log_enforcement_result(result)
        
        print(f"   ✅ Enforcement complete: {'ALLOWED' if is_execution_allowed else 'BLOCKED'}")
        print(f"   📊 Violations: {len(violations)}")
        print(f"   📈 Confidence: {confidence_score:.2f}")
        
        return result
        
    def _analyze_issue_phases(self, github_issues: List[Dict[str, Any]]) -> Dict[int, PhaseType]:
        """Analyze current phase for each GitHub issue"""
        
        issue_phases = {}
        
        for issue in github_issues:
            issue_number = issue.get("number")
            if not issue_number:
                continue
                
            # Extract issue state from labels
            issue_state = self._extract_issue_state(issue)
            
            # Map state to phase
            current_phase = self.state_to_phase.get(issue_state, PhaseType.ANALYSIS)
            issue_phases[issue_number] = current_phase
            
        return issue_phases
        
    def _validate_agent_launches(
        self,
        proposed_launches: List[Dict[str, Any]],
        issue_phases: Dict[int, PhaseType]
    ) -> List[PhaseViolation]:
        """Validate proposed agent launches against phase dependencies"""
        
        violations = []
        
        for launch in proposed_launches:
            # Determine what phase this agent launch targets
            target_phase = self._determine_agent_target_phase(launch)
            
            if not target_phase:
                continue  # Skip if can't determine phase
                
            # Extract issue numbers from launch
            target_issues = self._extract_issue_numbers_from_launch(launch)
            
            if not target_issues:
                continue  # Skip if no issues identified
                
            # Check each target issue
            for issue_number in target_issues:
                if issue_number not in issue_phases:
                    continue
                    
                current_phase = issue_phases[issue_number]
                
                # Check if prerequisites are met
                violation = self._check_phase_prerequisites(
                    issue_number, current_phase, target_phase, launch
                )
                
                if violation:
                    violations.append(violation)
                    
        return violations
        
    def _determine_agent_target_phase(self, launch: Dict[str, Any]) -> Optional[PhaseType]:
        """Determine what phase an agent launch is targeting"""
        
        description = launch.get("description", "").lower()
        prompt = launch.get("prompt", "").lower()
        
        # Map agent types to phases
        if "rif-analyst" in description or "analyze" in description:
            return PhaseType.ANALYSIS
        elif "rif-planner" in description or "plan" in description:
            return PhaseType.PLANNING  
        elif "rif-architect" in description or "architect" in description:
            return PhaseType.ARCHITECTURE
        elif "rif-implementer" in description or "implement" in description:
            return PhaseType.IMPLEMENTATION
        elif "rif-validator" in description or "validat" in description:
            return PhaseType.VALIDATION
        elif "rif-learner" in description or "learn" in description:
            return PhaseType.LEARNING
            
        # Check prompt for phase indicators
        if any(word in prompt for word in ["implement", "code", "build"]):
            return PhaseType.IMPLEMENTATION
        elif any(word in prompt for word in ["validate", "test", "verify"]):  
            return PhaseType.VALIDATION
        elif any(word in prompt for word in ["research", "analyze", "investigate"]):
            return PhaseType.ANALYSIS
            
        return None
        
    def _check_phase_prerequisites(
        self,
        issue_number: int,
        current_phase: PhaseType,
        target_phase: PhaseType,
        launch: Dict[str, Any]
    ) -> Optional[PhaseViolation]:
        """Check if phase prerequisites are met"""
        
        # Get required prerequisites for target phase
        required_prerequisites = self.phase_prerequisites.get(target_phase, [])
        
        if not required_prerequisites:
            return None  # No prerequisites required
            
        # Check if current phase satisfies prerequisites
        phase_order = [
            PhaseType.ANALYSIS,
            PhaseType.PLANNING, 
            PhaseType.ARCHITECTURE,
            PhaseType.IMPLEMENTATION,
            PhaseType.VALIDATION,
            PhaseType.LEARNING
        ]
        
        current_index = phase_order.index(current_phase) if current_phase in phase_order else 0
        
        for prerequisite in required_prerequisites:
            if prerequisite not in phase_order:
                continue
                
            prerequisite_index = phase_order.index(prerequisite)
            
            # If current phase hasn't reached prerequisite, we have a violation
            if current_index < prerequisite_index:
                return PhaseViolation(
                    violation_type=self._determine_violation_type(current_phase, target_phase),
                    issue_numbers=[issue_number],
                    attempted_phase=target_phase,
                    missing_prerequisite=prerequisite,
                    severity="high" if target_phase == PhaseType.IMPLEMENTATION else "medium",
                    description=f"Issue #{issue_number}: Attempted {target_phase.value} phase while in {current_phase.value} phase, missing {prerequisite.value} prerequisite",
                    remediation_action=f"Complete {prerequisite.value} phase for issue #{issue_number} before launching {target_phase.value} agents"
                )
                
        return None
        
    def _determine_violation_type(self, current_phase: PhaseType, target_phase: PhaseType) -> ViolationType:
        """Determine the type of violation"""
        
        if target_phase == PhaseType.IMPLEMENTATION and current_phase in [PhaseType.ANALYSIS, PhaseType.PLANNING]:
            return ViolationType.IMPLEMENTATION_WITHOUT_ANALYSIS
        elif target_phase == PhaseType.VALIDATION and current_phase != PhaseType.IMPLEMENTATION:
            return ViolationType.VALIDATION_WITHOUT_IMPLEMENTATION
        elif target_phase == PhaseType.ARCHITECTURE and current_phase == PhaseType.ANALYSIS:
            return ViolationType.ARCHITECTURE_WITHOUT_PLANNING
        else:
            return ViolationType.SEQUENTIAL_PHASE_SKIP
            
    def _categorize_tasks(
        self,
        proposed_launches: List[Dict[str, Any]],
        violations: List[PhaseViolation]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Categorize tasks into allowed, blocked, and prerequisite"""
        
        allowed_tasks = []
        blocked_tasks = []
        prerequisite_tasks = []
        
        # Get all issue numbers with violations
        violating_issues = set()
        for violation in violations:
            violating_issues.update(violation.issue_numbers)
            
        for launch in proposed_launches:
            launch_issues = self._extract_issue_numbers_from_launch(launch)
            
            # Check if this launch targets any violating issues
            if any(issue in violating_issues for issue in launch_issues):
                blocked_tasks.append(launch)
                
                # Generate prerequisite task
                prerequisite_task = self._generate_prerequisite_task(launch, violations)
                if prerequisite_task:
                    prerequisite_tasks.append(prerequisite_task)
            else:
                allowed_tasks.append(launch)
                
        return allowed_tasks, blocked_tasks, prerequisite_tasks
        
    def _generate_prerequisite_task(
        self,
        blocked_launch: Dict[str, Any],
        violations: List[PhaseViolation]
    ) -> Optional[Dict[str, Any]]:
        """Generate prerequisite task for blocked launch"""
        
        launch_issues = self._extract_issue_numbers_from_launch(blocked_launch)
        
        # Find relevant violations
        relevant_violations = [
            v for v in violations
            if any(issue in v.issue_numbers for issue in launch_issues)
        ]
        
        if not relevant_violations:
            return None
            
        violation = relevant_violations[0]
        missing_prerequisite = violation.missing_prerequisite
        issue_numbers = violation.issue_numbers
        
        # Generate appropriate prerequisite task
        if missing_prerequisite == PhaseType.ANALYSIS:
            return {
                "description": f"RIF-Analyst: Analysis for issues {issue_numbers}",
                "prompt": f"You are RIF-Analyst. Analyze requirements and patterns for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete analysis phase before proceeding. Follow all instructions in claude/agents/rif-analyst.md.",
                "subagent_type": "general-purpose"
            }
        elif missing_prerequisite == PhaseType.PLANNING:
            return {
                "description": f"RIF-Planner: Planning for issues {issue_numbers}",
                "prompt": f"You are RIF-Planner. Create execution plan for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete planning phase before proceeding. Follow all instructions in claude/agents/rif-planner.md.",
                "subagent_type": "general-purpose"
            }
        elif missing_prerequisite == PhaseType.ARCHITECTURE:
            return {
                "description": f"RIF-Architect: Architecture for issues {issue_numbers}",
                "prompt": f"You are RIF-Architect. Design system architecture for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete architecture phase before proceeding. Follow all instructions in claude/agents/rif-architect.md.",
                "subagent_type": "general-purpose"
            }
        elif missing_prerequisite == PhaseType.IMPLEMENTATION:
            return {
                "description": f"RIF-Implementer: Implementation for issues {issue_numbers}",
                "prompt": f"You are RIF-Implementer. Implement solution for issues {', '.join([f'#{i}' for i in issue_numbers])}. Complete implementation phase before proceeding. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }
            
        return None
        
    def _generate_enforcement_decision(
        self,
        violations: List[PhaseViolation],
        allowed_tasks: List[Dict[str, Any]],
        blocked_tasks: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Generate execution rationale and confidence score"""
        
        if not violations:
            rationale = f"Phase dependencies validated. {len(allowed_tasks)} tasks approved for parallel execution."
            confidence = 1.0
        else:
            critical_violations = [v for v in violations if v.severity == "high"]
            if critical_violations:
                rationale = f"Execution BLOCKED due to {len(critical_violations)} critical phase dependency violations. Sequential phase completion required."
                confidence = 0.95
            else:
                rationale = f"Execution BLOCKED due to {len(violations)} phase dependency violations. Prerequisites must complete first."
                confidence = 0.85
                
        return rationale, confidence
        
    def _extract_issue_state(self, issue: Dict[str, Any]) -> str:
        """Extract state from issue labels"""
        
        labels = [label.get("name", "") for label in issue.get("labels", [])]
        state_labels = [label for label in labels if label.startswith("state:")]
        return state_labels[0] if state_labels else "state:new"
        
    def _extract_issue_numbers_from_launch(self, launch: Dict[str, Any]) -> List[int]:
        """Extract issue numbers from launch description and prompt"""
        
        text = f"{launch.get('description', '')} {launch.get('prompt', '')}"
        matches = re.findall(r"#(\d+)", text)
        return [int(match) for match in matches]
        
    def _log_enforcement_result(self, result: OrchestrationEnforcementResult):
        """Log enforcement result to knowledge base"""
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_file = self.enforcement_log_path / f"simple_phase_enforcement_{timestamp}.json"
            
            log_data = {
                "is_execution_allowed": result.is_execution_allowed,
                "violations": [
                    {
                        "violation_type": v.violation_type.value,
                        "issue_numbers": v.issue_numbers,
                        "attempted_phase": v.attempted_phase.value,
                        "missing_prerequisite": v.missing_prerequisite.value,
                        "severity": v.severity,
                        "description": v.description,
                        "remediation_action": v.remediation_action
                    }
                    for v in result.violations
                ],
                "allowed_tasks": result.allowed_tasks,
                "blocked_tasks": result.blocked_tasks,
                "prerequisite_tasks": result.prerequisite_tasks,
                "execution_rationale": result.execution_rationale,
                "confidence_score": result.confidence_score,
                "enforcement_timestamp": result.enforcement_timestamp,
                "issue_reference": 223
            }
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not log enforcement result: {e}")
            
    def generate_orchestration_template(self, result: OrchestrationEnforcementResult) -> str:
        """Generate orchestration template with enforcement results"""
        
        template_parts = []
        
        # Header
        template_parts.append(f'''# PHASE DEPENDENCY ENFORCEMENT (Issue #223 Fix)
# Execution Allowed: {'✅ YES' if result.is_execution_allowed else '❌ NO'}  
# Violations: {len(result.violations)}
# Confidence: {result.confidence_score:.2f}
# Generated: {result.enforcement_timestamp}
# 
# {result.execution_rationale}
''')
        
        # Violations section
        if result.violations:
            template_parts.append('''
# ❌ PHASE DEPENDENCY VIOLATIONS:''')
            for i, violation in enumerate(result.violations):
                template_parts.append(f'''#   {i+1}. {violation.description}
#      → {violation.remediation_action}''')
                
        # Tasks section
        if result.is_execution_allowed and result.allowed_tasks:
            template_parts.append('''
# ✅ VALIDATED TASKS - EXECUTING IN PARALLEL:
''')
            for task in result.allowed_tasks:
                template_parts.append(f'''
Task(
    description="{task['description']}",
    subagent_type="{task['subagent_type']}",
    prompt="{task['prompt']}"
)''')
                
        elif result.prerequisite_tasks:
            template_parts.append('''
# 🔄 PREREQUISITE TASKS - EXECUTE THESE FIRST:
''')
            for task in result.prerequisite_tasks:
                template_parts.append(f'''
Task(
    description="{task['description']}",
    subagent_type="{task['subagent_type']}",
    prompt="{task['prompt']}"
)''')
                
        # Footer
        template_parts.append('''
# 
# ✅ Phase Dependency Enforcement: ACTIVE
# ✅ Issue #223 Fix: IMPLEMENTED
# ✅ Sequential Phase Validation: ENFORCED
''')
        
        return '\n'.join(template_parts)


# Main integration functions for CLAUDE.md

def enforce_orchestration_phase_dependencies(
    github_issues: List[Dict[str, Any]],
    proposed_agent_launches: List[Dict[str, Any]]
) -> OrchestrationEnforcementResult:
    """
    MAIN FUNCTION for CLAUDE.md orchestration template.
    
    This is the critical fix for issue #223. Use this function to enforce
    phase dependencies before any agent launching.
    """
    enforcer = SimplePhaseEnforcer()
    return enforcer.enforce_phase_dependencies(github_issues, proposed_agent_launches)


def generate_phase_validated_orchestration_template(
    github_issues: List[Dict[str, Any]],
    proposed_agent_launches: List[Dict[str, Any]]
) -> str:
    """Generate complete orchestration template with phase validation"""
    enforcer = SimplePhaseEnforcer()
    result = enforcer.enforce_phase_dependencies(github_issues, proposed_agent_launches)
    return enforcer.generate_orchestration_template(result)


# Test function
def test_simple_enforcement():
    """Test the simple phase enforcement"""
    
    test_issues = [
        {
            "number": 1,
            "title": "Research authentication patterns", 
            "labels": [{"name": "state:new"}]  # Issue is in analysis phase
        },
        {
            "number": 2,
            "title": "Implement authentication system",
            "labels": [{"name": "state:new"}]   # Issue is in analysis phase but we want to implement
        }
    ]
    
    test_launches = [
        {
            "description": "RIF-Implementer: Authentication implementation",
            "prompt": "You are RIF-Implementer. Implement authentication for issue #1 and #2",
            "subagent_type": "general-purpose"
        }
    ]
    
    print("🧪 Testing Simple Phase Enforcement...")
    result = enforce_orchestration_phase_dependencies(test_issues, test_launches)
    
    print(f"   Execution allowed: {result.is_execution_allowed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    if result.violations:
        print("   Violation details:")
        for violation in result.violations:
            print(f"     - {violation.description}")
            
    template = generate_phase_validated_orchestration_template(test_issues, test_launches)
    print(f"   Template length: {len(template)} characters")
    
    return result


if __name__ == "__main__":
    test_simple_enforcement()