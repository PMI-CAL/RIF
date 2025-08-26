"""
Phase Dependency Validation System

Implements comprehensive phase dependency validation to prevent orchestration errors
where implementation/validation agents are launched before research/architecture phases complete.

Issue #223: RIF Orchestration Error: Not Following Phase Dependencies
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class PhaseType(Enum):
    """Standard RIF workflow phases"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    LEARNING = "learning"


@dataclass
class PhaseCompletionCriteria:
    """Criteria that must be met for a phase to be considered complete"""
    phase: PhaseType
    required_states: List[str]
    required_labels: List[str]
    required_comments: List[str]  # Patterns that must exist in comments
    exclude_states: List[str]  # States that indicate phase is not complete
    validation_function: Optional[str] = None  # Custom validation function name


@dataclass
class IssuePhaseStatus:
    """Current phase status for an issue"""
    issue_number: int
    current_phase: PhaseType
    completed_phases: List[PhaseType]
    phase_completion_evidence: Dict[str, List[str]]
    blocking_dependencies: List[int]
    last_updated: str


@dataclass
class PhaseDependencyViolation:
    """Represents a phase dependency violation"""
    violation_type: str
    issue_numbers: List[int]
    attempted_phase: PhaseType
    missing_prerequisite_phases: List[PhaseType]
    severity: str  # "critical", "high", "medium", "low"
    description: str
    remediation_actions: List[str]


@dataclass
class PhaseValidationResult:
    """Result of phase dependency validation"""
    is_valid: bool
    violations: List[PhaseDependencyViolation]
    warnings: List[str]
    allowed_phases: List[PhaseType]
    blocked_phases: List[PhaseType]
    confidence_score: float
    validation_timestamp: str


class PhaseDependencyValidator:
    """
    Validates phase dependencies to ensure proper sequential workflow execution.
    
    Enforces rules:
    1. Research must complete before Architecture
    2. Architecture must complete before Implementation  
    3. Implementation must complete before Validation
    4. No implementation work on issues with incomplete research
    5. Foundation phases must complete before dependent phases
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.phase_criteria = self._initialize_phase_criteria()
        self.validation_history = []
        
    def _initialize_phase_criteria(self) -> Dict[PhaseType, PhaseCompletionCriteria]:
        """Initialize phase completion criteria"""
        return {
            PhaseType.RESEARCH: PhaseCompletionCriteria(
                phase=PhaseType.RESEARCH,
                required_states=["state:analyzed", "state:planning", "state:architecting"],
                required_labels=["research:complete", "analysis:complete"],
                required_comments=["research findings", "analysis complete", "requirements identified"],
                exclude_states=["state:new", "state:analyzing"]
            ),
            PhaseType.ANALYSIS: PhaseCompletionCriteria(
                phase=PhaseType.ANALYSIS,
                required_states=["state:planning", "state:architecting", "state:implementing"],
                required_labels=["analysis:complete", "complexity:assessed"],
                required_comments=["analysis complete", "patterns identified", "requirements clear"],
                exclude_states=["state:new", "state:analyzing"]
            ),
            PhaseType.PLANNING: PhaseCompletionCriteria(
                phase=PhaseType.PLANNING,
                required_states=["state:architecting", "state:implementing"],
                required_labels=["planning:complete", "plan:approved"],
                required_comments=["planning complete", "execution plan ready", "approach confirmed"],
                exclude_states=["state:new", "state:analyzing", "state:planning"]
            ),
            PhaseType.ARCHITECTURE: PhaseCompletionCriteria(
                phase=PhaseType.ARCHITECTURE,
                required_states=["state:implementing"],
                required_labels=["architecture:complete", "design:approved"],
                required_comments=["architecture complete", "design finalized", "technical approach confirmed"],
                exclude_states=["state:new", "state:analyzing", "state:planning", "state:architecting"]
            ),
            PhaseType.IMPLEMENTATION: PhaseCompletionCriteria(
                phase=PhaseType.IMPLEMENTATION,
                required_states=["state:validating", "state:documenting"],
                required_labels=["implementation:complete", "code:complete"],
                required_comments=["implementation complete", "code written", "tests added"],
                exclude_states=["state:new", "state:analyzing", "state:planning", "state:architecting", "state:implementing"]
            ),
            PhaseType.VALIDATION: PhaseCompletionCriteria(
                phase=PhaseType.VALIDATION,
                required_states=["state:documenting", "state:learning", "state:complete"],
                required_labels=["validation:complete", "tests:passing"],
                required_comments=["validation complete", "tests pass", "quality gates met"],
                exclude_states=["state:validating"]
            )
        }
        
    def validate_phase_dependencies(
        self, 
        github_issues: List[Dict[str, Any]],
        proposed_agent_launches: List[Dict[str, Any]]
    ) -> PhaseValidationResult:
        """
        Main validation function - checks if proposed agent launches violate phase dependencies
        """
        violations = []
        warnings = []
        allowed_phases = []
        blocked_phases = []
        
        # Step 1: Analyze current phase status for all issues
        issue_phase_statuses = {}
        for issue in github_issues:
            issue_number = issue.get("number")
            status = self._analyze_issue_phase_status(issue)
            issue_phase_statuses[issue_number] = status
            
        # Step 2: Check each proposed agent launch against dependencies
        for agent_launch in proposed_agent_launches:
            violation = self._validate_agent_launch_dependencies(
                agent_launch, 
                issue_phase_statuses,
                github_issues
            )
            if violation:
                violations.append(violation)
                blocked_phases.extend(violation.missing_prerequisite_phases)
            else:
                # Extract phase from agent launch
                attempted_phase = self._extract_phase_from_agent_launch(agent_launch)
                if attempted_phase and attempted_phase not in allowed_phases:
                    allowed_phases.append(attempted_phase)
                    
        # Step 3: Identify foundation dependency violations
        foundation_violations = self._check_foundation_dependencies(
            issue_phase_statuses, 
            proposed_agent_launches,
            github_issues
        )
        violations.extend(foundation_violations)
        
        # Step 4: Generate warnings for risky phase transitions
        phase_warnings = self._generate_phase_warnings(issue_phase_statuses, proposed_agent_launches)
        warnings.extend(phase_warnings)
        
        # Step 5: Calculate confidence score
        confidence_score = self._calculate_validation_confidence(violations, warnings, issue_phase_statuses)
        
        # Create result
        result = PhaseValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            allowed_phases=list(set(allowed_phases)),
            blocked_phases=list(set(blocked_phases)),
            confidence_score=confidence_score,
            validation_timestamp=datetime.utcnow().isoformat()
        )
        
        # Store validation result
        self._store_validation_result(result)
        
        return result
        
    def validate_phase_completion(self, issue: Dict[str, Any], phase: PhaseType) -> bool:
        """Check if a specific phase is complete for an issue"""
        criteria = self.phase_criteria.get(phase)
        if not criteria:
            return False
            
        issue_labels = [label.get("name", "") for label in issue.get("labels", [])]
        issue_state = self._extract_issue_state(issue)
        issue_comments = self._extract_issue_comments(issue)
        
        # Check required states
        if issue_state not in criteria.required_states:
            return False
            
        # Check excluded states
        if issue_state in criteria.exclude_states:
            return False
            
        # Check required labels
        has_required_labels = any(
            any(req_label in label for req_label in criteria.required_labels)
            for label in issue_labels
        )
        if criteria.required_labels and not has_required_labels:
            return False
            
        # Check required comments
        if criteria.required_comments:
            comment_text = " ".join(issue_comments).lower()
            has_required_comments = any(
                req_comment.lower() in comment_text
                for req_comment in criteria.required_comments
            )
            if not has_required_comments:
                return False
                
        return True
        
    def enforce_sequential_phases(
        self,
        issues: List[Dict[str, Any]], 
        target_phase: PhaseType
    ) -> Tuple[List[int], List[str]]:
        """
        Enforce sequential phase completion - return issues ready for target phase
        and reasons why others are blocked
        """
        ready_issues = []
        blocking_reasons = []
        
        for issue in issues:
            issue_number = issue.get("number")
            
            # Check if prerequisites are met
            prerequisites_met, blocking_reason = self._check_phase_prerequisites(issue, target_phase)
            
            if prerequisites_met:
                ready_issues.append(issue_number)
            else:
                blocking_reasons.append(f"Issue #{issue_number}: {blocking_reason}")
                
        return ready_issues, blocking_reasons
        
    def _analyze_issue_phase_status(self, issue: Dict[str, Any]) -> IssuePhaseStatus:
        """Analyze current phase status for an issue"""
        issue_number = issue.get("number")
        current_phase = self._determine_current_phase(issue)
        completed_phases = self._determine_completed_phases(issue)
        evidence = self._collect_phase_evidence(issue)
        dependencies = self._extract_blocking_dependencies(issue)
        
        return IssuePhaseStatus(
            issue_number=issue_number,
            current_phase=current_phase,
            completed_phases=completed_phases,
            phase_completion_evidence=evidence,
            blocking_dependencies=dependencies,
            last_updated=datetime.utcnow().isoformat()
        )
        
    def _determine_current_phase(self, issue: Dict[str, Any]) -> PhaseType:
        """Determine current phase from issue state"""
        issue_state = self._extract_issue_state(issue)
        issue_labels = [label.get("name", "") for label in issue.get("labels", [])]
        
        # Map states to phases
        if issue_state in ["state:new", "state:analyzing"]:
            if any("research" in label.lower() for label in issue_labels):
                return PhaseType.RESEARCH
            return PhaseType.ANALYSIS
        elif issue_state == "state:planning":
            return PhaseType.PLANNING
        elif issue_state == "state:architecting":
            return PhaseType.ARCHITECTURE
        elif issue_state == "state:implementing":
            return PhaseType.IMPLEMENTATION
        elif issue_state == "state:validating":
            return PhaseType.VALIDATION
        else:
            return PhaseType.ANALYSIS  # Default
            
    def _determine_completed_phases(self, issue: Dict[str, Any]) -> List[PhaseType]:
        """Determine which phases have been completed for an issue"""
        completed = []
        
        for phase_type in PhaseType:
            if self.validate_phase_completion(issue, phase_type):
                completed.append(phase_type)
                
        return completed
        
    def _collect_phase_evidence(self, issue: Dict[str, Any]) -> Dict[str, List[str]]:
        """Collect evidence of phase completion"""
        evidence = {}
        issue_comments = self._extract_issue_comments(issue)
        issue_labels = [label.get("name", "") for label in issue.get("labels", [])]
        
        for phase_type in PhaseType:
            phase_evidence = []
            
            # Check labels for evidence
            relevant_labels = [
                label for label in issue_labels 
                if phase_type.value in label.lower()
            ]
            phase_evidence.extend(relevant_labels)
            
            # Check comments for evidence  
            relevant_comments = [
                comment for comment in issue_comments
                if phase_type.value in comment.lower()
            ]
            phase_evidence.extend(relevant_comments[:3])  # Limit to first 3
            
            if phase_evidence:
                evidence[phase_type.value] = phase_evidence
                
        return evidence
        
    def _validate_agent_launch_dependencies(
        self,
        agent_launch: Dict[str, Any],
        issue_statuses: Dict[int, IssuePhaseStatus],
        github_issues: List[Dict[str, Any]]
    ) -> Optional[PhaseDependencyViolation]:
        """Validate a single agent launch against phase dependencies"""
        
        # Extract target phase and issues from agent launch
        target_phase = self._extract_phase_from_agent_launch(agent_launch)
        target_issues = self._extract_issues_from_agent_launch(agent_launch)
        
        if not target_phase or not target_issues:
            return None
            
        # Check each target issue
        violating_issues = []
        missing_phases = []
        
        for issue_num in target_issues:
            if issue_num not in issue_statuses:
                continue
                
            issue_status = issue_statuses[issue_num]
            prerequisites_met, missing = self._check_phase_prerequisites_from_status(
                issue_status, target_phase
            )
            
            if not prerequisites_met:
                violating_issues.append(issue_num)
                missing_phases.extend(missing)
                
        if violating_issues:
            severity = self._determine_violation_severity(target_phase, missing_phases)
            
            return PhaseDependencyViolation(
                violation_type="sequential_phase_violation",
                issue_numbers=violating_issues,
                attempted_phase=target_phase,
                missing_prerequisite_phases=list(set(missing_phases)),
                severity=severity,
                description=f"Attempted {target_phase.value} phase before completing prerequisite phases: {[p.value for p in set(missing_phases)]}",
                remediation_actions=self._generate_remediation_actions(target_phase, missing_phases)
            )
            
        return None
        
    def _check_foundation_dependencies(
        self,
        issue_statuses: Dict[int, IssuePhaseStatus],
        proposed_launches: List[Dict[str, Any]],
        github_issues: List[Dict[str, Any]]
    ) -> List[PhaseDependencyViolation]:
        """Check for foundation dependency violations"""
        violations = []
        
        # Identify foundation issues
        foundation_issues = []
        dependent_issues = []
        
        for issue in github_issues:
            title = issue.get("title", "").lower()
            labels = [label.get("name", "") for label in issue.get("labels", [])]
            
            if any(keyword in title for keyword in ["foundation", "core", "api", "framework", "infrastructure"]):
                foundation_issues.append(issue.get("number"))
            else:
                dependent_issues.append(issue.get("number"))
                
        # Check if dependent work is being launched while foundation is incomplete
        for launch in proposed_launches:
            launch_issues = self._extract_issues_from_agent_launch(launch)
            launch_phase = self._extract_phase_from_agent_launch(launch)
            
            # Skip if not implementation/validation phase
            if launch_phase not in [PhaseType.IMPLEMENTATION, PhaseType.VALIDATION]:
                continue
                
            dependent_launch_issues = [i for i in launch_issues if i in dependent_issues]
            
            if dependent_launch_issues:
                # Check if foundation issues are complete
                incomplete_foundation = []
                for foundation_issue in foundation_issues:
                    if foundation_issue in issue_statuses:
                        status = issue_statuses[foundation_issue]
                        if PhaseType.IMPLEMENTATION not in status.completed_phases:
                            incomplete_foundation.append(foundation_issue)
                            
                if incomplete_foundation:
                    violations.append(PhaseDependencyViolation(
                        violation_type="foundation_dependency_violation",
                        issue_numbers=dependent_launch_issues,
                        attempted_phase=launch_phase,
                        missing_prerequisite_phases=[PhaseType.IMPLEMENTATION],
                        severity="high",
                        description=f"Attempting {launch_phase.value} on dependent issues while foundation issues {incomplete_foundation} are incomplete",
                        remediation_actions=[
                            f"Complete implementation of foundation issues: {incomplete_foundation}",
                            "Wait for foundation APIs/frameworks to be ready",
                            "Launch foundation agents first, dependent agents second"
                        ]
                    ))
                    
        return violations
        
    def _generate_phase_warnings(
        self,
        issue_statuses: Dict[int, IssuePhaseStatus],
        proposed_launches: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate warnings for potentially risky phase transitions"""
        warnings = []
        
        for launch in proposed_launches:
            launch_issues = self._extract_issues_from_agent_launch(launch)
            launch_phase = self._extract_phase_from_agent_launch(launch)
            
            for issue_num in launch_issues:
                if issue_num in issue_statuses:
                    status = issue_statuses[issue_num]
                    
                    # Warn if jumping phases
                    if launch_phase == PhaseType.IMPLEMENTATION and PhaseType.ARCHITECTURE not in status.completed_phases:
                        warnings.append(f"Issue #{issue_num}: Jumping to implementation without architecture phase")
                        
                    # Warn if evidence is weak
                    phase_evidence = status.phase_completion_evidence
                    if len(phase_evidence) < 2:
                        warnings.append(f"Issue #{issue_num}: Weak evidence of phase completion")
                        
        return warnings
        
    def _check_phase_prerequisites(self, issue: Dict[str, Any], target_phase: PhaseType) -> Tuple[bool, str]:
        """Check if prerequisites are met for target phase"""
        
        # Define prerequisite chains
        prerequisites = {
            PhaseType.ANALYSIS: [],
            PhaseType.RESEARCH: [],
            PhaseType.PLANNING: [PhaseType.ANALYSIS],
            PhaseType.ARCHITECTURE: [PhaseType.ANALYSIS, PhaseType.PLANNING], 
            PhaseType.IMPLEMENTATION: [PhaseType.ANALYSIS, PhaseType.PLANNING, PhaseType.ARCHITECTURE],
            PhaseType.VALIDATION: [PhaseType.IMPLEMENTATION],
            PhaseType.DOCUMENTATION: [PhaseType.VALIDATION],
            PhaseType.LEARNING: [PhaseType.VALIDATION]
        }
        
        required_phases = prerequisites.get(target_phase, [])
        
        for required_phase in required_phases:
            if not self.validate_phase_completion(issue, required_phase):
                return False, f"Missing prerequisite phase: {required_phase.value}"
                
        return True, ""
        
    def _check_phase_prerequisites_from_status(
        self, 
        issue_status: IssuePhaseStatus, 
        target_phase: PhaseType
    ) -> Tuple[bool, List[PhaseType]]:
        """Check prerequisites from issue status object"""
        
        prerequisites = {
            PhaseType.ANALYSIS: [],
            PhaseType.RESEARCH: [],
            PhaseType.PLANNING: [PhaseType.ANALYSIS],
            PhaseType.ARCHITECTURE: [PhaseType.ANALYSIS, PhaseType.PLANNING], 
            PhaseType.IMPLEMENTATION: [PhaseType.ANALYSIS, PhaseType.PLANNING, PhaseType.ARCHITECTURE],
            PhaseType.VALIDATION: [PhaseType.IMPLEMENTATION],
            PhaseType.DOCUMENTATION: [PhaseType.VALIDATION],
            PhaseType.LEARNING: [PhaseType.VALIDATION]
        }
        
        required_phases = prerequisites.get(target_phase, [])
        missing_phases = []
        
        for required_phase in required_phases:
            if required_phase not in issue_status.completed_phases:
                missing_phases.append(required_phase)
                
        return len(missing_phases) == 0, missing_phases
        
    def _extract_phase_from_agent_launch(self, agent_launch: Dict[str, Any]) -> Optional[PhaseType]:
        """Extract target phase from agent launch description/prompt"""
        
        description = agent_launch.get("description", "").lower()
        prompt = agent_launch.get("prompt", "").lower()
        text = f"{description} {prompt}"
        
        # Map agent types to phases
        if "rif-analyst" in text:
            return PhaseType.ANALYSIS
        elif "rif-planner" in text:
            return PhaseType.PLANNING
        elif "rif-architect" in text:
            return PhaseType.ARCHITECTURE
        elif "rif-implementer" in text:
            return PhaseType.IMPLEMENTATION
        elif "rif-validator" in text:
            return PhaseType.VALIDATION
        elif "rif-documenter" in text:
            return PhaseType.DOCUMENTATION
        elif "rif-learner" in text:
            return PhaseType.LEARNING
            
        # Check for phase keywords
        if any(word in text for word in ["research", "investigate", "explore"]):
            return PhaseType.RESEARCH
        elif any(word in text for word in ["implement", "code", "build"]):
            return PhaseType.IMPLEMENTATION
        elif any(word in text for word in ["validate", "test", "verify"]):
            return PhaseType.VALIDATION
            
        return None
        
    def _extract_issues_from_agent_launch(self, agent_launch: Dict[str, Any]) -> List[int]:
        """Extract issue numbers from agent launch description/prompt"""
        
        description = agent_launch.get("description", "")
        prompt = agent_launch.get("prompt", "")
        text = f"{description} {prompt}"
        
        # Find issue numbers
        issue_numbers = []
        matches = re.findall(r"#(\d+)", text)
        for match in matches:
            try:
                issue_numbers.append(int(match))
            except ValueError:
                continue
                
        return issue_numbers
        
    def _extract_issue_state(self, issue: Dict[str, Any]) -> str:
        """Extract state label from issue"""
        labels = [label.get("name", "") for label in issue.get("labels", [])]
        state_labels = [label for label in labels if label.startswith("state:")]
        return state_labels[0] if state_labels else "state:unknown"
        
    def _extract_issue_comments(self, issue: Dict[str, Any]) -> List[str]:
        """Extract comments from issue (simplified - would use GitHub API in real implementation)"""
        # In real implementation, would fetch comments via GitHub API
        # For now, return empty list or mock data
        return []
        
    def _extract_blocking_dependencies(self, issue: Dict[str, Any]) -> List[int]:
        """Extract blocking dependencies from issue"""
        body = issue.get("body", "")
        dependencies = []
        
        # Look for dependency patterns
        patterns = [
            r"depends on #(\d+)",
            r"blocked by #(\d+)",
            r"requires #(\d+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, body.lower())
            dependencies.extend([int(match) for match in matches])
            
        return dependencies
        
    def _determine_violation_severity(
        self, 
        target_phase: PhaseType, 
        missing_phases: List[PhaseType]
    ) -> str:
        """Determine severity of phase violation"""
        
        # Critical violations
        if PhaseType.IMPLEMENTATION in missing_phases and target_phase == PhaseType.VALIDATION:
            return "critical"
        if PhaseType.ARCHITECTURE in missing_phases and target_phase == PhaseType.IMPLEMENTATION:
            return "critical"
            
        # High violations  
        if len(missing_phases) > 2:
            return "high"
        if PhaseType.ANALYSIS in missing_phases:
            return "high"
            
        # Medium violations
        if len(missing_phases) > 1:
            return "medium"
            
        return "low"
        
    def _generate_remediation_actions(
        self, 
        target_phase: PhaseType, 
        missing_phases: List[PhaseType]
    ) -> List[str]:
        """Generate remediation actions for violations"""
        
        actions = []
        
        for missing_phase in missing_phases:
            if missing_phase == PhaseType.ANALYSIS:
                actions.append("Launch RIF-Analyst to complete requirements analysis")
            elif missing_phase == PhaseType.PLANNING:
                actions.append("Launch RIF-Planner to create execution plan")
            elif missing_phase == PhaseType.ARCHITECTURE:
                actions.append("Launch RIF-Architect to complete technical design")
            elif missing_phase == PhaseType.IMPLEMENTATION:
                actions.append("Launch RIF-Implementer to complete code implementation")
                
        actions.append(f"Wait for prerequisite phases to complete before launching {target_phase.value} agents")
        actions.append("Use sequential agent launching instead of parallel launching")
        
        return actions
        
    def _calculate_validation_confidence(
        self,
        violations: List[PhaseDependencyViolation],
        warnings: List[str],
        issue_statuses: Dict[int, IssuePhaseStatus]
    ) -> float:
        """Calculate confidence in validation result"""
        
        base_confidence = 1.0
        
        # Reduce confidence for violations
        critical_violations = sum(1 for v in violations if v.severity == "critical")
        high_violations = sum(1 for v in violations if v.severity == "high")
        medium_violations = sum(1 for v in violations if v.severity == "medium")
        
        confidence_reduction = (
            critical_violations * 0.4 +
            high_violations * 0.2 + 
            medium_violations * 0.1 +
            len(warnings) * 0.05
        )
        
        # Increase confidence for good phase evidence
        total_issues = len(issue_statuses)
        if total_issues > 0:
            issues_with_evidence = sum(
                1 for status in issue_statuses.values()
                if len(status.phase_completion_evidence) > 1
            )
            evidence_bonus = (issues_with_evidence / total_issues) * 0.1
            base_confidence += evidence_bonus
            
        final_confidence = max(0.0, min(1.0, base_confidence - confidence_reduction))
        return final_confidence
        
    def _store_validation_result(self, result: PhaseValidationResult):
        """Store validation result in knowledge base"""
        try:
            validation_dir = Path(self.knowledge_base_path) / "validation"
            validation_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_dependency_validation_{timestamp}.json"
            filepath = validation_dir / filename
            
            # Convert result to JSON-serializable format
            result_dict = {
                "is_valid": result.is_valid,
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
                    for v in result.violations
                ],
                "warnings": result.warnings,
                "allowed_phases": [p.value for p in result.allowed_phases],
                "blocked_phases": [p.value for p in result.blocked_phases],
                "confidence_score": result.confidence_score,
                "validation_timestamp": result.validation_timestamp,
                "issue_reference": 223
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
            self.validation_history.append(result)
            
        except Exception as e:
            print(f"Warning: Could not store phase validation result: {e}")


# Convenience functions for integration
def validate_phase_completion(issue: Dict[str, Any], phase: str) -> bool:
    """Validate if a phase is complete for an issue"""
    validator = PhaseDependencyValidator()
    try:
        phase_enum = PhaseType(phase.lower())
        return validator.validate_phase_completion(issue, phase_enum)
    except ValueError:
        return False


def enforce_sequential_phases(issues: List[Dict[str, Any]], target_phase: str) -> Tuple[List[int], List[str]]:
    """Enforce sequential phase completion"""
    validator = PhaseDependencyValidator()
    try:
        phase_enum = PhaseType(target_phase.lower())
        return validator.enforce_sequential_phases(issues, phase_enum)
    except ValueError:
        return [], [f"Invalid phase: {target_phase}"]


if __name__ == "__main__":
    # Test the validator
    test_issues = [
        {
            "number": 1,
            "title": "Research user authentication patterns",
            "labels": [{"name": "state:new"}],
            "body": "Research authentication approaches"
        },
        {
            "number": 2,
            "title": "Implement user authentication",
            "labels": [{"name": "state:implementing"}],
            "body": "Implement user authentication system"
        }
    ]
    
    test_launches = [
        {
            "description": "RIF-Implementer: User authentication implementation",
            "prompt": "You are RIF-Implementer. Implement user authentication for issue #2.",
            "subagent_type": "general-purpose"
        }
    ]
    
    validator = PhaseDependencyValidator()
    result = validator.validate_phase_dependencies(test_issues, test_launches)
    
    print(f"Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Violations: {len(result.violations)}")
    for violation in result.violations:
        print(f"  - {violation.description}")
    print(f"Confidence: {result.confidence_score:.2f}")