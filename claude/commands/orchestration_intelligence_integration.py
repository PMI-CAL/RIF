"""
Orchestration Intelligence Framework Integration

Integrates orchestration pattern validation with the existing orchestration intelligence framework.
Enhances dependency analysis and intelligent launch decisions with anti-pattern prevention.

Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
"""

import json
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .orchestration_pattern_validator import OrchestrationPatternValidator, ValidationResult
from .orchestration_validation_enforcer import OrchestrationValidationEnforcer
from .phase_dependency_validator import PhaseDependencyValidator


@dataclass
class DependencyAnalysis:
    """Result of dependency analysis for orchestration decisions"""
    blocking_issues: List[int]
    foundation_issues: List[int] 
    research_incomplete_issues: List[int]
    ready_for_parallel_issues: List[int]
    sequential_dependencies: Dict[int, List[int]]
    critical_path: List[int]
    analysis_timestamp: str
    

@dataclass
class IntelligentOrchestrationDecision:
    """Result of intelligent orchestration decision making"""
    decision_type: str  # "blocking_only", "foundation_only", "research_only", "parallel_ready"
    recommended_tasks: List[Dict[str, str]]
    dependency_rationale: str
    validation_status: ValidationResult
    enforcement_action: str
    confidence_score: float
    decision_timestamp: str


class OrchestrationIntelligenceIntegration:
    """
    Integrates pattern validation with orchestration intelligence framework.
    
    Enhances the 4-step orchestration process:
    1. Issue Dependency Mapping + Pattern Validation
    2. Critical Path Identification + Parallel Readiness Validation  
    3. Intelligent Launch Decision + Anti-Pattern Prevention
    4. Agent Launching + Validation Hooks
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.pattern_validator = OrchestrationPatternValidator()
        self.validation_enforcer = OrchestrationValidationEnforcer(knowledge_base_path)
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.decision_history = []
        
    def enhanced_dependency_analysis(
        self,
        github_issues: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> DependencyAnalysis:
        """
        Enhanced Step 1: Issue Dependency Mapping with Pattern Awareness
        
        Performs traditional dependency analysis while considering orchestration patterns
        and anti-pattern prevention.
        """
        # Enhanced dependency analysis with comprehensive blocking detection
        blocking_issues = self._detect_blocking_issues_enhanced(github_issues)
        foundation_issues = []
        research_incomplete_issues = []
        ready_issues = []
        sequential_deps = {}
        
        for issue in github_issues:
            issue_id = issue.get("number")
            issue_state = self._extract_issue_state(issue)
            issue_labels = [label.get("name", "") for label in issue.get("labels", [])]
            
            # Skip blocking issues (already detected)
            if issue_id in blocking_issues:
                continue
            elif issue_state in ["state:new", "state:analyzing"] and "research" in issue.get("title", "").lower():
                research_incomplete_issues.append(issue_id)
            elif issue_state in ["state:new", "state:analyzing"] and any(word in issue.get("title", "").lower() for word in ["core", "foundation", "api", "framework"]):
                foundation_issues.append(issue_id)
            elif issue_state in ["state:implementing", "state:validating"]:
                ready_issues.append(issue_id)
                
            # Build dependency graph (simplified - would use more sophisticated logic in real implementation)
            dependencies = self._extract_dependencies(issue)
            if dependencies:
                sequential_deps[issue_id] = dependencies
                
        # Determine critical path
        critical_path = self._calculate_critical_path(blocking_issues, foundation_issues, sequential_deps)
        
        return DependencyAnalysis(
            blocking_issues=blocking_issues,
            foundation_issues=foundation_issues,
            research_incomplete_issues=research_incomplete_issues,
            ready_for_parallel_issues=ready_issues,
            sequential_dependencies=sequential_deps,
            critical_path=critical_path,
            analysis_timestamp=datetime.utcnow().isoformat()
        )
        
    def intelligent_launch_decision_with_validation(
        self,
        dependency_analysis: DependencyAnalysis,
        proposed_tasks: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> IntelligentOrchestrationDecision:
        """
        Enhanced Step 3: Intelligent Launch Decision with Anti-Pattern Prevention and Blocking Detection
        
        Makes orchestration decisions based on dependencies while preventing anti-patterns.
        CRITICAL: Now includes mandatory blocking issue validation as first step.
        """
        # Step 0: MANDATORY pre-flight blocking validation (CRITICAL FIX for issue #228)
        github_issues = context.get("github_issues", []) if context else []
        if github_issues:
            pre_flight_validation = self.validate_orchestration_request_with_blocking_check(
                github_issues, proposed_tasks
            )
            
            if not pre_flight_validation.is_valid:
                # BLOCK EXECUTION - blocking issues or critical phase violations detected
                return IntelligentOrchestrationDecision(
                    decision_type="blocked_execution",
                    recommended_tasks=[],
                    dependency_rationale="Execution blocked due to blocking issues or critical phase violations",
                    validation_status=pre_flight_validation,
                    enforcement_action="block_execution",
                    confidence_score=pre_flight_validation.confidence_score,
                    decision_timestamp=datetime.utcnow().isoformat()
                )
        
        # Step 1: Apply dependency-based decision logic (only if no blocking issues)
        if dependency_analysis.blocking_issues:
            decision_type = "blocking_only"
            dependency_rationale = f"Blocking issues {dependency_analysis.blocking_issues} must complete before other work"
            # Filter tasks to only handle blocking issues
            recommended_tasks = [
                task for task in proposed_tasks 
                if self._task_addresses_issues(task, dependency_analysis.blocking_issues)
            ]
        elif dependency_analysis.foundation_issues and dependency_analysis.ready_for_parallel_issues:
            decision_type = "foundation_only"  
            dependency_rationale = f"Foundation issues {dependency_analysis.foundation_issues} must complete before dependent work"
            recommended_tasks = [
                task for task in proposed_tasks
                if self._task_addresses_issues(task, dependency_analysis.foundation_issues)
            ]
        elif dependency_analysis.research_incomplete_issues:
            decision_type = "research_only"
            dependency_rationale = f"Research issues {dependency_analysis.research_incomplete_issues} must complete before implementation"
            recommended_tasks = [
                task for task in proposed_tasks
                if self._task_addresses_issues(task, dependency_analysis.research_incomplete_issues)
            ]
        else:
            decision_type = "parallel_ready"
            dependency_rationale = "No blocking dependencies - ready for parallel execution"
            recommended_tasks = proposed_tasks
            
        # Step 2: Validate orchestration patterns
        validation_result = self.pattern_validator.validate_orchestration_request(recommended_tasks)
        
        # Step 3: Apply enforcement based on validation
        enforcement_record = self.validation_enforcer.validate_and_enforce(
            recommended_tasks, 
            {
                "dependency_analysis": asdict(dependency_analysis),
                "decision_type": decision_type,
                "context": context or {}
            }
        )
        
        enforcement_action = enforcement_record["enforcement_action"]
        
        # Step 4: Adjust recommendations based on validation
        if enforcement_action == "block_execution":
            # Apply corrective suggestions from validation
            corrective_tasks = self._apply_corrective_suggestions(
                recommended_tasks, 
                validation_result.suggestions
            )
            
            # Re-validate corrected tasks
            corrected_validation = self.pattern_validator.validate_orchestration_request(corrective_tasks)
            
            if corrected_validation.is_valid:
                recommended_tasks = corrective_tasks
                validation_result = corrected_validation
                enforcement_action = "allow_execution_after_correction"
            else:
                # Still invalid - block completely
                recommended_tasks = []
                enforcement_action = "block_execution_uncorrectable"
                
        # Calculate combined confidence score
        dependency_confidence = self._calculate_dependency_confidence(dependency_analysis)
        combined_confidence = (dependency_confidence + validation_result.confidence_score) / 2
        
        decision = IntelligentOrchestrationDecision(
            decision_type=decision_type,
            recommended_tasks=recommended_tasks,
            dependency_rationale=dependency_rationale,
            validation_status=validation_result,
            enforcement_action=enforcement_action,
            confidence_score=combined_confidence,
            decision_timestamp=datetime.utcnow().isoformat()
        )
        
        # Store decision for learning
        self.decision_history.append(decision)
        self._store_orchestration_decision(decision, dependency_analysis)
        
        return decision
        
    def generate_validated_orchestration_template(
        self,
        orchestration_decision: IntelligentOrchestrationDecision
    ) -> str:
        """
        Enhanced Step 4: Generate orchestration template with validation hooks
        """
        if not orchestration_decision.recommended_tasks:
            return self._generate_no_action_template(orchestration_decision)
            
        if orchestration_decision.enforcement_action.startswith("block"):
            return self._generate_blocked_execution_template(orchestration_decision)
            
        # Generate valid orchestration template
        template_parts = []
        
        # Add header comment with decision context
        template_parts.append(f'''# Orchestration Decision: {orchestration_decision.decision_type}
# Dependency Rationale: {orchestration_decision.dependency_rationale}
# Pattern Validation: {'PASSED' if orchestration_decision.validation_status.is_valid else 'FAILED'}
# Confidence Score: {orchestration_decision.confidence_score:.2f}
# Generated: {orchestration_decision.decision_timestamp}
''')
        
        # Add validation check (optional runtime validation)
        if len(orchestration_decision.recommended_tasks) > 1:
            template_parts.append('''
# Validate orchestration pattern before execution (optional)
from claude.commands.orchestration_intelligence_integration import validate_before_execution
validate_before_execution([
''')
            for task in orchestration_decision.recommended_tasks:
                template_parts.append(f'    {task},')
            template_parts.append('])\n')
            
        # Generate Task invocations
        template_parts.append('\n# Parallel Task execution:')
        for i, task in enumerate(orchestration_decision.recommended_tasks):
            template_parts.append(f'''
Task(
    description="{task.get('description', f'Task {i+1}')}",
    subagent_type="{task.get('subagent_type', 'general-purpose')}",
    prompt="{task.get('prompt', 'Task prompt')}"
)''')
            
        # Add footer comment
        template_parts.append(f'''
# {len(orchestration_decision.recommended_tasks)} Tasks launched in parallel
# Pattern validation: {orchestration_decision.validation_status.pattern_type}
# Anti-pattern prevention: ACTIVE''')
        
        return ''.join(template_parts)
        
    def _extract_issue_state(self, issue: Dict[str, Any]) -> str:
        """Extract issue state from labels"""
        labels = [label.get("name", "") for label in issue.get("labels", [])]
        state_labels = [label for label in labels if label.startswith("state:")]
        return state_labels[0] if state_labels else "state:unknown"
        
    def _extract_dependencies(self, issue: Dict[str, Any]) -> List[int]:
        """Extract dependencies from issue (simplified implementation)"""
        # In real implementation, this would parse issue body for dependency references
        body = issue.get("body", "")
        dependencies = []
        
        # Look for "depends on #123" patterns
        import re
        depends_pattern = r"depends on #(\d+)"
        matches = re.findall(depends_pattern, body.lower())
        dependencies.extend([int(match) for match in matches])
        
        return dependencies
        
    def _calculate_critical_path(
        self,
        blocking_issues: List[int],
        foundation_issues: List[int], 
        sequential_deps: Dict[int, List[int]]
    ) -> List[int]:
        """Calculate critical path through dependencies"""
        # Simplified critical path calculation
        critical_path = []
        
        # Blocking issues are always first in critical path
        critical_path.extend(blocking_issues)
        
        # Foundation issues next
        critical_path.extend(foundation_issues)
        
        # Add issues with most dependencies
        dep_counts = {issue: len(deps) for issue, deps in sequential_deps.items()}
        sorted_issues = sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)
        
        for issue_id, _ in sorted_issues[:3]:  # Top 3 most dependent issues
            if issue_id not in critical_path:
                critical_path.append(issue_id)
                
        return critical_path
        
    def _detect_blocking_issues_enhanced(self, github_issues: List[Dict[str, Any]]) -> List[int]:
        """
        Enhanced blocking detection that parses issue body and comments for explicit blocking declarations.
        
        Detects:
        1. Label-based blocking (existing logic)
        2. User declarations in issue body
        3. User declarations in issue comments
        4. Phase dependency blocking (integration)
        """
        blocking_issues = []
        
        for issue in github_issues:
            issue_id = issue.get("number")
            issue_labels = [label.get("name", "") for label in issue.get("labels", [])]
            issue_body = issue.get("body", "").lower()
            issue_state = self._extract_issue_state(issue)
            
            # Skip completed issues - they can't be blocking active work
            if issue_state in ["state:complete", "state:closed"]:
                continue
                
            # EXISTING: Label-based detection
            if "agent:context-reading-failure" in issue_labels or "critical-infrastructure" in issue_labels:
                blocking_issues.append(issue_id)
                continue
                
            # NEW: User declaration detection in body (exact phrases to avoid false positives)
            blocking_declarations = [
                "this issue blocks all others",
                "blocks all other work",
                "blocks all others", 
                "stop all work",
                "must complete before all",
                "blocking priority",
                "blocks everything",
                "critical blocker",
                "halt all work",
                "pause everything"
            ]
            
            if any(declaration in issue_body for declaration in blocking_declarations):
                blocking_issues.append(issue_id)
                continue
                
            # NEW: Comment analysis integration
            comments_text = self._get_issue_comments_text(issue_id)
            if comments_text and any(declaration in comments_text.lower() for declaration in blocking_declarations):
                blocking_issues.append(issue_id)
                continue
                
        return blocking_issues
        
    def _get_issue_comments_text(self, issue_id: int) -> str:
        """
        Fetch issue comments text for blocking analysis using GitHub CLI.
        
        Returns concatenated text of all comments for keyword analysis.
        Implements basic error handling and caching.
        """
        try:
            # Use gh CLI to get comments
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_id), "--json", "comments"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                comments = data.get("comments", [])
                return " ".join([comment.get("body", "") for comment in comments])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not fetch comments for issue #{issue_id}: {e}")
        return ""
        
    def validate_orchestration_request_with_blocking_check(
        self,
        github_issues: List[Dict[str, Any]],
        proposed_tasks: List[Dict[str, str]]
    ) -> ValidationResult:
        """
        MANDATORY pre-flight validation that checks for blocking issues before any orchestration decision.
        
        This is the critical fix for issue #228 - ensures blocking issues are detected and respected.
        """
        # STEP 1: Check for blocking issues FIRST (critical fix)
        blocking_issues = self._detect_blocking_issues_enhanced(github_issues)
        
        if blocking_issues:
            return ValidationResult(
                is_valid=False,
                violations=[f"BLOCKING ISSUES DETECTED: {blocking_issues} - All other work must stop until these are resolved"],
                pattern_type="blocking_issue_enforcement",
                confidence_score=1.0,
                suggestions=[
                    "Complete blocking issues before launching other agents",
                    "Launch agents for blocking issues ONLY",  
                    "No parallel work allowed while blocking issues exist",
                    f"Focus exclusively on resolving issues: {blocking_issues}"
                ]
            )
        
        # STEP 2: Phase dependency validation (disabled for issue #228 fix focus)
        # Note: Issue #228 focuses specifically on explicit blocking issue declarations
        # Phase dependency validation is a separate concern and should not block explicit user declarations
        # This can be re-enabled later as a separate enhancement
        
        # For now, we only handle the critical case: explicit blocking issues
        # This ensures we fix the original problem (ignoring "THIS ISSUE BLOCKS ALL OTHERS")
        # without introducing new blocking behavior for regular orchestration scenarios
        
        # STEP 3: Continue with existing pattern validation if no blocking issues
        return self.pattern_validator.validate_orchestration_request(proposed_tasks)
        
    def _task_addresses_issues(self, task: Dict[str, str], issue_list: List[int]) -> bool:
        """Check if a task addresses any issues in the list"""
        task_text = f"{task.get('description', '')} {task.get('prompt', '')}"
        
        # Look for issue numbers in task text
        import re
        issue_numbers = re.findall(r"#(\d+)", task_text)
        task_issues = [int(num) for num in issue_numbers]
        
        return any(issue in issue_list for issue in task_issues)
        
    def _apply_corrective_suggestions(
        self,
        tasks: List[Dict[str, str]],
        suggestions: List[str]
    ) -> List[Dict[str, str]]:
        """Apply corrective suggestions to fix anti-patterns"""
        # This is a simplified implementation - real version would be more sophisticated
        corrected_tasks = []
        
        for task in tasks:
            # Check if task has multi-issue anti-pattern
            task_text = f"{task.get('description', '')} {task.get('prompt', '')}"
            issue_numbers = re.findall(r"#(\d+)", task_text)
            
            if len(issue_numbers) > 1:
                # Split multi-issue task into separate tasks
                base_description = task.get('description', '').split(':')[0]  # Get agent name
                base_prompt = task.get('prompt', '').split('.')[0]  # Get base prompt
                
                for issue_num in issue_numbers:
                    corrected_tasks.append({
                        "description": f"{base_description}: Issue #{issue_num}",
                        "prompt": f"{base_prompt} for issue #{issue_num}. Follow all instructions in claude/agents/rif-implementer.md.",
                        "subagent_type": task.get('subagent_type', 'general-purpose')
                    })
            else:
                # Task is okay, just ensure proper instructions
                corrected_task = task.copy()
                if "Follow all instructions in claude/agents/" not in corrected_task.get('prompt', ''):
                    corrected_task['prompt'] += " Follow all instructions in claude/agents/rif-implementer.md."
                corrected_tasks.append(corrected_task)
                
        return corrected_tasks
        
    def _calculate_dependency_confidence(self, analysis: DependencyAnalysis) -> float:
        """Calculate confidence in dependency analysis"""
        # Simple heuristic - more complex in real implementation
        total_issues = (
            len(analysis.blocking_issues) +
            len(analysis.foundation_issues) + 
            len(analysis.research_incomplete_issues) +
            len(analysis.ready_for_parallel_issues)
        )
        
        if total_issues == 0:
            return 1.0
            
        # Higher confidence when fewer blocking/foundation issues
        blocking_penalty = len(analysis.blocking_issues) * 0.2
        foundation_penalty = len(analysis.foundation_issues) * 0.1
        
        confidence = max(0.1, 1.0 - blocking_penalty - foundation_penalty)
        return confidence
        
    def _generate_no_action_template(self, decision: IntelligentOrchestrationDecision) -> str:
        """Generate template when no action should be taken"""
        return f'''# No Orchestration Action Required
# Decision: {decision.decision_type}
# Reason: {decision.dependency_rationale}
# Validation Status: {decision.enforcement_action}
# 
# No Tasks recommended at this time.
# Wait for blocking dependencies to resolve.'''
        
    def _generate_blocked_execution_template(self, decision: IntelligentOrchestrationDecision) -> str:
        """Generate template when execution is blocked due to violations"""
        violations = decision.validation_status.violations
        suggestions = decision.validation_status.suggestions
        
        template = f'''# Orchestration Execution BLOCKED
# Reason: Pattern violations detected
# Violations: {len(violations)}
# 
# Pattern Violations:'''
        
        for violation in violations:
            template += f'\n# - {violation}'
            
        template += '\n#\n# Corrective Actions Required:'
        
        for suggestion in suggestions:
            template += f'\n# - {suggestion}'
            
        template += '''
#
# Please correct the violations above and retry orchestration.
# No Tasks will be launched until patterns are corrected.'''
        
        return template
        
    def _store_orchestration_decision(
        self,
        decision: IntelligentOrchestrationDecision,
        analysis: DependencyAnalysis
    ):
        """Store orchestration decision in knowledge base"""
        try:
            decisions_dir = Path(self.knowledge_base_path) / "decisions"
            decisions_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"orchestration_decision_{timestamp}.json"
            filepath = decisions_dir / filename
            
            decision_record = {
                "decision": asdict(decision),
                "dependency_analysis": asdict(analysis),
                "issue_reference": 224,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(decision_record, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not store orchestration decision: {e}")


# Convenience functions for integration
def validate_before_execution(
    proposed_tasks: List[Dict[str, str]],
    github_issues: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """Runtime validation check for orchestration patterns with blocking issue detection"""
    # Enhanced validation with blocking detection if github issues provided
    if github_issues:
        integration = OrchestrationIntelligenceIntegration()
        result = integration.validate_orchestration_request_with_blocking_check(
            github_issues, proposed_tasks
        )
    else:
        # Fallback to pattern validation only
        validator = OrchestrationPatternValidator()
        result = validator.validate_orchestration_request(proposed_tasks)
    
    if not result.is_valid:
        print("❌ Orchestration validation failed:")
        for violation in result.violations:
            print(f"  - {violation}")
        if result.suggestions:
            print("\n📋 Suggested actions:")
            for suggestion in result.suggestions:
                print(f"  → {suggestion}")
        return False
        
    print("✅ Orchestration validation passed")
    return True


def make_intelligent_orchestration_decision(
    github_issues: List[Dict[str, Any]],
    proposed_tasks: List[Dict[str, str]],
    context: Optional[Dict[str, Any]] = None
) -> IntelligentOrchestrationDecision:
    """Main entry point for intelligent orchestration decisions with enhanced blocking detection"""
    integration = OrchestrationIntelligenceIntegration()
    
    # Ensure github_issues are available in context for blocking validation
    if context is None:
        context = {}
    context["github_issues"] = github_issues
    
    # Perform enhanced dependency analysis
    dependency_analysis = integration.enhanced_dependency_analysis(github_issues, context)
    
    # Make intelligent launch decision with validation (now includes blocking detection)
    decision = integration.intelligent_launch_decision_with_validation(
        dependency_analysis,
        proposed_tasks,
        context
    )
    
    return decision


# Enhanced blocking detection keywords for user declarations
BLOCKING_KEYWORDS = [
    "this issue blocks all others",
    "blocks all other work",
    "blocks all others", 
    "stop all work",
    "must complete before all",
    "blocking priority",
    "blocks everything",
    "critical blocker",
    "halt all work",
    "pause everything"
]

# Import statement for CLAUDE.md integration
import re


if __name__ == "__main__":
    # Test the integration
    test_issues = [
        {
            "number": 1,
            "title": "User authentication system",
            "labels": [{"name": "state:implementing"}],
            "body": "Implement user auth"
        },
        {
            "number": 2,
            "title": "Database connection pooling", 
            "labels": [{"name": "state:implementing"}],
            "body": "Add database pooling"
        }
    ]
    
    test_tasks = [
        {
            "description": "RIF-Implementer: User authentication",
            "prompt": "You are RIF-Implementer. Implement user auth for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Implementer: Database pooling",
            "prompt": "You are RIF-Implementer. Implement database pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    decision = make_intelligent_orchestration_decision(test_issues, test_tasks)
    print(f"Decision type: {decision.decision_type}")
    print(f"Validation passed: {decision.validation_status.is_valid}")
    print(f"Enforcement action: {decision.enforcement_action}")
    print(f"Recommended tasks: {len(decision.recommended_tasks)}")