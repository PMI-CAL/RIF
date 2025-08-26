"""
Orchestration Pattern Validation Framework

Prevents RIF orchestration anti-patterns like "Multi-Issue Accelerator" attempts.
Validates proper parallel Task() launching patterns before execution.

Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of orchestration pattern validation"""
    is_valid: bool
    pattern_type: str
    violations: List[str]
    suggestions: List[str]
    confidence_score: float


@dataclass
class TaskDescription:
    """Parsed Task description for validation"""
    description: str
    prompt: str
    agent_type: str
    issue_numbers: List[int]
    agent_name: str
    is_multi_issue: bool
    is_generic_accelerator: bool


class OrchestrationPatternValidator:
    """
    Validates orchestration patterns to prevent anti-patterns.
    
    Primary focus: Prevent "Multi-Issue Accelerator" anti-pattern where
    single Task attempts to handle multiple issues instead of proper
    parallel Task launching.
    """
    
    # Anti-pattern detection patterns
    MULTI_ISSUE_PATTERNS = [
        r"Multi-Issue",
        r"Accelerator", 
        r"Batch",
        r"Combined",
        r"Multiple.*Issues",
        r"Parallel.*Issues",
        r"Handle.*Issues"
    ]
    
    ISSUE_NUMBER_PATTERN = r"#(\d+)"
    
    # Valid RIF agent names
    VALID_RIF_AGENTS = [
        "RIF-Analyst",
        "RIF-Planner", 
        "RIF-Architect",
        "RIF-Implementer",
        "RIF-Validator",
        "RIF-Learner",
        "RIF-PR-Manager"
    ]
    
    def __init__(self):
        self.violation_log = []
        
    def validate_orchestration_request(
        self, 
        task_descriptions: List[Dict[str, str]]
    ) -> ValidationResult:
        """
        Validate an orchestration request containing Task descriptions.
        
        Args:
            task_descriptions: List of Task description dicts with keys:
                - description: Task description string
                - prompt: Task prompt string  
                - subagent_type: Agent type
                
        Returns:
            ValidationResult with validation status and recommendations
        """
        violations = []
        suggestions = []
        parsed_tasks = []
        
        # Parse all task descriptions
        for task_desc in task_descriptions:
            parsed_task = self._parse_task_description(task_desc)
            parsed_tasks.append(parsed_task)
            
        # Validate against anti-patterns
        multi_issue_violations = self._detect_multi_issue_anti_pattern(parsed_tasks)
        violations.extend(multi_issue_violations)
        
        # Validate parallel execution patterns
        parallel_violations = self._validate_parallel_patterns(parsed_tasks)
        violations.extend(parallel_violations)
        
        # Validate agent specialization
        agent_violations = self._validate_agent_specialization(parsed_tasks)
        violations.extend(agent_violations)
        
        # Generate suggestions for violations
        if violations:
            suggestions = self._generate_corrective_suggestions(parsed_tasks, violations)
            
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(parsed_tasks, violations)
        
        # Determine pattern type
        pattern_type = self._determine_pattern_type(parsed_tasks)
        
        # Log violations for learning system
        if violations:
            self._log_violations(parsed_tasks, violations, suggestions)
            
        return ValidationResult(
            is_valid=len(violations) == 0,
            pattern_type=pattern_type,
            violations=violations,
            suggestions=suggestions,
            confidence_score=confidence_score
        )
        
    def _parse_task_description(self, task_desc: Dict[str, str]) -> TaskDescription:
        """Parse a Task description for validation analysis"""
        description = task_desc.get("description", "")
        prompt = task_desc.get("prompt", "")
        agent_type = task_desc.get("subagent_type", "")
        
        # Extract issue numbers from description and prompt
        issue_numbers = self._extract_issue_numbers(description + " " + prompt)
        
        # Extract agent name from description
        agent_name = self._extract_agent_name(description)
        
        # Check for multi-issue indicators
        is_multi_issue = len(issue_numbers) > 1
        
        # Check for generic accelerator patterns
        is_generic_accelerator = self._is_generic_accelerator(description)
        
        return TaskDescription(
            description=description,
            prompt=prompt,
            agent_type=agent_type,
            issue_numbers=issue_numbers,
            agent_name=agent_name,
            is_multi_issue=is_multi_issue,
            is_generic_accelerator=is_generic_accelerator
        )
        
    def _extract_issue_numbers(self, text: str) -> List[int]:
        """Extract GitHub issue numbers from text"""
        matches = re.findall(self.ISSUE_NUMBER_PATTERN, text)
        return [int(match) for match in matches]
        
    def _extract_agent_name(self, description: str) -> str:
        """Extract agent name from Task description"""
        # Look for RIF agent patterns
        for agent in self.VALID_RIF_AGENTS:
            if agent in description:
                return agent
                
        # Look for other patterns
        if ":" in description:
            return description.split(":")[0].strip()
            
        return "Unknown"
        
    def _is_generic_accelerator(self, description: str) -> bool:
        """Check if description matches generic accelerator anti-pattern"""
        for pattern in self.MULTI_ISSUE_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                return True
        return False
        
    def _detect_multi_issue_anti_pattern(self, tasks: List[TaskDescription]) -> List[str]:
        """Detect Multi-Issue Accelerator anti-pattern"""
        violations = []
        
        for task in tasks:
            # Check for multi-issue single Task
            if task.is_multi_issue:
                violations.append(
                    f"Multi-Issue Anti-Pattern: Task '{task.description}' "
                    f"attempts to handle {len(task.issue_numbers)} issues "
                    f"({task.issue_numbers}) in single Task"
                )
                
            # Check for generic accelerator naming
            if task.is_generic_accelerator:
                violations.append(
                    f"Generic Accelerator Anti-Pattern: Task '{task.description}' "
                    f"uses generic accelerator naming pattern"
                )
                
        return violations
        
    def _validate_parallel_patterns(self, tasks: List[TaskDescription]) -> List[str]:
        """Validate proper parallel execution patterns"""
        violations = []
        
        # If multiple tasks, check they're properly structured for parallel execution
        if len(tasks) > 1:
            issue_sets = [set(task.issue_numbers) for task in tasks if task.issue_numbers]
            
            # Check for overlapping issues (potential conflict)
            for i, issues1 in enumerate(issue_sets):
                for j, issues2 in enumerate(issue_sets[i+1:], i+1):
                    overlap = issues1.intersection(issues2)
                    if overlap:
                        violations.append(
                            f"Issue Overlap: Tasks {i} and {j} both handle issue(s) {list(overlap)} "
                            f"which may cause conflicts"
                        )
                        
        return violations
        
    def _validate_agent_specialization(self, tasks: List[TaskDescription]) -> List[str]:
        """Validate proper agent specialization"""
        violations = []
        
        for task in tasks:
            # Check if agent name is valid RIF agent
            if task.agent_name not in self.VALID_RIF_AGENTS and task.agent_name != "Unknown":
                violations.append(
                    f"Invalid Agent: '{task.agent_name}' is not a standard RIF agent. "
                    f"Use: {', '.join(self.VALID_RIF_AGENTS)}"
                )
                
            # Check if prompt includes proper agent instructions
            if "Follow all instructions in claude/agents/" not in task.prompt:
                violations.append(
                    f"Missing Instructions: Task '{task.description}' prompt does not include "
                    f"'Follow all instructions in claude/agents/[agent-file].md'"
                )
                
        return violations
        
    def _generate_corrective_suggestions(
        self, 
        tasks: List[TaskDescription], 
        violations: List[str]
    ) -> List[str]:
        """Generate corrective suggestions for violations"""
        suggestions = []
        
        # Suggestions for multi-issue violations
        multi_issue_tasks = [t for t in tasks if t.is_multi_issue]
        if multi_issue_tasks:
            suggestions.append(
                "Split multi-issue Tasks: Create separate Task() for each issue number. "
                "Launch all Tasks in one response for parallel execution."
            )
            
        # Suggestions for generic accelerator violations  
        accelerator_tasks = [t for t in tasks if t.is_generic_accelerator]
        if accelerator_tasks:
            suggestions.append(
                "Use specialized agents: Replace generic 'Accelerator' with specific "
                f"RIF agents: {', '.join(self.VALID_RIF_AGENTS)}"
            )
            
        # Suggestions for instruction violations
        missing_instruction_tasks = [
            t for t in tasks 
            if "Follow all instructions in claude/agents/" not in t.prompt
        ]
        if missing_instruction_tasks:
            suggestions.append(
                "Add agent instructions: Include 'Follow all instructions in "
                "claude/agents/[agent-file].md' in all Task prompts"
            )
            
        return suggestions
        
    def _calculate_confidence_score(
        self, 
        tasks: List[TaskDescription], 
        violations: List[str]
    ) -> float:
        """Calculate confidence score for validation (0-1, higher is better)"""
        if not violations:
            return 1.0
            
        # Base score
        score = 1.0
        
        # Deduct for violations
        for violation in violations:
            if "Multi-Issue Anti-Pattern" in violation:
                score -= 0.4  # Major violation
            elif "Generic Accelerator" in violation:
                score -= 0.3  # Major violation
            elif "Issue Overlap" in violation:
                score -= 0.2  # Medium violation
            else:
                score -= 0.1  # Minor violation
                
        return max(0.0, score)
        
    def _determine_pattern_type(self, tasks: List[TaskDescription]) -> str:
        """Determine the orchestration pattern type"""
        if not tasks:
            return "empty"
            
        if len(tasks) == 1:
            if tasks[0].is_multi_issue:
                return "multi-issue-single-task"  # Anti-pattern
            else:
                return "single-task"
        else:
            return "parallel-tasks"  # Correct pattern
            
    def _log_violations(
        self,
        tasks: List[TaskDescription],
        violations: List[str], 
        suggestions: List[str]
    ):
        """Log violations for learning system"""
        violation_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "issue_id": 224,  # Reference issue
            "pattern_type": self._determine_pattern_type(tasks),
            "task_count": len(tasks),
            "violations": violations,
            "suggestions": suggestions,
            "tasks": [
                {
                    "description": task.description,
                    "agent_name": task.agent_name,
                    "issue_numbers": task.issue_numbers,
                    "is_multi_issue": task.is_multi_issue,
                    "is_generic_accelerator": task.is_generic_accelerator
                }
                for task in tasks
            ]
        }
        
        self.violation_log.append(violation_log)
        
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        return {
            "validator_version": "1.0.0",
            "issue_reference": 224,
            "anti_patterns_detected": [
                "Multi-Issue Accelerator",
                "Generic Accelerator Naming",
                "Issue Overlap",
                "Invalid Agent Names",
                "Missing Instructions"
            ],
            "violation_count": len(self.violation_log),
            "violations": self.violation_log
        }


def validate_task_request(task_descriptions: List[Dict[str, str]]) -> ValidationResult:
    """
    Convenience function for validating orchestration requests.
    
    Args:
        task_descriptions: List of Task description dicts
        
    Returns:
        ValidationResult with validation status and recommendations
    """
    validator = OrchestrationPatternValidator()
    return validator.validate_orchestration_request(task_descriptions)


# Example usage and testing
if __name__ == "__main__":
    # Test anti-pattern detection
    anti_pattern_request = [
        {
            "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
            "prompt": "You are a Multi-Issue Accelerator. Handle these issues in parallel: Issue #1: user auth, Issue #2: database pool, Issue #3: API validation",
            "subagent_type": "general-purpose"
        }
    ]
    
    # Test correct pattern
    correct_pattern_request = [
        {
            "description": "RIF-Implementer: User authentication system", 
            "prompt": "You are RIF-Implementer. Implement user authentication for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Implementer: Database connection pooling",
            "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.", 
            "subagent_type": "general-purpose"
        }
    ]
    
    print("Testing Anti-Pattern Detection:")
    result = validate_task_request(anti_pattern_request)
    print(f"Valid: {result.is_valid}")
    print(f"Violations: {result.violations}")
    print(f"Suggestions: {result.suggestions}")
    print()
    
    print("Testing Correct Pattern:")
    result = validate_task_request(correct_pattern_request)
    print(f"Valid: {result.is_valid}")
    print(f"Pattern Type: {result.pattern_type}")
    print(f"Confidence Score: {result.confidence_score}")