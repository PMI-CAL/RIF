"""
Orchestration Validation Enforcer

Real-time validation and enforcement of orchestration patterns.
Integrates with Claude Code to prevent anti-pattern execution.

Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .orchestration_pattern_validator import (
    OrchestrationPatternValidator,
    ValidationResult,
    validate_task_request
)


class OrchestrationValidationEnforcer:
    """
    Enforces orchestration patterns in real-time by intercepting
    Task invocation attempts and validating against anti-patterns.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.validator = OrchestrationPatternValidator()
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.enforcement_log = []
        
    def validate_and_enforce(
        self,
        proposed_tasks: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate proposed Task invocations and enforce patterns.
        
        Args:
            proposed_tasks: List of proposed Task descriptions
            context: Optional context about the orchestration request
            
        Returns:
            Dict with validation results and enforcement actions
        """
        # Perform validation
        validation_result = self.validator.validate_orchestration_request(proposed_tasks)
        
        # Create enforcement record
        enforcement_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_count": len(proposed_tasks),
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "pattern_type": validation_result.pattern_type,
                "violations": validation_result.violations,
                "suggestions": validation_result.suggestions,
                "confidence_score": validation_result.confidence_score
            },
            "context": context or {},
            "enforcement_action": self._determine_enforcement_action(validation_result),
            "corrective_guidance": self._generate_corrective_guidance(validation_result)
        }
        
        # Log enforcement action
        self.enforcement_log.append(enforcement_record)
        
        # Store in knowledge base
        self._store_enforcement_record(enforcement_record)
        
        return enforcement_record
        
    def _determine_enforcement_action(self, validation_result: ValidationResult) -> str:
        """Determine what enforcement action to take based on validation"""
        if validation_result.is_valid:
            return "allow_execution"
            
        # Check severity of violations
        has_critical_violations = any(
            "Multi-Issue Anti-Pattern" in violation or "Generic Accelerator" in violation
            for violation in validation_result.violations
        )
        
        if has_critical_violations:
            return "block_execution"
        else:
            return "warn_and_allow"
            
    def _generate_corrective_guidance(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Generate detailed corrective guidance"""
        if validation_result.is_valid:
            return {
                "status": "valid",
                "message": "Orchestration pattern validation passed. Proceeding with execution.",
                "recommendations": []
            }
            
        guidance = {
            "status": "violations_detected",
            "message": "Orchestration pattern violations detected. Review and correct before execution.",
            "violations": validation_result.violations,
            "suggestions": validation_result.suggestions,
            "corrective_examples": self._generate_corrective_examples(validation_result)
        }
        
        return guidance
        
    def _generate_corrective_examples(self, validation_result: ValidationResult) -> List[Dict[str, str]]:
        """Generate specific corrective examples for violations"""
        examples = []
        
        # Multi-Issue Anti-Pattern corrections
        if any("Multi-Issue Anti-Pattern" in v for v in validation_result.violations):
            examples.append({
                "violation_type": "Multi-Issue Anti-Pattern",
                "wrong_approach": '''Task(
    description="Multi-Issue Accelerator: Handle issues #1, #2, #3",
    prompt="Handle multiple issues in parallel..."
)''',
                "correct_approach": '''# Multiple Tasks in ONE response:
Task(
    description="RIF-Implementer: Issue #1 implementation",
    prompt="You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md."
)
Task(
    description="RIF-Implementer: Issue #2 implementation", 
    prompt="You are RIF-Implementer. Handle issue #2. Follow all instructions in claude/agents/rif-implementer.md."
)
Task(
    description="RIF-Implementer: Issue #3 implementation",
    prompt="You are RIF-Implementer. Handle issue #3. Follow all instructions in claude/agents/rif-implementer.md."
)'''
            })
            
        # Generic Accelerator corrections
        if any("Generic Accelerator" in v for v in validation_result.violations):
            examples.append({
                "violation_type": "Generic Accelerator Anti-Pattern",
                "wrong_approach": '''Task(
    description="Batch Processing Accelerator",
    prompt="Process multiple tasks efficiently..."
)''',
                "correct_approach": '''Task(
    description="RIF-Implementer: Specific implementation task",
    prompt="You are RIF-Implementer. Implement specific feature. Follow all instructions in claude/agents/rif-implementer.md."
)'''
            })
            
        return examples
        
    def _store_enforcement_record(self, record: Dict[str, Any]):
        """Store enforcement record in knowledge base"""
        try:
            enforcement_dir = Path(self.knowledge_base_path) / "enforcement_logs"
            enforcement_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"orchestration_enforcement_{timestamp}.json"
            filepath = enforcement_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(record, f, indent=2)
                
        except Exception as e:
            # Log error but don't fail enforcement
            print(f"Warning: Could not store enforcement record: {e}")
            
    def generate_enforcement_report(self) -> Dict[str, Any]:
        """Generate comprehensive enforcement report"""
        total_requests = len(self.enforcement_log)
        blocked_requests = sum(
            1 for record in self.enforcement_log 
            if record["enforcement_action"] == "block_execution"
        )
        warned_requests = sum(
            1 for record in self.enforcement_log
            if record["enforcement_action"] == "warn_and_allow" 
        )
        
        # Violation analysis
        all_violations = []
        for record in self.enforcement_log:
            all_violations.extend(record["validation_result"]["violations"])
            
        violation_types = {}
        for violation in all_violations:
            if "Multi-Issue Anti-Pattern" in violation:
                violation_types["multi_issue"] = violation_types.get("multi_issue", 0) + 1
            elif "Generic Accelerator" in violation:
                violation_types["generic_accelerator"] = violation_types.get("generic_accelerator", 0) + 1
            elif "Issue Overlap" in violation:
                violation_types["issue_overlap"] = violation_types.get("issue_overlap", 0) + 1
            else:
                violation_types["other"] = violation_types.get("other", 0) + 1
                
        return {
            "enforcement_summary": {
                "total_requests": total_requests,
                "blocked_requests": blocked_requests,
                "warned_requests": warned_requests,
                "allowed_requests": total_requests - blocked_requests - warned_requests,
                "block_rate": blocked_requests / total_requests if total_requests > 0 else 0,
                "violation_rate": (blocked_requests + warned_requests) / total_requests if total_requests > 0 else 0
            },
            "violation_analysis": {
                "total_violations": len(all_violations),
                "violation_types": violation_types,
                "most_common_violation": max(violation_types.items(), key=lambda x: x[1])[0] if violation_types else None
            },
            "recent_enforcements": self.enforcement_log[-5:] if len(self.enforcement_log) >= 5 else self.enforcement_log,
            "generated_at": datetime.utcnow().isoformat()
        }


# Integration helpers for Claude Code
def validate_orchestration_request(
    task_descriptions: List[Dict[str, str]], 
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main validation entry point for orchestration requests.
    
    Args:
        task_descriptions: List of proposed Task descriptions
        context: Optional context about the request
        
    Returns:
        Enforcement record with validation results and guidance
    """
    enforcer = OrchestrationValidationEnforcer()
    return enforcer.validate_and_enforce(task_descriptions, context)


def check_orchestration_pattern(task_descriptions: List[Dict[str, str]]) -> bool:
    """
    Quick check for orchestration pattern validity.
    
    Args:
        task_descriptions: List of proposed Task descriptions
        
    Returns:
        True if patterns are valid, False if violations detected
    """
    result = validate_task_request(task_descriptions)
    return result.is_valid


def get_orchestration_guidance(task_descriptions: List[Dict[str, str]]) -> str:
    """
    Get human-readable guidance for orchestration patterns.
    
    Args:
        task_descriptions: List of proposed Task descriptions
        
    Returns:
        Formatted guidance string
    """
    enforcement_record = validate_orchestration_request(task_descriptions)
    guidance = enforcement_record["corrective_guidance"]
    
    if guidance["status"] == "valid":
        return "✅ Orchestration pattern validation passed. Proceeding with execution."
        
    output = "❌ Orchestration pattern violations detected:\n\n"
    
    for violation in guidance["violations"]:
        output += f"• {violation}\n"
        
    output += "\n🔧 Corrective suggestions:\n\n"
    
    for suggestion in guidance["suggestions"]:
        output += f"• {suggestion}\n"
        
    if "corrective_examples" in guidance:
        output += "\n📚 Corrective examples:\n\n"
        for example in guidance["corrective_examples"]:
            output += f"**{example['violation_type']}**\n\n"
            output += f"❌ Wrong:\n```python\n{example['wrong_approach']}\n```\n\n"
            output += f"✅ Correct:\n```python\n{example['correct_approach']}\n```\n\n"
            
    return output


# Example usage and testing
if __name__ == "__main__":
    # Test enforcement with anti-pattern
    anti_pattern_tasks = [
        {
            "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
            "prompt": "Handle multiple issues efficiently",
            "subagent_type": "general-purpose"
        }
    ]
    
    print("Testing Orchestration Enforcement:")
    print("=" * 50)
    
    guidance = get_orchestration_guidance(anti_pattern_tasks)
    print(guidance)
    
    print("\nTesting Quick Validation Check:")
    print("=" * 50)
    
    is_valid = check_orchestration_pattern(anti_pattern_tasks)
    print(f"Pattern valid: {is_valid}")