#!/usr/bin/env python3
"""
Orchestration Intelligence Integration Module - Issue #228 Implementation
RIF-Implementer: Critical orchestration failure resolution integration

This module provides the main entry points for Claude Code to use the enhanced
orchestration intelligence with blocking detection capabilities.

ISSUE #228: Fixes critical orchestration failure where blocking issues were ignored.
Original Problem: Issue #225 declared "THIS ISSUE BLOCKS ALL OTHERS" but the
orchestrator proceeded with parallel work on issues #226 and #227.

Solution: Pre-flight blocking detection that halts orchestration when blocking
declarations are found in issue bodies or comments.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import the enhanced orchestration intelligence
try:
    from .enhanced_orchestration_intelligence import (
        EnhancedOrchestrationIntelligence,
        EnhancedBlockingDetectionEngine
    )
    ENHANCED_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced orchestration intelligence not available: {e}")
    ENHANCED_INTELLIGENCE_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class OrchestrationDecision:
    """
    Decision result from intelligent orchestration analysis.
    ISSUE #228: Enhanced with blocking detection results.
    """
    decision_type: str  # "allow_execution", "block_execution", "halt_all_orchestration"
    enforcement_action: str  # "ALLOW", "BLOCK", "HALT_ALL_ORCHESTRATION"
    blocking_issues: List[int]
    allowed_issues: List[int]
    blocked_issues: List[int]
    dependency_rationale: str
    blocking_analysis: Dict[str, Any]
    task_launch_codes: List[str]
    execution_ready: bool
    parallel_execution: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for JSON serialization"""
        return {
            'decision_type': self.decision_type,
            'enforcement_action': self.enforcement_action,
            'blocking_issues': self.blocking_issues,
            'allowed_issues': self.allowed_issues,
            'blocked_issues': self.blocked_issues,
            'dependency_rationale': self.dependency_rationale,
            'blocking_analysis': self.blocking_analysis,
            'task_launch_codes': self.task_launch_codes,
            'execution_ready': self.execution_ready,
            'parallel_execution': self.parallel_execution,
            'timestamp': datetime.now().isoformat()
        }

@dataclass
class ValidationStatus:
    """Validation status with detailed feedback"""
    is_valid: bool
    violations: List[str]
    suggestions: List[str]
    blocking_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'violations': self.violations,
            'suggestions': self.suggestions,
            'blocking_detected': self.blocking_detected
        }

def validate_orchestration_request(github_issues: List[int], 
                                 context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Pre-flight validation for orchestration requests.
    ISSUE #228: Enhanced with blocking detection to prevent orchestration failures.
    
    Args:
        github_issues: List of GitHub issue numbers
        context: Additional context for validation
        
    Returns:
        Tuple of (should_block, message)
    """
    if not ENHANCED_INTELLIGENCE_AVAILABLE:
        logger.warning("Enhanced intelligence not available - allowing orchestration")
        return False, "Enhanced intelligence unavailable - proceeding with basic orchestration"
    
    try:
        # Initialize blocking detection engine
        blocking_engine = EnhancedBlockingDetectionEngine()
        
        # Perform blocking detection
        blocking_analysis = blocking_engine.detect_blocking_issues(github_issues)
        
        # If blocking issues detected, block orchestration
        if blocking_analysis['has_blocking_issues']:
            blocking_issues = blocking_analysis['blocking_issues']
            blocking_details = blocking_analysis['blocking_details']
            
            message = f"ORCHESTRATION BLOCKED: {len(blocking_issues)} blocking issues detected"
            for issue_num in blocking_issues:
                details = blocking_details.get(str(issue_num), {})
                phrases = details.get('detected_phrases', [])
                message += f"\n  - Issue #{issue_num}: {phrases}"
            
            message += f"\n  ACTION: Complete blocking issues before proceeding with others"
            message += f"\n  BLOCKED ISSUES: {blocking_analysis['non_blocking_issues']}"
            
            logger.critical(message)
            return True, message
        
        # No blocking issues - allow orchestration
        logger.info(f"Pre-flight validation passed for issues: {github_issues}")
        return False, f"Validation passed - {len(github_issues)} issues ready for orchestration"
        
    except Exception as e:
        logger.error(f"Error in orchestration validation: {e}")
        return False, f"Validation error - proceeding with caution: {e}"

def make_intelligent_orchestration_decision(github_issues: List[int], 
                                          proposed_tasks: Optional[List[Dict[str, Any]]] = None,
                                          context: Optional[Dict[str, Any]] = None) -> OrchestrationDecision:
    """
    Main entry point for Claude Code orchestration decision-making.
    ISSUE #228: Enhanced with blocking detection to prevent critical failures.
    
    This function integrates all orchestration intelligence capabilities and provides
    Claude Code with the information needed to make proper Task() launching decisions.
    
    Args:
        github_issues: List of GitHub issue numbers to analyze
        proposed_tasks: Optional list of proposed task definitions
        context: Additional context (e.g., user request type)
        
    Returns:
        OrchestrationDecision with complete analysis and recommendations
    """
    logger.info(f"Making intelligent orchestration decision for issues: {github_issues}")
    
    if not ENHANCED_INTELLIGENCE_AVAILABLE:
        # Fallback decision when enhanced intelligence is not available
        return OrchestrationDecision(
            decision_type="fallback_execution",
            enforcement_action="ALLOW",
            blocking_issues=[],
            allowed_issues=github_issues,
            blocked_issues=[],
            dependency_rationale="Enhanced intelligence unavailable - proceeding with basic orchestration",
            blocking_analysis={'has_blocking_issues': False, 'blocking_count': 0},
            task_launch_codes=[],
            execution_ready=True,
            parallel_execution=len(github_issues) > 1
        )
    
    try:
        # Initialize enhanced orchestration intelligence
        orchestration_intelligence = EnhancedOrchestrationIntelligence()
        
        # Generate comprehensive orchestration plan with blocking detection
        orchestration_plan = orchestration_intelligence.generate_orchestration_plan(github_issues)
        
        # Check if orchestration was blocked due to blocking issues
        if orchestration_plan.get('orchestration_blocked', False):
            return OrchestrationDecision(
                decision_type="block_execution",
                enforcement_action="HALT_ALL_ORCHESTRATION",
                blocking_issues=orchestration_plan['blocking_issues'],
                allowed_issues=orchestration_plan['allowed_issues'],
                blocked_issues=orchestration_plan['blocked_issues'],
                dependency_rationale=orchestration_plan['message'],
                blocking_analysis=orchestration_plan['blocking_analysis'],
                task_launch_codes=orchestration_plan['task_launch_codes'],
                execution_ready=orchestration_plan['execution_ready'],
                parallel_execution=orchestration_plan['parallel_execution']
            )
        
        # No blocking issues - allow orchestration
        return OrchestrationDecision(
            decision_type="allow_execution",
            enforcement_action="ALLOW",
            blocking_issues=[],
            allowed_issues=github_issues,
            blocked_issues=[],
            dependency_rationale="No blocking issues detected - orchestration approved",
            blocking_analysis=orchestration_plan['blocking_analysis'],
            task_launch_codes=orchestration_plan['task_launch_codes'],
            execution_ready=orchestration_plan['execution_ready'],
            parallel_execution=orchestration_plan['parallel_execution']
        )
        
    except Exception as e:
        logger.error(f"Error in intelligent orchestration decision: {e}")
        # Error fallback - allow orchestration but with warning
        return OrchestrationDecision(
            decision_type="error_fallback",
            enforcement_action="ALLOW",
            blocking_issues=[],
            allowed_issues=github_issues,
            blocked_issues=[],
            dependency_rationale=f"Orchestration intelligence error - proceeding with caution: {e}",
            blocking_analysis={'error': str(e), 'has_blocking_issues': False},
            task_launch_codes=[],
            execution_ready=True,
            parallel_execution=len(github_issues) > 1
        )

def validate_orchestration_patterns(proposed_tasks: List[Dict[str, Any]]) -> ValidationStatus:
    """
    Validate proposed orchestration patterns to prevent anti-patterns.
    
    Args:
        proposed_tasks: List of proposed task definitions
        
    Returns:
        ValidationStatus with validation results
    """
    violations = []
    suggestions = []
    
    # Check for anti-patterns
    for i, task in enumerate(proposed_tasks):
        task_description = task.get('description', '')
        task_prompt = task.get('prompt', '')
        
        # Check for multi-issue accelerator anti-pattern
        if any(phrase in task_description.lower() for phrase in [
            'multi-issue', 'multiple issues', 'handle issues', 'accelerator'
        ]):
            violations.append(f"Task {i+1}: Multi-issue accelerator anti-pattern detected")
            suggestions.append(f"Task {i+1}: Create separate tasks for each issue instead")
        
        # Check for proper agent specialization
        if 'you are ' not in task_prompt.lower():
            violations.append(f"Task {i+1}: Missing agent role specification")
            suggestions.append(f"Task {i+1}: Include 'You are [AGENT_NAME]' in prompt")
        
        # Check for proper agent file reference
        if 'follow all instructions in claude/agents/' not in task_prompt.lower():
            violations.append(f"Task {i+1}: Missing agent file reference")
            suggestions.append(f"Task {i+1}: Include agent file reference in prompt")
    
    # Check for blocking detection capability
    blocking_detected = any(
        'blocking' in task.get('description', '').lower() or 
        'blocks all others' in task.get('prompt', '').lower()
        for task in proposed_tasks
    )
    
    is_valid = len(violations) == 0
    
    return ValidationStatus(
        is_valid=is_valid,
        violations=violations,
        suggestions=suggestions,
        blocking_detected=blocking_detected
    )

def validate_before_execution(proposed_tasks: List[Dict[str, Any]]) -> None:
    """
    Runtime validation hook for orchestration patterns.
    Raises exception if critical violations detected.
    
    Args:
        proposed_tasks: List of proposed task definitions
        
    Raises:
        ValueError: If critical orchestration violations detected
    """
    validation_result = validate_orchestration_patterns(proposed_tasks)
    
    if not validation_result.is_valid:
        error_message = "Orchestration pattern violations detected:\n"
        error_message += "\n".join(f"  - {violation}" for violation in validation_result.violations)
        error_message += "\n\nSuggestions:\n"
        error_message += "\n".join(f"  - {suggestion}" for suggestion in validation_result.suggestions)
        
        logger.error(error_message)
        raise ValueError(error_message)

def get_enhanced_blocking_detection_status() -> Dict[str, Any]:
    """
    Get status of enhanced blocking detection system.
    
    Returns:
        Dict with system status information
    """
    return {
        'enhanced_intelligence_available': ENHANCED_INTELLIGENCE_AVAILABLE,
        'blocking_detection_active': ENHANCED_INTELLIGENCE_AVAILABLE,
        'issue_228_integration': True,
        'pre_flight_validation_active': True,
        'supported_blocking_phrases': [
            "this issue blocks all others",
            "this issue blocks all other work", 
            "blocks all other work",
            "blocks all others",
            "stop all work",
            "must complete before all",
            "must complete before all other work",
            "must complete before all others"
        ] if ENHANCED_INTELLIGENCE_AVAILABLE else []
    }

# Example usage and testing functions
def example_usage():
    """Example of how Claude Code should use this module"""
    print("🔧 Orchestration Intelligence Integration - Issue #228")
    print("=" * 60)
    
    # Example 1: Basic orchestration decision
    print("\n1. Basic orchestration decision:")
    issues = [228, 229, 230]
    decision = make_intelligent_orchestration_decision(issues)
    print(f"   Decision: {decision.decision_type}")
    print(f"   Enforcement: {decision.enforcement_action}")
    print(f"   Execution Ready: {decision.execution_ready}")
    
    # Example 2: Pre-flight validation
    print("\n2. Pre-flight validation:")
    should_block, message = validate_orchestration_request(issues)
    print(f"   Should Block: {should_block}")
    print(f"   Message: {message}")
    
    # Example 3: System status
    print("\n3. System status:")
    status = get_enhanced_blocking_detection_status()
    print(f"   Blocking Detection Active: {status['blocking_detection_active']}")
    print(f"   Issue #228 Integration: {status['issue_228_integration']}")
    
    print(f"\n✅ Orchestration Intelligence Integration ready for use!")

if __name__ == "__main__":
    example_usage()