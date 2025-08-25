"""
Test Suite for Orchestration Pattern Validation

Comprehensive tests for Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
Tests validation framework, anti-pattern detection, and enforcement mechanisms.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

# Import the components we're testing
import sys
sys.path.append('/Users/cal/DEV/RIF')

from claude.commands.orchestration_pattern_validator import (
    OrchestrationPatternValidator,
    ValidationResult,
    TaskDescription,
    validate_task_request
)
from claude.commands.orchestration_validation_enforcer import (
    OrchestrationValidationEnforcer,
    validate_orchestration_request,
    check_orchestration_pattern,
    get_orchestration_guidance
)


class TestOrchestrationPatternValidator:
    """Unit tests for OrchestrationPatternValidator"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = OrchestrationPatternValidator()
        
    def test_multi_issue_anti_pattern_detection(self):
        """Test detection of Multi-Issue Accelerator anti-pattern"""
        # Anti-pattern: Single Task handling multiple issues
        anti_pattern_tasks = [
            {
                "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
                "prompt": "You are a Multi-Issue Accelerator. Handle these issues: Issue #1: user auth, Issue #2: database, Issue #3: API",
                "subagent_type": "general-purpose"
            }
        ]
        
        result = self.validator.validate_orchestration_request(anti_pattern_tasks)
        
        assert not result.is_valid
        assert result.pattern_type == "multi-issue-single-task"
        assert len(result.violations) >= 1
        assert any("Multi-Issue Anti-Pattern" in v for v in result.violations)
        assert any("Generic Accelerator Anti-Pattern" in v for v in result.violations)
        assert result.confidence_score < 0.5
        
    def test_correct_parallel_pattern_validation(self):
        """Test validation of correct parallel Task pattern"""
        # Correct pattern: Multiple Tasks for parallel execution
        correct_pattern_tasks = [
            {
                "description": "RIF-Implementer: User authentication system",
                "prompt": "You are RIF-Implementer. Implement user authentication for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Database connection pooling", 
                "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: API validation framework",
                "prompt": "You are RIF-Validator. Validate API framework for issue #3. Follow all instructions in claude/agents/rif-validator.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        result = self.validator.validate_orchestration_request(correct_pattern_tasks)
        
        assert result.is_valid
        assert result.pattern_type == "parallel-tasks"
        assert len(result.violations) == 0
        assert result.confidence_score == 1.0
        
    def test_generic_accelerator_detection(self):
        """Test detection of generic accelerator anti-patterns"""
        generic_patterns = [
            "Multi-Issue Accelerator",
            "Batch Processing Engine", 
            "Combined Task Handler",
            "Parallel Issues Manager"
        ]
        
        for pattern_name in generic_patterns:
            tasks = [
                {
                    "description": f"{pattern_name}: Handle multiple tasks",
                    "prompt": f"You are a {pattern_name}. Process these efficiently.",
                    "subagent_type": "general-purpose"
                }
            ]
            
            result = self.validator.validate_orchestration_request(tasks)
            assert not result.is_valid
            assert any("Generic Accelerator" in v for v in result.violations)
            
    def test_issue_number_extraction(self):
        """Test extraction of issue numbers from task descriptions"""
        text_samples = [
            ("Handle issue #123", [123]),
            ("Issues #1, #2, and #3", [1, 2, 3]),
            ("Process GitHub issue #456 and #789", [456, 789]),
            ("No issues mentioned here", []),
            ("Fix #42 and implement #99", [42, 99])
        ]
        
        for text, expected_issues in text_samples:
            issues = self.validator._extract_issue_numbers(text)
            assert issues == expected_issues
            
    def test_agent_name_validation(self):
        """Test validation of agent names"""
        valid_agents = [
            "RIF-Analyst",
            "RIF-Planner", 
            "RIF-Architect",
            "RIF-Implementer",
            "RIF-Validator",
            "RIF-Learner"
        ]
        
        for agent in valid_agents:
            tasks = [
                {
                    "description": f"{agent}: Valid agent task",
                    "prompt": f"You are {agent}. Follow all instructions in claude/agents/rif-implementer.md.",
                    "subagent_type": "general-purpose"
                }
            ]
            
            result = self.validator.validate_orchestration_request(tasks)
            # Should not have agent name violations
            agent_violations = [v for v in result.violations if "Invalid Agent" in v]
            assert len(agent_violations) == 0
            
    def test_missing_instructions_detection(self):
        """Test detection of missing agent instructions"""
        tasks_without_instructions = [
            {
                "description": "RIF-Implementer: Valid task",
                "prompt": "You are RIF-Implementer. Do the work.",  # Missing instructions
                "subagent_type": "general-purpose"
            }
        ]
        
        result = self.validator.validate_orchestration_request(tasks_without_instructions)
        
        instruction_violations = [v for v in result.violations if "Missing Instructions" in v]
        assert len(instruction_violations) > 0
        
    def test_issue_overlap_detection(self):
        """Test detection of overlapping issues between tasks"""
        overlapping_tasks = [
            {
                "description": "RIF-Implementer: Handle issue #1",
                "prompt": "You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: Also handle issue #1",  # Overlap!
                "prompt": "You are RIF-Validator. Validate issue #1. Follow all instructions in claude/agents/rif-validator.md.", 
                "subagent_type": "general-purpose"
            }
        ]
        
        result = self.validator.validate_orchestration_request(overlapping_tasks)
        
        overlap_violations = [v for v in result.violations if "Issue Overlap" in v]
        assert len(overlap_violations) > 0
        
    def test_empty_task_list(self):
        """Test validation of empty task list"""
        result = self.validator.validate_orchestration_request([])
        
        assert result.is_valid  # Empty is valid, just nothing to do
        assert result.pattern_type == "empty"
        assert len(result.violations) == 0
        
    def test_single_valid_task(self):
        """Test validation of single valid task"""
        single_task = [
            {
                "description": "RIF-Implementer: Implement feature X", 
                "prompt": "You are RIF-Implementer. Implement feature X for issue #42. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        result = self.validator.validate_orchestration_request(single_task)
        
        assert result.is_valid
        assert result.pattern_type == "single-task"
        assert len(result.violations) == 0
        assert result.confidence_score == 1.0


class TestOrchestrationValidationEnforcer:
    """Integration tests for OrchestrationValidationEnforcer"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Use temporary directory for knowledge base
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = OrchestrationValidationEnforcer(knowledge_base_path=self.temp_dir)
        
    def teardown_method(self):
        """Cleanup after each test method"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_enforcement_blocking(self):
        """Test that critical violations are blocked"""
        critical_anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3", 
                "prompt": "Handle multiple issues in parallel",
                "subagent_type": "general-purpose"
            }
        ]
        
        enforcement_record = self.enforcer.validate_and_enforce(critical_anti_pattern)
        
        assert enforcement_record["enforcement_action"] == "block_execution"
        assert not enforcement_record["validation_result"]["is_valid"]
        assert enforcement_record["corrective_guidance"]["status"] == "violations_detected"
        
    def test_enforcement_allowing(self):
        """Test that valid patterns are allowed"""
        valid_pattern = [
            {
                "description": "RIF-Implementer: Implement feature",
                "prompt": "You are RIF-Implementer. Implement feature for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        enforcement_record = self.enforcer.validate_and_enforce(valid_pattern)
        
        assert enforcement_record["enforcement_action"] == "allow_execution"
        assert enforcement_record["validation_result"]["is_valid"]
        assert enforcement_record["corrective_guidance"]["status"] == "valid"
        
    def test_corrective_guidance_generation(self):
        """Test generation of corrective guidance"""
        anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Handle issues #1, #2",
                "prompt": "Handle multiple issues",
                "subagent_type": "general-purpose" 
            }
        ]
        
        enforcement_record = self.enforcer.validate_and_enforce(anti_pattern)
        guidance = enforcement_record["corrective_guidance"]
        
        assert "corrective_examples" in guidance
        assert len(guidance["corrective_examples"]) > 0
        
        # Check that examples include both wrong and correct approaches
        example = guidance["corrective_examples"][0]
        assert "wrong_approach" in example
        assert "correct_approach" in example
        assert "violation_type" in example
        
    def test_knowledge_base_logging(self):
        """Test that enforcement records are logged to knowledge base"""
        anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Test logging",
                "prompt": "Test logging functionality", 
                "subagent_type": "general-purpose"
            }
        ]
        
        self.enforcer.validate_and_enforce(anti_pattern)
        
        # Check that enforcement log directory was created
        enforcement_dir = Path(self.temp_dir) / "enforcement_logs"
        assert enforcement_dir.exists()
        
        # Check that log files were created
        log_files = list(enforcement_dir.glob("orchestration_enforcement_*.json"))
        assert len(log_files) > 0
        
        # Verify log content
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
            
        assert "timestamp" in log_data
        assert "validation_result" in log_data
        assert "enforcement_action" in log_data
        
    def test_enforcement_report_generation(self):
        """Test generation of enforcement reports"""
        # Generate some test data
        test_patterns = [
            # Valid pattern
            [{
                "description": "RIF-Implementer: Valid task",
                "prompt": "You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }],
            # Anti-pattern
            [{
                "description": "Multi-Issue Accelerator: Invalid task", 
                "prompt": "Handle multiple issues",
                "subagent_type": "general-purpose"
            }]
        ]
        
        for pattern in test_patterns:
            self.enforcer.validate_and_enforce(pattern)
            
        report = self.enforcer.generate_enforcement_report()
        
        assert "enforcement_summary" in report
        assert "violation_analysis" in report
        assert "recent_enforcements" in report
        
        summary = report["enforcement_summary"] 
        assert summary["total_requests"] == 2
        assert summary["blocked_requests"] == 1
        assert summary["allowed_requests"] == 1
        assert summary["violation_rate"] == 0.5


class TestConvenienceFunctions:
    """Test the convenience functions for orchestration validation"""
    
    def test_validate_orchestration_request_function(self):
        """Test the main validation entry point function"""
        anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Test function",
                "prompt": "Test convenience function",
                "subagent_type": "general-purpose"
            }
        ]
        
        with patch('claude.commands.orchestration_validation_enforcer.OrchestrationValidationEnforcer'):
            # Mock the enforcer to avoid file system operations
            result = validate_orchestration_request(anti_pattern, {"test": True})
            
        # Should return a dictionary (enforcement record)
        assert isinstance(result, dict)
        
    def test_check_orchestration_pattern_function(self):
        """Test the quick validation check function"""
        valid_pattern = [
            {
                "description": "RIF-Implementer: Valid task",
                "prompt": "You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Invalid task",
                "prompt": "Handle multiple issues",
                "subagent_type": "general-purpose"
            }
        ]
        
        assert check_orchestration_pattern(valid_pattern) == True
        assert check_orchestration_pattern(anti_pattern) == False
        
    def test_get_orchestration_guidance_function(self):
        """Test the guidance generation function"""
        anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Test guidance",
                "prompt": "Test guidance generation", 
                "subagent_type": "general-purpose"
            }
        ]
        
        with patch('claude.commands.orchestration_validation_enforcer.validate_orchestration_request') as mock_validate:
            # Mock the validation result
            mock_validate.return_value = {
                "corrective_guidance": {
                    "status": "violations_detected",
                    "violations": ["Test violation"],
                    "suggestions": ["Test suggestion"],
                    "corrective_examples": [{
                        "violation_type": "Test Type",
                        "wrong_approach": "Wrong code",
                        "correct_approach": "Correct code"
                    }]
                }
            }
            
            guidance = get_orchestration_guidance(anti_pattern)
            
        assert isinstance(guidance, str)
        assert "❌ Orchestration pattern violations detected" in guidance
        assert "Test violation" in guidance
        assert "Test suggestion" in guidance
        assert "Wrong code" in guidance
        assert "Correct code" in guidance


class TestRegression:
    """Regression tests for known orchestration issues"""
    
    def test_issue_224_multi_issue_accelerator_regression(self):
        """Regression test for the specific Issue #224 Multi-Issue Accelerator pattern"""
        # This is the exact anti-pattern that caused Issue #224
        issue_224_anti_pattern = [
            {
                "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3", 
                "prompt": "You are a Multi-Issue Accelerator agent. Handle these issues in parallel: Issue #1: user auth, Issue #2: database pool, Issue #3: API validation",
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = OrchestrationPatternValidator()
        result = validator.validate_orchestration_request(issue_224_anti_pattern)
        
        # Ensure this specific pattern is caught
        assert not result.is_valid
        assert result.pattern_type == "multi-issue-single-task"
        
        # Ensure it detects both violation types
        multi_issue_violations = [v for v in result.violations if "Multi-Issue Anti-Pattern" in v]
        accelerator_violations = [v for v in result.violations if "Generic Accelerator" in v]
        
        assert len(multi_issue_violations) > 0
        assert len(accelerator_violations) > 0
        
        # Ensure low confidence score for this critical violation
        assert result.confidence_score < 0.3
        
    def test_correct_fix_for_issue_224(self):
        """Test that the correct fix for Issue #224 passes validation"""
        # This is the correct pattern that should be used instead
        correct_pattern_for_224 = [
            {
                "description": "RIF-Implementer: User authentication system",
                "prompt": "You are RIF-Implementer. Implement user authentication for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Database connection pooling",
                "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: API validation framework", 
                "prompt": "You are RIF-Validator. Validate API framework implementation for issue #3. Follow all instructions in claude/agents/rif-validator.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = OrchestrationPatternValidator()
        result = validator.validate_orchestration_request(correct_pattern_for_224)
        
        # Ensure the correct pattern passes validation
        assert result.is_valid
        assert result.pattern_type == "parallel-tasks"
        assert len(result.violations) == 0
        assert result.confidence_score == 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])