#!/usr/bin/env python3
"""
Comprehensive Tests for Workflow Loop-back Manager
Issue #53: Create workflow loop-back mechanism

This module provides comprehensive testing for the workflow loop-back system,
including loop detection, context preservation, and state rollback functionality.
"""

import pytest
import json
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from workflow_loopback_manager import (
    WorkflowLoopbackManager, 
    ValidationResult, 
    MaxLoopsExceeded,
    create_test_validation_result
)

class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(success=True, details="All tests passed")
        assert result.success == True
        assert result.details == "All tests passed"
        assert len(result.missing_requirements) == 0
        assert len(result.architectural_flaws) == 0
        assert len(result.implementation_errors) == 0
    
    def test_adding_failures(self):
        """Test adding various types of failures."""
        result = ValidationResult()
        
        # Add missing requirement
        result.add_missing_requirement("User authentication", "high")
        assert len(result.missing_requirements) == 1
        assert result.missing_requirements[0]['requirement'] == "User authentication"
        assert result.missing_requirements[0]['severity'] == "high"
        assert result.success == False
        
        # Add architectural flaw
        result.add_architectural_flaw("No connection pooling", "medium")
        assert len(result.architectural_flaws) == 1
        assert result.architectural_flaws[0]['flaw'] == "No connection pooling"
        
        # Add implementation error
        result.add_implementation_error("Null pointer exception", "src/main.py:42")
        assert len(result.implementation_errors) == 1
        assert result.implementation_errors[0]['error'] == "Null pointer exception"
        assert result.implementation_errors[0]['location'] == "src/main.py:42"
        
        # Add test failure
        result.add_test_failure("test_login", "Expected 200, got 500")
        assert len(result.test_failures) == 1
        
        # Add quality gate failure
        result.add_quality_gate_failure("coverage", "80%", "65%")
        assert len(result.quality_gate_failures) == 1
    
    def test_loop_back_priority(self):
        """Test loop-back priority determination."""
        result = ValidationResult()
        
        # Test with missing requirements (highest priority)
        result.add_missing_requirement("API spec", "high")
        priorities = result.get_loop_back_priority()
        assert len(priorities) > 0
        assert priorities[0][0] == 'analyzing'
        assert 'Missing requirements' in priorities[0][1]
        
        # Add architectural flaw
        result.add_architectural_flaw("Poor design", "medium")
        priorities = result.get_loop_back_priority()
        # Should still prioritize requirements first, then architecture
        assert priorities[0][0] == 'analyzing'
        assert priorities[1][0] == 'architecting'
        
        # Add implementation error
        result.add_implementation_error("Bug in code", "src/app.py")
        priorities = result.get_loop_back_priority()
        assert len(priorities) == 3  # All three types
        assert priorities[2][0] == 'implementing'


class TestWorkflowLoopbackManager:
    """Test cases for WorkflowLoopbackManager class."""
    
    @pytest.fixture
    def loop_manager(self):
        """Create a loop manager for testing."""
        return WorkflowLoopbackManager(max_loops=3)
    
    @pytest.fixture
    def test_context(self):
        """Create test context data."""
        return {
            'workflow_type': 'test',
            'github_issues': [53],
            'complexity': 'medium',
            'agent_performance': {'RIF-Validator': 0.85},
            'patterns_discovered': ['retry_pattern'],
            'validation_results': None
        }
    
    def test_loop_manager_creation(self, loop_manager):
        """Test loop manager initialization."""
        assert loop_manager.max_loops == 3
        assert isinstance(loop_manager.loop_counts, dict)
        assert isinstance(loop_manager.context_snapshots, dict)
        assert isinstance(loop_manager.loop_history, list)
        assert isinstance(loop_manager.metrics, dict)
    
    def test_should_loop_back_success(self, loop_manager):
        """Test loop-back decision with successful validation."""
        success_result = ValidationResult(success=True)
        decision = loop_manager.should_loop_back(success_result, 'validating')
        assert decision is None
    
    def test_should_loop_back_failure(self, loop_manager):
        """Test loop-back decision with validation failures."""
        # Create validation result with missing requirements
        failure_result = ValidationResult(success=False)
        failure_result.add_missing_requirement("Auth spec", "high")
        
        decision = loop_manager.should_loop_back(failure_result, 'validating')
        assert decision is not None
        target_state, reason = decision
        assert target_state == 'analyzing'
        assert 'Missing requirements' in reason
    
    def test_should_loop_back_with_limits(self, loop_manager):
        """Test loop-back decision respecting loop limits."""
        failure_result = ValidationResult(success=False)
        failure_result.add_implementation_error("Bug", "main.py")
        
        # First few should work
        for i in range(3):
            decision = loop_manager.should_loop_back(failure_result, 'validating')
            assert decision is not None
            target_state, reason = decision
            loop_manager.loop_counts[f'validating->{target_state}'] += 1
        
        # Fourth should be blocked by loop limit
        decision = loop_manager.should_loop_back(failure_result, 'validating')
        assert decision is None  # All loop-backs would exceed limits
    
    def test_execute_loopback_success(self, loop_manager, test_context):
        """Test successful loop-back execution."""
        # Execute loop-back
        updated_context = loop_manager.execute_loopback(
            'validating', 'implementing', test_context,
            "Test loop-back execution"
        )
        
        # Verify loop counting
        assert loop_manager.loop_counts['validating->implementing'] == 1
        
        # Verify context preservation
        assert 'loop_back_metadata' in updated_context
        assert len(updated_context['loop_back_metadata']) == 1
        
        metadata = updated_context['loop_back_metadata'][0]
        assert metadata['loop_back_reason'] == "Test loop-back execution"
        assert metadata['loop_from_state'] == 'validating'
        assert metadata['previous_attempts'] == 1
        
        # Verify retry count compatibility
        assert updated_context['retry_count'] == 1
        
        # Verify rollback history
        assert 'rollback_history' in updated_context
        assert len(updated_context['rollback_history']) == 1
        assert updated_context['rollback_history'][0]['target_state'] == 'implementing'
    
    def test_execute_loopback_max_loops_exceeded(self, loop_manager, test_context):
        """Test loop-back execution with max loops exceeded."""
        # Set loop count to maximum
        loop_manager.loop_counts['validating->implementing'] = 3
        
        # Should raise MaxLoopsExceeded
        with pytest.raises(MaxLoopsExceeded) as exc_info:
            loop_manager.execute_loopback(
                'validating', 'implementing', test_context,
                "This should fail"
            )
        
        assert "Maximum loops (3) exceeded" in str(exc_info.value)
        assert "validating->implementing" in str(exc_info.value)
    
    def test_context_snapshot_creation(self, loop_manager, test_context):
        """Test context snapshot creation and retrieval."""
        # Create snapshot
        snapshot_id = loop_manager._create_context_snapshot(
            test_context, 'validating', 'implementing'
        )
        
        assert snapshot_id is not None
        assert len(snapshot_id) > 0
        
        # Retrieve snapshot
        snapshot = loop_manager.get_context_snapshot(snapshot_id)
        assert snapshot is not None
        assert snapshot['from_state'] == 'validating'
        assert snapshot['to_state'] == 'implementing'
        assert 'checksum' in snapshot
        assert 'context' in snapshot
        
        # Verify context preservation
        assert snapshot['context']['workflow_type'] == 'test'
        assert snapshot['context']['github_issues'] == [53]
    
    def test_context_preservation(self, loop_manager, test_context):
        """Test context preservation logic."""
        # Add validation results to context
        validation_result = create_test_validation_result()
        test_context['validation_results'] = validation_result
        test_context['patterns_discovered'] = ['pattern1', 'pattern2']
        
        preserved_context = loop_manager._preserve_context(
            test_context, 'validating', 'implementing', 
            "Context preservation test"
        )
        
        # Verify original data preserved
        assert preserved_context['workflow_type'] == 'test'
        assert preserved_context['github_issues'] == [53]
        
        # Verify loop metadata added
        assert 'loop_back_metadata' in preserved_context
        metadata = preserved_context['loop_back_metadata'][0]
        assert metadata['loop_back_reason'] == "Context preservation test"
        assert metadata['loop_from_state'] == 'validating'
        
        # Verify accumulated learning preserved
        assert 'accumulated_patterns_discovered' in metadata
        assert metadata['accumulated_patterns_discovered'] == ['pattern1', 'pattern2']
        
        # Verify validation feedback preserved
        assert 'validation_feedback' in metadata
        assert metadata['validation_feedback'] == validation_result
    
    def test_state_rollback(self, loop_manager, test_context):
        """Test state rollback execution."""
        # Add state-specific context that should be cleaned
        test_context['test_results'] = {'passed': 10, 'failed': 2}
        test_context['code_artifacts'] = ['file1.py', 'file2.py']
        
        rollback_result = loop_manager._rollback_to_state('implementing', test_context)
        
        assert rollback_result['success'] == True
        assert rollback_result['target_state'] == 'implementing'
        assert rollback_result['integrity_validated'] == True
        
        # Verify context cleaning (test_results should be removed for implementing state)
        cleaned_context = rollback_result['context']
        assert 'test_results' not in cleaned_context  # Should be removed
        assert 'workflow_type' in cleaned_context  # Should be preserved
        
        # Verify rollback history added
        assert 'rollback_history' in cleaned_context
        assert len(cleaned_context['rollback_history']) == 1
        assert cleaned_context['rollback_history'][0]['target_state'] == 'implementing'
    
    def test_context_integrity_validation(self, loop_manager):
        """Test context integrity validation."""
        # Valid context
        valid_context = {
            'workflow_type': 'test',
            'github_issues': [1, 2, 3],
            'complexity': 'medium'
        }
        assert loop_manager._validate_context_integrity(valid_context) == True
        
        # Invalid context - missing required field
        invalid_context = {
            'github_issues': [1, 2, 3],
            'complexity': 'medium'
            # Missing workflow_type
        }
        assert loop_manager._validate_context_integrity(invalid_context) == False
        
        # Invalid context - not JSON serializable
        invalid_context_2 = {
            'workflow_type': 'test',
            'bad_data': set([1, 2, 3])  # Sets are not JSON serializable
        }
        assert loop_manager._validate_context_integrity(invalid_context_2) == False
    
    def test_context_checksum_calculation(self, loop_manager):
        """Test context checksum calculation."""
        context1 = {'a': 1, 'b': 2, 'c': 3}
        context2 = {'c': 3, 'a': 1, 'b': 2}  # Different order, same content
        context3 = {'a': 1, 'b': 2, 'c': 4}  # Different content
        
        checksum1 = loop_manager._calculate_context_checksum(context1)
        checksum2 = loop_manager._calculate_context_checksum(context2)
        checksum3 = loop_manager._calculate_context_checksum(context3)
        
        # Same content should produce same checksum regardless of order
        assert checksum1 == checksum2
        
        # Different content should produce different checksum
        assert checksum1 != checksum3
        
        # Checksums should be hex strings
        assert isinstance(checksum1, str)
        assert len(checksum1) == 64  # SHA-256 hex length
    
    def test_loop_statistics(self, loop_manager, test_context):
        """Test loop statistics collection."""
        # Execute several loop-backs
        for i in range(2):
            try:
                loop_manager.execute_loopback(
                    'validating', 'implementing', test_context,
                    f"Test loop {i+1}"
                )
            except MaxLoopsExceeded:
                pass
        
        # Get statistics
        stats = loop_manager.get_loop_statistics()
        
        assert stats['total_loop_backs'] == 2
        assert stats['successful_recoveries'] == 2
        assert stats['success_rate'] == 1.0
        assert stats['max_loops_exceeded_count'] == 0
        assert stats['context_preservation_failures'] == 0
        assert 'avg_rollback_time_ms' in stats
        assert 'loop_patterns' in stats
        assert 'current_loop_counts' in stats
        assert stats['max_loops_limit'] == 3
        
        # Check loop patterns
        assert 'validating_to_implementing' in stats['loop_patterns']
        assert stats['loop_patterns']['validating_to_implementing'] == 2
    
    def test_loop_count_reset(self, loop_manager, test_context):
        """Test loop count reset functionality."""
        # Execute loop-back to create count
        loop_manager.execute_loopback(
            'validating', 'implementing', test_context,
            "Test for reset"
        )
        
        assert loop_manager.loop_counts['validating->implementing'] == 1
        
        # Reset specific loop count
        loop_manager.reset_loop_counts('validating->implementing')
        assert 'validating->implementing' not in loop_manager.loop_counts
        
        # Execute another loop-back to create multiple counts
        loop_manager.execute_loopback(
            'validating', 'implementing', test_context, "Test 1"
        )
        loop_manager.execute_loopback(
            'implementing', 'analyzing', test_context, "Test 2"
        )
        
        assert len(loop_manager.loop_counts) == 2
        
        # Reset all loop counts
        loop_manager.reset_loop_counts()
        assert len(loop_manager.loop_counts) == 0
    
    def test_snapshot_cleanup(self, loop_manager, test_context):
        """Test context snapshot cleanup."""
        # Create multiple snapshots
        snapshot_ids = []
        for i in range(5):
            snapshot_id = loop_manager._create_context_snapshot(
                test_context, 'state1', 'state2'
            )
            snapshot_ids.append(snapshot_id)
        
        assert len(loop_manager.context_snapshots) == 5
        
        # Clean up with 0 max age (should remove all)
        loop_manager.cleanup_old_snapshots(max_age_hours=0)
        
        # All snapshots should be removed (they're all "old")
        assert len(loop_manager.context_snapshots) == 0
    
    def test_memory_management(self, loop_manager, test_context):
        """Test memory management with many snapshots."""
        # Create many snapshots to trigger automatic cleanup
        for i in range(150):  # More than the 100 snapshot limit
            loop_manager._create_context_snapshot(
                test_context, f'state{i}', f'target{i}'
            )
        
        # Should automatically limit to around 100 snapshots
        assert len(loop_manager.context_snapshots) <= 100


class TestIntegrationWithDynamicOrchestrator:
    """Integration tests with DynamicOrchestrator."""
    
    @patch('workflow_loopback_manager.OrchestratorStatePersistence')
    def test_orchestrator_integration(self, mock_persistence):
        """Test integration between loop-back manager and orchestrator."""
        # This test would require importing and testing with DynamicOrchestrator
        # For now, we'll test the interface compatibility
        
        loop_manager = WorkflowLoopbackManager()
        
        # Test that the interface methods exist and work as expected
        assert hasattr(loop_manager, 'should_loop_back')
        assert hasattr(loop_manager, 'execute_loopback')
        assert hasattr(loop_manager, 'get_loop_statistics')
        
        # Test with mock validation result
        validation_result = ValidationResult(success=False)
        validation_result.add_implementation_error("Test error", "test.py")
        
        decision = loop_manager.should_loop_back(validation_result, 'validating')
        assert decision is not None
        assert decision[0] == 'implementing'


@pytest.fixture
def performance_test_context():
    """Create context for performance testing."""
    return {
        'workflow_type': 'performance_test',
        'github_issues': list(range(10)),
        'complexity': 'high',
        'large_data': {
            'patterns': ['pattern_' + str(i) for i in range(100)],
            'metrics': {f'metric_{i}': i * 1.5 for i in range(50)},
            'history': [{'event': f'event_{i}', 'time': time.time()} for i in range(200)]
        }
    }


class TestPerformance:
    """Performance tests for loop-back system."""
    
    def test_loopback_execution_performance(self, performance_test_context):
        """Test that loop-back execution meets performance requirements."""
        loop_manager = WorkflowLoopbackManager()
        
        start_time = time.time()
        
        # Execute loop-back
        updated_context = loop_manager.execute_loopback(
            'validating', 'implementing', performance_test_context,
            "Performance test loop-back"
        )
        
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should complete within 50ms target (based on planning decisions)
        assert execution_time_ms < 50, f"Loop-back took {execution_time_ms:.2f}ms, target: <50ms"
        
        # Verify context was preserved correctly
        assert 'loop_back_metadata' in updated_context
        assert updated_context['workflow_type'] == 'performance_test'
        assert len(updated_context['github_issues']) == 10
    
    def test_context_snapshot_performance(self, performance_test_context):
        """Test context snapshot creation performance."""
        loop_manager = WorkflowLoopbackManager()
        
        start_time = time.time()
        
        # Create multiple snapshots
        snapshot_ids = []
        for i in range(10):
            snapshot_id = loop_manager._create_context_snapshot(
                performance_test_context, f'state{i}', f'target{i}'
            )
            snapshot_ids.append(snapshot_id)
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / 10
        
        # Each snapshot should be created quickly
        assert avg_time_ms < 10, f"Average snapshot creation: {avg_time_ms:.2f}ms, target: <10ms"
        
        # Verify all snapshots were created correctly
        assert len(snapshot_ids) == 10
        for snapshot_id in snapshot_ids:
            snapshot = loop_manager.get_context_snapshot(snapshot_id)
            assert snapshot is not None
            assert 'checksum' in snapshot


def test_create_test_validation_result():
    """Test the utility function for creating test validation results."""
    result = create_test_validation_result()
    
    assert result.success == False
    assert len(result.missing_requirements) == 1
    assert len(result.architectural_flaws) == 1
    assert len(result.implementation_errors) == 1
    assert len(result.test_failures) == 1
    assert len(result.quality_gate_failures) == 1
    
    # Test loop-back priorities
    priorities = result.get_loop_back_priority()
    assert len(priorities) == 3  # Should have all three types
    assert priorities[0][0] == 'analyzing'  # Missing requirements first
    assert priorities[1][0] == 'architecting'  # Architecture flaws second
    assert priorities[2][0] == 'implementing'  # Implementation errors third


if __name__ == "__main__":
    # Run basic functionality test
    print("🧪 Running Workflow Loop-back Manager Tests")
    
    # Test ValidationResult
    print("1. Testing ValidationResult...")
    result = create_test_validation_result()
    priorities = result.get_loop_back_priority()
    print(f"   ✅ Loop-back priorities: {len(priorities)} detected")
    
    # Test WorkflowLoopbackManager
    print("2. Testing WorkflowLoopbackManager...")
    manager = WorkflowLoopbackManager(max_loops=2)
    
    test_context = {
        'workflow_type': 'test',
        'github_issues': [53],
        'complexity': 'medium'
    }
    
    # Test loop-back decision
    decision = manager.should_loop_back(result, 'validating')
    if decision:
        target_state, reason = decision
        print(f"   ✅ Loop-back decision: {target_state} ({reason})")
        
        # Test execution
        try:
            updated_context = manager.execute_loopback(
                'validating', target_state, test_context, reason
            )
            print(f"   ✅ Loop-back executed successfully")
            print(f"   Context keys: {len(updated_context)}")
        except Exception as e:
            print(f"   ❌ Loop-back execution failed: {e}")
    
    # Test statistics
    print("3. Testing statistics...")
    stats = manager.get_loop_statistics()
    print(f"   Total loop-backs: {stats['total_loop_backs']}")
    print(f"   Success rate: {stats['success_rate']:.2f}")
    print(f"   Avg rollback time: {stats['avg_rollback_time_ms']:.1f}ms")
    
    print("✅ All tests completed!")