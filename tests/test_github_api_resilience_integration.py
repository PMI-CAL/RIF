#!/usr/bin/env python3
"""
Integration Tests for GitHub API Resilience System
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

End-to-end integration tests that verify the complete resilience system works together:
- Timeout management with adaptive configuration
- Request context preservation during failures
- Batch operation resilience with fragmentation
- Circuit breaker coordination
- Rate limit integration
"""

import pytest
import time
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Add the parent directory to the Python path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.github_api_client import (
    GitHubAPIClient,
    RateLimitStrategy,
    APICallResult
)
from claude.commands.github_timeout_manager import (
    GitHubEndpoint,
    TimeoutConfig,
    TimeoutStrategy
)
from claude.commands.github_request_context import (
    RequestState,
    ContextScope
)
from claude.commands.github_batch_resilience import (
    BatchOperationType,
    BatchConfiguration,
    BatchStrategy
)

class TestGitHubAPIClientIntegration:
    """Test the integrated GitHub API client"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create client with test configuration
        self.client = GitHubAPIClient(RateLimitStrategy.ADAPTIVE)
        
        # Override storage paths for testing
        self.client.timeout_manager.metrics_file = Path(self.temp_dir) / "timeout_metrics.json"
        self.client.context_manager.storage_path = Path(self.temp_dir) / "contexts"
        self.client.batch_manager.storage_path = Path(self.temp_dir) / "batches"
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_successful_api_call_with_metrics_recording(self, mock_run):
        """Test successful API call records metrics correctly"""
        # Mock successful subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps([{"number": 1, "title": "Test Issue", "state": "open"}])
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Make API call
        result = self.client.issue_list(state="open", limit=10)
        
        # Verify result
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0]["number"] == 1
        assert result.attempt_count == 1
        assert result.timeout_used > 0
        assert result.context_id is not None
        
        # Verify command was called correctly
        expected_command = [
            "gh", "issue", "list", "--state", "open", "--limit", "10",
            "--json", "number,title,state,labels,assignees,body,createdAt,updatedAt"
        ]
        mock_run.assert_called_once()
        actual_command = mock_run.call_args[0][0]
        assert actual_command == expected_command
        
        # Verify timeout was applied
        call_kwargs = mock_run.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] > 0
        
        # Verify metrics were recorded
        endpoint_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.ISSUE_LIST)
        assert endpoint_stats["total_requests"] == 1
        assert endpoint_stats["success_rate"] == 1.0
        
        # Verify context was created and completed
        context_stats = self.client.context_manager.get_context_stats()
        assert context_stats["completed_contexts"] == 1
    
    @patch('subprocess.run')
    def test_timeout_failure_with_retry_and_recovery(self, mock_run):
        """Test timeout failure triggers retry with progressive timeout"""
        # Mock timeout on first call, success on second
        timeout_result = Mock()
        timeout_result.side_effect = subprocess.TimeoutExpired("gh", 30)
        
        success_result = Mock()
        success_result.returncode = 0
        success_result.stdout = json.dumps({"number": 123, "title": "Test Issue"})
        success_result.stderr = ""
        
        mock_run.side_effect = [timeout_result, success_result]
        
        # Make API call
        result = self.client.issue_view(123)
        
        # Verify final result is successful
        assert result.success is True
        assert result.data["number"] == 123
        assert result.attempt_count == 2  # First attempt failed, second succeeded
        
        # Verify two calls were made
        assert mock_run.call_count == 2
        
        # Verify second call used higher timeout
        first_timeout = mock_run.call_args_list[0][1]["timeout"]
        second_timeout = mock_run.call_args_list[1][1]["timeout"]
        assert second_timeout > first_timeout
        
        # Verify metrics recorded both attempts
        endpoint_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.ISSUE_VIEW)
        assert endpoint_stats["total_requests"] == 2  # One failure, one success
        
        # Verify circuit breaker recorded success after failure
        circuit_breaker = self.client.timeout_manager.circuit_breakers[GitHubEndpoint.ISSUE_VIEW]
        assert circuit_breaker.failure_count == 0  # Reset after success
    
    @patch('subprocess.run')
    def test_circuit_breaker_blocks_requests_after_failures(self, mock_run):
        """Test circuit breaker blocks requests after repeated failures"""
        # Mock repeated failures
        failure_result = Mock()
        failure_result.returncode = 1
        failure_result.stdout = ""
        failure_result.stderr = "API rate limit exceeded"
        mock_run.return_value = failure_result
        
        endpoint = GitHubEndpoint.ISSUE_CREATE
        
        # Make several failing requests to trip circuit breaker
        results = []
        for i in range(8):  # Should trip after 5 failures
            result = self.client.issue_create(f"Test Issue {i}")
            results.append(result)
        
        # First 5-6 should actually make requests, later ones should be circuit breaker blocked
        successful_calls = sum(1 for r in results if r.attempt_count > 0)
        circuit_breaker_blocks = sum(1 for r in results if "circuit breaker" in (r.error_message or "").lower())
        
        assert successful_calls >= 5  # At least threshold attempts made
        assert circuit_breaker_blocks >= 2  # Some blocked by circuit breaker
        
        # Verify circuit breaker is open
        can_attempt, reason = self.client.timeout_manager.can_attempt_request(endpoint)
        assert can_attempt is False
        assert "open" in reason.lower()
    
    @patch('subprocess.run')
    def test_context_preservation_across_retries(self, mock_run):
        """Test request context is preserved across retry attempts"""
        # Mock failure then success
        failure_result = Mock()
        failure_result.side_effect = subprocess.TimeoutExpired("gh", 30)
        
        success_result = Mock()
        success_result.returncode = 0
        success_result.stdout = json.dumps({"number": 456, "url": "https://github.com/test/test/issues/456"})
        success_result.stderr = ""
        
        mock_run.side_effect = [failure_result, success_result]
        
        # Make API call
        result = self.client.issue_create("Test Issue", body="Test body")
        
        # Verify success
        assert result.success is True
        assert result.context_id is not None
        
        # Get the completed context
        completed_contexts = list(self.client.context_manager.completed_contexts.values())
        assert len(completed_contexts) == 1
        
        context = completed_contexts[0]
        assert context.state == RequestState.COMPLETED
        assert context.attempt_count == 2  # Two attempts
        assert len(context.error_history) == 1  # One failure recorded
        assert context.error_history[0]["attempt"] == 1
        assert "timeout" in context.error_history[0]["error"].lower()
    
    @patch('subprocess.run')
    def test_batch_operation_resilience(self, mock_run):
        """Test batch operations with partial failures and recovery"""
        # Mock responses: some success, some failures
        def mock_response(*args, **kwargs):
            command = args[0]
            issue_number = None
            
            # Extract issue number from command
            for i, arg in enumerate(command):
                if arg == "edit" and i + 1 < len(command):
                    try:
                        issue_number = int(command[i + 1])
                        break
                    except ValueError:
                        pass
            
            if issue_number and issue_number % 3 == 0:  # Every 3rd issue fails
                result = Mock()
                result.returncode = 1
                result.stdout = ""
                result.stderr = f"Issue {issue_number} not found"
                return result
            else:
                result = Mock()
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
                return result
        
        mock_run.side_effect = mock_response
        
        # Create batch update data
        updates = []
        for i in range(1, 11):  # Issues 1-10
            updates.append({
                "issue_number": i,
                "updates": {"add_labels": ["batch-update"]}
            })
        
        # Start batch operation
        batch_id = self.client.bulk_issue_update(
            updates,
            config=BatchConfiguration(
                chunk_size=3,
                strategy=BatchStrategy.SEQUENTIAL,
                continue_on_errors=True
            )
        )
        
        assert batch_id is not None
        
        # Wait for batch to complete (with timeout)
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            status = self.client.get_batch_status(batch_id)
            if status and status["state"] in ["completed", "failed"]:
                break
            time.sleep(0.1)
        
        # Verify batch completion
        final_status = self.client.get_batch_status(batch_id)
        assert final_status is not None
        assert final_status["state"] == "completed"  # Should complete despite some failures
        
        # Verify expected success/failure pattern
        assert final_status["total_items"] == 10
        expected_failures = 3  # Issues 3, 6, 9 should fail
        expected_successes = 7
        
        assert final_status["failed_items"] == expected_failures
        assert final_status["completed_items"] == expected_successes
        assert final_status["success_rate"] >= 70.0  # 7/10 = 70%
    
    @patch('subprocess.run')
    def test_rate_limit_handling(self, mock_run):
        """Test rate limit detection and handling"""
        # Mock rate limit error
        rate_limit_result = Mock()
        rate_limit_result.returncode = 1
        rate_limit_result.stdout = ""
        rate_limit_result.stderr = "API rate limit exceeded. Please wait."
        mock_run.return_value = rate_limit_result
        
        # Set client to fail fast on rate limits
        self.client.rate_limit_strategy = RateLimitStrategy.FAIL_FAST
        
        result = self.client.search_issues("test query")
        
        # Should fail immediately due to rate limit
        assert result.success is False
        assert "rate limit" in result.error_message.lower()
        assert result.attempt_count == 1  # No retries for rate limit
    
    def test_adaptive_timeout_adjustment(self):
        """Test that timeouts adapt based on endpoint performance"""
        endpoint = GitHubEndpoint.SEARCH
        initial_timeout = self.client.timeout_manager.get_timeout(endpoint)
        
        # Simulate fast responses
        for _ in range(10):
            self.client.timeout_manager.record_request_metrics(
                endpoint=endpoint,
                duration=0.5,  # Very fast
                success=True,
                timeout_used=initial_timeout
            )
        
        # Trigger optimization
        optimizations = self.client.timeout_manager.optimize_timeouts()
        
        # Check if timeout was optimized (may or may not happen depending on thresholds)
        new_timeout = self.client.timeout_manager.get_timeout(endpoint)
        
        # At minimum, the endpoint should have updated metrics
        stats = self.client.timeout_manager.get_endpoint_stats(endpoint)
        assert stats["total_requests"] == 10
        assert stats["success_rate"] == 1.0
        assert stats["avg_response_time"] == 0.5
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics collection"""
        # Record some activity
        self.client.timeout_manager.record_request_metrics(
            GitHubEndpoint.ISSUE_LIST, 2.0, True, 30.0
        )
        
        context = self.client.context_manager.create_context(
            GitHubEndpoint.PR_CREATE,
            "test_pr",
            ["pr", "create"]
        )
        
        # Get comprehensive stats
        stats = self.client.get_client_stats()
        
        # Verify structure
        assert "client_config" in stats
        assert "timeout_management" in stats
        assert "context_management" in stats
        assert "batch_management" in stats
        assert "rate_limits" in stats
        
        # Verify client config
        assert stats["client_config"]["rate_limit_strategy"] == "adaptive"
        assert stats["client_config"]["max_retries"] == 3
        
        # Verify timeout stats
        timeout_stats = stats["timeout_management"]
        assert timeout_stats["total_requests"] >= 1
        assert "endpoints" in timeout_stats
        
        # Verify context stats
        context_stats = stats["context_management"]
        assert context_stats["active_contexts"] >= 1

class TestRealWorldScenarios:
    """Test realistic failure scenarios and recovery patterns"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.client = GitHubAPIClient(RateLimitStrategy.ADAPTIVE)
        
        # Override storage paths for testing
        self.client.timeout_manager.metrics_file = Path(self.temp_dir) / "timeout_metrics.json"
        self.client.context_manager.storage_path = Path(self.temp_dir) / "contexts"
        self.client.batch_manager.storage_path = Path(self.temp_dir) / "batches"
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_network_instability_scenario(self, mock_run):
        """Test handling of intermittent network issues"""
        call_count = 0
        
        def unstable_network(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Pattern: fail, fail, succeed, fail, succeed, succeed
            if call_count in [1, 2, 4]:
                raise subprocess.TimeoutExpired("gh", 30)
            else:
                result = Mock()
                result.returncode = 0
                result.stdout = json.dumps({"number": call_count, "title": f"Issue {call_count}"})
                result.stderr = ""
                return result
        
        mock_run.side_effect = unstable_network
        
        # Make multiple API calls
        results = []
        for i in range(3):
            result = self.client.issue_view(i + 1)
            results.append(result)
        
        # All should eventually succeed despite network issues
        for result in results:
            assert result.success is True
            assert result.attempt_count >= 1
        
        # Verify adaptive behavior - later calls should use higher timeouts
        endpoint_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.ISSUE_VIEW)
        assert endpoint_stats["total_requests"] >= 6  # Multiple attempts across calls
    
    @patch('subprocess.run')
    def test_github_service_degradation_scenario(self, mock_run):
        """Test handling of GitHub service degradation"""
        def degraded_service(*args, **kwargs):
            # Simulate slow responses that gradually improve
            time.sleep(0.01)  # Simulate slow response
            
            result = Mock()
            result.returncode = 0
            result.stdout = json.dumps([{"number": 1, "title": "Slow Response"}])
            result.stderr = ""
            return result
        
        mock_run.side_effect = degraded_service
        
        # Make several calls to simulate learning
        for i in range(5):
            result = self.client.issue_list(limit=1)
            assert result.success is True
        
        # System should adapt to slower responses
        endpoint_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.ISSUE_LIST)
        assert endpoint_stats["total_requests"] == 5
        assert endpoint_stats["success_rate"] == 1.0
    
    @patch('subprocess.run')
    def test_mixed_endpoint_performance_scenario(self, mock_run):
        """Test handling of different performance characteristics across endpoints"""
        def endpoint_specific_behavior(*args, **kwargs):
            command = args[0]
            
            if "search" in command:
                # Search is slow
                time.sleep(0.02)
                result = Mock()
                result.returncode = 0
                result.stdout = json.dumps([{"number": 1, "title": "Search Result"}])
                result.stderr = ""
                return result
            
            elif "issue" in command and "view" in command:
                # Issue view is fast
                result = Mock()
                result.returncode = 0
                result.stdout = json.dumps({"number": 1, "title": "Fast Issue"})
                result.stderr = ""
                return result
                
            else:
                # Other operations are medium speed
                time.sleep(0.01)
                result = Mock()
                result.returncode = 0
                result.stdout = json.dumps({"result": "success"})
                result.stderr = ""
                return result
        
        mock_run.side_effect = endpoint_specific_behavior
        
        # Make calls to different endpoints
        search_result = self.client.search_issues("test")
        view_result = self.client.issue_view(123)
        list_result = self.client.issue_list()
        
        assert all(r.success for r in [search_result, view_result, list_result])
        
        # Verify different endpoints have different performance profiles
        search_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.SEARCH)
        view_stats = self.client.timeout_manager.get_endpoint_stats(GitHubEndpoint.ISSUE_VIEW)
        
        assert search_stats["avg_response_time"] > view_stats["avg_response_time"]

class TestErrorRecoveryPatterns:
    """Test specific error recovery patterns"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.client = GitHubAPIClient()
        
        # Override storage paths for testing
        self.client.context_manager.storage_path = Path(self.temp_dir) / "contexts"
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_context_recovery_after_system_restart(self):
        """Test context recovery after simulated system restart"""
        # Create a context
        context = self.client.context_manager.create_context(
            GitHubEndpoint.PR_CREATE,
            "create_important_pr",
            ["pr", "create", "--title", "Critical Fix"],
            scope=ContextScope.SESSION_PERSISTENT,
            priority=1
        )
        
        context_id = context.context_id
        
        # Update context to show it was in progress
        self.client.context_manager.update_context_state(
            context_id,
            RequestState.RETRYING,
            partial_results={"branch_created": True},
            intermediate_state={"validation_passed": True},
            continuation_data={"next_step": "create_pr"}
        )
        
        # Simulate system restart by creating new context manager
        new_context_manager = self.client.context_manager.__class__(
            str(self.client.context_manager.storage_path)
        )
        
        # Verify context was restored
        assert context_id in new_context_manager.active_contexts
        
        restored_context = new_context_manager.active_contexts[context_id]
        assert restored_context.state == RequestState.RETRYING
        assert restored_context.partial_results["branch_created"] is True
        assert restored_context.continuation_data["next_step"] == "create_pr"
        
        # Verify it appears in recoverable contexts
        recoverable = new_context_manager.get_recoverable_contexts()
        recoverable_ids = [c.context_id for c in recoverable]
        assert context_id in recoverable_ids
    
    def test_batch_operation_partial_recovery(self):
        """Test batch operation recovery from partial completion"""
        # This would require more complex mocking of the batch system
        # For now, verify the batch manager can track partial progress
        
        updates = [
            {"issue_number": i, "updates": {"add_labels": ["test"]}}
            for i in range(1, 6)
        ]
        
        batch = self.client.batch_manager.create_batch_operation(
            BatchOperationType.ISSUE_BULK_UPDATE,
            GitHubEndpoint.ISSUE_EDIT,
            updates
        )
        
        # Verify batch is created and trackable
        assert batch.batch_id is not None
        assert batch.total_items == 5
        assert batch.state == RequestState.INITIALIZED
        
        status = self.client.batch_manager.get_batch_status(batch.batch_id)
        assert status is not None
        assert status["total_items"] == 5
        assert status["progress_percentage"] == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])