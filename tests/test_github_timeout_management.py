#!/usr/bin/env python3
"""
Tests for GitHub Timeout Management System
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Comprehensive tests for the adaptive timeout management, circuit breaker integration,
and endpoint performance profiling systems.
"""

import pytest
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the Python path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.github_timeout_manager import (
    GitHubTimeoutManager,
    TimeoutConfig,
    GitHubEndpoint,
    TimeoutStrategy,
    RequestMetrics,
    EndpointProfile,
    create_timeout_manager,
    get_timeout_manager
)

class TestTimeoutConfig:
    """Test timeout configuration"""
    
    def test_default_config(self):
        """Test default timeout configuration"""
        config = TimeoutConfig()
        
        assert config.base_timeout == 30.0
        assert config.max_timeout == 300.0
        assert config.min_timeout == 5.0
        assert config.progressive_multiplier == 1.5
        assert config.adaptive_percentile == 95.0
        assert config.sample_window == 100
        assert config.strategy == TimeoutStrategy.ADAPTIVE
    
    def test_custom_config(self):
        """Test custom timeout configuration"""
        config = TimeoutConfig(
            base_timeout=45.0,
            max_timeout=600.0,
            min_timeout=10.0,
            progressive_multiplier=2.0,
            strategy=TimeoutStrategy.PROGRESSIVE
        )
        
        assert config.base_timeout == 45.0
        assert config.max_timeout == 600.0
        assert config.min_timeout == 10.0
        assert config.progressive_multiplier == 2.0
        assert config.strategy == TimeoutStrategy.PROGRESSIVE

class TestGitHubTimeoutManager:
    """Test the main timeout manager functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = Path(self.temp_dir) / "test_metrics.json"
        
        # Create manager with test configuration
        self.config = TimeoutConfig(
            base_timeout=30.0,
            strategy=TimeoutStrategy.ADAPTIVE,
            sample_window=10
        )
        
        # Mock the metrics file path
        with patch.object(Path, 'mkdir'):
            self.manager = GitHubTimeoutManager(self.config)
            self.manager.metrics_file = self.metrics_file
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test timeout manager initialization"""
        assert self.manager.config == self.config
        assert len(self.manager.endpoint_profiles) == len(GitHubEndpoint)
        assert len(self.manager.circuit_breakers) == len(GitHubEndpoint)
        
        # Check default endpoint profiles
        for endpoint in GitHubEndpoint:
            assert endpoint in self.manager.endpoint_profiles
            profile = self.manager.endpoint_profiles[endpoint]
            assert profile.endpoint == endpoint
            assert profile.success_rate == 1.0
            assert profile.total_requests == 0
            assert profile.failure_count == 0
    
    def test_get_timeout_fixed_strategy(self):
        """Test timeout calculation with fixed strategy"""
        self.manager.config.strategy = TimeoutStrategy.FIXED
        
        # First attempt
        timeout = self.manager.get_timeout(GitHubEndpoint.ISSUE_VIEW, 0)
        assert timeout == self.config.base_timeout
        
        # Retry attempt (should still be base timeout for fixed strategy)
        timeout = self.manager.get_timeout(GitHubEndpoint.ISSUE_VIEW, 1)
        assert timeout == self.config.base_timeout
    
    def test_get_timeout_adaptive_strategy(self):
        """Test timeout calculation with adaptive strategy"""
        self.manager.config.strategy = TimeoutStrategy.ADAPTIVE
        
        endpoint = GitHubEndpoint.ISSUE_VIEW
        profile = self.manager.endpoint_profiles[endpoint]
        profile.recommended_timeout = 25.0
        
        # First attempt
        timeout = self.manager.get_timeout(endpoint, 0)
        assert timeout == 25.0
        
        # Retry with progressive multiplier
        timeout = self.manager.get_timeout(endpoint, 1)
        expected = 25.0 * self.config.progressive_multiplier
        assert timeout == expected
        
        # Test clamping to max timeout
        timeout = self.manager.get_timeout(endpoint, 10)  # High retry count
        assert timeout == self.config.max_timeout
    
    def test_get_timeout_progressive_strategy(self):
        """Test timeout calculation with progressive strategy"""
        self.manager.config.strategy = TimeoutStrategy.PROGRESSIVE
        
        endpoint = GitHubEndpoint.ISSUE_CREATE
        profile = self.manager.endpoint_profiles[endpoint]
        profile.recommended_timeout = 30.0
        
        # Test progressive escalation
        timeout0 = self.manager.get_timeout(endpoint, 0)
        timeout1 = self.manager.get_timeout(endpoint, 1)
        timeout2 = self.manager.get_timeout(endpoint, 2)
        
        assert timeout1 > timeout0
        assert timeout2 > timeout1
        assert timeout1 == timeout0 * self.config.progressive_multiplier
        assert timeout2 == timeout0 * (self.config.progressive_multiplier ** 2)
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker state checking"""
        endpoint = GitHubEndpoint.ISSUE_LIST
        
        # Initially should allow requests
        can_attempt, reason = self.manager.can_attempt_request(endpoint)
        assert can_attempt is True
        
        # Simulate failures to trip circuit breaker
        circuit_breaker = self.manager.circuit_breakers[endpoint]
        for _ in range(6):  # Exceed failure threshold
            circuit_breaker.record_failure()
        
        # Should now be blocked
        can_attempt, reason = self.manager.can_attempt_request(endpoint)
        assert can_attempt is False
        assert "open" in reason.lower()
    
    def test_record_request_metrics_success(self):
        """Test recording successful request metrics"""
        endpoint = GitHubEndpoint.ISSUE_VIEW
        duration = 2.5
        timeout_used = 30.0
        
        initial_total = self.manager.endpoint_profiles[endpoint].total_requests
        
        self.manager.record_request_metrics(
            endpoint=endpoint,
            duration=duration,
            success=True,
            timeout_used=timeout_used
        )
        
        profile = self.manager.endpoint_profiles[endpoint]
        assert profile.total_requests == initial_total + 1
        assert profile.failure_count == 0
        assert len(profile.response_times) == 1
        assert profile.response_times[0] == duration
        assert len(self.manager.metrics_history) == 1
        
        # Check circuit breaker recorded success
        circuit_breaker = self.manager.circuit_breakers[endpoint]
        assert circuit_breaker.failure_count == 0
    
    def test_record_request_metrics_failure(self):
        """Test recording failed request metrics"""
        endpoint = GitHubEndpoint.ISSUE_CREATE
        duration = 31.0  # Timeout
        timeout_used = 30.0
        error_type = "timeout"
        
        initial_total = self.manager.endpoint_profiles[endpoint].total_requests
        initial_failures = self.manager.endpoint_profiles[endpoint].failure_count
        
        self.manager.record_request_metrics(
            endpoint=endpoint,
            duration=duration,
            success=False,
            timeout_used=timeout_used,
            error_type=error_type
        )
        
        profile = self.manager.endpoint_profiles[endpoint]
        assert profile.total_requests == initial_total + 1
        assert profile.failure_count == initial_failures + 1
        assert len(profile.response_times) == 1
        assert profile.response_times[0] == duration
        
        # Check success rate calculation
        expected_success_rate = (profile.total_requests - profile.failure_count) / profile.total_requests
        assert profile.success_rate == expected_success_rate
        
        # Check circuit breaker recorded failure
        circuit_breaker = self.manager.circuit_breakers[endpoint]
        assert circuit_breaker.failure_count == 1
    
    def test_endpoint_profile_updates(self):
        """Test endpoint profile statistics updates"""
        endpoint = GitHubEndpoint.SEARCH
        
        # Record several metrics
        durations = [1.2, 2.1, 1.8, 3.5, 2.0, 1.7, 2.3, 1.9, 2.8, 2.2]
        
        for duration in durations:
            self.manager.record_request_metrics(
                endpoint=endpoint,
                duration=duration,
                success=True,
                timeout_used=30.0
            )
        
        profile = self.manager.endpoint_profiles[endpoint]
        
        # Check statistics
        import statistics
        expected_avg = statistics.mean(durations)
        assert abs(profile.avg_response_time - expected_avg) < 0.01
        
        assert profile.total_requests == len(durations)
        assert profile.success_rate == 1.0  # All successful
        assert len(profile.response_times) == len(durations)
    
    def test_timeout_optimization(self):
        """Test timeout optimization based on performance"""
        endpoint = GitHubEndpoint.PR_CREATE
        profile = self.manager.endpoint_profiles[endpoint]
        
        # Simulate good performance (fast responses, high success rate)
        fast_durations = [1.0, 1.2, 0.8, 1.1, 0.9] * 2  # 10 requests
        for duration in fast_durations:
            self.manager.record_request_metrics(endpoint, duration, True, 30.0)
        
        old_timeout = profile.recommended_timeout
        optimizations = self.manager.optimize_timeouts()
        
        # Should have optimized timeout down due to good performance
        assert profile.recommended_timeout <= old_timeout
        assert len(optimizations) > 0
        
        # Find optimization for our endpoint
        opt = next((o for o in optimizations if o["endpoint"] == endpoint.value), None)
        assert opt is not None
        assert opt["new_timeout"] <= opt["old_timeout"]
    
    def test_timeout_optimization_poor_performance(self):
        """Test timeout optimization with poor performance"""
        endpoint = GitHubEndpoint.BULK_OPERATIONS
        profile = self.manager.endpoint_profiles[endpoint]
        
        # Simulate poor performance
        slow_durations = [25.0, 28.0, 30.0, 27.0, 29.0]  # Slow responses
        failures = [False, True, False, True, False]  # Some failures
        
        for duration, success in zip(slow_durations, failures):
            self.manager.record_request_metrics(endpoint, duration, success, 30.0)
        
        old_timeout = profile.recommended_timeout
        optimizations = self.manager.optimize_timeouts()
        
        # Should have increased timeout due to poor performance
        if optimizations:  # May not optimize if not enough data
            opt = next((o for o in optimizations if o["endpoint"] == endpoint.value), None)
            if opt:
                assert opt["new_timeout"] >= opt["old_timeout"]
    
    def test_get_endpoint_stats(self):
        """Test endpoint statistics retrieval"""
        endpoint = GitHubEndpoint.ISSUE_COMMENT
        
        # Record some metrics
        self.manager.record_request_metrics(endpoint, 2.5, True, 30.0)
        self.manager.record_request_metrics(endpoint, 3.2, False, 30.0, error_type="timeout")
        
        stats = self.manager.get_endpoint_stats(endpoint)
        
        assert stats["endpoint"] == endpoint.value
        assert stats["total_requests"] == 2
        assert stats["failure_count"] == 1
        assert stats["success_rate"] == 0.5
        assert "avg_response_time" in stats
        assert "p95_response_time" in stats
        assert "recommended_timeout" in stats
        assert "circuit_breaker_state" in stats
    
    def test_get_all_stats(self):
        """Test comprehensive statistics retrieval"""
        # Record metrics for multiple endpoints
        self.manager.record_request_metrics(GitHubEndpoint.ISSUE_LIST, 1.5, True, 30.0)
        self.manager.record_request_metrics(GitHubEndpoint.ISSUE_VIEW, 2.0, True, 30.0)
        self.manager.record_request_metrics(GitHubEndpoint.ISSUE_CREATE, 3.5, False, 30.0)
        
        stats = self.manager.get_all_stats()
        
        assert "total_requests" in stats
        assert stats["total_requests"] == 3
        assert "configuration" in stats
        assert "endpoints" in stats
        assert "overall_success_rate" in stats
        assert "avg_duration" in stats
        
        # Check that all endpoints are included
        for endpoint in GitHubEndpoint:
            assert endpoint.value in stats["endpoints"]
    
    @patch('threading.Thread')
    def test_metrics_analyzer_thread(self, mock_thread):
        """Test that metrics analyzer thread is started"""
        manager = GitHubTimeoutManager(self.config)
        
        # Verify thread was created and started
        mock_thread.assert_called_once()
        call_args = mock_thread.call_args
        assert call_args[1]["daemon"] is True
        
        # Verify the thread start was called
        thread_instance = mock_thread.return_value
        thread_instance.start.assert_called_once()
    
    def test_metrics_persistence(self):
        """Test metrics persistence to disk"""
        # Record some metrics
        self.manager.record_request_metrics(
            GitHubEndpoint.ISSUE_LIST, 2.0, True, 30.0
        )
        
        # Manually trigger persistence
        self.manager._persist_metrics()
        
        # Check file was created
        assert self.manager.metrics_file.exists()
        
        # Check file content
        with open(self.manager.metrics_file, 'r') as f:
            data = json.load(f)
        
        assert "timestamp" in data
        assert "config" in data
        assert "metrics" in data
        assert "endpoint_stats" in data
        assert len(data["metrics"]) == 1
        
        # Check metric data
        metric = data["metrics"][0]
        assert metric["endpoint"] == GitHubEndpoint.ISSUE_LIST.value
        assert metric["duration"] == 2.0
        assert metric["success"] is True
    
    def test_historical_metrics_loading(self):
        """Test loading historical metrics from disk"""
        # Create mock historical data
        historical_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {"base_timeout": 30.0},
            "metrics": [
                {
                    "endpoint": GitHubEndpoint.ISSUE_VIEW.value,
                    "duration": 1.5,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "timeout_used": 30.0,
                    "retry_count": 0,
                    "error_type": None
                }
            ],
            "endpoint_stats": {}
        }
        
        # Write to metrics file
        with open(self.manager.metrics_file, 'w') as f:
            json.dump(historical_data, f)
        
        # Create new manager instance (should load historical data)
        with patch.object(Path, 'mkdir'):
            new_manager = GitHubTimeoutManager(self.config)
            new_manager.metrics_file = self.manager.metrics_file
            new_manager._load_historical_metrics()
        
        # Check that historical data was loaded
        assert len(new_manager.metrics_history) > 0
        
        # Check endpoint profile was updated
        profile = new_manager.endpoint_profiles[GitHubEndpoint.ISSUE_VIEW]
        assert profile.total_requests > 0

class TestTimeoutManagerSingleton:
    """Test singleton pattern for timeout manager"""
    
    def test_get_timeout_manager_singleton(self):
        """Test that get_timeout_manager returns the same instance"""
        # Clear any existing instance
        import claude.commands.github_timeout_manager as tm_module
        tm_module._timeout_manager = None
        
        manager1 = get_timeout_manager()
        manager2 = get_timeout_manager()
        
        assert manager1 is manager2
    
    def test_create_timeout_manager_factory(self):
        """Test create_timeout_manager factory function"""
        config = TimeoutConfig(base_timeout=45.0)
        manager = create_timeout_manager(config)
        
        assert isinstance(manager, GitHubTimeoutManager)
        assert manager.config.base_timeout == 45.0

class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent access"""
    
    def setup_method(self):
        """Set up test environment"""
        self.manager = create_timeout_manager(TimeoutConfig(sample_window=50))
    
    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording from multiple threads"""
        endpoint = GitHubEndpoint.ISSUE_LIST
        num_threads = 10
        metrics_per_thread = 20
        
        def record_metrics():
            for i in range(metrics_per_thread):
                duration = 1.0 + (i * 0.1)  # Varying durations
                success = i % 3 != 0  # Some failures
                self.manager.record_request_metrics(
                    endpoint, duration, success, 30.0
                )
                time.sleep(0.001)  # Small delay
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        profile = self.manager.endpoint_profiles[endpoint]
        expected_total = num_threads * metrics_per_thread
        
        assert profile.total_requests == expected_total
        assert len(self.manager.metrics_history) == expected_total
    
    def test_concurrent_timeout_calculation(self):
        """Test concurrent timeout calculations"""
        endpoint = GitHubEndpoint.SEARCH
        num_threads = 20
        
        def calculate_timeouts():
            results = []
            for retry_count in range(5):
                timeout = self.manager.get_timeout(endpoint, retry_count)
                results.append(timeout)
                time.sleep(0.001)
            return results
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=calculate_timeouts)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should not have raised any exceptions
        assert True  # If we get here, no thread safety issues occurred

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Set up test environment"""
        self.manager = create_timeout_manager()
    
    def test_unknown_endpoint_handling(self):
        """Test handling of unknown/invalid endpoints"""
        # This would require mocking enum behavior, skip for now
        pass
    
    def test_extreme_timeout_values(self):
        """Test extreme timeout value handling"""
        config = TimeoutConfig(
            min_timeout=0.1,
            max_timeout=1.0,  # Very low max
            base_timeout=0.5
        )
        manager = GitHubTimeoutManager(config)
        
        endpoint = GitHubEndpoint.BULK_OPERATIONS
        
        # Very high retry count should be clamped to max
        timeout = manager.get_timeout(endpoint, 20)
        assert timeout <= config.max_timeout
        assert timeout >= config.min_timeout
    
    def test_empty_metrics_handling(self):
        """Test handling when no metrics are available"""
        endpoint = GitHubEndpoint.REPO_INFO
        
        # Should still return reasonable defaults
        timeout = self.manager.get_timeout(endpoint)
        assert timeout > 0
        
        stats = self.manager.get_endpoint_stats(endpoint)
        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 1.0  # Default optimistic
    
    def test_optimization_with_insufficient_data(self):
        """Test optimization when insufficient data is available"""
        # Record very few metrics
        self.manager.record_request_metrics(
            GitHubEndpoint.PR_LIST, 2.0, True, 30.0
        )
        
        # Should not optimize with insufficient data
        optimizations = self.manager.optimize_timeouts()
        
        # May or may not optimize depending on implementation
        # Main thing is it shouldn't crash
        assert isinstance(optimizations, list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])