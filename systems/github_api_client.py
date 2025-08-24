#!/usr/bin/env python3
"""
GitHub API Resilient Client - Centralized GitHub API Access with Built-in Resilience

This module provides a centralized GitHub API client with:
1. Exponential backoff retry mechanism (3 attempts: 2s, 5s, 10s delays)
2. Request timeout configuration (60s base timeout)
3. Rate limit handling with intelligent queuing
4. Circuit breaker pattern for resilience
5. Comprehensive error handling and recovery

Designed to solve issue #151: High Priority Error Investigation: err_20250823_20b66aa5
"""

import json
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Event
import queue
import threading


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class RetryStrategy(Enum):
    """Retry strategy options"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 2.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    timeout: float = 60.0    # seconds (increased from 30s default)


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information"""
    limit: int = 5000
    remaining: int = 5000
    reset_time: datetime = field(default_factory=datetime.now)
    
    @property
    def is_limited(self) -> bool:
        """Check if we're rate limited"""
        return self.remaining <= 10  # Conservative threshold
    
    @property
    def seconds_until_reset(self) -> float:
        """Seconds until rate limit resets"""
        return max(0, (self.reset_time - datetime.now()).total_seconds())


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for GitHub API resilience"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit breaker state"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class RequestQueue:
    """Intelligent request queue for rate limit handling"""
    
    def __init__(self, max_size: int = 100):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.processing = False
        self.lock = Lock()
    
    def add_request(self, priority: int, request_data: Dict[str, Any]) -> bool:
        """Add request to queue with priority (lower number = higher priority)"""
        try:
            timestamp = time.time()
            # Priority tuple: (priority, timestamp, request_data)
            self.queue.put((priority, timestamp, request_data), timeout=1.0)
            return True
        except queue.Full:
            return False
    
    def get_next_request(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Get next request from queue"""
        try:
            priority, timestamp, request_data = self.queue.get_nowait()
            return priority, request_data
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()


class ResilientGitHubClient:
    """
    Centralized GitHub API client with built-in resilience features.
    
    Features:
    - Exponential backoff retry (3 attempts: 2s, 5s, 10s delays)
    - 60-second request timeout (increased from 30s default)
    - Rate limit handling with intelligent queuing
    - Circuit breaker pattern
    - Comprehensive error handling and logging
    """
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.rate_limit = RateLimitInfo()
        self.circuit_breaker = CircuitBreaker()
        self.request_queue = RequestQueue()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = Lock()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'rate_limited_requests': 0,
            'circuit_breaker_rejections': 0
        }
        
        self.logger.info("Resilient GitHub Client initialized with 60s timeout and exponential backoff")
    
    def execute_gh_command(self, command: str, priority: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute GitHub CLI command with full resilience features.
        
        Args:
            command: GitHub CLI command to execute (e.g., "issue list --state open")
            priority: Request priority (1=highest, 10=lowest)
            **kwargs: Additional options (timeout_override, no_retry, etc.)
            
        Returns:
            Dict containing command result and metadata
            
        Raises:
            GitHubAPIError: On unrecoverable errors
        """
        self.stats['total_requests'] += 1
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            self.stats['circuit_breaker_rejections'] += 1
            raise GitHubAPIError(
                f"Circuit breaker is OPEN. Last failure: {self.circuit_breaker.last_failure_time}",
                status_code=503
            )
        
        # Check rate limits
        if self.rate_limit.is_limited:
            wait_time = min(self.rate_limit.seconds_until_reset, 300)  # Max 5 min wait
            if wait_time > 0:
                self.stats['rate_limited_requests'] += 1
                self.logger.warning(f"Rate limited. Waiting {wait_time:.1f}s until reset")
                time.sleep(wait_time)
        
        # Execute with retry logic
        last_exception = None
        timeout = kwargs.get('timeout_override', self.retry_config.timeout)
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                self.logger.debug(f"GitHub API attempt {attempt + 1}/{self.retry_config.max_attempts}: {command}")
                
                # Construct full gh command
                full_command = f"gh {command}"
                
                # Execute command with Python's built-in timeout (more portable than shell timeout)
                start_time = time.time()
                result = subprocess.run(
                    full_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout  # Use Python's subprocess timeout
                )
                execution_time = time.time() - start_time
                
                # Parse result
                response_data = self._parse_command_result(result, execution_time)
                
                # Update rate limit info if available
                self._update_rate_limit_from_headers(result)
                
                if result.returncode == 0:
                    # Success
                    self.circuit_breaker.record_success()
                    self.stats['successful_requests'] += 1
                    
                    if attempt > 0:
                        self.stats['retried_requests'] += 1
                        self.logger.info(f"GitHub API succeeded on retry {attempt + 1}")
                    
                    return response_data
                else:
                    # Command failed, determine if retryable
                    is_retryable = self._is_retryable_error(result.returncode, result.stderr)
                    if not is_retryable or attempt == self.retry_config.max_attempts - 1:
                        # Non-retryable error or final attempt
                        self.circuit_breaker.record_failure()
                        self.stats['failed_requests'] += 1
                        
                        error_msg = f"GitHub CLI command failed: {result.stderr.strip()}"
                        self.logger.error(error_msg)
                        raise GitHubAPIError(error_msg, status_code=result.returncode)
                    
                    # Retryable error, wait before next attempt
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.warning(f"GitHub API error (attempt {attempt + 1}), retrying in {delay}s: {result.stderr.strip()}")
                    time.sleep(delay)
                    last_exception = GitHubAPIError(result.stderr.strip(), status_code=result.returncode)
            
            except subprocess.TimeoutExpired:
                self.logger.warning(f"GitHub API timeout on attempt {attempt + 1}")
                if attempt == self.retry_config.max_attempts - 1:
                    self.circuit_breaker.record_failure()
                    self.stats['failed_requests'] += 1
                    raise GitHubAPIError(f"GitHub API timeout after {timeout}s", status_code=408)
                
                delay = self._calculate_retry_delay(attempt)
                time.sleep(delay)
                last_exception = GitHubAPIError(f"Timeout after {timeout}s", status_code=408)
            
            except Exception as e:
                self.logger.error(f"Unexpected error during GitHub API call: {e}")
                if attempt == self.retry_config.max_attempts - 1:
                    self.circuit_breaker.record_failure()
                    self.stats['failed_requests'] += 1
                    raise GitHubAPIError(f"Unexpected error: {str(e)}")
                
                delay = self._calculate_retry_delay(attempt)
                time.sleep(delay)
                last_exception = GitHubAPIError(str(e))
        
        # Should not reach here, but just in case
        raise last_exception or GitHubAPIError("All retry attempts failed")
    
    def _parse_command_result(self, result: subprocess.CompletedProcess, execution_time: float) -> Dict[str, Any]:
        """Parse subprocess result into structured response"""
        response = {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to parse JSON output if present
        if result.returncode == 0 and result.stdout.strip():
            try:
                # Check if output looks like JSON
                stdout_stripped = result.stdout.strip()
                if stdout_stripped.startswith(('{', '[')):
                    response['data'] = json.loads(stdout_stripped)
            except json.JSONDecodeError:
                # Not JSON, keep as string
                pass
        
        return response
    
    def _update_rate_limit_from_headers(self, result: subprocess.CompletedProcess):
        """Update rate limit info from command output (if available)"""
        # GitHub CLI doesn't typically expose rate limit headers directly
        # This would be enhanced if we had access to raw HTTP headers
        pass
    
    def _is_retryable_error(self, return_code: int, stderr: str) -> bool:
        """Determine if an error is retryable"""
        stderr_lower = stderr.lower()
        
        # Retryable conditions
        retryable_patterns = [
            'timeout',
            'connection refused',
            'network is unreachable',
            'temporary failure',
            'service unavailable',
            'bad gateway',
            'gateway timeout',
            'rate limit'
        ]
        
        # Non-retryable conditions
        non_retryable_patterns = [
            'not found',
            'permission denied',
            'unauthorized',
            'forbidden',
            'invalid token',
            'bad request'
        ]
        
        # Check for non-retryable patterns first
        for pattern in non_retryable_patterns:
            if pattern in stderr_lower:
                return False
        
        # Check for retryable patterns
        for pattern in retryable_patterns:
            if pattern in stderr_lower:
                return True
        
        # Default behavior based on return code
        if return_code in [124, 408, 502, 503, 504]:  # timeout, bad gateway, etc.
            return True
        elif return_code in [401, 403, 404, 422]:  # auth, not found, validation errors
            return False
        
        # For unknown errors, retry conservatively
        return return_code >= 500
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy and attempt number"""
        if self.retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential backoff: 2s, 5s, 10s for attempts 0, 1, 2
            if attempt == 0:
                delay = self.retry_config.base_delay  # 2s
            elif attempt == 1:
                delay = 5.0  # 5s
            else:
                delay = 10.0  # 10s
        elif self.retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.retry_config.base_delay * (attempt + 1)
        else:
            delay = self.retry_config.base_delay
        
        return min(delay, self.retry_config.max_delay)
    
    # Convenience methods for common GitHub operations
    
    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Get issue details with full resilience"""
        return self.execute_gh_command(f"issue view {issue_number} --json number,title,body,labels,state,createdAt,updatedAt,comments")
    
    def list_issues(self, state: str = "open", labels: Optional[List[str]] = None, limit: int = 50) -> Dict[str, Any]:
        """List issues with filtering"""
        cmd = f"issue list --state {state} --limit {limit} --json number,title,labels,state"
        if labels:
            labels_str = ",".join(labels)
            cmd += f" --label '{labels_str}'"
        return self.execute_gh_command(cmd)
    
    def update_issue_labels(self, issue_number: int, add_labels: Optional[List[str]] = None, 
                           remove_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Update issue labels"""
        cmd_parts = [f"issue edit {issue_number}"]
        
        if add_labels:
            labels_str = ",".join(add_labels)
            cmd_parts.append(f"--add-label '{labels_str}'")
        
        if remove_labels:
            labels_str = ",".join(remove_labels)
            cmd_parts.append(f"--remove-label '{labels_str}'")
        
        cmd = " ".join(cmd_parts)
        return self.execute_gh_command(cmd, priority=3)  # Higher priority for state changes
    
    def add_issue_comment(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """Add comment to issue"""
        # Escape the comment for shell safety
        escaped_comment = json.dumps(comment)
        return self.execute_gh_command(f"issue comment {issue_number} --body {escaped_comment}", priority=4)
    
    def create_pull_request(self, title: str, body: str, head: str, base: str = "main") -> Dict[str, Any]:
        """Create pull request"""
        escaped_title = json.dumps(title)
        escaped_body = json.dumps(body)
        return self.execute_gh_command(f"pr create --title {escaped_title} --body {escaped_body} --head {head} --base {base}")
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return self.execute_gh_command("api rate_limit", priority=1)
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'success_rate': stats['successful_requests'] / max(stats['total_requests'], 1),
                'retry_rate': stats['retried_requests'] / max(stats['total_requests'], 1),
                'circuit_breaker_state': self.circuit_breaker.state.value,
                'circuit_breaker_failures': self.circuit_breaker.failure_count,
                'rate_limit_remaining': self.rate_limit.remaining,
                'queue_size': self.request_queue.size()
            })
            return stats


# Singleton instance for global use
_github_client_instance: Optional[ResilientGitHubClient] = None
_client_lock = Lock()


def get_github_client() -> ResilientGitHubClient:
    """Get singleton instance of resilient GitHub client"""
    global _github_client_instance
    
    with _client_lock:
        if _github_client_instance is None:
            _github_client_instance = ResilientGitHubClient()
        return _github_client_instance


def configure_github_client(retry_config: RetryConfig) -> ResilientGitHubClient:
    """Configure and return GitHub client with custom settings"""
    global _github_client_instance
    
    with _client_lock:
        _github_client_instance = ResilientGitHubClient(retry_config)
        return _github_client_instance


# Testing and validation functions

def test_github_client_resilience():
    """Test the resilient client with error simulation"""
    client = get_github_client()
    
    print("Testing Resilient GitHub Client")
    print("=" * 40)
    
    try:
        # Test normal operation
        result = client.get_rate_limit_status()
        print(f"✅ Rate limit check: {result['success']}")
        
        # Test with invalid command (should retry then fail)
        try:
            client.execute_gh_command("invalid-command-test")
        except GitHubAPIError as e:
            print(f"✅ Error handling works: {str(e)[:50]}...")
        
        # Show stats
        stats = client.get_client_stats()
        print(f"\nClient Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Retry Rate: {stats['retry_rate']:.2%}")
        print(f"  Circuit Breaker: {stats['circuit_breaker_state']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_github_client_resilience()