#!/usr/bin/env python3
"""
GitHub API Client - Resilient GitHub API Client
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Resilient GitHub API client that integrates timeout management, request context preservation,
batch operation resilience, and rate limit coordination for comprehensive API reliability.
"""

import subprocess
import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import threading
import re

from .github_timeout_manager import GitHubEndpoint, get_timeout_manager
from .github_request_context import RequestState, ContextScope, get_context_manager
from .github_batch_resilience import BatchOperationType, BatchConfiguration, BatchStrategy, get_batch_manager

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limit handling strategies"""
    WAIT = "wait"              # Wait until rate limit resets
    FAIL_FAST = "fail_fast"    # Fail immediately if rate limited
    ADAPTIVE = "adaptive"      # Adapt based on available quota

@dataclass
class APICallResult:
    """Result of an API call"""
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    duration: float
    timeout_used: float
    rate_limit_remaining: Optional[int]
    rate_limit_reset: Optional[datetime]
    attempt_count: int
    context_id: Optional[str]

@dataclass
class RateLimitInfo:
    """GitHub API rate limit information"""
    remaining: int
    limit: int
    reset_time: datetime
    used: int
    resource: str  # e.g., "core", "search", "graphql"
    
    def time_until_reset(self) -> float:
        """Get seconds until rate limit resets"""
        return max(0, (self.reset_time - datetime.now()).total_seconds())
    
    def utilization_percentage(self) -> float:
        """Get rate limit utilization as percentage"""
        return (self.used / self.limit) * 100.0 if self.limit > 0 else 0.0

class GitHubAPIClient:
    """
    Resilient GitHub API client with:
    - Adaptive timeout management
    - Request context preservation for recovery
    - Batch operation support with fragmentation
    - Coordinated rate limit and timeout handling
    - Circuit breaker integration
    """
    
    def __init__(self, rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE):
        # Integration with resilience components
        self.timeout_manager = get_timeout_manager()
        self.context_manager = get_context_manager()
        self.batch_manager = get_batch_manager()
        
        # Rate limiting
        self.rate_limit_strategy = rate_limit_strategy
        self.rate_limit_info: Dict[str, RateLimitInfo] = {}
        self.rate_limit_lock = threading.RLock()
        
        # Client configuration
        self.default_timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Command construction
        self.base_command = ["gh"]
        self.common_flags = ["--json"]
        
        logger.info(f"Initialized GitHub API client with {rate_limit_strategy.value} rate limit strategy")
    
    def issue_list(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        limit: Optional[int] = None,
        assignee: Optional[str] = None,
        **kwargs
    ) -> APICallResult:
        """
        List GitHub issues with resilient handling.
        
        Args:
            state: Issue state ("open", "closed", "all")
            labels: List of labels to filter by
            limit: Maximum number of issues to return
            assignee: Filter by assignee
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with issue data
        """
        command_args = ["issue", "list", "--state", state]
        
        if labels:
            for label in labels:
                command_args.extend(["--label", label])
        
        if limit:
            command_args.extend(["--limit", str(limit)])
        
        if assignee:
            command_args.extend(["--assignee", assignee])
        
        # Add JSON output format
        command_args.extend(["--json", "number,title,state,labels,assignees,body,createdAt,updatedAt"])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.ISSUE_LIST,
            command_args=command_args,
            operation_type="issue_list",
            **kwargs
        )
    
    def issue_view(
        self,
        issue_number: int,
        include_comments: bool = False,
        **kwargs
    ) -> APICallResult:
        """
        View a specific GitHub issue.
        
        Args:
            issue_number: Issue number to view
            include_comments: Include comments in response
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with issue data
        """
        command_args = ["issue", "view", str(issue_number)]
        
        if include_comments:
            command_args.append("--comments")
        
        command_args.extend(["--json", "number,title,body,state,labels,assignees,comments,createdAt,updatedAt"])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.ISSUE_VIEW,
            command_args=command_args,
            operation_type="issue_view",
            **kwargs
        )
    
    def issue_create(
        self,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[str] = None,
        **kwargs
    ) -> APICallResult:
        """
        Create a new GitHub issue.
        
        Args:
            title: Issue title
            body: Issue body/description
            labels: List of labels to add
            assignees: List of assignees
            milestone: Milestone name
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with created issue data
        """
        command_args = ["issue", "create", "--title", title]
        
        if body:
            command_args.extend(["--body", body])
        
        if labels:
            for label in labels:
                command_args.extend(["--label", label])
        
        if assignees:
            for assignee in assignees:
                command_args.extend(["--assignee", assignee])
        
        if milestone:
            command_args.extend(["--milestone", milestone])
        
        command_args.extend(["--json", "number,title,url,state"])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.ISSUE_CREATE,
            command_args=command_args,
            operation_type="issue_create",
            priority=2,  # High priority for creation
            **kwargs
        )
    
    def issue_edit(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        add_labels: Optional[List[str]] = None,
        remove_labels: Optional[List[str]] = None,
        **kwargs
    ) -> APICallResult:
        """
        Edit an existing GitHub issue.
        
        Args:
            issue_number: Issue number to edit
            title: New title
            body: New body
            state: New state ("open" or "closed")
            add_labels: Labels to add
            remove_labels: Labels to remove
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with updated issue data
        """
        command_args = ["issue", "edit", str(issue_number)]
        
        if title:
            command_args.extend(["--title", title])
        
        if body:
            command_args.extend(["--body", body])
        
        if state:
            if state == "closed":
                command_args.append("--close")
            elif state == "open":
                command_args.append("--reopen")
        
        if add_labels:
            for label in add_labels:
                command_args.extend(["--add-label", label])
        
        if remove_labels:
            for label in remove_labels:
                command_args.extend(["--remove-label", label])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.ISSUE_EDIT,
            command_args=command_args,
            operation_type="issue_edit",
            **kwargs
        )
    
    def issue_comment(
        self,
        issue_number: int,
        body: str,
        **kwargs
    ) -> APICallResult:
        """
        Add a comment to a GitHub issue.
        
        Args:
            issue_number: Issue number to comment on
            body: Comment body
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with comment data
        """
        command_args = ["issue", "comment", str(issue_number), "--body", body]
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.ISSUE_COMMENT,
            command_args=command_args,
            operation_type="issue_comment",
            **kwargs
        )
    
    def pr_create(
        self,
        title: str,
        body: Optional[str] = None,
        head: Optional[str] = None,
        base: str = "main",
        draft: bool = False,
        **kwargs
    ) -> APICallResult:
        """
        Create a new pull request.
        
        Args:
            title: PR title
            body: PR body/description
            head: Head branch (defaults to current branch)
            base: Base branch to merge into
            draft: Create as draft PR
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with created PR data
        """
        command_args = ["pr", "create", "--title", title, "--base", base]
        
        if body:
            command_args.extend(["--body", body])
        
        if head:
            command_args.extend(["--head", head])
        
        if draft:
            command_args.append("--draft")
        
        command_args.extend(["--json", "number,title,url,state,headRefName,baseRefName"])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.PR_CREATE,
            command_args=command_args,
            operation_type="pr_create",
            priority=2,  # High priority for creation
            **kwargs
        )
    
    def search_issues(
        self,
        query: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> APICallResult:
        """
        Search GitHub issues.
        
        Args:
            query: Search query
            limit: Maximum number of results
            **kwargs: Additional GitHub CLI flags
            
        Returns:
            APICallResult with search results
        """
        command_args = ["search", "issues", query]
        
        if limit:
            command_args.extend(["--limit", str(limit)])
        
        command_args.extend(["--json", "number,title,state,labels,repository,url,createdAt"])
        
        return self._execute_api_call(
            endpoint=GitHubEndpoint.SEARCH,
            command_args=command_args,
            operation_type="search_issues",
            **kwargs
        )
    
    def bulk_issue_update(
        self,
        updates: List[Dict[str, Any]],
        config: Optional[BatchConfiguration] = None,
        **kwargs
    ) -> str:
        """
        Perform bulk issue updates using batch resilience.
        
        Args:
            updates: List of update specifications
                    Each dict should contain: {"issue_number": int, "updates": dict}
            config: Batch configuration
            **kwargs: Additional options
            
        Returns:
            Batch ID for tracking progress
        """
        # Create batch operation
        batch = self.batch_manager.create_batch_operation(
            operation_type=BatchOperationType.ISSUE_BULK_UPDATE,
            endpoint=GitHubEndpoint.ISSUE_EDIT,
            items_data=updates,
            config=config or BatchConfiguration(
                chunk_size=5,
                parallel_limit=2,
                strategy=BatchStrategy.ADAPTIVE
            )
        )
        
        # Define item executor
        def execute_update(update_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
            issue_number = update_data["issue_number"]
            updates = update_data["updates"]
            
            result = self.issue_edit(issue_number, **updates)
            return result.success, result.data, result.error_message
        
        # Start batch execution
        success = self.batch_manager.execute_batch(batch.batch_id, execute_update)
        
        if not success:
            raise RuntimeError(f"Failed to start batch operation {batch.batch_id}")
        
        return batch.batch_id
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch operation"""
        return self.batch_manager.get_batch_status(batch_id)
    
    def _execute_api_call(
        self,
        endpoint: GitHubEndpoint,
        command_args: List[str],
        operation_type: str,
        priority: int = 3,
        max_retries: Optional[int] = None,
        context_scope: ContextScope = ContextScope.SESSION_PERSISTENT,
        **kwargs
    ) -> APICallResult:
        """
        Execute a GitHub API call with full resilience handling.
        
        Args:
            endpoint: GitHub endpoint type
            command_args: Command arguments for gh CLI
            operation_type: Type of operation for context tracking
            priority: Operation priority (1=critical, 5=low)
            max_retries: Maximum retry attempts
            context_scope: Context preservation scope
            **kwargs: Additional options
            
        Returns:
            APICallResult with call results
        """
        max_retries = max_retries or self.max_retries
        
        # Check circuit breaker state
        can_attempt, reason = self.timeout_manager.can_attempt_request(endpoint)
        if not can_attempt:
            return APICallResult(
                success=False,
                data=None,
                error_message=f"Circuit breaker: {reason}",
                duration=0.0,
                timeout_used=0.0,
                rate_limit_remaining=None,
                rate_limit_reset=None,
                attempt_count=0,
                context_id=None
            )
        
        # Create request context
        context = self.context_manager.create_context(
            endpoint=endpoint,
            operation_type=operation_type,
            command_args=command_args,
            environment=dict(os.environ),
            working_directory=os.getcwd(),
            priority=priority,
            scope=context_scope,
            max_attempts=max_retries,
            tags=[f"endpoint:{endpoint.value}", f"operation:{operation_type}"]
        )
        
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            attempt += 1
            
            # Update context state
            self.context_manager.update_context_state(
                context.context_id,
                RequestState.EXECUTING if attempt == 1 else RequestState.RETRYING
            )
            
            # Get adaptive timeout
            timeout = self.timeout_manager.get_timeout(endpoint, attempt - 1)
            
            # Check rate limits
            rate_limit_delay = self._check_rate_limits()
            if rate_limit_delay > 0:
                if self.rate_limit_strategy == RateLimitStrategy.FAIL_FAST:
                    self.context_manager.complete_context(context.context_id, None, False)
                    return APICallResult(
                        success=False,
                        data=None,
                        error_message="Rate limit exceeded",
                        duration=0.0,
                        timeout_used=timeout,
                        rate_limit_remaining=0,
                        rate_limit_reset=None,
                        attempt_count=attempt,
                        context_id=context.context_id
                    )
                elif self.rate_limit_strategy == RateLimitStrategy.WAIT:
                    logger.info(f"Waiting {rate_limit_delay:.1f}s for rate limit reset")
                    time.sleep(rate_limit_delay)
            
            # Execute the command
            start_time = time.time()
            success, data, error_message = self._execute_gh_command(
                command_args, timeout, context.context_id
            )
            duration = time.time() - start_time
            
            # Update rate limit info from response
            self._update_rate_limit_info(data)
            
            # Record metrics
            self.timeout_manager.record_request_metrics(
                endpoint=endpoint,
                duration=duration,
                success=success,
                timeout_used=timeout,
                retry_count=attempt - 1,
                error_type=error_message if not success else None
            )
            
            if success:
                # Success - complete context and return
                self.context_manager.complete_context(context.context_id, data, True)
                
                rate_limit_info = self.rate_limit_info.get("core")
                return APICallResult(
                    success=True,
                    data=data,
                    error_message=None,
                    duration=duration,
                    timeout_used=timeout,
                    rate_limit_remaining=rate_limit_info.remaining if rate_limit_info else None,
                    rate_limit_reset=rate_limit_info.reset_time if rate_limit_info else None,
                    attempt_count=attempt,
                    context_id=context.context_id
                )
            
            else:
                # Failure - determine if we should retry
                last_error = error_message
                
                # Update context with error info
                self.context_manager.update_context_state(
                    context.context_id,
                    RequestState.FAILED if attempt >= max_retries else RequestState.RETRYING,
                    error_info={
                        "attempt": attempt,
                        "error": error_message,
                        "duration": duration,
                        "timeout_used": timeout
                    }
                )
                
                # Check if error is retryable
                if not self._is_retryable_error(error_message):
                    break
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries:
                    retry_delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
        
        # All retries exhausted - complete context with failure
        self.context_manager.complete_context(context.context_id, None, False)
        
        rate_limit_info = self.rate_limit_info.get("core")
        return APICallResult(
            success=False,
            data=None,
            error_message=last_error,
            duration=duration,
            timeout_used=timeout,
            rate_limit_remaining=rate_limit_info.remaining if rate_limit_info else None,
            rate_limit_reset=rate_limit_info.reset_time if rate_limit_info else None,
            attempt_count=attempt,
            context_id=context.context_id
        )
    
    def _execute_gh_command(
        self,
        command_args: List[str],
        timeout: float,
        context_id: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Execute GitHub CLI command with timeout handling"""
        try:
            # Construct full command
            full_command = self.base_command + command_args
            
            logger.debug(f"Executing: {' '.join(full_command)} (timeout: {timeout:.1f}s)")
            
            # Execute with timeout
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # Success - parse JSON response
                try:
                    data = json.loads(result.stdout) if result.stdout.strip() else {}
                    return True, data, None
                except json.JSONDecodeError as e:
                    return False, None, f"Failed to parse JSON response: {e}"
            
            else:
                # Command failed
                error_message = result.stderr.strip() or f"Command failed with exit code {result.returncode}"
                return False, None, error_message
        
        except subprocess.TimeoutExpired:
            return False, None, f"GitHub API timeout after {timeout} seconds"
        
        except Exception as e:
            return False, None, f"Command execution error: {e}"
    
    def _is_retryable_error(self, error_message: Optional[str]) -> bool:
        """Determine if an error is retryable"""
        if not error_message:
            return False
        
        error_lower = error_message.lower()
        
        # Retryable errors
        retryable_patterns = [
            "timeout",
            "network",
            "connection",
            "temporary",
            "rate limit",
            "server error",
            "503",
            "502",
            "500"
        ]
        
        # Non-retryable errors
        non_retryable_patterns = [
            "not found",
            "unauthorized",
            "forbidden",
            "validation failed",
            "422",
            "401",
            "403",
            "404"
        ]
        
        # Check non-retryable first
        for pattern in non_retryable_patterns:
            if pattern in error_lower:
                return False
        
        # Check retryable
        for pattern in retryable_patterns:
            if pattern in error_lower:
                return True
        
        # Default: retry unknown errors
        return True
    
    def _check_rate_limits(self) -> float:
        """
        Check rate limits and return delay needed (if any).
        
        Returns:
            Seconds to wait before making request (0 if no wait needed)
        """
        with self.rate_limit_lock:
            core_limits = self.rate_limit_info.get("core")
            if not core_limits:
                return 0.0  # No rate limit info available
            
            # If we have remaining quota, proceed
            if core_limits.remaining > 0:
                return 0.0
            
            # Calculate time until reset
            time_until_reset = core_limits.time_until_reset()
            
            if self.rate_limit_strategy == RateLimitStrategy.WAIT:
                return time_until_reset + 1.0  # Add 1 second buffer
            elif self.rate_limit_strategy == RateLimitStrategy.ADAPTIVE:
                # For adaptive, wait only if reset is soon
                if time_until_reset < 300:  # Less than 5 minutes
                    return time_until_reset + 1.0
            
            return 0.0
    
    def _update_rate_limit_info(self, response_data: Optional[Dict[str, Any]]):
        """Update rate limit information from API response headers"""
        # Note: GitHub CLI doesn't typically expose rate limit headers directly
        # This would need to be enhanced to parse rate limit info from responses
        # For now, we'll implement a placeholder that could be extended
        
        with self.rate_limit_lock:
            # This is a simplified implementation
            # In a real implementation, we'd parse rate limit headers from the response
            current_time = datetime.now()
            
            # Create a default rate limit info if we don't have one
            if "core" not in self.rate_limit_info:
                self.rate_limit_info["core"] = RateLimitInfo(
                    remaining=4999,  # Assume we have most of our quota
                    limit=5000,
                    reset_time=current_time + timedelta(hours=1),
                    used=1,
                    resource="core"
                )
            else:
                # Decrement remaining count
                core_info = self.rate_limit_info["core"]
                if core_info.remaining > 0:
                    core_info.remaining -= 1
                    core_info.used += 1
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        timeout_stats = self.timeout_manager.get_all_stats()
        context_stats = self.context_manager.get_context_stats()
        batch_stats = self.batch_manager.get_batch_stats()
        
        with self.rate_limit_lock:
            rate_limit_stats = {}
            for resource, info in self.rate_limit_info.items():
                rate_limit_stats[resource] = {
                    "remaining": info.remaining,
                    "limit": info.limit,
                    "reset_time": info.reset_time.isoformat(),
                    "utilization": info.utilization_percentage(),
                    "time_until_reset": info.time_until_reset()
                }
        
        return {
            "client_config": {
                "rate_limit_strategy": self.rate_limit_strategy.value,
                "default_timeout": self.default_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay
            },
            "timeout_management": timeout_stats,
            "context_management": context_stats,
            "batch_management": batch_stats,
            "rate_limits": rate_limit_stats
        }

# Global API client instance (singleton pattern)
_api_client = None

def get_api_client(rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE) -> GitHubAPIClient:
    """Get the global API client instance"""
    global _api_client
    if _api_client is None:
        _api_client = GitHubAPIClient(rate_limit_strategy)
    return _api_client