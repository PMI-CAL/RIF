#!/usr/bin/env python3
"""
Tests for GitHub Request Context System
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Comprehensive tests for request context preservation, state management,
recovery coordination, and persistence functionality.
"""

import pytest
import json
import time
import threading
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the Python path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.github_request_context import (
    GitHubRequestContextManager,
    RequestContext,
    RequestState,
    ContextScope,
    get_context_manager
)
from claude.commands.github_timeout_manager import GitHubEndpoint

class TestRequestContext:
    """Test RequestContext data structure"""
    
    def test_request_context_creation(self):
        """Test creating a request context"""
        context = RequestContext(
            context_id="test-123",
            endpoint=GitHubEndpoint.ISSUE_CREATE,
            operation_type="issue_create",
            command_args=["issue", "create", "--title", "Test"],
            environment={"USER": "test"},
            working_directory="/tmp",
            state=RequestState.INITIALIZED,
            created_at=datetime.now(),
            last_attempt=None,
            attempt_count=0,
            max_attempts=3,
            partial_results=None,
            intermediate_state=None,
            continuation_data=None,
            timeout_used=None,
            error_history=[],
            recovery_strategy=None,
            priority=3,
            scope=ContextScope.SESSION_PERSISTENT,
            expires_at=None,
            tags=["test"]
        )
        
        assert context.context_id == "test-123"
        assert context.endpoint == GitHubEndpoint.ISSUE_CREATE
        assert context.operation_type == "issue_create"
        assert context.state == RequestState.INITIALIZED
        assert context.attempt_count == 0
        assert context.priority == 3
        assert context.scope == ContextScope.SESSION_PERSISTENT
        assert "test" in context.tags
    
    def test_context_serialization(self):
        """Test context to_dict and from_dict"""
        original = RequestContext(
            context_id="serialize-test",
            endpoint=GitHubEndpoint.ISSUE_VIEW,
            operation_type="issue_view",
            command_args=["issue", "view", "123"],
            environment={"PATH": "/usr/bin"},
            working_directory="/home/user",
            state=RequestState.EXECUTING,
            created_at=datetime.now(),
            last_attempt=datetime.now(),
            attempt_count=1,
            max_attempts=5,
            partial_results={"status": "in_progress"},
            intermediate_state={"step": 2},
            continuation_data={"next_action": "retry"},
            timeout_used=30.0,
            error_history=[{"error": "timeout"}],
            recovery_strategy="exponential_backoff",
            priority=2,
            scope=ContextScope.OPERATION_CHAIN,
            expires_at=datetime.now() + timedelta(hours=1),
            tags=["high_priority", "retry"]
        )
        
        # Serialize
        data = original.to_dict()
        
        assert data["context_id"] == "serialize-test"
        assert data["endpoint"] == GitHubEndpoint.ISSUE_VIEW.value
        assert data["state"] == RequestState.EXECUTING.value
        assert data["scope"] == ContextScope.OPERATION_CHAIN.value
        assert data["partial_results"] == {"status": "in_progress"}
        
        # Deserialize
        restored = RequestContext.from_dict(data)
        
        assert restored.context_id == original.context_id
        assert restored.endpoint == original.endpoint
        assert restored.operation_type == original.operation_type
        assert restored.state == original.state
        assert restored.scope == original.scope
        assert restored.partial_results == original.partial_results
        assert restored.intermediate_state == original.intermediate_state
        assert restored.continuation_data == original.continuation_data
        assert restored.error_history == original.error_history
        assert restored.tags == original.tags

class TestGitHubRequestContextManager:
    """Test the main context manager functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "contexts"
        
        # Create manager with test storage path
        self.manager = GitHubRequestContextManager(str(self.storage_path))
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test context manager initialization"""
        assert self.manager.storage_path == self.storage_path
        assert len(self.manager.active_contexts) == 0
        assert len(self.manager.completed_contexts) == 0
        assert self.storage_path.exists()
    
    def test_create_context_basic(self):
        """Test creating a basic context"""
        context = self.manager.create_context(
            endpoint=GitHubEndpoint.ISSUE_LIST,
            operation_type="list_issues",
            command_args=["issue", "list", "--state", "open"]
        )
        
        assert context.context_id is not None
        assert len(context.context_id) > 0
        assert context.endpoint == GitHubEndpoint.ISSUE_LIST
        assert context.operation_type == "list_issues"
        assert context.state == RequestState.INITIALIZED
        assert context.attempt_count == 0
        assert context.max_attempts == 5  # default
        assert context.priority == 3  # default
        assert context.scope == ContextScope.SESSION_PERSISTENT  # default
        
        # Should be in active contexts
        assert context.context_id in self.manager.active_contexts
    
    def test_create_context_with_all_options(self):
        """Test creating a context with all options"""
        expiry = datetime.now() + timedelta(hours=2)
        
        context = self.manager.create_context(
            endpoint=GitHubEndpoint.PR_CREATE,
            operation_type="create_pr",
            command_args=["pr", "create", "--title", "Test PR"],
            environment={"GITHUB_TOKEN": "secret"},
            working_directory="/project",
            priority=1,
            scope=ContextScope.OPERATION_CHAIN,
            max_attempts=3,
            expiry_hours=2,
            tags=["critical", "pr_creation"]
        )
        
        assert context.priority == 1
        assert context.scope == ContextScope.OPERATION_CHAIN
        assert context.max_attempts == 3
        assert context.environment["GITHUB_TOKEN"] == "secret"
        assert context.working_directory == "/project"
        assert "critical" in context.tags
        assert "pr_creation" in context.tags
        assert context.expires_at is not None
    
    def test_update_context_state(self):
        """Test updating context state"""
        context = self.manager.create_context(
            GitHubEndpoint.ISSUE_COMMENT,
            "add_comment",
            ["issue", "comment", "123", "--body", "Test comment"]
        )
        
        context_id = context.context_id
        
        # Update to executing state
        success = self.manager.update_context_state(
            context_id,
            RequestState.EXECUTING,
            partial_results={"progress": "started"},
            intermediate_state={"step": 1}
        )
        
        assert success is True
        
        updated_context = self.manager.get_context(context_id)
        assert updated_context.state == RequestState.EXECUTING
        assert updated_context.attempt_count == 1
        assert updated_context.partial_results == {"progress": "started"}
        assert updated_context.intermediate_state == {"step": 1}
        assert updated_context.last_attempt is not None
    
    def test_update_context_with_error(self):
        """Test updating context with error information"""
        context = self.manager.create_context(
            GitHubEndpoint.SEARCH,
            "search_issues",
            ["search", "issues", "query"]
        )
        
        context_id = context.context_id
        
        # Update with error
        success = self.manager.update_context_state(
            context_id,
            RequestState.FAILED,
            error_info={
                "type": "timeout",
                "message": "Request timed out after 30 seconds",
                "code": "TIMEOUT"
            }
        )
        
        assert success is True
        
        updated_context = self.manager.get_context(context_id)
        assert updated_context.state == RequestState.FAILED
        assert len(updated_context.error_history) == 1
        
        error = updated_context.error_history[0]
        assert error["type"] == "timeout"
        assert error["message"] == "Request timed out after 30 seconds"
        assert error["attempt"] == 1
    
    def test_get_contexts_by_endpoint(self):
        """Test retrieving contexts by endpoint"""
        # Create contexts for different endpoints
        context1 = self.manager.create_context(
            GitHubEndpoint.ISSUE_LIST, "list1", ["issue", "list"]
        )
        context2 = self.manager.create_context(
            GitHubEndpoint.ISSUE_LIST, "list2", ["issue", "list", "--label", "bug"]
        )
        context3 = self.manager.create_context(
            GitHubEndpoint.PR_LIST, "pr_list", ["pr", "list"]
        )
        
        # Get contexts for ISSUE_LIST endpoint
        issue_contexts = self.manager.get_contexts_by_endpoint(GitHubEndpoint.ISSUE_LIST)
        
        assert len(issue_contexts) == 2
        context_ids = [c.context_id for c in issue_contexts]
        assert context1.context_id in context_ids
        assert context2.context_id in context_ids
        assert context3.context_id not in context_ids
    
    def test_get_contexts_by_state(self):
        """Test retrieving contexts by state"""
        # Create contexts in different states
        context1 = self.manager.create_context(
            GitHubEndpoint.ISSUE_CREATE, "create1", ["issue", "create"]
        )
        context2 = self.manager.create_context(
            GitHubEndpoint.ISSUE_CREATE, "create2", ["issue", "create"]
        )
        
        # Update states
        self.manager.update_context_state(context1.context_id, RequestState.EXECUTING)
        self.manager.update_context_state(context2.context_id, RequestState.FAILED)
        
        # Get contexts by state
        executing_contexts = self.manager.get_contexts_by_state(RequestState.EXECUTING)
        failed_contexts = self.manager.get_contexts_by_state(RequestState.FAILED)
        
        assert len(executing_contexts) == 1
        assert executing_contexts[0].context_id == context1.context_id
        
        assert len(failed_contexts) == 1
        assert failed_contexts[0].context_id == context2.context_id
    
    def test_get_recoverable_contexts(self):
        """Test retrieving recoverable contexts"""
        # Create contexts with different recoverability
        context1 = self.manager.create_context(
            GitHubEndpoint.ISSUE_EDIT, "edit1", ["issue", "edit", "1"], max_attempts=3
        )
        context2 = self.manager.create_context(
            GitHubEndpoint.ISSUE_EDIT, "edit2", ["issue", "edit", "2"], max_attempts=2
        )
        context3 = self.manager.create_context(
            GitHubEndpoint.ISSUE_EDIT, "edit3", ["issue", "edit", "3"], max_attempts=1
        )
        
        # Set states and attempt counts
        self.manager.update_context_state(context1.context_id, RequestState.FAILED)  # 1 attempt, recoverable
        self.manager.update_context_state(context2.context_id, RequestState.RETRYING)  # 1 attempt, recoverable
        
        # Context 3: exhaust attempts
        self.manager.update_context_state(context3.context_id, RequestState.FAILED)  # 1 attempt, max reached
        
        recoverable = self.manager.get_recoverable_contexts()
        
        # Should have 2 recoverable contexts
        assert len(recoverable) == 2
        recoverable_ids = [c.context_id for c in recoverable]
        assert context1.context_id in recoverable_ids
        assert context2.context_id in recoverable_ids
        assert context3.context_id not in recoverable_ids
    
    def test_complete_context_success(self):
        """Test completing a context successfully"""
        context = self.manager.create_context(
            GitHubEndpoint.REPO_INFO, "get_info", ["repo", "view"]
        )
        
        context_id = context.context_id
        final_results = {"name": "test-repo", "stars": 100}
        
        success = self.manager.complete_context(context_id, final_results, True)
        
        assert success is True
        
        # Should be moved to completed contexts
        assert context_id not in self.manager.active_contexts
        assert context_id in self.manager.completed_contexts
        
        completed_context = self.manager.completed_contexts[context_id]
        assert completed_context.state == RequestState.COMPLETED
        assert completed_context.partial_results == final_results
    
    def test_complete_context_failure(self):
        """Test completing a context with failure"""
        context = self.manager.create_context(
            GitHubEndpoint.BULK_OPERATIONS, "bulk_op", ["bulk", "operation"]
        )
        
        context_id = context.context_id
        
        success = self.manager.complete_context(context_id, None, False)
        
        assert success is True
        
        # Should be moved to completed contexts
        assert context_id not in self.manager.active_contexts
        assert context_id in self.manager.completed_contexts
        
        completed_context = self.manager.completed_contexts[context_id]
        assert completed_context.state == RequestState.FAILED
    
    def test_abandon_context(self):
        """Test abandoning a context"""
        context = self.manager.create_context(
            GitHubEndpoint.SEARCH, "search", ["search", "issues", "test"]
        )
        
        context_id = context.context_id
        reason = "Maximum attempts reached"
        
        success = self.manager.abandon_context(context_id, reason)
        
        assert success is True
        
        # Should be moved to completed contexts
        assert context_id not in self.manager.active_contexts
        assert context_id in self.manager.completed_contexts
        
        abandoned_context = self.manager.completed_contexts[context_id]
        assert abandoned_context.state == RequestState.ABANDONED
        assert len(abandoned_context.error_history) > 0
        
        abandonment_error = abandoned_context.error_history[-1]
        assert abandonment_error["type"] == "abandonment"
        assert abandonment_error["reason"] == reason
    
    def test_context_snapshot_and_restore(self):
        """Test creating and restoring context snapshots"""
        context = self.manager.create_context(
            GitHubEndpoint.PR_CREATE, "create_pr", ["pr", "create", "--title", "Test"],
            partial_results={"pr_id": 123},
            intermediate_state={"validation_passed": True}
        )
        
        context_id = context.context_id
        
        # Update context state
        self.manager.update_context_state(
            context_id,
            RequestState.EXECUTING,
            continuation_data={"next_step": "create_branch"}
        )
        
        # Create snapshot
        snapshot = self.manager.create_context_snapshot(context_id)
        
        assert snapshot is not None
        assert snapshot["context_id"] == context_id
        assert snapshot["state"] == RequestState.EXECUTING.value
        assert "snapshot_timestamp" in snapshot
        assert "snapshot_id" in snapshot
        
        # Remove original context
        self.manager.complete_context(context_id, None, False)
        assert context_id not in self.manager.active_contexts
        
        # Restore from snapshot
        restored_context = self.manager.restore_context_from_snapshot(snapshot)
        
        assert restored_context is not None
        assert restored_context.context_id == context_id
        assert restored_context.state == RequestState.RECOVERED
        assert restored_context.continuation_data == {"next_step": "create_branch"}
        
        # Should be back in active contexts
        assert context_id in self.manager.active_contexts
    
    def test_context_persistence_session_persistent(self):
        """Test context persistence for session persistent scope"""
        context = self.manager.create_context(
            GitHubEndpoint.ISSUE_CREATE,
            "create_issue",
            ["issue", "create", "--title", "Test"],
            scope=ContextScope.SESSION_PERSISTENT
        )
        
        context_id = context.context_id
        
        # Check that context file was created
        active_dir = self.storage_path / "active"
        context_file = active_dir / f"context_{context_id}.json"
        
        assert context_file.exists()
        
        # Verify file content
        with open(context_file, 'r') as f:
            data = json.load(f)
        
        assert data["context_id"] == context_id
        assert data["endpoint"] == GitHubEndpoint.ISSUE_CREATE.value
        assert data["scope"] == ContextScope.SESSION_PERSISTENT.value
    
    def test_context_persistence_memory_only(self):
        """Test that memory-only contexts are not persisted"""
        context = self.manager.create_context(
            GitHubEndpoint.ISSUE_VIEW,
            "view_issue",
            ["issue", "view", "123"],
            scope=ContextScope.MEMORY_ONLY
        )
        
        context_id = context.context_id
        
        # Check that no context file was created
        active_dir = self.storage_path / "active"
        context_file = active_dir / f"context_{context_id}.json"
        
        assert not context_file.exists()
    
    def test_context_loading_on_initialization(self):
        """Test loading persisted contexts on manager initialization"""
        # Create and persist a context manually
        test_context_data = {
            "context_id": "test-load-context",
            "endpoint": GitHubEndpoint.ISSUE_LIST.value,
            "operation_type": "list_issues",
            "command_args": ["issue", "list"],
            "environment": {},
            "working_directory": ".",
            "state": RequestState.RETRYING.value,
            "created_at": datetime.now().isoformat(),
            "last_attempt": datetime.now().isoformat(),
            "attempt_count": 2,
            "max_attempts": 5,
            "partial_results": None,
            "intermediate_state": None,
            "continuation_data": None,
            "timeout_used": None,
            "error_history": [],
            "recovery_strategy": None,
            "priority": 3,
            "scope": ContextScope.SESSION_PERSISTENT.value,
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "tags": ["test"]
        }
        
        # Write to active contexts directory
        active_dir = self.storage_path / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        
        context_file = active_dir / "context_test-load-context.json"
        with open(context_file, 'w') as f:
            json.dump(test_context_data, f)
        
        # Create new manager instance (should load the context)
        new_manager = GitHubRequestContextManager(str(self.storage_path))
        
        # Check that context was loaded
        assert "test-load-context" in new_manager.active_contexts
        
        loaded_context = new_manager.active_contexts["test-load-context"]
        assert loaded_context.context_id == "test-load-context"
        assert loaded_context.endpoint == GitHubEndpoint.ISSUE_LIST
        assert loaded_context.state == RequestState.RETRYING
        assert loaded_context.attempt_count == 2
    
    def test_context_stats(self):
        """Test context statistics generation"""
        # Create contexts in various states
        context1 = self.manager.create_context(
            GitHubEndpoint.ISSUE_LIST, "list1", ["issue", "list"], priority=1
        )
        context2 = self.manager.create_context(
            GitHubEndpoint.ISSUE_CREATE, "create1", ["issue", "create"], priority=2
        )
        context3 = self.manager.create_context(
            GitHubEndpoint.PR_LIST, "pr1", ["pr", "list"], priority=1
        )
        
        # Update states
        self.manager.update_context_state(context1.context_id, RequestState.EXECUTING)
        self.manager.update_context_state(context2.context_id, RequestState.FAILED)
        self.manager.complete_context(context3.context_id, {"result": "success"}, True)
        
        stats = self.manager.get_context_stats()
        
        assert stats["active_contexts"] == 2  # 1 and 2 are still active
        assert stats["completed_contexts"] == 1  # 3 was completed
        
        assert stats["contexts_by_state"]["executing"] == 1
        assert stats["contexts_by_state"]["failed"] == 1
        
        assert stats["contexts_by_endpoint"][GitHubEndpoint.ISSUE_LIST.value] == 1
        assert stats["contexts_by_endpoint"][GitHubEndpoint.ISSUE_CREATE.value] == 1
        assert stats["contexts_by_endpoint"][GitHubEndpoint.PR_LIST.value] == 1
        
        assert stats["contexts_by_priority"][1] == 1  # Only active context with priority 1
        assert stats["contexts_by_priority"][2] == 1

class TestContextManagerSingleton:
    """Test singleton pattern for context manager"""
    
    def test_get_context_manager_singleton(self):
        """Test that get_context_manager returns the same instance"""
        # Clear any existing instance
        import claude.commands.github_request_context as ctx_module
        ctx_module._context_manager = None
        
        manager1 = get_context_manager()
        manager2 = get_context_manager()
        
        assert manager1 is manager2

class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent access"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = GitHubRequestContextManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_concurrent_context_creation(self):
        """Test concurrent context creation from multiple threads"""
        num_threads = 10
        contexts_per_thread = 5
        created_contexts = []
        
        def create_contexts():
            thread_contexts = []
            for i in range(contexts_per_thread):
                context = self.manager.create_context(
                    GitHubEndpoint.ISSUE_LIST,
                    f"thread_operation_{threading.current_thread().ident}_{i}",
                    ["issue", "list"]
                )
                thread_contexts.append(context.context_id)
                time.sleep(0.001)  # Small delay
            created_contexts.extend(thread_contexts)
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=create_contexts)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        expected_total = num_threads * contexts_per_thread
        assert len(created_contexts) == expected_total
        assert len(self.manager.active_contexts) == expected_total
        
        # Check that all context IDs are unique
        assert len(set(created_contexts)) == expected_total
    
    def test_concurrent_context_updates(self):
        """Test concurrent context state updates"""
        # Create a context
        context = self.manager.create_context(
            GitHubEndpoint.BULK_OPERATIONS,
            "bulk_test",
            ["bulk", "operation"]
        )
        
        context_id = context.context_id
        num_threads = 20
        updates_per_thread = 10
        
        def update_context():
            for i in range(updates_per_thread):
                self.manager.update_context_state(
                    context_id,
                    RequestState.RETRYING,
                    partial_results={f"update_{threading.current_thread().ident}": i},
                    error_info={
                        "thread": threading.current_thread().ident,
                        "iteration": i
                    }
                )
                time.sleep(0.001)
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=update_context)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check final state
        final_context = self.manager.get_context(context_id)
        assert final_context is not None
        assert final_context.state == RequestState.RETRYING
        
        # Should have many error history entries
        expected_errors = num_threads * updates_per_thread
        assert len(final_context.error_history) == expected_errors

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = GitHubRequestContextManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_update_nonexistent_context(self):
        """Test updating a context that doesn't exist"""
        success = self.manager.update_context_state(
            "nonexistent-context",
            RequestState.EXECUTING
        )
        
        assert success is False
    
    def test_get_nonexistent_context(self):
        """Test retrieving a context that doesn't exist"""
        context = self.manager.get_context("nonexistent-context")
        assert context is None
    
    def test_complete_nonexistent_context(self):
        """Test completing a context that doesn't exist"""
        success = self.manager.complete_context("nonexistent-context")
        assert success is False
    
    def test_snapshot_nonexistent_context(self):
        """Test creating snapshot of nonexistent context"""
        snapshot = self.manager.create_context_snapshot("nonexistent-context")
        assert snapshot is None
    
    def test_context_expiry_handling(self):
        """Test handling of expired contexts"""
        # Create context that expires immediately
        context = self.manager.create_context(
            GitHubEndpoint.ISSUE_VIEW,
            "expiring_context",
            ["issue", "view", "123"],
            expiry_hours=0  # Expires immediately
        )
        
        context_id = context.context_id
        
        # Manually expire the context
        self.manager.active_contexts[context_id].expires_at = datetime.now() - timedelta(seconds=1)
        
        # Trigger cleanup
        self.manager._cleanup_expired_contexts()
        
        # Context should be abandoned
        assert context_id not in self.manager.active_contexts
        assert context_id in self.manager.completed_contexts
        
        abandoned_context = self.manager.completed_contexts[context_id]
        assert abandoned_context.state == RequestState.ABANDONED
    
    def test_invalid_snapshot_restore(self):
        """Test restoring from invalid snapshot data"""
        invalid_snapshot = {
            "context_id": "invalid",
            "endpoint": "invalid_endpoint",  # Invalid enum value
            "operation_type": "test",
            "state": "invalid_state"  # Invalid enum value
        }
        
        restored = self.manager.restore_context_from_snapshot(invalid_snapshot)
        assert restored is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])