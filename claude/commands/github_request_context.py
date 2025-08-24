#!/usr/bin/env python3
"""
GitHub Request Context - Request Context Preservation
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Request context preservation system for maintaining request state during timeout recovery,
enabling seamless continuation of operations after network failures.
"""

import json
import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pickle
import hashlib

from .github_timeout_manager import GitHubEndpoint

logger = logging.getLogger(__name__)

class RequestState(Enum):
    """Request execution state"""
    INITIALIZED = "initialized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    RECOVERED = "recovered"
    ABANDONED = "abandoned"

class ContextScope(Enum):
    """Scope of context preservation"""
    MEMORY_ONLY = "memory_only"      # Keep in memory during session
    SESSION_PERSISTENT = "session_persistent"  # Persist across session restarts
    OPERATION_CHAIN = "operation_chain"       # Preserve for chained operations

@dataclass
class RequestContext:
    """
    Context information for a GitHub API request that needs to be preserved
    across timeout/retry cycles.
    """
    context_id: str
    endpoint: GitHubEndpoint
    operation_type: str  # e.g., "issue_create", "comment_post", "pr_update"
    
    # Request details
    command_args: List[str]
    environment: Dict[str, str]
    working_directory: str
    
    # State information
    state: RequestState
    created_at: datetime
    last_attempt: Optional[datetime]
    attempt_count: int
    max_attempts: int
    
    # Execution context
    partial_results: Optional[Dict[str, Any]]
    intermediate_state: Optional[Dict[str, Any]]
    continuation_data: Optional[Dict[str, Any]]
    
    # Recovery information
    timeout_used: Optional[float]
    error_history: List[Dict[str, Any]]
    recovery_strategy: Optional[str]
    
    # Metadata
    priority: int  # 1=critical, 5=low
    scope: ContextScope
    expires_at: Optional[datetime]
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "context_id": self.context_id,
            "endpoint": self.endpoint.value,
            "operation_type": self.operation_type,
            "command_args": self.command_args,
            "environment": self.environment,
            "working_directory": self.working_directory,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "partial_results": self.partial_results,
            "intermediate_state": self.intermediate_state,
            "continuation_data": self.continuation_data,
            "timeout_used": self.timeout_used,
            "error_history": self.error_history,
            "recovery_strategy": self.recovery_strategy,
            "priority": self.priority,
            "scope": self.scope.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestContext':
        """Create from serialized dictionary"""
        return cls(
            context_id=data["context_id"],
            endpoint=GitHubEndpoint(data["endpoint"]),
            operation_type=data["operation_type"],
            command_args=data["command_args"],
            environment=data["environment"],
            working_directory=data["working_directory"],
            state=RequestState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_attempt=datetime.fromisoformat(data["last_attempt"]) if data["last_attempt"] else None,
            attempt_count=data["attempt_count"],
            max_attempts=data["max_attempts"],
            partial_results=data["partial_results"],
            intermediate_state=data["intermediate_state"],
            continuation_data=data["continuation_data"],
            timeout_used=data["timeout_used"],
            error_history=data["error_history"],
            recovery_strategy=data["recovery_strategy"],
            priority=data["priority"],
            scope=ContextScope(data["scope"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            tags=data["tags"]
        )

class GitHubRequestContextManager:
    """
    Manages request context preservation for GitHub API operations.
    Provides state preservation, recovery coordination, and continuation support.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "knowledge/context/github_requests")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory context storage
        self.active_contexts: Dict[str, RequestContext] = {}
        self.completed_contexts: Dict[str, RequestContext] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Cleanup configuration
        self.cleanup_interval = 3600  # 1 hour
        self.max_context_age = 24 * 3600  # 24 hours
        
        # Load persisted contexts
        self._load_persisted_contexts()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Initialized GitHub Request Context Manager with {len(self.active_contexts)} active contexts")
    
    def create_context(
        self,
        endpoint: GitHubEndpoint,
        operation_type: str,
        command_args: List[str],
        environment: Optional[Dict[str, str]] = None,
        working_directory: Optional[str] = None,
        priority: int = 3,
        scope: ContextScope = ContextScope.SESSION_PERSISTENT,
        max_attempts: int = 5,
        expiry_hours: int = 24,
        tags: Optional[List[str]] = None,
        partial_results: Optional[Any] = None,
        intermediate_state: Optional[Dict[str, Any]] = None
    ) -> RequestContext:
        """
        Create a new request context for preservation.
        
        Args:
            endpoint: GitHub endpoint type
            operation_type: Type of operation being performed
            command_args: Command arguments for the GitHub CLI call
            environment: Environment variables for the request
            working_directory: Working directory for command execution
            priority: Request priority (1=critical, 5=low)
            scope: Context preservation scope
            max_attempts: Maximum retry attempts
            expiry_hours: Hours until context expires
            tags: Optional tags for categorization
            partial_results: Optional partial results from previous attempts
            intermediate_state: Optional intermediate state data
            
        Returns:
            Created RequestContext
        """
        context_id = str(uuid.uuid4())
        
        context = RequestContext(
            context_id=context_id,
            endpoint=endpoint,
            operation_type=operation_type,
            command_args=command_args,
            environment=environment or {},
            working_directory=working_directory or ".",
            state=RequestState.INITIALIZED,
            created_at=datetime.now(),
            last_attempt=None,
            attempt_count=0,
            max_attempts=max_attempts,
            partial_results=partial_results,
            intermediate_state=intermediate_state,
            continuation_data=None,
            timeout_used=None,
            error_history=[],
            recovery_strategy=None,
            priority=priority,
            scope=scope,
            expires_at=datetime.now() + timedelta(hours=expiry_hours) if expiry_hours else None,
            tags=tags or []
        )
        
        with self.lock:
            self.active_contexts[context_id] = context
            
            # Persist if required
            if scope in [ContextScope.SESSION_PERSISTENT, ContextScope.OPERATION_CHAIN]:
                self._persist_context(context)
        
        logger.info(f"Created request context {context_id} for {operation_type}")
        return context
    
    def update_context_state(
        self,
        context_id: str,
        state: RequestState,
        partial_results: Optional[Dict[str, Any]] = None,
        intermediate_state: Optional[Dict[str, Any]] = None,
        continuation_data: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the state of an existing context.
        
        Args:
            context_id: Context identifier
            state: New request state
            partial_results: Partial results from the request
            intermediate_state: Intermediate state data
            continuation_data: Data needed for continuation
            error_info: Error information if applicable
            
        Returns:
            True if context was updated successfully
        """
        with self.lock:
            context = self.active_contexts.get(context_id)
            if not context:
                logger.warning(f"Context {context_id} not found for state update")
                return False
            
            context.state = state
            context.last_attempt = datetime.now()
            
            if state in [RequestState.EXECUTING, RequestState.RETRYING]:
                context.attempt_count += 1
            
            # If going to FAILED state and attempt count is still 0, increment it
            # This handles cases where we go directly to FAILED state
            if state == RequestState.FAILED and context.attempt_count == 0:
                context.attempt_count += 1
            
            if partial_results is not None:
                context.partial_results = partial_results
            
            if intermediate_state is not None:
                context.intermediate_state = intermediate_state
            
            if continuation_data is not None:
                context.continuation_data = continuation_data
            
            if error_info:
                context.error_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "attempt": context.attempt_count,
                    **error_info
                })
            
            # Persist if required
            if context.scope in [ContextScope.SESSION_PERSISTENT, ContextScope.OPERATION_CHAIN]:
                self._persist_context(context)
            
            logger.debug(f"Updated context {context_id} to state {state.value}")
            return True
    
    def get_context(self, context_id: str) -> Optional[RequestContext]:
        """Get context by ID"""
        with self.lock:
            return self.active_contexts.get(context_id)
    
    def get_contexts_by_endpoint(self, endpoint: GitHubEndpoint) -> List[RequestContext]:
        """Get all contexts for a specific endpoint"""
        with self.lock:
            return [ctx for ctx in self.active_contexts.values() if ctx.endpoint == endpoint]
    
    def get_contexts_by_state(self, state: RequestState) -> List[RequestContext]:
        """Get all contexts in a specific state"""
        with self.lock:
            return [ctx for ctx in self.active_contexts.values() if ctx.state == state]
    
    def get_recoverable_contexts(self) -> List[RequestContext]:
        """Get contexts that can be recovered/retried"""
        with self.lock:
            recoverable = []
            for context in self.active_contexts.values():
                if (context.state in [RequestState.FAILED, RequestState.RETRYING] and
                    context.attempt_count < context.max_attempts and
                    (context.expires_at is None or context.expires_at > datetime.now())):
                    recoverable.append(context)
            
            # Sort by priority and creation time
            recoverable.sort(key=lambda x: (x.priority, x.created_at))
            return recoverable
    
    def complete_context(
        self,
        context_id: str,
        final_results: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> bool:
        """
        Mark a context as completed and move to completed storage.
        
        Args:
            context_id: Context identifier
            final_results: Final results from the operation
            success: Whether the operation completed successfully
            
        Returns:
            True if context was completed successfully
        """
        with self.lock:
            context = self.active_contexts.get(context_id)
            if not context:
                return False
            
            context.state = RequestState.COMPLETED if success else RequestState.FAILED
            context.last_attempt = datetime.now()
            
            if final_results:
                context.partial_results = final_results
            
            # Move to completed contexts
            self.completed_contexts[context_id] = context
            del self.active_contexts[context_id]
            
            # Update persistence
            self._persist_context(context, completed=True)
            
            logger.info(f"Completed context {context_id} with success={success}")
            return True
    
    def abandon_context(self, context_id: str, reason: str = "abandoned") -> bool:
        """
        Abandon a context (max attempts reached, expired, etc.)
        
        Args:
            context_id: Context identifier
            reason: Reason for abandonment
            
        Returns:
            True if context was abandoned successfully
        """
        with self.lock:
            context = self.active_contexts.get(context_id)
            if not context:
                return False
            
            context.state = RequestState.ABANDONED
            context.error_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "abandonment",
                "reason": reason
            })
            
            # Move to completed contexts
            self.completed_contexts[context_id] = context
            del self.active_contexts[context_id]
            
            # Update persistence
            self._persist_context(context, completed=True)
            
            logger.warning(f"Abandoned context {context_id}: {reason}")
            return True
    
    def create_context_snapshot(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Create a snapshot of context state for recovery"""
        with self.lock:
            context = self.active_contexts.get(context_id)
            if not context:
                return None
            
            snapshot = context.to_dict()
            snapshot["snapshot_timestamp"] = datetime.now().isoformat()
            snapshot["snapshot_id"] = str(uuid.uuid4())
            
            return snapshot
    
    def restore_context_from_snapshot(self, snapshot: Dict[str, Any]) -> Optional[RequestContext]:
        """Restore context from a snapshot"""
        try:
            context = RequestContext.from_dict(snapshot)
            context.state = RequestState.RECOVERED
            
            with self.lock:
                self.active_contexts[context.context_id] = context
                
                if context.scope in [ContextScope.SESSION_PERSISTENT, ContextScope.OPERATION_CHAIN]:
                    self._persist_context(context)
            
            logger.info(f"Restored context {context.context_id} from snapshot")
            return context
            
        except Exception as e:
            logger.error(f"Failed to restore context from snapshot: {e}")
            return None
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get comprehensive context statistics"""
        with self.lock:
            stats = {
                "active_contexts": len(self.active_contexts),
                "completed_contexts": len(self.completed_contexts),
                "contexts_by_state": {},
                "contexts_by_endpoint": {},
                "contexts_by_priority": {},
                "average_attempt_count": 0,
                "recovery_success_rate": 0
            }
            
            # Count by state
            for context in self.active_contexts.values():
                state_name = context.state.value
                stats["contexts_by_state"][state_name] = stats["contexts_by_state"].get(state_name, 0) + 1
            
            # Count by endpoint
            for context in list(self.active_contexts.values()) + list(self.completed_contexts.values()):
                endpoint_name = context.endpoint.value
                stats["contexts_by_endpoint"][endpoint_name] = stats["contexts_by_endpoint"].get(endpoint_name, 0) + 1
            
            # Count by priority
            for context in self.active_contexts.values():
                priority = context.priority
                stats["contexts_by_priority"][priority] = stats["contexts_by_priority"].get(priority, 0) + 1
            
            # Calculate averages
            all_contexts = list(self.active_contexts.values()) + list(self.completed_contexts.values())
            if all_contexts:
                stats["average_attempt_count"] = sum(c.attempt_count for c in all_contexts) / len(all_contexts)
                
                # Calculate recovery success rate
                recovered_contexts = [c for c in all_contexts if c.state == RequestState.RECOVERED]
                failed_contexts = [c for c in all_contexts if c.state in [RequestState.FAILED, RequestState.ABANDONED]]
                total_recovery_attempts = len(recovered_contexts) + len(failed_contexts)
                
                if total_recovery_attempts > 0:
                    stats["recovery_success_rate"] = len(recovered_contexts) / total_recovery_attempts
            
            return stats
    
    def _persist_context(self, context: RequestContext, completed: bool = False):
        """Persist context to disk"""
        try:
            filename = f"context_{context.context_id}.json"
            subdir = "completed" if completed else "active"
            filepath = self.storage_path / subdir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
                
            # Remove from other directory if moving
            if completed:
                old_filepath = self.storage_path / "active" / filename
                if old_filepath.exists():
                    old_filepath.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to persist context {context.context_id}: {e}")
    
    def _load_persisted_contexts(self):
        """Load persisted contexts from disk"""
        try:
            # Load active contexts
            active_dir = self.storage_path / "active"
            if active_dir.exists():
                for context_file in active_dir.glob("context_*.json"):
                    try:
                        with open(context_file, 'r') as f:
                            data = json.load(f)
                        
                        context = RequestContext.from_dict(data)
                        
                        # Check if context has expired
                        if context.expires_at and context.expires_at < datetime.now():
                            self.abandon_context(context.context_id, "expired")
                            continue
                        
                        self.active_contexts[context.context_id] = context
                        
                    except Exception as e:
                        logger.warning(f"Failed to load context from {context_file}: {e}")
            
            # Load recent completed contexts (last 24 hours)
            completed_dir = self.storage_path / "completed"
            if completed_dir.exists():
                cutoff_time = datetime.now() - timedelta(hours=24)
                for context_file in completed_dir.glob("context_*.json"):
                    try:
                        if context_file.stat().st_mtime < cutoff_time.timestamp():
                            continue  # Skip old files
                        
                        with open(context_file, 'r') as f:
                            data = json.load(f)
                        
                        context = RequestContext.from_dict(data)
                        self.completed_contexts[context.context_id] = context
                        
                    except Exception as e:
                        logger.warning(f"Failed to load completed context from {context_file}: {e}")
            
            logger.info(f"Loaded {len(self.active_contexts)} active and {len(self.completed_contexts)} completed contexts")
            
        except Exception as e:
            logger.error(f"Failed to load persisted contexts: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_contexts()
                    self._cleanup_old_files()
                except Exception as e:
                    logger.error(f"Context cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
        logger.info("Started context cleanup thread")
    
    def _cleanup_expired_contexts(self):
        """Clean up expired contexts"""
        with self.lock:
            expired_contexts = []
            current_time = datetime.now()
            
            for context_id, context in self.active_contexts.items():
                if context.expires_at and context.expires_at < current_time:
                    expired_contexts.append(context_id)
            
            for context_id in expired_contexts:
                self.abandon_context(context_id, "expired")
            
            if expired_contexts:
                logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")
    
    def _cleanup_old_files(self):
        """Clean up old persisted context files"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.max_context_age)
            
            for subdir in ["active", "completed"]:
                context_dir = self.storage_path / subdir
                if not context_dir.exists():
                    continue
                
                old_files = []
                for context_file in context_dir.glob("context_*.json"):
                    if context_file.stat().st_mtime < cutoff_time.timestamp():
                        old_files.append(context_file)
                
                for old_file in old_files:
                    try:
                        old_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete old context file {old_file}: {e}")
                
                if old_files:
                    logger.info(f"Cleaned up {len(old_files)} old context files from {subdir}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old context files: {e}")

# Global context manager instance (singleton pattern)
_context_manager = None

def get_context_manager() -> GitHubRequestContextManager:
    """Get the global context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = GitHubRequestContextManager()
    return _context_manager