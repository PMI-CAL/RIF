#!/usr/bin/env python3
"""
Conversation Session Manager for RIF Agents

Provides robust session management for conversation tracking including:
- Session lifecycle management (start, end, pause, resume)
- Active session tracking and monitoring  
- Session recovery after interruptions
- Session metadata and context preservation
- Automated session cleanup
"""

import uuid
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .storage_backend import ConversationStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Rich metadata for conversation sessions"""
    session_id: str
    agent_type: str
    issue_number: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # active, paused, completed, failed, recovered
    event_count: int
    error_count: int  
    decision_count: int
    context_summary: Optional[str]
    last_activity: datetime
    recovery_count: int
    parent_session_id: Optional[str]  # For nested sessions
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds"""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status in ['active', 'paused']
    
    @property
    def activity_age_minutes(self) -> float:
        """Get minutes since last activity"""
        return (datetime.utcnow() - self.last_activity).total_seconds() / 60


class ConversationSessionManager:
    """
    Robust session management system for conversation tracking.
    
    Provides lifecycle management, recovery, monitoring, and cleanup
    capabilities for agent conversations.
    """
    
    def __init__(self, 
                 storage_backend: Optional[ConversationStorageBackend] = None,
                 session_timeout_minutes: int = 60,
                 cleanup_interval_minutes: int = 30,
                 max_recovery_attempts: int = 3):
        """
        Initialize conversation session manager.
        
        Args:
            storage_backend: Storage backend for persistence
            session_timeout_minutes: Session timeout for cleanup
            cleanup_interval_minutes: Interval for automated cleanup
            max_recovery_attempts: Maximum recovery attempts per session
        """
        self.storage = storage_backend or ConversationStorageBackend()
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.max_recovery_attempts = max_recovery_attempts
        
        # Active session tracking
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.paused_sessions: Dict[str, SessionMetadata] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring and cleanup
        self.cleanup_thread = None
        self.cleanup_enabled = True
        
        # Performance tracking
        self.session_stats = {
            'sessions_started': 0,
            'sessions_completed': 0,
            'sessions_failed': 0,
            'sessions_recovered': 0,
            'sessions_cleaned_up': 0,
            'avg_session_duration_minutes': 0.0,
            'total_events_captured': 0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        # Recover any interrupted sessions on startup
        self._recover_interrupted_sessions()
        
        logger.info(f"ConversationSessionManager initialized with {session_timeout_minutes}min timeout")
    
    def start_session(self, 
                     agent_type: str,
                     issue_number: Optional[int] = None,
                     context_summary: Optional[str] = None,
                     parent_session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            agent_type: Type of agent starting the session
            issue_number: Associated GitHub issue number
            context_summary: Summary of conversation context
            parent_session_id: Parent session ID for nested sessions
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        with self._lock:
            # Create session metadata
            session = SessionMetadata(
                session_id=session_id,
                agent_type=agent_type,
                issue_number=issue_number,
                start_time=start_time,
                end_time=None,
                status='active',
                event_count=0,
                error_count=0,
                decision_count=0,
                context_summary=context_summary,
                last_activity=start_time,
                recovery_count=0,
                parent_session_id=parent_session_id
            )
            
            # Store in active sessions
            self.active_sessions[session_id] = session
            
            # Persist to storage
            try:
                self.storage.start_conversation(
                    agent_type=agent_type,
                    issue_number=issue_number,
                    context_summary=context_summary
                )
                logger.info(f"Started session {session_id} for {agent_type}")
            except Exception as e:
                logger.error(f"Failed to persist session start: {e}")
                # Continue anyway - session exists in memory
            
            # Update stats
            self.session_stats['sessions_started'] += 1
        
        return session_id
    
    def end_session(self, 
                   session_id: str, 
                   success: bool = True,
                   context_summary: Optional[str] = None,
                   final_stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        End a conversation session.
        
        Args:
            session_id: Session to end
            success: Whether session completed successfully
            context_summary: Final context summary
            final_stats: Final session statistics
            
        Returns:
            True if session ended successfully
        """
        with self._lock:
            # Find session (check active and paused)
            session = self.active_sessions.get(session_id) or self.paused_sessions.get(session_id)
            
            if not session:
                logger.warning(f"Attempted to end non-existent session {session_id}")
                return False
            
            # Update session metadata
            session.end_time = datetime.utcnow()
            session.status = 'completed' if success else 'failed'
            
            if context_summary:
                session.context_summary = context_summary
            
            # Remove from active/paused sessions
            self.active_sessions.pop(session_id, None)
            self.paused_sessions.pop(session_id, None)
            
            # Persist to storage
            try:
                self.storage.end_conversation(
                    session_id,
                    success=success,
                    context_summary=session.context_summary
                )
            except Exception as e:
                logger.error(f"Failed to persist session end: {e}")
            
            # Update statistics
            if success:
                self.session_stats['sessions_completed'] += 1
            else:
                self.session_stats['sessions_failed'] += 1
            
            # Update average duration
            self._update_average_duration(session.duration_seconds / 60)
            
            logger.info(f"Ended session {session_id} ({'success' if success else 'failure'}) after {session.duration_seconds:.1f}s")
            return True
    
    def pause_session(self, session_id: str, reason: Optional[str] = None) -> bool:
        """
        Pause an active session.
        
        Args:
            session_id: Session to pause
            reason: Optional reason for pausing
            
        Returns:
            True if session paused successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            
            if not session:
                logger.warning(f"Cannot pause non-existent active session {session_id}")
                return False
            
            # Move from active to paused
            session.status = 'paused'
            session.last_activity = datetime.utcnow()
            
            self.paused_sessions[session_id] = session
            del self.active_sessions[session_id]
            
            logger.info(f"Paused session {session_id}" + (f" ({reason})" if reason else ""))
            return True
    
    def resume_session(self, session_id: str) -> bool:
        """
        Resume a paused session.
        
        Args:
            session_id: Session to resume
            
        Returns:
            True if session resumed successfully
        """
        with self._lock:
            session = self.paused_sessions.get(session_id)
            
            if not session:
                logger.warning(f"Cannot resume non-existent paused session {session_id}")
                return False
            
            # Move from paused to active
            session.status = 'active'
            session.last_activity = datetime.utcnow()
            
            self.active_sessions[session_id] = session
            del self.paused_sessions[session_id]
            
            logger.info(f"Resumed session {session_id}")
            return True
    
    def recover_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Recover an interrupted session from storage.
        
        Args:
            session_id: Session ID to recover
            
        Returns:
            Recovered session metadata or None if recovery fails
        """
        try:
            # Get conversation metadata from storage
            result = self.storage.connection.execute(
                "SELECT * FROM conversation_metadata WHERE conversation_id = ?",
                [session_id]
            )
            
            row = result.fetchone()
            if not row:
                logger.warning(f"No stored data found for session {session_id}")
                return None
            
            # Parse row data
            columns = [col[0] for col in result.description]
            data = dict(zip(columns, row))
            
            # Create recovered session metadata
            recovered_session = SessionMetadata(
                session_id=session_id,
                agent_type=data['agent_type'],
                issue_number=data.get('issue_number'),
                start_time=data['start_timestamp'],
                end_time=data.get('end_timestamp'),
                status='recovered',
                event_count=data.get('total_events', 0),
                error_count=data.get('error_count', 0),
                decision_count=data.get('decision_count', 0),
                context_summary=data.get('context_summary'),
                last_activity=datetime.utcnow(),
                recovery_count=1,  # First recovery
                parent_session_id=None
            )
            
            with self._lock:
                # Add to active sessions if not completed
                if not recovered_session.end_time:
                    self.active_sessions[session_id] = recovered_session
                
                # Update stats
                self.session_stats['sessions_recovered'] += 1
            
            logger.info(f"Recovered session {session_id} with {recovered_session.event_count} events")
            return recovered_session
            
        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return None
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get metadata for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session metadata or None if not found
        """
        with self._lock:
            return (self.active_sessions.get(session_id) or 
                   self.paused_sessions.get(session_id))
    
    def get_active_sessions(self) -> Dict[str, SessionMetadata]:
        """Get all active session metadata"""
        with self._lock:
            return self.active_sessions.copy()
    
    def get_paused_sessions(self) -> Dict[str, SessionMetadata]:
        """Get all paused session metadata"""
        with self._lock:
            return self.paused_sessions.copy()
    
    def get_all_sessions(self) -> Dict[str, SessionMetadata]:
        """Get all session metadata (active and paused)"""
        with self._lock:
            all_sessions = {}
            all_sessions.update(self.active_sessions)
            all_sessions.update(self.paused_sessions)
            return all_sessions
    
    def update_session_activity(self, 
                               session_id: str,
                               event_type: Optional[str] = None,
                               increment_events: bool = True,
                               increment_errors: bool = False,
                               increment_decisions: bool = False) -> bool:
        """
        Update session activity tracking.
        
        Args:
            session_id: Session ID to update
            event_type: Type of event (for logging)
            increment_events: Whether to increment event count
            increment_errors: Whether to increment error count
            increment_decisions: Whether to increment decision count
            
        Returns:
            True if session updated successfully
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            
            if not session:
                logger.debug(f"Session {session_id} not found for activity update")
                return False
            
            # Update activity timestamp
            session.last_activity = datetime.utcnow()
            
            # Update counters
            if increment_events:
                session.event_count += 1
                self.session_stats['total_events_captured'] += 1
            
            if increment_errors:
                session.error_count += 1
            
            if increment_decisions:
                session.decision_count += 1
            
            return True
    
    def cleanup_stale_sessions(self, force: bool = False) -> Dict[str, int]:
        """
        Clean up stale sessions that have timed out.
        
        Args:
            force: Force cleanup regardless of timeout
            
        Returns:
            Cleanup statistics
        """
        cleanup_stats = {
            'active_cleaned': 0,
            'paused_cleaned': 0,
            'orphaned_cleaned': 0
        }
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.session_timeout_minutes)
        
        with self._lock:
            # Clean up stale active sessions
            stale_active = []
            for session_id, session in self.active_sessions.items():
                if force or session.last_activity < cutoff_time:
                    stale_active.append(session_id)
            
            for session_id in stale_active:
                session = self.active_sessions.pop(session_id)
                self.end_session(session_id, success=False, context_summary="Session timed out")
                cleanup_stats['active_cleaned'] += 1
                logger.info(f"Cleaned up stale active session {session_id}")
            
            # Clean up old paused sessions
            stale_paused = []
            for session_id, session in self.paused_sessions.items():
                if force or session.last_activity < cutoff_time:
                    stale_paused.append(session_id)
            
            for session_id in stale_paused:
                session = self.paused_sessions.pop(session_id)
                self.end_session(session_id, success=False, context_summary="Paused session timed out")
                cleanup_stats['paused_cleaned'] += 1
                logger.info(f"Cleaned up stale paused session {session_id}")
        
        # Clean up orphaned sessions in database
        try:
            orphaned_count = self._cleanup_orphaned_sessions(cutoff_time)
            cleanup_stats['orphaned_cleaned'] = orphaned_count
        except Exception as e:
            logger.error(f"Failed to clean up orphaned sessions: {e}")
        
        total_cleaned = sum(cleanup_stats.values())
        if total_cleaned > 0:
            self.session_stats['sessions_cleaned_up'] += total_cleaned
            logger.info(f"Cleanup completed: {cleanup_stats}")
        
        return cleanup_stats
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        with self._lock:
            stats = self.session_stats.copy()
            
            # Add current state
            stats.update({
                'active_sessions_count': len(self.active_sessions),
                'paused_sessions_count': len(self.paused_sessions),
                'total_sessions_in_memory': len(self.active_sessions) + len(self.paused_sessions),
                'session_timeout_minutes': self.session_timeout_minutes,
                'cleanup_interval_minutes': self.cleanup_interval_minutes
            })
            
            # Add session duration breakdown
            if self.active_sessions or self.paused_sessions:
                all_sessions = self.get_all_sessions()
                durations = [s.duration_seconds / 60 for s in all_sessions.values()]
                
                stats.update({
                    'current_avg_duration_minutes': sum(durations) / len(durations),
                    'longest_active_session_minutes': max(durations) if durations else 0,
                    'shortest_active_session_minutes': min(durations) if durations else 0
                })
        
        return stats
    
    def generate_session_summary(self, session_id: str) -> Optional[str]:
        """
        Generate a summary of session activity.
        
        Args:
            session_id: Session to summarize
            
        Returns:
            Session summary string or None if session not found
        """
        session = self.get_session_metadata(session_id)
        if not session:
            return None
        
        duration_str = f"{session.duration_seconds / 60:.1f} minutes"
        
        summary_parts = [
            f"Session {session_id} ({session.agent_type})",
            f"Duration: {duration_str}",
            f"Events: {session.event_count}",
            f"Status: {session.status}"
        ]
        
        if session.issue_number:
            summary_parts.insert(1, f"Issue: #{session.issue_number}")
        
        if session.error_count > 0:
            summary_parts.append(f"Errors: {session.error_count}")
        
        if session.decision_count > 0:
            summary_parts.append(f"Decisions: {session.decision_count}")
        
        if session.recovery_count > 0:
            summary_parts.append(f"Recovered: {session.recovery_count}x")
        
        return " | ".join(summary_parts)
    
    def _recover_interrupted_sessions(self):
        """Recover any interrupted sessions on startup"""
        try:
            # Find active conversations in database that aren't in memory
            result = self.storage.connection.execute("""
                SELECT conversation_id, agent_type, issue_number, start_timestamp 
                FROM conversation_metadata 
                WHERE status = 'active' AND end_timestamp IS NULL
            """)
            
            interrupted_sessions = result.fetchall()
            
            for session_data in interrupted_sessions:
                session_id = session_data[0]
                
                # Only recover if not already in memory
                if session_id not in self.active_sessions:
                    recovered = self.recover_session(session_id)
                    if recovered:
                        logger.info(f"Auto-recovered interrupted session {session_id}")
        
        except Exception as e:
            logger.error(f"Failed to recover interrupted sessions: {e}")
    
    def _cleanup_orphaned_sessions(self, cutoff_time: datetime) -> int:
        """Clean up orphaned sessions in database"""
        try:
            # Mark old active sessions as failed
            result = self.storage.connection.execute("""
                UPDATE conversation_metadata 
                SET status = 'timeout', end_timestamp = ?
                WHERE status = 'active' AND start_timestamp < ?
            """, [datetime.utcnow(), cutoff_time])
            
            return result.rowcount
            
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned sessions: {e}")
            return 0
    
    def _update_average_duration(self, duration_minutes: float):
        """Update average session duration statistics"""
        current_avg = self.session_stats['avg_session_duration_minutes']
        completed_sessions = self.session_stats['sessions_completed']
        
        if completed_sessions > 0:
            self.session_stats['avg_session_duration_minutes'] = (
                (current_avg * (completed_sessions - 1) + duration_minutes) / completed_sessions
            )
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while self.cleanup_enabled:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    if self.cleanup_enabled:
                        self.cleanup_stale_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"Started cleanup thread with {self.cleanup_interval_minutes}min interval")
    
    def shutdown(self):
        """Shutdown session manager and cleanup resources"""
        logger.info("Shutting down ConversationSessionManager...")
        
        # Stop cleanup thread
        self.cleanup_enabled = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # End all active sessions
        with self._lock:
            active_session_ids = list(self.active_sessions.keys())
            for session_id in active_session_ids:
                self.end_session(session_id, success=False, context_summary="Manager shutdown")
            
            paused_session_ids = list(self.paused_sessions.keys())
            for session_id in paused_session_ids:
                self.end_session(session_id, success=False, context_summary="Manager shutdown")
        
        # Close storage
        if self.storage:
            self.storage.close()
        
        logger.info("ConversationSessionManager shutdown complete")


# Convenience functions for integration

def create_session_manager(**kwargs) -> ConversationSessionManager:
    """Factory function to create session manager with default configuration"""
    return ConversationSessionManager(**kwargs)


def get_default_session_manager() -> ConversationSessionManager:
    """Get or create default session manager instance"""
    global _default_session_manager
    
    if '_default_session_manager' not in globals():
        _default_session_manager = create_session_manager()
    
    return _default_session_manager


# Context manager for easy session usage
class managed_conversation_session:
    """
    Context manager for automatic session lifecycle management.
    
    Usage:
        with managed_conversation_session("rif-implementer", issue_number=123) as session_id:
            # Conversation operations
            pass
    """
    
    def __init__(self, 
                 agent_type: str,
                 issue_number: Optional[int] = None,
                 context_summary: Optional[str] = None,
                 session_manager: Optional[ConversationSessionManager] = None):
        """
        Initialize managed session context.
        
        Args:
            agent_type: Type of agent
            issue_number: Associated GitHub issue number
            context_summary: Context summary
            session_manager: Session manager instance (uses default if None)
        """
        self.agent_type = agent_type
        self.issue_number = issue_number
        self.context_summary = context_summary
        self.session_manager = session_manager or get_default_session_manager()
        self.session_id = None
        self.success = True
    
    def __enter__(self) -> str:
        """Start managed session"""
        self.session_id = self.session_manager.start_session(
            agent_type=self.agent_type,
            issue_number=self.issue_number,
            context_summary=self.context_summary
        )
        return self.session_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End managed session"""
        if self.session_id:
            success = exc_type is None
            self.session_manager.end_session(
                self.session_id,
                success=success,
                context_summary="Exception occurred" if not success else None
            )


if __name__ == "__main__":
    # Example usage and testing
    print("RIF Conversation Session Manager")
    print("=" * 40)
    
    # Create session manager
    manager = create_session_manager(session_timeout_minutes=30)
    
    # Start a test session
    session_id = manager.start_session("rif-implementer", issue_number=123)
    print(f"Started session: {session_id}")
    
    # Simulate some activity
    manager.update_session_activity(session_id, "tool_use")
    manager.update_session_activity(session_id, "decision", increment_decisions=True)
    
    # Get session info
    session = manager.get_session_metadata(session_id)
    if session:
        print(f"Session info: {manager.generate_session_summary(session_id)}")
    
    # Test pause/resume
    manager.pause_session(session_id, "Testing pause functionality")
    print("Session paused")
    
    manager.resume_session(session_id)
    print("Session resumed")
    
    # Get statistics
    stats = manager.get_session_statistics()
    print(f"Active sessions: {stats['active_sessions_count']}")
    print(f"Total events captured: {stats['total_events_captured']}")
    
    # Test context manager
    with managed_conversation_session("rif-validator", issue_number=456) as ctx_session_id:
        print(f"Context manager session: {ctx_session_id}")
        # Session automatically ends when context exits
    
    # End test session
    manager.end_session(session_id, success=True)
    print("Session ended")
    
    # Show final statistics
    final_stats = manager.get_session_statistics()
    print(f"Total sessions completed: {final_stats['sessions_completed']}")
    
    # Shutdown
    manager.shutdown()
    print("Session manager shutdown complete")