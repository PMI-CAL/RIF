#!/usr/bin/env python3
"""
Test suite for Conversation Session Manager.

Tests session lifecycle management, recovery, monitoring, and cleanup functionality.
"""

import unittest
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from knowledge.conversations.session_manager import (
    ConversationSessionManager,
    SessionMetadata,
    managed_conversation_session,
    create_session_manager,
    get_default_session_manager
)


class TestSessionMetadata(unittest.TestCase):
    """Test session metadata data class"""
    
    def test_session_metadata_creation(self):
        """Test creating session metadata"""
        start_time = datetime.utcnow()
        
        metadata = SessionMetadata(
            session_id="test-123",
            agent_type="rif-implementer",
            issue_number=42,
            start_time=start_time,
            end_time=None,
            status="active",
            event_count=5,
            error_count=1,
            decision_count=2,
            context_summary="Test session",
            last_activity=start_time,
            recovery_count=0,
            parent_session_id=None
        )
        
        self.assertEqual(metadata.session_id, "test-123")
        self.assertEqual(metadata.agent_type, "rif-implementer")
        self.assertEqual(metadata.issue_number, 42)
        self.assertTrue(metadata.is_active)
        self.assertEqual(metadata.event_count, 5)
    
    def test_session_duration_calculation(self):
        """Test session duration calculation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=5)
        
        metadata = SessionMetadata(
            session_id="test-123",
            agent_type="rif-implementer",
            issue_number=None,
            start_time=start_time,
            end_time=end_time,
            status="completed",
            event_count=0,
            error_count=0,
            decision_count=0,
            context_summary=None,
            last_activity=start_time,
            recovery_count=0,
            parent_session_id=None
        )
        
        # Should be approximately 5 minutes (300 seconds)
        self.assertAlmostEqual(metadata.duration_seconds, 300, delta=1)
    
    def test_activity_age_calculation(self):
        """Test activity age calculation"""
        old_time = datetime.utcnow() - timedelta(minutes=10)
        
        metadata = SessionMetadata(
            session_id="test-123",
            agent_type="rif-implementer",
            issue_number=None,
            start_time=old_time,
            end_time=None,
            status="active",
            event_count=0,
            error_count=0,
            decision_count=0,
            context_summary=None,
            last_activity=old_time,
            recovery_count=0,
            parent_session_id=None
        )
        
        # Should be approximately 10 minutes
        self.assertAlmostEqual(metadata.activity_age_minutes, 10, delta=1)


class TestConversationSessionManager(unittest.TestCase):
    """Test the main session manager class"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock storage backend
        self.mock_storage = Mock()
        self.mock_storage.start_conversation = Mock(return_value="test-conversation-id")
        self.mock_storage.end_conversation = Mock()
        self.mock_storage.close = Mock()
        
        # Mock database queries for recovery
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        self.mock_storage.connection.execute.return_value = mock_result
        
        # Create session manager with mocked storage and short timeouts for testing
        self.manager = ConversationSessionManager(
            storage_backend=self.mock_storage,
            session_timeout_minutes=1,  # 1 minute timeout for testing
            cleanup_interval_minutes=1,  # 1 minute cleanup interval
            max_recovery_attempts=2
        )
        
    def tearDown(self):
        """Clean up test environment"""
        self.manager.shutdown()
    
    def test_initialization(self):
        """Test proper initialization of session manager"""
        self.assertIsNotNone(self.manager.storage)
        self.assertEqual(self.manager.session_timeout_minutes, 1)
        self.assertEqual(len(self.manager.active_sessions), 0)
        self.assertEqual(len(self.manager.paused_sessions), 0)
        self.assertTrue(self.manager.cleanup_enabled)
    
    def test_start_session(self):
        """Test starting a new session"""
        session_id = self.manager.start_session(
            agent_type="rif-implementer",
            issue_number=123,
            context_summary="Test session"
        )
        
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 0)
        
        # Session should be in active sessions
        self.assertIn(session_id, self.manager.active_sessions)
        
        # Check session metadata
        session = self.manager.active_sessions[session_id]
        self.assertEqual(session.agent_type, "rif-implementer")
        self.assertEqual(session.issue_number, 123)
        self.assertEqual(session.status, "active")
        self.assertTrue(session.is_active)
        
        # Storage should have been called
        self.mock_storage.start_conversation.assert_called_once()
    
    def test_end_session(self):
        """Test ending a session"""
        # Start a session first
        session_id = self.manager.start_session("rif-implementer", issue_number=456)
        
        # End the session
        success = self.manager.end_session(session_id, success=True, context_summary="Completed successfully")
        
        self.assertTrue(success)
        
        # Session should be removed from active sessions
        self.assertNotIn(session_id, self.manager.active_sessions)
        
        # Storage should have been called
        self.mock_storage.end_conversation.assert_called()
    
    def test_end_nonexistent_session(self):
        """Test ending a session that doesn't exist"""
        success = self.manager.end_session("nonexistent-session-id")
        
        self.assertFalse(success)
    
    def test_pause_and_resume_session(self):
        """Test pausing and resuming a session"""
        # Start a session
        session_id = self.manager.start_session("rif-validator")
        
        # Pause the session
        pause_success = self.manager.pause_session(session_id, reason="Testing pause")
        self.assertTrue(pause_success)
        
        # Session should be in paused sessions, not active
        self.assertNotIn(session_id, self.manager.active_sessions)
        self.assertIn(session_id, self.manager.paused_sessions)
        
        paused_session = self.manager.paused_sessions[session_id]
        self.assertEqual(paused_session.status, "paused")
        
        # Resume the session
        resume_success = self.manager.resume_session(session_id)
        self.assertTrue(resume_success)
        
        # Session should be back in active sessions
        self.assertIn(session_id, self.manager.active_sessions)
        self.assertNotIn(session_id, self.manager.paused_sessions)
        
        resumed_session = self.manager.active_sessions[session_id]
        self.assertEqual(resumed_session.status, "active")
    
    def test_pause_nonexistent_session(self):
        """Test pausing a session that doesn't exist"""
        success = self.manager.pause_session("nonexistent-session-id")
        self.assertFalse(success)
    
    def test_resume_nonexistent_session(self):
        """Test resuming a session that doesn't exist"""
        success = self.manager.resume_session("nonexistent-session-id")
        self.assertFalse(success)
    
    def test_update_session_activity(self):
        """Test updating session activity"""
        # Start a session
        session_id = self.manager.start_session("rif-analyst")
        
        initial_session = self.manager.active_sessions[session_id]
        initial_event_count = initial_session.event_count
        initial_activity = initial_session.last_activity
        
        # Small delay to ensure timestamp difference
        time.sleep(0.01)
        
        # Update activity
        success = self.manager.update_session_activity(
            session_id, 
            event_type="tool_use",
            increment_events=True,
            increment_decisions=True
        )
        
        self.assertTrue(success)
        
        updated_session = self.manager.active_sessions[session_id]
        self.assertEqual(updated_session.event_count, initial_event_count + 1)
        self.assertEqual(updated_session.decision_count, 1)
        self.assertGreater(updated_session.last_activity, initial_activity)
    
    def test_get_session_metadata(self):
        """Test retrieving session metadata"""
        # Start a session
        session_id = self.manager.start_session("rif-security", issue_number=789)
        
        # Get metadata
        metadata = self.manager.get_session_metadata(session_id)
        
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata, SessionMetadata)
        self.assertEqual(metadata.session_id, session_id)
        self.assertEqual(metadata.agent_type, "rif-security")
        self.assertEqual(metadata.issue_number, 789)
    
    def test_get_all_sessions(self):
        """Test getting all session metadata"""
        # Start multiple sessions
        active_session = self.manager.start_session("rif-implementer", issue_number=100)
        paused_session = self.manager.start_session("rif-validator", issue_number=200)
        
        # Pause one session
        self.manager.pause_session(paused_session)
        
        # Get all sessions
        all_sessions = self.manager.get_all_sessions()
        
        self.assertEqual(len(all_sessions), 2)
        self.assertIn(active_session, all_sessions)
        self.assertIn(paused_session, all_sessions)
    
    def test_generate_session_summary(self):
        """Test generating session summary"""
        # Start a session
        session_id = self.manager.start_session("rif-planner", issue_number=321)
        
        # Update some activity
        self.manager.update_session_activity(session_id, increment_events=True)
        self.manager.update_session_activity(session_id, increment_errors=True)
        
        # Generate summary
        summary = self.manager.generate_session_summary(session_id)
        
        self.assertIsNotNone(summary)
        self.assertIn(session_id, summary)
        self.assertIn("rif-planner", summary)
        self.assertIn("Issue: #321", summary)
        self.assertIn("Events: 1", summary)
        self.assertIn("Errors: 1", summary)
    
    def test_cleanup_stale_sessions(self):
        """Test cleaning up stale sessions"""
        # Start a session
        session_id = self.manager.start_session("rif-implementer")
        
        # Manually set old last_activity to simulate stale session
        session = self.manager.active_sessions[session_id]
        session.last_activity = datetime.utcnow() - timedelta(minutes=5)  # 5 minutes old
        
        # Run cleanup (force=True to ignore timeout settings)
        cleanup_stats = self.manager.cleanup_stale_sessions(force=True)
        
        self.assertGreater(cleanup_stats['active_cleaned'], 0)
        
        # Session should be removed from active sessions
        self.assertNotIn(session_id, self.manager.active_sessions)
    
    def test_get_session_statistics(self):
        """Test getting session statistics"""
        # Start some sessions
        session1 = self.manager.start_session("rif-implementer")
        session2 = self.manager.start_session("rif-validator")
        
        # End one session
        self.manager.end_session(session1, success=True)
        
        # Get statistics
        stats = self.manager.get_session_statistics()
        
        self.assertIn('sessions_started', stats)
        self.assertIn('sessions_completed', stats)
        self.assertIn('active_sessions_count', stats)
        self.assertIn('session_timeout_minutes', stats)
        
        self.assertEqual(stats['sessions_started'], 2)
        self.assertEqual(stats['sessions_completed'], 1)
        self.assertEqual(stats['active_sessions_count'], 1)
    
    def test_recovery_functionality(self):
        """Test session recovery from storage"""
        # Mock storage to return recovery data
        mock_row = [
            "recovered-session-id",
            "rif-implementer",
            123,
            datetime.utcnow() - timedelta(minutes=10),  # start_timestamp
            None,  # end_timestamp (active session)
            10,  # total_events
            2,   # error_count
            1,   # decision_count
            "Recovered session"  # context_summary
        ]
        
        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_result.description = [
            ('conversation_id',), ('agent_type',), ('issue_number',), ('start_timestamp',),
            ('end_timestamp',), ('total_events',), ('error_count',), ('decision_count',),
            ('context_summary',)
        ]
        self.mock_storage.connection.execute.return_value = mock_result
        
        # Attempt recovery
        recovered_session = self.manager.recover_session("recovered-session-id")
        
        self.assertIsNotNone(recovered_session)
        self.assertEqual(recovered_session.session_id, "recovered-session-id")
        self.assertEqual(recovered_session.agent_type, "rif-implementer")
        self.assertEqual(recovered_session.issue_number, 123)
        self.assertEqual(recovered_session.status, "recovered")
        self.assertEqual(recovered_session.event_count, 10)
        
        # Should be added to active sessions
        self.assertIn("recovered-session-id", self.manager.active_sessions)


class TestManagedConversationSession(unittest.TestCase):
    """Test the context manager for automatic session management"""
    
    def setUp(self):
        # Create mock session manager
        self.mock_manager = Mock()
        self.mock_manager.start_session.return_value = "context-session-id"
        self.mock_manager.end_session.return_value = True
        
    def test_successful_context_manager(self):
        """Test successful context manager usage"""
        with managed_conversation_session(
            "rif-implementer",
            issue_number=123,
            session_manager=self.mock_manager
        ) as session_id:
            self.assertEqual(session_id, "context-session-id")
            # Simulate some work
            pass
        
        # Should have started and ended session
        self.mock_manager.start_session.assert_called_once_with(
            agent_type="rif-implementer",
            issue_number=123,
            context_summary=None
        )
        self.mock_manager.end_session.assert_called_once_with(
            "context-session-id",
            success=True,
            context_summary=None
        )
    
    def test_context_manager_with_exception(self):
        """Test context manager behavior when exception occurs"""
        try:
            with managed_conversation_session(
                "rif-validator",
                session_manager=self.mock_manager
            ) as session_id:
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception
        
        # Should have started session
        self.mock_manager.start_session.assert_called_once()
        
        # Should have ended session with success=False
        self.mock_manager.end_session.assert_called_once()
        call_args = self.mock_manager.end_session.call_args
        self.assertFalse(call_args[1]['success'])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_create_session_manager(self):
        """Test factory function"""
        manager = create_session_manager(session_timeout_minutes=45)
        
        self.assertIsInstance(manager, ConversationSessionManager)
        self.assertEqual(manager.session_timeout_minutes, 45)
        
        # Clean up
        manager.shutdown()
    
    @patch('knowledge.conversations.session_manager._default_session_manager', None)
    def test_get_default_session_manager(self):
        """Test getting default session manager"""
        manager1 = get_default_session_manager()
        manager2 = get_default_session_manager()
        
        # Should return same instance
        self.assertIs(manager1, manager2)
        
        # Clean up
        manager1.shutdown()


class TestIntegration(unittest.TestCase):
    """Integration tests for session manager"""
    
    def test_complete_session_lifecycle(self):
        """Test complete session lifecycle from start to end"""
        # Mock storage
        mock_storage = Mock()
        mock_storage.start_conversation.return_value = "lifecycle-session-id"
        mock_storage.end_conversation.return_value = None
        mock_storage.close.return_value = None
        
        # Mock database for recovery
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        mock_storage.connection.execute.return_value = mock_result
        
        manager = ConversationSessionManager(
            storage_backend=mock_storage,
            session_timeout_minutes=60,
            cleanup_interval_minutes=30
        )
        
        try:
            # Start session
            session_id = manager.start_session(
                "rif-implementer",
                issue_number=999,
                context_summary="Integration test session"
            )
            
            self.assertIsNotNone(session_id)
            self.assertIn(session_id, manager.active_sessions)
            
            # Update activity
            manager.update_session_activity(session_id, "tool_use", increment_events=True)
            manager.update_session_activity(session_id, "decision", increment_decisions=True)
            manager.update_session_activity(session_id, "error", increment_errors=True)
            
            # Check session state
            session = manager.get_session_metadata(session_id)
            self.assertEqual(session.event_count, 1)
            self.assertEqual(session.decision_count, 1)
            self.assertEqual(session.error_count, 1)
            
            # Pause and resume
            self.assertTrue(manager.pause_session(session_id, "Integration test pause"))
            self.assertIn(session_id, manager.paused_sessions)
            
            self.assertTrue(manager.resume_session(session_id))
            self.assertIn(session_id, manager.active_sessions)
            
            # Generate summary
            summary = manager.generate_session_summary(session_id)
            self.assertIn("rif-implementer", summary)
            self.assertIn("Issue: #999", summary)
            
            # End session
            self.assertTrue(manager.end_session(session_id, success=True))
            self.assertNotIn(session_id, manager.active_sessions)
            
            # Check final statistics
            stats = manager.get_session_statistics()
            self.assertEqual(stats['sessions_started'], 1)
            self.assertEqual(stats['sessions_completed'], 1)
            self.assertEqual(stats['total_events_captured'], 1)
            
        finally:
            manager.shutdown()


if __name__ == '__main__':
    # Run tests
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversation Session Manager Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ EXCELLENT: Session manager is working correctly!")
    elif success_rate >= 80:
        print("✅ GOOD: Session manager is mostly functional")
    elif success_rate >= 70:
        print("⚠️  WARNING: Some issues detected in session manager")
    else:
        print("❌ CRITICAL: Major issues in session manager")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)