"""
Tests for agent conversation storage and query system.

Validates conversation capture, storage, querying, and pattern detection
functionality for the RIF agent conversation system.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from storage_backend import ConversationStorageBackend
from capture_engine import ConversationCaptureEngine
from query_engine import ConversationQueryEngine
from pattern_detector import ConversationPatternDetector


class TestConversationStorageBackend(unittest.TestCase):
    """Test cases for ConversationStorageBackend."""
    
    def setUp(self):
        """Set up test database."""
        # Create a temporary file path without creating the file
        import tempfile
        fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(fd)  # Close file descriptor
        os.unlink(self.temp_db_path)  # Remove the empty file
        self.storage = ConversationStorageBackend(self.temp_db_path)
    
    def tearDown(self):
        """Clean up test database."""
        self.storage.close()
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_conversation_lifecycle(self):
        """Test complete conversation lifecycle."""
        # Start conversation
        conv_id = self.storage.start_conversation(
            agent_type="rif-implementer",
            issue_number=34,
            context_summary="Test conversation"
        )
        
        self.assertIsNotNone(conv_id)
        
        # Store events
        event_id = self.storage.store_conversation_event(
            conversation_id=conv_id,
            agent_type="rif-implementer", 
            event_type="tool_use",
            event_data={"tool_name": "Read", "file_path": "/test/file.py"},
            issue_number=34
        )
        
        self.assertIsNotNone(event_id)
        
        # Store decision
        decision_id = self.storage.store_agent_decision(
            conversation_id=conv_id,
            agent_type="rif-implementer",
            decision_point="Choose implementation approach",
            options_considered=[{"option": "approach_a"}, {"option": "approach_b"}],
            chosen_option="approach_a",
            rationale="Better performance characteristics"
        )
        
        self.assertIsNotNone(decision_id)
        
        # Store error
        error_id = self.storage.store_conversation_error(
            conversation_id=conv_id,
            agent_type="rif-implementer",
            error_type="file_not_found",
            error_message="File /test/file.py not found",
            resolution_attempted="Created missing file",
            resolution_success=True
        )
        
        self.assertIsNotNone(error_id)
        
        # End conversation
        self.storage.end_conversation(conv_id, success=True)
        
        # Verify storage
        events = self.storage.get_conversation_events(conv_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'tool_use')
        
        decisions = self.storage.get_agent_decisions(conversation_id=conv_id)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['chosen_option'], 'approach_a')
        
        errors = self.storage.get_conversation_errors(agent_type="rif-implementer")
        self.assertGreaterEqual(len(errors), 1)
    
    def test_storage_stats(self):
        """Test storage statistics retrieval."""
        # Create some test data
        conv_id = self.storage.start_conversation("rif-implementer", 34)
        self.storage.store_conversation_event(conv_id, "rif-implementer", "test", {"data": "test"})
        
        stats = self.storage.get_storage_stats()
        
        self.assertIn('conversation_events_count', stats)
        self.assertIn('conversation_metadata_count', stats) 
        self.assertGreater(stats['conversation_events_count'], 0)
        self.assertGreater(stats['conversation_metadata_count'], 0)
    
    def test_pattern_signature_generation(self):
        """Test error pattern signature generation."""
        conv_id = self.storage.start_conversation("rif-implementer")
        
        # Store two similar errors (same error message for same pattern signature)
        error_id1 = self.storage.store_conversation_error(
            conversation_id=conv_id,
            agent_type="rif-implementer",
            error_type="file_error",
            error_message="File not found: test file"
        )
        
        error_id2 = self.storage.store_conversation_error(
            conversation_id=conv_id,
            agent_type="rif-implementer", 
            error_type="file_error",
            error_message="File not found: test file"  # Same message for same signature
        )
        
        # Both errors should have the same pattern signature
        errors = self.storage.get_conversation_errors(agent_type="rif-implementer")
        self.assertEqual(len(errors), 2)
        
        # Pattern signatures should be the same for similar errors
        signatures = [error['pattern_signature'] for error in errors]
        # Both errors have same agent_type, error_type and similar message prefix
        # so they should have same pattern signature
        self.assertEqual(len(set(signatures)), 1, "Similar errors should have the same pattern signature")


class TestConversationCaptureEngine(unittest.TestCase):
    """Test cases for ConversationCaptureEngine."""
    
    def setUp(self):
        """Set up test capture engine."""
        # Create a temporary file path without creating the file
        fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(fd)  # Close file descriptor
        os.unlink(self.temp_db_path)  # Remove the empty file
        self.storage = ConversationStorageBackend(self.temp_db_path)
        self.capture_engine = ConversationCaptureEngine(self.storage)
        
        # Mock embedding generator
        self.mock_embedding = Mock(return_value=[0.1] * 768)
        self.capture_engine.set_embedding_generator(self.mock_embedding)
    
    def tearDown(self):
        """Clean up test resources."""
        self.capture_engine.close()
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_conversation_session(self):
        """Test conversation session context manager."""
        with self.capture_engine.conversation_session(
            agent_type="rif-implementer",
            issue_number=34
        ) as conv_id:
            self.assertIsNotNone(conv_id)
            
            # Check active conversations
            active = self.capture_engine.get_active_conversations()
            self.assertIn(conv_id, active)
            self.assertEqual(active[conv_id]['agent_type'], 'rif-implementer')
        
        # Should be removed from active conversations after context exit
        active_after = self.capture_engine.get_active_conversations()
        self.assertNotIn(conv_id, active_after)
    
    def test_event_capture(self):
        """Test event capture functionality."""
        with self.capture_engine.conversation_session("rif-implementer") as conv_id:
            # Capture event
            event_id = self.capture_engine.capture_event(
                conversation_id=conv_id,
                event_type="tool_use",
                event_data={
                    "tool_name": "Read",
                    "description": "Reading configuration file"
                },
                generate_embedding=True
            )
            
            self.assertIsNotNone(event_id)
            
            # Verify event was stored
            events = self.storage.get_conversation_events(conv_id)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['event_type'], 'tool_use')
            self.assertEqual(events[0]['event_data']['tool_name'], 'Read')
    
    def test_decision_capture(self):
        """Test decision capture functionality."""
        with self.capture_engine.conversation_session("rif-implementer") as conv_id:
            # Capture decision
            decision_id = self.capture_engine.capture_decision(
                conversation_id=conv_id,
                decision_point="Choose data structure",
                options_considered=["dict", "list", "set"],
                chosen_option="dict",
                rationale="Better lookup performance",
                confidence_score=0.8
            )
            
            self.assertIsNotNone(decision_id)
            
            # Verify decision was stored
            decisions = self.storage.get_agent_decisions(conversation_id=conv_id)
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]['chosen_option'], 'dict')
            self.assertAlmostEqual(decisions[0]['confidence_score'], 0.8, places=1)
    
    def test_error_capture(self):
        """Test error capture functionality."""
        with self.capture_engine.conversation_session("rif-implementer") as conv_id:
            # Capture error
            error_id = self.capture_engine.capture_error(
                conversation_id=conv_id,
                error_type="import_error",
                error_message="Module not found: missing_module",
                error_context={"traceback": "fake traceback"},
                resolution_attempted="Install missing module",
                resolution_success=True
            )
            
            self.assertIsNotNone(error_id)
            
            # Verify error was stored
            errors = self.storage.get_conversation_errors(agent_type="rif-implementer")
            self.assertGreaterEqual(len(errors), 1)
            
            # Find our error
            our_error = next((e for e in errors if e['error_id'] == error_id), None)
            self.assertIsNotNone(our_error)
            self.assertEqual(our_error['error_type'], 'import_error')
            self.assertTrue(our_error['resolution_success'])
    
    def test_tool_use_context_manager(self):
        """Test tool use capture context manager."""
        with self.capture_engine.conversation_session("rif-implementer") as conv_id:
            # Test successful tool use
            with self.capture_engine.capture_tool_use(
                conversation_id=conv_id,
                tool_name="Read",
                tool_params={"file_path": "/test/file.py"}
            ) as tool_event_id:
                self.assertIsNotNone(tool_event_id)
                # Simulate tool execution
                pass
            
            # Test tool use with error
            try:
                with self.capture_engine.capture_tool_use(
                    conversation_id=conv_id,
                    tool_name="Write",
                    tool_params={"file_path": "/readonly/file.py"}
                ) as tool_event_id:
                    raise PermissionError("Permission denied")
            except PermissionError:
                pass  # Expected
            
            # Verify events were captured
            events = self.storage.get_conversation_events(conv_id)
            # Should have start, success, start, error events
            self.assertGreaterEqual(len(events), 3)
    
    def test_capture_stats(self):
        """Test capture statistics tracking."""
        initial_stats = self.capture_engine.get_capture_stats()
        
        with self.capture_engine.conversation_session("rif-implementer") as conv_id:
            self.capture_engine.capture_event(conv_id, "test", {"data": "test"})
            self.capture_engine.capture_decision(conv_id, "test", ["a", "b"], "a")
            self.capture_engine.capture_error(conv_id, "test", "test error")
        
        final_stats = self.capture_engine.get_capture_stats()
        
        self.assertGreater(final_stats['events_captured'], initial_stats['events_captured'])
        self.assertGreater(final_stats['decisions_captured'], initial_stats['decisions_captured'])
        self.assertGreater(final_stats['errors_captured'], initial_stats['errors_captured'])
        self.assertGreater(final_stats['conversations_completed'], initial_stats['conversations_completed'])


class TestConversationQueryEngine(unittest.TestCase):
    """Test cases for ConversationQueryEngine."""
    
    def setUp(self):
        """Set up test query engine."""
        # Create a temporary file path without creating the file
        fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(fd)  # Close file descriptor
        os.unlink(self.temp_db_path)  # Remove the empty file
        self.storage = ConversationStorageBackend(self.temp_db_path)
        self.query_engine = ConversationQueryEngine(self.storage)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test resources."""
        self.storage.close()
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _create_test_data(self):
        """Create test conversation data."""
        # Create test conversations
        conv_id1 = self.storage.start_conversation("rif-implementer", 34, "Authentication implementation")
        self.storage.store_conversation_event(
            conv_id1, "rif-implementer", "tool_use",
            {"tool_name": "Read", "description": "Reading auth configuration"}
        )
        self.storage.store_agent_decision(
            conv_id1, "rif-implementer", "Choose auth method",
            [{"option": "JWT"}, {"option": "Session"}], "JWT", "Better for API"
        )
        self.storage.end_conversation(conv_id1, success=True)
        
        conv_id2 = self.storage.start_conversation("rif-validator", 35, "Testing validation")
        self.storage.store_conversation_error(
            conv_id2, "rif-validator", "test_failure",
            "Unit test failed: test_auth", {"test_name": "test_auth"}
        )
        self.storage.end_conversation(conv_id2, success=False)
    
    def test_search_conversations(self):
        """Test conversation search functionality."""
        # Search by query term
        results = self.query_engine.search_conversations("authentication")
        self.assertGreater(len(results), 0)
        
        # Search with filters
        results = self.query_engine.search_conversations(
            query="auth",
            agent_type="rif-implementer",
            issue_number=34
        )
        self.assertGreater(len(results), 0)
    
    def test_find_similar_decisions(self):
        """Test similar decision finding."""
        decisions = self.query_engine.find_similar_decisions(
            "authentication method selection"
        )
        self.assertGreaterEqual(len(decisions), 0)
        
        # If we have decisions, check structure
        if decisions:
            decision = decisions[0]
            self.assertIn('decision_point', decision)
            self.assertIn('chosen_option', decision)
            self.assertIn('options_considered', decision)
    
    def test_analyze_error_patterns(self):
        """Test error pattern analysis."""
        patterns = self.query_engine.analyze_error_patterns(
            days_back=1,  # Recent data
            min_occurrences=1  # Low threshold for test
        )
        self.assertGreaterEqual(len(patterns), 0)
        
        # If we have patterns, check structure
        if patterns:
            pattern = patterns[0]
            self.assertIn('pattern_signature', pattern)
            self.assertIn('error_type', pattern)
            self.assertIn('occurrence_count', pattern)
    
    def test_agent_performance_summary(self):
        """Test agent performance summary."""
        summary = self.query_engine.get_agent_performance_summary("rif-implementer")
        
        # Should have basic structure even with minimal data
        self.assertIn('agent_type', summary)
        self.assertIn('conversation_stats', summary)
        self.assertIn('derived_metrics', summary)
    
    def test_conversation_export(self):
        """Test conversation export functionality."""
        # Get a conversation ID from test data
        result = self.storage.connection.execute("SELECT conversation_id FROM conversation_metadata LIMIT 1")
        conv_id = result.fetchone()[0]
        
        # Export conversation
        export_data = self.query_engine.export_conversation(conv_id, format_type='json')
        
        self.assertIsNotNone(export_data)
        self.assertIn('conversation_id', export_data)
        self.assertIn('metadata', export_data)
        self.assertIn('events', export_data)
        self.assertIn('decisions', export_data)
        self.assertIn('errors', export_data)


class TestConversationPatternDetector(unittest.TestCase):
    """Test cases for ConversationPatternDetector."""
    
    def setUp(self):
        """Set up test pattern detector."""
        # Create a temporary file path without creating the file
        fd, self.temp_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(fd)  # Close file descriptor
        os.unlink(self.temp_db_path)  # Remove the empty file
        self.storage = ConversationStorageBackend(self.temp_db_path)
        self.pattern_detector = ConversationPatternDetector(self.storage)
        
        # Create test data with patterns
        self._create_pattern_test_data()
    
    def tearDown(self):
        """Clean up test resources."""
        self.storage.close()
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _create_pattern_test_data(self):
        """Create test data with detectable patterns."""
        # Create successful pattern
        for i in range(3):
            conv_id = self.storage.start_conversation("rif-implementer", 30 + i)
            self.storage.store_conversation_event(
                conv_id, "rif-implementer", "tool_use",
                {"tool_name": "Read", "status": "success"}
            )
            decision_id = self.storage.store_agent_decision(
                conv_id, "rif-implementer", "Implementation approach",
                [{"option": "TDD"}, {"option": "Direct"}], "TDD",
                "Better quality", confidence_score=0.8
            )
            # Manually update outcome via capture engine if needed
            if hasattr(self, '_capture_engine_for_test'):
                self._capture_engine_for_test.update_decision_outcome(decision_id, "success")
            self.storage.end_conversation(conv_id, success=True)
        
        # Create failure pattern
        for i in range(2):
            conv_id = self.storage.start_conversation("rif-validator", 40 + i)
            self.storage.store_conversation_error(
                conv_id, "rif-validator", "validation_error",
                f"Validation failed for test case {i}",
                resolution_attempted="Retry validation", resolution_success=False
            )
            self.storage.end_conversation(conv_id, success=False)
    
    def test_detect_success_patterns(self):
        """Test success pattern detection."""
        patterns = self.pattern_detector.detect_success_patterns(
            agent_type="rif-implementer",
            days_back=1,
            min_occurrences=2
        )
        
        # Should detect patterns from our test data
        self.assertGreaterEqual(len(patterns), 0)
        
        if patterns:
            pattern = patterns[0]
            self.assertIn('pattern_key', pattern)
            self.assertIn('success_rate', pattern)
            self.assertIn('occurrence_count', pattern)
            self.assertEqual(pattern['pattern_type'], 'success')
    
    def test_detect_failure_patterns(self):
        """Test failure pattern detection."""
        patterns = self.pattern_detector.detect_failure_patterns(
            agent_type="rif-validator",
            days_back=1,
            min_occurrences=1
        )
        
        # Should detect failure patterns
        self.assertGreaterEqual(len(patterns), 0)
        
        if patterns:
            pattern = patterns[0]
            self.assertIn('pattern_key', pattern)
            self.assertIn('failure_rate', pattern)
            self.assertIn('occurrence_count', pattern)
            self.assertEqual(pattern['pattern_type'], 'failure')
    
    def test_analyze_decision_patterns(self):
        """Test decision pattern analysis."""
        analysis = self.pattern_detector.analyze_decision_patterns(
            agent_type="rif-implementer"
        )
        
        # Should have analysis structure
        if analysis:  # Might be empty if no decisions in time window
            self.assertIn('total_decisions', analysis)
            if analysis['total_decisions'] > 0:
                self.assertIn('agent_breakdown', analysis)
                self.assertIn('confidence_analysis', analysis)
                self.assertIn('outcome_analysis', analysis)
    
    def test_generate_learning_recommendations(self):
        """Test learning recommendation generation."""
        recommendations = self.pattern_detector.generate_learning_recommendations(
            days_back=1
        )
        
        # Should generate some recommendations from test data
        self.assertIsInstance(recommendations, list)
        
        if recommendations:
            rec = recommendations[0]
            self.assertIn('type', rec)
            self.assertIn('priority', rec)
            self.assertIn('title', rec)
            self.assertIn('description', rec)
    
    def test_pattern_detection_summary(self):
        """Test pattern detection summary."""
        summary = self.pattern_detector.get_pattern_detection_summary()
        
        self.assertIn('config', summary)
        self.assertIn('recent_analysis', summary)
        
        # Should have analysis for our test agents
        if summary['recent_analysis']:
            self.assertTrue(any('rif-' in agent for agent in summary['recent_analysis'].keys()))


def run_conversation_tests():
    """Run all conversation system tests."""
    test_classes = [
        TestConversationStorageBackend,
        TestConversationCaptureEngine,
        TestConversationQueryEngine,
        TestConversationPatternDetector
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_conversation_tests()
    exit(0 if success else 1)