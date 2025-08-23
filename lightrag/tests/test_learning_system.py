"""
Comprehensive tests for RIF Learning System (Phase 4)
Tests feedback loops, analytics, and knowledge refinement.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.feedback_loop import FeedbackLoop, FeedbackEvent, PatternEffectiveness, record_agent_feedback
from learning.analytics import AnalyticsDashboard, generate_system_report
from learning.knowledge_refiner import KnowledgeRefiner, run_knowledge_refinement


class TestFeedbackLoop(unittest.TestCase):
    """Test feedback loop functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock LightRAG instance
        self.mock_rag = Mock()
        self.mock_rag.insert_document.return_value = "doc_123"
        self.mock_rag.search_documents.return_value = []
        
        # Create feedback loop with mocked dependencies
        with patch('learning.feedback_loop.get_lightrag_instance', return_value=self.mock_rag):
            self.feedback_loop = FeedbackLoop(feedback_window_hours=1)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self.feedback_loop, 'shutdown'):
            self.feedback_loop.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_feedback_event_creation(self):
        """Test creation of feedback events."""
        event = FeedbackEvent(
            timestamp="2025-01-01T12:00:00",
            agent_name="test_agent",
            action="test_action",
            success=True,
            pattern_id="pattern_123"
        )
        
        self.assertEqual(event.agent_name, "test_agent")
        self.assertEqual(event.action, "test_action")
        self.assertTrue(event.success)
        self.assertEqual(event.pattern_id, "pattern_123")
    
    def test_pattern_effectiveness_tracking(self):
        """Test pattern effectiveness calculation."""
        pattern = PatternEffectiveness(
            pattern_id="test_pattern",
            usage_count=10,
            success_count=8,
            failure_count=2,
            avg_performance=0.85,
            last_used="2025-01-01T12:00:00",
            context_tags=["implementation", "testing"]
        )
        
        self.assertEqual(pattern.success_rate, 0.8)
        self.assertEqual(pattern.usage_count, 10)
    
    def test_record_event(self):
        """Test recording feedback events."""
        self.feedback_loop.record_event(
            agent_name="analyst",
            action="analyze_requirements",
            success=True,
            pattern_id="analysis_pattern_1",
            performance_metrics={"score": 0.9},
            context={"issue_id": "123"}
        )
        
        # Give time for background processing
        time.sleep(0.1)
        
        # Check event was queued
        self.assertGreater(self.feedback_loop.event_queue.qsize(), 0)
    
    def test_system_health_metrics(self):
        """Test system health reporting."""
        # Record some events
        for i in range(5):
            self.feedback_loop.record_event(
                agent_name="test_agent",
                action="test_action",
                success=i % 2 == 0  # 60% success rate
            )
        
        time.sleep(0.2)  # Allow processing
        
        health = self.feedback_loop.get_system_health()
        
        self.assertIn("total_patterns", health)
        self.assertIn("recent_events", health)
        self.assertIn("active_agents", health)
        self.assertIn("avg_success_rate", health)
    
    def test_agent_performance_tracking(self):
        """Test agent-specific performance tracking."""
        # Record events for specific agent
        for i in range(3):
            self.feedback_loop.record_event(
                agent_name="implementer",
                action="implement_feature",
                success=True
            )
        
        time.sleep(0.1)
        
        performance = self.feedback_loop.get_agent_performance("implementer", hours=1)
        
        self.assertIn("events", performance)
        self.assertIn("success_rate", performance)
        self.assertIn("actions", performance)


class TestAnalyticsDashboard(unittest.TestCase):
    """Test analytics dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.mock_rag = Mock()
        self.mock_rag.search_documents.return_value = []
        
        self.mock_feedback_loop = Mock()
        self.mock_feedback_loop.get_system_health.return_value = {
            "total_patterns": 25,
            "recent_events": 15,
            "active_agents": 3,
            "avg_success_rate": 0.85,
            "queue_size": 5,
            "system_uptime": True
        }
        self.mock_feedback_loop.get_agent_performance.return_value = {
            "events": 10,
            "success_rate": 0.8,
            "actions": ["analyze", "implement"],
            "latest_activity": "2025-01-01T12:00:00"
        }
        
        with patch('learning.analytics.get_lightrag_instance', return_value=self.mock_rag), \
             patch('learning.analytics.get_feedback_loop', return_value=self.mock_feedback_loop):
            self.dashboard = AnalyticsDashboard()
    
    def test_system_report_generation(self):
        """Test generation of system reports."""
        report = self.dashboard.generate_system_report(hours=24)
        
        # Check report structure
        self.assertIn("report_generated", report)
        self.assertIn("system_health", report)
        self.assertIn("agent_performance", report)
        self.assertIn("pattern_effectiveness", report)
        self.assertIn("knowledge_growth", report)
        self.assertIn("recommendations", report)
        
        # Check system health analysis
        health = report["system_health"]
        self.assertIn("status", health)
        self.assertIn("metrics", health)
        self.assertIn("issues", health)
    
    def test_agent_performance_analysis(self):
        """Test agent performance analysis."""
        report = self.dashboard.generate_system_report(hours=24)
        
        agent_performance = report["agent_performance"]
        
        # Should analyze all core RIF agents
        expected_agents = ["analyst", "planner", "architect", "implementer", "validator", "learner"]
        for agent in expected_agents:
            self.assertIn(agent, agent_performance)
            
            agent_data = agent_performance[agent]
            self.assertIn("performance", agent_data)
            self.assertIn("grade", agent_data)
            self.assertIn("insights", agent_data)
    
    def test_markdown_export(self):
        """Test markdown report export."""
        report = self.dashboard.generate_system_report(hours=24)
        markdown = self.dashboard.export_report(report, "markdown")
        
        # Check markdown structure
        self.assertIn("# RIF Analytics Report", markdown)
        self.assertIn("## System Health", markdown)
        self.assertIn("## Agent Performance", markdown)
        self.assertIn("## Recommendations", markdown)
    
    def test_json_export(self):
        """Test JSON report export."""
        report = self.dashboard.generate_system_report(hours=24)
        json_str = self.dashboard.export_report(report, "json")
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["report_generated"], report["report_generated"])
    
    def test_recommendations_generation(self):
        """Test recommendation generation."""
        # Test with low performance scenario
        self.mock_feedback_loop.get_system_health.return_value = {
            "total_patterns": 25,
            "recent_events": 2,  # Low activity
            "active_agents": 1,
            "avg_success_rate": 0.5,  # Low success rate
            "queue_size": 5,
            "system_uptime": True
        }
        
        report = self.dashboard.generate_system_report(hours=24)
        recommendations = report["recommendations"]
        
        self.assertGreater(len(recommendations), 0)
        
        # Should have high priority recommendations for low success rate
        high_priority = [r for r in recommendations if r["priority"] == "high"]
        self.assertGreater(len(high_priority), 0)


class TestKnowledgeRefiner(unittest.TestCase):
    """Test knowledge refinement functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.mock_rag = Mock()
        self.mock_rag.search_documents.return_value = self._create_mock_documents()
        self.mock_rag.insert_document.return_value = "doc_123"
        
        self.mock_feedback_loop = Mock()
        
        with patch('learning.knowledge_refiner.get_lightrag_instance', return_value=self.mock_rag), \
             patch('learning.knowledge_refiner.get_feedback_loop', return_value=self.mock_feedback_loop):
            self.refiner = KnowledgeRefiner()
    
    def _create_mock_documents(self):
        """Create mock documents for testing."""
        docs = []
        
        # Create documents with various characteristics
        for i in range(5):
            doc = Mock()
            doc.content = f"Test document content {i} with some details and information."
            doc.metadata = {
                "timestamp": (datetime.now() - timedelta(days=i*10)).isoformat(),
                "type": "test_document",
                "agent": "test_agent"
            }
            docs.append(doc)
        
        return docs
    
    def test_full_refinement_process(self):
        """Test complete refinement process."""
        results = self.refiner.run_full_refinement()
        
        # Check results structure
        self.assertIn("start_time", results)
        self.assertIn("end_time", results)
        self.assertIn("duration_seconds", results)
        self.assertIn("operations", results)
        self.assertIn("errors", results)
        self.assertIn("recommendations", results)
        
        # Check operations were attempted
        operations = results["operations"]
        self.assertIn("duplicates_removed", operations)
        self.assertIn("outdated_archived", operations)
        self.assertIn("low_quality_improved", operations)
        self.assertIn("patterns_consolidated", operations)
        self.assertIn("metadata_enhanced", operations)
    
    def test_content_quality_assessment(self):
        """Test content quality assessment."""
        # Test high quality content
        high_quality = "This is a comprehensive implementation guide.\n\nIt includes detailed steps and examples."
        high_score = self.refiner._assess_content_quality(high_quality)
        
        # Test low quality content
        low_quality = "short"
        low_score = self.refiner._assess_content_quality(low_quality)
        
        self.assertGreater(high_score, low_score)
        self.assertGreater(high_score, 0.3)
    
    def test_duplicate_detection(self):
        """Test duplicate content detection."""
        # Create similar documents
        docs = []
        for i in range(3):
            doc = Mock()
            doc.content = "This is similar content with minor variations."
            docs.append(doc)
        
        # Add different document
        different_doc = Mock()
        different_doc.content = "This is completely different content about another topic."
        docs.append(different_doc)
        
        groups = self.refiner._find_duplicate_groups(docs)
        
        # Should find one group of 3 similar documents
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
    
    def test_content_similarity_calculation(self):
        """Test content similarity calculation."""
        content1 = "This is a test document with some content"
        content2 = "This is a test document with similar content"
        content3 = "Completely different topic and subject matter"
        
        # Similar content should have high similarity
        similarity_high = self.refiner._calculate_content_similarity(content1, content2)
        
        # Different content should have low similarity
        similarity_low = self.refiner._calculate_content_similarity(content1, content3)
        
        self.assertGreater(similarity_high, similarity_low)
        self.assertGreater(similarity_high, 0.5)
    
    def test_metadata_enhancement(self):
        """Test metadata enhancement."""
        doc = Mock()
        doc.content = "This is implementation content with patterns and code examples."
        doc.metadata = {"type": "unknown"}
        
        enhanced = self.refiner._generate_enhanced_metadata(doc)
        
        # Should add new metadata fields
        self.assertIn("word_count", enhanced)
        self.assertIn("quality_score", enhanced)
        self.assertIn("tags", enhanced)
        self.assertGreater(len(enhanced), len(doc.metadata))
    
    def test_refinement_status(self):
        """Test refinement status tracking."""
        # Run refinement first
        self.refiner.run_full_refinement()
        
        status = self.refiner.get_refinement_status()
        
        self.assertIn("last_refinement", status)
        self.assertIn("total_refinements", status)
        self.assertIn("next_recommended", status)
        
        self.assertIsNotNone(status["last_refinement"])
        self.assertEqual(status["total_refinements"], 1)


class TestIntegration(unittest.TestCase):
    """Test integration between learning system components."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock LightRAG
        self.mock_rag = Mock()
        self.mock_rag.insert_document.return_value = "doc_123"
        self.mock_rag.search_documents.return_value = []
    
    def test_feedback_to_analytics_flow(self):
        """Test data flow from feedback loop to analytics."""
        with patch('learning.feedback_loop.get_lightrag_instance', return_value=self.mock_rag), \
             patch('learning.analytics.get_lightrag_instance', return_value=self.mock_rag):
            
            # Create feedback loop and dashboard
            feedback_loop = FeedbackLoop(feedback_window_hours=1)
            dashboard = AnalyticsDashboard()
            
            # Record some events
            feedback_loop.record_event("analyst", "analyze", True)
            feedback_loop.record_event("implementer", "implement", False)
            
            time.sleep(0.1)  # Allow processing
            
            # Generate report
            with patch('learning.analytics.get_feedback_loop', return_value=feedback_loop):
                report = dashboard.generate_system_report(hours=1)
            
            # Should include system health data
            self.assertIn("system_health", report)
            
            feedback_loop.shutdown()
    
    def test_convenience_functions(self):
        """Test convenience functions work correctly."""
        with patch('learning.feedback_loop.get_lightrag_instance', return_value=self.mock_rag):
            # Test recording agent feedback
            record_agent_feedback("test_agent", "test_action", True, pattern_id="test_pattern")
            
            # Test system report generation
            with patch('learning.analytics.get_lightrag_instance', return_value=self.mock_rag):
                report = generate_system_report(hours=1)
                self.assertIn("report_generated", report)
            
            # Test knowledge refinement
            with patch('learning.knowledge_refiner.get_lightrag_instance', return_value=self.mock_rag):
                results = run_knowledge_refinement()
                self.assertIn("operations", results)


if __name__ == "__main__":
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)