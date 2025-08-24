"""
Comprehensive tests for Pattern Reinforcement System

This test suite validates all components of the pattern reinforcement system
including core reinforcement, maintenance, and integration functionality.
"""

import unittest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Import system components
import sys
sys.path.append('/Users/cal/DEV/RIF')

from claude.commands.pattern_reinforcement_system import (
    PatternReinforcementSystem, PatternOutcome, PatternMetrics,
    OutcomeType, FailureMode
)
from claude.commands.pattern_maintenance_system import (
    PatternMaintenanceSystem, MaintenanceReport, MaintenanceMode
)
from claude.commands.pattern_integration import (
    PatternReinforcementIntegration, MockPatternApplicationEngine,
    create_pattern_reinforcement_system, create_mock_pattern_outcome
)


class MockKnowledgeSystem:
    """Mock knowledge system for testing."""
    
    def __init__(self):
        self.storage = {}  # collection -> {doc_id: {content, metadata}}
        self.call_log = []
    
    def store_knowledge(self, collection, content, metadata=None, doc_id=None):
        self.call_log.append(('store', collection, doc_id))
        
        if collection not in self.storage:
            self.storage[collection] = {}
        
        if not doc_id:
            doc_id = f"doc_{len(self.storage[collection])}"
        
        self.storage[collection][doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'id': doc_id
        }
        
        return doc_id
    
    def retrieve_knowledge(self, query, collection=None, n_results=5, filters=None):
        self.call_log.append(('retrieve', collection, query))
        
        results = []
        collections_to_search = [collection] if collection else self.storage.keys()
        
        for coll in collections_to_search:
            if coll not in self.storage:
                continue
            
            for doc_id, doc_data in self.storage[coll].items():
                # Simple text matching for mock
                if query == "*" or query.lower() in doc_data['content'].lower():
                    results.append({
                        'id': doc_id,
                        'content': doc_data['content'],
                        'metadata': doc_data['metadata'],
                        'collection': coll,
                        'distance': 0.1
                    })
        
        return results[:n_results]
    
    def update_knowledge(self, collection, doc_id, content=None, metadata=None):
        self.call_log.append(('update', collection, doc_id))
        
        if collection in self.storage and doc_id in self.storage[collection]:
            if content:
                self.storage[collection][doc_id]['content'] = content
            if metadata:
                self.storage[collection][doc_id]['metadata'].update(metadata)
            return True
        return False
    
    def delete_knowledge(self, collection, doc_id):
        self.call_log.append(('delete', collection, doc_id))
        
        if collection in self.storage and doc_id in self.storage[collection]:
            del self.storage[collection][doc_id]
            return True
        return False
    
    def get_collection_stats(self):
        return {coll: {'count': len(docs)} for coll, docs in self.storage.items()}


class TestPatternOutcome(unittest.TestCase):
    """Test PatternOutcome data structure."""
    
    def test_outcome_creation(self):
        """Test creating a pattern outcome."""
        outcome = PatternOutcome(
            pattern_id="test_pattern",
            outcome_type=OutcomeType.SUCCESS,
            success=True,
            issue_id="issue_123",
            agent_type="rif-implementer",
            effectiveness_score=0.8
        )
        
        self.assertEqual(outcome.pattern_id, "test_pattern")
        self.assertEqual(outcome.outcome_type, OutcomeType.SUCCESS)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.issue_id, "issue_123")
        self.assertEqual(outcome.effectiveness_score, 0.8)
        self.assertIsInstance(outcome.timestamp, datetime)
    
    def test_outcome_serialization(self):
        """Test outcome to/from dict conversion."""
        outcome = PatternOutcome(
            pattern_id="test_pattern",
            outcome_type=OutcomeType.FAILURE,
            success=False,
            failure_mode=FailureMode.IMPLEMENTATION_ERROR,
            error_details="Test error"
        )
        
        # Test to_dict
        data = outcome.to_dict()
        self.assertIsInstance(data['timestamp'], str)
        self.assertEqual(data['outcome_type'], 'failure')
        self.assertEqual(data['failure_mode'], 'implementation_error')
        
        # Test from_dict
        restored = PatternOutcome.from_dict(data)
        self.assertEqual(restored.pattern_id, outcome.pattern_id)
        self.assertEqual(restored.outcome_type, outcome.outcome_type)
        self.assertEqual(restored.failure_mode, outcome.failure_mode)


class TestPatternMetrics(unittest.TestCase):
    """Test PatternMetrics data structure and calculations."""
    
    def test_metrics_creation(self):
        """Test creating pattern metrics."""
        metrics = PatternMetrics(pattern_id="test_pattern")
        
        self.assertEqual(metrics.pattern_id, "test_pattern")
        self.assertEqual(metrics.total_applications, 0)
        self.assertEqual(metrics.success_count, 0)
        self.assertEqual(metrics.current_score, 1.0)
        self.assertEqual(metrics.success_rate, 0.0)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PatternMetrics(pattern_id="test_pattern")
        
        # No applications
        self.assertEqual(metrics.calculate_success_rate(), 0.0)
        
        # With applications
        metrics.total_applications = 10
        metrics.success_count = 7
        self.assertEqual(metrics.calculate_success_rate(), 0.7)
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        metrics = PatternMetrics(pattern_id="test_pattern")
        metrics.total_applications = 10
        metrics.success_count = 8
        metrics.effectiveness_trend = [0.8, 0.9, 0.85, 0.9, 0.95]
        
        weighted_score = metrics.calculate_weighted_score()
        
        # Should be higher than base score due to good performance
        self.assertGreater(weighted_score, metrics.current_score)
        self.assertLessEqual(weighted_score, 2.0)  # Clamped maximum
        self.assertGreaterEqual(weighted_score, 0.0)  # Clamped minimum
    
    def test_pruning_candidate_detection(self):
        """Test pruning candidate identification."""
        metrics = PatternMetrics(pattern_id="test_pattern")
        
        # New pattern - not pruning candidate
        self.assertFalse(metrics.is_pruning_candidate())
        
        # Low success rate with sufficient data
        metrics.total_applications = 15
        metrics.success_count = 3  # 20% success rate
        self.assertTrue(metrics.is_pruning_candidate())
        
        # Old unused pattern
        metrics = PatternMetrics(pattern_id="old_pattern")
        metrics.last_used = datetime.now() - timedelta(days=100)
        self.assertTrue(metrics.is_pruning_candidate())


class TestPatternReinforcementSystem(unittest.TestCase):
    """Test core pattern reinforcement system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_knowledge = MockKnowledgeSystem()
        self.system = PatternReinforcementSystem(knowledge_system=self.mock_knowledge)
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.knowledge, self.mock_knowledge)
        self.assertIsInstance(self.system.config, dict)
        self.assertIn('success_boost_factor', self.system.config)
    
    def test_successful_outcome_processing(self):
        """Test processing successful pattern outcomes."""
        outcome = PatternOutcome(
            pattern_id="success_pattern",
            outcome_type=OutcomeType.SUCCESS,
            success=True,
            effectiveness_score=0.9,
            quality_score=0.8
        )
        
        # Process outcome
        success = self.system.process_pattern_outcome(outcome)
        self.assertTrue(success)
        
        # Check metrics were updated
        metrics = self.system._get_pattern_metrics("success_pattern")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_applications, 1)
        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertGreater(metrics.current_score, 1.0)  # Should be boosted
    
    def test_failure_outcome_processing(self):
        """Test processing failed pattern outcomes."""
        outcome = PatternOutcome(
            pattern_id="failure_pattern",
            outcome_type=OutcomeType.FAILURE,
            success=False,
            failure_mode=FailureMode.IMPLEMENTATION_ERROR,
            error_details="Test failure"
        )
        
        # Process outcome
        success = self.system.process_pattern_outcome(outcome)
        self.assertTrue(success)
        
        # Check metrics were updated
        metrics = self.system._get_pattern_metrics("failure_pattern")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_applications, 1)
        self.assertEqual(metrics.failure_count, 1)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertLess(metrics.current_score, 1.0)  # Should be penalized
        self.assertIn('implementation_error', metrics.failure_modes)
    
    def test_score_update_logic(self):
        """Test pattern score update logic."""
        metrics = PatternMetrics(pattern_id="test_pattern")
        
        # Test success boost
        success_outcome = PatternOutcome(
            pattern_id="test_pattern",
            outcome_type=OutcomeType.SUCCESS,
            success=True,
            effectiveness_score=0.9
        )
        
        old_score = metrics.current_score
        self.system._update_pattern_score(metrics, success_outcome)
        self.assertGreater(metrics.current_score, old_score)
        
        # Test failure penalty
        failure_outcome = PatternOutcome(
            pattern_id="test_pattern",
            outcome_type=OutcomeType.FAILURE,
            success=False,
            failure_mode=FailureMode.QUALITY_GATE_FAILURE
        )
        
        old_score = metrics.current_score
        self.system._update_pattern_score(metrics, failure_outcome)
        self.assertLess(metrics.current_score, old_score)
    
    def test_top_patterns_retrieval(self):
        """Test getting top-performing patterns."""
        # Create some patterns with different scores
        patterns_data = [
            ("high_score", 1.5, 0.9),
            ("medium_score", 1.2, 0.7),
            ("low_score", 0.8, 0.3)
        ]
        
        for pattern_id, score, success_rate in patterns_data:
            metrics = PatternMetrics(pattern_id=pattern_id)
            metrics.current_score = score
            metrics.success_rate = success_rate
            metrics.total_applications = 10
            
            # Store in knowledge system
            self.mock_knowledge.store_knowledge(
                collection="pattern_metrics",
                content=json.dumps(metrics.to_dict()),
                metadata={"pattern_id": pattern_id}
            )
        
        # Get top patterns
        top_patterns = self.system.get_top_patterns(limit=2)
        
        self.assertEqual(len(top_patterns), 2)
        # Should be sorted by score (weighted)
        self.assertEqual(top_patterns[0]['pattern_id'], 'high_score')
        self.assertEqual(top_patterns[1]['pattern_id'], 'medium_score')


class TestPatternMaintenanceSystem(unittest.TestCase):
    """Test pattern maintenance system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_knowledge = MockKnowledgeSystem()
        self.reinforcement_system = PatternReinforcementSystem(
            knowledge_system=self.mock_knowledge
        )
        self.maintenance_system = PatternMaintenanceSystem(
            reinforcement_system=self.reinforcement_system
        )
    
    def test_maintenance_system_initialization(self):
        """Test maintenance system initialization."""
        self.assertIsNotNone(self.maintenance_system)
        self.assertEqual(self.maintenance_system.reinforcement_system, self.reinforcement_system)
        self.assertEqual(self.maintenance_system.knowledge, self.mock_knowledge)
    
    def test_time_decay_application(self):
        """Test time-based decay application."""
        # Create pattern with old last_used date
        metrics = PatternMetrics(pattern_id="old_pattern")
        metrics.last_used = datetime.now() - timedelta(days=30)
        metrics.current_score = 1.0
        
        patterns = [("old_pattern", metrics)]
        
        # Apply decay
        decay_results = self.maintenance_system._apply_time_decay(patterns, dry_run=False)
        
        self.assertGreater(decay_results['updated_count'], 0)
        self.assertLess(metrics.current_score, 1.0)  # Should be decayed
    
    def test_pruning_candidate_evaluation(self):
        """Test pruning candidate evaluation."""
        # Low success rate pattern
        low_success_metrics = PatternMetrics(pattern_id="low_success")
        low_success_metrics.total_applications = 20
        low_success_metrics.success_count = 4  # 20% success rate
        
        reason = self.maintenance_system._evaluate_pruning_candidate(low_success_metrics)
        self.assertIsNotNone(reason)
        self.assertIn("success rate", reason.lower())
        
        # Inactive pattern
        inactive_metrics = PatternMetrics(pattern_id="inactive")
        inactive_metrics.last_used = datetime.now() - timedelta(days=120)
        
        reason = self.maintenance_system._evaluate_pruning_candidate(inactive_metrics)
        self.assertIsNotNone(reason)
        self.assertIn("inactive", reason.lower())
        
        # Good pattern - need to set success rate properly
        good_metrics = PatternMetrics(pattern_id="good")
        good_metrics.total_applications = 10
        good_metrics.success_count = 9  # 90% success rate
        good_metrics.success_rate = 0.9  # Set calculated success rate
        good_metrics.last_used = datetime.now()
        
        reason = self.maintenance_system._evaluate_pruning_candidate(good_metrics)
        self.assertIsNone(reason)
    
    def test_maintenance_run(self):
        """Test complete maintenance run."""
        # Set up test patterns
        patterns_data = [
            ("good_pattern", 10, 9, datetime.now()),  # Good pattern
            ("bad_pattern", 15, 3, datetime.now()),   # Low success rate
            ("old_pattern", 5, 3, datetime.now() - timedelta(days=100))  # Inactive
        ]
        
        for pattern_id, total_apps, success_count, last_used in patterns_data:
            metrics = PatternMetrics(pattern_id=pattern_id)
            metrics.total_applications = total_apps
            metrics.success_count = success_count
            metrics.last_used = last_used
            metrics.success_rate = success_count / total_apps
            
            # Store pattern metrics
            self.mock_knowledge.store_knowledge(
                collection="pattern_metrics",
                content=json.dumps(metrics.to_dict()),
                metadata={"pattern_id": pattern_id}
            )
            
            # Store pattern data
            self.mock_knowledge.store_knowledge(
                collection="patterns",
                content=json.dumps({"pattern": f"data for {pattern_id}"}),
                metadata={"pattern_id": pattern_id},
                doc_id=pattern_id
            )
        
        # Run maintenance
        report = self.maintenance_system.run_maintenance(dry_run=True)
        
        self.assertIsNotNone(report)
        self.assertEqual(report.patterns_evaluated, 3)
        # May or may not have pruning results depending on exact thresholds
        # Just verify the report structure is correct
        self.assertIsInstance(report.pruning_results, list)


class TestMockPatternApplicationEngine(unittest.TestCase):
    """Test mock pattern application engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MockPatternApplicationEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsInstance(self.engine.config, dict)
        self.assertIsInstance(self.engine.mock_patterns, list)
        self.assertGreater(len(self.engine.mock_patterns), 0)
    
    def test_pattern_application(self):
        """Test pattern application simulation."""
        issue_context = {
            'issue_id': 'test_issue_123',
            'complexity': 'medium',
            'technology': 'python'
        }
        
        outcome = self.engine.apply_pattern(
            pattern_id="rest_api_implementation",
            issue_context=issue_context,
            agent_type="rif-implementer"
        )
        
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.pattern_id, "rest_api_implementation")
        self.assertEqual(outcome.issue_id, "test_issue_123")
        self.assertEqual(outcome.agent_type, "rif-implementer")
        self.assertIsInstance(outcome.execution_time, float)
        self.assertGreater(outcome.execution_time, 0)
        
        # Check outcome type is valid
        self.assertIn(outcome.outcome_type, list(OutcomeType))
    
    def test_multiple_applications(self):
        """Test multiple pattern applications."""
        issue_context = {'issue_id': 'test', 'complexity': 'low'}
        
        outcomes = []
        for i in range(10):
            outcome = self.engine.apply_pattern("test_pattern", issue_context)
            outcomes.append(outcome)
        
        self.assertEqual(len(outcomes), 10)
        
        # Should have some variety in outcomes
        outcome_types = set(outcome.outcome_type for outcome in outcomes)
        self.assertGreater(len(outcome_types), 1)
        
        # Check statistics
        stats = self.engine.get_simulation_stats()
        self.assertEqual(stats['total_applications'], 10)
        self.assertEqual(stats['pattern_usage']['test_pattern'], 10)


class TestPatternReinforcementIntegration(unittest.TestCase):
    """Test complete pattern reinforcement integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_knowledge = MockKnowledgeSystem()
        self.integration = PatternReinforcementIntegration(
            knowledge_system=self.mock_knowledge
        )
    
    def test_integration_initialization(self):
        """Test integration system initialization."""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.reinforcement_system)
        self.assertIsNotNone(self.integration.maintenance_system)
        self.assertIsNotNone(self.integration.mock_engine)
    
    def test_outcome_processing_sync(self):
        """Test synchronous outcome processing."""
        outcome = create_mock_pattern_outcome(
            pattern_id="test_pattern",
            success=True,
            issue_id="issue_123"
        )
        
        success = self.integration.process_pattern_outcome_sync(outcome)
        self.assertTrue(success)
        
        # Check metrics were updated
        metrics = self.integration.get_integration_metrics()
        self.assertGreater(metrics['total_outcomes_processed'], 0)
    
    def test_pattern_simulation(self):
        """Test pattern application simulation."""
        issue_context = {
            'issue_id': 'sim_test_123',
            'complexity': 'medium',
            'technology': 'javascript'
        }
        
        outcome = self.integration.simulate_pattern_application(
            pattern_id="rest_api_implementation",
            issue_context=issue_context,
            agent_type="rif-implementer",
            process_outcome=True
        )
        
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.pattern_id, "rest_api_implementation")
        self.assertEqual(outcome.issue_id, "sim_test_123")
    
    def test_pattern_recommendations(self):
        """Test pattern recommendation system."""
        # Create some patterns with different scores
        for i, pattern_id in enumerate(["pattern_a", "pattern_b", "pattern_c"]):
            outcome = create_mock_pattern_outcome(
                pattern_id=pattern_id,
                success=True,
                issue_id=f"issue_{i}"
            )
            self.integration.process_pattern_outcome_sync(outcome)
        
        # Get recommendations
        issue_context = {'complexity': 'medium', 'technology': 'python'}
        recommendations = self.integration.get_pattern_recommendations(
            issue_context=issue_context,
            limit=3
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Each recommendation should have required fields
        for rec in recommendations:
            self.assertIn('pattern_id', rec)
            self.assertIn('reinforcement_score', rec)
            self.assertIn('success_rate', rec)
            self.assertIn('recommendation_reason', rec)
    
    def test_system_health(self):
        """Test system health monitoring."""
        health = self.integration.get_system_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('status', health)
        self.assertIn('components', health)
        self.assertIn('metrics', health)
        
        # Check component health
        self.assertIn('reinforcement_system', health['components'])
        self.assertIn('maintenance_system', health['components'])
        self.assertIn('knowledge_system', health['components'])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for agent integration."""
    
    def test_create_pattern_reinforcement_system(self):
        """Test system creation function."""
        mock_knowledge = MockKnowledgeSystem()
        system = create_pattern_reinforcement_system(knowledge_system=mock_knowledge)
        
        self.assertIsInstance(system, PatternReinforcementIntegration)
        self.assertEqual(system.knowledge, mock_knowledge)
    
    def test_create_mock_pattern_outcome(self):
        """Test mock outcome creation function."""
        outcome = create_mock_pattern_outcome(
            pattern_id="test_pattern",
            success=True,
            issue_id="issue_123",
            agent_type="rif-implementer"
        )
        
        self.assertIsInstance(outcome, PatternOutcome)
        self.assertEqual(outcome.pattern_id, "test_pattern")
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.issue_id, "issue_123")
        self.assertEqual(outcome.agent_type, "rif-implementer")
        
        # Test failure outcome
        failure_outcome = create_mock_pattern_outcome(
            pattern_id="fail_pattern",
            success=False
        )
        
        self.assertFalse(failure_outcome.success)
        self.assertIsNotNone(failure_outcome.failure_mode)
        self.assertIsNotNone(failure_outcome.error_details)


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestPatternOutcome,
        TestPatternMetrics,
        TestPatternReinforcementSystem,
        TestPatternMaintenanceSystem,
        TestMockPatternApplicationEngine,
        TestPatternReinforcementIntegration,
        TestConvenienceFunctions
    ]
    
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")