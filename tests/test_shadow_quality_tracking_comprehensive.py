#!/usr/bin/env python3
"""
Test Suite for Shadow Quality Tracking System - Issue #142
Comprehensive testing for quality monitoring, adversarial analysis, and evidence consolidation

Tests cover:
- Quality metric tracking and validation
- Adversarial finding recording and analysis  
- Quality decision making and evidence consolidation
- Integration with DPIBS benchmarking system
- Continuous monitoring and alert systems
"""

import os
import sys
import json
import time
import tempfile
import unittest
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add RIF to path
sys.path.insert(0, '/Users/cal/DEV/RIF')

from systems.shadow_quality_tracking import (
    ShadowQualityTracker, QualityMetricType, QualityMetric, QualitySession,
    create_shadow_quality_tracker
)


class TestShadowQualityTracker(unittest.TestCase):
    """Test suite for core shadow quality tracking functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix='shadow_quality_test_')
        self.test_db_path = os.path.join(self.temp_dir, 'test_quality.db')
        self.tracker = ShadowQualityTracker(self.test_db_path)
        
    def tearDown(self):
        """Clean up test environment"""
        self.tracker.stop_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quality_session_creation(self):
        """Test quality session creation and management"""
        session_id = self.tracker.start_quality_session(142, {
            'validation_phase': 1,
            'quality_focus': 'DPIBS_validation'
        })
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.tracker.active_sessions)
        
        session = self.tracker.active_sessions[session_id]
        self.assertEqual(session.issue_number, 142)
        self.assertTrue(session.is_active)
        self.assertEqual(len(session.metrics), 0)
    
    def test_quality_metric_recording(self):
        """Test quality metric recording and validation"""
        session_id = self.tracker.start_quality_session(142)
        
        # Record context relevance metric
        metric_id = self.tracker.record_quality_metric(
            session_id,
            QualityMetricType.CONTEXT_RELEVANCE,
            92.5,
            'test_agent',
            {'test_scenario': 'context_validation'},
            {'measurement_type': 'simulated_test'}
        )
        
        self.assertIsNotNone(metric_id)
        
        session = self.tracker.active_sessions[session_id]
        # Should have 1 explicitly recorded metric (continuous monitoring may add system performance metrics)
        cr_metrics = [m for m in session.metrics if m.metric_type == QualityMetricType.CONTEXT_RELEVANCE]
        self.assertEqual(len(cr_metrics), 1)
        
        metric = cr_metrics[0]
        self.assertEqual(metric.value, 92.5)
        self.assertEqual(metric.target_value, 90.0)  # From quality targets
        self.assertTrue(metric.target_met)
        self.assertEqual(metric.metric_type, QualityMetricType.CONTEXT_RELEVANCE)
        self.assertEqual(metric.source, 'test_agent')
    
    def test_benchmarking_accuracy_tracking(self):
        """Test benchmarking accuracy metric tracking"""
        session_id = self.tracker.start_quality_session(142)
        
        # Record benchmarking accuracy metric
        metric_id = self.tracker.record_quality_metric(
            session_id,
            QualityMetricType.BENCHMARKING_ACCURACY,
            87.3,
            'dpibs_benchmarking',
            {'benchmark_type': 'enhanced_nlp'},
            {'accuracy_breakdown': {'nlp': 87.3, 'evidence': 91.2}}
        )
        
        session = self.tracker.active_sessions[session_id]
        metric = session.get_latest_metric(QualityMetricType.BENCHMARKING_ACCURACY)
        
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 87.3)
        self.assertEqual(metric.target_value, 85.0)  # From quality targets
        self.assertTrue(metric.target_met)
        self.assertEqual(metric.quality_score, 1.0)  # Capped at 1.0
    
    def test_adversarial_finding_recording(self):
        """Test adversarial finding recording and management"""
        session_id = self.tracker.start_quality_session(142)
        
        finding_id = self.tracker.record_adversarial_finding(
            session_id,
            'performance_degradation',
            'medium',
            'Query response time occasionally exceeds threshold',
            {'max_response_time_ms': 234, 'threshold_ms': 200},
            'May impact user experience during validation',
            ['Optimize database queries', 'Implement caching']
        )
        
        self.assertIsNotNone(finding_id)
        
        session = self.tracker.active_sessions[session_id]
        self.assertEqual(len(session.adversarial_findings), 1)
        
        finding = session.adversarial_findings[0]
        self.assertEqual(finding['finding_type'], 'performance_degradation')
        self.assertEqual(finding['severity'], 'medium')
        self.assertEqual(finding['status'], 'open')
        self.assertIn('max_response_time_ms', finding['evidence'])
    
    def test_quality_decision_making(self):
        """Test quality decision recording and analysis"""
        session_id = self.tracker.start_quality_session(142)
        
        decision_id = self.tracker.record_quality_decision(
            session_id,
            'validation_approval',
            {'gate_type': 'context_relevance', 'threshold_met': True},
            'Context relevance consistently exceeds target',
            0.92,
            0.88
        )
        
        self.assertIsNotNone(decision_id)
        
        session = self.tracker.active_sessions[session_id]
        self.assertEqual(len(session.quality_decisions), 1)
        
        decision = session.quality_decisions[0]
        self.assertEqual(decision['decision_type'], 'validation_approval')
        self.assertEqual(decision['confidence_score'], 0.92)
        self.assertEqual(decision['evidence_quality'], 0.88)
    
    def test_session_status_reporting(self):
        """Test session status reporting functionality"""
        session_id = self.tracker.start_quality_session(142, {
            'validation_phase': 1,
            'expected_duration_days': 7
        })
        
        # Add some metrics
        self.tracker.record_quality_metric(
            session_id, QualityMetricType.CONTEXT_RELEVANCE, 91.5, 'test_agent'
        )
        self.tracker.record_quality_metric(
            session_id, QualityMetricType.BENCHMARKING_ACCURACY, 86.2, 'dpibs_engine'
        )
        
        status = self.tracker.get_session_status(session_id)
        
        self.assertEqual(status['status'], 'active')
        self.assertEqual(status['issue_number'], 142)
        # Should have at least 2 explicitly recorded metrics (may have more from monitoring)
        self.assertGreaterEqual(status['total_metrics'], 2)
        self.assertIn('context_relevance', status['latest_metrics'])
        self.assertIn('benchmarking_accuracy', status['latest_metrics'])
        
        # Check latest metrics
        cr_metric = status['latest_metrics']['context_relevance']
        self.assertEqual(cr_metric['value'], 91.5)
        self.assertEqual(cr_metric['target'], 90.0)
        self.assertTrue(cr_metric['target_met'])
    
    def test_session_completion_and_reporting(self):
        """Test session completion and final report generation"""
        session_id = self.tracker.start_quality_session(142)
        
        # Add comprehensive test data
        metrics_data = [
            (QualityMetricType.CONTEXT_RELEVANCE, 92.5, 'agent_1'),
            (QualityMetricType.BENCHMARKING_ACCURACY, 87.1, 'dpibs_engine'),
            (QualityMetricType.SYSTEM_PERFORMANCE, 99.8, 'system_monitor'),
            (QualityMetricType.AGENT_IMPROVEMENT, 78.3, 'improvement_tracker')
        ]
        
        for metric_type, value, source in metrics_data:
            self.tracker.record_quality_metric(session_id, metric_type, value, source)
        
        # Add adversarial findings
        self.tracker.record_adversarial_finding(
            session_id, 'edge_case', 'low',
            'Minor edge case in input validation',
            {'test_case': 'empty_input'}, 'Low impact', ['Add input validation']
        )
        
        # Add quality decisions
        self.tracker.record_quality_decision(
            session_id, 'quality_gate_pass', {'gate': 'context_relevance'},
            'Target exceeded', 0.95, 0.90
        )
        
        # End session and get report
        final_report = self.tracker.end_quality_session(session_id)
        
        self.assertIsInstance(final_report, dict)
        self.assertEqual(final_report['session_id'], session_id)
        self.assertEqual(final_report['issue_number'], 142)
        self.assertGreater(final_report['duration_hours'], 0)
        
        # Check quality summary
        quality_summary = final_report['quality_summary']
        self.assertIn('context_relevance', quality_summary)
        self.assertIn('benchmarking_accuracy', quality_summary)
        
        cr_summary = quality_summary['context_relevance']
        self.assertEqual(cr_summary['total_measurements'], 1)
        self.assertEqual(cr_summary['average_value'], 92.5)
        self.assertEqual(cr_summary['target_achievement_rate'], 100.0)
        
        # Check overall assessment
        overall = final_report['overall_assessment']
        self.assertIn('overall_quality_score', overall)
        self.assertTrue(overall['context_relevance_target_met'])
        self.assertTrue(overall['benchmarking_accuracy_target_met'])
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring functionality"""
        session_id = self.tracker.start_quality_session(142)
        
        # Start monitoring
        self.tracker.start_monitoring()
        self.assertTrue(self.tracker.monitoring_active)
        self.assertIsNotNone(self.tracker.monitoring_thread)
        
        # Let monitoring run briefly
        time.sleep(2)
        
        # Check that system performance metrics are being recorded
        session = self.tracker.active_sessions[session_id]
        system_metrics = session.get_metrics_by_type(QualityMetricType.SYSTEM_PERFORMANCE)
        
        # Should have at least one system performance metric from monitoring
        self.assertGreater(len(system_metrics), 0)
        
        # Stop monitoring
        self.tracker.stop_monitoring()
        self.assertFalse(self.tracker.monitoring_active)
    
    def test_quality_alert_generation(self):
        """Test quality alert generation for threshold violations"""
        session_id = self.tracker.start_quality_session(142)
        
        # Record metric below target (should trigger alert)
        with patch.object(self.tracker.logger, 'warning') as mock_warning:
            self.tracker.record_quality_metric(
                session_id,
                QualityMetricType.CONTEXT_RELEVANCE,
                75.0,  # Below 90% target
                'failing_agent'
            )
            
            # Should have logged a warning
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            self.assertIn('Quality Alert', call_args)
            self.assertIn('context_relevance', call_args)
    
    def test_dashboard_generation(self):
        """Test monitoring dashboard generation"""
        session_id = self.tracker.start_quality_session(142)
        
        # Add some test data
        self.tracker.record_quality_metric(
            session_id, QualityMetricType.CONTEXT_RELEVANCE, 93.2, 'test_agent'
        )
        
        # Generate dashboard
        dashboard_path = self.tracker.create_monitoring_dashboard()
        
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Read and validate dashboard content
        with open(dashboard_path, 'r') as f:
            content = f.read()
        
        self.assertIn('Shadow Quality Tracking Dashboard', content)
        self.assertIn('Issue #142', content)
        self.assertIn('context_relevance', content)
        self.assertIn('93.2', content)


class TestQualityMetrics(unittest.TestCase):
    """Test suite for quality metric functionality"""
    
    def test_quality_metric_creation(self):
        """Test quality metric object creation and validation"""
        metric = QualityMetric(
            id='test-metric-1',
            metric_type=QualityMetricType.CONTEXT_RELEVANCE,
            timestamp=datetime.now(),
            value=92.5,
            target_value=90.0,
            source='test_agent',
            context={'test': True},
            evidence={'measurement': 'simulated'}
        )
        
        self.assertEqual(metric.value, 92.5)
        self.assertEqual(metric.target_value, 90.0)
        self.assertTrue(metric.target_met)
        self.assertEqual(metric.quality_score, 1.0)  # Quality score is capped at 1.0
        
    def test_quality_session_management(self):
        """Test quality session object management"""
        session = QualitySession(
            session_id='test-session-1',
            issue_number=142,
            start_time=datetime.now(),
            end_time=None
        )
        
        self.assertTrue(session.is_active)
        self.assertEqual(len(session.metrics), 0)
        self.assertGreater(session.duration_hours, 0)
        
        # Add metrics
        metric1 = QualityMetric(
            id='m1', metric_type=QualityMetricType.CONTEXT_RELEVANCE,
            timestamp=datetime.now(), value=90.0, target_value=90.0,
            source='agent1', context={}, evidence={}
        )
        
        metric2 = QualityMetric(
            id='m2', metric_type=QualityMetricType.BENCHMARKING_ACCURACY,
            timestamp=datetime.now(), value=85.0, target_value=85.0,
            source='agent2', context={}, evidence={}
        )
        
        session.metrics.extend([metric1, metric2])
        
        # Test metric retrieval
        cr_metrics = session.get_metrics_by_type(QualityMetricType.CONTEXT_RELEVANCE)
        self.assertEqual(len(cr_metrics), 1)
        self.assertEqual(cr_metrics[0].value, 90.0)
        
        latest_ba = session.get_latest_metric(QualityMetricType.BENCHMARKING_ACCURACY)
        self.assertIsNotNone(latest_ba)
        self.assertEqual(latest_ba.value, 85.0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise in tests
    
    print("🧪 Running Shadow Quality Tracking Test Suite - Issue #142")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestShadowQualityTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results Summary:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Exit with appropriate code
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    exit(0 if success else 1)