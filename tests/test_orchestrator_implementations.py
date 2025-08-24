#!/usr/bin/env python3
"""
Comprehensive Tests for Orchestrator State Persistence and Monitoring Dashboard
Issues #55 and #56 Implementation Tests

This module provides thorough testing of the orchestrator persistence and monitoring
systems to ensure reliability and correctness.
"""

import pytest
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the commands directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'claude' / 'commands'))

try:
    from orchestrator_state_persistence import OrchestratorStatePersistence
    from orchestrator_monitoring_dashboard import OrchestratorMonitoringDashboard
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure DuckDB is installed: pip install duckdb")
    sys.exit(1)

class TestOrchestratorStatePersistence:
    """Test suite for orchestrator state persistence functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        # Use in-memory database for testing
        self.persistence = OrchestratorStatePersistence(':memory:')
        self.test_session_id = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if self.persistence:
            self.persistence.close()
    
    def test_database_initialization(self):
        """Test database schema creation and initialization."""
        # Verify tables were created
        result = self.persistence.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        
        table_names = [row[0] for row in result]
        expected_tables = ['orchestration_state', 'orchestration_decisions', 'orchestration_metrics']
        
        for table in expected_tables:
            assert table in table_names, f"Table {table} was not created"
    
    def test_session_creation(self):
        """Test orchestration session creation."""
        session_id = self.persistence.start_session('test_workflow', priority=1)
        
        assert session_id is not None
        assert len(session_id) > 0
        assert self.persistence.session_id == session_id
        
        # Verify session was stored in database
        result = self.persistence.db.execute(
            "SELECT session_id, current_state, workflow_type, priority FROM orchestration_state WHERE session_id = ?",
            [session_id]
        ).fetchone()
        
        assert result is not None
        assert result[0] == session_id
        assert result[1] == 'initialized'
        assert result[2] == 'test_workflow'
        assert result[3] == 1
        
        self.test_session_id = session_id
    
    def test_state_persistence(self):
        """Test state saving and retrieval."""
        # Create session first
        session_id = self.persistence.start_session('test_workflow')
        
        # Test state data
        test_state = {
            'current_state': 'implementing',
            'context': {
                'github_issues': [1, 2, 3],
                'active_agents': ['RIF-Implementer'],
                'start_time': datetime.now().isoformat()
            },
            'history': [
                {'state': 'initialized', 'timestamp': datetime.now().isoformat()},
                {'state': 'analyzing', 'timestamp': (datetime.now() + timedelta(minutes=1)).isoformat()},
                {'state': 'implementing', 'timestamp': (datetime.now() + timedelta(minutes=5)).isoformat()}
            ],
            'agent_assignments': {
                'RIF-Analyst': 'completed',
                'RIF-Implementer': 'active'
            }
        }
        
        # Save state
        success = self.persistence.save_state(test_state)
        assert success is True
        
        # Retrieve state
        recovered_state = self.persistence.recover_state(session_id)
        assert recovered_state is not None
        assert recovered_state['current_state'] == 'implementing'
        assert recovered_state['context']['github_issues'] == [1, 2, 3]
        assert len(recovered_state['history']) == 3
        assert recovered_state['agent_assignments']['RIF-Implementer'] == 'active'
    
    def test_decision_recording(self):
        """Test orchestration decision recording."""
        session_id = self.persistence.start_session('test_workflow')
        
        # Record a decision
        decision_id = self.persistence.record_decision(
            from_state='analyzing',
            to_state='implementing',
            reason='Requirements analysis completed successfully',
            agents_selected=['RIF-Implementer', 'RIF-Architect'],
            confidence_score=0.85,
            execution_time_ms=2500,
            success=True
        )
        
        assert decision_id != ""
        assert len(decision_id) > 0
        
        # Verify decision was stored
        decision_history = self.persistence.get_decision_history(session_id, limit=1)
        assert len(decision_history) == 1
        
        decision = decision_history[0]
        assert decision['from_state'] == 'analyzing'
        assert decision['to_state'] == 'implementing'
        assert decision['reason'] == 'Requirements analysis completed successfully'
        assert decision['agents_selected'] == ['RIF-Implementer', 'RIF-Architect']
        assert decision['confidence_score'] == 0.85
        assert decision['execution_time_ms'] == 2500
        assert decision['success'] is True
    
    def test_metric_recording(self):
        """Test performance metric recording."""
        session_id = self.persistence.start_session('test_workflow')
        
        # Record various metrics
        metrics = [
            ('performance', 'execution_time', 1500.5, {'agent': 'RIF-Implementer'}),
            ('quality', 'test_coverage', 87.5, {'test_suite': 'unit_tests'}),
            ('usage', 'api_calls', 42.0, {'endpoint': 'github_api'})
        ]
        
        metric_ids = []
        for metric_type, metric_name, value, metadata in metrics:
            metric_id = self.persistence.record_metric(metric_type, metric_name, value, metadata)
            metric_ids.append(metric_id)
            assert metric_id != ""
        
        # Verify metrics were stored
        result = self.persistence.db.execute(
            "SELECT COUNT(*) FROM orchestration_metrics WHERE session_id = ?",
            [session_id]
        ).fetchone()
        
        assert result[0] == len(metrics)
    
    def test_state_recovery(self):
        """Test complete state recovery after interruption."""
        # Create and populate session
        session_id = self.persistence.start_session('recovery_test', priority=2)
        
        # Simulate workflow progression
        states = ['analyzing', 'planning', 'implementing']
        for i, state in enumerate(states):
            test_state = {
                'current_state': state,
                'context': {'step': i + 1, 'issues': [10, 20, 30]},
                'history': [{'state': s, 'step': j} for j, s in enumerate(states[:i+1])],
                'agent_assignments': {f'Agent-{i}': 'active'}
            }
            
            self.persistence.save_state(test_state)
            
            # Record decision
            if i > 0:
                self.persistence.record_decision(
                    states[i-1], state, f'Transition to {state}',
                    [f'Agent-{i}'], 0.9, 1000
                )
        
        # Recover state
        recovered = self.persistence.recover_state(session_id)
        assert recovered is not None
        assert recovered['current_state'] == 'implementing'
        assert recovered['context']['step'] == 3
        assert recovered['priority'] == 2
        
        # Verify decision history
        decisions = self.persistence.get_decision_history(session_id)
        assert len(decisions) == 2  # Two transitions
    
    def test_active_sessions(self):
        """Test active session listing."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_id = self.persistence.start_session(f'workflow_{i}', priority=i)
            session_ids.append(session_id)
            
            # Set different states
            states = ['analyzing', 'implementing', 'completed']
            self.persistence.save_state({
                'current_state': states[i],
                'context': {},
                'history': [],
                'agent_assignments': {}
            })
        
        # Get active sessions
        active_sessions = self.persistence.get_active_sessions()
        
        # Should exclude 'completed' state
        assert len(active_sessions) == 2
        
        # Verify sorting by priority
        assert active_sessions[0]['priority'] >= active_sessions[1]['priority']
    
    def test_performance_stats(self):
        """Test performance statistics calculation."""
        session_id = self.persistence.start_session('perf_test')
        
        # Record multiple decisions with varying performance
        decisions_data = [
            ('state1', 'state2', 'reason1', ['agent1'], 0.9, 1000, True),
            ('state2', 'state3', 'reason2', ['agent2'], 0.8, 2000, True),
            ('state3', 'state4', 'reason3', ['agent3'], 0.7, 3000, False),
            ('state4', 'state5', 'reason4', ['agent4'], 0.95, 500, True)
        ]
        
        for from_state, to_state, reason, agents, confidence, exec_time, success in decisions_data:
            self.persistence.record_decision(
                from_state, to_state, reason, agents, confidence, exec_time, success
            )
        
        # Get performance stats
        stats = self.persistence.get_performance_stats(session_id)
        
        assert 'total_decisions' in stats
        assert stats['total_decisions'] == 4
        assert 'success_rate' in stats
        assert stats['success_rate'] == 0.75  # 3 successful out of 4
        assert 'avg_execution_time_ms' in stats
        assert stats['avg_execution_time_ms'] == 1625.0  # Average of 1000, 2000, 3000, 500
    
    def test_data_validation(self):
        """Test state data validation and integrity checks."""
        session_id = self.persistence.start_session('validation_test')
        
        # Save state with valid data
        valid_state = {
            'current_state': 'testing',
            'context': {'valid': True},
            'history': [{'action': 'test'}],
            'agent_assignments': {}
        }
        
        success = self.persistence.save_state(valid_state)
        assert success is True
        
        # Test validation
        validation_report = self.persistence.validate_state_integrity(session_id)
        assert validation_report['valid'] is True
        assert len(validation_report['errors']) == 0
        assert validation_report['sessions_checked'] == 1


class TestOrchestratorMonitoringDashboard:
    """Test suite for orchestrator monitoring dashboard functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.persistence = OrchestratorStatePersistence(':memory:')
        self.dashboard = OrchestratorMonitoringDashboard(self.persistence)
        self.test_session_id = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if self.persistence:
            self.persistence.close()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization with persistence system."""
        assert self.dashboard.persistence is not None
        assert self.dashboard.active_workflows == {}
        assert self.dashboard.refresh_interval == 1.0
    
    def test_dashboard_data_generation(self):
        """Test comprehensive dashboard data generation."""
        # Create test session with data
        session_id = self.persistence.start_session('dashboard_test', priority=1)
        self.persistence.save_state({
            'current_state': 'implementing',
            'context': {'github_issues': [1, 2]},
            'history': [{'state': 'initialized'}],
            'agent_assignments': {'RIF-Implementer': 'active'}
        })
        
        # Generate dashboard data
        dashboard_data = self.dashboard.get_dashboard_data()
        
        # Verify required sections
        required_sections = [
            'timestamp', 'active_workflows', 'state_distribution',
            'transition_metrics', 'performance_stats', 'recent_decisions',
            'agent_status', 'system_health', 'real_time_events'
        ]
        
        for section in required_sections:
            assert section in dashboard_data, f"Missing dashboard section: {section}"
    
    def test_active_workflows_display(self):
        """Test active workflow information display."""
        # Create multiple test sessions
        sessions = []
        for i in range(3):
            session_id = self.persistence.start_session(f'workflow_{i}', priority=i)
            sessions.append(session_id)
            
            self.persistence.save_state({
                'current_state': f'state_{i}',
                'context': {'github_issues': [i*10, i*10+1]},
                'history': [{'state': 'initialized'}],
                'agent_assignments': {f'Agent-{i}': 'active'}
            })
        
        # Get active workflows
        active_workflows = self.dashboard.get_active_workflows()
        
        assert len(active_workflows) == 3
        
        # Verify workflow information structure
        for workflow in active_workflows:
            required_fields = [
                'session_id', 'current_state', 'workflow_type', 'priority',
                'progress_percentage', 'duration_minutes', 'active_agents',
                'issues_count', 'created_at', 'updated_at'
            ]
            
            for field in required_fields:
                assert field in workflow, f"Missing workflow field: {field}"
    
    def test_state_distribution(self):
        """Test state distribution analysis."""
        # Create sessions in different states
        states_data = [
            ('analyzing', 'workflow_a'),
            ('implementing', 'workflow_b'),
            ('implementing', 'workflow_c'),
            ('validating', 'workflow_d')
        ]
        
        for state, workflow_type in states_data:
            session_id = self.persistence.start_session(workflow_type)
            self.persistence.save_state({
                'current_state': state,
                'context': {},
                'history': [],
                'agent_assignments': {}
            })
        
        # Get state distribution
        distribution = self.dashboard.get_state_distribution()
        
        assert 'total_active' in distribution
        assert distribution['total_active'] == 4
        
        assert 'by_state' in distribution
        assert distribution['by_state']['implementing'] == 2
        assert distribution['by_state']['analyzing'] == 1
        assert distribution['by_state']['validating'] == 1
    
    def test_workflow_visualization(self):
        """Test workflow visualization generation."""
        session_id = self.persistence.start_session('viz_test')
        
        # Create workflow progression with decisions
        states = ['initialized', 'analyzing', 'implementing', 'validating']
        for i in range(len(states) - 1):
            self.persistence.record_decision(
                states[i], states[i+1], f'Transition {i+1}',
                [f'Agent-{i+1}'], 0.9, 1000
            )
        
        # Update current state
        self.persistence.save_state({
            'current_state': 'validating',
            'context': {},
            'history': [{'state': s} for s in states],
            'agent_assignments': {}
        })
        
        # Generate visualization
        viz_data = self.dashboard.visualize_workflow(session_id)
        
        assert viz_data is not None
        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert len(viz_data['nodes']) == 4  # All states
        assert len(viz_data['edges']) == 3  # Three transitions
        
        # Verify current state marking
        current_node = next(node for node in viz_data['nodes'] if node['status'] == 'current')
        assert current_node['id'] == 'validating'
    
    def test_transition_metrics(self):
        """Test state transition metrics calculation."""
        session_id = self.persistence.start_session('metrics_test')
        
        # Record various transitions with different performance
        transitions = [
            ('analyzing', 'planning', 1500, 0.9),
            ('planning', 'implementing', 2000, 0.8),
            ('analyzing', 'planning', 1000, 0.95),  # Same transition again
            ('implementing', 'validating', 3000, 0.7)
        ]
        
        for from_state, to_state, exec_time, confidence in transitions:
            self.persistence.record_decision(
                from_state, to_state, 'Test transition',
                ['test_agent'], confidence, exec_time, True
            )
        
        # Get transition metrics
        metrics = self.dashboard.get_transition_metrics()
        
        assert 'transition_matrix' in metrics
        assert 'performance_by_transition' in metrics
        
        # Verify transition counting
        analyzing_planning_key = 'analyzing->planning'
        assert analyzing_planning_key in metrics['performance_by_transition']
        assert metrics['performance_by_transition'][analyzing_planning_key]['count'] == 2
    
    def test_real_time_event_tracking(self):
        """Test real-time event tracking functionality."""
        # Track various events
        events = [
            ('state_transition', {'from': 'analyzing', 'to': 'implementing', 'duration': 1500}),
            ('agent_launch', {'agent': 'RIF-Implementer', 'session_id': 'test'}),
            ('workflow_complete', {'session_id': 'test', 'total_duration': 5000}),
            ('error_occurred', {'error': 'Test error', 'severity': 'low'})
        ]
        
        for event_type, data in events:
            self.dashboard.track_metrics(event_type, data)
        
        # Verify events were tracked
        assert len(self.dashboard.real_time_events) == 4
        
        # Verify metric cache updates
        assert self.dashboard.metrics_cache['analyzing->implementing']['count'] == 1
        assert self.dashboard.metrics_cache['agent_RIF-Implementer']['count'] == 1
        assert self.dashboard.metrics_cache['completions']['count'] == 1
        assert self.dashboard.metrics_cache['errors']['count'] == 1
    
    def test_system_health_monitoring(self):
        """Test system health monitoring and alerting."""
        session_id = self.persistence.start_session('health_test')
        
        # Create some decisions with errors to test health calculation
        for i in range(10):
            success = i < 7  # 70% success rate
            self.persistence.record_decision(
                'test_from', 'test_to', f'Decision {i}',
                ['test_agent'], 0.8, 1000, success
            )
        
        # Get system health
        health = self.dashboard.get_system_health()
        
        assert 'status' in health
        assert 'health_score' in health
        assert 'error_rate' in health
        assert 'alerts' in health
        
        # Should detect the high error rate
        assert health['error_rate'] > 0.2
    
    def test_report_generation(self):
        """Test comprehensive report generation."""
        session_id = self.persistence.start_session('report_test')
        
        # Create some test data
        self.persistence.save_state({
            'current_state': 'implementing',
            'context': {'test': True},
            'history': [],
            'agent_assignments': {}
        })
        
        self.persistence.record_decision(
            'analyzing', 'implementing', 'Test decision',
            ['test_agent'], 0.9, 1500, True
        )
        
        # Generate report for specific session
        session_report = self.dashboard.generate_report(session_id, timeframe_hours=1)
        
        assert 'report_id' in session_report
        assert 'summary' in session_report
        assert 'performance' in session_report
        assert 'recommendations' in session_report
        assert session_report['session_id'] == session_id
        
        # Generate overall report
        overall_report = self.dashboard.generate_report(timeframe_hours=24)
        
        assert overall_report['session_id'] is None
        assert 'summary' in overall_report


class TestIntegration:
    """Integration tests for persistence and monitoring systems working together."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with persistence and monitoring."""
        # Initialize systems
        persistence = OrchestratorStatePersistence(':memory:')
        dashboard = OrchestratorMonitoringDashboard(persistence)
        
        try:
            # Start orchestration session
            session_id = persistence.start_session('integration_test', priority=1)
            
            # Simulate workflow progression
            workflow_states = [
                ('initialized', 'analyzing', 'Starting analysis'),
                ('analyzing', 'planning', 'Analysis complete'),
                ('planning', 'implementing', 'Planning approved'),
                ('implementing', 'validating', 'Implementation ready')
            ]
            
            for from_state, to_state, reason in workflow_states:
                # Record decision
                decision_id = persistence.record_decision(
                    from_state, to_state, reason,
                    [f'Agent-{to_state}'], 0.85, 2000, True
                )
                assert decision_id != ""
                
                # Update state
                success = persistence.save_state({
                    'current_state': to_state,
                    'context': {'step': to_state, 'issues': [1, 2, 3]},
                    'history': [{'state': to_state, 'timestamp': datetime.now().isoformat()}],
                    'agent_assignments': {f'Agent-{to_state}': 'active'}
                })
                assert success
                
                # Track metrics in dashboard
                dashboard.track_metrics('state_transition', {
                    'from': from_state,
                    'to': to_state,
                    'duration': 2000
                })
            
            # Generate final dashboard data
            dashboard_data = dashboard.get_dashboard_data()
            assert len(dashboard_data['active_workflows']) == 1
            assert dashboard_data['active_workflows'][0]['current_state'] == 'validating'
            
            # Generate workflow visualization
            viz = dashboard.visualize_workflow(session_id)
            assert viz is not None
            assert len(viz['nodes']) == 5  # All states including initial
            assert len(viz['edges']) == 4  # All transitions
            
            # Verify decision history
            decisions = persistence.get_decision_history(session_id)
            assert len(decisions) == 4
            
            # Generate comprehensive report
            report = dashboard.generate_report(session_id)
            assert 'error' not in report
            assert report['summary']['session_focus'] == session_id
            
        finally:
            persistence.close()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        persistence = OrchestratorStatePersistence(':memory:')
        dashboard = OrchestratorMonitoringDashboard(persistence)
        
        try:
            session_id = persistence.start_session('error_test')
            
            # Test invalid state data handling
            invalid_state = {
                'current_state': 'testing'
                # Missing required fields
            }
            
            # Should handle gracefully
            try:
                result = persistence.save_state(invalid_state)
                # Should return False or handle gracefully
            except Exception:
                pass  # Expected to fail
            
            # Test recovery from partial data
            valid_state = {
                'current_state': 'testing',
                'context': {},
                'history': [],
                'agent_assignments': {}
            }
            
            success = persistence.save_state(valid_state)
            assert success
            
            # Test dashboard with minimal data
            dashboard_data = dashboard.get_dashboard_data()
            assert 'error' not in dashboard_data or isinstance(dashboard_data.get('error'), str)
            
        finally:
            persistence.close()


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestOrchestratorStatePersistence,
        TestOrchestratorMonitoringDashboard,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            print(f"  Running {test_method}...", end=' ')
            
            try:
                # Create test instance and run test
                test_instance = test_class()
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test method
                getattr(test_instance, test_method)()
                
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                
                print("✅ PASSED")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ FAILED: {e}")
                failed_tests.append({
                    'class': test_class.__name__,
                    'method': test_method,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print(f"\n{'='*60}")
        print(f"FAILED TESTS")
        print(f"{'='*60}")
        for failure in failed_tests:
            print(f"\n{failure['class']}.{failure['method']}:")
            print(f"  Error: {failure['error']}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)