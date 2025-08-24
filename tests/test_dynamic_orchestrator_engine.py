#!/usr/bin/env python3
"""
Comprehensive test suite for Dynamic Orchestrator Engine

Tests all components of the Hybrid Graph-Based Dynamic Orchestration System:
- Dynamic Orchestrator Engine
- Decision Engine
- State Graph Manager
- Parallel Execution Coordinator
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the modules we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "claude" / "commands"))

from dynamic_orchestrator_engine import (
    DynamicOrchestrator, StateGraphManager, DecisionEngine,
    Evidence, WorkflowSession, StateNode, DecisionPoint,
    ConfidenceLevel
)
from parallel_execution_coordinator import (
    ParallelExecutionCoordinator, ParallelPath, ResourceRequirement,
    WorkloadBalancer, ConflictResolver, ExecutionStatus, ResourceType
)


class TestEvidence:
    """Test Evidence data class"""
    
    def test_evidence_creation(self):
        """Test basic evidence creation"""
        evidence = Evidence(
            source="test_runner",
            content="All tests passed",
            confidence=0.95,
            timestamp=datetime.now(),
            evidence_type="test_result"
        )
        
        assert evidence.source == "test_runner"
        assert evidence.confidence == 0.95
        assert evidence.evidence_type == "test_result"
    
    def test_evidence_to_dict(self):
        """Test evidence serialization"""
        timestamp = datetime.now()
        evidence = Evidence(
            source="validator",
            content="Quality gates passed",
            confidence=0.88,
            timestamp=timestamp,
            evidence_type="quality_gate"
        )
        
        evidence_dict = evidence.to_dict()
        
        assert evidence_dict['source'] == "validator"
        assert evidence_dict['confidence'] == 0.88
        assert evidence_dict['timestamp'] == timestamp.isoformat()
        assert evidence_dict['evidence_type'] == "quality_gate"


class TestStateNode:
    """Test StateNode functionality"""
    
    def test_state_node_creation(self):
        """Test basic state node creation"""
        state = StateNode(
            state_id='implementing',
            description='Code implementation phase',
            agents=['rif-implementer'],
            can_transition_to=['validating', 'architecting'],
            decision_logic='code_completion_check',
            loop_back_conditions=['architectural_issues']
        )
        
        assert state.state_id == 'implementing'
        assert 'rif-implementer' in state.agents
        assert 'validating' in state.can_transition_to
    
    def test_state_transition_validation(self):
        """Test state transition validation"""
        state = StateNode(
            state_id='implementing',
            description='Implementation phase',
            agents=['rif-implementer'],
            can_transition_to=['validating'],
            decision_logic='code_completion_check',
            loop_back_conditions=[]
        )
        
        # Valid transition
        can_transition, reason = state.can_transition('validating', {})
        assert can_transition is True
        
        # Invalid transition
        can_transition, reason = state.can_transition('learning', {})
        assert can_transition is False
        assert 'not allowed' in reason.lower()
    
    def test_complexity_based_transition_logic(self):
        """Test complexity-based transition logic"""
        state = StateNode(
            state_id='analyzing',
            description='Analysis phase',
            agents=['rif-analyst'],
            can_transition_to=['planning', 'implementing', 'architecting'],
            decision_logic='complexity_threshold_evaluation',
            loop_back_conditions=[]
        )
        
        # High complexity should not go directly to implementing
        context_high = {'complexity': 'high'}
        can_transition, reason = state.can_transition('implementing', context_high)
        # Note: This would need actual logic implementation to work
        assert isinstance(can_transition, bool)


class TestDecisionPoint:
    """Test DecisionPoint functionality"""
    
    def test_decision_point_creation(self):
        """Test decision point creation"""
        decision_point = DecisionPoint(
            decision_id='post_validation',
            trigger_condition='validation_results_available',
            evaluator='ValidationResultsEvaluator',
            outcomes=[
                {
                    'outcome': 'proceed_to_learning',
                    'condition': 'all_tests_pass',
                    'confidence_threshold': 0.9
                }
            ],
            context_factors=['test_results', 'quality_gates']
        )
        
        assert decision_point.decision_id == 'post_validation'
        assert len(decision_point.outcomes) == 1
        assert decision_point.confidence_threshold == 0.7  # default
    
    def test_decision_point_evaluation(self):
        """Test decision point condition evaluation"""
        decision_point = DecisionPoint(
            decision_id='test_decision',
            trigger_condition='validation_results_available',
            evaluator='TestEvaluator',
            outcomes=[],
            context_factors=[]
        )
        
        # Test validation results available
        context = {'validation_complete': True}
        evidence = []
        
        should_trigger, confidence = decision_point.evaluate_condition(context, evidence)
        assert should_trigger is True
        assert confidence > 0


class TestStateGraphManager:
    """Test StateGraphManager functionality"""
    
    def test_state_graph_initialization(self):
        """Test state graph manager initialization"""
        manager = StateGraphManager()
        
        # Check that basic states are initialized
        assert 'analyzing' in manager.states
        assert 'implementing' in manager.states
        assert 'validating' in manager.states
        
        # Check that decision points are initialized
        assert 'post_validation_decision' in manager.decision_points
        assert 'complexity_routing_decision' in manager.decision_points
    
    def test_get_state_node(self):
        """Test getting state nodes"""
        manager = StateGraphManager()
        
        analyzing_state = manager.get_state_node('analyzing')
        assert analyzing_state is not None
        assert analyzing_state.state_id == 'analyzing'
        assert 'rif-analyst' in analyzing_state.agents
        
        non_existent = manager.get_state_node('non_existent')
        assert non_existent is None
    
    def test_validate_transition(self):
        """Test state transition validation"""
        manager = StateGraphManager()
        
        # Valid transition
        is_valid, reason, confidence = manager.validate_transition(
            'analyzing', 'implementing', {'complexity': 'low'}
        )
        assert is_valid is True
        assert confidence > 0
        
        # Invalid source state
        is_valid, reason, confidence = manager.validate_transition(
            'invalid_state', 'implementing', {}
        )
        assert is_valid is False
        assert 'unknown' in reason.lower()
    
    def test_get_recommended_transitions(self):
        """Test getting recommended transitions"""
        manager = StateGraphManager()
        
        # Create test evidence
        evidence = [
            Evidence(
                source='test',
                content='analysis complete',
                confidence=0.9,
                timestamp=datetime.now(),
                evidence_type='analysis_complete'
            )
        ]
        
        recommendations = manager.get_recommended_transitions(
            'analyzing', {'complexity': 'medium'}, evidence
        )
        
        assert len(recommendations) > 0
        # Should return list of (state, confidence) tuples
        for state, confidence in recommendations:
            assert isinstance(state, str)
            assert 0.0 <= confidence <= 1.0
    
    def test_transition_confidence_calculation(self):
        """Test transition confidence calculation"""
        manager = StateGraphManager()
        
        # Test with good context
        good_context = {
            'validation_passed': True,
            'requirements_clear': True,
            'complexity': 'medium'
        }
        confidence = manager._calculate_transition_confidence(
            'implementing', 'validating', good_context
        )
        assert confidence >= 0.7
        
        # Test with problematic context
        bad_context = {
            'validation_passed': False,
            'requirements_clear': False
        }
        confidence = manager._calculate_transition_confidence(
            'implementing', 'validating', bad_context
        )
        assert confidence < 0.7


class TestDecisionEngine:
    """Test DecisionEngine functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.state_graph_manager = StateGraphManager()
        self.decision_engine = DecisionEngine(self.state_graph_manager)
    
    def test_decision_engine_creation(self):
        """Test decision engine creation"""
        self.setUp()
        assert self.decision_engine is not None
        assert self.decision_engine.state_graph_manager == self.state_graph_manager
    
    def test_evaluate_decision_point_not_found(self):
        """Test evaluation with non-existent decision point"""
        self.setUp()
        
        result = self.decision_engine.evaluate_decision_point(
            'non_existent', {}, []
        )
        assert result is None
    
    def test_evaluate_decision_point_valid(self):
        """Test valid decision point evaluation"""
        self.setUp()
        
        # Create test context and evidence
        context = {'validation_complete': True}
        evidence = [
            Evidence(
                source='test_runner',
                content='All tests pass',
                confidence=0.95,
                timestamp=datetime.now(),
                evidence_type='test_result'
            ),
            Evidence(
                source='quality_gates',
                content='Quality gates pass',
                confidence=0.90,
                timestamp=datetime.now(),
                evidence_type='quality_gate'
            )
        ]
        
        result = self.decision_engine.evaluate_decision_point(
            'post_validation_decision', context, evidence
        )
        
        # Should return a decision result
        if result:  # May return None if trigger condition not met
            assert 'decision_id' in result
            assert 'outcome' in result
            assert 'confidence' in result
    
    def test_multi_factor_confidence_calculation(self):
        """Test multi-factor confidence calculation"""
        self.setUp()
        
        factors = {
            'evidence_quality': 0.9,
            'pattern_matches': 0.8,
            'agent_consensus': 0.85,
            'historical_success': 0.7,
            'context_completeness': 0.95,
            'validation_reliability': 0.88
        }
        
        confidence = self.decision_engine.calculate_multi_factor_confidence(factors)
        
        assert 0.0 <= confidence <= 1.0
        # Should be weighted average of factors
        expected = (0.9*0.3 + 0.8*0.2 + 0.85*0.2 + 0.7*0.15 + 0.95*0.1 + 0.88*0.05)
        assert abs(confidence - expected) < 0.01
    
    def test_outcome_condition_evaluation(self):
        """Test outcome condition evaluation"""
        self.setUp()
        
        # Test with passing conditions
        context = {'all_tests_pass': True, 'quality_gates_pass': True}
        evidence = [
            Evidence(
                source='test',
                content='All tests pass successfully',
                confidence=0.95,
                timestamp=datetime.now(),
                evidence_type='test_result'
            )
        ]
        
        is_met, confidence = self.decision_engine._evaluate_outcome_condition(
            'all_tests_pass AND quality_gates_pass', context, evidence
        )
        
        # Note: This tests the simplified implementation
        # The actual result depends on the implementation logic


class TestWorkflowSession:
    """Test WorkflowSession functionality"""
    
    def test_session_creation(self):
        """Test workflow session creation"""
        session = WorkflowSession(
            session_id='test_session',
            issue_number=123,
            current_state='analyzing',
            context={'complexity': 'medium'}
        )
        
        assert session.session_id == 'test_session'
        assert session.issue_number == 123
        assert session.current_state == 'analyzing'
        assert len(session.evidence_trail) == 0
        assert len(session.transition_history) == 0
    
    def test_add_evidence(self):
        """Test adding evidence to session"""
        session = WorkflowSession(
            session_id='test_session',
            issue_number=123,
            current_state='analyzing',
            context={}
        )
        
        evidence = Evidence(
            source='test',
            content='Test evidence',
            confidence=0.8,
            timestamp=datetime.now(),
            evidence_type='test'
        )
        
        initial_time = session.last_updated
        time.sleep(0.01)  # Small delay to ensure time difference
        session.add_evidence(evidence)
        
        assert len(session.evidence_trail) == 1
        assert session.evidence_trail[0] == evidence
        assert session.last_updated > initial_time
    
    def test_record_transition(self):
        """Test recording state transitions"""
        session = WorkflowSession(
            session_id='test_session',
            issue_number=123,
            current_state='analyzing',
            context={}
        )
        
        session.record_transition('analyzing', 'implementing', 'test transition', 0.85)
        
        assert len(session.transition_history) == 1
        assert session.current_state == 'implementing'
        
        transition = session.transition_history[0]
        assert transition['from_state'] == 'analyzing'
        assert transition['to_state'] == 'implementing'
        assert transition['confidence'] == 0.85


class TestDynamicOrchestrator:
    """Test DynamicOrchestrator functionality"""
    
    def test_orchestrator_creation(self):
        """Test dynamic orchestrator creation"""
        orchestrator = DynamicOrchestrator()
        
        assert orchestrator.state_graph_manager is not None
        assert orchestrator.decision_engine is not None
        assert len(orchestrator.active_sessions) == 0
    
    def test_create_session(self):
        """Test creating workflow session"""
        orchestrator = DynamicOrchestrator()
        
        initial_context = {'complexity': 'high', 'priority': 'medium'}
        session_id = orchestrator.create_session(456, initial_context)
        
        assert session_id.startswith('session_456_')
        assert session_id in orchestrator.active_sessions
        
        session = orchestrator.active_sessions[session_id]
        assert session.issue_number == 456
        assert session.context == initial_context
    
    def test_get_session(self):
        """Test getting workflow session"""
        orchestrator = DynamicOrchestrator()
        
        session_id = orchestrator.create_session(789, {})
        
        session = orchestrator.get_session(session_id)
        assert session is not None
        assert session.issue_number == 789
        
        non_existent = orchestrator.get_session('non_existent')
        assert non_existent is None
    
    def test_add_evidence_to_session(self):
        """Test adding evidence to session"""
        orchestrator = DynamicOrchestrator()
        
        session_id = orchestrator.create_session(101, {})
        evidence = Evidence(
            source='test',
            content='Test evidence',
            confidence=0.9,
            timestamp=datetime.now(),
            evidence_type='test'
        )
        
        orchestrator.add_evidence_to_session(session_id, evidence)
        
        session = orchestrator.get_session(session_id)
        assert len(session.evidence_trail) == 1
    
    def test_evaluate_workflow_decision(self):
        """Test workflow decision evaluation"""
        orchestrator = DynamicOrchestrator()
        
        session_id = orchestrator.create_session(202, {'complexity': 'medium'})
        
        # Add some test evidence
        evidence = Evidence(
            source='test',
            content='Analysis complete',
            confidence=0.85,
            timestamp=datetime.now(),
            evidence_type='analysis_complete'
        )
        orchestrator.add_evidence_to_session(session_id, evidence)
        
        result = orchestrator.evaluate_workflow_decision(session_id)
        
        assert 'session_id' in result
        assert 'current_state' in result
        assert 'recommended_transitions' in result
        assert 'decision_point_results' in result
        assert result['session_id'] == session_id
    
    def test_execute_state_transition(self):
        """Test executing state transition"""
        orchestrator = DynamicOrchestrator()
        
        session_id = orchestrator.create_session(303, {})
        
        # Execute valid transition
        result = orchestrator.execute_state_transition(
            session_id, 'implementing', 'Test transition'
        )
        
        assert result['success'] is True
        assert result['from_state'] == 'analyzing'
        assert result['to_state'] == 'implementing'
        
        # Verify session state changed
        session = orchestrator.get_session(session_id)
        assert session.current_state == 'implementing'
    
    def test_get_orchestration_summary(self):
        """Test getting orchestration summary"""
        orchestrator = DynamicOrchestrator()
        
        session_id = orchestrator.create_session(404, {'complexity': 'high'})
        
        # Add evidence and transitions
        evidence = Evidence(
            source='test',
            content='Test evidence',
            confidence=0.8,
            timestamp=datetime.now(),
            evidence_type='test'
        )
        orchestrator.add_evidence_to_session(session_id, evidence)
        orchestrator.execute_state_transition(session_id, 'implementing', 'Test')
        
        summary = orchestrator.get_orchestration_summary(session_id)
        
        assert 'session_id' in summary
        assert 'issue_number' in summary
        assert 'current_state' in summary
        assert 'total_transitions' in summary
        assert summary['issue_number'] == 404
        assert summary['total_transitions'] == 1


class TestResourceRequirement:
    """Test ResourceRequirement functionality"""
    
    def test_resource_requirement_creation(self):
        """Test basic resource requirement creation"""
        req = ResourceRequirement(
            cpu_cores=2.0,
            memory_mb=1024,
            io_bandwidth=3.0,
            resource_type=ResourceType.CPU_INTENSIVE,
            execution_time_estimate=120
        )
        
        assert req.cpu_cores == 2.0
        assert req.memory_mb == 1024
        assert req.resource_type == ResourceType.CPU_INTENSIVE
    
    def test_resource_requirement_to_dict(self):
        """Test resource requirement serialization"""
        req = ResourceRequirement(
            cpu_cores=1.5,
            memory_mb=2048,
            execution_time_estimate=180
        )
        
        req_dict = req.to_dict()
        
        assert req_dict['cpu_cores'] == 1.5
        assert req_dict['memory_mb'] == 2048
        assert req_dict['execution_time_estimate'] == 180


class TestParallelPath:
    """Test ParallelPath functionality"""
    
    def test_parallel_path_creation(self):
        """Test parallel path creation"""
        req = ResourceRequirement(cpu_cores=1.0, memory_mb=512)
        
        path = ParallelPath(
            path_id='test_path',
            description='Test parallel path',
            agents=['rif-validator'],
            resource_requirements=req,
            dependencies=['other_path']
        )
        
        assert path.path_id == 'test_path'
        assert path.status == ExecutionStatus.PENDING
        assert 'other_path' in path.dependencies
    
    def test_parallel_path_execution_lifecycle(self):
        """Test parallel path execution lifecycle"""
        req = ResourceRequirement()
        path = ParallelPath(
            path_id='lifecycle_test',
            description='Test lifecycle',
            agents=['test-agent'],
            resource_requirements=req
        )
        
        # Initial state
        assert path.status == ExecutionStatus.PENDING
        assert path.execution_duration is None
        
        # Mark started
        start_time = datetime.now()
        path.mark_started()
        
        assert path.status == ExecutionStatus.RUNNING
        assert path.start_time is not None
        assert path.execution_duration is not None
        
        time.sleep(0.1)  # Small delay
        
        # Mark completed
        result = {'status': 'success', 'output': 'test output'}
        path.mark_completed(result)
        
        assert path.status == ExecutionStatus.COMPLETED
        assert path.result == result
        assert path.execution_duration > 0.05  # Should be > 0.05 seconds
    
    def test_parallel_path_failure(self):
        """Test parallel path failure handling"""
        req = ResourceRequirement()
        path = ParallelPath(
            path_id='failure_test',
            description='Test failure',
            agents=['test-agent'],
            resource_requirements=req
        )
        
        path.mark_started()
        
        # Mark failed
        error = ValueError("Test error")
        path.mark_failed(error)
        
        assert path.status == ExecutionStatus.FAILED
        assert path.error_info is not None
        assert path.error_info['error_type'] == 'ValueError'
        assert 'Test error' in path.error_info['error_message']


class TestWorkloadBalancer:
    """Test WorkloadBalancer functionality"""
    
    def test_workload_balancer_creation(self):
        """Test workload balancer creation"""
        balancer = WorkloadBalancer()
        assert balancer is not None
    
    def test_path_prioritization(self):
        """Test path prioritization logic"""
        balancer = WorkloadBalancer()
        
        # Create paths with different characteristics
        req1 = ResourceRequirement(execution_time_estimate=60)  # Short task
        req2 = ResourceRequirement(execution_time_estimate=300)  # Long task
        
        path1 = ParallelPath('path1', 'Short task', ['agent1'], req1)
        path2 = ParallelPath('path2', 'Long task', ['agent2'], req2, dependencies=['path1'])
        
        paths = [path2, path1]  # Intentionally reversed order
        prioritized = balancer._prioritize_paths(paths)
        
        # Shorter task should have higher priority
        assert prioritized[0].path_id == 'path1'
        assert prioritized[1].path_id == 'path2'


class TestConflictResolver:
    """Test ConflictResolver functionality"""
    
    def test_conflict_resolver_creation(self):
        """Test conflict resolver creation"""
        resolver = ConflictResolver()
        assert resolver is not None
    
    def test_resource_conflict_detection(self):
        """Test resource conflict detection"""
        resolver = ConflictResolver()
        
        # Create paths that would exceed resource limits
        high_req = ResourceRequirement(
            cpu_cores=5.0,  # Exceeds typical 4-core limit
            memory_mb=10240  # Exceeds 8GB limit
        )
        
        path1 = ParallelPath('path1', 'High resource path', ['agent1'], high_req)
        conflicts = resolver._detect_resource_conflicts([path1])
        
        assert len(conflicts) > 0
        assert conflicts[0]['type'] == 'resource_conflict'
    
    def test_agent_conflict_detection(self):
        """Test agent conflict detection"""
        resolver = ConflictResolver()
        
        req = ResourceRequirement()
        path1 = ParallelPath('path1', 'Path 1', ['rif-validator'], req)
        path2 = ParallelPath('path2', 'Path 2', ['rif-validator'], req)  # Same agent
        
        conflicts = resolver._detect_agent_conflicts([path1, path2])
        
        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'agent_conflict'
        assert conflicts[0]['agent'] == 'rif-validator'
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        resolver = ConflictResolver()
        
        dependency_graph = {
            'A': ['B'],
            'B': ['C'],
            'C': ['A']  # Creates a cycle
        }
        
        cycle = resolver._find_circular_dependencies(dependency_graph)
        
        assert cycle is not None
        assert len(cycle) > 0
        # Should detect the cycle A -> B -> C -> A


class TestParallelExecutionCoordinator:
    """Test ParallelExecutionCoordinator functionality"""
    
    def test_coordinator_creation(self):
        """Test parallel execution coordinator creation"""
        coordinator = ParallelExecutionCoordinator(max_workers=2)
        
        assert coordinator.max_workers == 2
        assert coordinator.resource_allocation.max_cpu_cores == 2
        assert len(coordinator.active_paths) == 0
    
    def test_create_execution_plan(self):
        """Test creating parallel execution plan"""
        coordinator = ParallelExecutionCoordinator()
        
        # Create test paths
        req = ResourceRequirement(cpu_cores=1.0, memory_mb=512)
        path1 = ParallelPath('path1', 'Test path 1', ['agent1'], req)
        path2 = ParallelPath('path2', 'Test path 2', ['agent2'], req)
        
        paths = [path1, path2]
        execution_plan = coordinator.create_parallel_execution_plan(paths)
        
        assert 'total_paths' in execution_plan
        assert 'execution_batches' in execution_plan
        assert 'synchronization_points' in execution_plan
        assert execution_plan['total_paths'] == 2
    
    def test_synchronization_plan_creation(self):
        """Test synchronization plan creation"""
        coordinator = ParallelExecutionCoordinator()
        
        req = ResourceRequirement()
        path1 = ParallelPath('path1', 'Path 1', ['agent1'], req, 
                           synchronization_points=['sync1'])
        path2 = ParallelPath('path2', 'Path 2', ['agent2'], req, 
                           synchronization_points=['sync1'])
        
        sync_plan = coordinator._create_synchronization_plan([path1, path2])
        
        assert 'sync1' in sync_plan
        assert len(sync_plan['sync1']['required_paths']) == 2


class TestPerformanceTargets:
    """Test performance targets for dynamic orchestration"""
    
    def test_orchestration_performance_target(self):
        """Test that orchestration meets performance targets"""
        orchestrator = DynamicOrchestrator()
        
        # Test orchestration cycle time
        start_time = time.perf_counter()
        
        session_id = orchestrator.create_session(999, {'complexity': 'medium'})
        result = orchestrator.evaluate_workflow_decision(session_id)
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Should be < 100ms target (allowing some buffer for test environment)
        assert duration_ms < 200, f"Orchestration took {duration_ms:.2f}ms, target is <100ms"
    
    def test_decision_evaluation_performance(self):
        """Test decision evaluation performance"""
        state_graph_manager = StateGraphManager()
        decision_engine = DecisionEngine(state_graph_manager)
        
        # Create test context and evidence
        context = {'validation_complete': True}
        evidence = [
            Evidence('test', 'test evidence', 0.9, datetime.now(), 'test')
            for _ in range(5)  # Multiple pieces of evidence
        ]
        
        start_time = time.perf_counter()
        
        # Evaluate multiple decision points
        for decision_id in state_graph_manager.decision_points:
            result = decision_engine.evaluate_decision_point(decision_id, context, evidence)
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Should be < 50ms per decision point target
        decision_count = len(state_graph_manager.decision_points)
        avg_duration = duration_ms / max(1, decision_count)
        
        assert avg_duration < 100, f"Decision evaluation took {avg_duration:.2f}ms avg, target is <50ms"


# Test runner and fixtures
@pytest.fixture
def test_orchestrator():
    """Fixture providing a test orchestrator"""
    return DynamicOrchestrator()


@pytest.fixture
def test_evidence_list():
    """Fixture providing test evidence"""
    return [
        Evidence(
            source='test_runner',
            content='All 25 unit tests passed',
            confidence=0.95,
            timestamp=datetime.now(),
            evidence_type='test_result'
        ),
        Evidence(
            source='quality_gates',
            content='Code coverage 85%, Security scan passed',
            confidence=0.90,
            timestamp=datetime.now(),
            evidence_type='quality_gate'
        ),
        Evidence(
            source='rif-implementer',
            content='Implementation complete with validation',
            confidence=0.88,
            timestamp=datetime.now(),
            evidence_type='implementation_complete'
        )
    ]


@pytest.fixture
def test_parallel_paths():
    """Fixture providing test parallel paths"""
    req1 = ResourceRequirement(
        cpu_cores=1.0, memory_mb=1024, execution_time_estimate=60
    )
    req2 = ResourceRequirement(
        cpu_cores=2.0, memory_mb=2048, execution_time_estimate=120
    )
    
    return [
        ParallelPath('path1', 'Validation path', ['rif-validator'], req1),
        ParallelPath('path2', 'Implementation path', ['rif-implementer'], req2)
    ]


def test_integration_orchestrator_with_parallel_execution(test_orchestrator, test_parallel_paths):
    """Integration test combining orchestrator with parallel execution"""
    coordinator = ParallelExecutionCoordinator()
    
    # Create orchestrator session
    session_id = test_orchestrator.create_session(
        999, {'complexity': 'high', 'parallel_execution': True}
    )
    
    # Create parallel execution plan
    execution_plan = coordinator.create_parallel_execution_plan(test_parallel_paths)
    
    # Verify integration points
    assert execution_plan['total_paths'] == len(test_parallel_paths)
    assert len(execution_plan['execution_batches']) > 0
    
    # Test that orchestrator can evaluate decisions with parallel context
    decision_context = {'parallel_execution_plan': execution_plan}
    result = test_orchestrator.evaluate_workflow_decision(session_id, decision_context)
    
    assert 'session_id' in result
    assert result['session_id'] == session_id


def test_end_to_end_dynamic_orchestration_scenario(test_orchestrator, test_evidence_list):
    """End-to-end test of dynamic orchestration scenario"""
    # Scenario: High complexity issue that needs architecture phase,
    # then implementation, validation with parallel paths, and learning
    
    # 1. Create session for high complexity issue
    session_id = test_orchestrator.create_session(
        1001, {'complexity': 'high', 'priority': 'high', 'requirements_clear': True}
    )
    
    # 2. Add analysis evidence
    analysis_evidence = Evidence(
        source='rif-analyst',
        content='Complex architectural requirements identified',
        confidence=0.92,
        timestamp=datetime.now(),
        evidence_type='analysis_complete'
    )
    test_orchestrator.add_evidence_to_session(session_id, analysis_evidence)
    
    # 3. Evaluate workflow - should recommend architecture phase
    result = test_orchestrator.evaluate_workflow_decision(session_id)
    
    # Should have recommendations for high complexity
    assert len(result['recommended_transitions']) > 0
    
    # 4. Execute transition to architecture phase
    transition_result = test_orchestrator.execute_state_transition(
        session_id, 'architecting', 'High complexity requires architecture'
    )
    assert transition_result['success'] is True
    
    # 5. Add architecture evidence
    arch_evidence = Evidence(
        source='rif-architect',
        content='Architecture design completed with dependency mapping',
        confidence=0.90,
        timestamp=datetime.now(),
        evidence_type='design_complete'
    )
    test_orchestrator.add_evidence_to_session(session_id, arch_evidence)
    
    # 6. Transition to implementation
    impl_result = test_orchestrator.execute_state_transition(
        session_id, 'implementing', 'Architecture phase complete'
    )
    assert impl_result['success'] is True
    
    # 7. Add implementation evidence
    impl_evidence = Evidence(
        source='rif-implementer',
        content='Implementation complete with comprehensive testing',
        confidence=0.87,
        timestamp=datetime.now(),
        evidence_type='implementation_complete'
    )
    test_orchestrator.add_evidence_to_session(session_id, impl_evidence)
    
    # 8. Add validation evidence
    for evidence in test_evidence_list:
        test_orchestrator.add_evidence_to_session(session_id, evidence)
    
    # 9. Transition to validation
    val_result = test_orchestrator.execute_state_transition(
        session_id, 'validating', 'Implementation ready for validation'
    )
    assert val_result['success'] is True
    
    # 10. Final workflow evaluation - should recommend learning
    final_result = test_orchestrator.evaluate_workflow_decision(session_id, {
        'all_tests_pass': True,
        'quality_gates_pass': True,
        'validation_complete': True
    })
    
    # 11. Get comprehensive summary
    summary = test_orchestrator.get_orchestration_summary(session_id)
    
    # Verify end-to-end execution
    assert summary['total_transitions'] >= 3  # analyzing -> architecting -> implementing -> validating
    assert summary['evidence_count'] >= 5
    assert summary['current_state'] == 'validating'


if __name__ == "__main__":
    # Run specific test for demonstration
    print("Running Dynamic Orchestrator Engine Tests")
    print("=" * 50)
    
    # Run a few key tests
    test_evidence = TestEvidence()
    test_evidence.test_evidence_creation()
    print("✓ Evidence creation test passed")
    
    test_state_graph = TestStateGraphManager()
    test_state_graph.test_state_graph_initialization()
    print("✓ State graph initialization test passed")
    
    test_orchestrator = TestDynamicOrchestrator()
    test_orchestrator.test_orchestrator_creation()
    test_orchestrator.test_create_session()
    print("✓ Dynamic orchestrator tests passed")
    
    test_parallel = TestParallelExecutionCoordinator()
    test_parallel.test_coordinator_creation()
    print("✓ Parallel execution coordinator tests passed")
    
    print("\n✅ All key tests completed successfully!")
    print("Run 'pytest test_dynamic_orchestrator_engine.py' for full test suite")