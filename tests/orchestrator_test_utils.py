#!/usr/bin/env python3
"""
Orchestrator Test Utilities
Supporting utilities for orchestrator testing including mocks, fixtures, and helpers.
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Add the commands directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'claude' / 'commands'))


class MockGitHubIssue:
    """Mock GitHub issue for testing."""
    
    def __init__(self, issue_id: int, complexity: str = 'medium', **kwargs):
        """Initialize mock GitHub issue."""
        self.issue_id = issue_id
        self.complexity = complexity
        self.title = kwargs.get('title', f'Test Issue #{issue_id}')
        self.body = kwargs.get('body', f'Test issue body for complexity {complexity}')
        self.labels = kwargs.get('labels', [f'complexity:{complexity}'])
        self.state = kwargs.get('state', 'open')
        self.assignees = kwargs.get('assignees', [])
        self.created_at = kwargs.get('created_at', datetime.now().isoformat())
        self.updated_at = kwargs.get('updated_at', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'issue_id': self.issue_id,
            'title': self.title,
            'body': self.body,
            'labels': self.labels,
            'state': self.state,
            'assignees': self.assignees,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'complexity': self.complexity
        }


class MockValidationResult:
    """Mock validation result for testing."""
    
    def __init__(self, success: bool, coverage: float = 85.0, **kwargs):
        """Initialize mock validation result."""
        self.success = success
        self.coverage = coverage
        self.test_count = kwargs.get('test_count', 50)
        self.failed_tests = kwargs.get('failed_tests', [] if success else ['test_example'])
        self.duration_seconds = kwargs.get('duration_seconds', 2.5)
        self.details = kwargs.get('details', 'All tests passed' if success else 'Some tests failed')
        self.quality_score = kwargs.get('quality_score', 0.9 if success else 0.6)


class MockAgentResponse:
    """Mock agent response for testing."""
    
    def __init__(self, agent_name: str, success: bool = True, **kwargs):
        """Initialize mock agent response."""
        self.agent_name = agent_name
        self.success = success
        self.execution_time_ms = kwargs.get('execution_time_ms', random.randint(500, 3000))
        self.confidence_score = kwargs.get('confidence_score', random.uniform(0.7, 0.95) if success else random.uniform(0.3, 0.6))
        self.output = kwargs.get('output', f'{agent_name} completed successfully' if success else f'{agent_name} encountered errors')
        self.metadata = kwargs.get('metadata', {})
        self.errors = kwargs.get('errors', [] if success else ['Example error'])


class OrchestrationTestFixture:
    """Test fixture for orchestration testing."""
    
    def __init__(self, scenario: str = 'standard'):
        """Initialize test fixture."""
        self.scenario = scenario
        self.issues = self._create_test_issues()
        self.workflows = self._create_test_workflows()
        self.contexts = self._create_test_contexts()
        self.expected_results = self._create_expected_results()
    
    def _create_test_issues(self) -> Dict[str, MockGitHubIssue]:
        """Create test issues for different scenarios."""
        issues = {
            'simple': MockGitHubIssue(1001, 'low', title='Simple bug fix'),
            'feature': MockGitHubIssue(1002, 'medium', title='New feature implementation'),
            'architecture': MockGitHubIssue(1003, 'high', title='System architecture changes'),
            'migration': MockGitHubIssue(1004, 'very_high', title='Database migration project'),
            'security': MockGitHubIssue(1005, 'high', title='Security vulnerability fix', 
                                       labels=['complexity:high', 'security-critical'])
        }
        return issues
    
    def _create_test_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Create test workflow graphs for different scenarios."""
        workflows = {
            'minimal': {
                'states': {
                    'initialized': {'transitions': ['completed'], 'agents': ['RIF-Implementer'], 'requirements': []},
                    'completed': {'transitions': [], 'agents': [], 'requirements': []}
                }
            },
            'standard': {
                'states': {
                    'initialized': {'transitions': ['analyzing'], 'agents': ['RIF-Analyst'], 'requirements': []},
                    'analyzing': {'transitions': ['implementing'], 'agents': ['RIF-Analyst'], 'requirements': ['github_issues']},
                    'implementing': {'transitions': ['validating'], 'agents': ['RIF-Implementer'], 'requirements': ['requirements']},
                    'validating': {'transitions': ['completed', 'implementing'], 'agents': ['RIF-Validator'], 'requirements': ['implementation']},
                    'completed': {'transitions': [], 'agents': [], 'requirements': []}
                }
            },
            'complex': {
                'states': {
                    'initialized': {'transitions': ['analyzing'], 'agents': ['RIF-Analyst'], 'requirements': []},
                    'analyzing': {'transitions': ['planning'], 'agents': ['RIF-Analyst'], 'requirements': ['github_issues']},
                    'planning': {'transitions': ['architecting'], 'agents': ['RIF-Planner'], 'requirements': ['requirements']},
                    'architecting': {'transitions': ['implementing'], 'agents': ['RIF-Architect'], 'requirements': ['plan']},
                    'implementing': {'transitions': ['validating'], 'agents': ['RIF-Implementer'], 'requirements': ['architecture']},
                    'validating': {'transitions': ['learning', 'implementing'], 'agents': ['RIF-Validator'], 'requirements': ['implementation']},
                    'learning': {'transitions': ['completed'], 'agents': ['RIF-Learner'], 'requirements': ['validation']},
                    'completed': {'transitions': [], 'agents': [], 'requirements': []}
                }
            }
        }
        return workflows
    
    def _create_test_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Create test contexts for different scenarios."""
        contexts = {
            'simple': {
                'github_issues': [self.issues['simple'].issue_id],
                'complexity': 'low',
                'workflow_type': 'bug_fix',
                'priority': 1
            },
            'feature_development': {
                'github_issues': [self.issues['feature'].issue_id],
                'complexity': 'medium',
                'workflow_type': 'feature_development',
                'priority': 2,
                'required_skills': ['requirements_analysis', 'code_implementation']
            },
            'architecture_change': {
                'github_issues': [self.issues['architecture'].issue_id],
                'complexity': 'high',
                'workflow_type': 'system_design',
                'priority': 3,
                'required_skills': ['system_design', 'architecture_planning'],
                'security_critical': False
            },
            'security_critical': {
                'github_issues': [self.issues['security'].issue_id],
                'complexity': 'high',
                'workflow_type': 'security_fix',
                'priority': 4,
                'security_critical': True,
                'required_skills': ['security_analysis', 'vulnerability_assessment']
            }
        }
        return contexts
    
    def _create_expected_results(self) -> Dict[str, Dict[str, Any]]:
        """Create expected results for test scenarios."""
        return {
            'simple': {
                'expected_states': ['initialized', 'completed'],
                'expected_agents': ['RIF-Implementer'],
                'max_iterations': 2,
                'min_confidence': 0.7
            },
            'standard': {
                'expected_states': ['initialized', 'analyzing', 'implementing', 'validating', 'completed'],
                'expected_agents': ['RIF-Analyst', 'RIF-Implementer', 'RIF-Validator'],
                'max_iterations': 5,
                'min_confidence': 0.8
            },
            'complex': {
                'expected_states': ['initialized', 'analyzing', 'planning', 'architecting', 'implementing', 'validating', 'learning', 'completed'],
                'expected_agents': ['RIF-Analyst', 'RIF-Planner', 'RIF-Architect', 'RIF-Implementer', 'RIF-Validator', 'RIF-Learner'],
                'max_iterations': 8,
                'min_confidence': 0.75
            }
        }
    
    def get_test_data(self, test_type: str) -> Dict[str, Any]:
        """Get test data for specific test type."""
        return {
            'issue': self.issues.get(test_type, self.issues['simple']),
            'workflow': self.workflows.get(test_type, self.workflows['standard']),
            'context': self.contexts.get(test_type, self.contexts['simple']),
            'expected': self.expected_results.get(test_type, self.expected_results['simple'])
        }


class PerformanceTestHarness:
    """Performance testing harness for orchestrator components."""
    
    def __init__(self):
        """Initialize performance test harness."""
        self.metrics = []
        self.baseline_metrics = self._load_baseline_metrics()
    
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline performance metrics."""
        # These would typically be loaded from a file or database
        return {
            'low_complexity_max_ms': 1000,
            'medium_complexity_max_ms': 5000,
            'high_complexity_max_ms': 15000,
            'very_high_complexity_max_ms': 30000,
            'memory_usage_max_mb': 100,
            'concurrent_workflows_min': 5
        }
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time of a function."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'duration_ms': (end_time - start_time) * 1000,
            'memory_usage_mb': end_memory - start_memory,
            'success': success,
            'error': error,
            'result': result
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def run_load_test(self, test_func: Callable, concurrent_count: int = 5, 
                     iterations: int = 10) -> Dict[str, Any]:
        """Run load test with concurrent execution."""
        import concurrent.futures
        
        def run_single_test(test_id):
            return self.measure_execution_time(test_func, test_id)
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(run_single_test, i) for i in range(iterations)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if successful_results:
            avg_duration = sum(r['duration_ms'] for r in successful_results) / len(successful_results)
            max_duration = max(r['duration_ms'] for r in successful_results)
            min_duration = min(r['duration_ms'] for r in successful_results)
            avg_memory = sum(r['memory_usage_mb'] for r in successful_results) / len(successful_results)
        else:
            avg_duration = max_duration = min_duration = avg_memory = 0
        
        return {
            'total_duration_s': end_time - start_time,
            'concurrent_count': concurrent_count,
            'total_iterations': iterations,
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'success_rate': len(successful_results) / iterations,
            'avg_duration_ms': avg_duration,
            'max_duration_ms': max_duration,
            'min_duration_ms': min_duration,
            'avg_memory_usage_mb': avg_memory,
            'throughput_per_second': iterations / (end_time - start_time)
        }
    
    def validate_performance(self, complexity: str, duration_ms: float) -> bool:
        """Validate performance against baseline metrics."""
        baseline_key = f'{complexity}_complexity_max_ms'
        if baseline_key in self.baseline_metrics:
            return duration_ms <= self.baseline_metrics[baseline_key]
        return True  # If no baseline, assume valid
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {'error': 'No performance data collected'}
        
        successful_metrics = [m for m in self.metrics if m['success']]
        
        if not successful_metrics:
            return {'error': 'No successful performance measurements'}
        
        return {
            'total_measurements': len(self.metrics),
            'successful_measurements': len(successful_metrics),
            'success_rate': len(successful_metrics) / len(self.metrics),
            'avg_duration_ms': sum(m['duration_ms'] for m in successful_metrics) / len(successful_metrics),
            'max_duration_ms': max(m['duration_ms'] for m in successful_metrics),
            'min_duration_ms': min(m['duration_ms'] for m in successful_metrics),
            'avg_memory_usage_mb': sum(m['memory_usage_mb'] for m in successful_metrics) / len(successful_metrics),
            'baseline_compliance': self._check_baseline_compliance(),
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _check_baseline_compliance(self) -> Dict[str, bool]:
        """Check compliance with baseline performance metrics."""
        compliance = {}
        
        if self.metrics:
            successful_metrics = [m for m in self.metrics if m['success']]
            if successful_metrics:
                max_duration = max(m['duration_ms'] for m in successful_metrics)
                avg_memory = sum(m['memory_usage_mb'] for m in successful_metrics) / len(successful_metrics)
                
                compliance['duration'] = max_duration <= self.baseline_metrics.get('high_complexity_max_ms', 15000)
                compliance['memory'] = avg_memory <= self.baseline_metrics.get('memory_usage_max_mb', 100)
        
        return compliance
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if self.metrics:
            successful_metrics = [m for m in self.metrics if m['success']]
            if successful_metrics:
                avg_duration = sum(m['duration_ms'] for m in successful_metrics) / len(successful_metrics)
                avg_memory = sum(m['memory_usage_mb'] for m in successful_metrics) / len(successful_metrics)
                
                if avg_duration > 10000:  # 10 seconds
                    recommendations.append("Consider optimizing workflow state transitions - average duration is high")
                
                if avg_memory > 50:  # 50 MB
                    recommendations.append("Monitor memory usage - consider implementing memory cleanup")
                
                failed_count = len(self.metrics) - len(successful_metrics)
                if failed_count > 0:
                    recommendations.append(f"Address {failed_count} failed test executions")
        
        return recommendations


class ConsensusTestSimulator:
    """Simulator for consensus system testing in orchestration."""
    
    def __init__(self, num_agents: int = 3):
        """Initialize consensus test simulator."""
        self.num_agents = num_agents
        self.agents = [f'Agent_{i}' for i in range(num_agents)]
        self.decisions = []
        self.consensus_threshold = 0.6  # 60% agreement required
    
    def simulate_agent_vote(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent voting on a decision."""
        votes = {}
        
        for agent in self.agents:
            # Simulate agent decision-making with some randomness
            base_confidence = random.uniform(0.5, 0.95)
            
            # Adjust based on context
            if context.get('complexity') == 'low':
                base_confidence += 0.1
            elif context.get('complexity') == 'high':
                base_confidence -= 0.1
            
            if context.get('security_critical'):
                base_confidence -= 0.05  # More cautious for security
            
            # Clamp confidence to valid range
            confidence = max(0.1, min(0.95, base_confidence))
            
            # Decision based on confidence threshold
            decision = 'approve' if confidence >= 0.7 else 'reject'
            
            votes[agent] = {
                'decision': decision,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        
        return votes
    
    def calculate_consensus(self, votes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from agent votes."""
        approve_votes = sum(1 for vote in votes.values() if vote['decision'] == 'approve')
        total_votes = len(votes)
        
        if total_votes == 0:
            return {'consensus': False, 'reason': 'No votes received'}
        
        approval_rate = approve_votes / total_votes
        has_consensus = approval_rate >= self.consensus_threshold
        
        avg_confidence = sum(vote['confidence'] for vote in votes.values()) / total_votes
        
        return {
            'consensus': has_consensus,
            'approval_rate': approval_rate,
            'avg_confidence': avg_confidence,
            'approve_votes': approve_votes,
            'total_votes': total_votes,
            'threshold': self.consensus_threshold
        }
    
    def simulate_consensus_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate complete consensus decision process."""
        decision_id = f"decision_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Simulate voting
        votes = self.simulate_agent_vote(decision_id, context)
        
        # Calculate consensus
        consensus_result = self.calculate_consensus(votes)
        
        # Record decision
        decision_record = {
            'decision_id': decision_id,
            'context': context,
            'votes': votes,
            'consensus': consensus_result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.decisions.append(decision_record)
        
        return decision_record


# Utility functions
def create_test_environment(scenario: str = 'standard') -> Dict[str, Any]:
    """Create a complete test environment for orchestrator testing."""
    fixture = OrchestrationTestFixture(scenario)
    performance_harness = PerformanceTestHarness()
    consensus_simulator = ConsensusTestSimulator()
    
    return {
        'fixture': fixture,
        'performance_harness': performance_harness,
        'consensus_simulator': consensus_simulator,
        'scenario': scenario
    }


def validate_test_results(results: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, bool]:
    """Validate test results against expected outcomes."""
    validation = {}
    
    # Check success status
    validation['success'] = results.get('success', False) == expected.get('success', True)
    
    # Check final state
    validation['final_state'] = results.get('final_state') == expected.get('final_state', 'completed')
    
    # Check iteration count
    max_iterations = expected.get('max_iterations', 10)
    validation['iteration_count'] = results.get('iterations', 0) <= max_iterations
    
    # Check confidence levels
    min_confidence = expected.get('min_confidence', 0.7)
    if 'history' in results and results['history']:
        avg_confidence = sum(step.get('confidence_score', 0) for step in results['history']) / len(results['history'])
        validation['confidence_level'] = avg_confidence >= min_confidence
    else:
        validation['confidence_level'] = True  # No history to validate
    
    # Check agent involvement
    expected_agents = expected.get('expected_agents', [])
    if expected_agents:
        agents_used = set()
        for step in results.get('history', []):
            agents_used.update(step.get('agents_selected', []))
        validation['agents_involved'] = all(agent in agents_used for agent in expected_agents)
    else:
        validation['agents_involved'] = True
    
    return validation


def generate_test_report(test_results: List[Dict[str, Any]], 
                        performance_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.get('success', False))
    
    report = {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'timestamp': datetime.now().isoformat()
        },
        'test_details': test_results,
        'performance': performance_results,
        'recommendations': []
    }
    
    # Add recommendations based on results
    if report['summary']['success_rate'] < 0.9:
        report['recommendations'].append("Improve test success rate - some orchestrator components may need fixes")
    
    if performance_results and performance_results.get('avg_duration_ms', 0) > 10000:
        report['recommendations'].append("Optimize orchestrator performance - execution time is above recommended threshold")
    
    if total_tests < 10:
        report['recommendations'].append("Expand test coverage - add more test scenarios")
    
    return report