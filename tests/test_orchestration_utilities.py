#!/usr/bin/env python3
"""
Test suite for RIF Orchestration Utilities - Pattern-Compliant Implementation

These tests verify that the orchestration utilities correctly support Claude Code
as the orchestrator, following the proper RIF pattern.
"""

import pytest
import json
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import utilities (should be available in sys.path)
import sys
sys.path.append('/Users/cal/DEV/RIF/claude/commands')
from orchestration_utilities import (
    ContextAnalyzer, StateValidator, OrchestrationHelper,
    GitHubStateManager, IssueContext
)


class TestIssueContext:
    """Test IssueContext data class functionality"""
    
    def test_issue_context_creation(self):
        """Test basic IssueContext creation"""
        context = IssueContext(
            number=42,
            title="Test Issue",
            body="Test body",
            labels=['state:new', 'complexity:medium'],
            state='open',
            complexity='medium',
            priority='medium',
            agent_history=['RIF-Analyst'],
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            comments_count=3
        )
        
        assert context.number == 42
        assert context.title == "Test Issue"
        assert context.current_state_label == 'state:new'
        assert context.complexity_score == 2
        assert context.requires_planning is False
    
    def test_complexity_scoring(self):
        """Test complexity score calculation"""
        # Test very high complexity
        context = IssueContext(
            number=1, title="Test", body="", labels=['complexity:very-high'],
            state='open', complexity='very-high', priority='medium',
            agent_history=[], created_at='', updated_at='', comments_count=0
        )
        assert context.complexity_score == 4
        assert context.requires_planning is True
        
        # Test low complexity
        context.labels = ['complexity:low']
        context.complexity = 'low'
        assert context.complexity_score == 1
        assert context.requires_planning is False


class TestContextAnalyzer:
    """Test ContextAnalyzer functionality"""
    
    def setUp(self):
        self.analyzer = ContextAnalyzer()
    
    @patch('subprocess.run')
    def test_analyze_issue_success(self, mock_run):
        """Test successful issue analysis"""
        # Mock successful GitHub API response
        mock_response = {
            'number': 52,
            'title': 'Test Issue',
            'body': 'This is a test issue requiring implementation',
            'state': 'open',
            'labels': [
                {'name': 'state:implementing'}, 
                {'name': 'complexity:high'}
            ],
            'createdAt': '2024-01-01T00:00:00Z',
            'updatedAt': '2024-01-01T01:00:00Z',
            'comments': []
        }
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_response)
        
        analyzer = ContextAnalyzer()
        context = analyzer.analyze_issue(52)
        
        assert context.number == 52
        assert context.title == 'Test Issue'
        assert 'state:implementing' in context.labels
        assert context.complexity == 'high'
    
    def test_complexity_analysis(self):
        """Test complexity analysis from content"""
        analyzer = ContextAnalyzer()
        
        # Test high complexity detection
        complexity = analyzer._analyze_complexity(
            "Need to implement microservice architecture with database integration",
            []
        )
        assert complexity == 'very-high'
        
        # Test low complexity detection  
        complexity = analyzer._analyze_complexity(
            "Fix minor typo in documentation",
            []
        )
        assert complexity == 'low'
        
        # Test explicit label override
        complexity = analyzer._analyze_complexity(
            "Major architectural change",
            ['complexity:low']  # Explicit label should override content analysis
        )
        assert complexity == 'low'
    
    def test_priority_analysis(self):
        """Test priority analysis from content"""
        analyzer = ContextAnalyzer()
        
        # Test high priority detection
        priority = analyzer._analyze_priority(
            "Critical production issue blocking users",
            []
        )
        assert priority == 'high'
        
        # Test medium priority
        priority = analyzer._analyze_priority(
            "Important feature needed for next release",
            []
        )
        assert priority == 'medium'


class TestStateValidator:
    """Test StateValidator functionality"""
    
    def test_valid_state_transitions(self):
        """Test valid state transition validation"""
        validator = StateValidator()
        
        # Test valid transitions
        is_valid, reason = validator.validate_state_transition('new', 'analyzing')
        assert is_valid is True
        
        is_valid, reason = validator.validate_state_transition('implementing', 'validating')
        assert is_valid is True
        
        # Test invalid transitions
        is_valid, reason = validator.validate_state_transition('new', 'validating')
        assert is_valid is False
        assert 'Invalid transition' in reason
    
    def test_required_agents(self):
        """Test getting required agents for states"""
        validator = StateValidator()
        
        assert validator.get_required_agent('new') == 'RIF-Analyst'
        assert validator.get_required_agent('implementing') == 'RIF-Implementer'
        assert validator.get_required_agent('validating') == 'RIF-Validator'
        assert validator.get_required_agent('unknown') is None
    
    def test_issue_readiness_validation(self):
        """Test issue readiness for state transitions"""
        validator = StateValidator()
        
        # Create test context
        context = IssueContext(
            number=1, title="Test", body="", labels=['state:implementing'],
            state='open', complexity='high', priority='medium',
            agent_history=['RIF-Analyst', 'RIF-Implementer'], 
            created_at='', updated_at='', comments_count=0
        )
        
        # Should be ready for validation after implementation
        is_ready, issues = validator.validate_issue_ready_for_state(context, 'validating')
        assert is_ready is True
        assert len(issues) == 0
    
    def test_next_state_recommendation(self):
        """Test next state recommendations"""
        validator = StateValidator()
        
        # High complexity after analyzing should go to architecting
        context = IssueContext(
            number=1, title="Test", body="", labels=['state:analyzing', 'complexity:high'],
            state='open', complexity='high', priority='medium',
            agent_history=['RIF-Analyst'], created_at='', updated_at='', comments_count=0
        )
        
        next_state = validator.get_next_recommended_state(context)
        assert next_state == 'architecting'
        
        # Medium complexity can skip to implementation
        context.labels = ['state:analyzing', 'complexity:medium']
        context.complexity = 'medium'
        next_state = validator.get_next_recommended_state(context)
        assert next_state == 'implementing'


class TestOrchestrationHelper:
    """Test OrchestrationHelper functionality - the core utility for Claude Code"""
    
    @patch('orchestration_utilities.ContextAnalyzer.analyze_issue')
    def test_task_launch_code_generation(self, mock_analyze):
        """Test Task() launch code generation"""
        # Mock context
        mock_context = IssueContext(
            number=52, title="Implement feature X", body="Feature description",
            labels=['state:implementing'], state='open', complexity='medium', 
            priority='medium', agent_history=[], created_at='', updated_at='', 
            comments_count=0
        )
        mock_analyze.return_value = mock_context
        
        helper = OrchestrationHelper()
        task_code = helper.generate_task_launch_code(mock_context, 'RIF-Implementer')
        
        # Verify task code structure
        assert 'Task(' in task_code
        assert 'RIF-Implementer' in task_code
        assert 'subagent_type="general-purpose"' in task_code
        assert 'claude/agents/implementer.md' in task_code
        assert f'issue #{mock_context.number}' in task_code
    
    @patch('orchestration_utilities.ContextAnalyzer.analyze_multiple_issues')
    def test_orchestration_plan_generation(self, mock_analyze_multiple):
        """Test complete orchestration plan generation"""
        # Mock multiple contexts
        contexts = [
            IssueContext(
                number=1, title="Issue 1", body="", labels=['state:new'],
                state='open', complexity='low', priority='high', agent_history=[],
                created_at='', updated_at='', comments_count=0
            ),
            IssueContext(
                number=2, title="Issue 2", body="", labels=['state:implementing'],
                state='open', complexity='medium', priority='medium', agent_history=['RIF-Analyst'],
                created_at='', updated_at='', comments_count=0
            )
        ]
        mock_analyze_multiple.return_value = contexts
        
        helper = OrchestrationHelper()
        plan = helper.generate_orchestration_plan([1, 2])
        
        # Verify plan structure
        assert 'timestamp' in plan
        assert plan['total_issues'] == 2
        assert len(plan['parallel_tasks']) == 2
        assert len(plan['task_launch_codes']) == 2
        
        # Verify task codes are generated
        for task_code in plan['task_launch_codes']:
            assert 'Task(' in task_code
            assert 'subagent_type="general-purpose"' in task_code
    
    @patch('orchestration_utilities.ContextAnalyzer.analyze_issue')
    def test_orchestration_recommendation(self, mock_analyze):
        """Test orchestration action recommendation"""
        # Mock context for new issue
        mock_context = IssueContext(
            number=99, title="New Issue", body="New issue description",
            labels=[], state='open', complexity='medium', priority='medium',
            agent_history=[], created_at='', updated_at='', comments_count=0
        )
        mock_analyze.return_value = mock_context
        
        helper = OrchestrationHelper()
        recommendation = helper.recommend_orchestration_action(99)
        
        # Should recommend launching RIF-Analyst for new issue
        assert recommendation['action'] == 'launch_agent'
        assert recommendation['recommended_agent'] == 'RIF-Analyst'
        assert recommendation['next_state'] == 'analyzing'
        assert 'task_launch_code' in recommendation
    
    def test_agent_prompt_generation(self):
        """Test agent prompt generation"""
        helper = OrchestrationHelper()
        context = IssueContext(
            number=42, title="Test Issue", body="", labels=[],
            state='open', complexity='medium', priority='medium',
            agent_history=[], created_at='', updated_at='', comments_count=0
        )
        
        # Test different agent prompts
        analyst_prompt = helper._generate_agent_prompt('RIF-Analyst', context)
        assert 'You are RIF-Analyst' in analyst_prompt
        assert 'issue #42' in analyst_prompt
        assert 'claude/agents/analyst.md' in analyst_prompt
        
        implementer_prompt = helper._generate_agent_prompt('RIF-Implementer', context)
        assert 'You are RIF-Implementer' in implementer_prompt
        assert 'Implement solution' in implementer_prompt


class TestGitHubStateManager:
    """Test GitHub state management functionality"""
    
    @patch('subprocess.run')
    def test_update_issue_state(self, mock_run):
        """Test GitHub issue state updates"""
        mock_run.return_value.returncode = 0
        
        manager = GitHubStateManager()
        success = manager.update_issue_state(42, 'validating', 'Updated to validation state')
        
        assert success is True
        # Verify correct commands were called
        assert mock_run.call_count >= 2  # Remove old labels + add new label
    
    @patch('subprocess.run')
    def test_get_active_issues(self, mock_run):
        """Test fetching active issues"""
        mock_response = [
            {
                'number': 1,
                'title': 'Issue 1',
                'labels': [{'name': 'state:implementing'}]
            },
            {
                'number': 2, 
                'title': 'Issue 2',
                'labels': [{'name': 'state:validating'}]
            }
        ]
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_response)
        
        manager = GitHubStateManager()
        issues = manager.get_active_issues()
        
        assert len(issues) == 2
        assert issues[0]['number'] == 1
    
    @patch('subprocess.run')
    def test_add_agent_tracking_label(self, mock_run):
        """Test adding agent tracking labels"""
        mock_run.return_value.returncode = 0
        
        manager = GitHubStateManager()
        success = manager.add_agent_tracking_label(42, 'RIF-Implementer')
        
        assert success is True


class TestIntegrationScenarios:
    """Test integration scenarios that mimic real Claude Code orchestration"""
    
    @patch('orchestration_utilities.ContextAnalyzer.analyze_issue')
    @patch('subprocess.run')
    def test_complete_orchestration_workflow(self, mock_run, mock_analyze):
        """Test complete orchestration workflow simulation"""
        # Mock GitHub API responses
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '[]'
        
        # Mock issue analysis
        mock_context = IssueContext(
            number=52, title="Implement DynamicOrchestrator class", 
            body="Need to implement orchestrator", labels=['state:implementing'],
            state='open', complexity='high', priority='high',
            agent_history=['RIF-Analyst', 'RIF-Planner'], 
            created_at='', updated_at='', comments_count=5
        )
        mock_analyze.return_value = mock_context
        
        # Test the full workflow
        helper = OrchestrationHelper()
        state_manager = GitHubStateManager()
        
        # 1. Analyze issue and get recommendation
        recommendation = helper.recommend_orchestration_action(52)
        assert recommendation['action'] == 'launch_agent'
        assert recommendation['recommended_agent'] == 'RIF-Implementer'
        
        # 2. Generate task launch code
        task_code = recommendation['task_launch_code']
        assert 'Task(' in task_code
        assert 'RIF-Implementer' in task_code
        
        # 3. Update state after agent completion
        success = state_manager.update_issue_state(52, 'validating')
        assert success is True
        
    def test_pattern_compliance_verification(self):
        """Verify the implementation follows RIF pattern correctly"""
        # Verify no orchestrator classes exist
        helper = OrchestrationHelper()
        
        # Should generate Task() calls, not orchestrator instances
        context = IssueContext(
            number=1, title="Test", body="", labels=['state:new'],
            state='open', complexity='low', priority='medium',
            agent_history=[], created_at='', updated_at='', comments_count=0
        )
        
        task_code = helper.generate_task_launch_code(context, 'RIF-Analyst')
        
        # Verify it's proper Task() function call
        assert task_code.startswith('Task(')
        assert 'subagent_type="general-purpose"' in task_code
        assert 'You are RIF-Analyst' in task_code
        
        # Verify no orchestrator instantiation
        assert 'DynamicOrchestrator(' not in task_code
        assert 'orchestrator =' not in task_code


def run_tests():
    """Run all tests"""
    print("Running RIF Orchestration Utilities Tests...")
    print("=" * 50)
    
    # Run basic functionality tests
    test_classes = [
        TestIssueContext,
        TestContextAnalyzer,
        TestStateValidator,
        TestOrchestrationHelper,
        TestGitHubStateManager,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_tests()