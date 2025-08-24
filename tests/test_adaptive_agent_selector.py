#!/usr/bin/env python3
"""
Test suite for Adaptive Agent Selection System.

Tests the 5-layer intelligence engine for agent selection including:
- Issue context analysis
- Historical pattern matching  
- Agent capability mapping
- Dynamic team composition
- Selection learning system
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.adaptive_agent_selector import (
    AdaptiveAgentSelector,
    IssueContextAnalyzer,
    HistoricalPatternMatcher,
    AgentCapabilityMapper,
    DynamicTeamComposer,
    SelectionLearningSystem,
    RequirementMapping,
    AgentCapability,
    TeamComposition,
    SelectionResult,
    create_adaptive_selector,
    select_agents_for_issue,
    generate_claude_code_task_launches
)


class TestIssueContextAnalyzer(unittest.TestCase):
    """Test Layer 1: Issue Context Analysis"""
    
    def setUp(self):
        self.analyzer = IssueContextAnalyzer()
        
    def test_extract_requirements_basic(self):
        """Test basic requirement extraction from issue context"""
        issue_context = {
            'title': 'Implement user authentication system',
            'body': 'Need to build secure login with encryption and validation',
            'labels': ['enhancement', 'security']
        }
        
        requirements = self.analyzer.extract_requirements(issue_context)
        
        # Should extract security and implementation requirements
        req_types = [req.requirement for req in requirements]
        self.assertIn('security', req_types)
        self.assertIn('implementation', req_types)
        # May also extract architecture depending on content analysis
        self.assertGreater(len(requirements), 0)
        
        # Security should have high priority due to content
        security_req = next(req for req in requirements if req.requirement == 'security')
        self.assertGreater(security_req.priority, 0.5)
    
    def test_assess_complexity_high(self):
        """Test complexity assessment for high complexity issue"""
        issue_context = {
            'title': 'Microservice architecture migration',
            'body': 'Migrate monolithic system to distributed microservices with orchestration',
            'labels': ['complexity:very-high']
        }
        
        complexity, breakdown = self.analyzer.assess_complexity(issue_context)
        
        self.assertEqual(complexity, 4)  # Very high complexity
        self.assertGreater(breakdown.get('very_high', 0), 0)
    
    def test_assess_complexity_low(self):
        """Test complexity assessment for low complexity issue"""
        issue_context = {
            'title': 'Fix typo in documentation',
            'body': 'Simple documentation fix',
            'labels': ['bug', 'documentation']
        }
        
        complexity, breakdown = self.analyzer.assess_complexity(issue_context)
        
        self.assertEqual(complexity, 1)  # Low complexity
    
    def test_get_capabilities_for_requirement(self):
        """Test capability mapping for different requirement types"""
        analysis_caps = self.analyzer._get_capabilities_for_requirement('analysis')
        self.assertIn('requirements', analysis_caps)
        self.assertIn('patterns', analysis_caps)
        
        security_caps = self.analyzer._get_capabilities_for_requirement('security')
        self.assertIn('vulnerabilities', security_caps)
        self.assertIn('auth', security_caps)


class TestHistoricalPatternMatcher(unittest.TestCase):
    """Test Layer 2: Historical Pattern Matching"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.matcher = HistoricalPatternMatcher(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_calculate_text_similarity(self):
        """Test text similarity calculation"""
        text1 = "implement user authentication system"
        text2 = "build user login authentication"
        
        similarity = self.matcher._calculate_text_similarity(text1, text2)
        
        self.assertGreater(similarity, 0.3)  # Should have reasonable similarity
        self.assertLessEqual(similarity, 1.0)
    
    def test_extract_agents_from_comments(self):
        """Test agent extraction from issue comments"""
        comments = [
            {
                'body': '## Analysis Complete\n\n**Agent**: RIF-Analyst\n**Status**: Complete'
            },
            {
                'body': '## Implementation Started\n\n**Agent**: RIF-Implementer\n**Progress**: 50%'
            }
        ]
        
        agents = self.matcher._extract_agents_from_comments(comments)
        
        self.assertIn('RIF-Analyst', agents)
        self.assertIn('RIF-Implementer', agents)
    
    def test_assess_success_indicators(self):
        """Test success indicator assessment"""
        comments = [
            {'body': 'Implementation complete and validated successfully'},
            {'body': 'All tests passing, ready for merge'},
            {'body': 'Failed to implement due to blocking issue'}
        ]
        
        indicators = self.matcher._assess_success_indicators(comments)
        
        self.assertIn('completion_rate', indicators)
        self.assertGreater(indicators['completion_rate'], 0.5)  # More success than failure
    
    @patch('subprocess.run')
    def test_find_similar_issues_mock(self, mock_subprocess):
        """Test finding similar issues with mocked GitHub response"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps([
            {
                'number': 123,
                'title': 'Build authentication system',
                'body': 'Implement user login with security',
                'labels': [{'name': 'security'}, {'name': 'enhancement'}]
            }
        ])
        
        issue_context = {
            'title': 'Create user authentication',
            'body': 'Need secure login system'
        }
        
        similar = self.matcher.find_similar_issues(issue_context)
        
        self.assertGreater(len(similar), 0)
        self.assertIn('similarity_score', similar[0])


class TestAgentCapabilityMapper(unittest.TestCase):
    """Test Layer 3: Agent Capability Mapping"""
    
    def setUp(self):
        self.mapper = AgentCapabilityMapper()
    
    def test_agent_capability_matrix(self):
        """Test that all agents have capabilities defined"""
        required_agents = ['rif-analyst', 'rif-architect', 'rif-implementer', 'rif-validator', 'rif-security', 'rif-performance']
        
        for agent in required_agents:
            self.assertIn(agent, self.mapper.agent_capability_matrix)
            capabilities = self.mapper.agent_capability_matrix[agent]
            self.assertIsInstance(capabilities, list)
            self.assertGreater(len(capabilities), 0)
    
    def test_map_requirements_to_capabilities(self):
        """Test requirement to capability mapping"""
        requirements = [
            RequirementMapping('security', ['vulnerabilities', 'auth'], 0.8, 1.5),
            RequirementMapping('implementation', ['coding', 'optimization'], 0.7, 1.2)
        ]
        
        capability_map = self.mapper.map_requirements_to_capabilities(requirements)
        
        self.assertIn('security', capability_map)
        self.assertIn('implementation', capability_map)
        self.assertEqual(capability_map['security'], ['vulnerabilities', 'auth'])
    
    def test_get_agents_with_capabilities(self):
        """Test finding agents with specific capabilities"""
        capabilities = ['vulnerabilities', 'auth']
        
        agents = self.mapper.get_agents_with_capabilities(capabilities)
        
        self.assertIn('rif-security', agents)
        self.assertIsInstance(agents['rif-security'], AgentCapability)
    
    def test_get_capability_coverage(self):
        """Test capability coverage calculation"""
        agents = ['rif-implementer', 'rif-validator']
        required_capabilities = ['coding', 'testing', 'quality']
        
        coverage = self.mapper.get_capability_coverage(agents, required_capabilities)
        
        # Should cover at least coding from implementer and testing/quality from validator
        self.assertGreater(coverage, 0.6)
        self.assertLessEqual(coverage, 1.0)


class TestDynamicTeamComposer(unittest.TestCase):
    """Test Layer 4: Dynamic Team Composition"""
    
    def setUp(self):
        self.mapper = AgentCapabilityMapper()
        self.composer = DynamicTeamComposer(self.mapper)
    
    def test_compose_optimal_team(self):
        """Test optimal team composition"""
        requirements = [
            RequirementMapping('analysis', ['requirements', 'patterns'], 0.9, 1.2),
            RequirementMapping('implementation', ['coding', 'optimization'], 0.8, 1.5),
            RequirementMapping('validation', ['testing', 'quality'], 0.7, 1.3)
        ]
        
        available_agents = self.mapper.agent_capabilities
        constraints = {'max_team_size': 4}
        
        team = self.composer.compose_optimal_team(requirements, available_agents, constraints)
        
        self.assertIsInstance(team, TeamComposition)
        self.assertGreater(len(team.agents), 0)
        self.assertLessEqual(len(team.agents), 4)
        self.assertGreater(team.confidence_score, 0.0)
        self.assertLessEqual(team.capability_coverage, 1.0)
    
    def test_validate_team_coverage(self):
        """Test team coverage validation"""
        team = ['rif-analyst', 'rif-implementer']
        requirements = [
            RequirementMapping('analysis', ['requirements'], 0.8, 1.0),
            RequirementMapping('implementation', ['coding'], 0.9, 1.2)
        ]
        
        coverage = self.composer.validate_team_coverage(team, requirements)
        
        self.assertGreater(coverage, 0.8)  # Should have good coverage
    
    def test_add_specialists_if_needed(self):
        """Test specialist addition logic"""
        base_team = ['rif-implementer']
        requirements = [
            RequirementMapping('security', ['vulnerabilities', 'auth'], 0.9, 2.0)  # High priority/complexity
        ]
        constraints = {'max_team_size': 4}
        
        enhanced_team = self.composer._add_specialists_if_needed(base_team, requirements, constraints)
        
        # Should add security specialist for high-priority security requirement
        self.assertIn('rif-security', enhanced_team)


class TestSelectionLearningSystem(unittest.TestCase):
    """Test Layer 5: Selection Learning System"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.learning_system = SelectionLearningSystem(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_record_selection_outcome(self):
        """Test recording selection outcomes"""
        # Create mock selection result
        team_composition = TeamComposition(
            agents=['rif-analyst', 'rif-implementer'],
            confidence_score=0.8,
            capability_coverage=0.9,
            estimated_effort=3.0,
            success_probability=0.85,
            historical_basis=[]
        )
        
        selection = SelectionResult(
            recommended_agents=['rif-analyst', 'rif-implementer'],
            team_composition=team_composition,
            rationale="Test selection",
            confidence_score=0.8,
            alternative_options=[],
            performance_metrics={}
        )
        
        actual_outcome = {
            'success_score': 0.9,
            'completion_time': 5.0,
            'quality_score': 0.85
        }
        
        self.learning_system.record_selection_outcome(selection, actual_outcome)
        
        # Check that outcome file was created
        self.assertTrue(self.learning_system.selection_outcomes_file.exists())
    
    def test_update_agent_performance_scores(self):
        """Test updating agent performance scores"""
        performances = {
            'rif-analyst': 0.85,
            'rif-implementer': 0.78,
            'rif-validator': 0.92
        }
        
        self.learning_system.update_agent_performance_scores(performances)
        
        # Check that trends file was created
        self.assertTrue(self.learning_system.performance_trends_file.exists())
        
        # Verify trends were recorded
        trends = self.learning_system._load_performance_trends()
        self.assertIn('rif-analyst', trends)
        self.assertEqual(trends['rif-analyst'][0]['performance'], 0.85)
    
    def test_get_learning_insights(self):
        """Test learning insights generation"""
        insights = self.learning_system.get_learning_insights()
        
        self.assertIn('total_selections_recorded', insights)
        self.assertIn('average_selection_accuracy', insights)
        self.assertIn('agent_performance_trends', insights)


class TestAdaptiveAgentSelector(unittest.TestCase):
    """Test main orchestrator class"""
    
    def setUp(self):
        self.selector = AdaptiveAgentSelector()
    
    def test_initialization(self):
        """Test proper initialization of all layers"""
        self.assertIsNotNone(self.selector.context_analyzer)
        self.assertIsNotNone(self.selector.pattern_matcher)
        self.assertIsNotNone(self.selector.capability_mapper)
        self.assertIsNotNone(self.selector.team_composer)
        self.assertIsNotNone(self.selector.learning_system)
    
    @patch('claude.commands.adaptive_agent_selector.subprocess.run')
    def test_select_optimal_agents(self, mock_subprocess):
        """Test end-to-end agent selection"""
        # Mock GitHub API response for similar issues
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps([])
        
        issue_context = {
            'number': 123,
            'title': 'Implement user authentication',
            'body': 'Build secure login system with encryption and validation',
            'labels': ['enhancement', 'security', 'complexity:high']
        }
        
        result = self.selector.select_optimal_agents(issue_context)
        
        self.assertIsInstance(result, SelectionResult)
        self.assertGreater(len(result.recommended_agents), 0)
        self.assertIsInstance(result.team_composition, TeamComposition)
        self.assertGreater(result.confidence_score, 0.0)
        self.assertIn('rationale', result.__dict__)
        self.assertIn('performance_metrics', result.__dict__)
    
    def test_create_fallback_selection(self):
        """Test fallback selection when normal process fails"""
        issue_context = {'number': 123}
        constraints = {}
        
        fallback = self.selector._create_fallback_selection(issue_context, constraints)
        
        self.assertIsInstance(fallback, SelectionResult)
        self.assertIn('rif-analyst', fallback.recommended_agents)
        self.assertIn('rif-implementer', fallback.recommended_agents)
        self.assertEqual(fallback.confidence_score, 0.5)
    
    def test_record_selection_outcome(self):
        """Test recording selection outcome for learning"""
        # Create mock selection result
        team_composition = TeamComposition(
            agents=['rif-analyst'],
            confidence_score=0.8,
            capability_coverage=0.9,
            estimated_effort=2.0,
            success_probability=0.85,
            historical_basis=[]
        )
        
        selection = SelectionResult(
            recommended_agents=['rif-analyst'],
            team_composition=team_composition,
            rationale="Test",
            confidence_score=0.8,
            alternative_options=[],
            performance_metrics={}
        )
        
        outcome = {'overall_success': 0.9, 'rif-analyst_performance': 0.85}
        
        # Should not raise exception
        self.selector.record_selection_outcome(selection, outcome)
    
    def test_get_selection_insights(self):
        """Test selection insights retrieval"""
        insights = self.selector.get_selection_insights()
        
        self.assertIn('selection_statistics', insights)
        self.assertIn('learning_insights', insights)
        self.assertIn('system_health', insights)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for Claude Code integration"""
    
    def test_create_adaptive_selector(self):
        """Test factory function for creating selector"""
        selector = create_adaptive_selector()
        
        self.assertIsInstance(selector, AdaptiveAgentSelector)
    
    @patch('claude.commands.adaptive_agent_selector.subprocess.run')
    def test_select_agents_for_issue_mock(self, mock_subprocess):
        """Test convenience function with mocked GitHub API"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps({
            'number': 123,
            'title': 'Test issue',
            'body': 'Test implementation',
            'labels': [{'name': 'enhancement'}]
        })
        
        # First call to get issue, second for similar issues
        mock_subprocess.side_effect = [
            mock_subprocess.return_value,
            Mock(returncode=0, stdout=json.dumps([]))
        ]
        
        result = select_agents_for_issue(123)
        
        self.assertIsInstance(result, SelectionResult)
    
    def test_generate_claude_code_task_launches(self):
        """Test Task launch code generation"""
        team_composition = TeamComposition(
            agents=['rif-analyst', 'rif-implementer'],
            confidence_score=0.8,
            capability_coverage=0.9,
            estimated_effort=3.0,
            success_probability=0.85,
            historical_basis=[]
        )
        
        selection = SelectionResult(
            recommended_agents=['rif-analyst', 'rif-implementer'],
            team_composition=team_composition,
            rationale="Test selection",
            confidence_score=0.8,
            alternative_options=[],
            performance_metrics={}
        )
        
        task_codes = generate_claude_code_task_launches(selection, 123)
        
        self.assertEqual(len(task_codes), 2)
        self.assertTrue(all('Task(' in code for code in task_codes))
        self.assertTrue(any('RIF-ANALYST' in code for code in task_codes))
        self.assertTrue(any('RIF-IMPLEMENTER' in code for code in task_codes))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_complete_selection_workflow(self):
        """Test complete agent selection workflow"""
        selector = create_adaptive_selector()
        
        # Complex issue requiring multiple agents
        issue_context = {
            'number': 456,
            'title': 'Microservice architecture with security and performance optimization',
            'body': '''
            Need to design and implement a distributed microservice architecture 
            with strong security measures and performance optimization. 
            This includes authentication, encryption, load balancing, and monitoring.
            ''',
            'labels': ['complexity:very-high', 'security', 'performance', 'architecture']
        }
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = json.dumps([])
            
            result = selector.select_optimal_agents(issue_context)
        
        # Should select multiple agents for very high complexity
        self.assertGreater(len(result.recommended_agents), 2)
        
        # Should include security specialist for security requirements
        self.assertIn('rif-security', result.recommended_agents)
        
        # Should have high confidence for well-covered requirements
        self.assertGreater(result.confidence_score, 0.6)
        
        # Should generate valid Task launch codes
        task_codes = generate_claude_code_task_launches(result, 456)
        self.assertGreater(len(task_codes), 0)
        
        # All task codes should be valid Python function calls
        for code in task_codes:
            self.assertIn('Task(', code)
            self.assertIn('description=', code)
            self.assertIn('subagent_type=', code)
            self.assertIn('prompt=', code)


if __name__ == '__main__':
    # Set up test environment
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Adaptive Agent Selector Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
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
        print("✅ EXCELLENT: Adaptive Agent Selector system is working correctly!")
    elif success_rate >= 80:
        print("✅ GOOD: Adaptive Agent Selector system is mostly functional")
    elif success_rate >= 70:
        print("⚠️  WARNING: Some issues detected in Adaptive Agent Selector")
    else:
        print("❌ CRITICAL: Major issues in Adaptive Agent Selector system")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)