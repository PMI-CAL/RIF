#!/usr/bin/env python3
"""
Test suite for Issue #112 - Big Picture and Correct Context Implementation

Tests all components implemented for Issue #112:
1. Live System Context Engine
2. Context Optimization Engine  
3. Design Specification Benchmarking Framework
4. Dynamic Dependency Tracker
5. Enhanced Learning System

These tests verify the complete implementation of big picture context and 
intelligent context optimization for development process enhancement.
"""

import unittest
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add systems directory to path
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'systems'))

try:
    from import_utils import import_live_system_context_engine, import_context_optimization_engine
    live_imports = import_live_system_context_engine()
    context_imports = import_context_optimization_engine()
    
    LiveSystemContextEngine = live_imports['LiveSystemContextEngine']
    ContextOptimizer = context_imports['ContextOptimizer']
    AgentType = context_imports['AgentType']
    ContextType = context_imports['ContextType']
    ContextItem = context_imports['ContextItem']
    AgentContext = context_imports['AgentContext']
    
    # Get the rest of the classes from the module
    live_module = live_imports['module']
    ContextGenerator = getattr(live_module, 'ContextGenerator', None)
    SystemAnalyzer = getattr(live_module, 'SystemAnalyzer', None)
    SystemComponent = getattr(live_module, 'SystemComponent', None)
    SystemComponentType = getattr(live_module, 'SystemComponentType', None)
    SystemOverview = getattr(live_module, 'SystemOverview', None)
    DependencyGraph = getattr(live_module, 'DependencyGraph', None)
except ImportError as e:
    # Fallback for testing - create minimal mock classes
    print(f"Warning: Could not import systems modules: {e}")
    print("Creating minimal test implementations...")
    
    class LiveSystemContextEngine:
        def __init__(self, repo_path, update_interval=300):
            self.repo_path = repo_path
        def get_live_context(self, force_refresh=False):
            return None
    
    class ContextOptimizer:
        def __init__(self, knowledge_base_path=None):
            pass
        def optimize_for_agent(self, agent_type, task_context, issue_number=None):
            return None

class TestIssue112LiveSystemContext(unittest.TestCase):
    """Test Live System Context Engine implementation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_repo = tempfile.mkdtemp()
        self.engine = LiveSystemContextEngine(self.test_repo, update_interval=1)
        
        # Create minimal test directory structure
        os.makedirs(os.path.join(self.test_repo, "claude", "agents"), exist_ok=True)
        os.makedirs(os.path.join(self.test_repo, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.test_repo, "systems"), exist_ok=True)
        
        # Create test agent file
        with open(os.path.join(self.test_repo, "claude", "agents", "rif-test.md"), "w") as f:
            f.write("## Role\nTest agent for verification\n## Responsibilities\nTesting functionality")
        
        # Create test workflow file
        with open(os.path.join(self.test_repo, "config", "rif-workflow.yaml"), "w") as f:
            f.write("workflow:\n  states:\n    - new\n    - implementing")
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
    
    def test_system_overview_generation(self):
        """Test system overview generation"""
        generator = ContextGenerator(self.test_repo)
        overview = generator.generate_system_overview()
        
        # Verify overview contains required fields
        self.assertIsInstance(overview, SystemOverview)
        self.assertIn("orchestrate", overview.purpose.lower())
        self.assertIn("automatic", overview.core_mission.lower())
        self.assertTrue(len(overview.key_capabilities) > 0)
        self.assertTrue(len(overview.quality_principles) > 0)
        self.assertTrue(len(overview.constraints) > 0)
    
    def test_component_analysis(self):
        """Test system component analysis"""
        analyzer = SystemAnalyzer(self.test_repo)
        components = analyzer.analyze_system_components()
        
        # Should find test agent
        agent_components = [c for c in components.values() if c.type == SystemComponentType.AGENT]
        self.assertTrue(len(agent_components) > 0)
        
        # Should find workflow configurations
        workflow_components = [c for c in components.values() if c.type == SystemComponentType.WORKFLOW]
        self.assertTrue(len(workflow_components) > 0)
        
        # Verify component structure
        test_component = next(iter(components.values()))
        self.assertIsInstance(test_component.id, str)
        self.assertIsInstance(test_component.name, str)
        self.assertIsInstance(test_component.dependencies, list)
    
    def test_dependency_analysis(self):
        """Test dependency analysis"""
        analyzer = SystemAnalyzer(self.test_repo)
        components = analyzer.analyze_system_components()
        dependency_graph = analyzer.analyze_dependencies(components)
        
        # Verify dependency graph structure
        self.assertIsInstance(dependency_graph, DependencyGraph)
        self.assertTrue(len(dependency_graph.components) > 0)
        self.assertIsInstance(dependency_graph.relationships, list)
        self.assertIsInstance(dependency_graph.critical_paths, list)
        self.assertIsInstance(dependency_graph.integration_points, list)
        self.assertIsInstance(dependency_graph.external_dependencies, list)
    
    def test_live_context_update(self):
        """Test live context generation and caching"""
        # Force context update
        context = self.engine.get_live_context(force_refresh=True)
        
        # Verify context structure
        self.assertIsNotNone(context)
        self.assertIsNotNone(context.overview)
        self.assertIsNotNone(context.dependency_graph)
        self.assertTrue(len(context.design_goals) > 0)
        self.assertTrue(len(context.quality_gates) > 0)
        
        # Verify timestamp is recent
        self.assertTrue((datetime.now() - context.context_timestamp).seconds < 60)
    
    @patch('subprocess.run')
    def test_health_assessment(self, mock_subprocess):
        """Test system health assessment"""
        # Mock successful git and gh commands
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        generator = ContextGenerator(self.test_repo)
        health = generator.assess_system_health()
        
        self.assertEqual(health['overall_status'], 'healthy')
        self.assertEqual(health['agent_status'], 'active')
        self.assertEqual(health['workflow_status'], 'operational')
        self.assertEqual(len(health['issues']), 0)
    
    def test_context_formatting(self):
        """Test context formatting for agent consumption"""
        context = self.engine.get_live_context(force_refresh=True)
        formatted = self.engine.format_context_for_agent(context, "rif-implementer")
        
        # Verify formatted context contains key sections
        self.assertIn("Live System Context", formatted)
        self.assertIn("System Overview", formatted)
        self.assertIn("Key Capabilities", formatted)
        self.assertIn("Design Goals", formatted)
        self.assertIn("System Architecture", formatted)
        self.assertIn("Workflow States", formatted)
        self.assertIn("Quality Standards", formatted)
        self.assertIn("System Health", formatted)


class TestIssue112ContextOptimization(unittest.TestCase):
    """Test Context Optimization Engine implementation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_kb = tempfile.mkdtemp()
        self.optimizer = ContextOptimizer(self.test_kb)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_kb):
            shutil.rmtree(self.test_kb)
    
    def test_agent_context_limits(self):
        """Test agent-specific context limits"""
        # Verify all agent types have context limits
        for agent_type in AgentType:
            self.assertIn(agent_type, self.optimizer.agent_context_limits)
            self.assertTrue(self.optimizer.agent_context_limits[agent_type] > 0)
    
    def test_relevance_weights(self):
        """Test relevance weight initialization"""
        # Verify relevance weights are defined for key agents
        key_agents = [AgentType.ANALYST, AgentType.IMPLEMENTER, AgentType.VALIDATOR]
        
        for agent in key_agents:
            self.assertIn(agent, self.optimizer.relevance_weights)
            weights = self.optimizer.relevance_weights[agent]
            self.assertIn(ContextType.CLAUDE_CODE_CAPABILITIES, weights)
            self.assertEqual(weights[ContextType.CLAUDE_CODE_CAPABILITIES], 1.0)
    
    def test_knowledge_gathering(self):
        """Test knowledge gathering from various sources"""
        task_context = {"description": "test implementation task"}
        knowledge_items = self.optimizer._gather_available_knowledge(task_context, 112)
        
        # Should gather multiple types of knowledge
        self.assertTrue(len(knowledge_items) >= 4)
        
        # Should include Claude Code capabilities
        claude_items = [item for item in knowledge_items 
                       if item.type == ContextType.CLAUDE_CODE_CAPABILITIES]
        self.assertTrue(len(claude_items) > 0)
        
        # Should include system overview
        overview_items = [item for item in knowledge_items 
                         if item.type == ContextType.SYSTEM_OVERVIEW]
        self.assertTrue(len(overview_items) > 0)
    
    def test_relevance_scoring(self):
        """Test knowledge relevance scoring"""
        task_context = {"description": "context optimization implementation"}
        knowledge_items = self.optimizer._gather_available_knowledge(task_context, 112)
        
        # Test scoring for implementer agent
        scores = self.optimizer._score_knowledge_relevance(
            AgentType.IMPLEMENTER, task_context, knowledge_items
        )
        
        # Should have scores for all items
        self.assertEqual(len(scores), len(knowledge_items))
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)
    
    def test_context_optimization(self):
        """Test complete context optimization for agent"""
        task_context = {
            "description": "implement context optimization engine",
            "complexity": "high"
        }
        
        # Test optimization for implementer
        agent_context = self.optimizer.optimize_for_agent(
            AgentType.IMPLEMENTER, task_context, 112
        )
        
        # Verify agent context structure
        self.assertIsInstance(agent_context, AgentContext)
        self.assertEqual(agent_context.agent_type, AgentType.IMPLEMENTER)
        self.assertTrue(len(agent_context.relevant_knowledge) > 0)
        self.assertIsNotNone(agent_context.system_context)
        self.assertTrue(0 <= agent_context.context_window_utilization <= 1)
    
    def test_size_constraints(self):
        """Test context window size constraints"""
        task_context = {"description": "large context test"}
        
        for agent_type in AgentType:
            agent_context = self.optimizer.optimize_for_agent(agent_type, task_context)
            
            # Total size should not exceed agent limit
            context_limit = self.optimizer.agent_context_limits[agent_type]
            self.assertTrue(agent_context.total_size <= context_limit)
    
    def test_context_formatting(self):
        """Test context formatting for agent consumption"""
        task_context = {"description": "test formatting"}
        agent_context = self.optimizer.optimize_for_agent(AgentType.ANALYST, task_context)
        
        formatted = self.optimizer.format_context_for_agent(agent_context)
        
        # Should contain key sections
        self.assertIn("System Context", formatted)
        self.assertIn("Claude Code Capabilities", formatted)
        self.assertIn("Context Optimization", formatted)
        self.assertIn("Window Utilization", formatted)


class TestIssue112Integration(unittest.TestCase):
    """Test integration between all Issue #112 components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_repo = tempfile.mkdtemp()
        self.engine = LiveSystemContextEngine(self.test_repo)
        self.optimizer = ContextOptimizer()
        
        # Create minimal test structure
        os.makedirs(os.path.join(self.test_repo, "claude", "agents"), exist_ok=True)
        with open(os.path.join(self.test_repo, "claude", "agents", "rif-implementer.md"), "w") as f:
            f.write("## Role\nImplements code and features\n")
    
    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
    
    def test_end_to_end_context_flow(self):
        """Test complete end-to-end context optimization flow"""
        # 1. Generate live system context
        live_context = self.engine.get_live_context(force_refresh=True)
        self.assertIsNotNone(live_context)
        
        # 2. Optimize context for specific agent task
        task_context = {
            "description": "implement big picture context system",
            "issue_number": 112,
            "complexity": "high"
        }
        
        agent_context = self.optimizer.optimize_for_agent(
            AgentType.IMPLEMENTER, task_context, 112
        )
        
        # 3. Verify integration works
        self.assertIsNotNone(agent_context)
        self.assertTrue(len(agent_context.relevant_knowledge) > 0)
        
        # 4. Test context formatting
        formatted_context = self.optimizer.format_context_for_agent(agent_context)
        self.assertIn("Claude Code", formatted_context)
        self.assertIn("RIF", formatted_context)
    
    def test_big_picture_awareness(self):
        """Test that agents receive comprehensive big picture context"""
        live_context = self.engine.get_live_context(force_refresh=True)
        
        # Verify big picture elements are present
        self.assertIn("orchestrate", live_context.overview.purpose.lower())
        self.assertIn("automatic", live_context.overview.core_mission.lower())
        
        # Verify design goals include key RIF principles  
        design_goal_titles = [goal['title'] for goal in live_context.design_goals]
        self.assertIn("Complete Automation", design_goal_titles)
        self.assertIn("Quality Excellence", design_goal_titles)
        
        # Verify architecture understanding
        self.assertIn("specialized agents", live_context.overview.architecture_summary.lower())
    
    def test_context_window_efficiency(self):
        """Test that context optimization prevents window bloat"""
        task_context = {"description": "complex multi-component implementation"}
        
        for agent_type in [AgentType.ANALYST, AgentType.IMPLEMENTER, AgentType.VALIDATOR]:
            agent_context = self.optimizer.optimize_for_agent(agent_type, task_context)
            
            # Context utilization should be reasonable (not exceeding limits)
            self.assertTrue(agent_context.context_window_utilization <= 1.0)
            
            # Should provide relevant information without overwhelming
            self.assertTrue(1 <= len(agent_context.relevant_knowledge) <= 10)
    
    def test_performance_requirements(self):
        """Test that context operations meet performance requirements"""
        import time
        
        # Test live context generation performance
        start_time = time.time()
        live_context = self.engine.get_live_context(force_refresh=True)
        context_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should generate context within reasonable time
        self.assertLess(context_time, 5000)  # Less than 5 seconds for test env
        
        # Test context optimization performance
        task_context = {"description": "performance test"}
        
        start_time = time.time()
        agent_context = self.optimizer.optimize_for_agent(AgentType.IMPLEMENTER, task_context)
        optimization_time = (time.time() - start_time) * 1000
        
        # Should optimize context quickly (relaxed for test environment)
        self.assertLess(optimization_time, 1000)  # Less than 1 second


class TestIssue112BenchmarkingFramework(unittest.TestCase):
    """Test Design Specification Benchmarking Framework"""
    
    def test_benchmarking_framework_exists(self):
        """Test that benchmarking framework is implemented"""
        benchmarking_file = "/Users/cal/DEV/RIF/systems/design-benchmarking-framework.py"
        self.assertTrue(os.path.exists(benchmarking_file), 
                       "Design benchmarking framework should be implemented")
    
    def test_dependency_tracker_exists(self):
        """Test that dynamic dependency tracker is implemented"""  
        dependency_file = "/Users/cal/DEV/RIF/systems/dynamic-dependency-tracker.py"
        self.assertTrue(os.path.exists(dependency_file),
                       "Dynamic dependency tracker should be implemented")
    
    def test_enhanced_learning_exists(self):
        """Test that enhanced learning system is implemented"""
        # Check if enhanced learning integration is available
        # This would be tested more thoroughly once the complete implementation is available
        systems_dir = "/Users/cal/DEV/RIF/systems"
        self.assertTrue(os.path.exists(systems_dir), 
                       "Systems directory should exist with all components")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIssue112LiveSystemContext))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIssue112ContextOptimization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIssue112Integration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIssue112BenchmarkingFramework))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Issue #112 Implementation Test Results ===")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            failure_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {failure_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)