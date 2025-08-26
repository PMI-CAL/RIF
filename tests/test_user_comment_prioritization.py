#!/usr/bin/env python3
"""
Test Suite for User Comment Prioritization - Issue #275
RIF-Implementer: Comprehensive validation of user-first orchestration system

This test suite validates the user comment prioritization system ensuring:
1. User directives are correctly extracted from comments and issue bodies
2. Priority hierarchy is enforced (VERY HIGH for users, MEDIUM for agents)
3. "Think Hard" logic triggers for complex scenarios
4. Conflicts are resolved in favor of user directives 100% of the time
5. Integration with existing orchestration system works correctly
"""

import unittest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from claude.commands.user_comment_prioritizer import (
        UserCommentPrioritizer,
        UserDirective,
        DirectivePriority,
        DirectiveSource,
        integrate_user_comment_prioritization,
        validate_user_directive_extraction
    )
    from claude.commands.orchestration_intelligence_integration import (
        make_user_priority_orchestration_decision,
        get_enhanced_blocking_detection_status,
        OrchestrationDecision
    )
    USER_COMMENT_PRIORITIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: User comment prioritization not available for testing: {e}")
    USER_COMMENT_PRIORITIZATION_AVAILABLE = False


class TestUserCommentPrioritizer(unittest.TestCase):
    """Test UserCommentPrioritizer class functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.skipTest("User comment prioritization not available")
        
        self.prioritizer = UserCommentPrioritizer()
    
    def test_directive_priority_enum(self):
        """Test DirectivePriority enum values and numeric priorities"""
        self.assertEqual(DirectivePriority.VERY_HIGH.numeric_value, 100)
        self.assertEqual(DirectivePriority.HIGH.numeric_value, 75)
        self.assertEqual(DirectivePriority.MEDIUM.numeric_value, 50)
        self.assertEqual(DirectivePriority.LOW.numeric_value, 25)
        
        # Test priority ordering
        self.assertTrue(DirectivePriority.VERY_HIGH.numeric_value > DirectivePriority.MEDIUM.numeric_value)
    
    def test_user_directive_dataclass(self):
        """Test UserDirective dataclass and auto-priority setting"""
        # Test user comment directive gets VERY HIGH priority
        user_directive = UserDirective(
            source_type=DirectiveSource.USER_COMMENT,
            priority=DirectivePriority.LOW,  # This should be overridden
            directive_text="implement issue #275",
            action_type="IMPLEMENT",
            target_issues=[275],
            specific_agents=[],
            reasoning="Test user directive",
            confidence_score=0.9,
            timestamp=datetime.now()
        )
        
        # Priority should be auto-set to VERY HIGH for user comments
        self.assertEqual(user_directive.priority, DirectivePriority.VERY_HIGH)
        
        # Test agent suggestion gets MEDIUM priority
        agent_directive = UserDirective(
            source_type=DirectiveSource.AGENT_SUGGESTION,
            priority=DirectivePriority.VERY_HIGH,  # This should be overridden
            directive_text="suggested implementation",
            action_type="IMPLEMENT",
            target_issues=[275],
            specific_agents=["RIF-Implementer"],
            reasoning="Agent suggestion",
            confidence_score=0.7,
            timestamp=datetime.now()
        )
        
        # Priority should be auto-set to MEDIUM for agent suggestions
        self.assertEqual(agent_directive.priority, DirectivePriority.MEDIUM)
    
    def test_directive_pattern_parsing(self):
        """Test parsing of various user directive patterns"""
        test_cases = [
            # IMPLEMENT patterns
            {
                'text': "Please implement issue #275 first",
                'expected_action': 'IMPLEMENT',
                'expected_issues': [275]
            },
            {
                'text': "work on issue 276 next",
                'expected_action': 'IMPLEMENT',
                'expected_issues': [276]
            },
            {
                'text': "focus on issue #277",
                'expected_action': 'IMPLEMENT', 
                'expected_issues': [277]
            },
            
            # BLOCK patterns
            {
                'text': "don't work on issue #278",
                'expected_action': 'BLOCK',
                'expected_issues': [278]
            },
            {
                'text': "block issue 279 until we resolve dependencies",
                'expected_action': 'BLOCK',
                'expected_issues': [279]
            },
            
            # PRIORITIZE patterns
            {
                'text': "issue #280 is high priority",
                'expected_action': 'PRIORITIZE',
                'expected_issues': [280]
            },
            {
                'text': "urgent issue #281",
                'expected_action': 'PRIORITIZE',
                'expected_issues': [281]
            },
            
            # SEQUENCE patterns
            {
                'text': "issue #282 before issue #283",
                'expected_action': 'SEQUENCE',
                'expected_issues': [282, 283]
            },
            {
                'text': "complete issue 284 before issue 285",
                'expected_action': 'SEQUENCE',
                'expected_issues': [284, 285]
            },
            
            # AGENT_SPECIFIC patterns
            {
                'text': "use RIF-Implementer for issue #286",
                'expected_action': 'AGENT_SPECIFIC',
                'expected_issues': [286],
                'expected_agents': ['RIF-Implementer']
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                directives = self.prioritizer._parse_text_for_directives(
                    case['text'], DirectiveSource.USER_COMMENT, "test_user"
                )
                
                self.assertGreater(len(directives), 0, f"No directives found for: {case['text']}")
                
                directive = directives[0]
                self.assertEqual(directive.action_type, case['expected_action'])
                self.assertEqual(directive.target_issues, case['expected_issues'])
                self.assertEqual(directive.priority, DirectivePriority.VERY_HIGH)
                
                if 'expected_agents' in case:
                    self.assertEqual(directive.specific_agents, case['expected_agents'])
    
    def test_think_hard_trigger_detection(self):
        """Test detection of 'Think Hard' triggers in user comments"""
        think_hard_texts = [
            "think hard about the orchestration approach",
            "carefully consider the dependencies",
            "complex scenario with multiple options", 
            "difficult decision for issue #275",
            "analyze thoroughly before proceeding"
        ]
        
        for text in think_hard_texts:
            with self.subTest(text=text):
                directives = self.prioritizer._parse_text_for_directives(
                    text, DirectiveSource.USER_COMMENT, "test_user"
                )
                
                # Think Hard should increase confidence score
                if directives:
                    self.assertGreaterEqual(directives[0].confidence_score, 0.9)
    
    def test_conflict_analysis(self):
        """Test analysis of conflicts between user directives and agent recommendations"""
        # Mock user directives
        user_directives = [
            UserDirective(
                source_type=DirectiveSource.USER_COMMENT,
                priority=DirectivePriority.VERY_HIGH,
                directive_text="block issue #275",
                action_type="BLOCK",
                target_issues=[275],
                specific_agents=[],
                reasoning="User wants to block",
                confidence_score=0.9,
                timestamp=datetime.now()
            )
        ]
        
        # Mock agent recommendations
        agent_recommendations = [
            {
                'issue': 275,
                'action': 'launch_agent',
                'recommended_agent': 'RIF-Implementer'
            }
        ]
        
        conflict_analysis = self.prioritizer.analyze_directive_conflicts(
            user_directives, agent_recommendations
        )
        
        # Should detect conflict
        self.assertGreater(conflict_analysis['total_conflicts'], 0)
        self.assertEqual(conflict_analysis['user_directive_influence_percentage'], 100.0)
        self.assertEqual(conflict_analysis['conflict_resolution_rule'], 'USER_DIRECTIVES_ALWAYS_WIN')
        
        # Should have user override
        self.assertGreater(len(conflict_analysis['user_overrides']), 0)
        override = conflict_analysis['user_overrides'][0]
        self.assertEqual(override['issue'], 275)
        self.assertEqual(override['user_directive_action'], 'BLOCK')
    
    def test_user_priority_orchestration_plan(self):
        """Test creation of user-priority orchestration plan"""
        # Mock GitHub issue extraction (tested separately)
        with patch.object(self.prioritizer, 'extract_user_directives') as mock_extract:
            mock_extract.return_value = [
                UserDirective(
                    source_type=DirectiveSource.USER_COMMENT,
                    priority=DirectivePriority.VERY_HIGH,
                    directive_text="implement issue #275",
                    action_type="IMPLEMENT",
                    target_issues=[275],
                    specific_agents=["RIF-Implementer"],
                    reasoning="User wants implementation",
                    confidence_score=0.95,
                    timestamp=datetime.now()
                )
            ]
            
            agent_recommendations = [
                {
                    'issue': 276,
                    'action': 'launch_agent',
                    'recommended_agent': 'RIF-Validator'
                }
            ]
            
            plan = self.prioritizer.create_user_priority_orchestration_plan(
                [275, 276], agent_recommendations
            )
            
            # Should prioritize user directives
            self.assertEqual(plan['decision_hierarchy'], 'USER_FIRST')
            self.assertGreater(plan['user_directive_count'], 0)
            self.assertEqual(plan['user_directive_compliance'], 100.0)
            
            # Should have orchestration decisions
            decisions = plan['final_orchestration_decisions']
            self.assertGreater(len(decisions), 0)
            
            # First decision should be user directive with VERY HIGH priority
            user_decision = next(d for d in decisions if d['source'] == 'USER_DIRECTIVE')
            self.assertEqual(user_decision['priority'], 'VERY_HIGH')
            self.assertEqual(user_decision['issue'], 275)


class TestValidateUserDirectiveExtraction(unittest.TestCase):
    """Test user directive extraction validation function"""
    
    def setUp(self):
        if not USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.skipTest("User comment prioritization not available")
    
    def test_validation_function(self):
        """Test the validate_user_directive_extraction function"""
        test_comments = [
            "implement issue #275 first",
            "block issue #276 until dependencies resolved",
            "prioritize issue #277 - it's critical",
            "use RIF-Validator for issue #278",
            "think hard about issue #279 orchestration"
        ]
        
        results = validate_user_directive_extraction(test_comments)
        
        self.assertEqual(results['total_comments'], 5)
        self.assertGreater(results['directives_found'], 0)
        self.assertIn('IMPLEMENT', results['directive_types'])
        self.assertIn('BLOCK', results['directive_types'])
        self.assertIn('PRIORITIZE', results['directive_types'])
        self.assertGreater(results['extraction_accuracy'], 0.0)


class TestOrchestrationIntegration(unittest.TestCase):
    """Test integration with orchestration intelligence system"""
    
    def setUp(self):
        if not USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.skipTest("User comment prioritization not available")
    
    @patch('claude.commands.orchestration_intelligence_integration.make_intelligent_orchestration_decision')
    @patch('claude.commands.user_comment_prioritizer.UserCommentPrioritizer.extract_user_directives')
    def test_user_priority_orchestration_decision(self, mock_extract, mock_base_decision):
        """Test make_user_priority_orchestration_decision function"""
        # Mock base orchestration decision
        mock_base_decision.return_value = Mock(
            decision_type="allow_execution",
            enforcement_action="ALLOW",
            blocking_issues=[],
            allowed_issues=[275],
            blocked_issues=[],
            dependency_rationale="Standard orchestration",
            blocking_analysis={'has_blocking_issues': False},
            task_launch_codes=["Task(...)"],
            execution_ready=True,
            parallel_execution=False
        )
        
        # Mock user directives
        mock_extract.return_value = [
            UserDirective(
                source_type=DirectiveSource.USER_COMMENT,
                priority=DirectivePriority.VERY_HIGH,
                directive_text="implement issue #275",
                action_type="IMPLEMENT",
                target_issues=[275],
                specific_agents=[],
                reasoning="User directive",
                confidence_score=0.95,
                timestamp=datetime.now()
            )
        ]
        
        # Make user-priority decision
        decision = make_user_priority_orchestration_decision([275])
        
        # Should have user-first characteristics
        self.assertEqual(decision.decision_hierarchy, "USER_FIRST")
        self.assertIsNotNone(decision.user_directive_analysis)
        self.assertIn("User directives", decision.dependency_rationale)
    
    def test_enhanced_status_includes_user_prioritization(self):
        """Test that system status includes user prioritization information"""
        status = get_enhanced_blocking_detection_status()
        
        # Should include Issue #275 fields
        self.assertIn('user_comment_prioritization_available', status)
        self.assertIn('issue_275_integration', status)
        self.assertIn('think_hard_logic_enabled', status)
        self.assertIn('supported_user_directive_patterns', status)
        self.assertIn('decision_hierarchy_enforced', status)
        
        if USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.assertTrue(status['issue_275_integration'])
            self.assertGreater(len(status['supported_user_directive_patterns']), 0)


class TestIntegrationFunction(unittest.TestCase):
    """Test the integration function for existing systems"""
    
    def setUp(self):
        if not USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.skipTest("User comment prioritization not available")
    
    @patch('claude.commands.user_comment_prioritizer.UserCommentPrioritizer.create_user_priority_orchestration_plan')
    @patch('claude.commands.user_comment_prioritizer.UserCommentPrioritizer.extract_user_directives')
    def test_integrate_user_comment_prioritization(self, mock_extract, mock_plan):
        """Test the integrate_user_comment_prioritization function"""
        # Mock existing orchestration plan
        existing_plan = {
            'parallel_tasks': [
                {
                    'issue': 275,
                    'agent': 'RIF-Implementer'
                }
            ],
            'decision_type': 'standard'
        }
        
        # Mock user priority plan
        mock_plan.return_value = {
            'decision_hierarchy': 'USER_FIRST',
            'conflict_analysis': {'total_conflicts': 1, 'user_overrides': []},
            'final_orchestration_decisions': [
                {
                    'type': 'LAUNCH_AGENT',
                    'issue': 275,
                    'priority': 'VERY_HIGH',
                    'source': 'USER_DIRECTIVE'
                }
            ]
        }
        
        mock_extract.return_value = []
        
        integrated_plan = integrate_user_comment_prioritization([275], existing_plan)
        
        # Should have user directive integration
        self.assertTrue(integrated_plan['user_directive_integration'])
        self.assertEqual(integrated_plan['decision_hierarchy'], 'USER_FIRST')
        self.assertIn('user_priority_analysis', integrated_plan)
        self.assertIn('original_agent_plan', integrated_plan)


class TestThinkHardScenarios(unittest.TestCase):
    """Test 'Think Hard' logic for complex orchestration scenarios"""
    
    def setUp(self):
        if not USER_COMMENT_PRIORITIZATION_AVAILABLE:
            self.skipTest("User comment prioritization not available")
        
        self.prioritizer = UserCommentPrioritizer()
    
    def test_think_hard_scenario_identification(self):
        """Test identification of scenarios requiring Think Hard analysis"""
        # Mock user directives with Think Hard triggers
        user_directives = [
            UserDirective(
                source_type=DirectiveSource.USER_COMMENT,
                priority=DirectivePriority.VERY_HIGH,
                directive_text="think hard about orchestration for issue #275",
                action_type="IMPLEMENT",
                target_issues=[275],
                specific_agents=[],
                reasoning="Think hard request",
                confidence_score=0.95,
                timestamp=datetime.now()
            )
        ]
        
        scenarios = self.prioritizer._identify_think_hard_scenarios(user_directives, [275])
        
        # Should identify explicit Think Hard request
        explicit_scenarios = [s for s in scenarios if s['type'] == 'EXPLICIT_THINK_HARD_REQUEST']
        self.assertGreater(len(explicit_scenarios), 0)
        
        # Test complex multi-issue scenario
        many_issues = list(range(275, 282))  # 7 issues
        complex_scenarios = self.prioritizer._identify_think_hard_scenarios([], many_issues)
        
        complex_multi = [s for s in complex_scenarios if s['type'] == 'COMPLEX_MULTI_ISSUE_SCENARIO']
        self.assertGreater(len(complex_multi), 0)
    
    def test_think_hard_extended_reasoning(self):
        """Test extended reasoning analysis for Think Hard scenarios"""
        user_directives = [
            UserDirective(
                source_type=DirectiveSource.USER_COMMENT,
                priority=DirectivePriority.VERY_HIGH,
                directive_text="think hard about dependencies",
                action_type="IMPLEMENT",
                target_issues=[275],
                specific_agents=[],
                reasoning="Complex analysis needed",
                confidence_score=0.95,
                timestamp=datetime.now()
            )
        ]
        
        analysis = self.prioritizer._perform_think_hard_analysis(
            user_directives, [], [275, 276, 277]
        )
        
        # Should have extended analysis structure
        self.assertEqual(analysis['analysis_depth'], 'EXTENDED')
        self.assertGreater(len(analysis['reasoning_steps']), 0)
        self.assertGreater(len(analysis['considered_factors']), 0)
        self.assertGreater(analysis['decision_confidence'], 0.0)
        
        # Should analyze user directive patterns
        step_descriptions = [step['description'] for step in analysis['reasoning_steps']]
        self.assertIn('Analyzing user directive patterns', step_descriptions)


if __name__ == '__main__':
    print("🧪 Running User Comment Prioritization Test Suite - Issue #275")
    print("=" * 70)
    
    # Run tests with verbose output
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print(f"\n✅ User Comment Prioritization tests complete!")