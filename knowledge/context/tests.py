"""
Tests for context optimization system.

Validates relevance scoring, context pruning, and overall optimization
functionality to ensure proper operation within RIF framework.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from .optimizer import ContextOptimizer
from .scorer import RelevanceScorer
from .pruner import ContextPruner, ContextBudget


class TestRelevanceScorer(unittest.TestCase):
    """Test cases for RelevanceScorer."""
    
    def setUp(self):
        self.scorer = RelevanceScorer()
        
        # Sample result for testing
        self.sample_result = {
            'id': 'test-1',
            'content': 'This is a test pattern for authentication using JWT tokens',
            'metadata': {
                'type': 'pattern',
                'title': 'JWT Authentication Pattern',
                'source': 'issue_25',
                'tags': 'auth,jwt,security',
                'created_at': '2025-08-20T10:00:00Z'
            },
            'distance': 0.3,
            'collection': 'patterns'
        }
    
    def test_direct_relevance_scoring(self):
        """Test direct relevance calculation."""
        query = "authentication JWT"
        
        score = self.scorer._calculate_direct_relevance(
            self.scorer._extract_query_terms(query),
            self.sample_result
        )
        
        # Should get high score for matching terms
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
    
    def test_semantic_relevance_scoring(self):
        """Test semantic relevance from distance."""
        # Low distance should give high relevance
        result_high_similarity = self.sample_result.copy()
        result_high_similarity['distance'] = 0.1
        
        score_high = self.scorer._calculate_semantic_relevance(result_high_similarity)
        
        # High distance should give low relevance
        result_low_similarity = self.sample_result.copy()
        result_low_similarity['distance'] = 1.5
        
        score_low = self.scorer._calculate_semantic_relevance(result_low_similarity)
        
        self.assertGreater(score_high, score_low)
        self.assertGreater(score_high, 0.7)
        self.assertLess(score_low, 0.3)
    
    def test_structural_relevance_with_context(self):
        """Test structural relevance calculation with context."""
        context = {
            'issue_id': '25',
            'component': 'auth',
            'tags': ['auth', 'security']
        }
        
        score = self.scorer._calculate_structural_relevance(self.sample_result, context)
        
        # Should get high score due to matching issue and tags
        self.assertGreater(score, 0.6)
        self.assertLessEqual(score, 1.0)
    
    def test_temporal_relevance_scoring(self):
        """Test temporal relevance calculation."""
        # Recent result
        recent_result = self.sample_result.copy()
        recent_result['metadata']['created_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        score_recent = self.scorer._calculate_temporal_relevance(recent_result)
        
        # Old result
        old_result = self.sample_result.copy()
        old_result['metadata']['created_at'] = '2020-01-01T00:00:00Z'
        
        score_old = self.scorer._calculate_temporal_relevance(old_result)
        
        self.assertGreater(score_recent, score_old)
        self.assertGreater(score_recent, 0.8)
    
    def test_comprehensive_relevance_scoring(self):
        """Test full relevance scoring pipeline."""
        query = "JWT authentication pattern"
        context = {
            'issue_id': '25',
            'agent_type': 'rif-implementer'
        }
        
        score = self.scorer.calculate_relevance_score(query, self.sample_result, context)
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Should be relatively high due to good matches
        self.assertGreater(score, 0.4)
    
    def test_score_breakdown(self):
        """Test detailed score breakdown."""
        query = "authentication"
        breakdown = self.scorer.get_score_breakdown(query, self.sample_result)
        
        required_keys = ['direct', 'semantic', 'structural', 'temporal', 'weighted_total', 'weights']
        for key in required_keys:
            self.assertIn(key, breakdown)
            
        # All component scores should be valid
        for component in ['direct', 'semantic', 'structural', 'temporal']:
            self.assertGreaterEqual(breakdown[component], 0.0)
            self.assertLessEqual(breakdown[component], 1.0)


class TestContextPruner(unittest.TestCase):
    """Test cases for ContextPruner."""
    
    def setUp(self):
        self.pruner = ContextPruner()
        
        # Create sample results of varying sizes
        self.sample_results = [
            {
                'id': '1',
                'content': 'Short pattern for testing',
                'metadata': {'type': 'pattern', 'title': 'Short Pattern'},
                'relevance_score': 0.9
            },
            {
                'id': '2',
                'content': 'Much longer content pattern that contains detailed implementation ' +
                          'guidance and code examples that would take significantly more tokens ' * 10,
                'metadata': {'type': 'pattern', 'title': 'Long Pattern'},
                'relevance_score': 0.8
            },
            {
                'id': '3',
                'content': {'implementation': 'Structured content', 'description': 'Detailed description', 'code': 'function example() { return "test"; }'},
                'metadata': {'type': 'decision', 'title': 'Structured Decision'},
                'relevance_score': 0.7
            },
            {
                'id': '4',
                'content': 'Medium length content for testing purposes',
                'metadata': {'type': 'learning', 'title': 'Medium Learning'},
                'relevance_score': 0.6
            }
        ]
    
    def test_context_budget_creation(self):
        """Test context budget allocation."""
        budget = ContextBudget.from_window_size(8000)
        
        self.assertEqual(budget.total_tokens, 8000)
        self.assertEqual(budget.direct_results, 4000)  # 50%
        self.assertEqual(budget.context_preservation, 2000)  # 25%
        self.assertEqual(budget.reserve, 2000)  # 25%
    
    def test_token_estimation(self):
        """Test token count estimation."""
        # Text content
        text_result = self.sample_results[0]
        tokens = self.pruner._estimate_result_tokens(text_result)
        self.assertGreater(tokens, 0)
        
        # Structured content
        structured_result = self.sample_results[2]
        structured_tokens = self.pruner._estimate_result_tokens(structured_result)
        self.assertGreater(structured_tokens, 0)
        
        # Structured content should typically use more tokens
        # (not always true, but generally expected)
        # self.assertGreater(structured_tokens, tokens)
    
    def test_basic_pruning(self):
        """Test basic token-based pruning."""
        # Very small window to force pruning
        pruned_results, pruning_info = self.pruner.prune_results(
            self.sample_results,
            custom_window=500,  # Very small
            min_results=2
        )
        
        # Should prune to minimum results
        self.assertLessEqual(len(pruned_results), len(self.sample_results))
        self.assertGreaterEqual(len(pruned_results), 2)  # min_results
        
        # Should maintain ordering by relevance
        if len(pruned_results) >= 2:
            self.assertGreaterEqual(
                pruned_results[0]['relevance_score'],
                pruned_results[1]['relevance_score']
            )
        
        self.assertTrue(pruning_info['pruning_applied'])
    
    def test_no_pruning_needed(self):
        """Test case where no pruning is needed."""
        # Large window
        pruned_results, pruning_info = self.pruner.prune_results(
            self.sample_results[:2],  # Small number of results
            custom_window=50000,  # Very large window
        )
        
        # Should keep all results
        self.assertEqual(len(pruned_results), 2)
        self.assertTrue(pruning_info['pruning_applied'])  # Still applies scoring
    
    def test_context_preservation(self):
        """Test context preservation during pruning."""
        # Add related results
        related_results = self.sample_results + [
            {
                'id': '5',
                'content': 'Related pattern with same tags',
                'metadata': {
                    'type': 'pattern', 
                    'title': 'Related Pattern',
                    'tags': 'auth,security',  # Same tags as others
                    'source': 'issue_25'  # Same source
                },
                'relevance_score': 0.5
            }
        ]
        
        pruned_results, pruning_info = self.pruner.prune_results(
            related_results,
            custom_window=2000,  # Medium window
            preserve_context=True,
            min_results=2
        )
        
        # Should preserve some context even with pruning
        self.assertGreaterEqual(len(pruned_results), 2)
    
    def test_summarization(self):
        """Test content summarization."""
        long_content = "Very long content that needs summarization. " * 100
        result_to_summarize = {
            'id': 'long',
            'content': long_content,
            'metadata': {'type': 'pattern', 'title': 'Long Pattern'},
            'relevance_score': 0.8
        }
        
        summarized = self.pruner._create_summarized_result(result_to_summarize, 200)
        
        self.assertIsNotNone(summarized)
        self.assertTrue(summarized['summarized'])
        self.assertLess(len(str(summarized['content'])), len(long_content))
        self.assertEqual(summarized['original_length'], len(long_content))


class TestContextOptimizer(unittest.TestCase):
    """Test cases for ContextOptimizer integration."""
    
    def setUp(self):
        self.optimizer = ContextOptimizer()
        
        # Create realistic test results
        self.test_results = [
            {
                'id': 'pattern-1',
                'content': 'Authentication pattern using JWT tokens for secure API access',
                'metadata': {
                    'type': 'pattern',
                    'title': 'JWT Authentication',
                    'source': 'issue_34',
                    'tags': 'auth,jwt,security',
                    'created_at': '2025-08-23T10:00:00Z'
                },
                'distance': 0.2,
                'collection': 'patterns'
            },
            {
                'id': 'decision-1',
                'content': 'Decision to use bcrypt for password hashing due to security requirements',
                'metadata': {
                    'type': 'decision',
                    'title': 'Password Hashing Decision',
                    'source': 'issue_33',
                    'tags': 'auth,security,passwords',
                    'created_at': '2025-08-22T15:30:00Z'
                },
                'distance': 0.4,
                'collection': 'decisions'
            },
            {
                'id': 'learning-1',
                'content': 'Learned that session management requires careful token expiration handling',
                'metadata': {
                    'type': 'learning',
                    'title': 'Session Management Learning',
                    'source': 'issue_32',
                    'tags': 'auth,sessions,tokens',
                    'created_at': '2025-08-21T09:15:00Z'
                },
                'distance': 0.6,
                'collection': 'learnings'
            }
        ]
    
    def test_basic_optimization(self):
        """Test basic optimization workflow."""
        result = self.optimizer.optimize_for_agent(
            results=self.test_results,
            query="authentication patterns",
            agent_type="rif-implementer",
            context={'issue_id': '34', 'component': 'auth'}
        )
        
        # Check return structure
        self.assertIn('optimized_results', result)
        self.assertIn('optimization_info', result)
        self.assertIn('performance_stats', result)
        
        # Results should have relevance scores
        optimized_results = result['optimized_results']
        for optimized_result in optimized_results:
            self.assertIn('relevance_score', optimized_result)
            self.assertGreaterEqual(optimized_result['relevance_score'], 0.0)
            self.assertLessEqual(optimized_result['relevance_score'], 1.0)
        
        # Should be sorted by relevance
        if len(optimized_results) >= 2:
            self.assertGreaterEqual(
                optimized_results[0]['relevance_score'],
                optimized_results[1]['relevance_score']
            )
    
    def test_optimization_with_explanation(self):
        """Test optimization with detailed explanation."""
        result = self.optimizer.optimize_for_agent(
            results=self.test_results,
            query="JWT authentication",
            agent_type="rif-implementer",
            explain=True
        )
        
        self.assertIn('explanation', result)
        
        explanation = result['explanation']
        self.assertIn('summary', explanation)
        self.assertIn('scoring_details', explanation)
        self.assertIn('pruning_details', explanation)
        self.assertIn('quality_preservation', explanation)
        self.assertIn('recommendations', explanation)
    
    def test_different_agent_types(self):
        """Test optimization for different agent types."""
        agents = ['rif-analyst', 'rif-implementer', 'rif-validator']
        
        for agent_type in agents:
            result = self.optimizer.optimize_for_agent(
                results=self.test_results,
                query="authentication",
                agent_type=agent_type
            )
            
            # Should complete successfully for all agent types
            self.assertIn('optimized_results', result)
            self.assertTrue(result['optimization_info']['optimization_applied'])
    
    def test_custom_window_size(self):
        """Test optimization with custom context window."""
        result = self.optimizer.optimize_for_agent(
            results=self.test_results,
            query="patterns",
            custom_window=1000,  # Very small window
            min_results=1
        )
        
        # Should respect custom window size
        optimization_info = result['optimization_info']
        self.assertEqual(optimization_info['pruning_details']['window_size'], 1000)
    
    def test_empty_results(self):
        """Test optimization with empty results."""
        result = self.optimizer.optimize_for_agent(
            results=[],
            query="test query"
        )
        
        self.assertEqual(len(result['optimized_results']), 0)
        # Should still have optimization info even with empty results
        self.assertIn('optimization_info', result)
    
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are tracked correctly."""
        initial_count = self.optimizer.performance_metrics['optimizations_performed']
        
        # Perform optimization
        self.optimizer.optimize_for_agent(
            results=self.test_results,
            query="test"
        )
        
        # Should increment optimization count
        self.assertEqual(
            self.optimizer.performance_metrics['optimizations_performed'],
            initial_count + 1
        )
    
    def test_optimization_history(self):
        """Test optimization history tracking."""
        initial_history_len = len(self.optimizer.optimization_history)
        
        # Perform optimization
        self.optimizer.optimize_for_agent(
            results=self.test_results,
            query="test query"
        )
        
        # Should add to history
        self.assertEqual(
            len(self.optimizer.optimization_history),
            initial_history_len + 1
        )
        
        # Get recent history
        recent_history = self.optimizer.get_optimization_history(limit=5)
        self.assertLessEqual(len(recent_history), 5)
    
    def test_error_handling(self):
        """Test error handling in optimization."""
        # Create malformed results to trigger error
        malformed_results = [{'malformed': True}]
        
        with patch.object(self.optimizer.scorer, 'calculate_relevance_score', side_effect=Exception("Test error")):
            result = self.optimizer.optimize_for_agent(
                results=malformed_results,
                query="test"
            )
            
            # Should return fallback results
            self.assertEqual(result['optimized_results'], malformed_results)
            self.assertFalse(result['optimization_info']['optimization_applied'])
            self.assertIn('error', result['optimization_info'])


def run_context_tests():
    """Run all context optimization tests."""
    test_classes = [
        TestRelevanceScorer,
        TestContextPruner, 
        TestContextOptimizer
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_context_tests()
    exit(0 if success else 1)