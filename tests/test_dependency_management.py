#!/usr/bin/env python3
"""
Comprehensive Unit Tests for RIF Dependency Management System

Tests all components of the dependency management system:
- DependencyParser: Pattern matching and parsing
- DependencyChecker: GitHub API integration and caching  
- DependencyManager: High-level orchestration interface

This ensures Issue #143 implementation is robust and reliable.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys
import yaml

# Add the project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from claude.commands.dependency_manager import (
    DependencyParser, 
    DependencyChecker, 
    DependencyManager, 
    Dependency,
    DependencyCheckResult
)


class TestDependencyParser(unittest.TestCase):
    """Test the DependencyParser class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary config file for testing
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        test_patterns = {
            'dependency_patterns': {
                'depends_on': [
                    {'pattern': r'depends on:?\s*#?(\d+)', 'confidence': 0.9}
                ],
                'blocked_by': [
                    {'pattern': r'blocked by:?\s*#?(\d+)', 'confidence': 0.95}
                ],
                'implementation_dependencies': [
                    {'pattern': r'Implementation Dependencies:.*?Issue #(\d+)', 'confidence': 0.95}
                ]
            }
        }
        yaml.dump(test_patterns, self.temp_config)
        self.temp_config.close()
        
        self.parser = DependencyParser(self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_config.name)
    
    def test_parse_simple_dependency(self):
        """Test parsing simple dependency declarations"""
        issue_body = """
        This feature depends on #123 being completed first.
        We also need to wait for the API changes.
        """
        
        dependencies = self.parser.parse_dependencies(issue_body, 100)
        
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].issue_number, 123)
        self.assertEqual(dependencies[0].dependency_type, 'depends_on')
        self.assertEqual(dependencies[0].confidence, 0.9)
    
    def test_parse_blocked_by_dependency(self):
        """Test parsing blocked by declarations"""
        issue_body = """
        This is blocked by #456 due to API compatibility.
        """
        
        dependencies = self.parser.parse_dependencies(issue_body, 100)
        
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].issue_number, 456)
        self.assertEqual(dependencies[0].dependency_type, 'blocked_by')
        self.assertEqual(dependencies[0].confidence, 0.95)
    
    def test_parse_implementation_dependencies(self):
        """Test parsing DPIBS-style implementation dependencies"""
        issue_body = """
        # DPIBS Phase 2: Advanced Analysis
        
        ## Implementation Dependencies
        Implementation Dependencies: Issue #129
        
        This phase requires the completion of basic analysis.
        """
        
        dependencies = self.parser.parse_dependencies(issue_body, 130)
        
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].issue_number, 129)
        self.assertEqual(dependencies[0].dependency_type, 'implementation_dependencies')
        self.assertEqual(dependencies[0].confidence, 0.95)
    
    def test_parse_multiple_dependencies(self):
        """Test parsing multiple dependencies in one issue"""
        issue_body = """
        This feature depends on #123 and is blocked by #456.
        Implementation Dependencies: Issue #789
        """
        
        dependencies = self.parser.parse_dependencies(issue_body, 100)
        
        self.assertEqual(len(dependencies), 3)
        
        issue_numbers = [dep.issue_number for dep in dependencies]
        self.assertIn(123, issue_numbers)
        self.assertIn(456, issue_numbers)
        self.assertIn(789, issue_numbers)
    
    def test_parse_no_dependencies(self):
        """Test parsing issue with no dependencies"""
        issue_body = """
        This is a standalone feature with no dependencies.
        It can be implemented independently.
        """
        
        dependencies = self.parser.parse_dependencies(issue_body, 100)
        
        self.assertEqual(len(dependencies), 0)
    
    def test_parse_empty_issue_body(self):
        """Test parsing empty or None issue body"""
        dependencies1 = self.parser.parse_dependencies("", 100)
        dependencies2 = self.parser.parse_dependencies(None, 100)
        
        self.assertEqual(len(dependencies1), 0)
        self.assertEqual(len(dependencies2), 0)
    
    def test_duplicate_dependencies_removed(self):
        """Test that duplicate dependencies are removed, keeping highest confidence"""
        issue_body = """
        depends on #123
        dependency: #123
        """
        
        # Mock the patterns to have different confidence levels
        with patch.object(self.parser, 'patterns', {
            'depends_on': [{'pattern': r'depends on:?\s*#?(\d+)', 'confidence': 0.9}],
            'dependency': [{'pattern': r'dependency:?\s*#?(\d+)', 'confidence': 0.8}]
        }):
            dependencies = self.parser.parse_dependencies(issue_body, 100)
        
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].confidence, 0.9)  # Should keep higher confidence
    
    def test_validate_patterns(self):
        """Test pattern validation"""
        results = self.parser.validate_patterns()
        
        # All patterns should be valid
        self.assertTrue(all(results.values()))
    
    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns"""
        # Create parser with invalid pattern
        with patch.object(self.parser, 'patterns', {
            'invalid': [{'pattern': r'[invalid regex(', 'confidence': 0.9}]
        }):
            results = self.parser.validate_patterns()
            self.assertIn('invalid_0', results)
            self.assertFalse(results['invalid_0'])


class TestDependencyChecker(unittest.TestCase):
    """Test the DependencyChecker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = DependencyChecker(cache_timeout_minutes=1)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.checker.clear_cache()
    
    @patch('subprocess.run')
    def test_check_satisfied_dependency_closed_issue(self, mock_run):
        """Test checking dependency on closed issue"""
        # Mock GitHub response for closed issue
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'number': 123,
            'title': 'Test Issue',
            'state': 'CLOSED',
            'labels': [],
            'closedAt': '2023-01-01T00:00:00Z'
        })
        mock_run.return_value = mock_result
        
        dependency = Dependency(
            issue_number=123,
            dependency_type='depends_on',
            source_line='depends on #123',
            confidence=0.9
        )
        
        result = self.checker.check_dependencies(100, [dependency])
        
        self.assertTrue(result.can_proceed)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(len(result.blocking_dependencies), 0)
    
    @patch('subprocess.run')
    def test_check_satisfied_dependency_complete_label(self, mock_run):
        """Test checking dependency on issue with state:complete label"""
        # Mock GitHub response for issue with state:complete label
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'number': 123,
            'title': 'Test Issue',
            'state': 'OPEN',
            'labels': [{'name': 'state:complete'}],
            'closedAt': None
        })
        mock_run.return_value = mock_result
        
        dependency = Dependency(
            issue_number=123,
            dependency_type='depends_on',
            source_line='depends on #123',
            confidence=0.9
        )
        
        result = self.checker.check_dependencies(100, [dependency])
        
        self.assertTrue(result.can_proceed)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(len(result.blocking_dependencies), 0)
    
    @patch('subprocess.run')
    def test_check_unsatisfied_dependency(self, mock_run):
        """Test checking dependency on open, incomplete issue"""
        # Mock GitHub response for open issue without complete label
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'number': 123,
            'title': 'Test Issue',
            'state': 'OPEN',
            'labels': [{'name': 'state:implementing'}],
            'closedAt': None
        })
        mock_run.return_value = mock_result
        
        dependency = Dependency(
            issue_number=123,
            dependency_type='depends_on',
            source_line='depends on #123',
            confidence=0.9
        )
        
        result = self.checker.check_dependencies(100, [dependency])
        
        self.assertFalse(result.can_proceed)
        self.assertEqual(len(result.satisfied_dependencies), 0)
        self.assertEqual(len(result.blocking_dependencies), 1)
        self.assertIn("123", result.reason)
    
    @patch('subprocess.run')
    def test_check_multiple_mixed_dependencies(self, mock_run):
        """Test checking multiple dependencies with mixed satisfaction"""
        # Mock different responses for different issues
        def mock_run_side_effect(cmd, **kwargs):
            if '#123' in cmd:
                # Satisfied dependency (closed)
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    'number': 123,
                    'state': 'CLOSED',
                    'labels': []
                })
                return mock_result
            elif '#456' in cmd:
                # Unsatisfied dependency (open)
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    'number': 456,
                    'state': 'OPEN',
                    'labels': [{'name': 'state:implementing'}]
                })
                return mock_result
        
        mock_run.side_effect = mock_run_side_effect
        
        dependencies = [
            Dependency(123, 'depends_on', 'depends on #123', 0.9),
            Dependency(456, 'blocked_by', 'blocked by #456', 0.95)
        ]
        
        result = self.checker.check_dependencies(100, dependencies)
        
        self.assertFalse(result.can_proceed)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(len(result.blocking_dependencies), 1)
    
    @patch('subprocess.run')
    def test_github_api_error_handling(self, mock_run):
        """Test handling of GitHub API errors"""
        # Mock GitHub API failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "API rate limit exceeded"
        mock_run.return_value = mock_result
        
        dependency = Dependency(
            issue_number=123,
            dependency_type='depends_on',
            source_line='depends on #123',
            confidence=0.9
        )
        
        result = self.checker.check_dependencies(100, [dependency])
        
        # Should treat API error as unsatisfied dependency (conservative approach)
        self.assertFalse(result.can_proceed)
        self.assertEqual(len(result.blocking_dependencies), 1)
    
    def test_caching_behavior(self):
        """Test that issue data is properly cached"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                'number': 123,
                'state': 'CLOSED',
                'labels': []
            })
            mock_run.return_value = mock_result
            
            # First call should hit API
            self.checker._get_issue_data(123)
            self.assertEqual(mock_run.call_count, 1)
            
            # Second call should use cache
            self.checker._get_issue_data(123)
            self.assertEqual(mock_run.call_count, 1)  # Should not increase
    
    def test_cache_expiry(self):
        """Test that cache expires after timeout"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({'number': 123, 'state': 'CLOSED', 'labels': []})
            mock_run.return_value = mock_result
            
            # Set very short cache timeout
            self.checker.cache_timeout = 0.001  # 1ms
            
            # First call
            self.checker._get_issue_data(123)
            self.assertEqual(mock_run.call_count, 1)
            
            # Wait for cache to expire
            import time
            time.sleep(0.002)
            
            # Second call should hit API again
            self.checker._get_issue_data(123)
            self.assertEqual(mock_run.call_count, 2)
    
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        # Add some data to cache
        self.checker.issue_cache[123] = ({'test': 'data'}, datetime.now().timestamp())
        
        self.assertEqual(len(self.checker.issue_cache), 1)
        
        self.checker.clear_cache()
        
        self.assertEqual(len(self.checker.issue_cache), 0)
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Add some data to cache
        self.checker.issue_cache[123] = ({'test': 'data'}, datetime.now().timestamp())
        
        stats = self.checker.get_cache_stats()
        
        self.assertEqual(stats['cache_size'], 1)
        self.assertIn('cache_timeout_seconds', stats)


class TestDependencyManager(unittest.TestCase):
    """Test the DependencyManager high-level interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = DependencyManager()
    
    @patch('subprocess.run')
    def test_can_work_on_issue_no_dependencies(self, mock_run):
        """Test issue with no dependencies can proceed"""
        # Mock GitHub response for issue data
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'number': 100,
            'title': 'Test Issue',
            'body': 'Simple feature with no dependencies',
            'state': 'OPEN',
            'labels': []
        })
        mock_run.return_value = mock_result
        
        can_proceed, reason, result = self.manager.can_work_on_issue(100)
        
        self.assertTrue(can_proceed)
        self.assertIsNone(reason)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.blocking_dependencies), 0)
    
    @patch('subprocess.run')
    def test_can_work_on_issue_with_satisfied_dependencies(self, mock_run):
        """Test issue with satisfied dependencies can proceed"""
        def mock_run_side_effect(cmd, **kwargs):
            if 'issue view 100' in cmd:
                # Main issue with dependency
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 100,
                        'title': 'Test Issue',
                        'body': 'This depends on #123',
                        'state': 'OPEN',
                        'labels': []
                    })
                )
            elif 'issue view 123' in cmd:
                # Dependency is satisfied (closed)
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 123,
                        'state': 'CLOSED',
                        'labels': []
                    })
                )
        
        mock_run.side_effect = mock_run_side_effect
        
        can_proceed, reason, result = self.manager.can_work_on_issue(100)
        
        self.assertTrue(can_proceed)
        self.assertIsNone(reason)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(len(result.blocking_dependencies), 0)
    
    @patch('subprocess.run')
    def test_can_work_on_issue_with_unsatisfied_dependencies(self, mock_run):
        """Test issue with unsatisfied dependencies cannot proceed"""
        def mock_run_side_effect(cmd, **kwargs):
            if 'issue view 100' in cmd:
                # Main issue with dependency
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 100,
                        'title': 'Test Issue',  
                        'body': 'This depends on #123',
                        'state': 'OPEN',
                        'labels': []
                    })
                )
            elif 'issue view 123' in cmd:
                # Dependency is not satisfied (still open)
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 123,
                        'state': 'OPEN',
                        'labels': [{'name': 'state:implementing'}]
                    })
                )
        
        mock_run.side_effect = mock_run_side_effect
        
        can_proceed, reason, result = self.manager.can_work_on_issue(100)
        
        self.assertFalse(can_proceed)
        self.assertIn("123", reason)
        self.assertEqual(len(result.satisfied_dependencies), 0)
        self.assertEqual(len(result.blocking_dependencies), 1)
    
    @patch('subprocess.run')
    def test_add_blocked_label(self, mock_run):
        """Test adding blocked label to issue"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = self.manager.add_blocked_label(100, "Blocked by issue #123")
        
        self.assertTrue(success)
        
        # Verify the correct commands were called
        self.assertEqual(mock_run.call_count, 2)  # Label command + comment command
        
        # Check that label command was called
        label_call = mock_run.call_args_list[0]
        self.assertIn('--add-label', label_call[0][0])
        self.assertIn('blocked', label_call[0][0])
    
    @patch('subprocess.run')
    def test_remove_blocked_label(self, mock_run):
        """Test removing blocked label from issue"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = self.manager.remove_blocked_label(100)
        
        self.assertTrue(success)
        
        # Verify the correct commands were called
        self.assertEqual(mock_run.call_count, 2)  # Label command + comment command
        
        # Check that remove label command was called
        label_call = mock_run.call_args_list[0]
        self.assertIn('--remove-label', label_call[0][0])
        self.assertIn('blocked', label_call[0][0])
    
    @patch('subprocess.run')
    def test_get_blocked_issues(self, mock_run):
        """Test getting list of blocked issues"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps([
            {'number': 100},
            {'number': 101},
            {'number': 102}
        ])
        mock_run.return_value = mock_result
        
        blocked_issues = self.manager.get_blocked_issues()
        
        self.assertEqual(len(blocked_issues), 3)
        self.assertIn(100, blocked_issues)
        self.assertIn(101, blocked_issues)
        self.assertIn(102, blocked_issues)
    
    @patch.object(DependencyManager, 'can_work_on_issue')
    @patch.object(DependencyManager, 'get_blocked_issues')
    @patch.object(DependencyManager, 'remove_blocked_label')
    def test_check_blocked_issues_for_unblocking(self, mock_remove_label, mock_get_blocked, mock_can_work):
        """Test checking blocked issues for potential unblocking"""
        # Mock blocked issues
        mock_get_blocked.return_value = [100, 101]
        
        # Mock dependency check results
        def mock_can_work_side_effect(issue_number):
            if issue_number == 100:
                # This issue can now proceed
                return True, None, MagicMock(can_proceed=True)
            else:
                # This issue still blocked
                return False, "Still blocked", MagicMock(can_proceed=False)
        
        mock_can_work.side_effect = mock_can_work_side_effect
        mock_remove_label.return_value = True
        
        unblocked = self.manager.check_blocked_issues_for_unblocking()
        
        self.assertEqual(len(unblocked), 1)
        self.assertIn(100, unblocked)
        
        # Verify remove_blocked_label was called for unblocked issue
        mock_remove_label.assert_called_once_with(100)
    
    def test_performance_metrics(self):
        """Test getting performance metrics"""
        metrics = self.manager.get_performance_metrics()
        
        self.assertIn('parser_patterns_count', metrics)
        self.assertIn('checker_cache_stats', metrics)
        self.assertIn('patterns_valid', metrics)
        
        # Verify metrics are reasonable
        self.assertGreater(metrics['parser_patterns_count'], 0)
        self.assertIsInstance(metrics['patterns_valid'], bool)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.manager = DependencyManager()
    
    @patch('subprocess.run')
    def test_dpibs_scenario_blocked(self, mock_run):
        """Test DPIBS scenario where Phase 2 is blocked by Phase 1"""
        def mock_run_side_effect(cmd, **kwargs):
            if 'issue view 130' in cmd:
                # Phase 2 issue with dependency on Phase 1
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 130,
                        'title': 'DPIBS Phase 2: Advanced Analysis',
                        'body': '''# DPIBS Phase 2: Advanced Analysis
                        
Implementation Dependencies: Issue #129

This phase builds upon the basic analysis infrastructure.''',
                        'state': 'OPEN',
                        'labels': [{'name': 'state:new'}]
                    })
                )
            elif 'issue view 129' in cmd:
                # Phase 1 is still implementing
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 129,
                        'state': 'OPEN',
                        'labels': [{'name': 'state:implementing'}]
                    })
                )
        
        mock_run.side_effect = mock_run_side_effect
        
        can_proceed, reason, result = self.manager.can_work_on_issue(130)
        
        self.assertFalse(can_proceed)
        self.assertIn("129", reason)
        self.assertEqual(len(result.blocking_dependencies), 1)
        self.assertEqual(result.blocking_dependencies[0].dependency_type, 'implementation_dependencies')
    
    @patch('subprocess.run')
    def test_dpibs_scenario_unblocked(self, mock_run):
        """Test DPIBS scenario where Phase 2 can proceed after Phase 1 completes"""
        def mock_run_side_effect(cmd, **kwargs):
            if 'issue view 130' in cmd:
                # Phase 2 issue with dependency on Phase 1
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 130,
                        'title': 'DPIBS Phase 2: Advanced Analysis',
                        'body': '''# DPIBS Phase 2: Advanced Analysis
                        
Implementation Dependencies: Issue #129

This phase builds upon the basic analysis infrastructure.''',
                        'state': 'OPEN',
                        'labels': [{'name': 'state:new'}]
                    })
                )
            elif 'issue view 129' in cmd:
                # Phase 1 is now complete
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 129,
                        'state': 'OPEN',
                        'labels': [{'name': 'state:complete'}]
                    })
                )
        
        mock_run.side_effect = mock_run_side_effect
        
        can_proceed, reason, result = self.manager.can_work_on_issue(130)
        
        self.assertTrue(can_proceed)
        self.assertIsNone(reason)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(len(result.blocking_dependencies), 0)
    
    @patch('subprocess.run')
    def test_complex_dependency_chain(self, mock_run):
        """Test complex dependency chain: A depends on B, B depends on C"""
        def mock_run_side_effect(cmd, **kwargs):
            if 'issue view 300' in cmd:
                # Issue A depends on B
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 300,
                        'title': 'Feature A',
                        'body': 'depends on #200',
                        'state': 'OPEN',
                        'labels': []
                    })
                )
            elif 'issue view 200' in cmd:
                # Issue B is complete
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        'number': 200,
                        'state': 'CLOSED'
                    })
                )
        
        mock_run.side_effect = mock_run_side_effect
        
        can_proceed, reason, result = self.manager.can_work_on_issue(300)
        
        self.assertTrue(can_proceed)
        self.assertEqual(len(result.satisfied_dependencies), 1)
        self.assertEqual(result.satisfied_dependencies[0].issue_number, 200)
    
    def test_performance_under_load(self):
        """Test performance with many dependency patterns"""
        # Create issue body with many different dependency formats
        issue_body = """
        depends on #100
        blocked by #101
        after #102
        requires #103
        prerequisite #104
        Implementation Dependencies: Issue #105
        child of #106
        builds on #107
        needs #108
        waiting for #109
        """
        
        start_time = datetime.now()
        
        dependencies = self.manager.parser.parse_dependencies(issue_body, 999)
        
        end_time = datetime.now()
        parsing_time = (end_time - start_time).total_seconds() * 1000  # milliseconds
        
        # Should parse quickly (under 200ms as per requirements)
        self.assertLess(parsing_time, 200)
        
        # Should find all 10 dependencies
        self.assertEqual(len(dependencies), 10)
        
        # All should be unique issue numbers
        issue_numbers = {dep.issue_number for dep in dependencies}
        self.assertEqual(len(issue_numbers), 10)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)