#!/usr/bin/env python3
"""
Test Suite for Automated PR Creation System - Issue #205
Comprehensive tests for all components of the automated PR creation functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.github_state_manager import GitHubStateManager
from claude.commands.pr_template_aggregator import PRTemplateAggregator, PRContext, CheckpointData
from claude.commands.pr_creation_service import PRCreationService
from claude.commands.state_machine_hooks import StateMachineHooks

class TestGitHubStateManagerPRExtensions(unittest.TestCase):
    """Test the PR-related extensions to GitHubStateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = GitHubStateManager()
        self.test_issue_number = 205
        
    def test_generate_branch_name(self):
        """Test branch name generation."""
        # Test normal case
        branch_name = self.manager.generate_branch_name(205, "Implement automated PR creation")
        self.assertEqual(branch_name, "issue-205-implement-automated-pr-creation")
        
        # Test special characters
        branch_name = self.manager.generate_branch_name(123, "Fix bug with @special #characters!")
        self.assertEqual(branch_name, "issue-123-fix-bug-with-special-characters")
        
        # Test long title truncation
        long_title = "This is a very long title that should be truncated to prevent overly long branch names that are hard to work with"
        branch_name = self.manager.generate_branch_name(456, long_title)
        self.assertLessEqual(len(branch_name), 70)  # Should be truncated
        self.assertTrue(branch_name.startswith("issue-456-"))
        
    @patch('subprocess.run')
    def test_get_file_modifications(self, mock_run):
        """Test file modification detection."""
        # Mock git diff output
        mock_run.return_value.stdout = "A\tfile1.py\nM\tfile2.py\nD\tfile3.py\nR100\tfile4.py\tfile4_new.py"
        mock_run.return_value.check = True
        
        modifications = self.manager.get_file_modifications("test-branch")
        
        self.assertEqual(modifications['added'], ['file1.py'])
        self.assertEqual(modifications['modified'], ['file2.py', 'file4_new.py'])  # Renamed files treated as modified
        self.assertEqual(modifications['deleted'], ['file3.py'])
        
        # Verify git command was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[:3], ['git', 'diff', '--name-status'])
        
    @patch('subprocess.run')
    def test_get_issue_metadata(self, mock_run):
        """Test issue metadata retrieval."""
        # Mock GitHub CLI response
        mock_response = {
            "title": "Test issue title",
            "body": "Test issue body",
            "labels": [{"name": "enhancement"}, {"name": "rif-managed"}],
            "assignees": [{"login": "test-user"}],
            "state": "open",
            "author": {"login": "issue-author"}
        }
        mock_run.return_value.stdout = json.dumps(mock_response)
        
        metadata = self.manager.get_issue_metadata(205)
        
        self.assertEqual(metadata['title'], "Test issue title")
        self.assertEqual(metadata['labels'], ['enhancement', 'rif-managed'])
        self.assertEqual(metadata['assignees'], ['test-user'])
        self.assertEqual(metadata['author'], 'issue-author')


class TestPRTemplateAggregator(unittest.TestCase):
    """Test the PR template aggregator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = PRTemplateAggregator()
        self.test_checkpoints = [
            CheckpointData(
                checkpoint_id="test-checkpoint-1",
                phase="Phase 1: Implementation",
                status="complete",
                description="Test implementation complete",
                timestamp="2025-01-24T18:00:00Z",
                components={"test_component": {"status": "complete", "description": "Test component"}}
            )
        ]
        
    def test_load_template_default(self):
        """Test loading default template when file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            template = self.aggregator.load_template()
            self.assertIn("# Pull Request", template)
            self.assertIn("{summary}", template)
            self.assertIn("Closes #{issue_number}", template)
            
    def test_generate_implementation_summary(self):
        """Test implementation summary generation."""
        summary = self.aggregator.generate_implementation_summary(self.test_checkpoints)
        
        self.assertIn("Completed 1 implementation phases", summary)
        self.assertIn("Phase 1: Implementation", summary)
        self.assertIn("Test implementation complete", summary)
        
    def test_generate_changes_summary(self):
        """Test changes summary generation."""
        file_mods = {
            'added': ['file1.py', 'file2.py'],
            'modified': ['file3.py'],
            'deleted': []
        }
        
        summary = self.aggregator.generate_changes_summary(file_mods, self.test_checkpoints)
        
        self.assertIn("Total files affected: 3", summary)
        self.assertIn("Added: 2 files", summary)
        self.assertIn("Modified: 1 files", summary)
        self.assertIn("test_component", summary)
        
    def test_determine_pr_type(self):
        """Test PR type determination."""
        # Test bug fix detection
        issue_metadata = {'title': 'Fix critical bug', 'labels': ['bug']}
        pr_type = self.aggregator.determine_pr_type(issue_metadata, [])
        self.assertEqual(pr_type, '🐛 Bug fix')
        
        # Test feature detection
        issue_metadata = {'title': 'Implement new feature', 'labels': ['enhancement']}
        pr_type = self.aggregator.determine_pr_type(issue_metadata, [])
        self.assertEqual(pr_type, '✨ New feature')
        
    def test_populate_template(self):
        """Test template population."""
        pr_context = PRContext(
            issue_number=205,
            issue_metadata={'title': 'Test issue', 'labels': ['enhancement']},
            checkpoints=self.test_checkpoints,
            file_modifications={'added': ['test.py'], 'modified': [], 'deleted': []},
            quality_results={'overall_status': 'ready'},
            implementation_summary="Test implementation"
        )
        
        populated = self.aggregator.populate_template(pr_context)
        
        self.assertIn("205", populated)
        self.assertIn("Test implementation", populated)
        self.assertIn("test.py", populated)


class TestPRCreationService(unittest.TestCase):
    """Test the PR creation service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = PRCreationService()
        
    def test_should_create_pr(self):
        """Test PR creation trigger detection."""
        # Test primary trigger
        should_create = self.service.should_create_pr(205, 'implementing', 'validating')
        self.assertTrue(should_create)
        
        # Test explicit trigger
        should_create = self.service.should_create_pr(205, 'analyzing', 'pr_creating')
        self.assertTrue(should_create)
        
        # Test non-trigger
        should_create = self.service.should_create_pr(205, 'new', 'analyzing')
        self.assertFalse(should_create)
        
    def test_check_quality_gates_no_evidence(self):
        """Test quality gate checking with no evidence file."""
        with patch.object(Path, 'exists', return_value=False):
            results = self.service.check_quality_gates(205)
            
            self.assertEqual(results['overall_status'], 'pending')
            self.assertTrue(results['draft_pr'])
            
    def test_check_quality_gates_with_evidence(self):
        """Test quality gate checking with evidence data."""
        mock_evidence = {
            "evidence": {
                "quality": {
                    "linting": {"errors": 0},
                    "type_check": {"passing": True},
                    "security": {"vulnerabilities": 0}
                },
                "tests": {
                    "unit": {"passing": 5},
                    "coverage": 85
                }
            }
        }
        
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_evidence))):
            
            results = self.service.check_quality_gates(205)
            
            self.assertEqual(results['overall_status'], 'ready')
            self.assertFalse(results['draft_pr'])
            self.assertTrue(results['gates_status']['code_quality'])
            self.assertTrue(results['gates_status']['testing'])
            
    def test_determine_pr_strategy(self):
        """Test PR strategy determination."""
        # Test ready strategy
        quality_results = {'overall_status': 'ready'}
        strategy = self.service.determine_pr_strategy(205, quality_results)
        
        self.assertFalse(strategy['draft'])
        self.assertEqual(strategy['title_prefix'], 'Ready:')
        self.assertIn('ready-for-review', strategy['labels'])
        
        # Test failing strategy
        quality_results = {'overall_status': 'failing'}
        strategy = self.service.determine_pr_strategy(205, quality_results)
        
        self.assertFalse(strategy['create_pr'])
        self.assertIn('reason', strategy)


class TestStateMachineHooks(unittest.TestCase):
    """Test the state machine hooks system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hooks = StateMachineHooks()
        
    def test_register_hook(self):
        """Test hook registration."""
        def test_hook(issue_number, from_state, to_state, context):
            return {'success': True, 'action': 'test_action'}
            
        success = self.hooks.register_hook('test_hook', test_hook)
        self.assertTrue(success)
        self.assertIn('test_hook', self.hooks.hooks)
        
    def test_execute_hooks(self):
        """Test hook execution."""
        # Register a test hook
        def test_hook(issue_number, from_state, to_state, context):
            return {'success': True, 'action': 'test_executed'}
            
        self.hooks.register_hook('test_hook', test_hook)
        
        # Execute hooks
        results = self.hooks.execute_hooks(205, 'implementing', 'validating')
        
        self.assertTrue(results['overall_success'])
        self.assertIn('test_hook', results['hook_results'])
        self.assertEqual(len(results['actions_taken']), 2)  # pr_creation + test_hook
        
    @patch.object(GitHubStateManager, 'transition_state')
    @patch.object(GitHubStateManager, 'get_current_state')
    def test_enhanced_transition_state(self, mock_get_state, mock_transition):
        """Test enhanced state transition."""
        mock_get_state.return_value = 'implementing'
        mock_transition.return_value = (True, "State transition successful")
        
        result = self.hooks.enhanced_transition_state(205, 'validating', "Implementation complete")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['from_state'], 'implementing')
        self.assertEqual(result['to_state'], 'validating')
        self.assertTrue(result['enhanced_transition'])


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""
    
    @patch('subprocess.run')
    @patch.object(Path, 'exists')
    @patch('builtins.open')
    def test_complete_pr_creation_workflow(self, mock_open_func, mock_path_exists, mock_subprocess):
        """Test the complete PR creation workflow."""
        # Setup mocks
        mock_path_exists.return_value = True
        
        # Mock GitHub CLI responses
        issue_response = {
            "title": "Test Implementation",
            "body": "Test body",
            "labels": [{"name": "enhancement"}],
            "assignees": [],
            "state": "open",
            "author": {"login": "test-user"}
        }
        
        # Mock checkpoint data
        checkpoint_data = {
            "checkpoint_id": "test-complete",
            "phase": "Implementation Complete",
            "status": "complete",
            "description": "All implementation finished",
            "timestamp": "2025-01-24T18:00:00Z",
            "components_implemented": {
                "test_component": {
                    "status": "complete",
                    "description": "Test component implemented"
                }
            }
        }
        
        # Mock quality evidence
        quality_evidence = {
            "evidence": {
                "quality": {
                    "linting": {"errors": 0},
                    "type_check": {"passing": True}
                },
                "tests": {
                    "unit": {"passing": 3},
                    "coverage": 85
                }
            }
        }
        
        # Setup file system mocks
        def mock_open_side_effect(filename, *args, **kwargs):
            if 'checkpoint' in str(filename):
                return mock_open(read_data=json.dumps(checkpoint_data))()
            elif 'evidence' in str(filename):
                return mock_open(read_data=json.dumps(quality_evidence))()
            elif 'pull_request_template' in str(filename):
                return mock_open(read_data="# PR Template\n{summary}")()
            else:
                return mock_open(read_data="{}")()
                
        mock_open_func.side_effect = mock_open_side_effect
        
        # Mock subprocess calls
        def subprocess_side_effect(cmd, **kwargs):
            mock_result = Mock()
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_result.returncode = 0
            
            if 'gh issue view' in ' '.join(cmd):
                mock_result.stdout = json.dumps(issue_response)
            elif 'git diff' in ' '.join(cmd):
                mock_result.stdout = "A\ttest_file.py"
            elif 'gh pr create' in ' '.join(cmd):
                mock_result.stdout = "https://github.com/test/repo/pull/123"
            elif 'git checkout' in ' '.join(cmd):
                pass  # Success
                
            return mock_result
            
        mock_subprocess.side_effect = subprocess_side_effect
        
        # Execute the complete workflow
        hooks = StateMachineHooks()
        result = hooks.enhanced_transition_state(
            205, 
            'pr_creating', 
            "Testing complete workflow"
        )
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertGreater(len(result['actions_taken']), 0)


def run_all_tests():
    """Run the complete test suite."""
    print("🧪 Running Automated PR Creation Test Suite")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGitHubStateManagerPRExtensions,
        TestPRTemplateAggregator, 
        TestPRCreationService,
        TestStateMachineHooks,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)