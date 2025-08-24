#!/usr/bin/env python3
"""
Unit Tests for GitHub State Manager
Issue #88: Critical fix for RIF state management system

These tests validate the GitHub state management functionality including
state transitions, label management, and conflict resolution.
"""

import unittest
import json
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.github_state_manager import GitHubStateManager

class TestGitHubStateManager(unittest.TestCase):
    """Test cases for GitHubStateManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary config for testing
        self.temp_config = {
            'workflow': {
                'states': {
                    'new': {'description': 'New issue'},
                    'analyzing': {'description': 'Analyzing'},
                    'implementing': {'description': 'Implementing'},
                    'validating': {'description': 'Validating'},
                    'complete': {'description': 'Complete'},
                    'failed': {'description': 'Failed'}
                }
            }
        }
        
        # Create manager instance with mocked config
        with patch.object(GitHubStateManager, 'load_workflow_config', return_value=self.temp_config['workflow']):
            self.manager = GitHubStateManager()
    
    def test_initialization(self):
        """Test GitHubStateManager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.state_prefix, "state:")
        self.assertIn('new', self.manager.workflow_config['states'])
        self.assertIn('analyzing', self.manager.workflow_config['states'])
    
    @patch('subprocess.run')
    def test_get_issue_labels_success(self, mock_run):
        """Test successful retrieval of issue labels."""
        # Mock successful GitHub CLI response
        mock_response = {
            'labels': [
                {'name': 'state:analyzing'},
                {'name': 'priority:high'},
                {'name': 'bug'}
            ]
        }
        mock_run.return_value.stdout = json.dumps(mock_response)
        mock_run.return_value.returncode = 0
        
        labels = self.manager.get_issue_labels(123)
        
        expected_labels = ['state:analyzing', 'priority:high', 'bug']
        self.assertEqual(labels, expected_labels)
        
        # Verify correct command was called
        mock_run.assert_called_with([
            'gh', 'issue', 'view', '123', '--json', 'labels'
        ], capture_output=True, text=True, check=True)
    
    @patch('subprocess.run')
    def test_get_issue_labels_failure(self, mock_run):
        """Test handling of GitHub CLI failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'gh')
        
        labels = self.manager.get_issue_labels(123)
        
        self.assertEqual(labels, [])
    
    @patch.object(GitHubStateManager, 'get_issue_labels')
    def test_get_current_state_single_label(self, mock_get_labels):
        """Test getting current state with single state label."""
        mock_get_labels.return_value = ['state:implementing', 'priority:high']
        
        state = self.manager.get_current_state(123)
        
        self.assertEqual(state, 'implementing')
    
    @patch.object(GitHubStateManager, 'get_issue_labels')
    def test_get_current_state_no_label(self, mock_get_labels):
        """Test getting current state with no state labels."""
        mock_get_labels.return_value = ['priority:high', 'bug']
        
        state = self.manager.get_current_state(123)
        
        self.assertIsNone(state)
    
    @patch.object(GitHubStateManager, 'get_issue_labels')
    def test_get_current_state_multiple_labels(self, mock_get_labels):
        """Test getting current state with multiple conflicting state labels."""
        mock_get_labels.return_value = ['state:analyzing', 'state:implementing', 'priority:high']
        
        with self.assertLogs(level='WARNING') as log:
            state = self.manager.get_current_state(123)
        
        # Should return first state but log warning
        self.assertEqual(state, 'analyzing')
        self.assertIn('multiple state labels', log.output[0])
    
    def test_validate_state_transition_valid(self):
        """Test validation of valid state transitions."""
        # Emergency states should always be allowed
        is_valid, reason = self.manager.validate_state_transition('analyzing', 'failed')
        self.assertTrue(is_valid)
        self.assertIn('Emergency state', reason)
        
        # Initial state assignment
        is_valid, reason = self.manager.validate_state_transition(None, 'new')
        self.assertTrue(is_valid)
        self.assertIn('Initial state', reason)
        
        # Normal transition
        is_valid, reason = self.manager.validate_state_transition('analyzing', 'implementing')
        self.assertTrue(is_valid)
    
    def test_validate_state_transition_invalid(self):
        """Test validation of invalid state transitions."""
        # Unknown target state
        is_valid, reason = self.manager.validate_state_transition('analyzing', 'nonexistent')
        self.assertFalse(is_valid)
        self.assertIn('Unknown state', reason)
    
    @patch('subprocess.run')
    def test_remove_conflicting_labels_success(self, mock_run):
        """Test successful removal of conflicting state labels."""
        # Setup mock for getting labels
        with patch.object(self.manager, 'get_issue_labels', 
                         return_value=['state:analyzing', 'state:implementing', 'priority:high']):
            
            # Mock successful label removal
            mock_run.return_value.returncode = 0
            
            removed_labels = self.manager.remove_conflicting_labels(123)
            
            expected_removed = ['state:analyzing', 'state:implementing']
            self.assertEqual(removed_labels, expected_removed)
            
            # Verify correct commands were called
            self.assertEqual(mock_run.call_count, 2)  # Two state labels to remove
    
    @patch('subprocess.run')
    def test_add_state_label_success(self, mock_run):
        """Test successful addition of state label."""
        mock_run.return_value.returncode = 0
        
        success = self.manager.add_state_label(123, 'implementing')
        
        self.assertTrue(success)
        mock_run.assert_called_with([
            'gh', 'issue', 'edit', '123', '--add-label', 'state:implementing'
        ], check=True, capture_output=True)
    
    @patch('subprocess.run')
    def test_add_state_label_failure(self, mock_run):
        """Test handling of label addition failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'gh')
        
        success = self.manager.add_state_label(123, 'implementing')
        
        self.assertFalse(success)
    
    def test_transition_state_success(self):
        """Test successful state transition."""
        # Mock all the methods used in transition_state
        with patch.object(self.manager, 'get_current_state', return_value='analyzing'), \
             patch.object(self.manager, 'validate_state_transition', return_value=(True, 'Valid')), \
             patch.object(self.manager, 'remove_conflicting_labels', return_value=['state:analyzing']), \
             patch.object(self.manager, 'add_state_label', return_value=True), \
             patch.object(self.manager, '_log_transition'):
            
            success, message = self.manager.transition_state(123, 'implementing', 'Test transition')
            
            self.assertTrue(success)
            self.assertIn('Successfully transitioned', message)
    
    def test_transition_state_invalid_transition(self):
        """Test state transition with invalid transition."""
        with patch.object(self.manager, 'get_current_state', return_value='analyzing'), \
             patch.object(self.manager, 'validate_state_transition', 
                         return_value=(False, 'Invalid transition')):
            
            success, message = self.manager.transition_state(123, 'invalid_state')
            
            self.assertFalse(success)
            self.assertIn('Invalid transition', message)
    
    def test_transition_state_add_label_failure(self):
        """Test state transition when label addition fails."""
        with patch.object(self.manager, 'get_current_state', return_value='analyzing'), \
             patch.object(self.manager, 'validate_state_transition', return_value=(True, 'Valid')), \
             patch.object(self.manager, 'remove_conflicting_labels', return_value=[]), \
             patch.object(self.manager, 'add_state_label', return_value=False):
            
            success, message = self.manager.transition_state(123, 'implementing')
            
            self.assertFalse(success)
            self.assertIn('Failed to add new state label', message)
    
    def test_validate_issue_state_valid(self):
        """Test validation of valid issue state."""
        with patch.object(self.manager, 'get_issue_labels', 
                         return_value=['state:analyzing', 'priority:high']):
            
            validation = self.manager.validate_issue_state(123)
            
            self.assertTrue(validation['is_valid'])
            self.assertEqual(validation['state_count'], 1)
            self.assertEqual(validation['state_labels'], ['state:analyzing'])
            self.assertEqual(validation['issues'], [])
    
    def test_validate_issue_state_no_labels(self):
        """Test validation of issue with no state labels."""
        with patch.object(self.manager, 'get_issue_labels', 
                         return_value=['priority:high', 'bug']):
            
            validation = self.manager.validate_issue_state(123)
            
            self.assertFalse(validation['is_valid'])
            self.assertEqual(validation['state_count'], 0)
            self.assertIn('No state label found', validation['issues'])
    
    def test_validate_issue_state_multiple_labels(self):
        """Test validation of issue with multiple state labels."""
        with patch.object(self.manager, 'get_issue_labels', 
                         return_value=['state:analyzing', 'state:implementing', 'priority:high']):
            
            validation = self.manager.validate_issue_state(123)
            
            self.assertFalse(validation['is_valid'])
            self.assertEqual(validation['state_count'], 2)
            self.assertIn('Multiple state labels', validation['issues'][0])
    
    def test_validate_issue_state_unknown_state(self):
        """Test validation of issue with unknown state."""
        # Mock issue with state not in workflow config
        with patch.object(self.manager, 'get_issue_labels', 
                         return_value=['state:unknown_state']):
            
            validation = self.manager.validate_issue_state(123)
            
            # Should be invalid because 'unknown_state' is not in the workflow config
            self.assertFalse(validation['is_valid'])
            self.assertIn('Unknown state: unknown_state', validation['issues'])
    
    @patch('subprocess.run')
    def test_get_all_open_issues_success(self, mock_run):
        """Test successful retrieval of all open issues."""
        mock_response = [
            {'number': 123},
            {'number': 124},
            {'number': 125}
        ]
        mock_run.return_value.stdout = json.dumps(mock_response)
        mock_run.return_value.returncode = 0
        
        issues = self.manager.get_all_open_issues()
        
        self.assertEqual(issues, [123, 124, 125])
    
    @patch('subprocess.run')
    def test_get_all_open_issues_failure(self, mock_run):
        """Test handling of failure to get open issues."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'gh')
        
        issues = self.manager.get_all_open_issues()
        
        self.assertEqual(issues, [])
    
    def test_audit_all_issues(self):
        """Test comprehensive audit of all issues."""
        mock_issues = [123, 124, 125]
        
        # Mock validation results for different scenarios
        mock_validations = [
            {'is_valid': True, 'state_count': 1, 'issues': []},  # Valid issue
            {'is_valid': False, 'state_count': 0, 'issues': ['No state label found']},  # No state
            {'is_valid': False, 'state_count': 2, 'issues': ['Multiple state labels: [...]']}  # Multiple states
        ]
        
        with patch.object(self.manager, 'get_all_open_issues', return_value=mock_issues), \
             patch.object(self.manager, 'validate_issue_state', side_effect=mock_validations):
            
            audit_report = self.manager.audit_all_issues()
            
            self.assertEqual(audit_report['total_issues'], 3)
            self.assertEqual(audit_report['valid_issues'], 1)
            self.assertEqual(audit_report['invalid_issues'], 2)
            self.assertEqual(len(audit_report['issues_by_problem']['no_state']), 1)
            self.assertEqual(len(audit_report['issues_by_problem']['multiple_states']), 1)
    
    @patch('subprocess.run')
    def test_post_comment_to_issue_success(self, mock_run):
        """Test successful posting of comment to issue."""
        mock_run.return_value.returncode = 0
        
        success = self.manager.post_comment_to_issue(123, "Test comment")
        
        self.assertTrue(success)
        mock_run.assert_called_with([
            'gh', 'issue', 'comment', '123', '--body', 'Test comment'
        ], check=True, capture_output=True)
    
    @patch('subprocess.run')
    def test_post_comment_to_issue_failure(self, mock_run):
        """Test handling of comment posting failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'gh')
        
        success = self.manager.post_comment_to_issue(123, "Test comment")
        
        self.assertFalse(success)
    
    def test_label_caching(self):
        """Test that label caching works correctly."""
        with patch('subprocess.run') as mock_run:
            # Mock first call
            mock_response = {'labels': [{'name': 'state:analyzing'}]}
            mock_run.return_value.stdout = json.dumps(mock_response)
            mock_run.return_value.returncode = 0
            
            # First call should hit the API
            labels1 = self.manager.get_issue_labels(123)
            self.assertEqual(mock_run.call_count, 1)
            
            # Second call should use cache
            labels2 = self.manager.get_issue_labels(123)
            self.assertEqual(mock_run.call_count, 1)  # No additional API call
            
            self.assertEqual(labels1, labels2)
    
    @patch('builtins.open', create=True)
    def test_log_transition(self, mock_open):
        """Test transition logging functionality."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        transition_log = {
            'timestamp': '2023-01-01T00:00:00Z',
            'issue_number': 123,
            'from_state': 'analyzing',
            'to_state': 'implementing',
            'reason': 'Test transition'
        }
        
        self.manager._log_transition(transition_log)
        
        # Verify file was opened for append
        mock_open.assert_called()
        
        # Verify JSON was written
        mock_file.write.assert_called()
        written_data = mock_file.write.call_args[0][0]
        self.assertIn('Test transition', written_data)


class TestGitHubStateManagerIntegration(unittest.TestCase):
    """Integration tests for GitHub state manager that test actual workflow scenarios."""
    
    def setUp(self):
        """Set up integration test environment."""
        with patch.object(GitHubStateManager, 'load_workflow_config') as mock_config:
            mock_config.return_value = {
                'states': {
                    'new': {'description': 'New issue'},
                    'analyzing': {'description': 'Analyzing'},
                    'implementing': {'description': 'Implementing'},
                    'validating': {'description': 'Validating'},
                    'complete': {'description': 'Complete'}
                }
            }
            self.manager = GitHubStateManager()
    
    def test_full_state_transition_workflow(self):
        """Test complete state transition workflow from new to complete."""
        issue_number = 999  # Use a test issue number
        
        # Mock all subprocess calls for complete workflow
        with patch('subprocess.run') as mock_run:
            # Mock successful operations
            mock_run.return_value.returncode = 0
            
            # Test each transition in the workflow
            transitions = [
                ('new', 'Initialization'),
                ('analyzing', 'Starting analysis'),
                ('implementing', 'Analysis complete'),
                ('validating', 'Implementation complete'),
                ('complete', 'Validation passed')
            ]
            
            previous_state = None
            for target_state, reason in transitions:
                # Mock get_current_state to return previous state
                with patch.object(self.manager, 'get_current_state', return_value=previous_state):
                    success, message = self.manager.transition_state(
                        issue_number, target_state, reason
                    )
                    
                    self.assertTrue(success, f"Failed transition to {target_state}: {message}")
                    self.assertIn('Successfully transitioned', message)
                
                previous_state = target_state
    
    def test_conflict_resolution_scenario(self):
        """Test handling of conflicting state labels scenario."""
        issue_number = 888
        conflicting_labels = ['state:analyzing', 'state:implementing', 'state:validating']
        
        with patch.object(self.manager, 'get_issue_labels', return_value=conflicting_labels), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value.returncode = 0
            
            # Attempt to clean up the conflicting states
            removed_labels = self.manager.remove_conflicting_labels(issue_number)
            
            # Should remove all state labels
            self.assertEqual(len(removed_labels), 3)
            for label in conflicting_labels:
                self.assertIn(label, removed_labels)
    
    def test_error_recovery_scenario(self):
        """Test error recovery during state transitions."""
        issue_number = 777
        
        # Simulate GitHub CLI failure during label removal
        with patch.object(self.manager, 'get_current_state', return_value='analyzing'), \
             patch.object(self.manager, 'remove_conflicting_labels', return_value=[]), \
             patch.object(self.manager, 'add_state_label', side_effect=[False, True]):
            
            # First attempt should fail
            success, message = self.manager.transition_state(
                issue_number, 'implementing', 'First attempt'
            )
            self.assertFalse(success)
            self.assertIn('Failed to add new state label', message)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)