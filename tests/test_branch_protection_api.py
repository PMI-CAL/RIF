#!/usr/bin/env python3
"""
Unit tests for GitHub Branch Protection API methods
Issue #203: Configure GitHub branch protection rules for main branch

Tests the new branch protection methods added to GitHubAPIClient.
"""

import unittest
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the dependencies before importing
sys.modules['claude.commands.github_timeout_manager'] = Mock()
sys.modules['claude.commands.github_request_context'] = Mock()  
sys.modules['claude.commands.github_batch_resilience'] = Mock()
sys.modules['systems.event_service_bus'] = Mock()

# Now we can import our modules
try:
    from claude.commands.github_api_client import GitHubAPIClient, APICallResult, RateLimitStrategy
except ImportError:
    # If still fails, create mock classes for testing
    class APICallResult:
        def __init__(self, success, data, error_message, duration, timeout_used, 
                     rate_limit_remaining, rate_limit_reset, attempt_count, context_id):
            self.success = success
            self.data = data
            self.error_message = error_message
            self.duration = duration
            self.timeout_used = timeout_used
            self.rate_limit_remaining = rate_limit_remaining
            self.rate_limit_reset = rate_limit_reset
            self.attempt_count = attempt_count
            self.context_id = context_id
    
    class RateLimitStrategy:
        ADAPTIVE = "adaptive"
    
    class GitHubAPIClient:
        def __init__(self, strategy):
            pass

class TestBranchProtectionAPI(unittest.TestCase):
    """Test cases for branch protection API methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = GitHubAPIClient(RateLimitStrategy.ADAPTIVE)
        
        # Mock the underlying components
        self.client.timeout_manager = Mock()
        self.client.context_manager = Mock()
        self.client.batch_manager = Mock()
        
        # Mock context manager responses
        self.client.timeout_manager.can_attempt_request.return_value = (True, "OK")
        self.client.timeout_manager.get_timeout.return_value = 30.0
        
        mock_context = Mock()
        mock_context.context_id = "test_context_123"
        self.client.context_manager.create_context.return_value = mock_context
        
        # Sample protection config for tests
        self.sample_protection_config = {
            "required_status_checks": {
                "strict": True,
                "contexts": ["continuous-integration", "security-scan"]
            },
            "required_pull_request_reviews": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews": True
            },
            "enforce_admins": False
        }
    
    def test_get_branch_protection_success(self):
        """Test successful branch protection retrieval."""
        # Mock successful API response
        expected_data = {"protection": "active"}
        
        with patch.object(self.client, '_execute_gh_command') as mock_execute:
            mock_execute.return_value = (True, expected_data, None)
            
            result = self.client.get_branch_protection("main")
            
            self.assertTrue(result.success)
            self.assertEqual(result.data, expected_data)
            self.assertIsNone(result.error_message)
            
            # Verify correct API call was made
            args, kwargs = mock_execute.call_args
            command_args = args[0]
            self.assertIn("api", command_args)
            self.assertIn("/repos/{owner}/{repo}/branches/main/protection", command_args)
    
    def test_get_branch_protection_not_found(self):
        """Test branch protection retrieval when no protection exists."""
        with patch.object(self.client, '_execute_gh_command') as mock_execute:
            mock_execute.return_value = (False, None, "Branch not protected")
            
            result = self.client.get_branch_protection("main")
            
            self.assertFalse(result.success)
            self.assertIsNone(result.data)
            self.assertEqual(result.error_message, "Branch not protected")
    
    def test_update_branch_protection_success(self):
        """Test successful branch protection update."""
        expected_data = {"protection": "updated"}
        
        with patch.object(self.client, '_execute_gh_command') as mock_execute:
            mock_execute.return_value = (True, expected_data, None)
            
            result = self.client.update_branch_protection(
                "main", 
                self.sample_protection_config
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.data, expected_data)
            
            # Verify correct API call was made
            args, kwargs = mock_execute.call_args
            command_args = args[0]
            self.assertIn("api", command_args)
            self.assertIn("/repos/{owner}/{repo}/branches/main/protection", command_args)
            self.assertIn("PUT", command_args)
            self.assertIn("--input", command_args)
            self.assertIn("-", command_args)
    
    def test_update_branch_protection_dry_run(self):
        """Test dry run mode for branch protection update."""
        result = self.client.update_branch_protection(
            "main", 
            self.sample_protection_config, 
            dry_run=True
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.context_id, "dry_run")
        self.assertIn("DRY RUN", result.data["message"])
        self.assertEqual(result.data["config"], self.sample_protection_config)
    
    def test_update_branch_protection_no_config(self):
        """Test branch protection update with no configuration."""
        result = self.client.update_branch_protection("main", None)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Protection config is required")
    
    def test_remove_branch_protection_success(self):
        """Test successful branch protection removal."""
        audit_reason = "Emergency deployment"
        
        with patch.object(self.client, '_execute_gh_command') as mock_execute:
            mock_execute.return_value = (True, {}, None)
            
            result = self.client.remove_branch_protection("main", audit_reason)
            
            self.assertTrue(result.success)
            
            # Verify correct API call was made
            args, kwargs = mock_execute.call_args
            command_args = args[0]
            self.assertIn("api", command_args)
            self.assertIn("/repos/{owner}/{repo}/branches/main/protection", command_args)
            self.assertIn("DELETE", command_args)
    
    def test_remove_branch_protection_no_reason(self):
        """Test branch protection removal without audit reason."""
        result = self.client.remove_branch_protection("main")
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Audit reason is required for removing branch protection")
    
    def test_validate_branch_protection_config_valid(self):
        """Test validation of valid branch protection configuration."""
        is_valid, error = self.client.validate_branch_protection_config(
            self.sample_protection_config
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_branch_protection_config_invalid_contexts(self):
        """Test validation with invalid contexts."""
        invalid_config = {
            "required_status_checks": {
                "strict": True,
                "contexts": "not_a_list"  # Should be list
            }
        }
        
        is_valid, error = self.client.validate_branch_protection_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("contexts must be a list", error)
    
    def test_validate_branch_protection_config_invalid_strict(self):
        """Test validation with invalid strict setting."""
        invalid_config = {
            "required_status_checks": {
                "strict": "yes",  # Should be boolean
                "contexts": ["test"]
            }
        }
        
        is_valid, error = self.client.validate_branch_protection_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("strict must be a boolean", error)
    
    def test_validate_branch_protection_config_invalid_review_count(self):
        """Test validation with invalid review count."""
        invalid_config = {
            "required_pull_request_reviews": {
                "required_approving_review_count": "one"  # Should be integer
            }
        }
        
        is_valid, error = self.client.validate_branch_protection_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("required_approving_review_count must be an integer", error)
    
    def test_validate_branch_protection_config_negative_review_count(self):
        """Test validation with negative review count."""
        invalid_config = {
            "required_pull_request_reviews": {
                "required_approving_review_count": -1
            }
        }
        
        is_valid, error = self.client.validate_branch_protection_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("cannot be negative", error)
    
    def test_validate_branch_protection_config_invalid_enforce_admins(self):
        """Test validation with invalid enforce_admins setting."""
        invalid_config = {
            "enforce_admins": "false"  # Should be boolean
        }
        
        is_valid, error = self.client.validate_branch_protection_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertIn("enforce_admins must be a boolean", error)
    
    def test_validate_branch_protection_config_exception(self):
        """Test validation with configuration causing exception."""
        # This should trigger the exception handling
        with patch('builtins.isinstance', side_effect=Exception("Test error")):
            is_valid, error = self.client.validate_branch_protection_config({})
            
            self.assertFalse(is_valid)
            self.assertIn("Configuration validation error", error)
    
    @patch('claude.commands.github_api_client.logger')
    def test_remove_branch_protection_logging(self, mock_logger):
        """Test that removal logs a warning."""
        audit_reason = "Emergency deployment"
        
        with patch.object(self.client, '_execute_gh_command') as mock_execute:
            mock_execute.return_value = (True, {}, None)
            
            self.client.remove_branch_protection("main", audit_reason)
            
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            self.assertIn("REMOVING BRANCH PROTECTION", call_args)
            self.assertIn("main", call_args)
            self.assertIn(audit_reason, call_args)

class TestBranchProtectionSetupScript(unittest.TestCase):
    """Test cases for the branch protection setup script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            "version": "1.0",
            "branches": {
                "main": {
                    "protection": {
                        "required_status_checks": {
                            "strict": True,
                            "contexts": ["test"]
                        }
                    },
                    "progressive_activation": {
                        "enabled": True,
                        "steps": [
                            {
                                "id": 1,
                                "name": "test_step",
                                "config": {"enforce_admins": False}
                            }
                        ]
                    }
                }
            }
        }
    
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_load_config_success(self, mock_exists, mock_open):
        """Test successful configuration loading."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(self.sample_config)
        
        from scripts.setup_branch_protection import BranchProtectionManager
        
        with patch('json.load', return_value=self.sample_config):
            manager = BranchProtectionManager(".test/config.json")
            self.assertEqual(manager.config, self.sample_config)
    
    @patch('pathlib.Path.exists')
    def test_load_config_file_not_found(self, mock_exists):
        """Test configuration loading with missing file."""
        mock_exists.return_value = False
        
        from scripts.setup_branch_protection import BranchProtectionManager
        
        with self.assertRaises(FileNotFoundError):
            BranchProtectionManager(".test/nonexistent.json")
    
    def test_merge_configs(self):
        """Test configuration merging functionality."""
        from scripts.setup_branch_protection import BranchProtectionManager
        
        with patch.object(BranchProtectionManager, '_load_config', return_value=self.sample_config):
            manager = BranchProtectionManager()
            
            base = {
                "a": 1,
                "b": {"c": 2, "d": 3},
                "e": 4
            }
            
            override = {
                "b": {"c": 99},
                "f": 5
            }
            
            result = manager._merge_configs(base, override)
            
            expected = {
                "a": 1,
                "b": {"c": 99, "d": 3},
                "e": 4,
                "f": 5
            }
            
            self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()