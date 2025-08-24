#!/usr/bin/env python3
"""
Basic validation tests for GitHub Branch Protection setup
Issue #203: Configure GitHub branch protection rules for main branch

Simple tests that verify configuration validation logic.
"""

import unittest
import json
import sys
from pathlib import Path

class TestBranchProtectionValidation(unittest.TestCase):
    """Test cases for branch protection validation logic."""
    
    def test_config_validation_logic(self):
        """Test the configuration validation logic directly."""
        
        def validate_branch_protection_config(protection_config):
            """Simplified validation logic from the API client."""
            try:
                # Required fields validation
                if "required_status_checks" in protection_config:
                    status_checks = protection_config["required_status_checks"]
                    if not isinstance(status_checks.get("contexts", []), list):
                        return False, "required_status_checks.contexts must be a list"
                    if not isinstance(status_checks.get("strict", True), bool):
                        return False, "required_status_checks.strict must be a boolean"
                
                if "required_pull_request_reviews" in protection_config:
                    pr_reviews = protection_config["required_pull_request_reviews"]
                    if not isinstance(pr_reviews.get("required_approving_review_count", 1), int):
                        return False, "required_pull_request_reviews.required_approving_review_count must be an integer"
                    if pr_reviews.get("required_approving_review_count", 1) < 0:
                        return False, "required_approving_review_count cannot be negative"
                
                # Validate enforce_admins
                if "enforce_admins" in protection_config:
                    if not isinstance(protection_config["enforce_admins"], bool):
                        return False, "enforce_admins must be a boolean"
                
                return True, None
                
            except Exception as e:
                return False, f"Configuration validation error: {str(e)}"
        
        # Test valid configuration
        valid_config = {
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
        
        is_valid, error = validate_branch_protection_config(valid_config)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test invalid contexts (not a list)
        invalid_config = {
            "required_status_checks": {
                "strict": True,
                "contexts": "not_a_list"
            }
        }
        
        is_valid, error = validate_branch_protection_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("contexts must be a list", error)
        
        # Test invalid strict (not a boolean)
        invalid_config = {
            "required_status_checks": {
                "strict": "yes",
                "contexts": ["test"]
            }
        }
        
        is_valid, error = validate_branch_protection_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("strict must be a boolean", error)
        
        # Test invalid review count (not an integer)
        invalid_config = {
            "required_pull_request_reviews": {
                "required_approving_review_count": "one"
            }
        }
        
        is_valid, error = validate_branch_protection_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("required_approving_review_count must be an integer", error)
        
        # Test negative review count
        invalid_config = {
            "required_pull_request_reviews": {
                "required_approving_review_count": -1
            }
        }
        
        is_valid, error = validate_branch_protection_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("cannot be negative", error)
        
        # Test invalid enforce_admins (not a boolean)
        invalid_config = {
            "enforce_admins": "false"
        }
        
        is_valid, error = validate_branch_protection_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("enforce_admins must be a boolean", error)
    
    def test_configuration_file_structure(self):
        """Test that the configuration file has the correct structure."""
        config_path = Path(__file__).parent.parent / ".github" / "branch-protection.json"
        
        # Check that config file exists
        self.assertTrue(config_path.exists(), "Branch protection configuration file should exist")
        
        # Load and validate JSON structure
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required top-level keys
        required_keys = ["version", "branches", "emergency_procedures", "monitoring", "metadata"]
        for key in required_keys:
            self.assertIn(key, config, f"Configuration should contain {key}")
        
        # Check branch configuration
        self.assertIn("main", config["branches"], "Configuration should include main branch")
        
        main_config = config["branches"]["main"]
        self.assertIn("protection", main_config, "Main branch should have protection config")
        self.assertIn("progressive_activation", main_config, "Main branch should have progressive activation config")
        
        # Check protection configuration structure
        protection = main_config["protection"]
        expected_protection_keys = [
            "required_status_checks",
            "required_pull_request_reviews", 
            "enforce_admins"
        ]
        for key in expected_protection_keys:
            self.assertIn(key, protection, f"Protection config should contain {key}")
        
        # Check status checks
        status_checks = protection["required_status_checks"]
        self.assertIn("contexts", status_checks, "Status checks should specify contexts")
        self.assertIsInstance(status_checks["contexts"], list, "Contexts should be a list")
        
        # Verify expected status checks are present
        expected_contexts = [
            "Code Quality Analysis",
            "Security Scanning", 
            "Test Coverage",
            "Performance Testing",
            "RIF Validation"
        ]
        contexts = status_checks["contexts"]
        for expected_context in expected_contexts:
            self.assertIn(expected_context, contexts, f"Should include {expected_context} status check")
        
        # Check progressive activation
        progressive = main_config["progressive_activation"]
        self.assertIn("steps", progressive, "Progressive activation should have steps")
        self.assertIsInstance(progressive["steps"], list, "Steps should be a list")
        self.assertTrue(len(progressive["steps"]) > 0, "Should have at least one activation step")

class TestSetupScriptValidation(unittest.TestCase):
    """Test cases for the setup script."""
    
    def test_setup_script_exists(self):
        """Test that the setup script exists and is executable."""
        script_path = Path(__file__).parent.parent / "scripts" / "setup_branch_protection.py"
        
        self.assertTrue(script_path.exists(), "Setup script should exist")
        self.assertTrue(script_path.is_file(), "Setup script should be a file")
        
        # Check if script is executable (on Unix-like systems)
        import stat
        file_stat = script_path.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
        self.assertTrue(is_executable, "Setup script should be executable")
    
    def test_config_merge_logic(self):
        """Test the configuration merging logic."""
        
        def merge_configs(base, override):
            """Simplified merge logic from setup script."""
            result = base.copy()
            
            for key, value in override.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        # Test configuration merging
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": 4
        }
        
        override = {
            "b": {"c": 99},
            "f": 5
        }
        
        result = merge_configs(base, override)
        
        expected = {
            "a": 1,
            "b": {"c": 99, "d": 3},
            "e": 4,
            "f": 5
        }
        
        self.assertEqual(result, expected)
        
        # Test deep merging
        base = {
            "protection": {
                "required_status_checks": {
                    "strict": False,
                    "contexts": ["old-check"]
                },
                "enforce_admins": True
            }
        }
        
        override = {
            "protection": {
                "required_status_checks": {
                    "strict": True
                }
            }
        }
        
        result = merge_configs(base, override)
        
        # Should preserve contexts but update strict
        self.assertTrue(result["protection"]["required_status_checks"]["strict"])
        self.assertEqual(result["protection"]["required_status_checks"]["contexts"], ["old-check"])
        self.assertTrue(result["protection"]["enforce_admins"])

if __name__ == '__main__':
    unittest.main()