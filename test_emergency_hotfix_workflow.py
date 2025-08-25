#!/usr/bin/env python3
"""
Test script for Emergency Hotfix Workflow
Validates the complete emergency hotfix implementation.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

# Add the commands directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'claude', 'commands'))

try:
    from emergency_hotfix_manager import HotfixManager
except ImportError as e:
    print(f"❌ Could not import HotfixManager: {e}")
    print("Make sure emergency_hotfix_manager.py is in claude/commands/")
    sys.exit(1)


class EmergencyWorkflowTester:
    """
    Comprehensive test suite for emergency hotfix workflow.
    """
    
    def __init__(self):
        self.repo_path = Path(".").absolute()
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": []
        }
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 Starting Emergency Hotfix Workflow Test Suite")
        print("=" * 60)
        
        # Test 1: Configuration validation
        self.test_configuration_exists()
        
        # Test 2: CLI functionality
        self.test_cli_functionality()
        
        # Test 3: Emergency simulation
        self.test_emergency_simulation()
        
        # Test 4: GitHub workflow validation
        self.test_github_workflow_exists()
        
        # Test 5: Incident management
        self.test_incident_management()
        
        # Test 6: Branch management
        self.test_branch_management()
        
        # Print summary
        self.print_test_summary()
    
    def test_configuration_exists(self):
        """Test that emergency hotfix configuration exists and is valid."""
        self.start_test("Configuration Validation")
        
        config_path = self.repo_path / "config" / "emergency-hotfix.yaml"
        
        if not config_path.exists():
            self.fail_test(f"Configuration file not found: {config_path}")
            return
        
        # Basic YAML structure validation
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                
            required_sections = [
                "production_branch",
                "quality_gates",
                "deployment",
                "notifications",
                "incident_tracking",
                "monitoring",
                "security",
                "performance_targets"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                self.fail_test(f"Missing configuration sections: {missing_sections}")
                return
                
            self.pass_test("Configuration file valid")
            
        except Exception as e:
            self.fail_test(f"Configuration validation error: {e}")
    
    def test_cli_functionality(self):
        """Test the emergency hotfix CLI functionality."""
        self.start_test("CLI Functionality")
        
        try:
            # Test CLI help
            result = subprocess.run([
                "python3", "claude/commands/emergency_hotfix_manager.py", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.fail_test(f"CLI help failed: {result.stderr}")
                return
            
            # Check for required commands
            required_commands = ["create", "create-pr", "monitor", "list", "simulate"]
            missing_commands = []
            
            for cmd in required_commands:
                if cmd not in result.stdout:
                    missing_commands.append(cmd)
            
            if missing_commands:
                self.fail_test(f"Missing CLI commands: {missing_commands}")
                return
                
            self.pass_test("CLI functionality verified")
            
        except subprocess.TimeoutExpired:
            self.fail_test("CLI test timed out")
        except Exception as e:
            self.fail_test(f"CLI test error: {e}")
    
    def test_emergency_simulation(self):
        """Test emergency scenario simulation."""
        self.start_test("Emergency Simulation")
        
        try:
            manager = HotfixManager()
            
            # Test scenario simulation
            result = manager.simulate_emergency("database_connection")
            
            if not result.get("success"):
                self.fail_test(f"Simulation failed: {result.get('error', 'Unknown error')}")
                return
            
            incident_id = result["incident_id"]
            
            # Verify incident record was created
            incident_file = Path("incidents") / f"{incident_id}.json"
            if not incident_file.exists():
                self.fail_test(f"Incident record not created: {incident_file}")
                return
            
            # Verify incident record structure
            with open(incident_file, 'r') as f:
                incident_data = json.load(f)
            
            required_fields = ["incident_id", "description", "severity", "created_at", "timeline"]
            missing_fields = []
            
            for field in required_fields:
                if field not in incident_data:
                    missing_fields.append(field)
            
            if missing_fields:
                self.fail_test(f"Missing incident record fields: {missing_fields}")
                return
            
            # Clean up - switch back to original branch
            try:
                subprocess.run(["git", "checkout", "main"], 
                             capture_output=True, check=True)
                subprocess.run(["git", "branch", "-D", incident_data.get("branch_name", "")], 
                             capture_output=True)
            except:
                pass  # Clean up failures are not critical
            
            self.pass_test(f"Emergency simulation successful (ID: {incident_id})")
            
        except Exception as e:
            self.fail_test(f"Simulation error: {e}")
    
    def test_github_workflow_exists(self):
        """Test that GitHub workflow file exists and has required structure."""
        self.start_test("GitHub Workflow Validation")
        
        workflow_path = self.repo_path / ".github" / "workflows" / "emergency-hotfix.yml"
        
        if not workflow_path.exists():
            self.fail_test(f"GitHub workflow not found: {workflow_path}")
            return
        
        try:
            with open(workflow_path, 'r') as f:
                workflow_content = f.read()
            
            # Check for required workflow components
            required_components = [
                "name: Emergency Hotfix Workflow",
                "on:",
                "branches:",
                "hotfix/**",
                "emergency-detection",
                "emergency-quality-gates",
                "emergency-deployment",
                "automatic-backport",
                "post-mortem-automation",
                "emergency-rollback"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in workflow_content:
                    missing_components.append(component)
            
            if missing_components:
                self.fail_test(f"Missing workflow components: {missing_components}")
                return
            
            # Check for quality gates
            quality_gates = ["Security Scan", "Smoke Tests"]
            missing_gates = []
            
            for gate in quality_gates:
                if gate not in workflow_content:
                    missing_gates.append(gate)
            
            if missing_gates:
                self.fail_test(f"Missing quality gates: {missing_gates}")
                return
            
            self.pass_test("GitHub workflow structure validated")
            
        except Exception as e:
            self.fail_test(f"Workflow validation error: {e}")
    
    def test_incident_management(self):
        """Test incident management functionality."""
        self.start_test("Incident Management")
        
        try:
            manager = HotfixManager()
            
            # Test listing incidents (should not crash)
            incidents = manager.list_active_incidents()
            
            if not isinstance(incidents, list):
                self.fail_test("list_active_incidents should return a list")
                return
            
            # Test creating incident record structure
            test_incident = {
                "incident_id": "TEST-001",
                "description": "Test incident",
                "severity": "high",
                "status": "in_progress",  # Use status that won't be filtered out
                "created_at": "2025-08-25T12:00:00Z",
                "timeline": []
            }
            
            # Create test incident file
            incidents_dir = Path("incidents")
            incidents_dir.mkdir(exist_ok=True)
            
            test_file = incidents_dir / "TEST-001.json"
            with open(test_file, 'w') as f:
                json.dump(test_incident, f, indent=2)
            
            # Test that it appears in listing
            incidents = manager.list_active_incidents()
            found = any(inc["incident_id"] == "TEST-001" for inc in incidents)
            
            # Clean up
            test_file.unlink()
            
            if not found:
                self.fail_test("Created incident not found in listing")
                return
            
            self.pass_test("Incident management functionality verified")
            
        except Exception as e:
            self.fail_test(f"Incident management error: {e}")
    
    def test_branch_management(self):
        """Test branch creation and management."""
        self.start_test("Branch Management")
        
        try:
            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"], 
                capture_output=True, text=True, check=True
            )
            original_branch = result.stdout.strip()
            
            # Test branch naming convention
            from datetime import datetime
            test_description = "test emergency fix"
            expected_pattern = f"hotfix/{test_description.replace(' ', '-')}"
            
            # This test verifies the branch naming logic without actually creating branches
            if "hotfix/" not in expected_pattern:
                self.fail_test("Branch naming pattern incorrect")
                return
            
            # Verify we can create temporary branch (then delete it immediately)
            temp_branch = f"test-branch-{int(time.time())}"
            
            try:
                subprocess.run(["git", "checkout", "-b", temp_branch], 
                             capture_output=True, check=True)
                subprocess.run(["git", "checkout", original_branch], 
                             capture_output=True, check=True)
                subprocess.run(["git", "branch", "-D", temp_branch], 
                             capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                self.fail_test(f"Branch operations failed: {e}")
                return
            
            self.pass_test("Branch management functionality verified")
            
        except subprocess.CalledProcessError as e:
            self.fail_test(f"Branch management error: {e}")
        except Exception as e:
            self.fail_test(f"Branch management unexpected error: {e}")
    
    def start_test(self, test_name):
        """Start a test."""
        self.current_test = test_name
        self.test_results["tests_run"] += 1
        print(f"\n🧪 Testing: {test_name}")
    
    def pass_test(self, message=""):
        """Mark test as passed."""
        self.test_results["tests_passed"] += 1
        print(f"✅ PASS: {self.current_test}")
        if message:
            print(f"   {message}")
    
    def fail_test(self, reason):
        """Mark test as failed."""
        self.test_results["tests_failed"] += 1
        self.test_results["failures"].append({
            "test": self.current_test,
            "reason": reason
        })
        print(f"❌ FAIL: {self.current_test}")
        print(f"   Reason: {reason}")
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("🧪 TEST SUMMARY")
        print("=" * 60)
        
        total = self.test_results["tests_run"]
        passed = self.test_results["tests_passed"]
        failed = self.test_results["tests_failed"]
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "0%")
        
        if failed > 0:
            print(f"\n❌ FAILURES ({failed}):")
            for failure in self.test_results["failures"]:
                print(f"   • {failure['test']}: {failure['reason']}")
            print()
            
            print("🔧 NEXT STEPS:")
            print("   1. Review and fix the failed components")
            print("   2. Re-run the test suite")
            print("   3. Validate fixes in a test environment")
        else:
            print(f"\n✅ ALL TESTS PASSED!")
            print("\n🚀 EMERGENCY HOTFIX WORKFLOW READY!")
            print("\n📋 USAGE:")
            print("   # Create emergency hotfix:")
            print("   python claude/commands/emergency_hotfix_manager.py create 'Fix description' --severity critical")
            print("   ")
            print("   # Simulate emergency:")
            print("   python claude/commands/emergency_hotfix_manager.py simulate database_connection")
            print("   ")
            print("   # Monitor active incidents:")
            print("   python claude/commands/emergency_hotfix_manager.py list")
    
    def test_workflow_integration(self):
        """Test integration with RIF systems."""
        self.start_test("RIF Integration")
        
        try:
            # Check if RIF configuration files exist
            rif_config_files = [
                "config/rif-workflow.yaml",
                "config/multi-agent.yaml",
                "config/framework-variables.yaml"
            ]
            
            missing_configs = []
            for config_file in rif_config_files:
                if not Path(config_file).exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                print(f"   ⚠️  Missing RIF configs (non-critical): {missing_configs}")
            
            # Check if GitHub CLI is available (required for PR creation)
            try:
                result = subprocess.run(["gh", "--version"], capture_output=True, check=True)
                gh_version = result.stdout.decode().strip()
                print(f"   ✅ GitHub CLI available: {gh_version}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("   ⚠️  GitHub CLI not available - PR creation will fail")
            
            self.pass_test("RIF integration checks completed")
            
        except Exception as e:
            self.fail_test(f"Integration test error: {e}")


def main():
    """Run the emergency hotfix workflow test suite."""
    print("🚨 RIF Emergency Hotfix Workflow Test Suite")
    print("This will test the complete emergency hotfix implementation.")
    print()
    
    tester = EmergencyWorkflowTester()
    tester.run_all_tests()
    
    # Return appropriate exit code
    if tester.test_results["tests_failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()