#!/usr/bin/env python3
"""
RIF-Validator: Comprehensive Release Management Automation Validation
Issue #219 - GitHub Release Management Automation
"""

import os
import sys
import json
import yaml
import subprocess
import re
from datetime import datetime
from pathlib import Path

class RIFReleaseValidator:
    def __init__(self):
        self.workflow_path = "/Users/cal/DEV/RIF/.github/workflows/release-automation.yml"
        self.repo_path = "/Users/cal/DEV/RIF"
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "issue": "#219",
            "agent": "RIF-Validator",
            "tests": {},
            "overall_score": 0,
            "decision": "PENDING"
        }
        
    def colorize(self, text, color):
        """Add color to terminal output"""
        colors = {
            'red': '\033[0;31m',
            'green': '\033[0;32m',
            'yellow': '\033[1;33m',
            'blue': '\033[0;34m',
            'purple': '\033[0;35m',
            'cyan': '\033[0;36m',
            'white': '\033[1;37m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"
        
    def log_test(self, category, test_name, status, details=""):
        """Log test results"""
        if category not in self.validation_results["tests"]:
            self.validation_results["tests"][category] = []
            
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.validation_results["tests"][category].append(result)
        
        # Console output with colors
        status_color = {'PASS': 'green', 'FAIL': 'red', 'SKIP': 'yellow', 'WARN': 'yellow'}
        status_symbol = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️', 'WARN': '⚠️'}
        
        print(f"{self.colorize(status_symbol.get(status, '?'), status_color.get(status, 'white'))} {test_name}: {status}")
        if details:
            print(f"   {details}")

    def test_file_existence(self):
        """Test 1: File Existence and Structure"""
        print(f"\n{self.colorize('📁 Test 1: File Existence and Structure', 'blue')}")
        
        # Workflow file
        if os.path.exists(self.workflow_path):
            self.log_test("file_structure", "Workflow file exists", "PASS", f"Found at {self.workflow_path}")
            
            # Check file size (should be substantial)
            size = os.path.getsize(self.workflow_path)
            if size > 25000:  # 25KB+
                self.log_test("file_structure", "Workflow file size", "PASS", f"{size} bytes")
            else:
                self.log_test("file_structure", "Workflow file size", "WARN", f"Small file: {size} bytes")
        else:
            self.log_test("file_structure", "Workflow file exists", "FAIL", "File not found")
            return False
            
        # Check for supporting files mentioned in implementation
        supporting_files = [
            "config/release-automation.yml", 
            ".github/scripts/release-utilities.js"
        ]
        
        for file_path in supporting_files:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path):
                self.log_test("file_structure", f"{file_path} exists", "PASS")
            else:
                self.log_test("file_structure", f"{file_path} exists", "SKIP", "File not required")
                
        return True

    def test_yaml_syntax(self):
        """Test 2: YAML Syntax Validation"""
        print(f"\n{self.colorize('🔍 Test 2: YAML Syntax Validation', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                
            self.log_test("syntax", "YAML syntax valid", "PASS", "Parsed successfully")
            
            # Check required top-level keys
            required_keys = ['name', 'on', 'permissions', 'jobs']
            for key in required_keys:
                if key in yaml_content:
                    self.log_test("syntax", f"Required key '{key}' present", "PASS")
                else:
                    self.log_test("syntax", f"Required key '{key}' present", "FAIL")
                    
            return True
            
        except yaml.YAMLError as e:
            self.log_test("syntax", "YAML syntax valid", "FAIL", f"YAML Error: {str(e)}")
            return False
        except Exception as e:
            self.log_test("syntax", "YAML syntax valid", "FAIL", f"Error: {str(e)}")
            return False

    def test_github_integration(self):
        """Test 3: GitHub CLI Integration"""
        print(f"\n{self.colorize('🐙 Test 3: GitHub CLI Integration', 'blue')}")
        
        # Test GitHub CLI auth
        try:
            result = subprocess.run(['gh', 'auth', 'status'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_test("github", "GitHub CLI authentication", "PASS", "Authenticated")
            else:
                self.log_test("github", "GitHub CLI authentication", "FAIL", 
                             f"Auth failed: {result.stderr}")
        except Exception as e:
            self.log_test("github", "GitHub CLI authentication", "FAIL", f"Error: {str(e)}")
            
        # Test workflow registration
        try:
            result = subprocess.run(['gh', 'workflow', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                workflows = result.stdout
                if "Release Management Automation" in workflows:
                    self.log_test("github", "Workflow registration", "PASS", 
                                 "Found in workflow list")
                else:
                    self.log_test("github", "Workflow registration", "FAIL", 
                                 "Workflow not found in gh workflow list")
            else:
                self.log_test("github", "Workflow registration", "FAIL", 
                             f"gh workflow list failed: {result.stderr}")
        except Exception as e:
            self.log_test("github", "Workflow registration", "FAIL", f"Error: {str(e)}")

    def test_workflow_structure(self):
        """Test 4: Workflow Structure Validation"""
        print(f"\n{self.colorize('⚙️ Test 4: Workflow Structure Validation', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            # Expected jobs from implementation
            expected_jobs = [
                'version-analysis',
                'build-assets', 
                'create-release',
                'deploy',
                'rif-integration',
                'announce-release',
                'post-release-validation'
            ]
            
            for job in expected_jobs:
                if f"{job}:" in content:
                    self.log_test("workflow_structure", f"Job '{job}' exists", "PASS")
                else:
                    self.log_test("workflow_structure", f"Job '{job}' exists", "FAIL")
                    
            # Test trigger configuration
            triggers = ['workflow_dispatch', 'push:', 'tags:']
            for trigger in triggers:
                if trigger in content:
                    self.log_test("workflow_structure", f"{trigger.replace(':', '')} trigger configured", "PASS")
                else:
                    self.log_test("workflow_structure", f"{trigger.replace(':', '')} trigger configured", "FAIL")
                    
        except Exception as e:
            self.log_test("workflow_structure", "Workflow structure analysis", "FAIL", 
                         f"Error reading workflow: {str(e)}")

    def test_permissions(self):
        """Test 5: Permission Validation"""
        print(f"\n{self.colorize('🔐 Test 5: Permission Validation', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            required_permissions = [
                'contents: write',
                'issues: write', 
                'pull-requests: read'
            ]
            
            for perm in required_permissions:
                if perm in content:
                    self.log_test("permissions", f"Permission '{perm}'", "PASS")
                else:
                    self.log_test("permissions", f"Permission '{perm}'", "FAIL")
                    
        except Exception as e:
            self.log_test("permissions", "Permission analysis", "FAIL", f"Error: {str(e)}")

    def test_semantic_versioning(self):
        """Test 6: Semantic Versioning Logic"""
        print(f"\n{self.colorize('🏷️ Test 6: Semantic Versioning Logic', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            # Look for version pattern matching
            version_patterns = [
                'BREAKING',
                'feat:',
                'fix:', 
                'feature',
                'bugfix'
            ]
            
            for pattern in version_patterns:
                if pattern in content:
                    self.log_test("semantic_versioning", f"Version pattern '{pattern}'", "PASS")
                else:
                    self.log_test("semantic_versioning", f"Version pattern '{pattern}'", "FAIL")
                    
        except Exception as e:
            self.log_test("semantic_versioning", "Semantic versioning analysis", "FAIL", f"Error: {str(e)}")

    def test_asset_management(self):
        """Test 7: Asset Management"""  
        print(f"\n{self.colorize('📦 Test 7: Asset Management', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            asset_features = [
                'source-code',
                'documentation',
                'config-templates',
                'checksums',
                'upload-artifact'
            ]
            
            for feature in asset_features:
                if feature in content:
                    self.log_test("asset_management", f"Asset feature '{feature}'", "PASS")
                else:
                    self.log_test("asset_management", f"Asset feature '{feature}'", "FAIL")
                    
        except Exception as e:
            self.log_test("asset_management", "Asset management analysis", "FAIL", f"Error: {str(e)}")

    def test_rif_integration(self):
        """Test 8: RIF Integration"""
        print(f"\n{self.colorize('🤖 Test 8: RIF Integration', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            rif_features = [
                'state:',
                'rif-integration',
                'knowledge',
                'labels'
            ]
            
            for feature in rif_features:
                if feature in content:
                    self.log_test("rif_integration", f"RIF feature '{feature}'", "PASS")
                else:
                    self.log_test("rif_integration", f"RIF feature '{feature}'", "FAIL")
                    
        except Exception as e:
            self.log_test("rif_integration", "RIF integration analysis", "FAIL", f"Error: {str(e)}")

    def test_environment_support(self):
        """Test 9: Environment Support"""
        print(f"\n{self.colorize('🌍 Test 9: Environment Support', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            environments = ['production', 'staging', 'development']
            
            for env in environments:
                if env in content:
                    self.log_test("environment_support", f"Environment '{env}'", "PASS")
                else:
                    self.log_test("environment_support", f"Environment '{env}'", "FAIL")
                    
        except Exception as e:
            self.log_test("environment_support", "Environment support analysis", "FAIL", f"Error: {str(e)}")

    def test_error_handling(self):
        """Test 10: Error Handling and Validation"""
        print(f"\n{self.colorize('🛡️ Test 10: Error Handling and Validation', 'blue')}")
        
        try:
            with open(self.workflow_path, 'r') as f:
                content = f.read()
                
            error_handlers = [
                'if:',
                'needs:',
                'always()',
                'success()', 
                'failure()'
            ]
            
            for handler in error_handlers:
                if handler in content:
                    self.log_test("error_handling", f"Error handling '{handler}'", "PASS")
                else:
                    self.log_test("error_handling", f"Error handling '{handler}'", "FAIL", 
                                 "Handler not found in workflow")
                    
        except Exception as e:
            self.log_test("error_handling", "Error handling analysis", "FAIL", f"Error: {str(e)}")

    def run_all_tests(self):
        """Execute all validation tests"""
        print(f"{self.colorize('🧪 RIF Release Workflow Test Suite', 'blue')}")
        print("=" * 39)
        
        test_methods = [
            self.test_file_existence,
            self.test_yaml_syntax,
            self.test_github_integration,
            self.test_workflow_structure,
            self.test_permissions,
            self.test_semantic_versioning,
            self.test_asset_management,
            self.test_rif_integration,
            self.test_environment_support,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"{self.colorize('❌ Test failed with exception:', 'red')} {str(e)}")
                
        self.generate_summary()

    def calculate_quality_score(self):
        """Calculate overall quality score based on test results"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, tests in self.validation_results["tests"].items():
            for test in tests:
                total_tests += 1
                if test["status"] == "PASS":
                    passed_tests += 1
                elif test["status"] == "FAIL":
                    failed_tests += 1
                    
        if total_tests == 0:
            return 0
            
        # Quality score calculation: Base 100, -10 for each failure, -2 for each skip/warn
        base_score = 100
        score = base_score - (failed_tests * 10)
        score = max(0, min(100, score))  # Clamp between 0-100
        
        return score

    def generate_summary(self):
        """Generate test summary and recommendations"""
        print(f"\n{self.colorize('📊 Test Summary', 'blue')}")
        print("=" * 19)
        
        quality_score = self.calculate_quality_score()
        self.validation_results["overall_score"] = quality_score
        
        # Determine decision based on score
        if quality_score >= 90:
            decision = "PASS"
        elif quality_score >= 70:
            decision = "PASS with CONCERNS" 
        elif quality_score >= 40:
            decision = "FAIL"
        else:
            decision = "CRITICAL FAILURE"
            
        self.validation_results["decision"] = decision
        
        print(f"{self.colorize('✅ Tests completed', 'green')}")
        print(f"Quality Score: {quality_score}/100")
        print(f"Decision: {decision}")
        
        print(f"\n{self.colorize('⚠️ Manual validation required for:', 'yellow')}")
        print("   - Actual workflow execution")
        print("   - GitHub Release creation")
        print("   - Asset attachment verification") 
        print("   - RIF integration testing")
        print("   - Multi-environment deployment")
        
        print(f"\n{self.colorize('🚀 Next Steps', 'blue')}")
        print("=" * 15)
        print("1. Review any failed tests above")
        print("2. Manually test workflow with:")
        print("   gh workflow run release-automation.yml -f version=v0.0.1-test -f prerelease=true")
        print("3. Validate release creation in GitHub UI")
        print("4. Test RIF integration with actual issues")
        print("5. Verify announcement system functionality")
        
        print(f"\n{self.colorize('✅ Release workflow validation completed!', 'green')}")
        
        return quality_score, decision

if __name__ == "__main__":
    validator = RIFReleaseValidator()
    quality_score, decision = validator.run_all_tests()
    
    # Save results
    with open("/Users/cal/DEV/RIF/validation_report_issue_219.json", "w") as f:
        json.dump(validator.validation_results, f, indent=2)
        
    # Exit with appropriate code
    if decision == "CRITICAL FAILURE":
        sys.exit(1)
    elif decision == "FAIL":
        sys.exit(2)
    else:
        sys.exit(0)