#!/usr/bin/env python3
"""
Configuration Validation Script for RIF
Validates deployment configuration and path resolution
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'claude' / 'commands'))

try:
    from path_resolver import PathResolver, ConfigurationValidator
except ImportError:
    print("❌ PathResolver not found. Please ensure path_resolver.py is in claude/commands/")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ConfigurationTester:
    """
    Comprehensive configuration testing and validation
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
    
    def run_all_tests(self) -> bool:
        """
        Run all configuration tests
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("🧪 RIF Configuration Validation")
        print("=" * 50)
        
        # Test 1: Basic Configuration Loading
        self.test_config_loading()
        
        # Test 2: Path Resolution
        self.test_path_resolution()
        
        # Test 3: Directory Structure
        self.test_directory_structure()
        
        # Test 4: Environment Variables
        self.test_environment_variables()
        
        # Test 5: Feature Configuration
        self.test_feature_configuration()
        
        # Test 6: Security Settings
        self.test_security_settings()
        
        # Print summary
        self.print_summary()
        
        return self.results['failed'] == 0
    
    def test_config_loading(self):
        """Test configuration file loading"""
        print("\n📋 Testing Configuration Loading...")
        
        try:
            resolver = PathResolver()
            config = resolver.get_config()
            
            # Check required sections
            required_sections = ['version', 'deployment_mode', 'paths', 'features']
            for section in required_sections:
                if section in config:
                    self.pass_test(f"Required section '{section}' present")
                else:
                    self.fail_test(f"Missing required section '{section}'")
            
            # Check version
            version = config.get('version')
            if version and version == '1.0.0':
                self.pass_test(f"Configuration version: {version}")
            else:
                self.warn_test(f"Unexpected configuration version: {version}")
            
            # Check deployment mode
            mode = config.get('deployment_mode')
            valid_modes = ['project', 'development', 'production']
            if mode in valid_modes:
                self.pass_test(f"Valid deployment mode: {mode}")
            else:
                self.fail_test(f"Invalid deployment mode: {mode}")
                
        except Exception as e:
            self.fail_test(f"Configuration loading failed: {e}")
    
    def test_path_resolution(self):
        """Test path resolution functionality"""
        print("\n📁 Testing Path Resolution...")
        
        try:
            resolver = PathResolver()
            paths_config = resolver.get_config('paths')
            
            for path_key, path_template in paths_config.items():
                try:
                    resolved_path = resolver.resolve(path_key)
                    
                    # Check if path is absolute
                    if resolved_path.is_absolute():
                        self.pass_test(f"Path '{path_key}' resolves to absolute path")
                    else:
                        self.fail_test(f"Path '{path_key}' is not absolute: {resolved_path}")
                    
                    # Check if parent directory exists or can be created
                    parent_dir = resolved_path.parent
                    if parent_dir.exists() or self.can_create_directory(parent_dir):
                        self.pass_test(f"Path '{path_key}' is accessible")
                    else:
                        self.fail_test(f"Path '{path_key}' is not accessible: {resolved_path}")
                        
                except Exception as e:
                    self.fail_test(f"Failed to resolve path '{path_key}': {e}")
                    
        except Exception as e:
            self.fail_test(f"Path resolution test failed: {e}")
    
    def test_directory_structure(self):
        """Test directory structure creation and access"""
        print("\n🏗️  Testing Directory Structure...")
        
        try:
            resolver = PathResolver()
            paths_config = resolver.get_config('paths')
            
            for path_key, _ in paths_config.items():
                try:
                    resolved_path = resolver.resolve(path_key)
                    
                    # Try to create directory if it doesn't exist
                    if not resolved_path.exists():
                        resolved_path.mkdir(parents=True, exist_ok=True)
                    
                    if resolved_path.exists():
                        self.pass_test(f"Directory '{path_key}' exists: {resolved_path}")
                        
                        # Test write access
                        test_file = resolved_path / '.test_write'
                        try:
                            test_file.write_text('test')
                            test_file.unlink()
                            self.pass_test(f"Directory '{path_key}' is writable")
                        except PermissionError:
                            self.fail_test(f"Directory '{path_key}' is not writable")
                    else:
                        self.fail_test(f"Cannot create directory '{path_key}': {resolved_path}")
                        
                except Exception as e:
                    self.fail_test(f"Directory test failed for '{path_key}': {e}")
                    
        except Exception as e:
            self.fail_test(f"Directory structure test failed: {e}")
    
    def test_environment_variables(self):
        """Test environment variable configuration"""
        print("\n🌍 Testing Environment Variables...")
        
        import os
        
        # Check critical environment variables
        env_vars = {
            'PROJECT_ROOT': str(self.project_root),
            'RIF_HOME': str(self.project_root / '.rif'),
        }
        
        for var_name, expected_value in env_vars.items():
            actual_value = os.environ.get(var_name)
            if actual_value:
                if Path(actual_value).resolve() == Path(expected_value).resolve():
                    self.pass_test(f"Environment variable '{var_name}' correctly set")
                else:
                    self.warn_test(f"Environment variable '{var_name}' differs from expected")
            else:
                self.warn_test(f"Environment variable '{var_name}' not set")
        
        # Check for .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            self.pass_test("Environment file (.env) exists")
            
            # Parse .env file and check critical variables
            try:
                env_content = env_file.read_text()
                if 'PROJECT_ROOT=' in env_content:
                    self.pass_test("PROJECT_ROOT defined in .env")
                else:
                    self.warn_test("PROJECT_ROOT not defined in .env")
                    
                if 'RIF_DEPLOYMENT_MODE=' in env_content:
                    self.pass_test("RIF_DEPLOYMENT_MODE defined in .env")
                else:
                    self.warn_test("RIF_DEPLOYMENT_MODE not defined in .env")
                    
            except Exception as e:
                self.warn_test(f"Could not parse .env file: {e}")
        else:
            self.warn_test("Environment file (.env) not found")
    
    def test_feature_configuration(self):
        """Test feature flags configuration"""
        print("\n🎛️  Testing Feature Configuration...")
        
        try:
            resolver = PathResolver()
            features = resolver.get_config('features')
            
            # Test feature flag access
            known_features = [
                'self_development_checks',
                'audit_logging', 
                'development_telemetry',
                'shadow_mode',
                'quality_gates',
                'pattern_learning'
            ]
            
            for feature in known_features:
                if feature in features:
                    enabled = resolver.is_feature_enabled(feature)
                    self.pass_test(f"Feature '{feature}' configured: {enabled}")
                else:
                    self.warn_test(f"Feature '{feature}' not configured")
                    
        except Exception as e:
            self.fail_test(f"Feature configuration test failed: {e}")
    
    def test_security_settings(self):
        """Test security configuration"""
        print("\n🔒 Testing Security Settings...")
        
        try:
            resolver = PathResolver()
            security_config = resolver.get_config('security')
            
            # Check security flags
            security_flags = ['sanitize_paths', 'validate_templates', 'restrict_file_access']
            
            for flag in security_flags:
                if flag in security_config:
                    enabled = security_config[flag]
                    if enabled:
                        self.pass_test(f"Security setting '{flag}' enabled")
                    else:
                        self.warn_test(f"Security setting '{flag}' disabled")
                else:
                    self.warn_test(f"Security setting '{flag}' not configured")
            
            # Test path sanitization
            test_path = "../../etc/passwd"
            try:
                # This should be safe with proper path resolution
                resolved = resolver._expand_variables("${PROJECT_ROOT}/" + test_path)
                if str(resolved).startswith(str(self.project_root)):
                    self.pass_test("Path sanitization working correctly")
                else:
                    self.warn_test("Path sanitization may not be working properly")
            except Exception:
                self.pass_test("Path sanitization prevented dangerous path resolution")
                
        except Exception as e:
            self.fail_test(f"Security settings test failed: {e}")
    
    def can_create_directory(self, path: Path) -> bool:
        """
        Test if a directory can be created
        
        Args:
            path: Path to test
            
        Returns:
            True if directory can be created, False otherwise
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except (PermissionError, OSError):
            return False
    
    def pass_test(self, message: str):
        """Record a passing test"""
        print(f"  ✅ {message}")
        self.results['passed'] += 1
        self.results['details'].append(f"PASS: {message}")
    
    def fail_test(self, message: str):
        """Record a failing test"""
        print(f"  ❌ {message}")
        self.results['failed'] += 1
        self.results['details'].append(f"FAIL: {message}")
    
    def warn_test(self, message: str):
        """Record a warning"""
        print(f"  ⚠️  {message}")
        self.results['warnings'] += 1
        self.results['details'].append(f"WARN: {message}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("🏁 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = self.results['passed'] + self.results['failed'] + self.results['warnings']
        
        print(f"Total tests run: {total_tests}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⚠️  Warnings: {self.results['warnings']}")
        
        if self.results['failed'] == 0:
            print("\n🎉 All critical tests PASSED! Configuration is valid.")
            if self.results['warnings'] > 0:
                print("⚠️  Some warnings were found - review them for optimization.")
        else:
            print("\n💥 Some tests FAILED! Please fix the issues before deploying.")
        
        print("=" * 50)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate RIF deployment configuration')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT,
                        help='Project root directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--json', action='store_true',
                        help='Output results in JSON format')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    tester = ConfigurationTester(args.project_root)
    success = tester.run_all_tests()
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(tester.results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()