"""
Test Runner for Pattern Application Engine

This script runs all tests for the Pattern Application Engine
and generates a comprehensive test report.
"""

import sys
import os
import pytest
import time
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def run_tests():
    """
    Run all pattern application tests and generate report.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=" * 80)
    print("PATTERN APPLICATION ENGINE - TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Test configuration
    test_args = [
        '--verbose',
        '--tb=short',
        '--durations=10',
        '--color=yes',
        os.path.dirname(__file__)  # Run tests in this directory
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        test_args.extend([
            '--cov=knowledge.pattern_application',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-fail-under=80'
        ])
        print("Coverage reporting enabled (target: 80%)")
    except ImportError:
        print("Coverage reporting not available (install pytest-cov for coverage)")
    
    print(f"Running tests with args: {' '.join(test_args)}")
    print()
    
    # Run the tests
    start_time = time.time()
    exit_code = pytest.main(test_args)
    end_time = time.time()
    
    # Generate summary
    print()
    print("=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Exit code: {exit_code}")
    
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
        print()
        print("The Pattern Application Engine implementation is ready for validation!")
        print("Next steps:")
        print("1. Review test coverage report (if generated)")
        print("2. Run integration tests with real knowledge system")
        print("3. Update GitHub issue with implementation status")
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please review the test output above and fix any issues.")
        print("Common issues:")
        print("- Import errors: Check that all modules are properly structured")
        print("- Missing dependencies: Ensure all required packages are installed")
        print("- Logic errors: Review implementation against test expectations")
    
    print("=" * 80)
    
    return exit_code == 0

def check_dependencies():
    """Check if required test dependencies are available."""
    print("Checking test dependencies...")
    
    required_modules = [
        'pytest',
        'knowledge.pattern_application.core',
        'knowledge.pattern_application.engine',
        'knowledge.pattern_application.context_extractor',
        'knowledge.pattern_application.pattern_matcher'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            missing_modules.append(module)
    
    if missing_modules:
        print()
        print("Missing dependencies detected!")
        print("Please ensure all modules are properly implemented and accessible.")
        return False
    
    print()
    print("All dependencies available!")
    return True

def run_quick_test():
    """Run a quick smoke test to verify basic functionality."""
    print("Running quick smoke test...")
    
    try:
        # Test core imports
        from knowledge.pattern_application.core import (
            Pattern, IssueContext, TechStack, IssueConstraints,
            PatternApplicationStatus, AdaptationStrategy
        )
        print("✅ Core data models import successfully")
        
        # Test basic object creation
        tech_stack = TechStack(primary_language='python')
        constraints = IssueConstraints()
        context = IssueContext(
            issue_id='test',
            title='Test Issue',
            description='Test description',
            complexity='medium',
            tech_stack=tech_stack,
            constraints=constraints,
            domain='general'
        )
        print("✅ Core objects create successfully")
        
        # Test enum values
        assert PatternApplicationStatus.PENDING.value == 'pending'
        assert AdaptationStrategy.FULL_ADAPTATION.value == 'full_adaptation'
        print("✅ Enums work correctly")
        
        # Test context extractor
        from knowledge.pattern_application.context_extractor import ContextExtractor
        extractor = ContextExtractor()
        print("✅ Context extractor initializes successfully")
        
        # Test pattern matcher
        from knowledge.pattern_application.pattern_matcher import BasicPatternMatcher
        # Note: This might fail due to knowledge system dependency - that's expected
        print("✅ Pattern matcher imports successfully")
        
        print("🎉 Smoke test passed! Core functionality is working.")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Pattern Application Engine Test Suite")
    print("====================================")
    print()
    
    # Check if we should run quick test only
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Check dependencies first
    if not check_dependencies():
        print("Cannot run tests due to missing dependencies.")
        sys.exit(1)
    
    # Run smoke test
    print()
    if not run_quick_test():
        print("Smoke test failed. Skipping full test suite.")
        sys.exit(1)
    
    print()
    print("Proceeding with full test suite...")
    print()
    
    # Run full test suite
    success = run_tests()
    
    sys.exit(0 if success else 1)