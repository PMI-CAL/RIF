#!/usr/bin/env python3
"""
Test runner for RIF Knowledge Interface Tests

This script runs the comprehensive test suite for the knowledge management 
interface and implementations to validate the decoupling implementation.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required test dependencies are available."""
    try:
        import pytest
        logger.info("✓ pytest is available")
        return True
    except ImportError:
        logger.error("✗ pytest is not available. Install with: pip install pytest")
        return False

def run_knowledge_tests():
    """Run the knowledge interface tests."""
    test_file = Path(__file__).parent / "test_knowledge_interface.py"
    
    if not test_file.exists():
        logger.error(f"✗ Test file not found: {test_file}")
        return False
    
    logger.info("Running knowledge interface tests...")
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.stdout:
            print("\n=== TEST OUTPUT ===")
            print(result.stdout)
        
        if result.stderr:
            print("\n=== TEST ERRORS ===")
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info("✓ All tests passed!")
            return True
        else:
            logger.error(f"✗ Tests failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to run tests: {e}")
        return False

def run_manual_validation():
    """Run manual validation tests for interface functionality."""
    logger.info("Running manual validation tests...")
    
    try:
        # Add knowledge module to path
        knowledge_path = Path(__file__).parent.parent / "knowledge"
        sys.path.insert(0, str(knowledge_path))
        
        from knowledge import get_knowledge_system, MockKnowledgeAdapter
        from knowledge.lightrag_adapter import LIGHTRAG_AVAILABLE
        
        # Test 1: Factory creation
        logger.info("Test 1: Factory system...")
        try:
            knowledge = get_knowledge_system()
            logger.info(f"✓ Default knowledge system created: {type(knowledge).__name__}")
        except Exception as e:
            logger.error(f"✗ Factory creation failed: {e}")
            return False
        
        # Test 2: Mock adapter functionality
        logger.info("Test 2: Mock adapter basic functionality...")
        mock_adapter = MockKnowledgeAdapter()
        
        # Store a pattern
        pattern_data = {
            "title": "Validation Test Pattern",
            "description": "Testing pattern storage and retrieval",
            "complexity": "low",
            "tags": ["validation", "test"]
        }
        
        pattern_id = mock_adapter.store_pattern(pattern_data)
        if pattern_id:
            logger.info(f"✓ Pattern stored with ID: {pattern_id}")
        else:
            logger.error("✗ Pattern storage failed")
            return False
        
        # Search for pattern
        results = mock_adapter.search_patterns("validation")
        if results and len(results) > 0:
            logger.info(f"✓ Pattern search returned {len(results)} results")
        else:
            logger.error("✗ Pattern search failed")
            return False
        
        # Test 3: LightRAG adapter if available
        if LIGHTRAG_AVAILABLE:
            logger.info("Test 3: LightRAG adapter availability...")
            logger.info("✓ LightRAG is available for testing")
        else:
            logger.info("Test 3: LightRAG adapter...")
            logger.info("⚠ LightRAG not available - using mock adapter as default")
        
        logger.info("✓ Manual validation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Manual validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_test_report():
    """Generate a test report with system information."""
    logger.info("Generating test report...")
    
    try:
        # Add knowledge module to path
        knowledge_path = Path(__file__).parent.parent / "knowledge"
        sys.path.insert(0, str(knowledge_path))
        
        from knowledge import get_knowledge_system
        from knowledge.lightrag_adapter import LIGHTRAG_AVAILABLE
        
        knowledge = get_knowledge_system()
        system_info = knowledge.get_system_info()
        
        report = f"""
=== RIF KNOWLEDGE INTERFACE TEST REPORT ===

Implementation: {system_info.get('implementation', 'Unknown')}
Backend: {system_info.get('backend', 'Unknown')}
Version: {system_info.get('version', 'Unknown')}
LightRAG Available: {LIGHTRAG_AVAILABLE}

Features:
{chr(10).join('- ' + feature for feature in system_info.get('features', []))}

Collections:
{chr(10).join('- ' + collection for collection in system_info.get('collections', []))}

Test Status: ✓ PASSED
Decoupling Status: ✓ SUCCESSFULLY IMPLEMENTED

The RIF agents are now decoupled from LightRAG implementation and can use 
any knowledge system that implements the KnowledgeInterface.
"""
        
        report_path = Path(__file__).parent / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"✓ Test report saved to: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Report generation failed: {e}")
        return False

def main():
    """Main test runner function."""
    logger.info("=== RIF Knowledge Interface Test Runner ===")
    
    success = True
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Run automated tests
    if success:
        success = run_knowledge_tests()
    
    # Run manual validation
    if success:
        success = run_manual_validation()
    
    # Generate report
    if success:
        success = generate_test_report()
    
    # Final status
    if success:
        logger.info("🎉 ALL TESTS PASSED - Knowledge interface implementation successful!")
        return 0
    else:
        logger.error("❌ TESTS FAILED - Please check the errors above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)