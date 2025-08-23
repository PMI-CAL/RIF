"""
Test script for RIF Agent LightRAG integration.
Verifies that agent integration modules work correctly.
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import agent_integration
        print("✓ agent_integration imported successfully")
    except Exception as e:
        print(f"✗ Failed to import agent_integration: {e}")
        return False
    
    try:
        import utils
        print("✓ utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        import examples
        print("✓ examples imported successfully")
    except Exception as e:
        print(f"✗ Failed to import examples: {e}")
        return False
    
    return True


def test_agent_creation():
    """Test that agent RAG instances can be created."""
    print("\nTesting agent creation...")
    
    try:
        from agent_integration import create_agent_rag
        
        agent_types = ['analyst', 'architect', 'implementer', 'validator', 'planner']
        
        for agent_type in agent_types:
            try:
                agent = create_agent_rag(agent_type)
                print(f"✓ Created {agent_type} agent successfully")
            except Exception as e:
                print(f"✗ Failed to create {agent_type} agent: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test agent creation: {e}")
        return False


def test_lightrag_connection():
    """Test connection to LightRAG core."""
    print("\nTesting LightRAG connection...")
    
    try:
        from agent_integration import get_analyst_rag
        
        analyst = get_analyst_rag()
        
        # Try to get collection stats
        stats = analyst.get_agent_stats()
        print(f"✓ LightRAG connection successful - {len(stats)} collections found")
        
        return True
        
    except Exception as e:
        print(f"✗ LightRAG connection failed: {e}")
        traceback.print_exc()
        return False


def test_knowledge_operations():
    """Test basic knowledge operations."""
    print("\nTesting knowledge operations...")
    
    try:
        from agent_integration import get_analyst_rag
        
        analyst = get_analyst_rag()
        
        # Test storing knowledge
        test_content = json.dumps({
            "test_pattern": "This is a test pattern for integration testing",
            "timestamp": datetime.utcnow().isoformat(),
            "test_id": "integration_test_001"
        })
        
        test_metadata = {
            "type": "test",
            "source": "integration_test",
            "complexity": "low"
        }
        
        try:
            doc_id = analyst.capture_knowledge(test_content, "pattern", test_metadata)
            print(f"✓ Knowledge stored successfully: {doc_id}")
        except Exception as e:
            print(f"✗ Failed to store knowledge: {e}")
            return False
        
        # Test querying knowledge
        try:
            results = analyst.query_similar_work("test pattern integration")
            print(f"✓ Knowledge query successful: {len(results)} results")
        except Exception as e:
            print(f"✗ Failed to query knowledge: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Knowledge operations test failed: {e}")
        traceback.print_exc()
        return False


def test_agent_specific_methods():
    """Test agent-specific methods."""
    print("\nTesting agent-specific methods...")
    
    try:
        from agent_integration import (
            get_analyst_rag, get_architect_rag, get_implementer_rag,
            get_validator_rag, get_planner_rag
        )
        
        # Test analyst methods
        try:
            analyst = get_analyst_rag()
            similar_issues = analyst.find_similar_issues("test issue description")
            patterns = analyst.find_relevant_patterns("test requirements")
            print(f"✓ Analyst methods work: {len(similar_issues)} issues, {len(patterns)} patterns")
        except Exception as e:
            print(f"✗ Analyst methods failed: {e}")
            return False
        
        # Test architect methods
        try:
            architect = get_architect_rag()
            decisions = architect.find_architectural_decisions("test architecture context")
            patterns = architect.find_design_patterns("test requirements")
            print(f"✓ Architect methods work: {len(decisions)} decisions, {len(patterns)} patterns")
        except Exception as e:
            print(f"✗ Architect methods failed: {e}")
            return False
        
        # Test implementer methods
        try:
            implementer = get_implementer_rag()
            code = implementer.find_code_examples("test functionality")
            patterns = implementer.find_implementation_patterns("test task")
            print(f"✓ Implementer methods work: {len(code)} examples, {len(patterns)} patterns")
        except Exception as e:
            print(f"✗ Implementer methods failed: {e}")
            return False
        
        # Test validator methods
        try:
            validator = get_validator_rag()
            test_patterns = validator.find_test_patterns("test functionality")
            gates = validator.find_quality_gates("web")
            print(f"✓ Validator methods work: {len(test_patterns)} patterns, {len(gates)} gates")
        except Exception as e:
            print(f"✗ Validator methods failed: {e}")
            return False
        
        # Test planner methods
        try:
            planner = get_planner_rag()
            templates = planner.find_planning_templates("medium")
            workflows = planner.find_workflow_patterns("test requirements")
            print(f"✓ Planner methods work: {len(templates)} templates, {len(workflows)} workflows")
        except Exception as e:
            print(f"✗ Planner methods failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Agent-specific methods test failed: {e}")
        traceback.print_exc()
        return False


def test_utils_functionality():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import agent_knowledge_session, KnowledgePatternMatcher
        from agent_integration import get_analyst_rag
        
        # Test session management
        try:
            with agent_knowledge_session('analyst', 'test_session') as session:
                results = session.query_knowledge("test query")
                summary = session.get_session_summary()
                print(f"✓ Session management works: {summary['queries_performed']} queries performed")
        except Exception as e:
            print(f"✗ Session management failed: {e}")
            return False
        
        # Test pattern matcher
        try:
            analyst = get_analyst_rag()
            matcher = KnowledgePatternMatcher(analyst)
            evolution = matcher.find_evolution_patterns("test entity", 30)
            success = matcher.find_success_patterns("test domain")
            anti = matcher.find_anti_patterns("test domain")
            print(f"✓ Pattern matcher works: {len(evolution)} evolution, {len(success)} success, {len(anti)} anti-patterns")
        except Exception as e:
            print(f"✗ Pattern matcher failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Utils functionality test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("RIF Agent LightRAG Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Creation Test", test_agent_creation),
        ("LightRAG Connection Test", test_lightrag_connection),
        ("Knowledge Operations Test", test_knowledge_operations),
        ("Agent-Specific Methods Test", test_agent_specific_methods),
        ("Utils Functionality Test", test_utils_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent integration is working correctly.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)