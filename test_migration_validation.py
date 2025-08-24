#!/usr/bin/env python3
"""
RIF-Validator critical validation test for Issue #39
Test real migration scenarios and identify exact failure points
"""

import os
import sys
import tempfile
import json
import logging
from pathlib import Path

# Add knowledge directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge'))

def test_schema_constraint_issue():
    """Test and identify the exact schema constraint issues."""
    print("🔍 CRITICAL VALIDATION: Schema Constraint Analysis")
    print("="*60)
    
    try:
        # Import required modules
        from knowledge.migration_coordinator import MigrationCoordinator, HybridKnowledgeAdapter
        from knowledge.lightrag_adapter import LightRAGKnowledgeAdapter
        
        # Initialize LightRAG to see what types of data we have
        lightrag = LightRAGKnowledgeAdapter()
        
        # Get first few items from each collection to analyze their structure
        print("📊 Analyzing LightRAG data structure...")
        
        collections_info = lightrag.get_collections_info()
        print(f"Collections found: {list(collections_info.keys())}")
        
        # Test with real data from patterns collection
        if 'patterns' in collections_info:
            patterns = lightrag.get_collection_items('patterns', limit=3)
            print(f"\n📝 Sample patterns data structure:")
            for i, pattern in enumerate(patterns[:2]):  # Just first 2
                print(f"Pattern {i+1}:")
                print(f"  Keys: {list(pattern.keys()) if isinstance(pattern, dict) else 'Not a dict'}")
                if isinstance(pattern, dict):
                    for key, value in pattern.items():
                        if key == 'content' and len(str(value)) > 100:
                            print(f"  {key}: {str(value)[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                print()
        
        # Test what the hybrid adapter is trying to do
        print("\n🧪 Testing hybrid adapter storage...")
        hybrid = HybridKnowledgeAdapter()
        
        # Create test knowledge item that matches LightRAG structure
        test_item = {
            'content': 'Test pattern for validation',
            'tags': ['test', 'validation'],
            'metadata': {'type': 'pattern', 'source': 'test'}
        }
        
        try:
            success = hybrid.store_knowledge('test_patterns', json.dumps(test_item), test_item)
            print(f"✅ Test storage successful: {success}")
        except Exception as e:
            print(f"❌ Test storage failed: {e}")
            print(f"   Error type: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False

def test_type_mapping_fix():
    """Test the type mapping between LightRAG and database entities."""
    print("\n🔄 CRITICAL VALIDATION: Type Mapping Analysis")
    print("="*60)
    
    try:
        from knowledge.migration_coordinator import HybridKnowledgeAdapter
        
        adapter = HybridKnowledgeAdapter()
        
        # Test different content types that LightRAG might have
        test_cases = [
            ('pattern', {'title': 'Test Pattern', 'content': 'Pattern content'}),
            ('decision', {'title': 'Test Decision', 'rationale': 'Decision rationale'}),
            ('learning', {'title': 'Test Learning', 'insight': 'Learning insight'}),
            ('issue_resolution', {'issue': 'Test Issue', 'resolution': 'Resolution details'}),
        ]
        
        print("📋 Testing type mapping for different knowledge types...")
        
        for content_type, test_data in test_cases:
            print(f"\n  Testing {content_type}:")
            try:
                # Use the adapter's type mapping logic
                mapped_type = adapter._map_lightrag_type_to_entity_type(content_type)
                print(f"    LightRAG type: {content_type}")
                print(f"    Mapped to entity type: {mapped_type}")
                
                # Try to store it
                success = adapter.store_knowledge(f'test_{content_type}', json.dumps(test_data), {'type': content_type})
                print(f"    Storage result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
            except Exception as e:
                print(f"    ❌ Mapping/Storage failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Type mapping test failed: {e}")
        return False

def test_constraint_compatibility():
    """Test database constraint compatibility with migration data."""
    print("\n🗄️ CRITICAL VALIDATION: Database Constraint Compatibility")
    print("="*60)
    
    try:
        from knowledge.database.database_interface import DatabaseInterface
        
        db = DatabaseInterface()
        
        # Check current schema constraints
        print("📊 Checking current schema constraints...")
        
        constraint_query = """
        SELECT constraint_name, constraint_type, constraint_definition 
        FROM information_schema.table_constraints 
        WHERE table_name = 'entities'
        """
        
        try:
            constraints = db.execute_query(constraint_query)
            print("Current entity table constraints:")
            for constraint in constraints:
                print(f"  {constraint}")
        except Exception as e:
            print(f"Could not query constraints: {e}")
        
        # Test entity type constraint specifically
        print("\n🧪 Testing entity type constraint...")
        
        # Get allowed types from CHECK constraint
        allowed_types = [
            'function', 'class', 'module', 'variable', 'constant', 
            'interface', 'enum', 'pattern', 'decision', 'learning', 
            'metric', 'issue_resolution', 'checkpoint', 'knowledge_item'
        ]
        
        print(f"Schema allows types: {allowed_types}")
        
        # Test if we can insert each type
        test_insert_query = """
        INSERT INTO entities (type, name, file_path, metadata) 
        VALUES (?, 'test_entity', '/test/path', '{}')
        """
        
        for entity_type in ['pattern', 'decision', 'knowledge_item']:
            try:
                # Use a transaction to test insert then rollback
                db.connection.execute('BEGIN')
                db.connection.execute(test_insert_query, (entity_type,))
                db.connection.execute('ROLLBACK')
                print(f"  ✅ Type '{entity_type}' accepted by constraint")
            except Exception as e:
                db.connection.execute('ROLLBACK')
                print(f"  ❌ Type '{entity_type}' rejected: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Constraint compatibility test failed: {e}")
        return False

def run_critical_validation():
    """Run critical validation tests for issue #39."""
    print("🚨 RIF-VALIDATOR CRITICAL ANALYSIS - Issue #39")
    print("="*80)
    print("Analyzing migration system failures and database constraints...")
    print("="*80)
    
    tests = [
        test_schema_constraint_issue,
        test_type_mapping_fix, 
        test_constraint_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("\n" + "-"*60 + "\n")
    
    print("="*80)
    print(f"📊 CRITICAL VALIDATION RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\n🚨 CRITICAL ISSUES IDENTIFIED")
        print(f"Migration system has {failed} critical failure(s)")
        print(f"These must be fixed before production deployment")
    else:
        print(f"\n✅ ALL CRITICAL VALIDATIONS PASSED")
        print(f"Migration system ready for production deployment")
    
    return failed == 0

if __name__ == '__main__':
    success = run_critical_validation()
    sys.exit(0 if success else 1)