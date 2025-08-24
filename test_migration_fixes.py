#!/usr/bin/env python3
"""
Test script to validate critical migration fixes.
Issue #39: Migrate from LightRAG to new system
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_database_schema_fix():
    """Test that the database schema supports knowledge types."""
    print("🧪 Testing database schema fixes...")
    
    try:
        from knowledge.database.database_interface import RIFDatabase
        from knowledge.database.database_config import DatabaseConfig
        
        # Create test database
        db_config = DatabaseConfig(
            database_path="test_migration_fixes.duckdb",
            memory_limit="100MB"
        )
        
        with RIFDatabase(db_config) as db:
            # Test storing a knowledge item with new type
            test_entity = {
                'type': 'pattern',  # This should now be allowed
                'name': 'test_pattern',
                'file_path': 'knowledge/patterns/test.json',
                'metadata': {'test': True}
            }
            
            entity_id = db.store_entity(test_entity)
            print(f"   ✅ Successfully stored entity with 'pattern' type: {entity_id}")
            
            # Test another knowledge type
            test_entity2 = {
                'type': 'decision', 
                'name': 'test_decision',
                'file_path': 'knowledge/decisions/test.json',
                'metadata': {'test': True}
            }
            
            entity_id2 = db.store_entity(test_entity2)
            print(f"   ✅ Successfully stored entity with 'decision' type: {entity_id2}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Schema test failed: {e}")
        return False
    finally:
        # Cleanup test database
        test_db = Path("test_migration_fixes.duckdb")
        if test_db.exists():
            test_db.unlink()

def test_type_mapping():
    """Test the knowledge type mapping system."""
    print("🧪 Testing knowledge type mapping...")
    
    try:
        from knowledge.migration_coordinator import HybridKnowledgeAdapter
        
        # Test the type mapping
        mapping = HybridKnowledgeAdapter.KNOWLEDGE_TYPE_MAPPING
        
        # Check expected mappings
        expected_mappings = {
            'patterns': 'pattern',
            'decisions': 'decision',
            'learnings': 'learning',
            'metrics': 'metric'
        }
        
        for collection, expected_type in expected_mappings.items():
            actual_type = mapping.get(collection, mapping['default'])
            if actual_type == expected_type:
                print(f"   ✅ {collection} -> {actual_type}")
            else:
                print(f"   ❌ {collection} -> {actual_type} (expected {expected_type})")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Type mapping test failed: {e}")
        return False

def test_state_persistence():
    """Test migration state persistence."""
    print("🧪 Testing migration state persistence...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase
        import tempfile
        import shutil
        
        # Create temporary knowledge directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create first coordinator instance
            coord1 = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Verify initial state
            assert coord1.current_phase == MigrationPhase.NOT_STARTED
            print("   ✅ Initial state loaded correctly")
            
            # Change state and save
            from datetime import datetime
            coord1.current_phase = MigrationPhase.PHASE_1_PARALLEL
            coord1.migration_start_time = datetime.now()
            coord1._save_migration_state()
            print("   ✅ State saved successfully")
            
            # Create second coordinator instance (simulating restart)
            coord2 = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Verify state was loaded
            if coord2.current_phase == MigrationPhase.PHASE_1_PARALLEL:
                print("   ✅ State persisted across instances")
                return True
            else:
                print(f"   ❌ State not persisted: {coord2.current_phase}")
                return False
                
        finally:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ❌ State persistence test failed: {e}")
        return False

def test_hybrid_adapter():
    """Test the hybrid knowledge adapter with type mapping."""
    print("🧪 Testing hybrid adapter with type mapping...")
    
    try:
        from knowledge.migration_coordinator import HybridKnowledgeAdapter
        from knowledge.database.database_interface import RIFDatabase
        from knowledge.database.database_config import DatabaseConfig
        
        # Create test database
        db_config = DatabaseConfig(
            database_path="test_hybrid_adapter.duckdb",
            memory_limit="100MB"
        )
        
        try:
            with RIFDatabase(db_config) as db:
                adapter = HybridKnowledgeAdapter(db)
                
                # Test storing knowledge items from different collections
                test_cases = [
                    ('patterns', {'title': 'Test Pattern', 'description': 'A test pattern'}),
                    ('decisions', {'title': 'Test Decision', 'rationale': 'Test reasoning'}),
                    ('metrics', {'name': 'test_metric', 'value': 42})
                ]
                
                for collection, content in test_cases:
                    result_id = adapter.store_knowledge(
                        collection=collection,
                        content=content,
                        doc_id=f"test_{collection}_1"
                    )
                    
                    if result_id:
                        print(f"   ✅ Stored {collection} knowledge item: {result_id}")
                    else:
                        print(f"   ❌ Failed to store {collection} knowledge item")
                        return False
                
                return True
                
        finally:
            # Cleanup
            test_db = Path("test_hybrid_adapter.duckdb")
            if test_db.exists():
                test_db.unlink()
        
    except Exception as e:
        print(f"   ❌ Hybrid adapter test failed: {e}")
        return False

def main():
    """Run all migration fix tests."""
    print("🔧 Testing Critical Migration Fixes - Issue #39")
    print("=" * 60)
    
    # Configure logging to show warnings and errors
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("Database Schema Fix", test_database_schema_fix),
        ("Knowledge Type Mapping", test_type_mapping), 
        ("State Persistence", test_state_persistence),
        ("Hybrid Adapter", test_hybrid_adapter)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 SUMMARY: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All critical fixes validated successfully!")
        return 0
    else:
        print("💥 Some tests failed - fixes need more work")
        return 1

if __name__ == "__main__":
    sys.exit(main())