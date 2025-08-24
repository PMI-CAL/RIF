#!/usr/bin/env python3
"""
Test script for Migration Phase 1 implementation.
Validates that the migration coordinator and Phase 1 execution work correctly.
"""

import os
import sys
import tempfile
import shutil
import json
import logging
from pathlib import Path

# Add knowledge directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge'))

def test_migration_coordinator_init():
    """Test migration coordinator initialization."""
    print("🧪 Testing MigrationCoordinator initialization...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase
        
        # Create temporary knowledge path for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Verify initial state
            assert coordinator.current_phase == MigrationPhase.NOT_STARTED
            assert coordinator.migration_start_time is None
            assert coordinator.lightrag_system is None
            assert coordinator.hybrid_system is None
            
            print("✅ MigrationCoordinator initialization test passed")
            return True
            
    except Exception as e:
        print(f"❌ MigrationCoordinator initialization test failed: {e}")
        return False


def test_migration_config():
    """Test migration configuration loading and saving."""
    print("🧪 Testing migration configuration...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Verify default configuration is loaded
            assert 'phase_durations' in coordinator.config
            assert 'rollback_conditions' in coordinator.config
            assert 'monitoring' in coordinator.config
            
            # Test config saving
            coordinator._save_migration_config()
            config_path = coordinator.migration_config_path
            assert os.path.exists(config_path)
            
            # Verify saved config can be loaded
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            assert saved_config == coordinator.config
            
            print("✅ Migration configuration test passed")
            return True
            
    except Exception as e:
        print(f"❌ Migration configuration test failed: {e}")
        return False


def test_migration_metrics():
    """Test migration metrics tracking."""
    print("🧪 Testing migration metrics...")
    
    try:
        from knowledge.migration_coordinator import MigrationMetrics
        
        metrics = MigrationMetrics()
        
        # Test operation recording
        metrics.record_operation('test_operation', 0.1, True)
        metrics.record_operation('failed_operation', 0.2, False, 'Test error')
        
        # Verify metrics
        assert metrics.operation_counts['errors'] == 1
        assert len(metrics.performance_samples) == 2
        assert len(metrics.errors) == 1
        
        # Test performance summary
        summary = metrics.get_performance_summary()
        assert summary['total_operations'] == 2
        assert summary['successful_operations'] == 1
        assert summary['failed_operations'] == 1
        assert summary['success_rate'] == 0.5
        
        print("✅ Migration metrics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Migration metrics test failed: {e}")
        return False


def test_hybrid_adapter():
    """Test hybrid knowledge adapter."""
    print("🧪 Testing HybridKnowledgeAdapter...")
    
    try:
        from knowledge.migration_coordinator import HybridKnowledgeAdapter
        
        # Mock RIFDatabase for testing
        class MockRIFDatabase:
            def store_entity(self, entity_data):
                return "test_id_123"
        
        adapter = HybridKnowledgeAdapter(MockRIFDatabase())
        
        # Test storing knowledge
        result_id = adapter.store_knowledge(
            collection="test_collection",
            content="test content",
            metadata={"test": True}
        )
        assert result_id == "test_id_123"
        
        # Test other operations
        assert adapter.update_knowledge("test", "id", "content") == True
        assert adapter.delete_knowledge("test", "id") == True
        
        stats = adapter.get_collection_stats()
        assert "hybrid_system" in stats
        
        print("✅ HybridKnowledgeAdapter test passed")
        return True
        
    except Exception as e:
        print(f"❌ HybridKnowledgeAdapter test failed: {e}")
        return False


def test_phase1_components():
    """Test Phase 1 migration components."""
    print("🧪 Testing Phase 1 migration components...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Test rollback point creation
            coordinator._create_rollback_point("test_point")
            assert "test_point" in coordinator.rollback_points
            
            # Test status reporting
            status = coordinator.get_migration_status()
            assert 'current_phase' in status
            assert 'migration_start_time' in status
            assert 'systems_status' in status
            
            print("✅ Phase 1 components test passed")
            return True
            
    except Exception as e:
        print(f"❌ Phase 1 components test failed: {e}")
        return False


def run_all_tests():
    """Run all migration tests."""
    print("🚀 Running Migration Phase 1 Tests")
    print("="*50)
    
    tests = [
        test_migration_coordinator_init,
        test_migration_config,
        test_migration_metrics,
        test_hybrid_adapter,
        test_phase1_components
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
        print()  # Empty line between tests
    
    print("="*50)
    print(f"📊 TEST RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Migration Phase 1 implementation is ready.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review implementation before proceeding.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)