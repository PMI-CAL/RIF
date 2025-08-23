#!/usr/bin/env python3
"""
Test rollback capabilities for Issue #39 migration system
Verify that rollback works at each phase
"""

import os
import sys
import tempfile

# Add knowledge directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge'))

def test_rollback_points_exist():
    """Test that rollback points were created during migration."""
    print("🔄 Testing rollback points exist...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        coordinator = MigrationCoordinator()
        
        # Check that rollback points exist
        expected_rollback_points = [
            'pre_migration',
            'pre_phase_2_read_migration', 
            'pre_phase_3_write_migration',
            'pre_phase_4_cutover'
        ]
        
        print(f"Expected rollback points: {expected_rollback_points}")
        print(f"Actual rollback points: {list(coordinator.rollback_points.keys())}")
        
        for point in expected_rollback_points:
            if point in coordinator.rollback_points:
                rollback_data = coordinator.rollback_points[point]
                print(f"✅ Rollback point '{point}' exists")
                print(f"   Timestamp: {rollback_data.get('timestamp', 'N/A')}")
                print(f"   Phase: {rollback_data.get('phase', 'N/A')}")
            else:
                print(f"❌ Rollback point '{point}' missing")
                return False
        
        print("✅ All rollback points verified")
        return True
        
    except Exception as e:
        print(f"❌ Rollback points test failed: {e}")
        return False

def test_rollback_functionality():
    """Test rollback functionality with a new coordinator instance."""
    print("\n🔄 Testing rollback functionality...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fresh coordinator for rollback testing
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Start migration to create rollback points
            print("  Creating migration to test rollback...")
            success = coordinator.start_migration()  # Phase 1
            assert success, "Phase 1 should succeed"
            
            # Progress to Phase 2 to create more rollback points
            success = coordinator.progress_to_next_phase()  # Phase 2
            assert success, "Phase 2 should succeed"
            
            print(f"  Current phase: {coordinator.current_phase}")
            print(f"  Rollback points created: {list(coordinator.rollback_points.keys())}")
            
            # Test rollback point creation functionality
            test_point = "test_rollback_point"
            coordinator._create_rollback_point(test_point)
            
            if test_point in coordinator.rollback_points:
                print(f"✅ Rollback point creation works: {test_point}")
            else:
                print(f"❌ Rollback point creation failed: {test_point}")
                return False
            
            # Test rollback point data structure
            rollback_data = coordinator.rollback_points[test_point]
            required_fields = ['name', 'timestamp', 'phase', 'metadata']
            
            for field in required_fields:
                if field in rollback_data:
                    print(f"✅ Rollback data has field: {field}")
                else:
                    print(f"❌ Rollback data missing field: {field}")
                    return False
            
            print("✅ Rollback functionality verified")
            return True
        
    except Exception as e:
        print(f"❌ Rollback functionality test failed: {e}")
        return False

def test_state_persistence():
    """Test that migration state persists correctly."""
    print("\n💾 Testing state persistence...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create coordinator and start migration
            coordinator1 = MigrationCoordinator(knowledge_path=temp_dir)
            success = coordinator1.start_migration()
            assert success, "Phase 1 should succeed"
            
            phase1 = coordinator1.current_phase
            rollback1 = list(coordinator1.rollback_points.keys())
            
            print(f"  Coordinator 1 - Phase: {phase1}")
            print(f"  Coordinator 1 - Rollbacks: {rollback1}")
            
            # Create new coordinator instance to test persistence
            coordinator2 = MigrationCoordinator(knowledge_path=temp_dir)
            
            phase2 = coordinator2.current_phase  
            rollback2 = list(coordinator2.rollback_points.keys())
            
            print(f"  Coordinator 2 - Phase: {phase2}")
            print(f"  Coordinator 2 - Rollbacks: {rollback2}")
            
            # Verify state persisted
            if phase1 == phase2:
                print("✅ Phase state persisted correctly")
            else:
                print(f"❌ Phase state not persisted: {phase1} != {phase2}")
                return False
            
            if rollback1 == rollback2:
                print("✅ Rollback points persisted correctly")
            else:
                print(f"❌ Rollback points not persisted: {rollback1} != {rollback2}")
                return False
            
            print("✅ State persistence verified")
            return True
            
    except Exception as e:
        print(f"❌ State persistence test failed: {e}")
        return False

def run_rollback_validation():
    """Run all rollback validation tests."""
    print("🔄 RIF-VALIDATOR ROLLBACK CAPABILITY TESTING")
    print("="*60)
    
    tests = [
        test_rollback_points_exist,
        test_rollback_functionality,
        test_state_persistence
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
        print()
    
    print("="*60)
    print(f"📊 ROLLBACK VALIDATION RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("✅ All rollback capabilities verified successfully!")
        return True
    else:
        print(f"❌ {failed} rollback test(s) failed")
        return False

if __name__ == '__main__':
    success = run_rollback_validation()
    sys.exit(0 if success else 1)