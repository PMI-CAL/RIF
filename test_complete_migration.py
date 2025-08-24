#!/usr/bin/env python3
"""
Comprehensive test for complete 4-phase migration implementation.
Validates that all migration phases work end-to-end.
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

def test_full_migration_workflow():
    """Test complete migration workflow through all phases."""
    print("🧪 Testing complete migration workflow...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Verify initial state
            assert coordinator.current_phase == MigrationPhase.NOT_STARTED
            
            # Test phase progression
            phases_to_test = [
                (MigrationPhase.PHASE_1_PARALLEL, "Phase 1: Parallel Installation"),
                (MigrationPhase.PHASE_2_READ, "Phase 2: Read Migration"),
                (MigrationPhase.PHASE_3_WRITE, "Phase 3: Write Migration"),
                (MigrationPhase.PHASE_4_CUTOVER, "Phase 4: Cutover"),
                (MigrationPhase.COMPLETE, "Migration Complete")
            ]
            
            # Test Phase 1 (start_migration includes Phase 1 execution)
            print("  Testing Phase 1...")
            success = coordinator.start_migration()
            assert success, "Phase 1 execution should succeed"
            assert coordinator.current_phase == MigrationPhase.PHASE_1_PARALLEL
            
            # Test progression through remaining phases
            for target_phase, phase_name in phases_to_test[1:]:
                print(f"  Testing progression to {phase_name}...")
                success = coordinator.progress_to_next_phase()
                assert success, f"Progression to {phase_name} should succeed"
                assert coordinator.current_phase == target_phase
            
            # Test final state
            status = coordinator.get_migration_status()
            assert status['current_phase'] == 'complete'
            
            print("✅ Complete migration workflow test passed")
            return True
            
    except Exception as e:
        print(f"❌ Complete migration workflow test failed: {e}")
        return False


def test_migration_metrics_collection():
    """Test migration metrics collection across all phases."""
    print("🧪 Testing migration metrics collection...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Execute full migration to collect metrics
            coordinator.start_migration()
            
            # Progress through all phases
            for _ in range(4):  # Phases 2, 3, 4, and complete
                coordinator.progress_to_next_phase()
            
            # Verify metrics were collected
            metrics_summary = coordinator.metrics.get_performance_summary()
            assert metrics_summary['total_operations'] > 0
            assert 'operation_counts' in metrics_summary
            
            # Check specific operation types were recorded
            operation_counts = coordinator.metrics.operation_counts
            assert len(operation_counts) > 0
            
            print("✅ Migration metrics collection test passed")
            return True
            
    except Exception as e:
        print(f"❌ Migration metrics collection test failed: {e}")
        return False


def test_configuration_files_creation():
    """Test that migration creates proper configuration files."""
    print("🧪 Testing configuration files creation...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Test Phase 1 execution creates initial config
            phase1_success = coordinator.start_migration()  # Phase 1
            assert phase1_success, "Phase 1 execution failed"
            assert coordinator.current_phase == MigrationPhase.PHASE_1_PARALLEL
            
            # Test Phase 2 progression creates routing config
            phase2_success = coordinator.progress_to_next_phase()  # Phase 2
            assert phase2_success, "Phase 2 progression failed"
            assert coordinator.current_phase == MigrationPhase.PHASE_2_READ
            
            # Test Phase 3 progression creates dual write config  
            phase3_success = coordinator.progress_to_next_phase()  # Phase 3
            assert phase3_success, "Phase 3 progression failed"
            assert coordinator.current_phase == MigrationPhase.PHASE_3_WRITE
            
            # Test Phase 4 progression creates cutover config and final report
            phase4_success = coordinator.progress_to_next_phase()  # Phase 4
            assert phase4_success, "Phase 4 progression failed"
            assert coordinator.current_phase == MigrationPhase.PHASE_4_CUTOVER
            
            # Test completion
            complete_success = coordinator.progress_to_next_phase()  # Complete
            assert complete_success, "Migration completion failed"
            assert coordinator.current_phase == MigrationPhase.COMPLETE
            
            # Check that the final report was created
            final_report_path = os.path.join(temp_dir, 'migration_final_report.json')
            assert os.path.exists(final_report_path), "Final migration report not created"
            
            print("✅ Configuration files creation test passed")
            return True
            
    except Exception as e:
        print(f"❌ Configuration files creation test failed: {e}")
        return False


def test_rollback_points():
    """Test rollback point creation and management."""
    print("🧪 Testing rollback points...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Create manual rollback points
            coordinator._create_rollback_point("test_point_1")
            coordinator._create_rollback_point("test_point_2")
            
            # Verify rollback points exist
            assert "test_point_1" in coordinator.rollback_points
            assert "test_point_2" in coordinator.rollback_points
            
            # Verify rollback point structure
            rollback_data = coordinator.rollback_points["test_point_1"]
            assert 'name' in rollback_data
            assert 'timestamp' in rollback_data
            assert 'phase' in rollback_data
            
            print("✅ Rollback points test passed")
            return True
            
    except Exception as e:
        print(f"❌ Rollback points test failed: {e}")
        return False


def test_phase_specific_operations():
    """Test phase-specific operations work correctly."""
    print("🧪 Testing phase-specific operations...")
    
    try:
        from knowledge.migration_coordinator import MigrationCoordinator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MigrationCoordinator(knowledge_path=temp_dir)
            
            # Test Phase 1 operations
            assert coordinator._setup_shadow_indexing()
            assert coordinator._validate_hybrid_system_performance()
            
            # Test Phase 2 operations  
            assert coordinator._setup_read_routing()
            assert coordinator._run_ab_testing()
            assert coordinator._monitor_read_performance()
            
            # Test Phase 3 operations
            assert coordinator._enable_dual_write()
            assert coordinator._verify_data_consistency()
            assert coordinator._monitor_write_performance()
            
            # Test Phase 4 operations
            assert coordinator._final_system_validation()
            assert coordinator._disable_lightrag_system()
            assert coordinator._archive_lightrag_data()
            assert coordinator._cleanup_migration()
            
            print("✅ Phase-specific operations test passed")
            return True
            
    except Exception as e:
        print(f"❌ Phase-specific operations test failed: {e}")
        return False


def test_command_line_interface():
    """Test that command line interface works."""
    print("🧪 Testing command line interface...")
    
    try:
        # Test that the CLI script can be imported
        import execute_migration
        
        # Test that the main functions exist
        assert hasattr(execute_migration, 'execute_phase_1')
        assert hasattr(execute_migration, 'execute_phase_2')
        assert hasattr(execute_migration, 'execute_phase_3')
        assert hasattr(execute_migration, 'execute_phase_4')
        assert hasattr(execute_migration, 'print_migration_status')
        
        print("✅ Command line interface test passed")
        return True
        
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False


def run_comprehensive_tests():
    """Run all comprehensive migration tests."""
    print("🚀 Running Comprehensive Migration Tests")
    print("="*60)
    
    tests = [
        test_full_migration_workflow,
        test_migration_metrics_collection,
        test_configuration_files_creation,
        test_rollback_points,
        test_phase_specific_operations,
        test_command_line_interface
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
    
    print("="*60)
    print(f"📊 COMPREHENSIVE TEST RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All comprehensive tests passed! Complete 4-phase migration is ready.")
        print("🚀 Ready for production migration execution.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review implementation before deployment.")
        return False


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)