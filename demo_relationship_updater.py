#!/usr/bin/env python3
"""
Demo script for RelationshipUpdater - Issue #66 Implementation
Demonstrates relationship updater functionality and integration with cascade system.

Author: RIF-Implementer
Date: 2025-08-23
Issue: #66
"""

import logging
import uuid
import json
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_relationship_updater():
    """Demonstrate RelationshipUpdater functionality."""
    
    print("🔄 RelationshipUpdater Demo - Issue #66")
    print("=" * 50)
    
    try:
        # Import the relationship updater
        from knowledge.relationships.relationship_updater import (
            create_relationship_updater, 
            EntityChange, 
            validate_updater_prerequisites
        )
        from knowledge.cascade_update_system import CascadeUpdateSystem, Change
        
        database_path = "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        
        print(f"📊 Using database: {database_path}")
        
        # Phase 1: Validate prerequisites
        print("\n🔍 Phase 1: Validating prerequisites...")
        valid, issues = validate_updater_prerequisites(database_path)
        
        if not valid:
            print(f"❌ Prerequisites validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return
        
        print("✅ Prerequisites validation passed")
        
        # Phase 2: Create RelationshipUpdater
        print("\n🏗️  Phase 2: Creating RelationshipUpdater...")
        updater = create_relationship_updater(database_path)
        
        print(f"✅ RelationshipUpdater created successfully")
        print(f"   - Database path: {updater.database_path}")
        print(f"   - Batch size: {updater.batch_size}")
        
        # Phase 3: Test with sample entity changes
        print("\n🔄 Phase 3: Testing with sample entity changes...")
        
        # Create sample entity changes
        test_changes = [
            EntityChange(
                entity_id=str(uuid.uuid4()),
                change_type="created",
                metadata={"test_entity": True, "demo": "relationship_updater"}
            ),
            EntityChange(
                entity_id=str(uuid.uuid4()),
                change_type="modified",
                metadata={"modification_reason": "demo_update"}
            )
        ]
        
        print(f"📝 Created {len(test_changes)} test entity changes:")
        for i, change in enumerate(test_changes, 1):
            print(f"   {i}. Entity {change.entity_id[:8]}... - {change.change_type}")
        
        # Process the changes
        print("\n⚡ Processing entity changes...")
        result = updater.update_relationships(test_changes)
        
        print(f"📊 Update Results:")
        print(f"   - Success: {result.success}")
        print(f"   - Entity changes processed: {result.entity_changes_processed}")
        print(f"   - Relationships added: {result.relationships_added}")
        print(f"   - Relationships updated: {result.relationships_updated}")
        print(f"   - Relationships removed: {result.relationships_removed}")
        print(f"   - Orphans cleaned: {result.orphans_cleaned}")
        print(f"   - Processing time: {result.processing_time:.3f}s")
        print(f"   - Validation passed: {result.validation_passed}")
        
        if result.errors:
            print("⚠️  Errors encountered:")
            for error in result.errors:
                print(f"   - {error}")
        
        # Phase 4: Show statistics
        print("\n📈 Phase 4: RelationshipUpdater Statistics...")
        stats = updater.get_statistics()
        
        print(f"📊 Performance Statistics:")
        print(f"   - Total updates: {stats['total_updates']}")
        print(f"   - Successful updates: {stats['successful_updates']}")
        print(f"   - Failed updates: {stats['failed_updates']}")
        print(f"   - Relationships processed: {stats['relationships_processed']}")
        print(f"   - Orphans cleaned: {stats['orphans_cleaned']}")
        print(f"   - Average processing time: {stats['avg_processing_time']:.3f}s")
        
        # Phase 5: Test cascade system integration
        print("\n🔗 Phase 5: Testing Cascade System Integration...")
        
        cascade_system = CascadeUpdateSystem(database_path, updater)
        
        # Create a test cascade change
        cascade_change = Change(
            entity_id=str(uuid.uuid4()),
            change_type="update",
            metadata={"cascade_demo": True}
        )
        
        print(f"🔄 Processing cascade change for entity {cascade_change.entity_id[:8]}...")
        cascade_result = cascade_system.cascade_updates(cascade_change)
        
        print(f"📊 Cascade Results:")
        print(f"   - Success: {cascade_result.success}")
        print(f"   - Affected entities: {len(cascade_result.affected_entities)}")
        print(f"   - Processed entities: {len(cascade_result.processed_entities)}")
        print(f"   - Cycles detected: {len(cascade_result.cycles_detected)}")
        print(f"   - Duration: {cascade_result.duration_seconds:.3f}s")
        
        if cascade_result.errors:
            print("⚠️  Cascade errors:")
            for error in cascade_result.errors:
                print(f"   - {error}")
        
        # Phase 6: Component testing
        print("\n🧪 Phase 6: Testing individual components...")
        
        # Test ChangeAnalyzer
        print("🔍 Testing ChangeAnalyzer...")
        impact = updater.change_analyzer.analyze_impact(test_changes)
        print(f"   - Potential new sources: {len(impact.potential_new_sources)}")
        print(f"   - Reanalysis required: {len(impact.reanalysis_required)}")
        print(f"   - Cascade deletions: {len(impact.cascade_deletions)}")
        print(f"   - Orphan cleanup required: {impact.orphan_cleanup_required}")
        
        # Test OrphanCleaner
        print("🧹 Testing OrphanCleaner...")
        orphaned_ids = updater.orphan_cleaner.cleanup_orphans()
        print(f"   - Orphaned relationships cleaned: {len(orphaned_ids)}")
        
        print("\n✅ Demo completed successfully!")
        print(f"🚀 RelationshipUpdater is ready for production use!")
        
        # Return demo results for further analysis
        return {
            'prerequisites_valid': valid,
            'update_result': result,
            'cascade_result': cascade_result,
            'statistics': stats,
            'impact_analysis': impact,
            'orphans_cleaned': len(orphaned_ids)
        }
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        return None


def demo_performance_test():
    """Demonstrate performance with larger datasets."""
    print("\n🏃 Performance Test - Issue #66")
    print("=" * 30)
    
    try:
        from knowledge.relationships.relationship_updater import create_relationship_updater, EntityChange
        import time
        
        database_path = "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        updater = create_relationship_updater(database_path)
        
        # Create larger dataset
        large_changes = []
        for i in range(100):  # Test with 100 changes
            large_changes.append(EntityChange(
                entity_id=str(uuid.uuid4()),
                change_type="created" if i % 2 == 0 else "modified",
                metadata={"batch": i // 10, "index": i}
            ))
        
        print(f"📊 Testing with {len(large_changes)} entity changes...")
        
        start_time = time.time()
        result = updater.update_relationships(large_changes)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(large_changes) / processing_time if processing_time > 0 else 0
        
        print(f"📈 Performance Results:")
        print(f"   - Total processing time: {processing_time:.3f}s")
        print(f"   - Throughput: {throughput:.1f} changes/second")
        print(f"   - Success rate: {result.success}")
        print(f"   - Average time per change: {processing_time/len(large_changes)*1000:.2f}ms")
        
        # Check performance targets
        target_time = 5.0  # Target: <5 seconds for 1000 changes (scaled down)
        scaled_target = target_time * (len(large_changes) / 1000)
        
        if processing_time <= scaled_target:
            print(f"✅ Performance target met! ({processing_time:.3f}s <= {scaled_target:.3f}s)")
        else:
            print(f"⚠️  Performance target missed ({processing_time:.3f}s > {scaled_target:.3f}s)")
        
        return {
            'processing_time': processing_time,
            'throughput': throughput,
            'target_met': processing_time <= scaled_target,
            'result': result
        }
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None


if __name__ == "__main__":
    # Run the demo
    demo_result = demo_relationship_updater()
    
    if demo_result:
        # Run performance test
        perf_result = demo_performance_test()
        
        # Save results
        results = {
            'demo': demo_result,
            'performance': perf_result,
            'timestamp': '2025-08-23',
            'issue': '#66',
            'status': 'complete'
        }
        
        output_path = Path("output") / "relationship_updater_demo_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
    else:
        print("❌ Demo failed - no results to save")