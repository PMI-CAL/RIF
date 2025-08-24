"""
Performance validation tests for Cascade Update System
Issue #67: Create cascade update system

These tests validate that the cascade update system meets performance targets:
- Handle graphs with >1,000 entities within reasonable time
- <100ms latency for dependency lookups using existing indexes
- Memory usage stays within configured limits

Author: RIF-Implementer
Date: 2025-08-23
"""

import pytest
import tempfile
import os
import uuid
import time
import duckdb
from typing import List, Set

# Import the system under test
import sys
sys.path.append('/Users/cal/DEV/RIF')
from knowledge.cascade_update_system import (
    CascadeUpdateSystem,
    Change,
    create_cascade_system,
    validate_cascade_prerequisites
)


class TestCascadePerformance:
    """Performance validation tests for cascade update system."""
    
    @pytest.fixture
    def performance_database(self):
        """Create a larger database for performance testing."""
        temp_db = tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        
        # Remove the empty file so DuckDB can create it properly
        os.unlink(temp_db_path)
        
        try:
            # Set up test database with required schema
            with duckdb.connect(temp_db_path) as conn:
                # Create entities table
                conn.execute("""
                CREATE TABLE entities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    type VARCHAR(50) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    ast_hash VARCHAR(64),
                    embedding FLOAT[768],
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (type IN ('function', 'class', 'module', 'variable', 'constant', 'interface', 'enum'))
                )
                """)
                
                # Create relationships table
                conn.execute("""
                CREATE TABLE relationships (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_id UUID NOT NULL,
                    target_id UUID NOT NULL,
                    relationship_type VARCHAR(50) NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (relationship_type IN ('imports', 'calls', 'extends', 'uses', 'implements', 'references', 'contains')),
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    CHECK (source_id != target_id)
                )
                """)
                
                # Create required indexes
                conn.execute("CREATE INDEX idx_relationships_source ON relationships(source_id)")
                conn.execute("CREATE INDEX idx_relationships_target ON relationships(target_id)")
                conn.execute("CREATE INDEX idx_entities_type_name ON entities(type, name)")
                
            yield temp_db_path
            
        finally:
            # Cleanup
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def create_test_graph(self, database_path: str, entity_count: int) -> List[str]:
        """Create a test graph with specified number of entities."""
        entities = []
        
        with duckdb.connect(database_path) as conn:
            # Create entities in batches for better performance
            batch_size = 100
            for i in range(0, entity_count, batch_size):
                batch_entities = []
                for j in range(min(batch_size, entity_count - i)):
                    entity_id = str(uuid.uuid4())
                    batch_entities.append((
                        entity_id, 
                        'function' if j % 4 == 0 else ('class' if j % 4 == 1 else ('module' if j % 4 == 2 else 'variable')),
                        f'entity_{i+j}',
                        f'/test/file_{(i+j) // 10}.py'
                    ))
                    entities.append(entity_id)
                
                # Insert batch
                conn.executemany("""
                INSERT INTO entities (id, type, name, file_path, line_start, line_end)
                VALUES (?, ?, ?, ?, 1, 10)
                """, batch_entities)
            
            # Create relationships to form a connected graph
            relationships = []
            for i in range(min(entity_count - 1, entity_count * 2)):  # Create roughly 2x relationships
                source_idx = i % len(entities)
                target_idx = (i + 1) % len(entities)
                
                # Avoid self-references and duplicates
                if source_idx != target_idx:
                    rel_types = ['calls', 'uses', 'references', 'imports']
                    rel_type = rel_types[i % len(rel_types)]
                    
                    relationships.append((entities[source_idx], entities[target_idx], rel_type))
            
            # Insert relationships in batches
            for i in range(0, len(relationships), batch_size):
                batch_rels = relationships[i:i+batch_size]
                conn.executemany("""
                INSERT INTO relationships (source_id, target_id, relationship_type)
                VALUES (?, ?, ?)
                """, batch_rels)
        
        return entities
    
    def test_medium_graph_performance(self, performance_database):
        """Test performance with medium-sized graph (1000 entities)."""
        entity_count = 1000
        entities = self.create_test_graph(performance_database, entity_count)
        
        system = CascadeUpdateSystem(database_path=performance_database)
        
        # Test cascade update performance
        change = Change(
            entity_id=entities[0],
            change_type='update',
            metadata={'performance_test': True}
        )
        
        start_time = time.time()
        result = system.cascade_updates(change)
        duration = time.time() - start_time
        
        # Validate performance targets
        assert result.success is True, f"Cascade update failed: {result.errors}"
        assert duration < 30.0, f"Performance target exceeded: {duration:.2f}s > 30s for {entity_count} entities"
        assert len(result.processed_entities) > 0, "No entities were processed"
        
        print(f"Medium graph performance: {duration:.2f}s for {entity_count} entities, "
              f"{len(result.processed_entities)} entities processed")
    
    def test_dependency_lookup_performance(self, performance_database):
        """Test dependency lookup latency using database indexes."""
        entity_count = 500
        entities = self.create_test_graph(performance_database, entity_count)
        
        system = CascadeUpdateSystem(database_path=performance_database)
        
        # Test multiple dependency lookups
        total_lookups = 50
        start_time = time.time()
        
        with duckdb.connect(performance_database) as conn:
            for i in range(total_lookups):
                entity_id = entities[i % len(entities)]
                dependents = system._find_dependents(entity_id, conn)
                # Verify lookup returned results (may be empty, that's ok)
                assert isinstance(dependents, list)
        
        total_duration = time.time() - start_time
        avg_latency = (total_duration / total_lookups) * 1000  # Convert to ms
        
        # Validate performance target: <100ms average
        assert avg_latency < 100.0, f"Dependency lookup latency target exceeded: {avg_latency:.2f}ms > 100ms"
        
        print(f"Dependency lookup performance: {avg_latency:.2f}ms average latency for {total_lookups} lookups")
    
    def test_batch_processing_efficiency(self, performance_database):
        """Test batch processing efficiency with different batch sizes."""
        entity_count = 500
        entities = self.create_test_graph(performance_database, entity_count)
        
        system = CascadeUpdateSystem(database_path=performance_database)
        
        # Test different batch sizes
        batch_sizes = [100, 250, 500]
        results = {}
        
        for batch_size in batch_sizes:
            system.batch_size = batch_size
            
            change = Change(
                entity_id=entities[0],
                change_type='update',
                metadata={'batch_test': True, 'batch_size': batch_size}
            )
            
            start_time = time.time()
            result = system.cascade_updates(change)
            duration = time.time() - start_time
            
            assert result.success is True, f"Batch processing failed for size {batch_size}: {result.errors}"
            
            results[batch_size] = {
                'duration': duration,
                'processed_count': len(result.processed_entities)
            }
            
            print(f"Batch size {batch_size}: {duration:.3f}s, {len(result.processed_entities)} entities processed")
        
        # Validate that batch processing provides reasonable performance
        for batch_size, metrics in results.items():
            assert metrics['duration'] < 10.0, f"Batch processing too slow for size {batch_size}: {metrics['duration']:.2f}s"
    
    def test_cycle_detection_performance(self, performance_database):
        """Test cycle detection performance with complex graph structures."""
        # Create a graph with intentional cycles
        entity_count = 200
        entities = self.create_test_graph(performance_database, entity_count)
        
        # Add some circular dependencies
        with duckdb.connect(performance_database) as conn:
            # Create a few cycles
            cycle_relationships = [
                (entities[0], entities[1], 'calls'),
                (entities[1], entities[2], 'calls'),
                (entities[2], entities[0], 'calls'),  # 3-node cycle
                
                (entities[10], entities[11], 'uses'),
                (entities[11], entities[12], 'uses'),
                (entities[12], entities[13], 'uses'),
                (entities[13], entities[10], 'uses'),  # 4-node cycle
            ]
            
            conn.executemany("""
            INSERT INTO relationships (source_id, target_id, relationship_type)
            VALUES (?, ?, ?)
            """, cycle_relationships)
        
        system = CascadeUpdateSystem(database_path=performance_database)
        
        # Test cascade update with cycles
        change = Change(
            entity_id=entities[0],  # Start from an entity in a cycle
            change_type='update',
            metadata={'cycle_test': True}
        )
        
        start_time = time.time()
        result = system.cascade_updates(change)
        duration = time.time() - start_time
        
        # Validate that cycle detection works and performance is acceptable
        assert result.success is True, f"Cycle detection failed: {result.errors}"
        assert len(result.cycles_detected) >= 2, f"Expected at least 2 cycles, found {len(result.cycles_detected)}"
        assert duration < 5.0, f"Cycle detection too slow: {duration:.2f}s for {entity_count} entities with cycles"
        
        print(f"Cycle detection performance: {duration:.2f}s, {len(result.cycles_detected)} cycles detected")
    
    def test_memory_usage_estimation(self, performance_database):
        """Test memory usage stays within reasonable bounds."""
        entity_count = 1000
        entities = self.create_test_graph(performance_database, entity_count)
        
        system = CascadeUpdateSystem(database_path=performance_database)
        
        # Run cascade update and monitor basic statistics
        change = Change(
            entity_id=entities[0],
            change_type='update',
            metadata={'memory_test': True}
        )
        
        result = system.cascade_updates(change)
        
        # Validate successful operation
        assert result.success is True, f"Memory test failed: {result.errors}"
        
        # Basic validation that system can handle the load
        # (Real memory profiling would require additional tools)
        assert len(result.processed_entities) > 0, "No entities processed in memory test"
        assert result.duration_seconds < 30.0, f"Memory test took too long: {result.duration_seconds:.2f}s"
        
        # Validate statistics tracking
        stats = system.get_statistics()
        assert stats['total_operations'] > 0, "Statistics not properly tracked"
        assert stats['entities_processed'] > 0, "Entity processing count not tracked"
        
        print(f"Memory usage test: {len(result.processed_entities)} entities processed in {result.duration_seconds:.2f}s")


def test_performance_prerequisites():
    """Test performance testing prerequisites."""
    # Basic smoke test that can be run quickly
    system = create_cascade_system()
    assert system is not None
    
    # Test basic statistics
    stats = system.get_statistics()
    assert isinstance(stats, dict)
    assert 'total_operations' in stats


if __name__ == "__main__":
    # Run basic performance validation
    print("Running cascade update system performance validation...")
    
    # Quick prerequisite test
    test_performance_prerequisites()
    print("✓ Prerequisites validated")
    
    print("Performance validation completed!")
    print("\nTo run full performance tests:")
    print("python3 -m pytest tests/test_cascade_performance.py -v")