"""
Comprehensive tests for RelationshipUpdater - Issue #66
Tests all aspects of relationship updating including change detection, cascade handling, and validation.

Author: RIF-Implementer  
Date: 2025-08-23
Issue: #66
"""

import unittest
import tempfile
import uuid
import json
from pathlib import Path
from typing import List, Dict
import duckdb

from .relationship_updater import (
    RelationshipUpdater, EntityChange, RelationshipDiff, RelationshipUpdateResult,
    ChangeAnalyzer, RelationshipDiffer, CascadeHandler, OrphanCleaner,
    create_relationship_updater, validate_updater_prerequisites
)
from .relationship_types import CodeRelationship, RelationshipType
from .relationship_detector import RelationshipDetector
from .storage_integration import RelationshipStorage
from ..extraction.entity_types import CodeEntity
from ..parsing.parser_manager import ParserManager


class TestRelationshipUpdater(unittest.TestCase):
    """Test cases for RelationshipUpdater functionality."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.duckdb")
        
        # Initialize test database with schema
        self._setup_test_database()
        
        # Create test dependencies
        self.parser_manager = ParserManager()
        self.detector = RelationshipDetector(self.parser_manager)
        self.storage = RelationshipStorage(self.db_path)
        
        # Create test entities
        self.test_entities = self._create_test_entities()
        
        # Initialize updater
        self.updater = RelationshipUpdater(self.detector, self.storage, self.db_path)
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_test_database(self):
        """Set up test database with required schema."""
        with duckdb.connect(self.db_path) as conn:
            # Create entities table
            conn.execute("""
                CREATE TABLE entities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    file_path VARCHAR(512) NOT NULL,
                    start_line INTEGER DEFAULT 0,
                    end_line INTEGER DEFAULT 0,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CHECK (relationship_type IN ('imports', 'calls', 'extends', 'uses', 'implements', 'references', 'contains')),
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    CHECK (source_id != target_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX idx_relationships_source ON relationships(source_id)")
            conn.execute("CREATE INDEX idx_relationships_target ON relationships(target_id)")
    
    def _create_test_entities(self) -> List[CodeEntity]:
        """Create test entities in database."""
        entities = []
        entity_data = [
            ("function_a", "function", "/test/file1.py", 1, 5),
            ("function_b", "function", "/test/file1.py", 7, 10),
            ("class_x", "class", "/test/file2.py", 1, 20),
            ("method_y", "method", "/test/file2.py", 5, 8),
        ]
        
        with duckdb.connect(self.db_path) as conn:
            for name, entity_type, file_path, start_line, end_line in entity_data:
                entity_id = str(uuid.uuid4())
                
                # Insert entity into database
                conn.execute("""
                    INSERT INTO entities (id, name, type, file_path, start_line, end_line, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [entity_id, name, entity_type, file_path, start_line, end_line, '{}'])
                
                # Create CodeEntity object
                from ..extraction.entity_types import EntityType, SourceLocation
                
                entity = CodeEntity(
                    id=uuid.UUID(entity_id) if isinstance(entity_id, str) else entity_id,
                    name=name,
                    type=EntityType(entity_type),
                    file_path=file_path,
                    location=SourceLocation(line_start=start_line, line_end=end_line),
                    metadata={}
                )
                entities.append(entity)
        
        return entities
    
    def _create_test_relationships(self) -> List[CodeRelationship]:
        """Create test relationships between entities."""
        relationships = []
        
        # function_a calls function_b
        rel1 = CodeRelationship(
            id=str(uuid.uuid4()),
            source_id=self.test_entities[0].id,  # function_a
            target_id=self.test_entities[1].id,  # function_b
            relationship_type=RelationshipType.CALLS,
            confidence=0.9,
            metadata={"line": 3}
        )
        
        # method_y extends class_x
        rel2 = CodeRelationship(
            id=str(uuid.uuid4()),
            source_id=self.test_entities[3].id,  # method_y
            target_id=self.test_entities[2].id,  # class_x
            relationship_type=RelationshipType.USES,
            confidence=0.8,
            metadata={"line": 6}
        )
        
        relationships.extend([rel1, rel2])
        
        # Store relationships in database
        self.storage.store_relationships(relationships)
        
        return relationships
    
    def test_entity_change_impact_analysis(self):
        """Test ChangeAnalyzer.analyze_impact()."""
        analyzer = ChangeAnalyzer(self.storage)
        
        entity_changes = [
            EntityChange(entity_id=self.test_entities[0].id, change_type="modified"),
            EntityChange(entity_id=self.test_entities[1].id, change_type="deleted"),
            EntityChange(entity_id=str(uuid.uuid4()), change_type="created")
        ]
        
        impact = analyzer.analyze_impact(entity_changes)
        
        # Check impact analysis results
        self.assertIn(self.test_entities[0].id, impact.reanalysis_required)
        self.assertIn(self.test_entities[1].id, impact.cascade_deletions)
        self.assertTrue(impact.orphan_cleanup_required)
        self.assertEqual(len(impact.potential_new_sources), 1)
    
    def test_relationship_diff_calculation(self):
        """Test RelationshipDiffer.calculate_relationship_diff()."""
        differ = RelationshipDiffer(self.storage)
        
        # Create old and new relationship sets
        old_relationships = self._create_test_relationships()
        
        # Modify relationships for new set
        new_relationships = old_relationships.copy()
        new_relationships[0].confidence = 0.95  # Modified confidence
        
        # Add a new relationship
        new_rel = CodeRelationship(
            id=str(uuid.uuid4()),
            source_id=self.test_entities[2].id,  # class_x
            target_id=self.test_entities[3].id,  # method_y
            relationship_type=RelationshipType.CONTAINS,
            confidence=1.0,
            metadata={}
        )
        new_relationships.append(new_rel)
        
        # Calculate diff
        diff = differ.calculate_relationship_diff(
            self.test_entities[0].id, old_relationships, new_relationships
        )
        
        # Verify diff results
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(len(diff.modified), 1)
        self.assertEqual(len(diff.removed), 0)
        self.assertGreater(len(diff.unchanged), 0)
    
    def test_cascade_handler_deletion(self):
        """Test CascadeHandler.handle_entity_deletion()."""
        # Create test relationships
        test_relationships = self._create_test_relationships()
        
        cascade_handler = CascadeHandler(self.storage, self.db_path)
        
        # Delete an entity and handle cascade
        deleted_entity_id = self.test_entities[0].id
        deleted_rel_ids = cascade_handler.handle_entity_deletion(deleted_entity_id)
        
        # Verify relationships were deleted
        self.assertGreater(len(deleted_rel_ids), 0)
        
        # Verify relationships are actually gone from database
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM relationships WHERE source_id = ? OR target_id = ?",
                [deleted_entity_id, deleted_entity_id]
            ).fetchone()
            self.assertEqual(result[0], 0)
    
    def test_cascade_handler_move(self):
        """Test CascadeHandler.handle_entity_move()."""
        # Create test relationships
        self._create_test_relationships()
        
        cascade_handler = CascadeHandler(self.storage, self.db_path)
        
        old_entity_id = self.test_entities[0].id
        new_entity_id = str(uuid.uuid4())
        
        # Handle entity move
        updated_count = cascade_handler.handle_entity_move(old_entity_id, new_entity_id)
        
        # Verify relationships were updated
        self.assertGreater(updated_count, 0)
        
        # Verify old entity ID is no longer referenced
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM relationships WHERE source_id = ? OR target_id = ?",
                [old_entity_id, old_entity_id]
            ).fetchone()
            self.assertEqual(result[0], 0)
    
    def test_orphan_cleaner(self):
        """Test OrphanCleaner.cleanup_orphans()."""
        # Create test relationships
        self._create_test_relationships()
        
        # Delete an entity to create orphans (directly from database, not through cascade)
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", [self.test_entities[0].id])
        
        orphan_cleaner = OrphanCleaner(self.db_path)
        
        # Clean up orphans
        orphaned_ids = orphan_cleaner.cleanup_orphans()
        
        # Verify orphans were found and cleaned
        self.assertGreater(len(orphaned_ids), 0)
        
        # Verify no orphans remain
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT COUNT(*) FROM relationships r
                WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.source_id)
                   OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.target_id)
            """).fetchone()
            self.assertEqual(result[0], 0)
    
    def test_relationship_updater_full_workflow(self):
        """Test complete RelationshipUpdater.update_relationships() workflow."""
        # Create initial relationships
        self._create_test_relationships()
        
        # Create entity changes
        entity_changes = [
            EntityChange(
                entity_id=self.test_entities[0].id,
                change_type="modified",
                old_entity=self.test_entities[0],
                new_entity=self.test_entities[0]  # Simplified - same entity
            ),
            EntityChange(
                entity_id=self.test_entities[1].id,
                change_type="deleted"
            )
        ]
        
        # Process updates
        result = self.updater.update_relationships(entity_changes)
        
        # Verify results
        self.assertTrue(result.success)
        self.assertEqual(result.entity_changes_processed, 2)
        self.assertGreater(result.relationships_removed, 0)  # Deletions should occur
        self.assertTrue(result.validation_passed)
        self.assertGreater(result.processing_time, 0)
        
        # Verify statistics were updated
        stats = self.updater.get_statistics()
        self.assertEqual(stats['successful_updates'], 1)
    
    def test_relationship_updater_validation(self):
        """Test RelationshipUpdater graph consistency validation."""
        # Create test relationships
        self._create_test_relationships()
        
        # Test validation on clean graph
        validation_result = self.updater._validate_graph_consistency(
            {entity.id for entity in self.test_entities}
        )
        self.assertTrue(validation_result)
        
        # Create inconsistent state (orphaned relationship)
        with duckdb.connect(self.db_path) as conn:
            # Insert relationship with non-existent target
            conn.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, [str(uuid.uuid4()), self.test_entities[0].id, str(uuid.uuid4()), "calls", 0.9])
        
        # Test validation on inconsistent graph
        validation_result = self.updater._validate_graph_consistency(
            {entity.id for entity in self.test_entities}
        )
        self.assertFalse(validation_result)
    
    def test_performance_requirements(self):
        """Test that performance requirements are met."""
        import time
        
        # Create larger dataset for performance testing
        large_entity_changes = []
        for i in range(50):  # Smaller scale for unit test
            entity_id = str(uuid.uuid4())
            large_entity_changes.append(
                EntityChange(entity_id=entity_id, change_type="created")
            )
        
        start_time = time.time()
        result = self.updater.update_relationships(large_entity_changes)
        processing_time = time.time() - start_time
        
        # Performance requirements (scaled down for unit test)
        # Original: 1000 changes in <5 seconds
        # Test: 50 changes in <1 second
        self.assertLess(processing_time, 1.0)
        self.assertTrue(result.success)
    
    def test_factory_function(self):
        """Test create_relationship_updater factory function."""
        updater = create_relationship_updater(self.db_path)
        
        self.assertIsInstance(updater, RelationshipUpdater)
        self.assertEqual(updater.database_path, self.db_path)
        self.assertIsNotNone(updater.detector)
        self.assertIsNotNone(updater.storage)
    
    def test_prerequisites_validation(self):
        """Test validate_updater_prerequisites function."""
        # Test with valid database
        valid, issues = validate_updater_prerequisites(self.db_path)
        self.assertTrue(valid)
        self.assertEqual(len(issues), 0)
        
        # Test with invalid database
        invalid_db_path = str(Path(self.temp_dir) / "nonexistent.duckdb")
        valid, issues = validate_updater_prerequisites(invalid_db_path)
        self.assertFalse(valid)
        self.assertGreater(len(issues), 0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid entity changes
        invalid_changes = [
            EntityChange(entity_id="invalid-uuid", change_type="invalid_type")
        ]
        
        result = self.updater.update_relationships(invalid_changes)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(result, RelationshipUpdateResult)
        # May succeed or fail depending on implementation, but should not crash
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create many entity changes to test batching
        many_changes = []
        for i in range(self.updater.batch_size + 10):  # Exceed batch size
            entity_id = str(uuid.uuid4())
            many_changes.append(
                EntityChange(entity_id=entity_id, change_type="created")
            )
        
        result = self.updater.update_relationships(many_changes)
        
        # Should handle batching without issues
        self.assertIsInstance(result, RelationshipUpdateResult)
        self.assertEqual(result.entity_changes_processed, len(many_changes))


class TestRelationshipUpdaterIntegration(unittest.TestCase):
    """Integration tests for RelationshipUpdater with existing systems."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "integration_test.duckdb")
        
        # Set up database with schema
        with duckdb.connect(self.db_path) as conn:
            # Load schema from existing schema file if available
            try:
                from ..schema.duckdb_schema import get_schema_sql
                schema_sql = get_schema_sql()
                conn.execute(schema_sql)
            except ImportError:
                # Fallback to basic schema
                conn.execute("""
                    CREATE TABLE entities (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        file_path VARCHAR(512) NOT NULL,
                        start_line INTEGER DEFAULT 0,
                        end_line INTEGER DEFAULT 0,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE relationships (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_id UUID NOT NULL,
                        target_id UUID NOT NULL,
                        relationship_type VARCHAR(50) NOT NULL,
                        confidence FLOAT DEFAULT 1.0,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
    
    def tearDown(self):
        """Clean up integration test resources."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cascade_system_integration(self):
        """Test integration with CascadeUpdateSystem."""
        # Import cascade system
        from ..cascade_update_system import CascadeUpdateSystem, Change
        
        # Create updater and cascade system
        updater = create_relationship_updater(self.db_path)
        cascade_system = CascadeUpdateSystem(self.db_path, updater)
        
        # Create test change
        test_change = Change(
            entity_id=str(uuid.uuid4()),
            change_type="update"
        )
        
        # Process through cascade system
        result = cascade_system.cascade_updates(test_change)
        
        # Verify integration works
        self.assertIsInstance(result.success, bool)
    
    def test_storage_integration(self):
        """Test integration with RelationshipStorage system."""
        updater = create_relationship_updater(self.db_path)
        
        # Test that storage operations work
        stats = updater.storage.get_relationship_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_relationships', stats)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)