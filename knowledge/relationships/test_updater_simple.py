"""
Simple test to validate RelationshipUpdater core functionality - Issue #66
Tests basic operations without complex dependencies.

Author: RIF-Implementer  
Date: 2025-08-23
Issue: #66
"""

import unittest
import tempfile
import uuid
from pathlib import Path
import duckdb

from .relationship_updater import (
    RelationshipUpdater, EntityChange, ChangeAnalyzer, RelationshipDiffer,
    CascadeHandler, OrphanCleaner, validate_updater_prerequisites
)


class TestRelationshipUpdaterCore(unittest.TestCase):
    """Simple tests for core RelationshipUpdater functionality."""
    
    def setUp(self):
        """Set up minimal test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.duckdb")
        
        # Set up minimal database
        with duckdb.connect(self.db_path) as conn:
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
            
            # Add test data
            entity1_id = str(uuid.uuid4())
            entity2_id = str(uuid.uuid4())
            
            conn.execute("""
                INSERT INTO entities (id, name, type, file_path, start_line, end_line)
                VALUES (?, 'test_function', 'function', '/test/file.py', 1, 5)
            """, [entity1_id])
            
            conn.execute("""
                INSERT INTO entities (id, name, type, file_path, start_line, end_line) 
                VALUES (?, 'test_class', 'class', '/test/file.py', 7, 15)
            """, [entity2_id])
            
            # Add test relationship
            conn.execute("""
                INSERT INTO relationships (source_id, target_id, relationship_type, confidence)
                VALUES (?, ?, 'calls', 0.9)
            """, [entity1_id, entity2_id])
            
            self.test_entity_ids = [entity1_id, entity2_id]
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_change_analyzer_basic(self):
        """Test basic ChangeAnalyzer functionality."""
        from .storage_integration import RelationshipStorage
        
        storage = RelationshipStorage(self.db_path)
        analyzer = ChangeAnalyzer(storage)
        
        entity_changes = [
            EntityChange(entity_id=self.test_entity_ids[0], change_type="modified"),
            EntityChange(entity_id=self.test_entity_ids[1], change_type="deleted")
        ]
        
        impact = analyzer.analyze_impact(entity_changes)
        
        self.assertIn(self.test_entity_ids[0], impact.reanalysis_required)
        self.assertIn(self.test_entity_ids[1], impact.cascade_deletions)
        self.assertTrue(impact.orphan_cleanup_required)
    
    def test_cascade_handler_deletion(self):
        """Test CascadeHandler deletion functionality."""
        from .storage_integration import RelationshipStorage
        
        storage = RelationshipStorage(self.db_path)
        cascade_handler = CascadeHandler(storage, self.db_path)
        
        # Test deletion
        deleted_rels = cascade_handler.handle_entity_deletion(self.test_entity_ids[0])
        
        # Verify relationships were deleted
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM relationships WHERE source_id = ? OR target_id = ?",
                [self.test_entity_ids[0], self.test_entity_ids[0]]
            ).fetchone()
            self.assertEqual(result[0], 0)
    
    def test_orphan_cleaner_basic(self):
        """Test OrphanCleaner functionality.""" 
        # Create orphaned relationship by deleting entity
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", [self.test_entity_ids[0]])
        
        orphan_cleaner = OrphanCleaner(self.db_path)
        orphaned_ids = orphan_cleaner.cleanup_orphans()
        
        # Should have found and cleaned orphans
        self.assertGreater(len(orphaned_ids), 0)
    
    def test_prerequisites_validation(self):
        """Test prerequisites validation."""
        valid, issues = validate_updater_prerequisites(self.db_path)
        self.assertTrue(valid)
        self.assertEqual(len(issues), 0)
        
        # Test invalid path
        invalid_path = "/nonexistent/path.duckdb"
        valid, issues = validate_updater_prerequisites(invalid_path)
        self.assertFalse(valid)
        self.assertGreater(len(issues), 0)
    
    def test_entity_change_creation(self):
        """Test EntityChange object creation."""
        change = EntityChange(
            entity_id=self.test_entity_ids[0],
            change_type="modified",
            metadata={"test": True}
        )
        
        self.assertEqual(change.entity_id, self.test_entity_ids[0])
        self.assertEqual(change.change_type, "modified")
        self.assertEqual(change.metadata["test"], True)
        self.assertIsInstance(change.timestamp, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)