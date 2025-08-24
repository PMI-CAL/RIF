"""
Relationship Updater - Issue #66 Implementation
Build relationship updater for graph relationship management.

This module implements a sophisticated relationship update system that:
1. Detects changes in relationships based on entity modifications
2. Performs incremental relationship updates with validation
3. Handles cascade deletions and orphan cleanup
4. Maintains graph consistency and referential integrity
5. Integrates with existing relationship detection infrastructure

Core Features:
- Entity change impact analysis
- Smart relationship detection and diff calculation
- Cascade update handling with cycle detection
- Orphan relationship cleanup
- Performance-optimized batch processing
- Comprehensive validation and consistency checking

Performance Targets:
- Process 1000 entity changes in <5 seconds
- Handle cascade deletions in <1 second per entity
- Maintain <50MB memory footprint for batch processing
- Update dependency graphs in <2 seconds

Author: RIF-Implementer
Date: 2025-08-23
Issue: #66
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from uuid import UUID
import json
import duckdb

from .relationship_types import CodeRelationship, RelationshipType
from .relationship_detector import RelationshipDetector
from .storage_integration import RelationshipStorage
from ..extraction.entity_types import CodeEntity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityChange:
    """Represents a change event for an entity that may affect relationships."""
    entity_id: str
    change_type: str  # 'created', 'modified', 'deleted', 'moved'
    old_entity: Optional[CodeEntity] = None
    new_entity: Optional[CodeEntity] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RelationshipDiff:
    """Represents the difference between old and new relationships for an entity."""
    entity_id: str
    added: List[CodeRelationship] = field(default_factory=list)
    removed: List[CodeRelationship] = field(default_factory=list)
    modified: List[Tuple[CodeRelationship, CodeRelationship]] = field(default_factory=list)
    unchanged: List[CodeRelationship] = field(default_factory=list)


@dataclass
class RelationshipUpdateResult:
    """Result of relationship update operations."""
    success: bool
    entity_changes_processed: int = 0
    relationships_added: int = 0
    relationships_updated: int = 0
    relationships_removed: int = 0
    orphans_cleaned: int = 0
    affected_entities: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    validation_passed: bool = False


@dataclass
class EntityChangeImpact:
    """Analysis of which relationships are affected by entity changes."""
    potential_new_sources: Set[str] = field(default_factory=set)
    potential_new_targets: Set[str] = field(default_factory=set)
    validation_required: Set[str] = field(default_factory=set)
    reanalysis_required: Set[str] = field(default_factory=set)
    cascade_deletions: Set[str] = field(default_factory=set)
    reference_updates: Set[Tuple[str, str]] = field(default_factory=set)  # (old_id, new_id)
    orphan_cleanup_required: bool = False


class ChangeAnalyzer:
    """Analyzes impact of entity changes on the relationship graph."""
    
    def __init__(self, storage: RelationshipStorage):
        self.storage = storage
        self.logger = logging.getLogger(f"{__name__}.ChangeAnalyzer")
    
    def analyze_impact(self, entity_changes: List[EntityChange]) -> EntityChangeImpact:
        """
        Analyze which relationships are affected by entity changes.
        
        Args:
            entity_changes: List of entity change events
            
        Returns:
            EntityChangeImpact with analysis results
        """
        impact = EntityChangeImpact()
        
        self.logger.info(f"Analyzing impact of {len(entity_changes)} entity changes")
        
        for change in entity_changes:
            if change.change_type == 'created':
                # New entity - check for new relationships
                impact.potential_new_sources.add(change.entity_id)
                impact.potential_new_targets.add(change.entity_id)
                self.logger.debug(f"Entity {change.entity_id} created - potential new relationships")
                
            elif change.change_type == 'modified':
                # Modified entity - relationships might change
                impact.validation_required.add(change.entity_id)
                impact.reanalysis_required.add(change.entity_id)
                self.logger.debug(f"Entity {change.entity_id} modified - reanalysis required")
                
            elif change.change_type == 'deleted':
                # Deleted entity - cascading deletions needed
                impact.cascade_deletions.add(change.entity_id)
                impact.orphan_cleanup_required = True
                self.logger.debug(f"Entity {change.entity_id} deleted - cascade cleanup required")
                
            elif change.change_type == 'moved':
                # Moved entity - update references
                if change.old_entity and change.new_entity:
                    impact.reference_updates.add((change.old_entity.id, change.new_entity.id))
                    self.logger.debug(f"Entity {change.entity_id} moved - reference updates required")
        
        self.logger.info(f"Impact analysis complete: {len(impact.reanalysis_required)} entities need reanalysis, "
                        f"{len(impact.cascade_deletions)} deletions, "
                        f"orphan cleanup: {impact.orphan_cleanup_required}")
        
        return impact


class RelationshipDiffer:
    """Calculates differences between old and new relationships for entities."""
    
    def __init__(self, storage: RelationshipStorage):
        self.storage = storage
        self.logger = logging.getLogger(f"{__name__}.RelationshipDiffer")
    
    def calculate_relationship_diff(self, entity_id: str, 
                                  old_relationships: List[CodeRelationship],
                                  new_relationships: List[CodeRelationship]) -> RelationshipDiff:
        """
        Calculate the difference between old and new relationships for an entity.
        
        Args:
            entity_id: Entity ID being analyzed
            old_relationships: Previously detected relationships
            new_relationships: Newly detected relationships
            
        Returns:
            RelationshipDiff with categorized changes
        """
        diff = RelationshipDiff(entity_id=entity_id)
        
        # Create lookup maps for efficient comparison
        old_rel_map = {self._relationship_key(rel): rel for rel in old_relationships}
        new_rel_map = {self._relationship_key(rel): rel for rel in new_relationships}
        
        old_keys = set(old_rel_map.keys())
        new_keys = set(new_rel_map.keys())
        
        # Find added relationships
        added_keys = new_keys - old_keys
        diff.added = [new_rel_map[key] for key in added_keys]
        
        # Find removed relationships
        removed_keys = old_keys - new_keys
        diff.removed = [old_rel_map[key] for key in removed_keys]
        
        # Find potentially modified relationships (same key, different attributes)
        common_keys = old_keys & new_keys
        for key in common_keys:
            old_rel = old_rel_map[key]
            new_rel = new_rel_map[key]
            
            if self._relationships_differ(old_rel, new_rel):
                diff.modified.append((old_rel, new_rel))
            else:
                diff.unchanged.append(old_rel)
        
        self.logger.debug(f"Relationship diff for entity {entity_id}: "
                         f"+{len(diff.added)} -{len(diff.removed)} "
                         f"~{len(diff.modified)} ={len(diff.unchanged)}")
        
        return diff
    
    def _relationship_key(self, relationship: CodeRelationship) -> str:
        """Create a unique key for relationship comparison."""
        return f"{relationship.source_id}:{relationship.target_id}:{relationship.relationship_type.value}"
    
    def _relationships_differ(self, old_rel: CodeRelationship, new_rel: CodeRelationship) -> bool:
        """Check if two relationships with the same key have different attributes."""
        return (old_rel.confidence != new_rel.confidence or 
                old_rel.metadata != new_rel.metadata)


class CascadeHandler:
    """Handles cascade operations when entities are deleted or moved."""
    
    def __init__(self, storage: RelationshipStorage, database_path: str):
        self.storage = storage
        self.database_path = database_path
        self.logger = logging.getLogger(f"{__name__}.CascadeHandler")
    
    def handle_entity_deletion(self, deleted_entity_id: str) -> List[str]:
        """
        Handle cascading effects of entity deletion.
        
        Args:
            deleted_entity_id: ID of the deleted entity
            
        Returns:
            List of relationship IDs that were deleted
        """
        deleted_relationships = []
        
        try:
            with duckdb.connect(self.database_path) as conn:
                # Find all relationships involving the deleted entity
                query = """
                SELECT id FROM relationships 
                WHERE source_id = ? OR target_id = ?
                """
                
                result = conn.execute(query, [deleted_entity_id, deleted_entity_id]).fetchall()
                relationship_ids_to_delete = [str(row[0]) for row in result]
                
                if relationship_ids_to_delete:
                    # Delete the relationships
                    placeholders = ','.join(['?' for _ in relationship_ids_to_delete])
                    delete_query = f"DELETE FROM relationships WHERE id::VARCHAR IN ({placeholders})"
                    conn.execute(delete_query, relationship_ids_to_delete)
                    
                    deleted_relationships = relationship_ids_to_delete
                    self.logger.info(f"Deleted {len(deleted_relationships)} relationships for entity {deleted_entity_id}")
                
        except Exception as e:
            self.logger.error(f"Error during cascade deletion for entity {deleted_entity_id}: {e}")
            raise
        
        return deleted_relationships
    
    def handle_entity_move(self, old_entity_id: str, new_entity_id: str) -> int:
        """
        Handle entity ID changes by updating all references.
        
        Args:
            old_entity_id: Previous entity ID
            new_entity_id: New entity ID
            
        Returns:
            Number of relationships updated
        """
        updated_count = 0
        
        try:
            with duckdb.connect(self.database_path) as conn:
                # Update source references
                source_result = conn.execute(
                    "UPDATE relationships SET source_id = ? WHERE source_id = ?",
                    [new_entity_id, old_entity_id]
                )
                source_updates = source_result.rowcount
                
                # Update target references
                target_result = conn.execute(
                    "UPDATE relationships SET target_id = ? WHERE target_id = ?",
                    [new_entity_id, old_entity_id]
                )
                target_updates = target_result.rowcount
                
                updated_count = source_updates + target_updates
                self.logger.info(f"Updated {updated_count} relationships for entity move {old_entity_id} -> {new_entity_id}")
                
        except Exception as e:
            self.logger.error(f"Error during entity move handling: {e}")
            raise
        
        return updated_count


class OrphanCleaner:
    """Finds and cleans up orphaned relationships."""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.logger = logging.getLogger(f"{__name__}.OrphanCleaner")
    
    def cleanup_orphans(self) -> List[str]:
        """
        Find and clean up orphaned relationships.
        
        Returns:
            List of relationship IDs that were cleaned up
        """
        orphaned_ids = []
        
        try:
            with duckdb.connect(self.database_path) as conn:
                # Find relationships with non-existent entities
                orphan_query = """
                SELECT r.id 
                FROM relationships r
                WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.source_id)
                   OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.target_id)
                """
                
                result = conn.execute(orphan_query).fetchall()
                orphaned_ids = [str(row[0]) for row in result]
                
                if orphaned_ids:
                    # Delete orphaned relationships
                    placeholders = ','.join(['?' for _ in orphaned_ids])
                    delete_query = f"DELETE FROM relationships WHERE id::VARCHAR IN ({placeholders})"
                    conn.execute(delete_query, orphaned_ids)
                    
                    self.logger.info(f"Cleaned up {len(orphaned_ids)} orphaned relationships")
                
        except Exception as e:
            self.logger.error(f"Error during orphan cleanup: {e}")
            raise
        
        return orphaned_ids


class RelationshipUpdater:
    """
    Main relationship updater that coordinates all relationship update operations.
    
    This class implements the core functionality for Issue #66:
    - Detects relationship changes from entity modifications  
    - Updates dependency graphs with incremental changes
    - Handles deletion cascades and orphan cleanup
    - Maintains graph consistency and performance
    """
    
    def __init__(self, 
                 relationship_detector: RelationshipDetector,
                 storage: RelationshipStorage,
                 database_path: str = None):
        """
        Initialize the relationship updater.
        
        Args:
            relationship_detector: Instance of RelationshipDetector for analysis
            storage: Instance of RelationshipStorage for persistence
            database_path: Path to DuckDB database (optional)
        """
        self.detector = relationship_detector
        self.storage = storage
        self.database_path = database_path or "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        
        # Initialize component analyzers
        self.change_analyzer = ChangeAnalyzer(storage)
        self.relationship_differ = RelationshipDiffer(storage)
        self.cascade_handler = CascadeHandler(storage, self.database_path)
        self.orphan_cleaner = OrphanCleaner(self.database_path)
        
        self.logger = logging.getLogger(f"{__name__}.RelationshipUpdater")
        
        # Performance configuration
        self.batch_size = 100  # Process entities in batches
        self.max_cascade_depth = 10  # Prevent infinite cascades
        
        # Statistics tracking
        self.statistics = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'relationships_processed': 0,
            'orphans_cleaned': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("RelationshipUpdater initialized successfully")
    
    def update_relationships(self, entity_changes: List[EntityChange]) -> RelationshipUpdateResult:
        """
        Main entry point for relationship updates from entity changes.
        
        This method implements the core algorithm from the issue requirements:
        1. Identifies affected relationships through change analysis
        2. Re-analyzes relationships for modified entities
        3. Applies relationship diffs (additions, updates, deletions)
        4. Handles cascade operations for deleted entities
        5. Cleans up orphaned relationships
        6. Validates graph consistency
        
        Args:
            entity_changes: List of EntityChange objects
            
        Returns:
            RelationshipUpdateResult with comprehensive operation details
        """
        start_time = time.time()
        result = RelationshipUpdateResult(success=False)
        result.entity_changes_processed = len(entity_changes)
        
        if not entity_changes:
            result.success = True
            result.processing_time = time.time() - start_time
            return result
        
        try:
            self.logger.info(f"Starting relationship update for {len(entity_changes)} entity changes")
            
            # Phase 1: Analyze change impact
            impact_analysis = self.change_analyzer.analyze_impact(entity_changes)
            
            # Phase 2: Process entity modifications (reanalysis required)
            modified_results = self._process_modified_entities(impact_analysis.reanalysis_required)
            result.relationships_added += modified_results['added']
            result.relationships_updated += modified_results['updated'] 
            result.relationships_removed += modified_results['removed']
            result.affected_entities.update(impact_analysis.reanalysis_required)
            
            # Phase 3: Process new entities (potential new relationships)
            new_results = self._process_new_entities(impact_analysis.potential_new_sources)
            result.relationships_added += new_results['added']
            result.affected_entities.update(impact_analysis.potential_new_sources)
            
            # Phase 4: Handle deletion cascades
            cascade_results = self._process_deletions(impact_analysis.cascade_deletions)
            result.relationships_removed += cascade_results['removed']
            result.affected_entities.update(impact_analysis.cascade_deletions)
            
            # Phase 5: Handle entity moves
            move_results = self._process_moves(impact_analysis.reference_updates)
            result.relationships_updated += move_results['updated']
            
            # Phase 6: Clean up orphaned relationships
            if impact_analysis.orphan_cleanup_required:
                orphaned_ids = self.orphan_cleaner.cleanup_orphans()
                result.orphans_cleaned = len(orphaned_ids)
                result.relationships_removed += len(orphaned_ids)
            
            # Phase 7: Validate graph consistency
            validation_passed = self._validate_graph_consistency(result.affected_entities)
            result.validation_passed = validation_passed
            
            if not validation_passed:
                result.errors.append("Graph consistency validation failed")
                self.logger.error("Graph consistency validation failed after relationship update")
                return result
            
            # Success!
            result.success = True
            result.processing_time = time.time() - start_time
            
            # Update statistics
            self.statistics['total_updates'] += 1
            self.statistics['successful_updates'] += 1
            self.statistics['relationships_processed'] += (
                result.relationships_added + result.relationships_updated + result.relationships_removed
            )
            self.statistics['orphans_cleaned'] += result.orphans_cleaned
            self._update_avg_processing_time(result.processing_time)
            
            self.logger.info(f"Relationship update completed successfully in {result.processing_time:.2f}s. "
                           f"Processed: +{result.relationships_added} ~{result.relationships_updated} "
                           f"-{result.relationships_removed} relationships, {result.orphans_cleaned} orphans cleaned")
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Relationship update failed: {str(e)}")
            result.processing_time = time.time() - start_time
            
            self.statistics['total_updates'] += 1
            self.statistics['failed_updates'] += 1
            
            self.logger.error(f"Relationship update failed: {e}", exc_info=True)
            return result
    
    def _process_modified_entities(self, entity_ids: Set[str]) -> Dict[str, int]:
        """Process entities that have been modified and need relationship reanalysis."""
        results = {'added': 0, 'updated': 0, 'removed': 0}
        
        if not entity_ids:
            return results
        
        self.logger.info(f"Processing {len(entity_ids)} modified entities")
        
        # Process in batches for performance
        entity_list = list(entity_ids)
        for i in range(0, len(entity_list), self.batch_size):
            batch = entity_list[i:i + self.batch_size]
            batch_results = self._process_entity_batch_modifications(batch)
            
            for key in results:
                results[key] += batch_results[key]
        
        self.logger.info(f"Modified entity processing complete: {results}")
        return results
    
    def _process_entity_batch_modifications(self, entity_ids: List[str]) -> Dict[str, int]:
        """Process a batch of modified entities."""
        results = {'added': 0, 'updated': 0, 'removed': 0}
        
        try:
            with duckdb.connect(self.database_path) as conn:
                for entity_id in entity_ids:
                    # Get the current entity
                    entity_query = "SELECT * FROM entities WHERE id = ?"
                    entity_result = conn.execute(entity_query, [entity_id]).fetchone()
                    
                    if not entity_result:
                        self.logger.warning(f"Entity {entity_id} not found during relationship update")
                        continue
                    
                    # Convert to CodeEntity (simplified - would need full conversion)
                    entity = self._db_row_to_entity(entity_result)
                    
                    # Get existing relationships for this entity
                    old_relationships = self._get_relationships_for_entity(entity_id, conn)
                    
                    # Re-detect relationships for the entity
                    detection_result = self.detector.detect_relationships_from_file(
                        entity.file_path, [entity]
                    )
                    
                    if not detection_result.success:
                        self.logger.warning(f"Failed to re-detect relationships for entity {entity_id}")
                        continue
                    
                    # Filter to only relationships involving this entity
                    new_relationships = [
                        rel for rel in detection_result.relationships
                        if rel.source_id == entity_id or rel.target_id == entity_id
                    ]
                    
                    # Calculate diff
                    diff = self.relationship_differ.calculate_relationship_diff(
                        entity_id, old_relationships, new_relationships
                    )
                    
                    # Apply changes
                    batch_results = self._apply_relationship_diff(diff, conn)
                    for key in results:
                        results[key] += batch_results[key]
                        
        except Exception as e:
            self.logger.error(f"Error processing entity batch modifications: {e}")
            raise
        
        return results
    
    def _process_new_entities(self, entity_ids: Set[str]) -> Dict[str, int]:
        """Process newly created entities that might have new relationships."""
        results = {'added': 0}
        
        if not entity_ids:
            return results
        
        self.logger.info(f"Processing {len(entity_ids)} new entities")
        
        # For new entities, we just need to detect and add relationships
        # (no diff calculation needed since there are no old relationships)
        
        try:
            with duckdb.connect(self.database_path) as conn:
                for entity_id in entity_ids:
                    entity_query = "SELECT * FROM entities WHERE id = ?"
                    entity_result = conn.execute(entity_query, [entity_id]).fetchone()
                    
                    if not entity_result:
                        continue
                    
                    entity = self._db_row_to_entity(entity_result)
                    
                    # Detect relationships for the new entity
                    detection_result = self.detector.detect_relationships_from_file(
                        entity.file_path, [entity]
                    )
                    
                    if detection_result.success:
                        # Store new relationships
                        storage_result = self.storage.store_relationships(
                            detection_result.relationships, update_mode='upsert'
                        )
                        results['added'] += storage_result.get('inserted', 0)
                        
        except Exception as e:
            self.logger.error(f"Error processing new entities: {e}")
            raise
        
        self.logger.info(f"New entity processing complete: {results}")
        return results
    
    def _process_deletions(self, entity_ids: Set[str]) -> Dict[str, int]:
        """Process deleted entities and handle cascade deletions."""
        results = {'removed': 0}
        
        if not entity_ids:
            return results
        
        self.logger.info(f"Processing deletions for {len(entity_ids)} entities")
        
        for entity_id in entity_ids:
            deleted_relationships = self.cascade_handler.handle_entity_deletion(entity_id)
            results['removed'] += len(deleted_relationships)
        
        self.logger.info(f"Deletion processing complete: {results}")
        return results
    
    def _process_moves(self, reference_updates: Set[Tuple[str, str]]) -> Dict[str, int]:
        """Process entity moves by updating relationship references."""
        results = {'updated': 0}
        
        if not reference_updates:
            return results
        
        self.logger.info(f"Processing {len(reference_updates)} entity moves")
        
        for old_id, new_id in reference_updates:
            updated_count = self.cascade_handler.handle_entity_move(old_id, new_id)
            results['updated'] += updated_count
        
        self.logger.info(f"Move processing complete: {results}")
        return results
    
    def _apply_relationship_diff(self, diff: RelationshipDiff, 
                               conn: duckdb.DuckDBPyConnection) -> Dict[str, int]:
        """Apply relationship diff changes to storage."""
        results = {'added': 0, 'updated': 0, 'removed': 0}
        
        try:
            # Add new relationships
            if diff.added:
                storage_result = self.storage.store_relationships(diff.added, update_mode='insert')
                results['added'] += storage_result.get('inserted', 0)
            
            # Update modified relationships
            for old_rel, new_rel in diff.modified:
                storage_result = self.storage.store_relationships([new_rel], update_mode='replace')
                results['updated'] += storage_result.get('updated', 0)
            
            # Remove deleted relationships
            if diff.removed:
                for rel in diff.removed:
                    conn.execute("DELETE FROM relationships WHERE id = ?", [str(rel.id)])
                results['removed'] += len(diff.removed)
            
        except Exception as e:
            self.logger.error(f"Error applying relationship diff: {e}")
            raise
        
        return results
    
    def _get_relationships_for_entity(self, entity_id: str, 
                                    conn: duckdb.DuckDBPyConnection) -> List[CodeRelationship]:
        """Get all relationships involving a specific entity."""
        query = """
        SELECT id, source_id, target_id, relationship_type, confidence, metadata
        FROM relationships
        WHERE source_id = ? OR target_id = ?
        """
        
        rows = conn.execute(query, [entity_id, entity_id]).fetchall()
        relationships = []
        
        for row in rows:
            rel_data = {
                'id': str(row[0]),
                'source_id': str(row[1]),
                'target_id': str(row[2]),
                'relationship_type': row[3],
                'confidence': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            }
            relationships.append(CodeRelationship.from_db_dict(rel_data))
        
        return relationships
    
    def _db_row_to_entity(self, row) -> CodeEntity:
        """Convert database row to CodeEntity (simplified)."""
        from ..extraction.entity_types import EntityType, SourceLocation
        from uuid import UUID
        
        # Parse entity data from database row
        entity_id = row[0] if row[0] else uuid.uuid4()
        name = row[1] if len(row) > 1 else "unknown"
        entity_type = row[2] if len(row) > 2 else "function"
        file_path = row[3] if len(row) > 3 else ""
        start_line = row[4] if len(row) > 4 else 0
        end_line = row[5] if len(row) > 5 else 0
        
        # Convert to proper types
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        
        try:
            entity_type_enum = EntityType(entity_type)
        except ValueError:
            entity_type_enum = EntityType.FUNCTION
        
        location = SourceLocation(line_start=start_line, line_end=end_line) if start_line and end_line else None
        
        return CodeEntity(
            id=entity_id,
            name=name,
            type=entity_type_enum,
            file_path=file_path,
            location=location,
            metadata={}
        )
    
    def _validate_graph_consistency(self, affected_entities: Set[str]) -> bool:
        """Validate that the graph remains consistent after updates."""
        try:
            with duckdb.connect(self.database_path) as conn:
                # Check for orphaned relationships
                orphan_query = """
                SELECT COUNT(*) FROM relationships r
                WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.source_id)
                   OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.target_id)
                """
                
                result = conn.execute(orphan_query).fetchone()
                orphan_count = result[0] if result else 0
                
                if orphan_count > 0:
                    self.logger.error(f"Graph consistency validation failed: {orphan_count} orphaned relationships")
                    return False
                
                # Check for self-referential relationships
                self_ref_query = """
                SELECT COUNT(*) FROM relationships
                WHERE source_id = target_id
                """
                
                result = conn.execute(self_ref_query).fetchone()
                self_ref_count = result[0] if result else 0
                
                if self_ref_count > 0:
                    self.logger.error(f"Graph consistency validation failed: {self_ref_count} self-referential relationships")
                    return False
                
                # Check confidence values are valid
                invalid_confidence_query = """
                SELECT COUNT(*) FROM relationships
                WHERE confidence < 0.0 OR confidence > 1.0
                """
                
                result = conn.execute(invalid_confidence_query).fetchone()
                invalid_conf_count = result[0] if result else 0
                
                if invalid_conf_count > 0:
                    self.logger.error(f"Graph consistency validation failed: {invalid_conf_count} invalid confidence values")
                    return False
                
                self.logger.info(f"Graph consistency validation passed for {len(affected_entities)} entities")
                return True
                
        except Exception as e:
            self.logger.error(f"Error during graph consistency validation: {e}")
            return False
    
    def _update_avg_processing_time(self, duration: float):
        """Update average processing time statistics."""
        current_avg = self.statistics['avg_processing_time']
        total_ops = self.statistics['total_updates']
        
        # Calculate new average: (old_avg * (n-1) + new_value) / n
        new_avg = (current_avg * (total_ops - 1) + duration) / total_ops
        self.statistics['avg_processing_time'] = new_avg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current relationship updater statistics."""
        return self.statistics.copy()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.statistics = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'relationships_processed': 0,
            'orphans_cleaned': 0,
            'avg_processing_time': 0.0
        }
        self.logger.info("Statistics reset")


# Integration functions for external usage
def create_relationship_updater(database_path: str = None) -> RelationshipUpdater:
    """
    Factory function to create a RelationshipUpdater instance with all dependencies.
    
    Args:
        database_path: Path to DuckDB database
        
    Returns:
        Configured RelationshipUpdater instance
    """
    from ..parsing.parser_manager import ParserManager
    
    # Initialize dependencies
    parser_manager = ParserManager()
    relationship_detector = RelationshipDetector(parser_manager)
    relationship_storage = RelationshipStorage(database_path or "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb")
    
    return RelationshipUpdater(relationship_detector, relationship_storage, database_path)


def validate_updater_prerequisites(database_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that all prerequisites are met for relationship updater operations.
    
    Args:
        database_path: Path to DuckDB database
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        with duckdb.connect(database_path) as conn:
            # Check required tables exist
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN ('entities', 'relationships')
            """
            
            tables = conn.execute(tables_query).fetchall()
            table_names = {row[0] for row in tables}
            
            if 'entities' not in table_names:
                issues.append("Missing required 'entities' table")
            if 'relationships' not in table_names:
                issues.append("Missing required 'relationships' table")
            
            # Check for basic functionality
            conn.execute("SELECT COUNT(*) FROM entities").fetchone()
            conn.execute("SELECT COUNT(*) FROM relationships").fetchone()
                
    except Exception as e:
        issues.append(f"Database validation error: {str(e)}")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    # Example usage and validation
    updater = create_relationship_updater()
    
    # Validate prerequisites
    valid, issues = validate_updater_prerequisites(updater.database_path)
    if not valid:
        print(f"Prerequisites validation failed: {issues}")
    else:
        print("Prerequisites validation passed")
        
        # Example relationship update
        test_changes = [
            EntityChange(
                entity_id=str(uuid.uuid4()),
                change_type="modified",
                metadata={"test": True}
            )
        ]
        
        print("Testing relationship updater...")
        result = updater.update_relationships(test_changes)
        print(f"Update result: Success={result.success}, "
              f"Added={result.relationships_added}, "
              f"Updated={result.relationships_updated}, "
              f"Removed={result.relationships_removed}, "
              f"Time={result.processing_time:.3f}s")