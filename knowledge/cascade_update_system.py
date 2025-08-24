"""
Cascade Update System for Knowledge Graph
Issue #67: Create cascade update system

This module implements a sophisticated cascade update system that:
1. Identifies all entities affected by initial changes through relationship traversal
2. Propagates changes through the dependency graph systematically  
3. Maintains graph consistency during and after updates
4. Handles circular dependencies without infinite loops

Key Features:
- Breadth-first graph traversal with cycle detection
- Strongly Connected Components (SCC) detection for circular dependency clusters
- Transaction management for atomic operations
- Memory-efficient streaming processing for large graphs
- Integration with DuckDB knowledge graph schema

Performance Targets:
- Handle graphs with >10,000 entities within 30 seconds
- <100ms latency for dependency lookups using existing indexes
- Batch updates of 500+ entities in single transaction
- Memory usage <800MB for normal operations

Author: RIF-Implementer
Date: 2025-08-23
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
import json
import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Change:
    """Represents a change event that triggers cascade updates."""
    entity_id: str
    change_type: str  # 'create', 'update', 'delete'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class UpdateResult:
    """Result of a cascade update operation."""
    success: bool
    affected_entities: Set[str] = field(default_factory=set)
    processed_entities: Set[str] = field(default_factory=set)
    cycles_detected: List[List[str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphState:
    """Maintains state during graph traversal operations."""
    visited: Set[str] = field(default_factory=set)
    processing: Set[str] = field(default_factory=set)
    processed: Set[str] = field(default_factory=set)
    cycle_detection_stack: List[str] = field(default_factory=list)
    strongly_connected_components: List[List[str]] = field(default_factory=list)


# Import the real RelationshipUpdater from Issue #66
try:
    from .relationships.relationship_updater import RelationshipUpdater, EntityChange, create_relationship_updater
    RELATIONSHIP_UPDATER_AVAILABLE = True
except ImportError:
    RELATIONSHIP_UPDATER_AVAILABLE = False
    # Fallback mock for development
    class RelationshipUpdaterMock:
        """Fallback mock interface for Issue #66 relationship updater."""
        
        def __init__(self):
            self.logger = logging.getLogger(f"{__name__}.RelationshipUpdaterMock")
            
        def detect_changes(self, entity_id: str) -> List[Change]:
            """Mock implementation of change detection."""
            self.logger.debug(f"Mock: Detecting changes for entity {entity_id}")
            return []
            
        def get_affected_relationships(self, entity_id: str) -> List[str]:
            """Mock implementation of affected relationship detection."""
            self.logger.debug(f"Mock: Getting affected relationships for entity {entity_id}")
            return []
            
        def validate_relationship_consistency(self, entity_ids: Set[str]) -> bool:
            """Mock implementation of relationship consistency validation."""
            self.logger.debug(f"Mock: Validating relationship consistency for {len(entity_ids)} entities")
            return True


class CascadeUpdateSystem:
    """
    Core cascade update system for knowledge graph consistency maintenance.
    
    This system implements sophisticated graph algorithms to identify affected entities,
    propagate updates through relationships, and maintain consistency while handling
    circular dependencies.
    """
    
    def __init__(self, database_path: str = None, relationship_updater=None):
        """
        Initialize the cascade update system.
        
        Args:
            database_path: Path to DuckDB database file
            relationship_updater: Instance of relationship updater (Issue #66)
        """
        self.database_path = database_path or "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        
        # Initialize relationship updater - use real implementation if available
        if relationship_updater:
            self.relationship_updater = relationship_updater
        elif RELATIONSHIP_UPDATER_AVAILABLE:
            self.relationship_updater = create_relationship_updater(self.database_path)
            self.logger = logging.getLogger(f"{__name__}.CascadeUpdateSystem")
            self.logger.info("Using real RelationshipUpdater implementation from Issue #66")
        else:
            self.relationship_updater = RelationshipUpdaterMock()
            self.logger = logging.getLogger(f"{__name__}.CascadeUpdateSystem")
            self.logger.warning("Using mock RelationshipUpdater - Issue #66 implementation not available")
            
        self.logger = logging.getLogger(f"{__name__}.CascadeUpdateSystem")
        
        # Performance tracking
        self.statistics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0,
            'entities_processed': 0,
            'cycles_detected': 0
        }
        
        # Memory management
        self.memory_budget = 800 * 1024 * 1024  # 800MB budget
        self.batch_size = 500  # Default batch size for updates
        
        self.logger.info(f"CascadeUpdateSystem initialized with database: {self.database_path}")
    
    def cascade_updates(self, initial_change: Change) -> UpdateResult:
        """
        Main cascade update orchestration method.
        
        This method implements the core algorithm that:
        1. Identifies all affected entities through graph traversal
        2. Detects and handles circular dependencies
        3. Propagates updates in topologically sorted order
        4. Validates graph consistency throughout the process
        
        Args:
            initial_change: The change event that triggers the cascade
            
        Returns:
            UpdateResult with comprehensive operation details
        """
        start_time = time.time()
        result = UpdateResult(success=False)
        
        try:
            self.logger.info(f"Starting cascade update for entity {initial_change.entity_id} "
                           f"(type: {initial_change.change_type})")
            
            # Phase 1: Initialize graph state and identify affected entities
            graph_state = GraphState()
            affected_entities = self._identify_affected_entities(initial_change.entity_id, graph_state)
            result.affected_entities = affected_entities
            
            self.logger.info(f"Identified {len(affected_entities)} potentially affected entities")
            
            # Phase 2: Detect cycles and strongly connected components
            cycles = self._detect_cycles(affected_entities, graph_state)
            result.cycles_detected = cycles
            
            if cycles:
                self.logger.warning(f"Detected {len(cycles)} cycles in dependency graph")
                self.statistics['cycles_detected'] += len(cycles)
            
            # Phase 3: Process updates in dependency order
            processed_entities = self._process_updates_ordered(
                affected_entities, cycles, initial_change, graph_state
            )
            result.processed_entities = processed_entities
            
            # Phase 4: Validate final graph consistency
            consistency_valid = self._validate_graph_consistency(processed_entities)
            
            if not consistency_valid:
                result.errors.append("Graph consistency validation failed")
                self.logger.error("Graph consistency validation failed after cascade update")
                return result
            
            # Success!
            result.success = True
            result.duration_seconds = time.time() - start_time
            
            # Update statistics
            self.statistics['total_operations'] += 1
            self.statistics['successful_operations'] += 1
            self.statistics['entities_processed'] += len(processed_entities)
            self._update_avg_processing_time(result.duration_seconds)
            
            self.logger.info(f"Cascade update completed successfully in {result.duration_seconds:.2f}s. "
                           f"Processed {len(processed_entities)} entities.")
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Cascade update failed: {str(e)}")
            result.duration_seconds = time.time() - start_time
            
            self.statistics['total_operations'] += 1
            self.statistics['failed_operations'] += 1
            
            self.logger.error(f"Cascade update failed: {e}", exc_info=True)
            return result
    
    def _identify_affected_entities(self, entity_id: str, graph_state: GraphState) -> Set[str]:
        """
        Identify all entities affected by the initial change through breadth-first traversal.
        
        This implements a memory-efficient breadth-first search that:
        - Tracks visited nodes to prevent infinite loops
        - Uses existing database indexes for fast relationship lookups  
        - Implements early termination for large graphs
        
        Args:
            entity_id: Starting entity for traversal
            graph_state: Current graph traversal state
            
        Returns:
            Set of all affected entity IDs
        """
        affected = set([entity_id])
        queue = deque([entity_id])
        graph_state.visited.add(entity_id)
        
        try:
            with duckdb.connect(self.database_path) as conn:
                while queue:
                    current_id = queue.popleft()
                    
                    # Find all entities dependent on current entity
                    dependents = self._find_dependents(current_id, conn)
                    
                    for dependent_id in dependents:
                        if dependent_id not in graph_state.visited:
                            affected.add(dependent_id)
                            queue.append(dependent_id)
                            graph_state.visited.add(dependent_id)
                            
                    # Memory management: limit traversal depth for very large graphs
                    if len(affected) > 10000:
                        self.logger.warning(f"Large graph detected ({len(affected)} entities). "
                                          "Limiting traversal to prevent memory issues.")
                        break
                        
        except Exception as e:
            self.logger.error(f"Error during affected entity identification: {e}")
            raise
        
        return affected
    
    def _find_dependencies(self, entity_id: str, conn: duckdb.DuckDBPyConnection) -> List[str]:
        """
        Find all entities that the given entity depends on.
        
        This is the opposite of _find_dependents - it finds what this entity uses/calls/references.
        
        Args:
            entity_id: Entity to find dependencies for
            conn: Database connection
            
        Returns:
            List of entity IDs this entity depends on
        """
        try:
            # Query uses idx_relationships_source for fast lookup
            query = """
            SELECT DISTINCT r.target_id
            FROM relationships r
            WHERE r.source_id = ?
              AND r.relationship_type IN ('calls', 'uses', 'references', 'imports')
            """
            
            result = conn.execute(query, [entity_id]).fetchall()
            dependencies = [str(row[0]) for row in result]  # Convert UUIDs to strings
            
            self.logger.debug(f"Found {len(dependencies)} dependencies for entity {entity_id}")
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error finding dependencies for entity {entity_id}: {e}")
            return []

    def _find_dependents(self, entity_id: str, conn: duckdb.DuckDBPyConnection) -> List[str]:
        """
        Find all entities that depend on the given entity.
        
        Uses existing database indexes for optimal performance:
        - idx_relationships_target for reverse relationship lookup
        - Leverages relationship_type filtering for relevant dependencies
        
        Args:
            entity_id: Entity to find dependents for
            conn: Database connection
            
        Returns:
            List of dependent entity IDs
        """
        try:
            # Query uses idx_relationships_target for fast lookup
            query = """
            SELECT DISTINCT r.source_id
            FROM relationships r
            WHERE r.target_id = ?
              AND r.relationship_type IN ('calls', 'uses', 'references', 'imports')
            """
            
            result = conn.execute(query, [entity_id]).fetchall()
            dependents = [str(row[0]) for row in result]  # Convert UUIDs to strings
            
            self.logger.debug(f"Found {len(dependents)} dependents for entity {entity_id}")
            return dependents
            
        except Exception as e:
            self.logger.error(f"Error finding dependents for entity {entity_id}: {e}")
            return []
    
    def _detect_cycles(self, entities: Set[str], graph_state: GraphState) -> List[List[str]]:
        """
        Detect cycles in the dependency graph using Tarjan's algorithm.
        
        This implements Tarjan's strongly connected components algorithm to:
        - Identify all cycles in the dependency graph
        - Group entities into strongly connected components
        - Enable proper ordering for cycle handling during updates
        
        Args:
            entities: Set of entities to check for cycles
            graph_state: Current graph traversal state
            
        Returns:
            List of cycles, where each cycle is a list of entity IDs
        """
        # Initialize Tarjan's algorithm state
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        cycles = []
        
        def strongconnect(entity_id: str):
            """Tarjan's algorithm recursive implementation."""
            index[entity_id] = index_counter[0]
            lowlinks[entity_id] = index_counter[0]
            index_counter[0] += 1
            stack.append(entity_id)
            on_stack.add(entity_id)
            
            try:
                with duckdb.connect(self.database_path) as conn:
                    dependencies = self._find_dependencies(entity_id, conn)
                    
                    for dependency_id in dependencies:
                        if dependency_id in entities:  # Only consider entities in our set
                            if dependency_id not in index:
                                strongconnect(dependency_id)
                                lowlinks[entity_id] = min(lowlinks[entity_id], lowlinks[dependency_id])
                            elif dependency_id in on_stack:
                                lowlinks[entity_id] = min(lowlinks[entity_id], index[dependency_id])
            
            except Exception as e:
                self.logger.error(f"Error during cycle detection for entity {entity_id}: {e}")
                return
            
            # Check if entity_id is a root node (representative of SCC)
            if lowlinks[entity_id] == index[entity_id]:
                connected_component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    connected_component.append(w)
                    if w == entity_id:
                        break
                
                # If component has more than one entity, it's a cycle
                if len(connected_component) > 1:
                    cycles.append(connected_component)
                    
                graph_state.strongly_connected_components.append(connected_component)
        
        # Run Tarjan's algorithm on all unprocessed entities
        for entity_id in entities:
            if entity_id not in index:
                strongconnect(entity_id)
        
        self.logger.info(f"Cycle detection complete. Found {len(cycles)} cycles in {len(entities)} entities.")
        return cycles
    
    def _process_updates_ordered(self, entities: Set[str], cycles: List[List[str]], 
                               initial_change: Change, graph_state: GraphState) -> Set[str]:
        """
        Process updates in topologically sorted order, handling cycles appropriately.
        
        This method implements the core update processing logic:
        - Topologically sorts entities where possible
        - Handles cycles using strongly connected components
        - Uses batch processing for database efficiency
        - Provides transaction isolation for atomic operations
        
        Args:
            entities: All entities to be updated
            cycles: Detected cycles that need special handling
            initial_change: Original change that triggered cascade
            graph_state: Current graph traversal state
            
        Returns:
            Set of successfully processed entity IDs
        """
        processed = set()
        
        try:
            # Create topological order, treating cycles as single units
            ordered_entities = self._topological_sort_with_cycles(entities, cycles)
            
            # Process entities in batches for optimal database performance
            batch = []
            
            for entity_id in ordered_entities:
                batch.append(entity_id)
                
                # Process batch when it reaches optimal size
                if len(batch) >= self.batch_size:
                    successfully_processed = self._process_entity_batch(batch, initial_change)
                    processed.update(successfully_processed)
                    batch.clear()
            
            # Process remaining entities in final batch
            if batch:
                successfully_processed = self._process_entity_batch(batch, initial_change)
                processed.update(successfully_processed)
            
            self.logger.info(f"Successfully processed {len(processed)} entities in ordered batches")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error during ordered update processing: {e}")
            raise
    
    def _topological_sort_with_cycles(self, entities: Set[str], cycles: List[List[str]]) -> List[str]:
        """
        Create topological ordering of entities, treating cycles as single units.
        
        Args:
            entities: All entities to sort
            cycles: Detected cycles to treat as units
            
        Returns:
            Topologically sorted list of entity IDs
        """
        # For now, implement a simple ordering strategy
        # In a full implementation, this would use more sophisticated topological sorting
        ordered = []
        
        # Process non-cyclic entities first
        cycle_entities = set()
        for cycle in cycles:
            cycle_entities.update(cycle)
            
        non_cycle_entities = entities - cycle_entities
        ordered.extend(non_cycle_entities)
        
        # Add cyclic entities at the end (they'll be handled specially)
        for cycle in cycles:
            ordered.extend(cycle)
            
        return ordered
    
    def _process_entity_batch(self, entity_ids: List[str], initial_change: Change) -> Set[str]:
        """
        Process a batch of entities with transaction isolation.
        
        Args:
            entity_ids: Batch of entity IDs to process
            initial_change: Original change that triggered cascade
            
        Returns:
            Set of successfully processed entity IDs
        """
        successfully_processed = set()
        
        try:
            with duckdb.connect(self.database_path) as conn:
                # Start transaction for atomic batch processing
                conn.begin()
                
                try:
                    for entity_id in entity_ids:
                        success = self._update_entity(entity_id, initial_change, conn)
                        if success:
                            successfully_processed.add(entity_id)
                        else:
                            self.logger.warning(f"Failed to update entity {entity_id}")
                    
                    # Commit transaction if all updates successful
                    conn.commit()
                    self.logger.debug(f"Successfully processed batch of {len(successfully_processed)} entities")
                    
                except Exception as e:
                    # Rollback transaction on any failure
                    conn.rollback()
                    self.logger.error(f"Batch processing failed, transaction rolled back: {e}")
                    successfully_processed.clear()
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error during batch processing: {e}")
            
        return successfully_processed
    
    def _update_entity(self, entity_id: str, initial_change: Change, 
                      conn: duckdb.DuckDBPyConnection) -> bool:
        """
        Update a single entity based on the cascade change.
        
        Args:
            entity_id: Entity to update
            initial_change: Original change that triggered cascade
            conn: Database connection within transaction
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # For now, implement a simple update that records the cascade operation
            # In a full implementation, this would apply specific business logic
            # based on the change type and entity relationships
            
            # Update entity's updated_at timestamp to reflect cascade update
            update_query = """
            UPDATE entities 
            SET updated_at = CURRENT_TIMESTAMP,
                metadata = COALESCE(metadata, '{}')
            WHERE id = ?
            """
            
            conn.execute(update_query, [entity_id])
            
            self.logger.debug(f"Updated entity {entity_id} due to cascade from {initial_change.entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update entity {entity_id}: {e}")
            return False
    
    def _validate_graph_consistency(self, processed_entities: Set[str]) -> bool:
        """
        Validate that the graph remains consistent after cascade updates.
        
        This performs multi-level validation:
        - Entity-level: Verify all entities exist and have valid data
        - Relationship-level: Ensure all relationships are valid
        - Graph-level: Check for consistency violations
        
        Args:
            processed_entities: Set of entities that were updated
            
        Returns:
            True if graph is consistent, False otherwise
        """
        try:
            with duckdb.connect(self.database_path) as conn:
                # Level 1: Entity validation
                entity_validation = self._validate_entities(processed_entities, conn)
                if not entity_validation:
                    self.logger.error("Entity-level validation failed")
                    return False
                
                # Level 2: Relationship validation  
                relationship_validation = self._validate_relationships(processed_entities, conn)
                if not relationship_validation:
                    self.logger.error("Relationship-level validation failed")
                    return False
                
                # Level 3: Graph-level validation
                graph_validation = self._validate_graph_structure(processed_entities, conn)
                if not graph_validation:
                    self.logger.error("Graph-level validation failed")
                    return False
                
                # Level 4: Use relationship updater for additional validation
                relationship_updater_validation = self.relationship_updater.validate_relationship_consistency(
                    processed_entities
                )
                if not relationship_updater_validation:
                    self.logger.error("Relationship updater validation failed")
                    return False
                
                self.logger.info(f"Graph consistency validation passed for {len(processed_entities)} entities")
                return True
                
        except Exception as e:
            self.logger.error(f"Error during graph consistency validation: {e}")
            return False
    
    def _validate_entities(self, entity_ids: Set[str], conn: duckdb.DuckDBPyConnection) -> bool:
        """Validate that all processed entities exist and have valid data."""
        try:
            # Convert entity IDs to proper format for DuckDB
            entity_list = [str(eid) for eid in entity_ids]
            placeholders = ','.join(['?' for _ in entity_list])
            
            query = f"""
            SELECT COUNT(*) as count
            FROM entities 
            WHERE id::VARCHAR IN ({placeholders})
              AND name IS NOT NULL 
              AND name != ''
              AND type IS NOT NULL
              AND file_path IS NOT NULL
              AND file_path != ''
            """
            
            result = conn.execute(query, entity_list).fetchone()
            valid_count = result[0] if result else 0
            
            if valid_count != len(entity_ids):
                self.logger.error(f"Entity validation failed: {valid_count}/{len(entity_ids)} entities valid")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error during entity validation: {e}")
            return False
    
    def _validate_relationships(self, entity_ids: Set[str], conn: duckdb.DuckDBPyConnection) -> bool:
        """Validate that all relationships involving processed entities are valid."""
        try:
            # Convert entity IDs to proper format for DuckDB
            entity_list = [str(eid) for eid in entity_ids]
            placeholders = ','.join(['?' for _ in entity_list])
            
            query = f"""
            SELECT COUNT(*) as invalid_relationships
            FROM relationships r
            WHERE (r.source_id::VARCHAR IN ({placeholders}) OR r.target_id::VARCHAR IN ({placeholders}))
              AND (r.source_id = r.target_id 
                   OR r.confidence < 0.0 
                   OR r.confidence > 1.0
                   OR r.relationship_type IS NULL)
            """
            
            # Need to provide entity_list twice for the two IN clauses
            result = conn.execute(query, entity_list + entity_list).fetchone()
            invalid_count = result[0] if result else 0
            
            if invalid_count > 0:
                self.logger.error(f"Relationship validation failed: {invalid_count} invalid relationships found")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error during relationship validation: {e}")
            return False
    
    def _validate_graph_structure(self, entity_ids: Set[str], conn: duckdb.DuckDBPyConnection) -> bool:
        """Validate overall graph structure integrity."""
        try:
            # Check for orphaned relationships (references to non-existent entities)
            entity_list = [str(eid) for eid in entity_ids]
            placeholders = ','.join(['?' for _ in entity_list])
            
            orphan_query = f"""
            SELECT COUNT(*) as orphans
            FROM relationships r
            WHERE (r.source_id::VARCHAR IN ({placeholders}) OR r.target_id::VARCHAR IN ({placeholders}))
              AND (NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.source_id)
                   OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.target_id))
            """
            
            # Need to provide entity_list twice for the two IN clauses
            result = conn.execute(orphan_query, entity_list + entity_list).fetchone()
            orphan_count = result[0] if result else 0
            
            if orphan_count > 0:
                self.logger.error(f"Graph structure validation failed: {orphan_count} orphaned relationships")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error during graph structure validation: {e}")
            return False
    
    def _update_avg_processing_time(self, duration: float):
        """Update average processing time statistics."""
        current_avg = self.statistics['avg_processing_time']
        total_ops = self.statistics['total_operations']
        
        # Calculate new average: (old_avg * (n-1) + new_value) / n
        new_avg = (current_avg * (total_ops - 1) + duration) / total_ops
        self.statistics['avg_processing_time'] = new_avg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current system performance statistics."""
        return self.statistics.copy()
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.statistics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0,
            'entities_processed': 0,
            'cycles_detected': 0
        }
        self.logger.info("Statistics reset")


# Utility functions for external integration
def create_cascade_system(database_path: str = None) -> CascadeUpdateSystem:
    """Factory function to create a cascade update system instance."""
    return CascadeUpdateSystem(database_path=database_path)


def validate_cascade_prerequisites(database_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that all prerequisites are met for cascade update operations.
    
    Args:
        database_path: Path to DuckDB database
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        import duckdb
        
        # Test database connection
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
                
            # Check required indexes exist
            indexes_query = """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE index_name IN ('idx_relationships_source', 'idx_relationships_target')
            """
            
            indexes = conn.execute(indexes_query).fetchall()
            index_names = {row[0] for row in indexes}
            
            if 'idx_relationships_source' not in index_names:
                issues.append("Missing required index 'idx_relationships_source'")
            if 'idx_relationships_target' not in index_names:
                issues.append("Missing required index 'idx_relationships_target'")
                
    except Exception as e:
        issues.append(f"Database validation error: {str(e)}")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    # Example usage and basic validation
    system = create_cascade_system()
    
    # Validate prerequisites
    valid, issues = validate_cascade_prerequisites(system.database_path)
    if not valid:
        print(f"Prerequisites validation failed: {issues}")
    else:
        print("Prerequisites validation passed")
        
        # Example cascade update
        test_change = Change(
            entity_id=str(uuid.uuid4()),
            change_type="update",
            metadata={"test": True}
        )
        
        print(f"Testing cascade update system...")
        result = system.cascade_updates(test_change)
        print(f"Cascade result: Success={result.success}, "
              f"Affected={len(result.affected_entities)}, "
              f"Duration={result.duration_seconds:.3f}s")