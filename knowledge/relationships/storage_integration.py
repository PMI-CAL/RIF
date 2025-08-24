"""
Storage integration for detected relationships using DuckDB.

This module handles storing, retrieving, and querying relationships in the DuckDB
knowledge graph database.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import UUID
from datetime import datetime

import duckdb

from .relationship_types import (
    CodeRelationship, RelationshipDetectionResult, RelationshipQuery, 
    RelationshipType, RelationshipPattern
)
from .relationship_detector import RelationshipDetector


class RelationshipStorage:
    """
    Handles storage and retrieval of code relationships in DuckDB.
    
    Provides:
    - Batch relationship insert operations
    - Incremental relationship updates
    - Relationship deduplication and conflict resolution
    - Complex relationship queries
    - Pattern analysis and statistics
    """
    
    def __init__(self, db_path: str = "knowledge/chromadb/entities.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection
        self._conn = None
        self._ensure_schema()
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _ensure_schema(self):
        """Ensure the relationships table exists with proper schema."""
        conn = self._get_connection()
        
        # Check if relationships table exists
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'relationships'"
        ).fetchone()
        
        if result[0] == 0:
            self.logger.info("Creating relationships table...")
            
            # Load and execute schema from schema file
            schema_path = Path(__file__).parent.parent / "schema" / "duckdb_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                conn.execute(schema_sql)
            else:
                # Fallback: create basic relationships table
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
        
        self.logger.info("Relationship storage initialized successfully")
    
    def store_relationships(self, relationships: List[CodeRelationship], 
                          update_mode: str = 'upsert') -> Dict[str, int]:
        """
        Store detected relationships in the database.
        
        Args:
            relationships: List of CodeRelationship objects to store
            update_mode: 'insert', 'upsert', or 'replace'
            
        Returns:
            Dictionary with counts of inserted, updated, and skipped relationships
        """
        if not relationships:
            return {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        conn = self._get_connection()
        results = {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        try:
            conn.begin()
            
            for relationship in relationships:
                if update_mode == 'upsert':
                    result = self._upsert_relationship(conn, relationship)
                elif update_mode == 'replace':
                    result = self._replace_relationship(conn, relationship)
                else:  # insert
                    result = self._insert_relationship(conn, relationship)
                
                results[result] += 1
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error storing relationships: {e}")
            raise
        
        self.logger.info(f"Stored relationships: {results}")
        return results
    
    def _upsert_relationship(self, conn: duckdb.DuckDBPyConnection, 
                           relationship: CodeRelationship) -> str:
        """Insert or update relationship based on source, target, and type."""
        
        # Check if relationship exists
        existing = conn.execute("""
            SELECT id, confidence FROM relationships 
            WHERE source_id = ? AND target_id = ? AND relationship_type = ?
        """, [
            str(relationship.source_id), 
            str(relationship.target_id), 
            relationship.relationship_type.value
        ]).fetchone()
        
        if existing:
            existing_id, existing_confidence = existing
            
            # Update if new confidence is higher or metadata has changed
            if relationship.confidence > existing_confidence:
                conn.execute("""
                    UPDATE relationships SET 
                        confidence = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [
                    relationship.confidence,
                    json.dumps(relationship.to_db_dict()['metadata']),
                    existing_id
                ])
                return 'updated'
            else:
                return 'skipped'
        else:
            # Insert new relationship
            db_dict = relationship.to_db_dict()
            conn.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                db_dict['id'],
                db_dict['source_id'],
                db_dict['target_id'],
                db_dict['relationship_type'],
                db_dict['confidence'],
                json.dumps(db_dict['metadata'])
            ])
            return 'inserted'
    
    def _replace_relationship(self, conn: duckdb.DuckDBPyConnection, 
                            relationship: CodeRelationship) -> str:
        """Replace relationship regardless of existing data."""
        
        # Try to update first
        result = conn.execute("""
            UPDATE relationships SET 
                confidence = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
            WHERE source_id = ? AND target_id = ? AND relationship_type = ?
        """, [
            relationship.confidence,
            json.dumps(relationship.to_db_dict()['metadata']),
            str(relationship.source_id),
            str(relationship.target_id),
            relationship.relationship_type.value
        ])
        
        if result.rowcount > 0:
            return 'updated'
        else:
            # Insert if update didn't affect any rows
            db_dict = relationship.to_db_dict()
            conn.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                db_dict['id'],
                db_dict['source_id'],
                db_dict['target_id'],
                db_dict['relationship_type'],
                db_dict['confidence'],
                json.dumps(db_dict['metadata'])
            ])
            return 'inserted'
    
    def _insert_relationship(self, conn: duckdb.DuckDBPyConnection, 
                           relationship: CodeRelationship) -> str:
        """Insert relationship (will fail if duplicate exists)."""
        try:
            db_dict = relationship.to_db_dict()
            conn.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                db_dict['id'],
                db_dict['source_id'],
                db_dict['target_id'],
                db_dict['relationship_type'],
                db_dict['confidence'],
                json.dumps(db_dict['metadata'])
            ])
            return 'inserted'
        except Exception:
            return 'skipped'  # Likely a duplicate
    
    def store_detection_result(self, result: RelationshipDetectionResult, 
                              update_mode: str = 'upsert') -> Dict[str, int]:
        """Store all relationships from a RelationshipDetectionResult."""
        if not result.success:
            self.logger.warning(f"Skipping storage for failed detection: {result.file_path}")
            return {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        return self.store_relationships(result.relationships, update_mode)
    
    def query_relationships(self, query: RelationshipQuery) -> List[CodeRelationship]:
        """Execute a relationship query and return matching relationships."""
        conn = self._get_connection()
        
        # Build SQL query from RelationshipQuery
        where_clause, params = query.to_sql_conditions()
        
        sql = f"""
            SELECT id, source_id, target_id, relationship_type, confidence, metadata, created_at
            FROM relationships 
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
        """
        
        rows = conn.execute(sql, params).fetchall()
        
        relationships = []
        for row in rows:
            relationship_data = {
                'id': row[0],
                'source_id': row[1],
                'target_id': row[2],
                'relationship_type': row[3],
                'confidence': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            }
            relationships.append(CodeRelationship.from_db_dict(relationship_data))
        
        return relationships
    
    def get_relationships_for_entity(self, entity_id: UUID, 
                                   direction: str = 'both') -> List[CodeRelationship]:
        """Get all relationships for a specific entity."""
        query = RelationshipQuery()
        
        if direction == 'outgoing':
            query.from_source(entity_id)
        elif direction == 'incoming':
            query.to_target(entity_id)
        else:  # both
            # This requires a more complex query
            conn = self._get_connection()
            
            sql = """
                SELECT id, source_id, target_id, relationship_type, confidence, metadata, created_at
                FROM relationships 
                WHERE source_id = ? OR target_id = ?
                ORDER BY confidence DESC, created_at DESC
            """
            
            rows = conn.execute(sql, [str(entity_id), str(entity_id)]).fetchall()
            
            relationships = []
            for row in rows:
                relationship_data = {
                    'id': row[0],
                    'source_id': row[1],
                    'target_id': row[2],
                    'relationship_type': row[3],
                    'confidence': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                }
                relationships.append(CodeRelationship.from_db_dict(relationship_data))
            
            return relationships
        
        return self.query_relationships(query)
    
    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored relationships."""
        conn = self._get_connection()
        
        stats = {}
        
        # Total counts
        result = conn.execute("""
            SELECT 
                COUNT(*) as total_relationships,
                COUNT(DISTINCT source_id) as unique_sources,
                COUNT(DISTINCT target_id) as unique_targets,
                AVG(confidence) as avg_confidence
            FROM relationships
        """).fetchone()
        
        stats['total_relationships'] = result[0]
        stats['unique_sources'] = result[1]
        stats['unique_targets'] = result[2]
        stats['avg_confidence'] = float(result[3]) if result[3] else 0.0
        
        # Counts by type
        type_counts = conn.execute("""
            SELECT relationship_type, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM relationships
            GROUP BY relationship_type
            ORDER BY count DESC
        """).fetchall()
        
        stats['by_type'] = {
            row[0]: {'count': row[1], 'avg_confidence': float(row[2])}
            for row in type_counts
        }
        
        # Confidence distribution
        confidence_dist = conn.execute("""
            SELECT 
                COUNT(CASE WHEN confidence >= 0.9 THEN 1 END) as very_high,
                COUNT(CASE WHEN confidence >= 0.7 AND confidence < 0.9 THEN 1 END) as high,
                COUNT(CASE WHEN confidence >= 0.5 AND confidence < 0.7 THEN 1 END) as medium,
                COUNT(CASE WHEN confidence < 0.5 THEN 1 END) as low
            FROM relationships
        """).fetchone()
        
        stats['confidence_distribution'] = {
            'very_high': confidence_dist[0],
            'high': confidence_dist[1],
            'medium': confidence_dist[2],
            'low': confidence_dist[3]
        }
        
        # Cross-file relationships
        cross_file = conn.execute("""
            SELECT COUNT(*) FROM relationships r
            JOIN entities e1 ON r.source_id = e1.id
            JOIN entities e2 ON r.target_id = e2.id
            WHERE e1.file_path != e2.file_path
        """).fetchone()
        
        stats['cross_file_relationships'] = cross_file[0] if cross_file else 0
        
        return stats
    
    def find_relationship_patterns(self, min_frequency: int = 3) -> List[RelationshipPattern]:
        """Find common patterns in relationships."""
        conn = self._get_connection()
        
        patterns = []
        
        # Pattern 1: Common import relationships
        import_patterns = conn.execute("""
            SELECT 
                r.metadata->>'module_path' as module_path,
                COUNT(*) as frequency,
                AVG(r.confidence) as avg_confidence
            FROM relationships r
            WHERE r.relationship_type = 'imports' 
              AND r.metadata->>'module_path' IS NOT NULL
            GROUP BY r.metadata->>'module_path'
            HAVING COUNT(*) >= ?
            ORDER BY frequency DESC
        """, [min_frequency]).fetchall()
        
        for row in import_patterns:
            pattern = RelationshipPattern(
                pattern_type='common_import',
                frequency=row[1],
                confidence=float(row[2]),
                description=f"Module '{row[0]}' is commonly imported"
            )
            patterns.append(pattern)
        
        # Pattern 2: Function call hotspots
        call_patterns = conn.execute("""
            SELECT 
                e.name as function_name,
                COUNT(*) as frequency,
                AVG(r.confidence) as avg_confidence
            FROM relationships r
            JOIN entities e ON r.target_id = e.id
            WHERE r.relationship_type = 'calls' AND e.type = 'function'
            GROUP BY e.name
            HAVING COUNT(*) >= ?
            ORDER BY frequency DESC
        """, [min_frequency]).fetchall()
        
        for row in call_patterns:
            pattern = RelationshipPattern(
                pattern_type='function_hotspot',
                frequency=row[1],
                confidence=float(row[2]),
                description=f"Function '{row[0]}' is frequently called"
            )
            patterns.append(pattern)
        
        # Pattern 3: Inheritance hierarchies
        inheritance_patterns = conn.execute("""
            SELECT 
                e.name as base_class,
                COUNT(*) as frequency,
                AVG(r.confidence) as avg_confidence
            FROM relationships r
            JOIN entities e ON r.target_id = e.id
            WHERE r.relationship_type IN ('extends', 'implements') AND e.type = 'class'
            GROUP BY e.name
            HAVING COUNT(*) >= ?
            ORDER BY frequency DESC
        """, [min_frequency]).fetchall()
        
        for row in inheritance_patterns:
            pattern = RelationshipPattern(
                pattern_type='inheritance_base',
                frequency=row[1],
                confidence=float(row[2]),
                description=f"Class '{row[0]}' is commonly extended/implemented"
            )
            patterns.append(pattern)
        
        return patterns
    
    def cleanup_stale_relationships(self, entity_ids_to_keep: Optional[List[UUID]] = None) -> int:
        """Remove relationships for entities that no longer exist."""
        conn = self._get_connection()
        
        if entity_ids_to_keep:
            # Remove relationships where source or target is not in the keep list
            entity_ids_str = [str(eid) for eid in entity_ids_to_keep]
            placeholders = ','.join(['?' for _ in entity_ids_str])
            
            result = conn.execute(f"""
                DELETE FROM relationships 
                WHERE source_id NOT IN ({placeholders}) 
                   OR target_id NOT IN ({placeholders})
            """, entity_ids_str + entity_ids_str)
        else:
            # Remove relationships where entities don't exist in entities table
            result = conn.execute("""
                DELETE FROM relationships 
                WHERE source_id NOT IN (SELECT id FROM entities)
                   OR target_id NOT IN (SELECT id FROM entities)
            """)
        
        deleted_count = result.rowcount
        self.logger.info(f"Cleaned up {deleted_count} stale relationships")
        
        return deleted_count
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class RelationshipAnalysisPipeline:
    """
    Complete pipeline for detecting and storing relationships.
    
    Combines RelationshipDetector and RelationshipStorage for end-to-end processing.
    """
    
    def __init__(self, parser_manager, db_path: str = "knowledge/chromadb/entities.duckdb"):
        self.detector = RelationshipDetector(parser_manager)
        self.storage = RelationshipStorage(db_path)
        self.logger = logging.getLogger(__name__)
    
    def process_file(self, file_path: str, entities: List[Any], 
                    update_mode: str = 'upsert') -> Dict[str, Any]:
        """Process a single file: detect relationships and store them."""
        
        # Detect relationships
        detection_result = self.detector.detect_relationships_from_file(file_path, entities)
        
        # Store relationships
        storage_result = self.storage.store_detection_result(detection_result, update_mode)
        
        return {
            'file_path': file_path,
            'detection_success': detection_result.success,
            'detection_time': detection_result.detection_time,
            'relationships_found': len(detection_result.relationships),
            'relationships_stored': storage_result['inserted'] + storage_result['updated'],
            'storage_result': storage_result,
            'error_message': detection_result.error_message
        }
    
    def process_directory(self, directory_path: str, 
                         extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         exclude_patterns: Optional[List[str]] = None,
                         update_mode: str = 'upsert') -> Dict[str, Any]:
        """Process all files in a directory."""
        
        start_time = time.time()
        
        # Detect relationships from all files
        detection_results = self.detector.detect_relationships_from_directory(
            directory_path, extensions, recursive, exclude_patterns
        )
        
        # Store all results
        total_storage_stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        for result in detection_results:
            storage_stats = self.storage.store_detection_result(result, update_mode)
            for key in total_storage_stats:
                total_storage_stats[key] += storage_stats[key]
        
        # Calculate summary statistics
        successful_detections = [r for r in detection_results if r.success]
        failed_detections = [r for r in detection_results if not r.success]
        
        total_relationships = sum(len(r.relationships) for r in successful_detections)
        total_time = time.time() - start_time
        
        summary = {
            'directory': directory_path,
            'total_files': len(detection_results),
            'successful_detections': len(successful_detections),
            'failed_detections': len(failed_detections),
            'total_relationships_detected': total_relationships,
            'total_relationships_stored': total_storage_stats['inserted'] + total_storage_stats['updated'],
            'storage_stats': total_storage_stats,
            'processing_time': total_time,
            'avg_time_per_file': total_time / len(detection_results) if detection_results else 0
        }
        
        self.logger.info(f"Directory processing complete: {summary}")
        
        return summary
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the entire pipeline."""
        
        detection_metrics = self.detector.get_analysis_metrics()
        storage_stats = self.storage.get_relationship_statistics()
        
        return {
            'detection': detection_metrics,
            'storage': storage_stats,
            'combined': {
                'total_files_processed': detection_metrics['files_processed'],
                'total_relationships_in_db': storage_stats['total_relationships'],
                'detection_success_rate': (
                    detection_metrics['files_processed'] / max(detection_metrics['files_processed'], 1)
                )
            }
        }
    
    def close(self):
        """Clean shutdown of pipeline."""
        self.storage.close()