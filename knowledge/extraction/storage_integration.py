"""
Storage integration for extracted entities using DuckDB.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import UUID
from datetime import datetime

import duckdb

from .entity_types import CodeEntity, ExtractionResult
from .entity_extractor import EntityExtractor


class EntityStorage:
    """
    Handles storage and retrieval of extracted entities in DuckDB.
    
    Provides:
    - Batch insert operations for performance
    - Incremental updates using AST hashes
    - Entity deduplication and conflict resolution
    - Query interface for retrieved entities
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
        """Ensure the entities table exists with proper schema."""
        conn = self._get_connection()
        
        # Check if entities table exists
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'entities'"
        ).fetchone()
        
        if result[0] == 0:
            self.logger.info("Creating entities table...")
            
            # Load and execute schema from schema file
            schema_path = Path(__file__).parent.parent / "schema" / "duckdb_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                conn.execute(schema_sql)
            else:
                # Fallback: create basic entities table
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
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        
        self.logger.info("Entity storage initialized successfully")
    
    def store_entities(self, entities: List[CodeEntity], 
                      update_mode: str = 'upsert') -> Dict[str, int]:
        """
        Store extracted entities in the database.
        
        Args:
            entities: List of CodeEntity objects to store
            update_mode: 'insert', 'upsert', or 'replace'
            
        Returns:
            Dictionary with counts of inserted, updated, and skipped entities
        """
        if not entities:
            return {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        conn = self._get_connection()
        
        results = {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        try:
            conn.begin()
            
            for entity in entities:
                if update_mode == 'upsert':
                    result = self._upsert_entity(conn, entity)
                elif update_mode == 'replace':
                    result = self._replace_entity(conn, entity)
                else:  # insert
                    result = self._insert_entity(conn, entity)
                
                results[result] += 1
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error storing entities: {e}")
            raise
        
        self.logger.info(f"Stored entities: {results}")
        return results
    
    def _upsert_entity(self, conn: duckdb.DuckDBPyConnection, entity: CodeEntity) -> str:
        """Insert or update entity based on file_path + name + type."""
        
        # Check if entity exists
        existing = conn.execute("""
            SELECT id, ast_hash FROM entities 
            WHERE file_path = ? AND name = ? AND type = ?
        """, [entity.file_path, entity.name, entity.type.value]).fetchone()
        
        if existing:
            existing_id, existing_hash = existing
            
            # Check if AST hash has changed
            if existing_hash != entity.ast_hash:
                # Update existing entity
                conn.execute("""
                    UPDATE entities SET 
                        line_start = ?, line_end = ?, ast_hash = ?, 
                        metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [
                    entity.location.line_start if entity.location else None,
                    entity.location.line_end if entity.location else None,
                    entity.ast_hash,
                    json.dumps(entity.metadata),
                    existing_id
                ])
                return 'updated'
            else:
                return 'skipped'
        else:
            # Insert new entity
            conn.execute("""
                INSERT INTO entities (id, type, name, file_path, line_start, line_end, ast_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                str(entity.id),
                entity.type.value,
                entity.name,
                entity.file_path,
                entity.location.line_start if entity.location else None,
                entity.location.line_end if entity.location else None,
                entity.ast_hash,
                json.dumps(entity.metadata)
            ])
            return 'inserted'
    
    def _replace_entity(self, conn: duckdb.DuckDBPyConnection, entity: CodeEntity) -> str:
        """Replace entity regardless of hash."""
        
        # Try to update first
        result = conn.execute("""
            UPDATE entities SET 
                line_start = ?, line_end = ?, ast_hash = ?, 
                metadata = ?, updated_at = CURRENT_TIMESTAMP
            WHERE file_path = ? AND name = ? AND type = ?
        """, [
            entity.location.line_start if entity.location else None,
            entity.location.line_end if entity.location else None,
            entity.ast_hash,
            json.dumps(entity.metadata),
            entity.file_path,
            entity.name,
            entity.type.value
        ])
        
        if result.rowcount > 0:
            return 'updated'
        else:
            # Insert if update didn't affect any rows
            conn.execute("""
                INSERT INTO entities (id, type, name, file_path, line_start, line_end, ast_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                str(entity.id),
                entity.type.value,
                entity.name,
                entity.file_path,
                entity.location.line_start if entity.location else None,
                entity.location.line_end if entity.location else None,
                entity.ast_hash,
                json.dumps(entity.metadata)
            ])
            return 'inserted'
    
    def _insert_entity(self, conn: duckdb.DuckDBPyConnection, entity: CodeEntity) -> str:
        """Insert entity (will fail if duplicate exists)."""
        try:
            conn.execute("""
                INSERT INTO entities (id, type, name, file_path, line_start, line_end, ast_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                str(entity.id),
                entity.type.value,
                entity.name,
                entity.file_path,
                entity.location.line_start if entity.location else None,
                entity.location.line_end if entity.location else None,
                entity.ast_hash,
                json.dumps(entity.metadata)
            ])
            return 'inserted'
        except Exception:
            return 'skipped'  # Likely a duplicate
    
    def store_extraction_result(self, result: ExtractionResult, 
                               update_mode: str = 'upsert') -> Dict[str, int]:
        """Store all entities from an ExtractionResult."""
        if not result.success:
            self.logger.warning(f"Skipping storage for failed extraction: {result.file_path}")
            return {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        return self.store_entities(result.entities, update_mode)
    
    def get_entities_by_file(self, file_path: str) -> List[CodeEntity]:
        """Retrieve all entities for a specific file."""
        conn = self._get_connection()
        
        rows = conn.execute("""
            SELECT id, type, name, file_path, line_start, line_end, 
                   ast_hash, metadata, created_at, updated_at
            FROM entities 
            WHERE file_path = ?
            ORDER BY line_start
        """, [file_path]).fetchall()
        
        entities = []
        for row in rows:
            entity_data = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'file_path': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'ast_hash': row[6],
                'metadata': json.loads(row[7]) if row[7] else {}
            }
            entities.append(CodeEntity.from_db_dict(entity_data))
        
        return entities
    
    def get_entities_by_type(self, entity_type: str, limit: int = 1000) -> List[CodeEntity]:
        """Retrieve entities of a specific type."""
        conn = self._get_connection()
        
        rows = conn.execute("""
            SELECT id, type, name, file_path, line_start, line_end, 
                   ast_hash, metadata, created_at, updated_at
            FROM entities 
            WHERE type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, [entity_type, limit]).fetchall()
        
        entities = []
        for row in rows:
            entity_data = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'file_path': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'ast_hash': row[6],
                'metadata': json.loads(row[7]) if row[7] else {}
            }
            entities.append(CodeEntity.from_db_dict(entity_data))
        
        return entities
    
    def search_entities(self, name_pattern: str, entity_type: Optional[str] = None) -> List[CodeEntity]:
        """Search entities by name pattern."""
        conn = self._get_connection()
        
        query = """
            SELECT id, type, name, file_path, line_start, line_end, 
                   ast_hash, metadata, created_at, updated_at
            FROM entities 
            WHERE name LIKE ?
        """
        params = [f"%{name_pattern}%"]
        
        if entity_type:
            query += " AND type = ?"
            params.append(entity_type)
        
        query += " ORDER BY name LIMIT 100"
        
        rows = conn.execute(query, params).fetchall()
        
        entities = []
        for row in rows:
            entity_data = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'file_path': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'ast_hash': row[6],
                'metadata': json.loads(row[7]) if row[7] else {}
            }
            entities.append(CodeEntity.from_db_dict(entity_data))
        
        return entities
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored entities."""
        conn = self._get_connection()
        
        stats = {}
        
        # Total counts
        result = conn.execute("""
            SELECT 
                COUNT(*) as total_entities,
                COUNT(DISTINCT file_path) as total_files,
                COUNT(DISTINCT type) as entity_types
            FROM entities
        """).fetchone()
        
        stats['total_entities'] = result[0]
        stats['total_files'] = result[1]
        stats['entity_types'] = result[2]
        
        # Counts by type
        type_counts = conn.execute("""
            SELECT type, COUNT(*) as count
            FROM entities
            GROUP BY type
            ORDER BY count DESC
        """).fetchall()
        
        stats['by_type'] = {row[0]: row[1] for row in type_counts}
        
        # Recent activity
        recent = conn.execute("""
            SELECT COUNT(*) as recent_updates
            FROM entities
            WHERE updated_at >= NOW() - INTERVAL '1 hour'
        """).fetchone()
        
        stats['recent_updates'] = recent[0]
        
        return stats
    
    def cleanup_orphaned_entities(self) -> int:
        """Remove entities for files that no longer exist."""
        conn = self._get_connection()
        
        # Get all unique file paths
        file_paths = conn.execute("""
            SELECT DISTINCT file_path FROM entities
        """).fetchall()
        
        orphaned_paths = []
        for (file_path,) in file_paths:
            if not Path(file_path).exists():
                orphaned_paths.append(file_path)
        
        if not orphaned_paths:
            return 0
        
        # Delete orphaned entities
        placeholders = ','.join(['?' for _ in orphaned_paths])
        result = conn.execute(f"""
            DELETE FROM entities WHERE file_path IN ({placeholders})
        """, orphaned_paths)
        
        deleted_count = result.rowcount
        self.logger.info(f"Cleaned up {deleted_count} orphaned entities from {len(orphaned_paths)} files")
        
        return deleted_count
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class EntityExtractionPipeline:
    """
    Complete pipeline for extracting and storing entities from source code.
    
    Combines EntityExtractor and EntityStorage for end-to-end processing.
    """
    
    def __init__(self, db_path: str = "knowledge/chromadb/entities.duckdb"):
        self.extractor = EntityExtractor()
        self.storage = EntityStorage(db_path)
        self.logger = logging.getLogger(__name__)
    
    def process_file(self, file_path: str, update_mode: str = 'upsert') -> Dict[str, Any]:
        """Process a single file: extract entities and store them."""
        
        # Extract entities
        result = self.extractor.extract_from_file(file_path)
        
        # Store entities
        storage_result = self.storage.store_extraction_result(result, update_mode)
        
        return {
            'file_path': file_path,
            'extraction_success': result.success,
            'extraction_time': result.extraction_time,
            'entities_found': len(result.entities),
            'entities_stored': storage_result['inserted'] + storage_result['updated'],
            'storage_result': storage_result,
            'error_message': result.error_message
        }
    
    def process_directory(self, directory_path: str, 
                         extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         exclude_patterns: Optional[List[str]] = None,
                         update_mode: str = 'upsert') -> Dict[str, Any]:
        """Process all files in a directory."""
        
        start_time = time.time()
        
        # Extract entities from all files
        extraction_results = self.extractor.extract_from_directory(
            directory_path, extensions, recursive, exclude_patterns
        )
        
        # Store all results
        total_storage_stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        for result in extraction_results:
            storage_stats = self.storage.store_extraction_result(result, update_mode)
            for key in total_storage_stats:
                total_storage_stats[key] += storage_stats[key]
        
        # Calculate summary statistics
        successful_extractions = [r for r in extraction_results if r.success]
        failed_extractions = [r for r in extraction_results if not r.success]
        
        total_entities = sum(len(r.entities) for r in successful_extractions)
        total_time = time.time() - start_time
        
        summary = {
            'directory': directory_path,
            'total_files': len(extraction_results),
            'successful_extractions': len(successful_extractions),
            'failed_extractions': len(failed_extractions),
            'total_entities_extracted': total_entities,
            'total_entities_stored': total_storage_stats['inserted'] + total_storage_stats['updated'],
            'storage_stats': total_storage_stats,
            'processing_time': total_time,
            'avg_time_per_file': total_time / len(extraction_results) if extraction_results else 0
        }
        
        self.logger.info(f"Directory processing complete: {summary}")
        
        return summary
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the entire pipeline."""
        
        extraction_metrics = self.extractor.get_extraction_metrics()
        storage_stats = self.storage.get_file_statistics()
        
        return {
            'extraction': extraction_metrics,
            'storage': storage_stats,
            'combined': {
                'total_files_processed': extraction_metrics['files_processed'],
                'total_entities_in_db': storage_stats['total_entities'],
                'extraction_success_rate': extraction_metrics['success_rate']
            }
        }
    
    def close(self):
        """Clean shutdown of pipeline."""
        self.storage.close()