#!/usr/bin/env python3
"""
Extract relationships for all existing entities in the knowledge database.

This script processes all code entities that were extracted but don't have
relationships detected yet. It groups entities by file and runs relationship
detection to find calls, imports, inheritances, and other connections.
"""

import os
import sys
import json
import logging
import duckdb
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from uuid import UUID
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.relationships.relationship_detector import RelationshipDetector
from knowledge.parsing.parser_manager import ParserManager
from knowledge.extraction.entity_types import CodeEntity, EntityType, SourceLocation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RelationshipExtractor')


@dataclass
class EntityRecord:
    """Represents an entity from the database."""
    id: str
    type: str
    name: str
    file_path: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    metadata: Dict[str, Any]


class RelationshipExtractor:
    """Extract relationships for all entities in the knowledge database."""
    
    def __init__(self, db_path: str = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"):
        self.db_path = db_path
        self.parser_manager = ParserManager()
        self.detector = RelationshipDetector(self.parser_manager)
        # Don't initialize storage here to avoid connection conflicts
        
    def load_entities_by_file(self) -> Dict[str, List[EntityRecord]]:
        """Load all entities grouped by file path."""
        entities_by_file = {}
        
        with duckdb.connect(self.db_path, read_only=True) as conn:
            # Get all code entities with file paths
            query = """
                SELECT id, type, name, file_path, line_start, line_end, metadata
                FROM entities
                WHERE type IN ('module', 'class', 'function', 'variable', 'constant')
                  AND file_path IS NOT NULL
                ORDER BY file_path, line_start
            """
            
            results = conn.execute(query).fetchall()
            
            for row in results:
                entity = EntityRecord(
                    id=str(row[0]),
                    type=row[1],
                    name=row[2],
                    file_path=row[3],
                    line_start=row[4],
                    line_end=row[5],
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                
                if entity.file_path not in entities_by_file:
                    entities_by_file[entity.file_path] = []
                entities_by_file[entity.file_path].append(entity)
        
        logger.info(f"Loaded {sum(len(e) for e in entities_by_file.values())} entities from {len(entities_by_file)} files")
        return entities_by_file
    
    def store_relationships_direct(self, relationships: List[Any]) -> int:
        """Store relationships directly in the database."""
        stored = 0
        
        with duckdb.connect(self.db_path) as conn:
            for rel in relationships:
                try:
                    # Check if relationship already exists
                    existing = conn.execute("""
                        SELECT id FROM relationships 
                        WHERE source_id = ? AND target_id = ? AND relationship_type = ?
                    """, [str(rel.source_id), str(rel.target_id), rel.relationship_type.value]).fetchone()
                    
                    if not existing:
                        # Insert new relationship
                        conn.execute("""
                            INSERT INTO relationships (source_id, target_id, relationship_type, confidence, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        """, [
                            str(rel.source_id),
                            str(rel.target_id), 
                            rel.relationship_type.value,
                            rel.confidence,
                            json.dumps(rel.metadata) if rel.metadata else '{}'
                        ])
                        stored += 1
                except Exception as e:
                    logger.debug(f"Error storing relationship: {e}")
        
        return stored
    
    def convert_to_code_entity(self, entity: EntityRecord) -> CodeEntity:
        """Convert database entity to CodeEntity for relationship detection."""
        # Map database types to EntityType enum
        type_map = {
            'module': EntityType.MODULE,
            'class': EntityType.CLASS,
            'function': EntityType.FUNCTION,
            'variable': EntityType.VARIABLE,
            'constant': EntityType.CONSTANT
        }
        
        # Create location object if line numbers exist
        location = None
        if entity.line_start is not None and entity.line_end is not None:
            location = SourceLocation(
                line_start=entity.line_start,
                line_end=entity.line_end,
                column_start=0,
                column_end=0
            )
        
        return CodeEntity(
            id=UUID(entity.id),
            name=entity.name,
            type=type_map.get(entity.type, EntityType.FUNCTION),  # Default to function if unknown
            file_path=entity.file_path or '',
            location=location,
            metadata=entity.metadata or {}
        )
    
    def extract_relationships(self) -> Dict[str, Any]:
        """Extract relationships for all entities."""
        stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'entities_processed': 0,
            'relationships_found': 0,
            'relationships_stored': 0,
            'errors': []
        }
        
        entities_by_file = self.load_entities_by_file()
        
        for file_path, entities in entities_by_file.items():
            # Skip non-existent files
            if not Path(file_path).exists():
                logger.debug(f"Skipping non-existent file: {file_path}")
                stats['files_skipped'] += 1
                continue
            
            # Skip test files and __pycache__
            if '__pycache__' in file_path or '/test' in file_path or '\\test' in file_path:
                stats['files_skipped'] += 1
                continue
            
            try:
                logger.debug(f"Processing {file_path} with {len(entities)} entities")
                
                # Convert to CodeEntity objects
                code_entities = [self.convert_to_code_entity(e) for e in entities]
                
                # Detect relationships - fix parser incompatibility
                # The detector expects (tree, language) but parser returns dict
                # Patch the parser temporarily
                original_parse = self.detector.parser_manager.parse_file
                def patched_parse(fp):
                    result = original_parse(fp)
                    if isinstance(result, dict):
                        return result.get('tree'), result.get('language', 'python')
                    return result
                
                self.detector.parser_manager.parse_file = patched_parse
                
                detection_result = self.detector.detect_relationships_from_file(
                    file_path, code_entities
                )
                
                # Restore original parser
                self.detector.parser_manager.parse_file = original_parse
                
                if detection_result.success:
                    stats['files_processed'] += 1
                    stats['entities_processed'] += len(entities)
                    stats['relationships_found'] += len(detection_result.relationships)
                    
                    # Store relationships directly using DuckDB
                    if detection_result.relationships:
                        stored = self.store_relationships_direct(detection_result.relationships)
                        stats['relationships_stored'] += stored
                        
                        if stored > 0:
                            logger.info(f"Found {len(detection_result.relationships)} relationships in {file_path}")
                else:
                    logger.warning(f"Failed to detect relationships in {file_path}: {detection_result.error_message}")
                    stats['errors'].append(f"{file_path}: {detection_result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'].append(f"{file_path}: {str(e)}")
        
        return stats
    
    def get_current_stats(self) -> Dict[str, int]:
        """Get current database statistics."""
        with duckdb.connect(self.db_path, read_only=True) as conn:
            total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            total_relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            
            rel_types = conn.execute("""
                SELECT relationship_type, COUNT(*) as count
                FROM relationships
                GROUP BY relationship_type
                ORDER BY count DESC
            """).fetchall()
            
            return {
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'relationship_types': {r[0]: r[1] for r in rel_types}
            }
    
    def run(self):
        """Run the complete relationship extraction process."""
        logger.info("=" * 60)
        logger.info("RELATIONSHIP EXTRACTION STARTING")
        logger.info("=" * 60)
        
        # Get initial stats
        initial_stats = self.get_current_stats()
        logger.info(f"Initial state: {initial_stats['total_entities']} entities, {initial_stats['total_relationships']} relationships")
        
        # Extract relationships
        start_time = time.time()
        extraction_stats = self.extract_relationships()
        elapsed_time = time.time() - start_time
        
        # Get final stats
        final_stats = self.get_current_stats()
        
        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files processed: {extraction_stats['files_processed']}")
        logger.info(f"Files skipped: {extraction_stats['files_skipped']}")
        logger.info(f"Entities processed: {extraction_stats['entities_processed']}")
        logger.info(f"Relationships found: {extraction_stats['relationships_found']}")
        logger.info(f"Relationships stored: {extraction_stats['relationships_stored']}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        logger.info("\nFinal database state:")
        logger.info(f"  Total entities: {final_stats['total_entities']:,}")
        logger.info(f"  Total relationships: {final_stats['total_relationships']:,}")
        logger.info(f"  New relationships added: {final_stats['total_relationships'] - initial_stats['total_relationships']}")
        
        if final_stats['relationship_types']:
            logger.info("\nRelationship types:")
            for rel_type, count in final_stats['relationship_types'].items():
                logger.info(f"  {rel_type}: {count}")
        
        if extraction_stats['errors']:
            logger.warning(f"\nErrors encountered: {len(extraction_stats['errors'])}")
            for error in extraction_stats['errors'][:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
        
        # Success check
        if final_stats['total_relationships'] >= 1000:
            logger.info("\n✅ SUCCESS: Knowledge graph now has 1000+ relationships!")
        elif final_stats['total_relationships'] > initial_stats['total_relationships']:
            logger.info(f"\n⚠️  Added {final_stats['total_relationships'] - initial_stats['total_relationships']} relationships, but total is still under 1000")
        else:
            logger.warning("\n❌ No new relationships were added")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract relationships for all entities')
    parser.add_argument('--db', default='/Users/cal/DEV/RIF/knowledge/knowledge.duckdb',
                       help='Path to knowledge database')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = RelationshipExtractor(args.db)
    extractor.run()


if __name__ == "__main__":
    main()