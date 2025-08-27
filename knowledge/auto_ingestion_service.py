#!/usr/bin/env python3
"""
Automatic Knowledge Ingestion Service

This service monitors for new knowledge files and automatically ingests them
into the knowledge database. It runs separately from the MCP server to avoid
locking conflicts.

Usage:
    python3 auto_ingestion_service.py          # Run once
    python3 auto_ingestion_service.py --watch  # Watch for changes
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import duckdb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IngestionService')


class KnowledgeIngestionService:
    """Service for automatic knowledge ingestion"""
    
    def __init__(self, db_path: str = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"):
        self.db_path = db_path
        self.knowledge_path = Path("/Users/cal/DEV/RIF/knowledge")
        self.last_ingestion = {}
        
    def check_mcp_server_lock(self) -> bool:
        """Check if MCP server has database locked"""
        try:
            # Try to open in write mode
            conn = duckdb.connect(self.db_path)
            conn.close()
            return False  # No lock
        except Exception as e:
            if "Conflicting lock" in str(e):
                return True  # Locked
            raise
    
    def ingest_new_files(self) -> Dict[str, int]:
        """Ingest new or modified knowledge files"""
        stats = {
            'patterns': 0,
            'issues': 0,
            'decisions': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # Check for lock
        if self.check_mcp_server_lock():
            logger.warning("Database is locked by MCP server. Waiting for it to release...")
            # Could implement retry logic here
            return stats
        
        conn = duckdb.connect(self.db_path)
        
        try:
            # Process patterns
            pattern_dir = self.knowledge_path / "patterns"
            if pattern_dir.exists():
                for file_path in pattern_dir.glob("*.json"):
                    if self._should_process(file_path):
                        if self._ingest_file(conn, file_path, 'pattern'):
                            stats['patterns'] += 1
                        else:
                            stats['errors'] += 1
            
            # Process issues
            issues_dir = self.knowledge_path / "issues"
            if issues_dir.exists():
                for file_path in issues_dir.glob("*.json"):
                    if self._should_process(file_path):
                        if self._ingest_file(conn, file_path, 'issue'):
                            stats['issues'] += 1
                        else:
                            stats['errors'] += 1
            
            # Process decisions
            decisions_dir = self.knowledge_path / "decisions"
            if decisions_dir.exists():
                for file_path in decisions_dir.glob("*.json"):
                    if self._should_process(file_path):
                        if self._ingest_file(conn, file_path, 'decision'):
                            stats['decisions'] += 1
                        else:
                            stats['errors'] += 1
            
            conn.commit()
            
        finally:
            conn.close()
        
        return stats
    
    def _should_process(self, file_path: Path) -> bool:
        """Check if file should be processed based on modification time"""
        mtime = file_path.stat().st_mtime
        last_time = self.last_ingestion.get(str(file_path), 0)
        
        if mtime > last_time:
            self.last_ingestion[str(file_path)] = mtime
            return True
        return False
    
    def _ingest_file(self, conn, file_path: Path, item_type: str) -> bool:
        """Ingest a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            item_id = data.get(f'{item_type}_id') or file_path.stem
            name = data.get(f'{item_type}_name') or data.get('name') or file_path.stem
            description = data.get('description') or str(data.get('problem_description', ''))
            
            # Check if already exists
            existing = conn.execute("""
                SELECT id FROM entities 
                WHERE type = ? AND name = ?
            """, [item_type, name]).fetchone()
            
            if existing:
                # Update
                conn.execute("""
                    UPDATE entities 
                    SET description = ?, content = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [description, json.dumps(data), existing[0]])
                logger.debug(f"Updated {item_type}: {name}")
            else:
                # Insert
                conn.execute("""
                    INSERT INTO entities (type, name, description, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, [item_type, name, description, json.dumps(data), json.dumps({'source': str(file_path)})])
                logger.info(f"Added new {item_type}: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return False
    
    def extract_code_entities(self, target_path: str = None) -> Dict[str, int]:
        """Extract entities from Python code files"""
        stats = {'files': 0, 'entities': 0, 'errors': 0}
        
        if self.check_mcp_server_lock():
            logger.warning("Database locked. Skipping code extraction.")
            return stats
        
        # Import extraction modules
        sys.path.append('/Users/cal/DEV/RIF')
        from knowledge.extraction.entity_extractor import EntityExtractor
        from knowledge.extraction.storage_integration import EntityStorage
        
        extractor = EntityExtractor()
        storage = EntityStorage(self.db_path)
        
        # Default to knowledge directory if no target specified
        target = Path(target_path) if target_path else self.knowledge_path
        
        # Extract from Python files
        for py_file in target.rglob("*.py"):
            try:
                result = extractor.extract_from_file(str(py_file))
                if result.success:
                    storage_stats = storage.store_entities(result.entities)
                    stats['files'] += 1
                    stats['entities'] += storage_stats.get('inserted', 0)
                else:
                    stats['errors'] += 1
            except Exception as e:
                logger.error(f"Error extracting from {py_file}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def run_once(self):
        """Run ingestion once"""
        logger.info("Starting knowledge ingestion...")
        
        # Ingest knowledge files
        stats = self.ingest_new_files()
        logger.info(f"Ingested: {stats['patterns']} patterns, {stats['issues']} issues, {stats['decisions']} decisions")
        
        # Extract code entities (optional)
        # code_stats = self.extract_code_entities()
        # logger.info(f"Extracted: {code_stats['entities']} entities from {code_stats['files']} files")
        
        # Show database stats
        self.show_stats()
    
    def watch(self, interval: int = 60):
        """Watch for changes and ingest periodically"""
        logger.info(f"Watching for changes (checking every {interval} seconds)...")
        
        while True:
            try:
                stats = self.ingest_new_files()
                if any(stats[k] > 0 for k in ['patterns', 'issues', 'decisions']):
                    logger.info(f"Ingested: {stats}")
                    self.show_stats()
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping ingestion service...")
                break
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                time.sleep(interval)
    
    def show_stats(self):
        """Show current database statistics"""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            
            total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            patterns = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'pattern'").fetchone()[0]
            issues = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'issue'").fetchone()[0]
            decisions = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'decision'").fetchone()[0]
            relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            
            logger.info(f"Database stats: {total} entities ({patterns} patterns, {issues} issues, {decisions} decisions), {relationships} relationships")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")


def main():
    parser = argparse.ArgumentParser(description='Knowledge Ingestion Service')
    parser.add_argument('--watch', action='store_true', help='Watch for changes continuously')
    parser.add_argument('--interval', type=int, default=60, help='Watch interval in seconds')
    parser.add_argument('--extract-code', action='store_true', help='Also extract code entities')
    parser.add_argument('--db', default="/Users/cal/DEV/RIF/knowledge/knowledge.duckdb", help='Database path')
    
    args = parser.parse_args()
    
    service = KnowledgeIngestionService(args.db)
    
    if args.extract_code:
        logger.info("Extracting code entities...")
        stats = service.extract_code_entities()
        logger.info(f"Extracted: {stats}")
    
    if args.watch:
        service.watch(args.interval)
    else:
        service.run_once()


if __name__ == "__main__":
    main()