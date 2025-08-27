#!/usr/bin/env python3
"""
Knowledge Ingestion Script - Populates the consolidated knowledge database
This ingests all JSON knowledge files (patterns, issues, decisions) into the database
"""

import duckdb
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class KnowledgeIngester:
    """Ingests knowledge from JSON files into the database"""
    
    def __init__(self, db_path: str = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.stats = {
            'patterns': 0,
            'issues': 0,
            'decisions': 0,
            'capabilities': 0,
            'errors': 0,
            'skipped': 0
        }
    
    def ingest_json_file(self, file_path: Path, item_type: str) -> bool:
        """Ingest a single JSON knowledge file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract common fields
            item_id = data.get('pattern_id') or data.get('issue_id') or data.get('decision_id') or file_path.stem
            title = data.get('pattern_name') or data.get('issue_title') or data.get('decision_name') or data.get('name') or file_path.stem
            
            # Get description from various possible fields
            description = None
            if 'description' in data:
                description = data['description']
            elif 'problem_description' in data:
                description = str(data['problem_description'])
            elif 'problem_context' in data:
                description = str(data['problem_context'])
            elif 'summary' in data:
                description = data['summary']
            
            category = data.get('category') or data.get('issue_type') or data.get('decision_type')
            
            # Extract tags from various fields
            tags = []
            if 'tags' in data:
                tags = data['tags'] if isinstance(data['tags'], list) else [data['tags']]
            if 'metadata' in data and isinstance(data.get('metadata'), dict):
                if 'tags' in data['metadata']:
                    tags.extend(data['metadata']['tags'] if isinstance(data['metadata']['tags'], list) else [data['metadata']['tags']])
            
            # Check if already exists
            existing = self.conn.execute("""
                SELECT id FROM knowledge_items WHERE item_id = ?
            """, [item_id]).fetchone()
            
            if existing:
                # Update existing
                self.conn.execute("""
                    UPDATE knowledge_items 
                    SET title = ?, category = ?, description = ?, content = ?, 
                        tags = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = ?
                """, [title, category, description, json.dumps(data), json.dumps(tags), json.dumps(data.get('metadata', {})), item_id])
                self.stats['skipped'] += 1
            else:
                # Insert new
                self.conn.execute("""
                    INSERT INTO knowledge_items (item_type, item_id, title, category, description, content, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [item_type, item_id, title, category, description, json.dumps(data), json.dumps(tags), json.dumps(data.get('metadata', {}))])
                self.stats[f'{item_type}s'] += 1
            
            # Also insert as entity for compatibility with MCP server queries
            self.conn.execute("""
                INSERT INTO entities (type, name, description, content, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
            """, [item_type, title, description, json.dumps(data), json.dumps(data.get('metadata', {}))])
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def ingest_patterns(self):
        """Ingest all pattern files"""
        print("\n📚 Ingesting patterns...")
        pattern_dir = Path("/Users/cal/DEV/RIF/knowledge/patterns")
        
        if not pattern_dir.exists():
            print("  ⚠️ Patterns directory not found")
            return
        
        pattern_files = list(pattern_dir.glob("*.json"))
        print(f"  Found {len(pattern_files)} pattern files")
        
        for file_path in pattern_files:
            self.ingest_json_file(file_path, 'pattern')
    
    def ingest_issues(self):
        """Ingest all issue resolution files"""
        print("\n🐛 Ingesting issue resolutions...")
        issues_dir = Path("/Users/cal/DEV/RIF/knowledge/issues")
        
        if not issues_dir.exists():
            print("  ⚠️ Issues directory not found")
            return
        
        issue_files = list(issues_dir.glob("*.json"))
        print(f"  Found {len(issue_files)} issue files")
        
        for file_path in issue_files:
            self.ingest_json_file(file_path, 'issue')
    
    def ingest_decisions(self):
        """Ingest all decision files"""
        print("\n🎯 Ingesting decisions...")
        decisions_dir = Path("/Users/cal/DEV/RIF/knowledge/decisions")
        
        if not decisions_dir.exists():
            print("  ⚠️ Decisions directory not found")
            return
        
        decision_files = list(decisions_dir.glob("*.json"))
        print(f"  Found {len(decision_files)} decision files")
        
        for file_path in decision_files:
            self.ingest_json_file(file_path, 'decision')
    
    def ingest_claude_knowledge(self):
        """Ingest Claude Code capabilities and limitations"""
        print("\n🤖 Ingesting Claude Code knowledge...")
        
        capabilities = [
            {
                'name': 'File Operations',
                'description': 'Can read, write, edit, and manipulate files directly',
                'tools': ['Read', 'Write', 'Edit', 'MultiEdit']
            },
            {
                'name': 'Command Execution',
                'description': 'Can execute shell commands and scripts',
                'tools': ['Bash']
            },
            {
                'name': 'Code Analysis',
                'description': 'Can search, analyze and understand code across large codebases',
                'tools': ['Grep', 'Glob', 'Read']
            },
            {
                'name': 'Web Access',
                'description': 'Can search the web and fetch content from URLs',
                'tools': ['WebSearch', 'WebFetch']
            },
            {
                'name': 'Task Delegation',
                'description': 'Can launch specialized subagents for complex tasks',
                'tools': ['Task']
            },
            {
                'name': 'MCP Server Integration',
                'description': 'Can connect to external tools, databases, and APIs via MCP',
                'tools': []
            }
        ]
        
        limitations = [
            {
                'name': 'No persistent background processes',
                'description': 'Cannot run persistent processes beyond session scope',
                'workaround': 'Use Bash with run_in_background for session-scoped tasks'
            },
            {
                'name': 'Task.parallel() is pseudocode',
                'description': 'Task.parallel() is not a real function',
                'workaround': 'Launch multiple Task tools in one response for parallel execution'
            },
            {
                'name': 'No direct agent communication',
                'description': 'Agents cannot communicate directly with each other',
                'workaround': 'Agents coordinate through files or GitHub issues'
            },
            {
                'name': 'Session-scoped memory',
                'description': 'Memory is limited to current session',
                'workaround': 'Explicitly save state to files for persistence'
            }
        ]
        
        # Insert capabilities
        for cap in capabilities:
            self.conn.execute("""
                INSERT INTO entities (type, name, description, metadata)
                VALUES ('capability', ?, ?, ?)
                ON CONFLICT DO NOTHING
            """, ['Claude: ' + cap['name'], cap['description'], json.dumps({'tools': cap.get('tools', [])})])
            self.stats['capabilities'] += 1
        
        # Insert limitations
        for lim in limitations:
            self.conn.execute("""
                INSERT INTO entities (type, name, description, metadata)
                VALUES ('limitation', ?, ?, ?)
                ON CONFLICT DO NOTHING
            """, ['Claude Limitation: ' + lim['name'], lim['description'], json.dumps({'workaround': lim.get('workaround', '')})])
            self.stats['capabilities'] += 1
        
        print(f"  Added {len(capabilities)} capabilities and {len(limitations)} limitations")
    
    def run(self):
        """Run the full ingestion process"""
        print("🚀 Starting knowledge ingestion...")
        print(f"Database: {self.db_path}")
        
        # Ingest all knowledge types
        self.ingest_patterns()
        self.ingest_issues()
        self.ingest_decisions()
        self.ingest_claude_knowledge()
        
        # Commit changes
        self.conn.commit()
        
        # Print statistics
        print("\n📊 Ingestion Statistics:")
        print(f"  Patterns: {self.stats['patterns']}")
        print(f"  Issues: {self.stats['issues']}")
        print(f"  Decisions: {self.stats['decisions']}")
        print(f"  Claude Knowledge: {self.stats['capabilities']}")
        print(f"  Skipped (already exists): {self.stats['skipped']}")
        print(f"  Errors: {self.stats['errors']}")
        
        # Verify final counts
        total_knowledge = self.conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]
        total_entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        
        print(f"\n✅ Database now contains:")
        print(f"  Knowledge items: {total_knowledge}")
        print(f"  Entities: {total_entities}")
        
        self.conn.close()

if __name__ == "__main__":
    ingester = KnowledgeIngester()
    ingester.run()