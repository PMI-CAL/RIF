#!/usr/bin/env python3
"""
Fix Knowledge Loader - Properly populate the RIF knowledge graph
Extracts meaningful names and creates searchable content from JSON files
"""

import json
import duckdb
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import re

class KnowledgeGraphFixer:
    """Load JSON knowledge into database with meaningful names"""
    
    def __init__(self):
        self.db_path = "/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb"
        self.knowledge_path = Path("/Users/cal/DEV/RIF/knowledge")
        self.conn = None
        self.loaded_count = 0
        self.error_count = 0
        
    def connect_db(self):
        """Connect to database"""
        self.conn = duckdb.connect(self.db_path)
        print(f"Connected to {self.db_path}")
        
    def clear_existing_patterns(self):
        """Clear existing pattern entities that have generic names"""
        # Delete entities with generic timestamp-based names
        deleted = self.conn.execute("""
            DELETE FROM entities 
            WHERE type = 'pattern' 
            AND name LIKE 'patterns_item_doc_%'
        """).fetchall()
        
        print(f"Cleared generic pattern entities")
        
        # Also clear any existing relationships
        self.conn.execute("DELETE FROM relationships WHERE source_id NOT IN (SELECT id FROM entities)")
        print("Cleaned up orphaned relationships")
        
    def extract_pattern_info(self, data: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """Extract meaningful information from pattern JSON"""
        
        # Try different name fields
        name = None
        if 'pattern_name' in data:
            name = data['pattern_name']
        elif 'pattern_id' in data:
            name = data['pattern_id'].replace('-', ' ').title()
        elif 'name' in data:
            name = data['name']
        else:
            # Use filename as fallback
            name = filename.replace('.json', '').replace('-', ' ').replace('_', ' ').title()
        
        if not name:
            return None
            
        # Extract description
        description = ""
        if 'description' in data:
            description = data['description']
        elif 'problem_context' in data and isinstance(data['problem_context'], dict):
            description = data['problem_context'].get('solution_pattern', '')
        elif 'summary' in data:
            description = data['summary']
        
        # Extract all searchable keywords
        keywords = []
        
        # Add common fields
        for field in ['category', 'domain', 'technology', 'solution_approach']:
            if field in data:
                keywords.append(str(data[field]))
        
        # Extract from nested structures
        if 'implementation' in data:
            impl = data['implementation']
            if isinstance(impl, dict):
                for key, value in impl.items():
                    if isinstance(value, str):
                        keywords.append(value)
                    elif isinstance(value, list):
                        keywords.extend([str(v) for v in value if isinstance(v, str)])
        
        # Extract from problem context
        if 'problem_context' in data:
            ctx = data['problem_context']
            if isinstance(ctx, dict):
                for key, value in ctx.items():
                    if isinstance(value, str):
                        keywords.append(value)
        
        # Extract from architectural components
        if 'architectural_components' in data:
            arch = data['architectural_components']
            if isinstance(arch, dict):
                for comp_name, comp_data in arch.items():
                    keywords.append(comp_name.replace('_', ' '))
                    if isinstance(comp_data, dict):
                        for key, value in comp_data.items():
                            if isinstance(value, str):
                                keywords.append(value)
        
        # Create searchable metadata
        metadata = {
            'name': name,
            'description': description,
            'keywords': ' '.join(keywords),
            'source_file': filename,
            'type': 'pattern'
        }
        
        # Add original fields for completeness
        for key in ['category', 'complexity', 'reusability', 'effectiveness']:
            if key in data:
                metadata[key] = data[key]
                
        return metadata
    
    def load_pattern_files(self):
        """Load all pattern JSON files with meaningful names"""
        pattern_dir = self.knowledge_path / "patterns"
        pattern_files = list(pattern_dir.glob("*.json"))
        
        print(f"\nLoading {len(pattern_files)} pattern files...")
        
        for file_path in pattern_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract meaningful info
                info = self.extract_pattern_info(data, file_path.name)
                if not info:
                    print(f"⚠️ Could not extract info from {file_path.name}")
                    self.error_count += 1
                    continue
                
                # Create entity with meaningful name
                entity_id = str(uuid.uuid4())
                
                # Insert into database
                self.conn.execute("""
                    INSERT INTO entities (id, type, name, file_path, metadata, created_at)
                    VALUES (?, 'pattern', ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    entity_id,
                    info['name'],
                    str(file_path),
                    json.dumps(info)
                ])
                
                self.loaded_count += 1
                print(f"✅ Loaded: {info['name']}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
                self.error_count += 1
    
    def load_claude_research(self):
        """Load Claude Code research with searchable structure"""
        research_file = self.knowledge_path / "research" / "claude-code-research.json"
        
        if not research_file.exists():
            print("⚠️ Claude Code research file not found")
            return
            
        print("\\nLoading Claude Code research...")
        
        try:
            with open(research_file, 'r') as f:
                data = json.load(f)
            
            # Create multiple searchable entities from the research
            entities = []
            
            if 'claude_code_reality' in data:
                reality = data['claude_code_reality']
                
                # Core identity
                if 'core_identity' in reality:
                    identity = reality['core_identity']
                    entities.append({
                        'name': 'Claude Code Core Identity',
                        'description': f"What it is: {identity.get('what_it_is', '')}. What it's not: {identity.get('what_it_is_not', '')}",
                        'keywords': f"Claude Code identity {identity.get('what_it_is', '')} {identity.get('what_it_is_not', '')} {identity.get('primary_function', '')}",
                        'category': 'claude_identity'
                    })
                
                # Architecture
                if 'actual_architecture' in reality:
                    arch = reality['actual_architecture']
                    tools_list = []
                    if 'tools' in arch:
                        for tool_category, tools in arch['tools'].items():
                            if isinstance(tools, list):
                                tools_list.extend(tools)
                    
                    entities.append({
                        'name': 'Claude Code Architecture',
                        'description': f"Core component: {arch.get('core_component', '')}. Tools: {', '.join(tools_list)}",
                        'keywords': f"Claude Code architecture tools {' '.join(tools_list)} {arch.get('core_component', '')}",
                        'category': 'claude_architecture'
                    })
                
                # Capabilities
                if 'capabilities' in reality:
                    caps = reality['capabilities']
                    cap_list = []
                    if isinstance(caps, dict):
                        for cap_name, cap_details in caps.items():
                            cap_list.append(cap_name)
                            if isinstance(cap_details, str):
                                cap_list.append(cap_details)
                    
                    entities.append({
                        'name': 'Claude Code Capabilities',
                        'description': f"Capabilities: {', '.join(cap_list[:10])}",
                        'keywords': f"Claude Code capabilities {' '.join(cap_list)}",
                        'category': 'claude_capabilities'
                    })
                
                # Limitations
                if 'limitations' in reality:
                    limits = reality['limitations']
                    limit_list = []
                    if isinstance(limits, dict):
                        for limit_name, limit_details in limits.items():
                            limit_list.append(limit_name)
                            if isinstance(limit_details, str):
                                limit_list.append(limit_details)
                    
                    entities.append({
                        'name': 'Claude Code Limitations',
                        'description': f"Limitations: {', '.join(limit_list[:10])}",
                        'keywords': f"Claude Code limitations {' '.join(limit_list)} cannot Task.parallel pseudocode persistent background",
                        'category': 'claude_limitations'
                    })
            
            # Insert all entities
            for entity_info in entities:
                entity_id = str(uuid.uuid4())
                
                self.conn.execute("""
                    INSERT INTO entities (id, type, name, file_path, metadata, created_at)
                    VALUES (?, 'knowledge_item', ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    entity_id,
                    entity_info['name'],
                    str(research_file),
                    json.dumps(entity_info)
                ])
                
                self.loaded_count += 1
                print(f"✅ Loaded: {entity_info['name']}")
                
        except Exception as e:
            print(f"❌ Error loading Claude research: {e}")
            self.error_count += 1
    
    def load_decision_files(self):
        """Load decision files with meaningful names"""
        decision_dir = self.knowledge_path / "decisions"
        if not decision_dir.exists():
            return
            
        decision_files = list(decision_dir.glob("*.json"))
        print(f"\\nLoading {len(decision_files)} decision files...")
        
        for file_path in decision_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract decision name
                name = data.get('decision_title') or data.get('title') or file_path.stem.replace('-', ' ').title()
                
                # Create searchable content
                keywords = [name]
                if 'problem' in data:
                    keywords.append(str(data['problem']))
                if 'solution' in data:
                    keywords.append(str(data['solution']))
                if 'rationale' in data:
                    keywords.append(str(data['rationale']))
                
                metadata = {
                    'name': name,
                    'description': data.get('summary', ''),
                    'keywords': ' '.join(keywords),
                    'source_file': file_path.name,
                    'type': 'decision'
                }
                
                entity_id = str(uuid.uuid4())
                
                self.conn.execute("""
                    INSERT INTO entities (id, type, name, file_path, metadata, created_at)
                    VALUES (?, 'decision', ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    entity_id,
                    name,
                    str(file_path),
                    json.dumps(metadata)
                ])
                
                self.loaded_count += 1
                print(f"✅ Loaded decision: {name}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
                self.error_count += 1
    
    def verify_loading(self):
        """Verify that loading was successful"""
        print("\\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        
        # Count entities
        counts = self.conn.execute("""
            SELECT type, COUNT(*) FROM entities GROUP BY type
        """).fetchall()
        
        print("Entity counts:")
        for entity_type, count in counts:
            print(f"  {entity_type}: {count}")
        
        # Test searches
        test_queries = [
            "github",
            "error", 
            "orchestrat",
            "claude",
            "monitoring",
            "enterprise"
        ]
        
        print("\\nSearch verification:")
        for query in test_queries:
            count = self.conn.execute("""
                SELECT COUNT(*) FROM entities
                WHERE LOWER(name) LIKE LOWER(?) OR LOWER(CAST(metadata AS VARCHAR)) LIKE LOWER(?)
            """, [f"%{query}%", f"%{query}%"]).fetchone()[0]
            
            print(f"  '{query}': {count} results")
        
        # Show sample pattern names
        patterns = self.conn.execute("""
            SELECT name FROM entities WHERE type = 'pattern' LIMIT 5
        """).fetchall()
        
        print("\\nSample pattern names:")
        for name, in patterns:
            print(f"  - {name}")
    
    def run(self):
        """Run the complete knowledge loading process"""
        print("🔧 FIXING RIF KNOWLEDGE GRAPH")
        print("="*60)
        
        self.connect_db()
        
        # Clear existing bad data
        print("\\n1. Clearing generic pattern entities...")
        self.clear_existing_patterns()
        
        # Load pattern files
        print("\\n2. Loading pattern files...")
        self.load_pattern_files()
        
        # Load Claude research
        print("\\n3. Loading Claude Code research...")
        self.load_claude_research()
        
        # Load decision files
        print("\\n4. Loading decision files...")
        self.load_decision_files()
        
        # Verify
        self.verify_loading()
        
        print(f"\\n✅ COMPLETE: Loaded {self.loaded_count} entities")
        print(f"❌ Errors: {self.error_count}")
        
        self.conn.close()

if __name__ == "__main__":
    fixer = KnowledgeGraphFixer()
    fixer.run()