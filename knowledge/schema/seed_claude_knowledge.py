#!/usr/bin/env python3
"""
Claude Knowledge Seeder (Issue #97 Phase 1)

Seeds the knowledge graph with Claude Code capabilities, limitations, tools,
and implementation patterns based on the research findings.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

# Add parent directory to path to import RIF modules
sys.path.append(str(Path(__file__).parent.parent))

from database.database_interface import RIFDatabase


class ClaudeKnowledgeSeeder:
    """Seeds Claude Code knowledge into the knowledge graph."""
    
    def __init__(self, db: Optional[RIFDatabase] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize database connection
        if db:
            self.db = db
            self._external_db = True
        else:
            self.db = RIFDatabase()
            self._external_db = False
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._external_db:
            self.db.close()
    
    def _create_entity_hash(self, entity_type: str, name: str) -> str:
        """Create a hash for entity identification."""
        content = f"{entity_type}:{name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def seed_claude_capabilities(self) -> List[str]:
        """Seed Claude Code capabilities into knowledge graph."""
        self.logger.info("Seeding Claude capabilities...")
        
        capabilities = [
            {
                'name': 'File Operations',
                'description': 'Can read, write, edit, and manipulate files directly',
                'tools': ['Read', 'Write', 'Edit', 'MultiEdit'],
                'evidence': 'Built-in tools documented in Claude Code documentation'
            },
            {
                'name': 'Command Execution',
                'description': 'Can execute shell commands and scripts',
                'tools': ['Bash'],
                'evidence': 'Bash tool for running shell commands'
            },
            {
                'name': 'Code Analysis',
                'description': 'Can search, analyze and understand code across large codebases',
                'tools': ['Grep', 'Glob', 'Read'],
                'evidence': 'Built-in search and analysis tools'
            },
            {
                'name': 'Git Integration',
                'description': 'Can create commits, branches, and manage Git operations',
                'tools': ['Bash'],
                'evidence': 'Direct Git command execution through Bash tool'
            },
            {
                'name': 'MCP Server Integration',
                'description': 'Can connect to external tools, databases, and APIs via MCP',
                'tools': [],
                'evidence': 'Model Context Protocol servers for external integrations'
            },
            {
                'name': 'Subagent Delegation',
                'description': 'Can create specialized subagents for domain-specific tasks',
                'tools': [],
                'evidence': 'Official subagent documentation and features'
            },
            {
                'name': 'Hook-based Automation',
                'description': 'Can trigger automated actions based on tool usage events',
                'tools': [],
                'evidence': 'Hooks system in .claude/settings.json'
            },
            {
                'name': 'Web Search',
                'description': 'Can search the web for up-to-date information',
                'tools': ['WebSearch', 'WebFetch'],
                'evidence': 'Built-in web search capabilities'
            }
        ]
        
        entity_ids = []
        for cap in capabilities:
            try:
                entity_id = self.db.store_entity({
                    'type': 'claude_capability',
                    'name': cap['name'],
                    'file_path': '/claude/capabilities',
                    'ast_hash': self._create_entity_hash('claude_capability', cap['name']),
                    'metadata': {
                        'description': cap['description'],
                        'supporting_tools': cap['tools'],
                        'evidence': cap['evidence'],
                        'seeded_at': datetime.now().isoformat()
                    }
                })
                entity_ids.append(entity_id)
                self.logger.debug(f"Created capability: {cap['name']} (ID: {entity_id})")
            except Exception as e:
                self.logger.error(f"Failed to create capability {cap['name']}: {e}")
        
        self.logger.info(f"✅ Seeded {len(entity_ids)} Claude capabilities")
        return entity_ids
    
    def seed_claude_limitations(self) -> List[str]:
        """Seed Claude Code limitations into knowledge graph."""
        self.logger.info("Seeding Claude limitations...")
        
        limitations = [
            {
                'name': 'No Task-based Orchestration',
                'description': 'Cannot launch independent agent processes via Task() tool',
                'evidence': 'No Task() tool found in official documentation'
            },
            {
                'name': 'No Independent Agent Processes',
                'description': 'Agents are contextual specialists, not independent processes',
                'evidence': 'Subagents operate within same Claude session'
            },
            {
                'name': 'No Background Processing',
                'description': 'Cannot run continuous background processes or scheduling',
                'evidence': 'Hooks are event-triggered, not interval-based'
            },
            {
                'name': 'No Persistent Agent State',
                'description': 'Agent state and memory is limited to conversation context',
                'evidence': 'No persistent agent memory beyond conversation'
            },
            {
                'name': 'No Autonomous Agent Operation',
                'description': 'All agent operations happen within Claude Code session',
                'evidence': 'No evidence of autonomous agent capabilities'
            },
            {
                'name': 'Limited Parallel Execution',
                'description': 'Parallel execution limited to subagent delegation within session',
                'evidence': 'No multi-process or truly parallel agent execution'
            }
        ]
        
        entity_ids = []
        for limit in limitations:
            try:
                entity_id = self.db.store_entity({
                    'type': 'claude_limitation',
                    'name': limit['name'],
                    'file_path': '/claude/limitations',
                    'ast_hash': self._create_entity_hash('claude_limitation', limit['name']),
                    'metadata': {
                        'description': limit['description'],
                        'evidence': limit['evidence'],
                        'seeded_at': datetime.now().isoformat()
                    }
                })
                entity_ids.append(entity_id)
                self.logger.debug(f"Created limitation: {limit['name']} (ID: {entity_id})")
            except Exception as e:
                self.logger.error(f"Failed to create limitation {limit['name']}: {e}")
        
        self.logger.info(f"✅ Seeded {len(entity_ids)} Claude limitations")
        return entity_ids
    
    def seed_claude_tools(self) -> List[str]:
        """Seed Claude Code tools into knowledge graph."""
        self.logger.info("Seeding Claude tools...")
        
        tools = [
            {
                'name': 'Read',
                'description': 'Reads file contents from filesystem',
                'category': 'file_operations',
                'supports_capabilities': ['File Operations', 'Code Analysis']
            },
            {
                'name': 'Write',
                'description': 'Writes content to files, creating or overwriting',
                'category': 'file_operations', 
                'supports_capabilities': ['File Operations']
            },
            {
                'name': 'Edit',
                'description': 'Makes targeted edits to existing files',
                'category': 'file_operations',
                'supports_capabilities': ['File Operations']
            },
            {
                'name': 'MultiEdit',
                'description': 'Makes multiple edits to a single file in one operation',
                'category': 'file_operations',
                'supports_capabilities': ['File Operations']
            },
            {
                'name': 'Bash',
                'description': 'Executes shell commands and scripts',
                'category': 'command_execution',
                'supports_capabilities': ['Command Execution', 'Git Integration']
            },
            {
                'name': 'Grep',
                'description': 'Searches for patterns in files using ripgrep',
                'category': 'code_analysis',
                'supports_capabilities': ['Code Analysis']
            },
            {
                'name': 'Glob',
                'description': 'Finds files matching glob patterns',
                'category': 'code_analysis',
                'supports_capabilities': ['Code Analysis']
            },
            {
                'name': 'WebSearch',
                'description': 'Searches the web for information',
                'category': 'web_integration',
                'supports_capabilities': ['Web Search']
            },
            {
                'name': 'WebFetch',
                'description': 'Fetches content from web URLs',
                'category': 'web_integration',
                'supports_capabilities': ['Web Search']
            },
            {
                'name': 'TodoWrite',
                'description': 'Manages task lists and progress tracking',
                'category': 'workflow_management',
                'supports_capabilities': []
            }
        ]
        
        entity_ids = []
        for tool in tools:
            try:
                entity_id = self.db.store_entity({
                    'type': 'claude_tool',
                    'name': tool['name'],
                    'file_path': '/claude/tools',
                    'ast_hash': self._create_entity_hash('claude_tool', tool['name']),
                    'metadata': {
                        'description': tool['description'],
                        'category': tool['category'],
                        'supports_capabilities': tool['supports_capabilities'],
                        'seeded_at': datetime.now().isoformat()
                    }
                })
                entity_ids.append(entity_id)
                self.logger.debug(f"Created tool: {tool['name']} (ID: {entity_id})")
            except Exception as e:
                self.logger.error(f"Failed to create tool {tool['name']}: {e}")
        
        self.logger.info(f"✅ Seeded {len(entity_ids)} Claude tools")
        return entity_ids
    
    def seed_implementation_patterns(self) -> List[str]:
        """Seed correct implementation patterns into knowledge graph."""
        self.logger.info("Seeding implementation patterns...")
        
        patterns = [
            {
                'name': 'MCP-based External Integration',
                'description': 'Use MCP servers to connect with external tools and APIs',
                'context': 'External system integration',
                'example': 'Connect to GitHub via MCP server instead of assuming direct API access'
            },
            {
                'name': 'Subagent Specialization',
                'description': 'Create specialized subagents for domain-specific tasks',
                'context': 'Task delegation and specialization',
                'example': 'Use /agents command to create specialized subagents for different code areas'
            },
            {
                'name': 'Hook-based Automation',
                'description': 'Use .claude/settings.json hooks for event-triggered automation',
                'context': 'Process automation',
                'example': 'Trigger commands on file changes using PostToolUse hooks'
            },
            {
                'name': 'Direct Tool Usage',
                'description': 'Use built-in tools directly for file and command operations',
                'context': 'File and command operations',
                'example': 'Use Read, Write, Edit, Bash tools directly instead of external processes'
            },
            {
                'name': 'Conversation-based Workflow',
                'description': 'Manage workflows through conversation context and subagent delegation',
                'context': 'Workflow management',
                'example': 'Use conversation context for state management instead of persistent processes'
            }
        ]
        
        entity_ids = []
        for pattern in patterns:
            try:
                entity_id = self.db.store_entity({
                    'type': 'implementation_pattern',
                    'name': pattern['name'],
                    'file_path': '/claude/patterns/correct',
                    'ast_hash': self._create_entity_hash('implementation_pattern', pattern['name']),
                    'metadata': {
                        'description': pattern['description'],
                        'context': pattern['context'],
                        'example': pattern['example'],
                        'seeded_at': datetime.now().isoformat()
                    }
                })
                entity_ids.append(entity_id)
                self.logger.debug(f"Created pattern: {pattern['name']} (ID: {entity_id})")
            except Exception as e:
                self.logger.error(f"Failed to create pattern {pattern['name']}: {e}")
        
        self.logger.info(f"✅ Seeded {len(entity_ids)} implementation patterns")
        return entity_ids
    
    def seed_anti_patterns(self) -> List[str]:
        """Seed anti-patterns (incorrect approaches) into knowledge graph."""
        self.logger.info("Seeding anti-patterns...")
        
        anti_patterns = [
            {
                'name': 'Task-based Orchestration Assumption',
                'description': 'Incorrectly assuming Task() tool exists for launching independent agents',
                'why_wrong': 'No Task() tool exists in Claude Code - use subagent delegation instead'
            },
            {
                'name': 'Independent Agent Process Assumption', 
                'description': 'Assuming agents can run as independent processes posting to GitHub',
                'why_wrong': 'All operations happen within Claude Code session - use direct integration instead'
            },
            {
                'name': 'Background Process Assumption',
                'description': 'Assuming continuous background processes or scheduling capabilities',
                'why_wrong': 'Hooks are event-triggered only - use event-based patterns instead'
            },
            {
                'name': 'Complex Automation Assumptions',
                'description': 'Over-engineering automation beyond Claude Code capabilities',
                'why_wrong': 'Keep automation simple and event-based using hooks and MCP'
            },
            {
                'name': 'Persistent Agent State Assumption',
                'description': 'Assuming agents have persistent memory beyond conversation context',
                'why_wrong': 'Use conversation context and external storage via MCP instead'
            }
        ]
        
        entity_ids = []
        for anti_pattern in anti_patterns:
            try:
                entity_id = self.db.store_entity({
                    'type': 'anti_pattern',
                    'name': anti_pattern['name'],
                    'file_path': '/claude/patterns/incorrect',
                    'ast_hash': self._create_entity_hash('anti_pattern', anti_pattern['name']),
                    'metadata': {
                        'description': anti_pattern['description'],
                        'why_wrong': anti_pattern['why_wrong'],
                        'seeded_at': datetime.now().isoformat()
                    }
                })
                entity_ids.append(entity_id)
                self.logger.debug(f"Created anti-pattern: {anti_pattern['name']} (ID: {entity_id})")
            except Exception as e:
                self.logger.error(f"Failed to create anti-pattern {anti_pattern['name']}: {e}")
        
        self.logger.info(f"✅ Seeded {len(entity_ids)} anti-patterns")
        return entity_ids
    
    def create_relationships(self) -> int:
        """Create relationships between capabilities, tools, patterns, etc."""
        self.logger.info("Creating relationships between Claude knowledge entities...")
        
        relationship_count = 0
        
        try:
            # Get all entities by type
            capabilities = self.db.search_entities(entity_types=['claude_capability'])
            tools = self.db.search_entities(entity_types=['claude_tool'])  
            patterns = self.db.search_entities(entity_types=['implementation_pattern'])
            anti_patterns = self.db.search_entities(entity_types=['anti_pattern'])
            limitations = self.db.search_entities(entity_types=['claude_limitation'])
            
            # Create tool -> capability "supports" relationships
            for tool in tools:
                tool_metadata = tool.get('metadata', {})
                supported_caps = tool_metadata.get('supports_capabilities', [])
                
                for cap_name in supported_caps:
                    # Find matching capability
                    matching_caps = [c for c in capabilities if c['name'] == cap_name]
                    if matching_caps:
                        rel_id = self.db.store_relationship(
                            tool['id'], matching_caps[0]['id'], 'supports', 0.9,
                            {'description': f"{tool['name']} tool supports {cap_name} capability"}
                        )
                        relationship_count += 1
                        self.logger.debug(f"Created supports relationship: {tool['name']} -> {cap_name}")
            
            # Create pattern -> anti-pattern "alternative_to" relationships
            pattern_alternatives = {
                'MCP-based External Integration': 'Task-based Orchestration Assumption',
                'Subagent Specialization': 'Independent Agent Process Assumption',
                'Hook-based Automation': 'Background Process Assumption',
                'Conversation-based Workflow': 'Persistent Agent State Assumption'
            }
            
            for pattern_name, anti_pattern_name in pattern_alternatives.items():
                pattern_entities = [p for p in patterns if p['name'] == pattern_name]
                anti_pattern_entities = [ap for ap in anti_patterns if ap['name'] == anti_pattern_name]
                
                if pattern_entities and anti_pattern_entities:
                    rel_id = self.db.store_relationship(
                        pattern_entities[0]['id'], anti_pattern_entities[0]['id'], 
                        'alternative_to', 1.0,
                        {'description': f"{pattern_name} is the correct alternative to {anti_pattern_name}"}
                    )
                    relationship_count += 1
                    self.logger.debug(f"Created alternative_to relationship: {pattern_name} -> {anti_pattern_name}")
            
            # Create limitation -> anti-pattern "conflicts_with" relationships
            limitation_conflicts = {
                'No Task-based Orchestration': 'Task-based Orchestration Assumption',
                'No Independent Agent Processes': 'Independent Agent Process Assumption', 
                'No Background Processing': 'Background Process Assumption',
                'No Persistent Agent State': 'Persistent Agent State Assumption'
            }
            
            for limitation_name, anti_pattern_name in limitation_conflicts.items():
                limitation_entities = [l for l in limitations if l['name'] == limitation_name]
                anti_pattern_entities = [ap for ap in anti_patterns if ap['name'] == anti_pattern_name]
                
                if limitation_entities and anti_pattern_entities:
                    rel_id = self.db.store_relationship(
                        limitation_entities[0]['id'], anti_pattern_entities[0]['id'],
                        'conflicts_with', 1.0,
                        {'description': f"{limitation_name} conflicts with {anti_pattern_name}"}
                    )
                    relationship_count += 1
                    self.logger.debug(f"Created conflicts_with relationship: {limitation_name} -> {anti_pattern_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create relationships: {e}")
        
        self.logger.info(f"✅ Created {relationship_count} relationships")
        return relationship_count
    
    def seed_all(self) -> Dict[str, Any]:
        """Seed all Claude knowledge into the knowledge graph."""
        results = {
            'start_time': datetime.now().isoformat(),
            'capabilities': [],
            'limitations': [],
            'tools': [],
            'patterns': [],
            'anti_patterns': [],
            'relationships_count': 0,
            'success': False,
            'errors': []
        }
        
        try:
            self.logger.info("Starting Claude knowledge seeding process...")
            
            # Seed all entity types
            results['capabilities'] = self.seed_claude_capabilities()
            results['limitations'] = self.seed_claude_limitations() 
            results['tools'] = self.seed_claude_tools()
            results['patterns'] = self.seed_implementation_patterns()
            results['anti_patterns'] = self.seed_anti_patterns()
            
            # Create relationships
            results['relationships_count'] = self.create_relationships()
            
            # Summary
            total_entities = (
                len(results['capabilities']) + len(results['limitations']) + 
                len(results['tools']) + len(results['patterns']) + 
                len(results['anti_patterns'])
            )
            
            results['success'] = True
            results['end_time'] = datetime.now().isoformat()
            results['total_entities'] = total_entities
            
            self.logger.info(f"🎉 Claude knowledge seeding completed successfully!")
            self.logger.info(f"   Seeded {total_entities} entities and {results['relationships_count']} relationships")
            
        except Exception as e:
            results['errors'].append(f"Seeding process failed: {e}")
            self.logger.error(f"❌ Claude knowledge seeding failed: {e}")
        
        return results


def main():
    """Main entry point for seeder script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Knowledge Seeder Script")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        with ClaudeKnowledgeSeeder() as seeder:
            logger.info("Running Claude knowledge seeding...")
            results = seeder.seed_all()
            
            # Print results
            print(json.dumps(results, indent=2, default=str))
            
            sys.exit(0 if results['success'] else 1)
                
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()