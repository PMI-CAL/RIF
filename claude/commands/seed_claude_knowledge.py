"""
Seed Claude Code Knowledge into RIF Knowledge Graph

This script demonstrates how Claude Code capability knowledge is loaded into the 
existing RIF knowledge graph system, extending it with new entity types and 
relationships specific to Claude Code capabilities, limitations, and patterns.

Usage:
    python seed_claude_knowledge.py --source /knowledge/claude-code-capabilities.md
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add RIF knowledge system to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from knowledge.database.database_interface import RIFDatabase
from knowledge.embeddings.embedding_generator import EmbeddingGenerator


class ClaudeKnowledgeSeeder:
    """
    Seeds Claude Code knowledge into the existing RIF knowledge graph.
    
    Demonstrates integration with existing infrastructure:
    - Uses RIFDatabase for entity and relationship storage
    - Uses EmbeddingGenerator for vector embeddings
    - Extends existing schema with new entity types
    - Creates relationships between capabilities and patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rif_db = RIFDatabase()
        self.embedding_gen = EmbeddingGenerator()
        
        # Sample Claude Code knowledge data
        self.knowledge_data = self._load_claude_knowledge()
        
    def _load_claude_knowledge(self) -> Dict[str, List[Dict]]:
        """
        Load Claude Code knowledge from various sources.
        
        In production, this would:
        1. Parse /knowledge/claude-code-capabilities.md
        2. Extract from official documentation
        3. Load from curated knowledge files
        4. Validate against known working examples
        """
        return {
            "capabilities": [
                {
                    "name": "file_operations_read",
                    "type": "claude_capability", 
                    "category": "core_capabilities",
                    "description": "Read files from filesystem with configurable limits",
                    "parameters": {
                        "file_path": "Absolute path to file (required)",
                        "limit": "Number of lines to read (optional, default 2000)",
                        "offset": "Line number to start from (optional)"
                    },
                    "limitations": [
                        "Absolute paths required, not relative",
                        "2000 lines default limit",
                        "Lines > 2000 characters truncated"
                    ],
                    "version_support": "all",
                    "confidence": 1.0,
                    "examples": [
                        "Read('/Users/project/file.py')",
                        "Read('/path/to/file.js', limit=500, offset=100)"
                    ]
                },
                {
                    "name": "file_operations_edit",
                    "type": "claude_capability",
                    "category": "core_capabilities", 
                    "description": "Exact string replacement in files",
                    "parameters": {
                        "file_path": "Absolute path to file",
                        "old_string": "Text to replace (must match exactly)",
                        "new_string": "Replacement text",
                        "replace_all": "Replace all occurrences (optional, default false)"
                    },
                    "limitations": [
                        "Must use Read tool first before editing",
                        "old_string must be unique unless replace_all=true",
                        "Exact whitespace/indentation matching required"
                    ],
                    "version_support": "all",
                    "confidence": 1.0
                },
                {
                    "name": "subagent_delegation",
                    "type": "claude_capability",
                    "category": "agent_system",
                    "description": "Delegate tasks to specialized subagents within same session",
                    "parameters": {
                        "subagent_type": "Type of specialist subagent",
                        "context": "Task context and requirements",
                        "tools": "Tool restrictions for subagent"
                    },
                    "limitations": [
                        "Subagents operate in separate context windows",
                        "No persistent memory between sessions",
                        "Cannot run truly independently or in parallel"
                    ],
                    "version_support": "all",
                    "confidence": 1.0
                },
                {
                    "name": "mcp_server_integration", 
                    "type": "claude_capability",
                    "category": "external_integration",
                    "description": "Connect to external tools and services via MCP protocol",
                    "parameters": {
                        "server_type": "stdio, sse, or http",
                        "configuration": "Server-specific config",
                        "authentication": "OAuth2 or other auth methods"
                    },
                    "limitations": [
                        "Must be configured in .claude/settings.json",
                        "No runtime MCP server discovery",
                        "Limited to configured servers"
                    ],
                    "version_support": "all", 
                    "confidence": 1.0
                }
            ],
            
            "limitations": [
                {
                    "name": "no_task_parallel_execution",
                    "type": "claude_limitation",
                    "category": "orchestration",
                    "description": "No built-in Task() tool for parallel agent execution",
                    "impact": "Cannot launch multiple independent agents simultaneously",
                    "severity": "high",
                    "workarounds": [
                        "Use subagent delegation within same session",
                        "Sequential task execution with context passing",
                        "MCP server coordination for external parallelism"
                    ],
                    "version_support": "all",
                    "confidence": 1.0
                },
                {
                    "name": "no_persistent_background_processes",
                    "type": "claude_limitation", 
                    "category": "process_management",
                    "description": "Cannot maintain persistent background processes or daemons",
                    "impact": "No continuous monitoring or scheduled tasks",
                    "severity": "high",
                    "workarounds": [
                        "Use hooks for event-triggered automation",
                        "External cron jobs or schedulers",
                        "Session-based workflow execution"
                    ],
                    "version_support": "all",
                    "confidence": 1.0
                },
                {
                    "name": "session_scope_state_limit",
                    "type": "claude_limitation",
                    "category": "state_management", 
                    "description": "State and memory limited to current conversation session",
                    "impact": "No persistent agent memory between sessions",
                    "severity": "medium",
                    "workarounds": [
                        "Use external storage (files, databases)",
                        "MCP servers for persistent state",
                        "Context reconstruction from stored data"
                    ],
                    "version_support": "all",
                    "confidence": 1.0
                }
            ],
            
            "implementation_patterns": [
                {
                    "name": "mcp_github_integration",
                    "type": "implementation_pattern",
                    "category": "github_integration",
                    "technology": "github",
                    "task_type": "issue_management",
                    "description": "Integrate with GitHub via MCP server for issue and PR operations",
                    "code_example": '''
# .claude/settings.json configuration
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}

# Usage in Claude Code
# GitHub tools become available automatically
# Can create issues, manage PRs, query repositories
''',
                    "best_practices": [
                        "Use environment variables for tokens",
                        "Configure appropriate scopes for PAT",
                        "Handle rate limiting gracefully",
                        "Validate permissions before operations"
                    ],
                    "anti_patterns": [
                        "Direct GitHub API calls without MCP",
                        "Hardcoded tokens in configuration",
                        "Assuming unlimited rate limits"
                    ],
                    "confidence": 0.9
                },
                
                {
                    "name": "subagent_specialization",
                    "type": "implementation_pattern",
                    "category": "agent_architecture",
                    "technology": "general",
                    "task_type": "task_delegation",
                    "description": "Create specialized subagents for different domains within Claude Code",
                    "code_example": '''
# Create specialized subagents via /agents command
# Example: Backend Specialist
{
  "name": "Backend Specialist",
  "description": "Expert in backend development, APIs, and database design",
  "system_prompt": "You are a backend development expert. Focus on server-side logic, API design, database optimization, and system architecture.",
  "tool_restrictions": ["filesystem", "database", "testing"],
  "scope": "project"
}

# Delegate tasks to appropriate specialist
# Claude automatically routes backend questions to Backend Specialist
# Maintains context and expertise focus
''',
                    "best_practices": [
                        "Define clear specialization boundaries",
                        "Use descriptive system prompts", 
                        "Restrict tools to relevant domains",
                        "Create project-scoped for team consistency"
                    ],
                    "anti_patterns": [
                        "Overly broad subagent definitions",
                        "Too many overlapping specialists", 
                        "Generic system prompts"
                    ],
                    "confidence": 0.95
                },
                
                {
                    "name": "hook_based_automation",
                    "type": "implementation_pattern",
                    "category": "automation",
                    "technology": "general", 
                    "task_type": "workflow_automation",
                    "description": "Use Claude Code hooks for event-driven automation",
                    "code_example": '''
# .claude/settings.json hooks configuration
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'File modified' >> ./changes.log"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": ".*test.*",
        "hooks": [
          {
            "type": "command", 
            "command": "npm test",
            "output": "context"
          }
        ]
      }
    ]
  }
}
''',
                    "best_practices": [
                        "Use specific regex patterns for matching",
                        "Keep hook commands lightweight",
                        "Use 'output': 'context' for relevant information",
                        "Test hooks thoroughly before deployment"
                    ],
                    "anti_patterns": [
                        "Long-running commands in hooks",
                        "Overly broad pattern matching",
                        "Hooks that modify critical files"
                    ],
                    "confidence": 0.85
                }
            ],
            
            "anti_patterns": [
                {
                    "name": "task_orchestration_assumption",
                    "type": "anti_pattern",
                    "category": "orchestration",
                    "description": "Assuming Task() tool exists for parallel agent orchestration",
                    "why_wrong": "Claude Code has no built-in Task() tool or parallel agent execution",
                    "common_mistakes": [
                        "Task(description='...', subagent_type='...')",
                        "Expecting multiple agents to run simultaneously",
                        "Complex orchestration workflows"
                    ],
                    "correct_approach": "Use subagent delegation within single session",
                    "confidence": 1.0
                },
                {
                    "name": "independent_agent_processes",
                    "type": "anti_pattern", 
                    "category": "agent_architecture",
                    "description": "Treating subagents as independent processes with persistent state",
                    "why_wrong": "Subagents are contextual specialists within Claude Code, not separate processes",
                    "common_mistakes": [
                        "Expecting agents to run autonomously",
                        "Assuming persistent agent memory",
                        "Cross-agent communication patterns"
                    ],
                    "correct_approach": "Use subagents for specialization, external storage for persistence",
                    "confidence": 1.0
                },
                {
                    "name": "background_process_monitoring",
                    "type": "anti_pattern",
                    "category": "monitoring",
                    "description": "Expecting Claude Code to maintain background monitoring processes",
                    "why_wrong": "Claude Code operates in session-based model, no background processes",
                    "common_mistakes": [
                        "Continuous file monitoring",
                        "Background health checks",
                        "Scheduled maintenance tasks"
                    ],
                    "correct_approach": "Use external tools, cron jobs, or hook-based triggers",
                    "confidence": 1.0
                }
            ]
        }
    
    def seed_knowledge(self) -> Dict[str, Any]:
        """
        Seed Claude Code knowledge into the existing RIF knowledge graph.
        
        Process:
        1. Create entities for capabilities, limitations, patterns, anti-patterns
        2. Generate embeddings for semantic search
        3. Create relationships between entities
        4. Validate data integrity
        5. Update knowledge graph statistics
        """
        results = {
            'entities_created': 0,
            'relationships_created': 0, 
            'embeddings_generated': 0,
            'errors': []
        }
        
        try:
            self.logger.info("Starting Claude Code knowledge seeding...")
            
            # Seed each knowledge category
            entity_ids = {}
            
            # 1. Seed capabilities
            capability_ids = self._seed_capabilities()
            entity_ids.update(capability_ids)
            results['entities_created'] += len(capability_ids)
            
            # 2. Seed limitations
            limitation_ids = self._seed_limitations()
            entity_ids.update(limitation_ids)
            results['entities_created'] += len(limitation_ids)
            
            # 3. Seed implementation patterns
            pattern_ids = self._seed_implementation_patterns()
            entity_ids.update(pattern_ids)
            results['entities_created'] += len(pattern_ids)
            
            # 4. Seed anti-patterns
            anti_pattern_ids = self._seed_anti_patterns()
            entity_ids.update(anti_pattern_ids)
            results['entities_created'] += len(anti_pattern_ids)
            
            # 5. Create relationships between entities
            relationship_count = self._create_relationships(entity_ids)
            results['relationships_created'] = relationship_count
            
            # 6. Generate embeddings for semantic search
            embedding_count = self._generate_embeddings(entity_ids)
            results['embeddings_generated'] = embedding_count
            
            # 7. Validate seeded data
            validation_results = self._validate_seeded_data()
            if not validation_results['valid']:
                results['errors'].extend(validation_results['errors'])
            
            self.logger.info(f"Knowledge seeding completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Knowledge seeding failed: {e}")
            results['errors'].append(str(e))
            return results
    
    def _seed_capabilities(self) -> Dict[str, str]:
        """Seed Claude Code capabilities as entities."""
        capability_ids = {}
        
        for capability in self.knowledge_data['capabilities']:
            try:
                entity_data = {
                    'type': capability['type'],
                    'name': capability['name'], 
                    'file_path': 'claude-code-capabilities',  # Logical grouping
                    'metadata': {
                        'category': capability['category'],
                        'description': capability['description'],
                        'parameters': capability.get('parameters', {}),
                        'limitations': capability.get('limitations', []),
                        'version_support': capability['version_support'],
                        'confidence': capability['confidence'],
                        'examples': capability.get('examples', [])
                    }
                }
                
                entity_id = self.rif_db.store_entity(entity_data)
                capability_ids[capability['name']] = entity_id
                
                self.logger.debug(f"Created capability entity: {capability['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create capability {capability['name']}: {e}")
        
        return capability_ids
    
    def _seed_limitations(self) -> Dict[str, str]:
        """Seed Claude Code limitations as entities."""
        limitation_ids = {}
        
        for limitation in self.knowledge_data['limitations']:
            try:
                entity_data = {
                    'type': limitation['type'],
                    'name': limitation['name'],
                    'file_path': 'claude-code-limitations',
                    'metadata': {
                        'category': limitation['category'],
                        'description': limitation['description'], 
                        'impact': limitation['impact'],
                        'severity': limitation['severity'],
                        'workarounds': limitation.get('workarounds', []),
                        'version_support': limitation['version_support'],
                        'confidence': limitation['confidence']
                    }
                }
                
                entity_id = self.rif_db.store_entity(entity_data)
                limitation_ids[limitation['name']] = entity_id
                
                self.logger.debug(f"Created limitation entity: {limitation['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create limitation {limitation['name']}: {e}")
        
        return limitation_ids
    
    def _seed_implementation_patterns(self) -> Dict[str, str]:
        """Seed implementation patterns as entities."""
        pattern_ids = {}
        
        for pattern in self.knowledge_data['implementation_patterns']:
            try:
                entity_data = {
                    'type': pattern['type'],
                    'name': pattern['name'],
                    'file_path': 'claude-code-patterns',
                    'metadata': {
                        'category': pattern['category'],
                        'technology': pattern['technology'],
                        'task_type': pattern['task_type'],
                        'description': pattern['description'],
                        'code_example': pattern['code_example'],
                        'best_practices': pattern.get('best_practices', []),
                        'anti_patterns': pattern.get('anti_patterns', []),
                        'confidence': pattern['confidence']
                    }
                }
                
                entity_id = self.rif_db.store_entity(entity_data)
                pattern_ids[pattern['name']] = entity_id
                
                self.logger.debug(f"Created pattern entity: {pattern['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create pattern {pattern['name']}: {e}")
        
        return pattern_ids
    
    def _seed_anti_patterns(self) -> Dict[str, str]:
        """Seed anti-patterns as entities."""
        anti_pattern_ids = {}
        
        for anti_pattern in self.knowledge_data['anti_patterns']:
            try:
                entity_data = {
                    'type': anti_pattern['type'],
                    'name': anti_pattern['name'],
                    'file_path': 'claude-code-anti-patterns',
                    'metadata': {
                        'category': anti_pattern['category'],
                        'description': anti_pattern['description'],
                        'why_wrong': anti_pattern['why_wrong'],
                        'common_mistakes': anti_pattern.get('common_mistakes', []),
                        'correct_approach': anti_pattern['correct_approach'],
                        'confidence': anti_pattern['confidence']
                    }
                }
                
                entity_id = self.rif_db.store_entity(entity_data)
                anti_pattern_ids[anti_pattern['name']] = entity_id
                
                self.logger.debug(f"Created anti-pattern entity: {anti_pattern['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create anti-pattern {anti_pattern['name']}: {e}")
        
        return anti_pattern_ids
    
    def _create_relationships(self, entity_ids: Dict[str, str]) -> int:
        """Create relationships between Claude Code knowledge entities."""
        relationship_count = 0
        
        # Define relationships based on knowledge structure
        relationships = [
            # Capabilities support patterns
            ('file_operations_read', 'supports', 'mcp_github_integration', 0.8),
            ('file_operations_edit', 'supports', 'mcp_github_integration', 0.9),
            ('subagent_delegation', 'supports', 'subagent_specialization', 1.0),
            ('mcp_server_integration', 'supports', 'mcp_github_integration', 1.0),
            
            # Limitations conflict with patterns
            ('no_task_parallel_execution', 'conflicts_with', 'task_orchestration_assumption', 1.0),
            ('no_persistent_background_processes', 'conflicts_with', 'background_process_monitoring', 1.0),
            ('session_scope_state_limit', 'conflicts_with', 'independent_agent_processes', 0.9),
            
            # Patterns are alternatives to anti-patterns
            ('subagent_specialization', 'alternative_to', 'independent_agent_processes', 0.9),
            ('hook_based_automation', 'alternative_to', 'background_process_monitoring', 0.8),
            ('mcp_github_integration', 'alternative_to', 'task_orchestration_assumption', 0.7),
        ]
        
        for source_name, relation_type, target_name, confidence in relationships:
            try:
                source_id = entity_ids.get(source_name)
                target_id = entity_ids.get(target_name)
                
                if source_id and target_id:
                    self.rif_db.store_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=relation_type,
                        confidence=confidence,
                        metadata={'created_by': 'claude_knowledge_seeder'}
                    )
                    relationship_count += 1
                    
                    self.logger.debug(f"Created relationship: {source_name} {relation_type} {target_name}")
                else:
                    self.logger.warning(f"Missing entity IDs for relationship: {source_name} -> {target_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to create relationship {source_name} -> {target_name}: {e}")
        
        return relationship_count
    
    def _generate_embeddings(self, entity_ids: Dict[str, str]) -> int:
        """Generate vector embeddings for semantic search."""
        embedding_count = 0
        
        for entity_name, entity_id in entity_ids.items():
            try:
                # Get entity data
                entity = self.rif_db.get_entity(entity_id)
                if not entity:
                    continue
                
                # Create text for embedding generation
                text_content = f"{entity['name']} {entity['metadata'].get('description', '')}"
                
                # Generate embedding
                embedding = self.embedding_gen.generate_embedding(text_content)
                
                # Update entity with embedding
                self.rif_db.update_entity_embedding(
                    entity_id=entity_id,
                    embedding=embedding,
                    metadata=entity['metadata']
                )
                
                embedding_count += 1
                self.logger.debug(f"Generated embedding for: {entity_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for {entity_name}: {e}")
        
        return embedding_count
    
    def _validate_seeded_data(self) -> Dict[str, Any]:
        """Validate the seeded Claude Code knowledge data."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check entity counts
            capabilities = self.rif_db.search_entities(entity_types=['claude_capability'])
            limitations = self.rif_db.search_entities(entity_types=['claude_limitation'])
            patterns = self.rif_db.search_entities(entity_types=['implementation_pattern'])
            anti_patterns = self.rif_db.search_entities(entity_types=['anti_pattern'])
            
            expected_counts = {
                'capabilities': len(self.knowledge_data['capabilities']),
                'limitations': len(self.knowledge_data['limitations']),
                'patterns': len(self.knowledge_data['implementation_patterns']),
                'anti_patterns': len(self.knowledge_data['anti_patterns'])
            }
            
            actual_counts = {
                'capabilities': len(capabilities),
                'limitations': len(limitations), 
                'patterns': len(patterns),
                'anti_patterns': len(anti_patterns)
            }
            
            # Compare counts
            for category, expected in expected_counts.items():
                actual = actual_counts[category]
                if actual != expected:
                    validation_results['errors'].append(
                        f"Entity count mismatch for {category}: expected {expected}, got {actual}"
                    )
                    validation_results['valid'] = False
            
            # Check for embeddings
            entities_with_embeddings = 0
            for entity_type in ['claude_capability', 'claude_limitation', 'implementation_pattern', 'anti_pattern']:
                entities = self.rif_db.search_entities(entity_types=[entity_type])
                for entity in entities:
                    full_entity = self.rif_db.get_entity(entity['id'])
                    if full_entity and full_entity.get('embedding'):
                        entities_with_embeddings += 1
            
            total_entities = sum(actual_counts.values())
            if entities_with_embeddings < total_entities:
                validation_results['warnings'].append(
                    f"Only {entities_with_embeddings}/{total_entities} entities have embeddings"
                )
            
            self.logger.info(f"Data validation completed: {validation_results}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['valid'] = False
        
        return validation_results


def main():
    """Main entry point for knowledge seeding."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed Claude Code Knowledge into RIF Knowledge Graph')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create seeder
    seeder = ClaudeKnowledgeSeeder()
    
    if args.validate_only:
        # Only validate existing data
        validation_results = seeder._validate_seeded_data()
        if validation_results['valid']:
            print("✅ Existing Claude Code knowledge data is valid")
        else:
            print("❌ Validation errors found:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        return
    
    # Seed knowledge
    results = seeder.seed_knowledge()
    
    print("\n🌱 Claude Code Knowledge Seeding Results:")
    print(f"  Entities created: {results['entities_created']}")
    print(f"  Relationships created: {results['relationships_created']}")
    print(f"  Embeddings generated: {results['embeddings_generated']}")
    
    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    else:
        print("\n✅ Knowledge seeding completed successfully!")
        print("\nNext steps:")
        print("  1. Start the Claude Code Knowledge MCP Server")
        print("  2. Test compatibility checking tools")
        print("  3. Validate pattern recommendations")


if __name__ == '__main__':
    main()