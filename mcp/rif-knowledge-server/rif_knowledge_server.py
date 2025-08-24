#!/usr/bin/env python3
"""
RIF Knowledge MCP Server - Universal Knowledge Gateway
=======================================================
The primary interface for ALL knowledge graph operations in the RIF system.

Phase 1: Claude Code documentation and capabilities
Phase 2: All patterns, decisions, and resolutions
Phase 3: Vector search and relationship traversal
Phase 4: Knowledge updates and management
"""

import json
import sys
import os
import logging
import duckdb
import chromadb
import hashlib
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/rif_knowledge_server.log'),
        logging.StreamHandler(sys.stderr) if os.getenv('DEBUG') else logging.NullHandler()
    ]
)
logger = logging.getLogger('RIFKnowledgeServer')


class RIFKnowledgeServer:
    """
    Universal MCP server for RIF knowledge graph access.
    Provides comprehensive query, search, and management capabilities.
    """
    
    def __init__(self):
        """Initialize connections to all knowledge stores"""
        self.base_path = Path(__file__).parent.parent.parent
        self.knowledge_path = self.base_path / "knowledge"
        
        # Initialize database connections
        self.duckdb_path = self.knowledge_path / "hybrid_knowledge.duckdb"
        self.chromadb_path = self.knowledge_path / "chromadb"
        
        self.duckdb_conn = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # Knowledge categories for better organization
        self.knowledge_categories = {
            'claude': ['Claude Code', 'claude', 'assistant', 'MCP', 'tools'],
            'patterns': ['pattern', 'anti-pattern', 'implementation'],
            'architecture': ['decision', 'design', 'architecture'],
            'issues': ['issue', 'resolution', 'fix', 'bug'],
            'agents': ['agent', 'RIF-', 'orchestrat', 'delegat'],
            'testing': ['test', 'validation', 'quality'],
            'monitoring': ['monitor', 'metric', 'performance']
        }
        
        # Initialize performance caching
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        
        self._connect_databases()
    
    def _connect_databases(self):
        """Connect to DuckDB and ChromaDB"""
        try:
            # Connect to DuckDB (read-only to avoid lock conflicts)
            self.duckdb_conn = duckdb.connect(str(self.duckdb_path), read_only=True)
            logger.info(f"Connected to DuckDB at {self.duckdb_path}")
            
            # Get database stats
            stats = self.duckdb_conn.execute("""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(DISTINCT type) as entity_types
                FROM entities
            """).fetchone()
            logger.info(f"Database contains {stats[0]} entities of {stats[1]} types")
            
            # Connect to ChromaDB for vector search
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(self.chromadb_path))
                self.chroma_collection = self.chroma_client.get_or_create_collection("rif_knowledge")
                logger.info(f"Connected to ChromaDB at {self.chromadb_path}")
            except Exception as e:
                logger.warning(f"ChromaDB connection failed (vector search disabled): {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            raise
    
    # =========================================================================
    # PHASE 1: CLAUDE CODE DOCUMENTATION
    # =========================================================================
    
    def get_claude_documentation(self, topic: str) -> Dict[str, Any]:
        """
        Retrieve Claude Code specific documentation and capabilities.
        This is our PRIMARY goal - make Claude Code knowledge accessible.
        """
        logger.info(f"Getting Claude documentation for: {topic}")
        
        # First, check for stored Claude research
        research_path = self.knowledge_path / "research" / "claude-code-research.json"
        claude_knowledge = {}
        
        if research_path.exists():
            with open(research_path, 'r') as f:
                claude_research = json.load(f)
                claude_knowledge['research'] = claude_research
        
        # Query database for Claude-specific patterns
        claude_patterns = self.duckdb_conn.execute("""
            SELECT 
                type, name, file_path, metadata
            FROM entities
            WHERE 
                LOWER(name) LIKE '%claude%' OR
                LOWER(CAST(metadata AS VARCHAR)) LIKE '%claude%' OR
                LOWER(name) LIKE '%mcp%' OR
                type IN ('pattern', 'decision', 'issue_resolution')
            ORDER BY 
                CASE 
                    WHEN LOWER(name) LIKE '%claude%' THEN 0
                    WHEN type = 'pattern' THEN 1
                    WHEN type = 'decision' THEN 2
                    ELSE 3
                END
            LIMIT 20
        """).fetchall()
        
        # Format results
        patterns = []
        decisions = []
        issues = []
        
        for row in claude_patterns:
            entry = {
                'type': row[0],
                'name': row[1],
                'source': row[2],
                'metadata': json.loads(row[3]) if row[3] else {}
            }
            
            if row[0] == 'pattern':
                patterns.append(entry)
            elif row[0] == 'decision':
                decisions.append(entry)
            elif row[0] == 'issue_resolution':
                issues.append(entry)
        
        claude_knowledge['patterns'] = patterns
        claude_knowledge['decisions'] = decisions
        claude_knowledge['issues'] = issues
        
        # Check for specific Claude capabilities (expanded matching)
        capability_keywords = ['capability', 'can', 'file', 'bash', 'command', 'web', 'search', 'task', 'tools', 'grep', 'glob', 'read', 'write', 'edit', 'mcp', 'subagent', 'hook']
        if any(keyword in topic.lower() for keyword in capability_keywords):
            claude_knowledge['capabilities'] = self._get_claude_capabilities()
        
        # Check for limitations (expanded matching)
        limitation_keywords = ['limit', 'cannot', 'persistent', 'background', 'task.parallel', 'agent communication', 'session']
        if any(keyword in topic.lower() for keyword in limitation_keywords):
            claude_knowledge['limitations'] = self._get_claude_limitations()
        
        return claude_knowledge
    
    def _get_claude_capabilities(self) -> List[Dict[str, str]]:
        """Get specific Claude Code capabilities from stored knowledge"""
        capabilities = [
            {
                'category': 'File Operations',
                'tools': ['Read', 'Write', 'Edit', 'MultiEdit'],
                'description': 'Full file system access for reading and modifying code'
            },
            {
                'category': 'Command Execution',
                'tools': ['Bash'],
                'description': 'Execute shell commands, including git and other CLI tools'
            },
            {
                'category': 'Web Access',
                'tools': ['WebSearch', 'WebFetch'],
                'description': 'Search the web and fetch content from URLs'
            },
            {
                'category': 'Task Delegation',
                'tools': ['Task'],
                'description': 'Launch specialized subagents for complex tasks'
            },
            {
                'category': 'Code Analysis',
                'tools': ['Grep', 'Glob'],
                'description': 'Search codebases with powerful pattern matching'
            },
            {
                'category': 'Extension',
                'tools': ['MCP Servers'],
                'description': 'Connect to external tools via Model Context Protocol'
            },
            {
                'category': 'Automation',
                'tools': ['Hooks'],
                'description': 'Event-triggered automation scripts on tool events, not continuous background processes'
            },
            {
                'category': 'Subagent Delegation',
                'tools': ['Task with specialized prompts'],
                'description': 'Contextual specialist subagents within same session, not independent processes'
            },
            {
                'category': 'Agent Coordination',
                'tools': ['Files', 'GitHub Issues'],
                'description': 'Agents coordinate through files and GitHub issues, no direct communication or shared memory'
            }
        ]
        return capabilities
    
    def _get_claude_limitations(self) -> List[Dict[str, str]]:
        """Get known Claude Code limitations from stored knowledge"""
        limitations = [
            {
                'limitation': 'No persistent background processes',
                'workaround': 'Use Bash with run_in_background for session-scoped tasks'
            },
            {
                'limitation': 'Task.parallel() is pseudocode',
                'workaround': 'Launch multiple Task tools in one response for parallel execution'
            },
            {
                'limitation': 'No direct agent communication',
                'workaround': 'Agents coordinate through files or GitHub issues'
            },
            {
                'limitation': 'Session-scoped memory',
                'workaround': 'Explicitly save state to files for persistence'
            },
            {
                'limitation': 'Cannot be externally orchestrated',
                'workaround': 'Claude Code IS the orchestrator, not orchestrated by external systems'
            }
        ]
        return limitations
    
    # =========================================================================
    # PHASE 2: GENERAL KNOWLEDGE QUERIES
    # =========================================================================
    
    def query_knowledge(
        self, 
        query: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
        use_vector_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        General knowledge query interface - the main gateway for all knowledge.
        """
        logger.info(f"Querying knowledge: {query} (types: {entity_types}, vector: {use_vector_search})")
        
        # Check cache first
        cache_key = self._get_cache_key(query, entity_types, limit, use_vector_search)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {query}")
            return cached_result
        
        # Execute search
        if use_vector_search and self.chroma_collection:
            # Use vector similarity search
            results = self._vector_search(query, entity_types, limit)
        else:
            # Use SQL text search
            results = self._text_search(query, entity_types, limit)
        
        # Cache the results
        self._put_in_cache(cache_key, results)
        
        return results
    
    def _text_search(
        self, 
        query: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Enhanced SQL-based text search with multiple strategies"""
        
        # Strategy 1: Try exact phrase first
        results = self._search_with_query(query, entity_types, limit)
        
        # Strategy 2: If no results, try individual words
        if not results and ' ' in query:
            words = query.split()
            for word in words:
                if len(word) > 2:  # Skip short words
                    word_results = self._search_with_query(word, entity_types, limit)
                    results.extend(word_results)
                    if len(results) >= limit:
                        break
        
        # Strategy 3: If still no results, try stemmed/root words
        if not results:
            # Simple stemming - remove common suffixes
            query_stem = query
            for suffix in ['ing', 'ed', 'er', 'est', 's']:
                if query.endswith(suffix) and len(query) > len(suffix) + 2:
                    query_stem = query[:-len(suffix)]
                    stem_results = self._search_with_query(query_stem, entity_types, limit)
                    if stem_results:
                        results.extend(stem_results)
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in results[:limit]:
            result_id = result['id']
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)
        
        return unique_results[:limit]
    
    def _search_with_query(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute single search query"""
        # Build query
        type_filter = ""
        params = [f"%{query}%", f"%{query}%"]
        
        if entity_types:
            placeholders = ','.join(['?' for _ in entity_types])
            type_filter = f"AND type IN ({placeholders})"
            params.extend(entity_types)
        
        params.append(limit)
        
        sql = f"""
            SELECT 
                id,
                type,
                name,
                file_path,
                line_start,
                line_end,
                metadata,
                created_at
            FROM entities
            WHERE (
                LOWER(name) LIKE LOWER(?) OR 
                LOWER(CAST(metadata AS VARCHAR)) LIKE LOWER(?)
            )
            {type_filter}
            ORDER BY 
                CASE 
                    WHEN LOWER(name) LIKE LOWER(?) THEN 0
                    WHEN type = 'pattern' THEN 1
                    WHEN type = 'decision' THEN 2
                    WHEN type = 'issue_resolution' THEN 3
                    ELSE 4
                END,
                created_at DESC
            LIMIT ?
        """
        
        # Add extra param for ORDER BY
        params.insert(-1, f"%{query}%")
        
        results = self.duckdb_conn.execute(sql, params).fetchall()
        
        # Format results
        formatted = []
        for row in results:
            entry = {
                'id': str(row[0]),
                'type': row[1],
                'name': row[2],
                'source': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'metadata': json.loads(row[6]) if row[6] else {},
                'created_at': str(row[7])
            }
            formatted.append(entry)
        
        return formatted
    
    def _vector_search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """ChromaDB vector similarity search"""
        # This would use the actual embedding pipeline
        # For now, return empty until we implement embeddings
        logger.warning("Vector search not yet implemented")
        return []
    
    # =========================================================================
    # CACHING SYSTEM FOR PERFORMANCE
    # =========================================================================
    
    def _get_cache_key(self, query: str, entity_types: Optional[List[str]], limit: int, use_vector: bool) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.lower().strip(),
            str(sorted(entity_types) if entity_types else []),
            str(limit),
            str(use_vector)
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get results from cache if not expired"""
        if cache_key in self.query_cache:
            cached_item = self.query_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['results']
            else:
                # Expired - remove from cache
                del self.query_cache[cache_key]
        return None
    
    def _put_in_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Put results in cache with TTL"""
        # Clean cache if too large
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = sorted(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k]['timestamp'])[:100]
            for old_key in oldest_keys:
                del self.query_cache[old_key]
        
        self.query_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
    
    # =========================================================================
    # PHASE 3: PATTERN AND RELATIONSHIP QUERIES
    # =========================================================================
    
    def get_patterns(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all patterns, optionally filtered by category"""
        sql = """
            SELECT 
                name, file_path, metadata
            FROM entities
            WHERE type = 'pattern'
        """
        
        if category:
            sql += " AND LOWER(CAST(metadata AS VARCHAR)) LIKE LOWER(?)"
            results = self.duckdb_conn.execute(sql, [f"%{category}%"]).fetchall()
        else:
            results = self.duckdb_conn.execute(sql).fetchall()
        
        patterns = []
        for row in results:
            patterns.append({
                'name': row[0],
                'source': row[1],
                'metadata': json.loads(row[2]) if row[2] else {}
            })
        
        return patterns
    
    def get_relationships(self, entity_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all relationships for an entity"""
        # Check if relationships table exists
        try:
            relationships = self.duckdb_conn.execute("""
                SELECT 
                    r.relationship_type,
                    r.target_id,
                    e.name as target_name,
                    e.type as target_type
                FROM relationships r
                JOIN entities e ON r.target_id = e.id
                WHERE r.source_id = ?
            """, [entity_id]).fetchall()
            
            grouped = {}
            for rel in relationships:
                rel_type = rel[0]
                if rel_type not in grouped:
                    grouped[rel_type] = []
                grouped[rel_type].append({
                    'id': str(rel[1]),
                    'name': rel[2],
                    'type': rel[3]
                })
            
            return grouped
        except:
            return {}
    
    # =========================================================================
    # MCP PROTOCOL IMPLEMENTATION
    # =========================================================================
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get('method')
        
        if method == 'initialize':
            return self._handle_initialize()
        elif method == 'tools/list':
            return self._handle_tools_list()
        elif method == 'tools/call':
            return self._handle_tool_call(request.get('params', {}))
        else:
            return {'error': {'code': -32601, 'message': f'Method not found: {method}'}}
    
    def _handle_initialize(self) -> Dict[str, Any]:
        """Initialize MCP server"""
        return {
            'protocolVersion': '2024-11-05',
            'capabilities': {
                'tools': {},
                'resources': {}  # Could add resource support later
            },
            'serverInfo': {
                'name': 'rif-knowledge',
                'version': '3.0.0',
                'description': 'Universal RIF Knowledge Graph Gateway'
            }
        }
    
    def _handle_tools_list(self) -> Dict[str, Any]:
        """List all available tools"""
        return {
            'tools': [
                # PHASE 1: Claude-specific tools
                {
                    'name': 'get_claude_documentation',
                    'description': 'Get Claude Code documentation, capabilities, and limitations',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'topic': {
                                'type': 'string',
                                'description': 'Topic to get documentation for (e.g., "capabilities", "limitations", "Task tool")'
                            }
                        },
                        'required': ['topic']
                    }
                },
                
                # PHASE 2: General knowledge tools
                {
                    'name': 'query_knowledge',
                    'description': 'Query the RIF knowledge graph for any information',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'Search query'
                            },
                            'entity_types': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Filter by types: pattern, decision, issue_resolution'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Max results (default: 10)'
                            }
                        },
                        'required': ['query']
                    }
                },
                
                # PHASE 3: Pattern tools
                {
                    'name': 'get_patterns',
                    'description': 'Get all patterns or filter by category',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'category': {
                                'type': 'string',
                                'description': 'Optional category filter'
                            }
                        }
                    }
                },
                
                # Relationship tools
                {
                    'name': 'get_relationships',
                    'description': 'Get all relationships for an entity',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'entity_id': {
                                'type': 'string',
                                'description': 'Entity ID to get relationships for'
                            }
                        },
                        'required': ['entity_id']
                    }
                },
                
                # Compatibility checking
                {
                    'name': 'check_compatibility',
                    'description': 'Check if an approach is compatible with Claude Code and RIF patterns',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'approach': {
                                'type': 'string',
                                'description': 'Approach or pattern to validate'
                            }
                        },
                        'required': ['approach']
                    }
                }
            ]
        }
    
    def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool invocation"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'get_claude_documentation':
                result = self.get_claude_documentation(arguments.get('topic', ''))
                return self._format_claude_documentation(result)
                
            elif tool_name == 'query_knowledge':
                results = self.query_knowledge(
                    arguments.get('query', ''),
                    arguments.get('entity_types'),
                    arguments.get('limit', 10)
                )
                return self._format_query_results(results)
                
            elif tool_name == 'get_patterns':
                patterns = self.get_patterns(arguments.get('category'))
                return self._format_patterns(patterns)
                
            elif tool_name == 'get_relationships':
                relationships = self.get_relationships(arguments.get('entity_id'))
                return self._format_relationships(relationships)
                
            elif tool_name == 'check_compatibility':
                # Special compatibility checking combining Claude knowledge
                approach = arguments.get('approach', '')
                
                # Check Claude limitations
                claude_docs = self.get_claude_documentation('limitations')
                
                # Search for anti-patterns
                anti_patterns = self.query_knowledge(approach, ['pattern'], limit=5)
                
                return self._format_compatibility_check(approach, claude_docs, anti_patterns)
                
            else:
                return {'error': {'code': -32601, 'message': f'Unknown tool: {tool_name}'}}
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {'error': {'code': -32603, 'message': f'Internal error: {str(e)}'}}
    
    # =========================================================================
    # RESPONSE FORMATTING
    # =========================================================================
    
    def _format_claude_documentation(self, docs: Dict[str, Any]) -> Dict[str, Any]:
        """Format Claude documentation for MCP response"""
        text = "# Claude Code Documentation\n\n"
        
        if 'research' in docs and docs['research']:
            research = docs['research']
            if 'claude_code_reality' in research:
                reality = research['claude_code_reality']
                text += "## Core Identity\n"
                text += f"- **What it is**: {reality.get('core_identity', {}).get('what_it_is', 'N/A')}\n"
                text += f"- **What it's not**: {reality.get('core_identity', {}).get('what_it_is_not', 'N/A')}\n\n"
        
        if 'capabilities' in docs:
            text += "## Capabilities\n"
            for cap in docs['capabilities']:
                text += f"- **{cap['category']}**: {cap['description']}\n"
                text += f"  Tools: {', '.join(cap['tools'])}\n"
        
        if 'limitations' in docs:
            text += "\n## Limitations\n"
            for lim in docs['limitations']:
                text += f"- **{lim['limitation']}**\n"
                text += f"  Workaround: {lim['workaround']}\n"
        
        if 'patterns' in docs and docs['patterns']:
            text += "\n## Related Patterns\n"
            for p in docs['patterns'][:5]:
                text += f"- {p['name']}\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _format_query_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format query results for MCP response"""
        if not results:
            return {'content': [{'type': 'text', 'text': 'No results found.'}]}
        
        text = f"# Knowledge Query Results ({len(results)} items)\n\n"
        
        for r in results:
            text += f"## {r['type'].upper()}: {r['name']}\n"
            
            if r.get('metadata'):
                meta = r['metadata']
                if meta.get('description'):
                    text += f"**Description**: {meta['description']}\n"
                if meta.get('solution'):
                    text += f"**Solution**: {meta['solution']}\n"
                if meta.get('rationale'):
                    text += f"**Rationale**: {meta['rationale']}\n"
            
            text += f"**Source**: {r['source']}"
            if r.get('line_start'):
                text += f" (lines {r['line_start']}-{r.get('line_end', r['line_start'])})"
            text += "\n\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format patterns for MCP response"""
        if not patterns:
            return {'content': [{'type': 'text', 'text': 'No patterns found.'}]}
        
        text = f"# Patterns ({len(patterns)} total)\n\n"
        
        for p in patterns:
            text += f"- **{p['name']}**\n"
            if p.get('metadata', {}).get('description'):
                text += f"  {p['metadata']['description']}\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _format_relationships(self, relationships: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Format relationships for MCP response"""
        if not relationships:
            return {'content': [{'type': 'text', 'text': 'No relationships found.'}]}
        
        text = "# Entity Relationships\n\n"
        
        for rel_type, targets in relationships.items():
            text += f"## {rel_type}\n"
            for target in targets:
                text += f"- {target['name']} ({target['type']})\n"
            text += "\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    def _format_compatibility_check(
        self, 
        approach: str, 
        claude_docs: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format compatibility check results"""
        text = f"# Compatibility Check: {approach}\n\n"
        
        # Check against limitations
        incompatible = False
        approach_lower = approach.lower()
        
        # Check for specific incompatibilities
        if 'agent' in approach_lower and ('communicat' in approach_lower or 'memory' in approach_lower or 'share' in approach_lower):
            incompatible = True
            text += "⚠️ **INCOMPATIBLE**: Agents cannot communicate directly or share memory\n"
            text += "   Suggested workaround: Use files or GitHub issues for coordination\n\n"
        
        if 'task.parallel()' in approach_lower:
            incompatible = True
            text += "⚠️ **INCOMPATIBLE**: Task.parallel() is pseudocode, not a real function\n"
            text += "   Suggested workaround: Launch multiple Task tools in one response for parallel execution\n\n"
        
        if 'persistent' in approach_lower or 'background' in approach_lower:
            incompatible = True
            text += "⚠️ **INCOMPATIBLE**: No persistent background processes\n"
            text += "   Suggested workaround: Use Bash with run_in_background for session-scoped tasks\n\n"
        
        if 'external' in approach_lower and ('monitor' in approach_lower or 'orchestrat' in approach_lower):
            incompatible = True
            text += "⚠️ **INCOMPATIBLE**: Cannot be externally orchestrated or monitor continuously\n"
            text += "   Suggested workaround: Claude Code IS the orchestrator, use session-based checking\n\n"
        
        # Check patterns
        anti_patterns_found = False
        for p in patterns:
            if 'anti' in p['name'].lower():
                anti_patterns_found = True
                text += f"❌ **Anti-pattern detected**: {p['name']}\n"
        
        if not incompatible and not anti_patterns_found:
            text += "✅ **COMPATIBLE**: This approach appears to be compatible with Claude Code and RIF patterns.\n"
        
        return {'content': [{'type': 'text', 'text': text}]}
    
    # =========================================================================
    # MAIN SERVER LOOP
    # =========================================================================
    
    def run_stdio(self):
        """Run as stdio MCP server"""
        logger.info("RIF Knowledge MCP Server starting...")
        logger.info(f"Database path: {self.duckdb_path}")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                logger.debug(f"Request: {request}")
                
                result = self.handle_request(request)
                
                response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id')
                }
                
                if 'error' in result:
                    response['error'] = result['error']
                else:
                    response['result'] = result
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {'code': -32700, 'message': 'Parse error'}
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except KeyboardInterrupt:
                logger.info("Server shutting down...")
                break
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
                error_response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id') if 'request' in locals() else None,
                    'error': {'code': -32603, 'message': str(e)}
                }
                print(json.dumps(error_response))
                sys.stdout.flush()


if __name__ == '__main__':
    server = RIFKnowledgeServer()
    server.run_stdio()