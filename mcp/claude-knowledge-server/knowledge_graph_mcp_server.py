#!/usr/bin/env python3
"""
Claude Code Knowledge MCP Server - Knowledge Graph Integration
Queries the RIF knowledge graph for actual stored knowledge about Claude Code
"""

import json
import sys
import os
import logging
import duckdb
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/tmp/mcp_knowledge_server.log')]
)
logger = logging.getLogger(__name__)

class KnowledgeGraphMCPServer:
    """MCP server that queries the RIF knowledge graph"""
    
    def __init__(self):
        """Initialize connection to knowledge graph"""
        self.db_path = str(Path(__file__).parent.parent.parent / "knowledge" / "hybrid_knowledge.duckdb")
        self.conn = None
        self.connect_db()
        
    def connect_db(self):
        """Connect to DuckDB knowledge graph"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to knowledge graph at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def query_knowledge(self, query: str, entity_types: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge graph for relevant information"""
        if not self.conn:
            return []
        
        try:
            # Build query based on entity types
            if entity_types:
                type_filter = f"AND type IN ({','.join(['?' for _ in entity_types])})"
                params = [f"%{query}%"] + entity_types + [limit]
            else:
                type_filter = ""
                params = [f"%{query}%", limit]
            
            sql = f"""
                SELECT 
                    type,
                    name,
                    file_path,
                    metadata
                FROM entities
                WHERE (
                    LOWER(name) LIKE LOWER(?) OR 
                    LOWER(CAST(metadata AS VARCHAR)) LIKE LOWER(?)
                )
                {type_filter}
                ORDER BY 
                    CASE 
                        WHEN type = 'pattern' THEN 1
                        WHEN type = 'decision' THEN 2
                        WHEN type = 'issue_resolution' THEN 3
                        ELSE 4
                    END
                LIMIT ?
            """
            
            # Duplicate query param for name and metadata search
            if entity_types:
                params = [f"%{query}%", f"%{query}%"] + entity_types + [limit]
            else:
                params = [f"%{query}%", f"%{query}%", limit]
            
            results = self.conn.execute(sql, params).fetchall()
            
            # Format results
            formatted = []
            for row in results:
                entry = {
                    'type': row[0],
                    'name': row[1],
                    'source': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {}
                }
                formatted.append(entry)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def check_pattern_compatibility(self, approach: str) -> Dict[str, Any]:
        """Check if an approach matches known patterns or anti-patterns"""
        # Search for patterns related to the approach
        patterns = self.query_knowledge(approach, ['pattern'], limit=10)
        
        # Look for anti-patterns
        anti_patterns = []
        compatible_patterns = []
        
        for pattern in patterns:
            metadata = pattern.get('metadata', {})
            if 'anti-pattern' in pattern['name'].lower() or metadata.get('category') == 'anti-pattern':
                anti_patterns.append(pattern)
            else:
                compatible_patterns.append(pattern)
        
        # Search for specific issue resolutions
        issues = self.query_knowledge(approach, ['issue_resolution'], limit=5)
        
        return {
            'anti_patterns': anti_patterns,
            'compatible_patterns': compatible_patterns,
            'related_issues': issues
        }
    
    def get_aggregated_context(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive context about a topic from all sources"""
        context = {
            'patterns': self.query_knowledge(topic, ['pattern'], limit=5),
            'decisions': self.query_knowledge(topic, ['decision'], limit=3),
            'issues': self.query_knowledge(topic, ['issue_resolution'], limit=3),
            'all_knowledge': self.query_knowledge(topic, limit=10)
        }
        
        # Load specific Claude Code research if available
        if 'claude' in topic.lower() or 'agent' in topic.lower():
            claude_patterns = self.query_knowledge('claude', ['pattern'], limit=5)
            context['claude_specific'] = claude_patterns
        
        return context
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get('method')
        request_id = request.get('id')
        
        if method == 'initialize':
            return {
                'protocolVersion': '2024-11-05',
                'capabilities': {'tools': {}},
                'serverInfo': {
                    'name': 'claude-knowledge-graph',
                    'version': '2.0.0'
                }
            }
            
        elif method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': 'query_claude_knowledge',
                        'description': 'Query the RIF knowledge graph for Claude Code patterns and knowledge',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'query': {
                                    'type': 'string',
                                    'description': 'Search query (e.g., "agent communication", "Task.parallel", "MCP servers")'
                                },
                                'entity_types': {
                                    'type': 'array',
                                    'items': {'type': 'string'},
                                    'description': 'Optional: Filter by entity types (pattern, decision, issue_resolution)'
                                },
                                'limit': {
                                    'type': 'integer',
                                    'description': 'Maximum results to return (default: 5)'
                                }
                            },
                            'required': ['query']
                        }
                    },
                    {
                        'name': 'check_compatibility',
                        'description': 'Check if an approach is compatible based on stored patterns',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'approach': {
                                    'type': 'string',
                                    'description': 'The approach or pattern to validate'
                                }
                            },
                            'required': ['approach']
                        }
                    },
                    {
                        'name': 'get_context',
                        'description': 'Get comprehensive context about a topic from all knowledge sources',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'topic': {
                                    'type': 'string',
                                    'description': 'Topic to get context about (e.g., "orchestration", "file monitoring")'
                                }
                            },
                            'required': ['topic']
                        }
                    }
                ]
            }
            
        elif method == 'tools/call':
            params = request.get('params', {})
            tool_name = params.get('name')
            arguments = params.get('arguments', {})
            
            try:
                if tool_name == 'query_claude_knowledge':
                    query = arguments.get('query', '')
                    entity_types = arguments.get('entity_types')
                    limit = arguments.get('limit', 5)
                    
                    results = self.query_knowledge(query, entity_types, limit)
                    
                    # Format response
                    if results:
                        text = f"Found {len(results)} relevant knowledge items:\n\n"
                        for r in results:
                            text += f"**{r['type'].upper()}: {r['name']}**\n"
                            if r.get('metadata', {}).get('description'):
                                text += f"Description: {r['metadata']['description']}\n"
                            if r.get('metadata', {}).get('solution'):
                                text += f"Solution: {r['metadata']['solution']}\n"
                            text += f"Source: {r['source']}\n\n"
                    else:
                        text = f"No knowledge found for query: {query}"
                    
                    return {
                        'content': [{'type': 'text', 'text': text}]
                    }
                    
                elif tool_name == 'check_compatibility':
                    approach = arguments.get('approach', '')
                    result = self.check_pattern_compatibility(approach)
                    
                    # Format response
                    text = ""
                    if result['anti_patterns']:
                        text += "⚠️ POTENTIAL INCOMPATIBILITIES FOUND:\n\n"
                        for ap in result['anti_patterns']:
                            text += f"- **{ap['name']}**\n"
                            if ap.get('metadata', {}).get('reason'):
                                text += f"  Reason: {ap['metadata']['reason']}\n"
                    
                    if result['compatible_patterns']:
                        text += "\n✅ COMPATIBLE PATTERNS:\n\n"
                        for cp in result['compatible_patterns']:
                            text += f"- **{cp['name']}**\n"
                            if cp.get('metadata', {}).get('description'):
                                text += f"  {cp['metadata']['description']}\n"
                    
                    if result['related_issues']:
                        text += "\n📋 RELATED ISSUE RESOLUTIONS:\n\n"
                        for issue in result['related_issues']:
                            text += f"- {issue['name']}\n"
                    
                    if not text:
                        text = f"No specific patterns found for: {approach}"
                    
                    return {
                        'content': [{'type': 'text', 'text': text}]
                    }
                    
                elif tool_name == 'get_context':
                    topic = arguments.get('topic', '')
                    context = self.get_aggregated_context(topic)
                    
                    # Format comprehensive response
                    text = f"# Context for: {topic}\n\n"
                    
                    if context['patterns']:
                        text += "## Patterns\n"
                        for p in context['patterns'][:3]:
                            text += f"- **{p['name']}**"
                            if p.get('metadata', {}).get('description'):
                                text += f": {p['metadata']['description']}"
                            text += "\n"
                    
                    if context['decisions']:
                        text += "\n## Architectural Decisions\n"
                        for d in context['decisions'][:3]:
                            text += f"- **{d['name']}**\n"
                    
                    if context['issues']:
                        text += "\n## Related Issues\n"
                        for i in context['issues'][:3]:
                            text += f"- {i['name']}\n"
                    
                    if 'claude_specific' in context and context['claude_specific']:
                        text += "\n## Claude Code Specific Knowledge\n"
                        for c in context['claude_specific'][:3]:
                            text += f"- {c['name']}\n"
                    
                    return {
                        'content': [{'type': 'text', 'text': text}]
                    }
                    
                else:
                    return {
                        'error': {
                            'code': -32601,
                            'message': f'Unknown tool: {tool_name}'
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {
                    'error': {
                        'code': -32603,
                        'message': f'Internal error: {str(e)}'
                    }
                }
                
        else:
            return {
                'error': {
                    'code': -32601,
                    'message': f'Method not found: {method}'
                }
            }
    
    def run_stdio(self):
        """Run as stdio MCP server"""
        logger.info("MCP Knowledge Graph Server starting...")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                logger.info(f"Received request: {request.get('method')}")
                
                result = self.handle_request(request)
                
                # Build JSON-RPC response
                response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id')
                }
                
                if 'error' in result:
                    response['error'] = result['error']
                else:
                    response['result'] = result
                
                # Send response
                print(json.dumps(response))
                sys.stdout.flush()
                logger.info(f"Sent response for: {request.get('method')}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {
                        'code': -32700,
                        'message': 'Parse error'
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Server error: {e}")
                error_response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id') if 'request' in locals() else None,
                    'error': {
                        'code': -32603,
                        'message': f'Internal error: {str(e)}'
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()

if __name__ == '__main__':
    server = KnowledgeGraphMCPServer()
    server.run_stdio()