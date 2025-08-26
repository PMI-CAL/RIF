#!/usr/bin/env python3
"""
Simple MCP Server for RIF Knowledge Base
Provides access to the knowledge system via MCP protocol
"""

import json
import sys
import asyncio
import warnings
import os
from typing import Dict, Any, List
from pathlib import Path

# Suppress ALL warnings before importing anything
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add RIF to path
sys.path.insert(0, str(Path(__file__).parent))

# Redirect stderr to devnull during import
import os
old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from knowledge import get_knowledge_system
sys.stderr = old_stderr

class RIFKnowledgeMCPServer:
    """Simple MCP server for RIF knowledge base."""
    
    def __init__(self):
        self.ks = get_knowledge_system()
        # Don't print to stderr - it breaks MCP protocol
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests."""
        method = request.get('method', '')
        params = request.get('params', {})
        
        try:
            if method == 'initialize':
                return {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {
                        'tools': {}
                    },
                    'serverInfo': {
                        'name': 'rif-knowledge',
                        'version': '1.0.0'
                    }
                }
            elif method == 'tools/list':
                return {
                    'tools': [
                        {
                            'name': 'query_knowledge',
                            'description': 'Query the RIF knowledge base',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'query': {'type': 'string', 'description': 'Query string'},
                                    'collection': {'type': 'string', 'description': 'Optional collection name'}
                                },
                                'required': ['query']
                            }
                        },
                        {
                            'name': 'search_patterns',
                            'description': 'Search for patterns in the knowledge base',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'query': {'type': 'string', 'description': 'Search query'},
                                    'limit': {'type': 'integer', 'description': 'Max results', 'default': 10}
                                },
                                'required': ['query']
                            }
                        },
                        {
                            'name': 'find_similar_issues',
                            'description': 'Find similar issues in the knowledge base',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'description': {'type': 'string', 'description': 'Issue description'},
                                    'limit': {'type': 'integer', 'description': 'Max results', 'default': 5}
                                },
                                'required': ['description']
                            }
                        }
                    ]
                }
            
            elif method == 'tools/call':
                tool_name = params.get('name', '')
                arguments = params.get('arguments', {})
                
                if tool_name == 'query_knowledge':
                    query = arguments.get('query', '')
                    collection = arguments.get('collection')
                    
                    if collection:
                        # Query specific collection
                        results = self.ks.retrieve_knowledge(query)
                        filtered = [r for r in results if r.get('collection') == collection]
                        return {'content': filtered[:10]}
                    else:
                        results = self.ks.retrieve_knowledge(query)
                        return {'content': results[:10]}
                
                elif tool_name == 'search_patterns':
                    query = arguments.get('query', '')
                    limit = arguments.get('limit', 10)
                    results = self.ks.search_patterns(query, limit=limit)
                    return {'content': results}
                
                elif tool_name == 'find_similar_issues':
                    description = arguments.get('description', '')
                    limit = arguments.get('limit', 5)
                    results = self.ks.find_similar_issues(description, limit=limit)
                    return {'content': results}
                
                else:
                    return {'error': f'Unknown tool: {tool_name}'}
            
            else:
                return {'error': f'Unknown method: {method}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def run(self):
        """Run the MCP server, reading from stdin and writing to stdout."""
        # Don't print to stderr - it breaks MCP protocol
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                
                # Handle request
                response = await self.handle_request(request)
                
                # Add request ID to response
                response['id'] = request.get('id')
                
                # Write response to stdout
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                error_response = {
                    'error': f'Invalid JSON: {e}',
                    'id': None
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                print(f"Server error: {e}", file=sys.stderr)
                break

def main():
    """Main entry point."""
    server = RIFKnowledgeMCPServer()
    asyncio.run(server.run())

if __name__ == '__main__':
    main()