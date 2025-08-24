#!/usr/bin/env python3
"""
Debug version of the MCP server to see what's going wrong
"""

import json
import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rif_knowledge_server import RIFKnowledgeServer

def debug_server():
    """Run debug version"""
    print(f"DEBUG: Working directory: {os.getcwd()}", file=sys.stderr)
    print(f"DEBUG: Script location: {__file__}", file=sys.stderr)
    
    # Check if we're in the right directory
    expected_path = Path("/Users/cal/DEV/RIF/knowledge/hybrid_knowledge.duckdb")
    print(f"DEBUG: Database path exists: {expected_path.exists()}", file=sys.stderr)
    
    server = RIFKnowledgeServer()
    print(f"DEBUG: Server database path: {server.duckdb_path}", file=sys.stderr)
    print(f"DEBUG: Server database exists: {server.duckdb_path.exists()}", file=sys.stderr)
    
    # Test a simple query
    try:
        results = server.query_knowledge('orchestration')
        print(f"DEBUG: Query found {len(results)} results", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Query failed: {e}", file=sys.stderr)
    
    # Process single request from stdin
    line = sys.stdin.readline()
    if line:
        print(f"DEBUG: Received: {line.strip()}", file=sys.stderr)
        try:
            request = json.loads(line.strip())
            result = server.handle_request(request)
            response = {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'result': result
            }
            print(json.dumps(response))
        except Exception as e:
            print(f"DEBUG: Error: {e}", file=sys.stderr)
            error_response = {
                'jsonrpc': '2.0',
                'id': request.get('id') if 'request' in locals() else None,
                'error': {'code': -32603, 'message': str(e)}
            }
            print(json.dumps(error_response))

if __name__ == '__main__':
    debug_server()