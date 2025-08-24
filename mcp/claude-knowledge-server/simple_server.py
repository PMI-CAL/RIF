#!/usr/bin/env python3
"""
Simple Claude Code Knowledge MCP Server
A minimal synchronous implementation that works with Claude Code
"""

import json
import sys

def send_response(response):
    """Send JSON-RPC response"""
    sys.stdout.write(json.dumps(response) + '\n')
    sys.stdout.flush()

def main():
    """Main server loop"""
    while True:
        try:
            # Read line from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            # Parse request
            request = json.loads(line.strip())
            method = request.get('method')
            request_id = request.get('id')
            
            # Handle different methods
            if method == 'initialize':
                response = {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'result': {
                        'protocolVersion': '2024-11-05',
                        'capabilities': {
                            'tools': {}
                        },
                        'serverInfo': {
                            'name': 'claude-knowledge-server',
                            'version': '1.0.0'
                        }
                    }
                }
                
            elif method == 'tools/list':
                response = {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'result': {
                        'tools': [
                            {
                                'name': 'check_claude_capability',
                                'description': 'Check if Claude Code can perform a specific action',
                                'inputSchema': {
                                    'type': 'object',
                                    'properties': {
                                        'action': {
                                            'type': 'string',
                                            'description': 'The action to check'
                                        }
                                    },
                                    'required': ['action']
                                }
                            },
                            {
                                'name': 'get_implementation_pattern',
                                'description': 'Get correct implementation pattern for a task',
                                'inputSchema': {
                                    'type': 'object',
                                    'properties': {
                                        'task': {
                                            'type': 'string',
                                            'description': 'The task type (e.g., github, mcp, orchestration)'
                                        }
                                    },
                                    'required': ['task']
                                }
                            },
                            {
                                'name': 'check_compatibility',
                                'description': 'Check if an approach is compatible with Claude Code',
                                'inputSchema': {
                                    'type': 'object',
                                    'properties': {
                                        'approach': {
                                            'type': 'string',
                                            'description': 'The approach to validate'
                                        }
                                    },
                                    'required': ['approach']
                                }
                            }
                        ]
                    }
                }
                
            elif method == 'tools/call':
                params = request.get('params', {})
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                
                if tool_name == 'check_claude_capability':
                    action = arguments.get('action', '').lower()
                    
                    # Simple capability check
                    if any(word in action for word in ['file', 'read', 'write', 'edit']):
                        result_text = "Yes, Claude Code can work with files using Read, Write, Edit, and MultiEdit tools."
                    elif any(word in action for word in ['bash', 'command', 'execute']):
                        result_text = "Yes, Claude Code can execute bash commands."
                    elif any(word in action for word in ['mcp', 'server']):
                        result_text = "Yes, Claude Code can use MCP servers for extended capabilities."
                    elif 'parallel' in action and 'task' in action:
                        result_text = "No, Task.parallel() is pseudocode. Launch multiple Task tools in one response for parallel execution."
                    elif 'persistent' in action or 'background process' in action:
                        result_text = "No, Claude Code cannot run persistent background processes. Only session-scoped background tasks via Bash run_in_background."
                    elif 'search' in action:
                        result_text = "Yes, Claude Code can search code using Grep for content and Glob for file patterns."
                    elif 'task' in action and 'delegat' in action:
                        result_text = "Yes, Claude Code can delegate tasks to subagents using the Task tool."
                    elif 'web' in action:
                        result_text = "Yes, Claude Code can access web content using WebSearch and WebFetch tools."
                    else:
                        result_text = f"Unknown capability: {arguments.get('action')}. Check documentation."
                    
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'content': [
                                {
                                    'type': 'text',
                                    'text': result_text
                                }
                            ]
                        }
                    }
                    
                elif tool_name == 'get_implementation_pattern':
                    task = arguments.get('task', '').lower()
                    
                    patterns = {
                        'github': 'Use gh CLI via Bash tool. Example: gh issue list --state open',
                        'mcp': 'Add MCP servers with: claude mcp add name -- command args',
                        'orchestration': 'Launch multiple Task tools in one response for parallel execution',
                        'file': 'Use Read/Write/Edit/MultiEdit tools for file operations',
                        'search': 'Use Grep for content search, Glob for file patterns'
                    }
                    
                    if task in patterns:
                        result_text = f"Pattern for {task}: {patterns[task]}"
                    else:
                        result_text = f"Available patterns: {', '.join(patterns.keys())}"
                    
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'content': [
                                {
                                    'type': 'text',
                                    'text': result_text
                                }
                            ]
                        }
                    }
                    
                elif tool_name == 'check_compatibility':
                    approach = arguments.get('approach', '').lower()
                    
                    if 'task.parallel()' in approach:
                        result_text = "INCOMPATIBLE: Task.parallel() is pseudocode. Use multiple Task tools in one response instead."
                    elif 'background process' in approach or 'persistent' in approach:
                        result_text = "INCOMPATIBLE: No persistent background processes. Use Bash with run_in_background for session-scoped background tasks."
                    elif ('orchestrat' in approach and 'external' in approach) or ('external' in approach and 'claude code' in approach):
                        result_text = "INCOMPATIBLE: Claude Code cannot be externally orchestrated. Claude Code IS the orchestrator."
                    elif 'agent' in approach and ('communicat' in approach or 'shared memory' in approach):
                        result_text = "INCOMPATIBLE: Agents cannot communicate directly or share memory. They must use files or GitHub issues for coordination."
                    elif 'external' in approach and 'monitoring' in approach:
                        result_text = "INCOMPATIBLE: Cannot monitor external systems continuously. Only session-based checking is possible."
                    else:
                        result_text = f"COMPATIBLE: {approach} appears to be compatible with Claude Code."
                    
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'content': [
                                {
                                    'type': 'text',
                                    'text': result_text
                                }
                            ]
                        }
                    }
                    
                else:
                    response = {
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'error': {
                            'code': -32601,
                            'message': f'Unknown tool: {tool_name}'
                        }
                    }
                    
            else:
                response = {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'error': {
                        'code': -32601,
                        'message': f'Method not found: {method}'
                    }
                }
            
            send_response(response)
            
        except json.JSONDecodeError:
            error_response = {
                'jsonrpc': '2.0',
                'id': None,
                'error': {
                    'code': -32700,
                    'message': 'Parse error'
                }
            }
            send_response(error_response)
        except Exception as e:
            error_response = {
                'jsonrpc': '2.0',
                'id': request_id if 'request_id' in locals() else None,
                'error': {
                    'code': -32603,
                    'message': f'Internal error: {str(e)}'
                }
            }
            send_response(error_response)

if __name__ == '__main__':
    main()