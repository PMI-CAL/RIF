#!/usr/bin/env python3
"""
Claude Code Knowledge MCP Server - Synchronous Version
Provides accurate knowledge about Claude Code's capabilities via MCP protocol

This server implements the MCP (Model Context Protocol) specification
using synchronous I/O for better compatibility with MCP health checks.
"""

import json
import sys
import logging
from typing import Dict, List, Any

# Disable logging to avoid interfering with stdio
logging.disable(logging.CRITICAL)


class ClaudeKnowledgeMCPServer:
    """MCP server providing Claude Code capability knowledge"""
    
    def __init__(self):
        """Initialize the server with knowledge data"""
        self.capabilities = {
            "file_operations": "Read, Write, Edit, MultiEdit files",
            "bash_execution": "Execute bash commands with optional background execution",
            "web_access": "WebSearch and WebFetch for internet content",
            "task_delegation": "Launch subagents via Task tool",
            "mcp_integration": "Use MCP servers for extended capabilities",
            "code_analysis": "Search code with Grep and Glob tools"
        }
        self.limitations = {
            "no_task_parallel": "Task.parallel() is pseudocode, not a real function",
            "session_scope": "No persistence between sessions without explicit state management", 
            "no_persistent_background": "Cannot run truly persistent background processes",
            "parallel_execution": "Multiple Task tools in one response run in parallel"
        }
        
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "claude-knowledge-server",
                "version": "1.0.0"
            }
        }
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        return {
            "tools": [
                {
                    "name": "check_claude_capability",
                    "description": "Check if Claude Code can perform a specific action",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string", 
                                "description": "The action to check"
                            }
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "get_implementation_pattern", 
                    "description": "Get correct implementation pattern for a task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to implement"
                            }
                        },
                        "required": ["task"]
                    }
                },
                {
                    "name": "check_compatibility",
                    "description": "Check if an approach is compatible with Claude Code",
                    "inputSchema": {
                        "type": "object", 
                        "properties": {
                            "approach": {
                                "type": "string",
                                "description": "The approach to validate"
                            }
                        },
                        "required": ["approach"]
                    }
                }
            ]
        }
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "check_claude_capability":
            return self.check_capability(arguments.get("action", ""))
        elif tool_name == "get_implementation_pattern":
            return self.get_pattern(arguments.get("task", ""))
        elif tool_name == "check_compatibility":
            return self.check_compatibility_impl(arguments.get("approach", ""))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def check_capability(self, action: str) -> Dict[str, Any]:
        """Check if Claude Code can perform an action"""
        action_lower = action.lower()
        
        # Check capabilities
        for cap_key, cap_desc in self.capabilities.items():
            if (cap_key in action_lower or 
                any(word in action_lower for word in cap_desc.lower().split())):
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Yes, Claude Code can: {cap_desc}"
                    }]
                }
        
        # Check limitations
        for lim_key, lim_desc in self.limitations.items():
            if any(word in action_lower for word in ["parallel", "background", "persistent"]):
                return {
                    "content": [{
                        "type": "text", 
                        "text": f"Limitation: {lim_desc}"
                    }]
                }
        
        return {
            "content": [{
                "type": "text",
                "text": f"Unknown capability: {action}. Available tools include: file operations, bash execution, web access, task delegation, MCP integration, and code analysis."
            }]
        }
    
    def get_pattern(self, task: str) -> Dict[str, Any]:
        """Get implementation pattern for a task"""
        patterns = {
            "github": {
                "pattern": "github-cli-integration",
                "description": "Use gh CLI via Bash tool for GitHub operations",
                "example": "Bash(command='gh issue list --state open', description='List open GitHub issues')"
            },
            "mcp": {
                "pattern": "mcp-server-integration",
                "description": "Configure MCP servers using claude mcp add command", 
                "example": "Bash(command='claude mcp add server-name \"python server.py\"', description='Add MCP server')"
            },
            "orchestration": {
                "pattern": "task-delegation",
                "description": "Use Task tool to launch subagents, multiple in one response for parallel",
                "example": "Task(subagent_type='general-purpose', prompt='You are a specialist agent...', description='Agent task')"
            },
            "file": {
                "pattern": "file-operations",
                "description": "Read files first, then edit with exact strings",
                "example": "Read(file_path='/path/file.py') then Edit(file_path='/path/file.py', old_string='exact text', new_string='new text')"
            }
        }
        
        task_lower = task.lower()
        for key, pattern in patterns.items():
            if key in task_lower:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(pattern, indent=2)
                    }]
                }
        
        return {
            "content": [{
                "type": "text",
                "text": f"No specific pattern found for: {task}. Available patterns: github, mcp, orchestration, file operations."
            }]
        }
    
    def check_compatibility_impl(self, approach: str) -> Dict[str, Any]:
        """Check if an approach is compatible with Claude Code"""
        approach_lower = approach.lower()
        
        # Check anti-patterns
        if "task.parallel()" in approach_lower:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "compatible": False,
                        "reason": "Task.parallel() is pseudocode, not a real function",
                        "alternatives": ["Launch multiple Task tools in one response for parallel execution"]
                    }, indent=2)
                }]
            }
        
        if "background process" in approach_lower or "persistent daemon" in approach_lower:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "compatible": False,
                        "reason": "Claude Code cannot run persistent background processes between sessions",
                        "alternatives": ["Use Bash with run_in_background for commands within session scope"]
                    }, indent=2)
                }]
            }
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "compatible": True,
                    "approach": approach,
                    "note": "Approach appears compatible with Claude Code capabilities"
                }, indent=2)
            }]
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return self.handle_initialize(params)
        elif method == "tools/list":
            return self.handle_tools_list(params)
        elif method == "tools/call":
            return self.handle_tools_call(params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run(self):
        """Run the MCP server using stdio transport"""        
        try:
            while True:
                # Read request from stdin
                line = sys.stdin.readline()
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON-RPC request
                    request = json.loads(line)
                    
                    # Handle the request
                    result = self.handle_request(request)
                    
                    # Build JSON-RPC response
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": result
                    }
                    
                except json.JSONDecodeError as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                            "data": str(e)
                        }
                    }
                
                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id") if 'request' in locals() else None,
                        "error": {
                            "code": -32603,
                            "message": "Internal error",
                            "data": str(e)
                        }
                    }
                
                # Send response
                print(json.dumps(response), flush=True)
                
        except KeyboardInterrupt:
            pass
        except Exception:
            pass


def main():
    """Main entry point"""
    server = ClaudeKnowledgeMCPServer()
    server.run()


if __name__ == "__main__":
    main()