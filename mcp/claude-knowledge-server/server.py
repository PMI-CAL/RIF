#!/usr/bin/env python3
"""
Claude Code Knowledge MCP Server
Provides accurate knowledge about Claude Code's capabilities via MCP protocol

This server implements the MCP (Model Context Protocol) specification
and provides tools to query Claude Code capabilities and limitations.
"""

import json
import sys
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

# Set up logging to file instead of stderr to not interfere with stdio
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/mcp_server.log'),
        # Don't log to stderr which could interfere with MCP stdio protocol
    ]
)
logger = logging.getLogger(__name__)


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
        
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "check_claude_capability":
            return await self.check_capability(arguments.get("action", ""))
        elif tool_name == "get_implementation_pattern":
            return await self.get_pattern(arguments.get("task", ""))
        elif tool_name == "check_compatibility":
            return await self.check_compatibility_impl(arguments.get("approach", ""))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def check_capability(self, action: str) -> Dict[str, Any]:
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
    
    async def get_pattern(self, task: str) -> Dict[str, Any]:
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
    
    async def check_compatibility_impl(self, approach: str) -> Dict[str, Any]:
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
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return await self.handle_initialize(params)
        elif method == "tools/list":
            return await self.handle_tools_list(params)
        elif method == "tools/call":
            return await self.handle_tools_call(params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    async def run(self):
        """Run the MCP server using stdio transport"""
        logger.info("Starting Claude Code Knowledge MCP Server")
        
        try:
            while True:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    logger.info("End of input, shutting down")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON-RPC request
                    request = json.loads(line)
                    logger.debug(f"Received request: {request}")
                    
                    # Handle the request
                    result = await self.handle_request(request)
                    
                    # Build JSON-RPC response
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": result
                    }
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
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
                    logger.error(f"Request handling error: {e}")
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
                response_str = json.dumps(response)
                print(response_str, flush=True)
                logger.debug(f"Sent response: {response_str}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


async def main():
    """Main entry point"""
    server = ClaudeKnowledgeMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())