#!/usr/bin/env python3
"""
Test MCP health check protocol by monitoring what Claude sends
"""

import sys
import json
import time

def log_debug(msg):
    with open("/tmp/mcp_debug.log", "a") as f:
        f.write(f"{time.time()}: {msg}\n")
        f.flush()

def main():
    log_debug("MCP server started")
    
    try:
        while True:
            log_debug("Waiting for input...")
            line = sys.stdin.readline()
            
            if not line:
                log_debug("No input received, breaking")
                break
            
            log_debug(f"Received: {line.strip()}")
            
            try:
                request = json.loads(line.strip())
                log_debug(f"Parsed request: {request}")
                
                # Simple response to any request
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"status": "ok"}
                }
                
                response_str = json.dumps(response)
                log_debug(f"Sending response: {response_str}")
                
                print(response_str)
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                log_debug(f"JSON decode error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
                
    except Exception as e:
        log_debug(f"Exception: {e}")

if __name__ == "__main__":
    main()