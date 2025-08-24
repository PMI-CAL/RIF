#!/usr/bin/env python3
"""
Test script for RIF Knowledge MCP Server
Tests both Claude-specific and general knowledge queries
"""

import json
import subprocess
import sys

def send_request(request):
    """Send request to MCP server"""
    process = subprocess.Popen(
        ["python3", "/Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=json.dumps(request))
    try:
        response = json.loads(stdout.strip())
        return response
    except:
        print(f"Failed to parse response: {stdout}")
        print(f"Stderr: {stderr}")
        return None

def test_claude_documentation():
    """Test Claude Code documentation retrieval"""
    print("\n" + "="*60)
    print("TEST 1: Claude Code Documentation")
    print("="*60)
    
    # Test capabilities
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "get_claude_documentation",
            "arguments": {"topic": "capabilities"}
        }
    }
    
    response = send_request(request)
    if response and 'result' in response:
        text = response['result']['content'][0]['text']
        print("✅ Claude capabilities retrieved:")
        print(text[:500] + "..." if len(text) > 500 else text)
    else:
        print("❌ Failed to get Claude documentation")
        print(response)
    
    # Test limitations
    request['params']['arguments']['topic'] = 'limitations'
    request['id'] = 2
    
    response = send_request(request)
    if response and 'result' in response:
        text = response['result']['content'][0]['text']
        if 'Limitations' in text or 'limitation' in text.lower():
            print("\n✅ Claude limitations retrieved")
        else:
            print("\n⚠️ Limitations section missing")
    else:
        print("\n❌ Failed to get limitations")

def test_general_queries():
    """Test general knowledge queries"""
    print("\n" + "="*60)
    print("TEST 2: General Knowledge Queries")
    print("="*60)
    
    # Query for patterns
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "query_knowledge",
            "arguments": {
                "query": "orchestrator",
                "entity_types": ["pattern", "decision"],
                "limit": 3
            }
        }
    }
    
    response = send_request(request)
    if response and 'result' in response:
        text = response['result']['content'][0]['text']
        if 'No results' not in text:
            print("✅ Pattern query successful")
            print(f"Results preview: {text[:300]}...")
        else:
            print("⚠️ No patterns found for 'orchestrator'")
    else:
        print("❌ Pattern query failed")
    
    # Query for issues
    request['params']['arguments'] = {
        "query": "error",
        "entity_types": ["issue_resolution"],
        "limit": 3
    }
    request['id'] = 4
    
    response = send_request(request)
    if response and 'result' in response:
        text = response['result']['content'][0]['text']
        print(f"\n{'✅' if 'No results' not in text else '⚠️'} Issue query: {text[:100]}...")
    else:
        print("\n❌ Issue query failed")

def test_compatibility_check():
    """Test compatibility checking"""
    print("\n" + "="*60)
    print("TEST 3: Compatibility Checking")
    print("="*60)
    
    test_cases = [
        ("Using Task.parallel() for orchestration", False),
        ("Using gh CLI for GitHub automation", True),
        ("Running persistent background processes", False),
        ("Using MCP servers for extended functionality", True)
    ]
    
    for approach, should_be_compatible in test_cases:
        request = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "check_compatibility",
                "arguments": {"approach": approach}
            }
        }
        
        response = send_request(request)
        if response and 'result' in response:
            text = response['result']['content'][0]['text']
            is_compatible = "COMPATIBLE" in text and "INCOMPATIBLE" not in text
            
            if is_compatible == should_be_compatible:
                print(f"✅ Correctly assessed: {approach[:40]}...")
            else:
                print(f"❌ Wrong assessment for: {approach[:40]}...")
                print(f"   Expected: {'Compatible' if should_be_compatible else 'Incompatible'}")
                print(f"   Got: {text[:100]}...")
        else:
            print(f"❌ Failed to check: {approach}")

def test_pattern_retrieval():
    """Test pattern retrieval"""
    print("\n" + "="*60)
    print("TEST 4: Pattern Retrieval")
    print("="*60)
    
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "get_patterns",
            "arguments": {}
        }
    }
    
    response = send_request(request)
    if response and 'result' in response:
        text = response['result']['content'][0]['text']
        pattern_count = text.count('pattern')
        print(f"✅ Retrieved patterns (found {pattern_count} references)")
        print(text[:400] + "..." if len(text) > 400 else text)
    else:
        print("❌ Failed to retrieve patterns")

def main():
    print("🧪 Testing RIF Knowledge MCP Server")
    print("="*60)
    
    # First test initialization
    init_request = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {}
    }
    
    response = send_request(init_request)
    if response and 'result' in response:
        server_info = response['result'].get('serverInfo', {})
        print(f"✅ Server initialized: {server_info.get('name')} v{server_info.get('version')}")
        print(f"   Description: {server_info.get('description')}")
    else:
        print("❌ Server initialization failed")
        return 1
    
    # Run all tests
    test_claude_documentation()
    test_general_queries()
    test_compatibility_check()
    test_pattern_retrieval()
    
    print("\n" + "="*60)
    print("🎉 Test suite complete!")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())