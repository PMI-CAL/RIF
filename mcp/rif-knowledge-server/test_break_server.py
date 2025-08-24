#!/usr/bin/env python3
"""
AGGRESSIVE TEST SUITE - Try to BREAK the RIF Knowledge MCP Server
Tests edge cases, malformed input, SQL injection, performance issues
"""

import json
import subprocess
import sys
import time
import random
import string

def send_request(request, timeout=5):
    """Send request to MCP server"""
    try:
        process = subprocess.Popen(
            ["python3", "/Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=json.dumps(request), timeout=timeout)
        
        if stdout:
            return json.loads(stdout.strip())
        else:
            return {"error": "No response", "stderr": stderr}
    except subprocess.TimeoutExpired:
        process.kill()
        return {"error": "Timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "stdout": stdout}
    except Exception as e:
        return {"error": str(e)}

def test_sql_injection():
    """Try SQL injection attacks"""
    print("\n" + "="*60)
    print("🔨 SQL INJECTION TESTS")
    print("="*60)
    
    injections = [
        "'; DROP TABLE entities; --",
        "' OR '1'='1",
        "'; SELECT * FROM entities WHERE '1'='1'; --",
        "\\'; DROP TABLE entities; --",
        "' UNION SELECT * FROM entities --",
        "'; DELETE FROM entities WHERE id IS NOT NULL; --",
        "${1+1}",
        "$(echo hacked)",
        "`rm -rf /`"
    ]
    
    for injection in injections:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge",
                "arguments": {"query": injection}
            }
        }
        
        response = send_request(request)
        
        # Check if injection caused damage
        if 'error' in response and 'DROP' in str(response):
            print(f"❌ VULNERABLE TO SQL INJECTION: {injection[:30]}...")
        elif 'result' in response:
            # Verify database still works
            test_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_patterns",
                    "arguments": {}
                }
            }
            verify = send_request(test_req)
            if 'error' in verify:
                print(f"❌ DATABASE CORRUPTED BY: {injection[:30]}...")
            else:
                print(f"✅ Survived injection: {injection[:30]}...")
        else:
            print(f"✅ Blocked injection: {injection[:30]}...")

def test_malformed_requests():
    """Send malformed/invalid requests"""
    print("\n" + "="*60)
    print("🔨 MALFORMED REQUEST TESTS")
    print("="*60)
    
    malformed = [
        {},  # Empty request
        {"method": "tools/call"},  # Missing jsonrpc
        {"jsonrpc": "2.0", "method": "tools/call", "params": None},  # Null params
        {"jsonrpc": "2.0", "method": "INVALID_METHOD"},  # Invalid method
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "FAKE_TOOL"}},  # Fake tool
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "query_knowledge"}},  # Missing args
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"arguments": {"query": "test"}}},  # Missing name
        {"jsonrpc": "1.0", "method": "initialize"},  # Wrong version
        "NOT JSON AT ALL",  # Not even JSON
        {"jsonrpc": "2.0", "id": "NOT_A_NUMBER", "method": "initialize"},  # Invalid ID type
    ]
    
    for i, req in enumerate(malformed):
        if isinstance(req, str):
            # Send raw string
            try:
                process = subprocess.Popen(
                    ["python3", "/Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=req, timeout=2)
                if stdout and 'error' in stdout:
                    print(f"✅ Handled malformed #{i}: Returns error")
                else:
                    print(f"❌ Malformed #{i} crashed: {stdout[:50]}")
            except:
                print(f"❌ Malformed #{i} caused exception")
        else:
            response = send_request(req, timeout=2)
            if 'error' in response or (isinstance(response, dict) and response.get('error')):
                print(f"✅ Handled malformed #{i}: {str(req)[:30]}...")
            else:
                print(f"❌ Accepted invalid #{i}: {str(req)[:30]}...")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n" + "="*60)
    print("🔨 EDGE CASE TESTS")
    print("="*60)
    
    # Test 1: Empty strings
    response = send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": ""}}
    })
    print(f"Empty query: {'✅ Handled' if 'result' in response else '❌ Failed'}")
    
    # Test 2: Very long query
    long_query = "a" * 10000
    response = send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": long_query}}
    })
    print(f"10K char query: {'✅ Handled' if 'result' in response else '❌ Failed'}")
    
    # Test 3: Unicode and special chars
    unicode_query = "🔥💀 τεστ テスト 测试 \\n\\r\\t\\0"
    response = send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": unicode_query}}
    })
    print(f"Unicode query: {'✅ Handled' if 'result' in response else '❌ Failed'}")
    
    # Test 4: Negative limit
    response = send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": "test", "limit": -1}}
    })
    print(f"Negative limit: {'✅ Handled' if 'result' in response or 'error' in response else '❌ Crashed'}")
    
    # Test 5: Huge limit
    response = send_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": "test", "limit": 999999}}
    })
    print(f"Huge limit: {'✅ Handled' if 'result' in response else '❌ Failed'}")
    
    # Test 6: Invalid entity types
    response = send_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": "test", "entity_types": ["FAKE_TYPE", 123, None]}}
    })
    print(f"Invalid types: {'✅ Handled' if 'result' in response or 'error' in response else '❌ Crashed'}")

def test_performance():
    """Test performance under load"""
    print("\n" + "="*60)
    print("🔨 PERFORMANCE TESTS")
    print("="*60)
    
    # Test 1: Rapid fire requests
    print("Sending 20 rapid requests...")
    start = time.time()
    successes = 0
    for i in range(20):
        response = send_request({
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": "get_patterns", "arguments": {}}
        }, timeout=1)
        if 'result' in response:
            successes += 1
    
    elapsed = time.time() - start
    print(f"Rapid fire: {successes}/20 succeeded in {elapsed:.2f}s")
    if successes < 18:
        print("❌ Poor performance under load")
    else:
        print("✅ Good performance")
    
    # Test 2: Large result set
    response = send_request({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "query_knowledge", "arguments": {"query": "a", "limit": 100}}
    }, timeout=10)
    if 'result' in response:
        text = response['result']['content'][0]['text']
        print(f"✅ Large result handled: {len(text)} chars")
    else:
        print("❌ Failed on large result set")

def test_real_world_queries():
    """Test with actual real-world questions"""
    print("\n" + "="*60)
    print("🔨 REAL-WORLD USEFULNESS TESTS")
    print("="*60)
    
    queries = [
        ("How do I run multiple agents in parallel?", "check_compatibility", 
         {"approach": "Running multiple Task() calls in parallel"}),
        ("Can Claude Code monitor external systems?", "get_claude_documentation",
         {"topic": "monitoring capabilities"}),
        ("What patterns exist for error handling?", "query_knowledge",
         {"query": "error handling", "entity_types": ["pattern"]}),
        ("Is Task.parallel() a real function?", "check_compatibility",
         {"approach": "Using Task.parallel() for orchestration"}),
        ("How do agents communicate?", "get_claude_documentation",
         {"topic": "agent communication"}),
    ]
    
    useful_count = 0
    for question, tool, args in queries:
        print(f"\nQ: {question}")
        response = send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args}
        })
        
        if 'result' in response:
            text = response['result']['content'][0]['text']
            # Check if response is actually useful
            if len(text) > 50 and not text.startswith("No results"):
                print(f"✅ Useful response: {text[:150]}...")
                useful_count += 1
                
                # Verify accuracy for known facts
                if "Task.parallel()" in question and "pseudocode" in text:
                    print("   ✅ Correctly identifies Task.parallel() as pseudocode")
                elif "monitor external" in question and ("limitation" in text.lower() or "cannot" in text.lower()):
                    print("   ✅ Correctly identifies monitoring limitations")
            else:
                print(f"❌ Not useful: {text[:100]}")
        else:
            print(f"❌ Failed to answer: {response}")
    
    print(f"\nUsefulness score: {useful_count}/{len(queries)}")
    if useful_count < 3:
        print("❌ SERVER IS NOT USEFUL FOR REAL QUERIES")

def test_concurrent_access():
    """Test concurrent access issues"""
    print("\n" + "="*60)
    print("🔨 CONCURRENT ACCESS TESTS")
    print("="*60)
    
    import threading
    results = []
    
    def make_request(thread_id):
        response = send_request({
            "jsonrpc": "2.0",
            "id": thread_id,
            "method": "tools/call",
            "params": {"name": "query_knowledge", "arguments": {"query": f"thread{thread_id}"}}
        })
        results.append('result' in response)
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    success_count = sum(results)
    print(f"Concurrent requests: {success_count}/10 succeeded")
    if success_count < 8:
        print("❌ Poor concurrent handling")
    else:
        print("✅ Good concurrent handling")

def main():
    print("🔥 AGGRESSIVE MCP SERVER TESTING - TRYING TO BREAK IT")
    print("="*60)
    
    # Check server is running first
    init_response = send_request({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {}
    })
    
    if 'result' not in init_response:
        print("❌ Server not responding to initialization")
        return 1
    
    print("✅ Server initialized, beginning torture tests...\n")
    
    # Run all test suites
    test_sql_injection()
    test_malformed_requests()
    test_edge_cases()
    test_performance()
    test_real_world_queries()
    test_concurrent_access()
    
    # Final database integrity check
    print("\n" + "="*60)
    print("FINAL DATABASE INTEGRITY CHECK")
    print("="*60)
    
    final_check = send_request({
        "jsonrpc": "2.0",
        "id": 999,
        "method": "tools/call",
        "params": {"name": "get_patterns", "arguments": {}}
    })
    
    if 'result' in final_check:
        print("✅ Database still functional after all tests")
    else:
        print("❌ DATABASE CORRUPTED - Server is broken!")
        
    print("\n" + "="*60)
    print("TEST COMPLETE - Check results above for failures")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())