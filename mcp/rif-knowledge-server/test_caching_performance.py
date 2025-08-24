#!/usr/bin/env python3
"""
Test caching performance improvements
"""

import json
import subprocess
import time
import sys

def query_server(tool, arguments):
    """Query the MCP server using echo approach"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": arguments
        }
    }
    
    cmd = f'echo \'{json.dumps(request)}\' | python3 /Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py'
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd='/Users/cal/DEV/RIF'
    )
    
    stdout, stderr = process.communicate(timeout=10)
    
    try:
        response = json.loads(stdout.strip())
        if 'result' in response:
            return response['result']['content'][0]['text']
        else:
            return f"ERROR: {response}"
    except Exception as e:
        return f"FAILED: {stdout[:100]} | {str(e)}"

def performance_test():
    """Test caching performance"""
    print("=" * 70)
    print("TESTING CACHING PERFORMANCE")
    print("=" * 70)
    
    # Test queries - same queries repeated to test caching
    test_queries = [
        ("query_knowledge", {"query": "orchestration"}),
        ("query_knowledge", {"query": "GitHub"}),
        ("query_knowledge", {"query": "error"}),
        ("query_knowledge", {"query": "monitoring"}),
        ("query_knowledge", {"query": "pattern"}),
    ]
    
    print("\nFirst run (no cache):")
    first_run_times = []
    for i, (tool, args) in enumerate(test_queries):
        start_time = time.time()
        result = query_server(tool, args)
        end_time = time.time()
        elapsed = end_time - start_time
        first_run_times.append(elapsed)
        
        # Check if we got results
        has_results = "No results found" not in result and len(result) > 100
        print(f"  Query {i+1}: {elapsed:.3f}s - {'✅' if has_results else '❌'}")
    
    print("\nSecond run (with cache):")
    second_run_times = []
    for i, (tool, args) in enumerate(test_queries):
        start_time = time.time()
        result = query_server(tool, args)
        end_time = time.time()
        elapsed = end_time - start_time
        second_run_times.append(elapsed)
        
        # Check if we got results
        has_results = "No results found" not in result and len(result) > 100
        print(f"  Query {i+1}: {elapsed:.3f}s - {'✅' if has_results else '❌'}")
    
    # Calculate performance improvement
    avg_first = sum(first_run_times) / len(first_run_times)
    avg_second = sum(second_run_times) / len(second_run_times)
    improvement = (avg_first - avg_second) / avg_first * 100 if avg_first > 0 else 0
    
    print(f"\n📊 Performance Results:")
    print(f"   Average first run: {avg_first:.3f}s")
    print(f"   Average second run: {avg_second:.3f}s")
    print(f"   Performance improvement: {improvement:.1f}%")
    
    if improvement > 10:
        print("✅ CACHING WORKING - Significant performance improvement")
        return True
    else:
        print("⚠️ CACHING MAY NOT BE WORKING - Minimal improvement")
        return False

def rapid_requests_test():
    """Test handling many rapid requests"""
    print("\n" + "=" * 70)
    print("RAPID REQUESTS TEST")
    print("=" * 70)
    
    # Test 20 rapid requests
    successful = 0
    total_time = 0
    
    start_test = time.time()
    
    for i in range(20):
        try:
            start_time = time.time()
            result = query_server("query_knowledge", {"query": "orchestration"})
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_time += elapsed
            
            if "No results found" not in result and len(result) > 100:
                successful += 1
                print(f"  Request {i+1}: {elapsed:.3f}s ✅")
            else:
                print(f"  Request {i+1}: {elapsed:.3f}s ❌")
                
        except Exception as e:
            print(f"  Request {i+1}: FAILED - {e}")
    
    end_test = time.time()
    total_test_time = end_test - start_test
    
    print(f"\n📊 Rapid Requests Results:")
    print(f"   Successful requests: {successful}/20")
    print(f"   Average request time: {total_time/20:.3f}s")
    print(f"   Total test time: {total_test_time:.3f}s")
    print(f"   Requests per second: {20/total_test_time:.2f}")
    
    success_rate = successful / 20 * 100
    if success_rate >= 90:
        print(f"✅ RAPID REQUESTS PASSED - {success_rate:.0f}% success rate")
        return True
    else:
        print(f"❌ RAPID REQUESTS FAILED - Only {success_rate:.0f}% success rate")
        return False

if __name__ == '__main__':
    cache_working = performance_test()
    rapid_working = rapid_requests_test()
    
    if cache_working and rapid_working:
        print("\n🎉 ALL PERFORMANCE TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME PERFORMANCE TESTS FAILED")
        sys.exit(1)