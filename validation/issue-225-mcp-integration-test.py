#!/usr/bin/env python3
"""
Adversarial Validation Test for Issue #225 - MCP Server Integration Fix

This test performs independent verification of the claimed MCP server fixes:
1. MCP Protocol Compliance 
2. Tool Functionality
3. Performance Validation
4. Error Handling
5. Integration Testing

Run independently to validate implementation claims.
"""

import json
import subprocess
import sys
import time
import asyncio
from typing import Dict, Any, List
import tempfile
import os


class MCPServerTester:
    """Independent tester for MCP Server functionality."""
    
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.test_results = []
        self.server_process = None
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite and return results."""
        print("🔍 Starting Adversarial Validation of Issue #225 MCP Server Fix")
        print(f"Testing server at: {self.server_path}")
        print("-" * 70)
        
        results = {
            "overall_status": "unknown",
            "timestamp": time.time(),
            "server_path": self.server_path,
            "tests_run": [],
            "failures": [],
            "performance_metrics": {}
        }
        
        # Test 1: MCP Protocol Compliance
        print("\n📋 Test 1: MCP Protocol Compliance")
        protocol_result = self.test_mcp_protocol_compliance()
        results["tests_run"].append(protocol_result)
        if not protocol_result["passed"]:
            results["failures"].append(protocol_result)
        
        # Test 2: Tool Functionality 
        print("\n🔧 Test 2: Tool Functionality")
        tools_result = self.test_tool_functionality()
        results["tests_run"].append(tools_result)
        if not tools_result["passed"]:
            results["failures"].append(tools_result)
        
        # Test 3: Performance Validation
        print("\n⚡ Test 3: Performance Validation")
        performance_result = self.test_performance()
        results["tests_run"].append(performance_result)
        results["performance_metrics"] = performance_result.get("metrics", {})
        if not performance_result["passed"]:
            results["failures"].append(performance_result)
        
        # Test 4: Error Handling
        print("\n🚨 Test 4: Error Handling")
        error_result = self.test_error_handling()
        results["tests_run"].append(error_result)
        if not error_result["passed"]:
            results["failures"].append(error_result)
        
        # Overall assessment
        results["overall_status"] = "PASSED" if len(results["failures"]) == 0 else "FAILED"
        
        return results
    
    def test_mcp_protocol_compliance(self) -> Dict[str, Any]:
        """Test MCP protocol compliance."""
        test_result = {
            "test_name": "MCP Protocol Compliance",
            "passed": False,
            "details": [],
            "evidence": []
        }
        
        try:
            # Test initialize method
            print("  Testing initialize method...")
            init_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            })
            
            if init_response and init_response.get("result"):
                test_result["details"].append("✅ initialize method responds correctly")
                test_result["evidence"].append(f"Initialize response: {json.dumps(init_response, indent=2)}")
            else:
                test_result["details"].append("❌ initialize method failed")
                return test_result
            
            # Test tools/list method
            print("  Testing tools/list method...")
            tools_response = self.send_mcp_request({
                "jsonrpc": "2.0", 
                "id": 2,
                "method": "tools/list",
                "params": {}
            })
            
            if tools_response and tools_response.get("result", {}).get("tools"):
                tools = tools_response["result"]["tools"]
                test_result["details"].append(f"✅ tools/list returns {len(tools)} tools")
                test_result["evidence"].append(f"Tools: {[t['name'] for t in tools]}")
                
                # Verify expected tools are present
                expected_tools = ["check_compatibility", "recommend_pattern", "find_alternatives", "validate_architecture", "query_limitations"]
                actual_tools = [t['name'] for t in tools]
                missing_tools = set(expected_tools) - set(actual_tools)
                
                if missing_tools:
                    test_result["details"].append(f"❌ Missing expected tools: {missing_tools}")
                    return test_result
                else:
                    test_result["details"].append("✅ All 5 sophisticated tools present")
            else:
                test_result["details"].append("❌ tools/list method failed")
                return test_result
            
            # Test JSON-RPC 2.0 compliance
            print("  Testing JSON-RPC 2.0 compliance...")
            if (init_response.get("jsonrpc") == "2.0" and 
                tools_response.get("jsonrpc") == "2.0"):
                test_result["details"].append("✅ JSON-RPC 2.0 format compliance verified")
            else:
                test_result["details"].append("❌ JSON-RPC 2.0 format compliance failed")
                return test_result
            
            test_result["passed"] = True
            
        except Exception as e:
            test_result["details"].append(f"❌ Protocol test failed with exception: {e}")
        
        return test_result
    
    def test_tool_functionality(self) -> Dict[str, Any]:
        """Test individual tool functionality."""
        test_result = {
            "test_name": "Tool Functionality", 
            "passed": False,
            "details": [],
            "evidence": [],
            "tool_results": {}
        }
        
        try:
            # Test check_compatibility tool
            print("  Testing check_compatibility tool...")
            compat_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "check_compatibility",
                    "arguments": {
                        "issue_description": "Implement Task() orchestration for parallel agent execution",
                        "approach": "Use Task() to launch multiple agents in parallel"
                    }
                }
            })
            
            if compat_response and compat_response.get("result"):
                # Parse the tool result
                content = compat_response["result"]["content"][0]["text"]
                tool_result = json.loads(content)
                test_result["tool_results"]["check_compatibility"] = tool_result
                
                if "compatible" in tool_result:
                    test_result["details"].append("✅ check_compatibility tool functional")
                    test_result["evidence"].append(f"Compatibility result: {tool_result.get('compatible')}")
                else:
                    test_result["details"].append("❌ check_compatibility tool returned invalid format")
                    return test_result
            else:
                test_result["details"].append("❌ check_compatibility tool failed")
                return test_result
            
            # Test validate_architecture tool  
            print("  Testing validate_architecture tool...")
            arch_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "validate_architecture",
                    "arguments": {
                        "system_design": "Microservices architecture with queue-based orchestrator and multiple agents"
                    }
                }
            })
            
            if arch_response and arch_response.get("result"):
                content = arch_response["result"]["content"][0]["text"]
                tool_result = json.loads(content)
                test_result["tool_results"]["validate_architecture"] = tool_result
                
                if "valid" in tool_result and "components_analyzed" in tool_result:
                    test_result["details"].append("✅ validate_architecture tool functional")
                    test_result["evidence"].append(f"Components analyzed: {tool_result.get('components_analyzed')}")
                else:
                    test_result["details"].append("❌ validate_architecture tool returned invalid format")
                    return test_result
            else:
                test_result["details"].append("❌ validate_architecture tool failed")
                return test_result
            
            # Test query_limitations tool
            print("  Testing query_limitations tool...")
            limit_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": 5, 
                "method": "tools/call",
                "params": {
                    "name": "query_limitations",
                    "arguments": {
                        "capability_area": "orchestration"
                    }
                }
            })
            
            if limit_response and limit_response.get("result"):
                content = limit_response["result"]["content"][0]["text"]
                tool_result = json.loads(content)
                test_result["tool_results"]["query_limitations"] = tool_result
                
                if "limitations" in tool_result:
                    test_result["details"].append("✅ query_limitations tool functional")
                    test_result["evidence"].append(f"Limitations found: {len(tool_result.get('limitations', []))}")
                else:
                    test_result["details"].append("❌ query_limitations tool returned invalid format")
                    return test_result
            else:
                test_result["details"].append("❌ query_limitations tool failed")
                return test_result
                
            test_result["passed"] = True
            
        except Exception as e:
            test_result["details"].append(f"❌ Tool functionality test failed: {e}")
        
        return test_result
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance claims."""
        test_result = {
            "test_name": "Performance Validation",
            "passed": False,
            "details": [],
            "evidence": [],
            "metrics": {}
        }
        
        try:
            # Performance target: <200ms (claimed: 2-5ms typical)
            print("  Testing response time performance...")
            
            response_times = []
            for i in range(5):
                start_time = time.time()
                
                response = self.send_mcp_request({
                    "jsonrpc": "2.0",
                    "id": f"perf_{i}",
                    "method": "tools/call",
                    "params": {
                        "name": "check_compatibility",
                        "arguments": {
                            "issue_description": "Simple test for performance measurement",
                            "approach": "Standard approach"
                        }
                    }
                })
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response and response.get("result"):
                    print(f"    Request {i+1}: {response_time_ms:.1f}ms")
                else:
                    test_result["details"].append(f"❌ Performance test request {i+1} failed")
                    return test_result
            
            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            test_result["metrics"] = {
                "average_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "min_response_time_ms": min_response_time,
                "target_response_time_ms": 200,
                "claimed_typical_range_ms": [2, 5]
            }
            
            # Validate against requirements
            if avg_response_time < 200:
                test_result["details"].append(f"✅ Average response time: {avg_response_time:.1f}ms (target: <200ms)")
            else:
                test_result["details"].append(f"❌ Average response time: {avg_response_time:.1f}ms exceeds 200ms target")
                return test_result
            
            # Validate against claims
            if avg_response_time <= 10:  # Within reasonable range of claimed 2-5ms
                test_result["details"].append(f"✅ Performance claims verified (avg: {avg_response_time:.1f}ms)")
            else:
                test_result["details"].append(f"⚠️ Performance slower than claimed but meets requirements")
            
            test_result["evidence"].append(f"Response times: {[f'{t:.1f}ms' for t in response_times]}")
            test_result["passed"] = True
            
        except Exception as e:
            test_result["details"].append(f"❌ Performance test failed: {e}")
        
        return test_result
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling capabilities."""
        test_result = {
            "test_name": "Error Handling",
            "passed": False,
            "details": [],
            "evidence": []
        }
        
        try:
            # Test invalid method
            print("  Testing invalid method handling...")
            invalid_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": "error_1",
                "method": "invalid_method",
                "params": {}
            })
            
            if (invalid_response and invalid_response.get("error") and 
                invalid_response["error"]["code"] == -32601):
                test_result["details"].append("✅ Invalid method error handled correctly")
                test_result["evidence"].append(f"Error response: {invalid_response['error']['message']}")
            else:
                test_result["details"].append("❌ Invalid method error handling failed")
                return test_result
            
            # Test invalid tool name
            print("  Testing invalid tool handling...")
            invalid_tool_response = self.send_mcp_request({
                "jsonrpc": "2.0",
                "id": "error_2", 
                "method": "tools/call",
                "params": {
                    "name": "invalid_tool",
                    "arguments": {}
                }
            })
            
            if (invalid_tool_response and invalid_tool_response.get("error") and
                invalid_tool_response["error"]["code"] == -32601):
                test_result["details"].append("✅ Invalid tool error handled correctly")
                test_result["evidence"].append(f"Available tools provided: {invalid_tool_response['error'].get('data', {}).get('available_tools', [])}")
            else:
                test_result["details"].append("❌ Invalid tool error handling failed")
                return test_result
            
            # Test malformed JSON
            print("  Testing malformed JSON handling...")
            malformed_response = self.send_raw_request("invalid json content")
            
            if (malformed_response and malformed_response.get("error") and
                malformed_response["error"]["code"] == -32700):
                test_result["details"].append("✅ Malformed JSON error handled correctly")
            else:
                test_result["details"].append("❌ Malformed JSON error handling failed")
                return test_result
            
            test_result["passed"] = True
            
        except Exception as e:
            test_result["details"].append(f"❌ Error handling test failed: {e}")
        
        return test_result
    
    def send_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send an MCP request and return response."""
        request_json = json.dumps(request) + "\n"
        return self.send_raw_request(request_json)
    
    def send_raw_request(self, request_data: str) -> Dict[str, Any]:
        """Send raw request data and return parsed response."""
        try:
            # Start server process if not running
            if not self.server_process:
                self.server_process = subprocess.Popen(
                    ["python3", self.server_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={
                        **os.environ,
                        "PYTHONPATH": "/Users/cal/DEV/RIF",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONWARNINGS": "ignore"
                    }
                )
            
            # Send request
            self.server_process.stdin.write(request_data)
            self.server_process.stdin.flush()
            
            # Read response
            response_line = self.server_process.stdout.readline()
            
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response received"}
                
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def cleanup(self):
        """Cleanup test resources."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()


def main():
    """Main test execution."""
    server_path = "/Users/cal/DEV/RIF/mcp/claude-code-knowledge/server.py"
    
    if not os.path.exists(server_path):
        print(f"❌ Server not found at: {server_path}")
        sys.exit(1)
    
    tester = MCPServerTester(server_path)
    
    try:
        results = tester.run_test_suite()
        
        # Print results summary
        print("\n" + "=" * 70)
        print("🔍 ADVERSARIAL VALIDATION RESULTS")
        print("=" * 70)
        
        print(f"Overall Status: {'✅ PASSED' if results['overall_status'] == 'PASSED' else '❌ FAILED'}")
        print(f"Tests Run: {len(results['tests_run'])}")
        print(f"Failures: {len(results['failures'])}")
        
        # Detailed results
        for test in results["tests_run"]:
            status = "✅ PASSED" if test["passed"] else "❌ FAILED" 
            print(f"\n{test['test_name']}: {status}")
            for detail in test["details"]:
                print(f"  {detail}")
        
        # Performance metrics
        if results["performance_metrics"]:
            print(f"\nPerformance Metrics:")
            metrics = results["performance_metrics"]
            print(f"  Average: {metrics.get('average_response_time_ms', 0):.1f}ms")
            print(f"  Target: <{metrics.get('target_response_time_ms', 200)}ms")
            print(f"  Claimed: {metrics.get('claimed_typical_range_ms', [])}")
        
        # Save detailed results
        results_file = "/tmp/issue-225-validation-results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        return 0 if results["overall_status"] == "PASSED" else 1
        
    finally:
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())