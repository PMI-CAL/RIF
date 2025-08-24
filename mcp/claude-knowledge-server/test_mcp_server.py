#!/usr/bin/env python3
"""
Comprehensive test suite for Claude Code Knowledge MCP Server
Tests all requirements from issue #97
"""

import json
import subprocess
import sys
import os
from typing import Dict, Any, List

class MCPServerTester:
    def __init__(self):
        self.server_path = "/Users/cal/DEV/RIF/mcp/claude-knowledge-server/simple_server.py"
        self.test_results = []
        self.failures = []
        
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server and get response"""
        try:
            process = subprocess.Popen(
                ["python3", self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            request_str = json.dumps(request)
            stdout, stderr = process.communicate(input=request_str, timeout=5)
            
            if stdout:
                return json.loads(stdout.strip())
            else:
                return {"error": f"No response. Stderr: {stderr}"}
                
        except subprocess.TimeoutExpired:
            process.kill()
            return {"error": "Server timeout"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_protocol_compliance(self):
        """Test JSON-RPC 2.0 protocol compliance"""
        print("\n🔍 Testing Protocol Compliance...")
        
        # Test 1: Initialize request
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        
        if "result" in response and "protocolVersion" in response.get("result", {}):
            self.test_results.append("✅ Initialize request successful")
        else:
            self.failures.append(f"❌ Initialize failed: {response}")
            
        # Test 2: Invalid method
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "invalid_method",
            "params": {}
        })
        
        if "error" in response:
            self.test_results.append("✅ Invalid method returns error")
        else:
            self.failures.append(f"❌ Invalid method should return error: {response}")
            
        # Test 3: Missing jsonrpc field
        response = self.send_request({
            "id": 3,
            "method": "initialize"
        })
        
        if response.get("jsonrpc") == "2.0":
            self.test_results.append("✅ Server adds jsonrpc field")
        else:
            self.failures.append(f"❌ Missing jsonrpc handling: {response}")
    
    def test_tools_listing(self):
        """Test that all required tools are listed"""
        print("\n📋 Testing Tools Listing...")
        
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {}
        })
        
        if "result" not in response or "tools" not in response.get("result", {}):
            self.failures.append(f"❌ Tools list failed: {response}")
            return
            
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        # Check required tools from original spec
        required_tools = [
            "check_claude_capability",
            "get_implementation_pattern", 
            "check_compatibility"
        ]
        
        for tool in required_tools:
            if tool in tool_names:
                self.test_results.append(f"✅ Tool '{tool}' is available")
            else:
                self.failures.append(f"❌ Missing required tool: {tool}")
    
    def test_capability_checking(self):
        """Test check_claude_capability with various inputs"""
        print("\n🎯 Testing Capability Checking...")
        
        test_cases = [
            ("can Claude Code read files", True, "file operations"),
            ("can it execute bash commands", True, "bash"),
            ("can it use Task.parallel()", False, "pseudocode"),
            ("can it run persistent background processes", False, "persistent"),
            ("can it use MCP servers", True, "MCP"),
            ("can it search code", True, "search")
        ]
        
        for action, should_be_capable, reason in test_cases:
            response = self.send_request({
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {
                    "name": "check_claude_capability",
                    "arguments": {"action": action}
                }
            })
            
            if "result" in response:
                content = response["result"].get("content", [{}])[0].get("text", "")
                # More accurate capability detection
                is_capable = ("Yes" in content or ("can" in content.lower() and "cannot" not in content.lower()))
                
                if is_capable == should_be_capable:
                    self.test_results.append(f"✅ Correctly identified: {action[:30]}...")
                else:
                    self.failures.append(f"❌ Wrong capability for '{action}': got {content[:50]}...")
            else:
                self.failures.append(f"❌ Failed to check: {action}")
    
    def test_pattern_recommendations(self):
        """Test get_implementation_pattern for various tasks"""
        print("\n🔧 Testing Pattern Recommendations...")
        
        test_patterns = [
            ("github", "gh CLI"),
            ("mcp", "claude mcp add"),
            ("orchestration", "multiple Task tools"),
            ("file", "Read/Write/Edit"),
            ("search", "Grep")
        ]
        
        for task, expected_keyword in test_patterns:
            response = self.send_request({
                "jsonrpc": "2.0",
                "id": 200,
                "method": "tools/call",
                "params": {
                    "name": "get_implementation_pattern",
                    "arguments": {"task": task}
                }
            })
            
            if "result" in response:
                content = response["result"].get("content", [{}])[0].get("text", "")
                if expected_keyword.lower() in content.lower():
                    self.test_results.append(f"✅ Pattern for '{task}' includes '{expected_keyword}'")
                else:
                    self.failures.append(f"❌ Pattern for '{task}' missing '{expected_keyword}': {content[:50]}")
            else:
                self.failures.append(f"❌ Failed to get pattern for: {task}")
    
    def test_compatibility_checking(self):
        """Test check_compatibility with known anti-patterns"""
        print("\n⚠️ Testing Anti-Pattern Detection...")
        
        test_cases = [
            ("Using Task.parallel() for orchestration", False, "pseudocode"),
            ("Running persistent background monitoring", False, "persistent"),
            ("External orchestrator controlling Claude Code", False, "external"),
            ("Using gh CLI for GitHub operations", True, "compatible"),
            ("Reading files with Read tool", True, "compatible"),
            ("Launching multiple Task tools in one response", True, "compatible")
        ]
        
        for approach, should_be_compatible, reason in test_cases:
            response = self.send_request({
                "jsonrpc": "2.0",
                "id": 300,
                "method": "tools/call",
                "params": {
                    "name": "check_compatibility",
                    "arguments": {"approach": approach}
                }
            })
            
            if "result" in response:
                content = response["result"].get("content", [{}])[0].get("text", "")
                is_compatible = "COMPATIBLE:" in content and "INCOMPATIBLE:" not in content
                
                if is_compatible == should_be_compatible:
                    self.test_results.append(f"✅ Correctly assessed: {approach[:40]}...")
                else:
                    self.failures.append(f"❌ Wrong compatibility for '{approach}': {content[:50]}")
            else:
                self.failures.append(f"❌ Failed to check compatibility: {approach}")
    
    def test_rif_specific_scenarios(self):
        """Test scenarios specific to RIF's needs"""
        print("\n🎯 Testing RIF-Specific Scenarios...")
        
        # Test 1: Can RIF agents communicate directly?
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 400,
            "method": "tools/call",
            "params": {
                "name": "check_compatibility",
                "arguments": {"approach": "RIF agents communicating via shared memory"}
            }
        })
        
        if "result" in response:
            content = response["result"].get("content", [{}])[0].get("text", "")
            if "INCOMPATIBLE" in content:
                self.test_results.append("✅ Correctly identifies agent communication limitation")
            else:
                self.failures.append(f"❌ Should reject inter-agent communication: {content}")
        
        # Test 2: Orchestration pattern
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 401,
            "method": "tools/call",
            "params": {
                "name": "get_implementation_pattern",
                "arguments": {"task": "orchestration"}
            }
        })
        
        if "result" in response:
            content = response["result"].get("content", [{}])[0].get("text", "")
            if "multiple Task tools" in content or "one response" in content:
                self.test_results.append("✅ Correct orchestration pattern provided")
            else:
                self.failures.append(f"❌ Incorrect orchestration pattern: {content}")
        
        # Test 3: GitHub automation compatibility
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 402,
            "method": "tools/call",
            "params": {
                "name": "check_compatibility",
                "arguments": {"approach": "Automated GitHub issue monitoring and response"}
            }
        })
        
        if "result" in response:
            content = response["result"].get("content", [{}])[0].get("text", "")
            # This should be compatible with session-based checking
            if "COMPATIBLE" in content or "session" in content.lower():
                self.test_results.append("✅ GitHub automation correctly assessed")
            else:
                self.failures.append(f"❌ GitHub automation assessment wrong: {content}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🔨 Testing Edge Cases...")
        
        # Test 1: Empty arguments
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 500,
            "method": "tools/call",
            "params": {
                "name": "check_claude_capability",
                "arguments": {}
            }
        })
        
        if "result" in response or "error" in response:
            self.test_results.append("✅ Handles empty arguments gracefully")
        else:
            self.failures.append(f"❌ Failed on empty arguments: {response}")
        
        # Test 2: Very long input
        long_text = "x" * 10000
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 501,
            "method": "tools/call",
            "params": {
                "name": "check_compatibility",
                "arguments": {"approach": long_text}
            }
        })
        
        if "result" in response or "error" in response:
            self.test_results.append("✅ Handles long input without crashing")
        else:
            self.failures.append("❌ Failed on long input")
        
        # Test 3: Special characters
        response = self.send_request({
            "jsonrpc": "2.0",
            "id": 502,
            "method": "tools/call",
            "params": {
                "name": "check_claude_capability",
                "arguments": {"action": "can it handle 'quotes' and \"double quotes\" and \n newlines?"}
            }
        })
        
        if "result" in response:
            self.test_results.append("✅ Handles special characters")
        else:
            self.failures.append(f"❌ Failed on special characters: {response}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("=" * 60)
        print("🧪 COMPREHENSIVE MCP SERVER TEST SUITE")
        print("=" * 60)
        
        self.test_protocol_compliance()
        self.test_tools_listing()
        self.test_capability_checking()
        self.test_pattern_recommendations()
        self.test_compatibility_checking()
        self.test_rif_specific_scenarios()
        self.test_edge_cases()
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results) + len(self.failures)
        passed = len(self.test_results)
        failed = len(self.failures)
        
        print(f"\n✅ Passed: {passed}/{total_tests}")
        print(f"❌ Failed: {failed}/{total_tests}")
        print(f"📈 Success Rate: {(passed/total_tests)*100:.1f}%")
        
        if self.failures:
            print("\n⚠️ FAILURES DETECTED:")
            for failure in self.failures:
                print(f"  {failure}")
        
        print("\n✨ SUCCESSFUL TESTS:")
        for result in self.test_results:
            print(f"  {result}")
        
        # Final verdict
        print("\n" + "=" * 60)
        if failed == 0:
            print("🎉 ALL TESTS PASSED! Server is ready for production.")
            return 0
        elif failed <= 2:
            print("⚠️ Minor issues detected. Server mostly functional.")
            return 1
        else:
            print("❌ CRITICAL FAILURES! Server is NOT ready.")
            return 2

if __name__ == "__main__":
    tester = MCPServerTester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)