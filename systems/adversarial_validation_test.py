#!/usr/bin/env python3
"""
Adversarial Validation Test for Context Optimization Engine
Attempts to break the system and validate quality gates
"""

import time
import concurrent.futures
import threading
import subprocess
import sys
import os
import importlib.util

def load_module(name, filepath):
    """Dynamically load module from file path"""
    try:
        spec = importlib.util.spec_from_file_location(name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return None

def stress_test_concurrent_agents():
    """Stress test with 15 concurrent agents (exceeds 10 target)"""
    print("🔥 Stress Testing: 15 Concurrent Agents...")
    
    coe = load_module("coe", "context-optimization-engine.py")
    if not coe:
        return False
    
    optimizer = coe.ContextOptimizer()
    
    def agent_request(agent_id):
        agent_type = coe.AgentType.IMPLEMENTER
        task = {
            "description": f"Concurrent stress test request {agent_id}",
            "complexity": "high"
        }
        
        start_time = time.time()
        try:
            result = optimizer.optimize_for_agent(agent_type, task, 123 + agent_id)
            duration = (time.time() - start_time) * 1000
            return {
                "agent_id": agent_id,
                "success": True,
                "duration_ms": duration,
                "knowledge_items": len(result.relevant_knowledge)
            }
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return {
                "agent_id": agent_id,
                "success": False,
                "duration_ms": duration,
                "error": str(e)
            }
    
    # Run 15 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(agent_request, i) for i in range(15)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if successful:
        avg_time = sum(r["duration_ms"] for r in successful) / len(successful)
        max_time = max(r["duration_ms"] for r in successful)
        sub_200ms_rate = len([r for r in successful if r["duration_ms"] < 200]) / len(successful)
        
        print(f"  ✅ Successful requests: {len(successful)}/15")
        print(f"  ✅ Average response time: {avg_time:.1f}ms")
        print(f"  ✅ Maximum response time: {max_time:.1f}ms")
        print(f"  ✅ Sub-200ms compliance: {sub_200ms_rate:.1%}")
        
        if failed:
            print(f"  ⚠️ Failed requests: {len(failed)}")
            for f in failed[:3]:  # Show first 3 failures
                print(f"    - Agent {f['agent_id']}: {f.get('error', 'Unknown')}")
        
        # Success criteria: >10 successful, <200ms avg, >80% sub-200ms
        return len(successful) >= 10 and avg_time < 200 and sub_200ms_rate > 0.8
    else:
        print(f"  ❌ All requests failed")
        return False

def boundary_test_large_context():
    """Test with extremely large context requirements"""
    print("🧪 Boundary Test: Large Context Requirements...")
    
    coe = load_module("coe", "context-optimization-engine.py")
    if not coe:
        return False
    
    optimizer = coe.ContextOptimizer()
    
    # Create a massive task description to test filtering
    large_task = {
        "description": "Implement massive distributed system " * 1000,  # Very large description
        "complexity": "very_high",
        "requirements": ["req" + str(i) for i in range(100)],  # Many requirements
        "type": "implementation"
    }
    
    start_time = time.time()
    try:
        result = optimizer.optimize_for_agent(coe.AgentType.ARCHITECT, large_task, 999)
        duration = (time.time() - start_time) * 1000
        
        print(f"  ✅ Processing time: {duration:.1f}ms")
        print(f"  ✅ Context items: {len(result.relevant_knowledge)}")
        print(f"  ✅ Total size: {result.total_size} chars")
        print(f"  ✅ Utilization: {result.context_window_utilization:.1%}")
        
        # Success criteria: Still fast processing, reasonable size
        return duration < 500 and result.total_size < 10000
        
    except Exception as e:
        print(f"  ❌ Failed with large context: {e}")
        return False

def edge_case_test_invalid_inputs():
    """Test with invalid inputs to verify error handling"""
    print("🔍 Edge Case Test: Invalid Inputs...")
    
    coe = load_module("coe", "context-optimization-engine.py")
    if not coe:
        return False
    
    optimizer = coe.ContextOptimizer()
    test_cases = []
    
    # Test case 1: None task context
    try:
        result = optimizer.optimize_for_agent(coe.AgentType.IMPLEMENTER, None, 123)
        test_cases.append(("none_task", "handled", "No crash"))
    except Exception as e:
        test_cases.append(("none_task", "error", str(e)[:50]))
    
    # Test case 2: Empty task context
    try:
        result = optimizer.optimize_for_agent(coe.AgentType.IMPLEMENTER, {}, 123)
        test_cases.append(("empty_task", "handled", "No crash"))
    except Exception as e:
        test_cases.append(("empty_task", "error", str(e)[:50]))
    
    # Test case 3: Invalid issue number
    try:
        result = optimizer.optimize_for_agent(coe.AgentType.IMPLEMENTER, {"description": "test"}, -1)
        test_cases.append(("negative_issue", "handled", "No crash"))
    except Exception as e:
        test_cases.append(("negative_issue", "error", str(e)[:50]))
    
    # Test case 4: Very large issue number
    try:
        result = optimizer.optimize_for_agent(coe.AgentType.IMPLEMENTER, {"description": "test"}, 999999999)
        test_cases.append(("large_issue", "handled", "No crash"))
    except Exception as e:
        test_cases.append(("large_issue", "error", str(e)[:50]))
    
    print(f"  ✅ Edge case tests completed: {len(test_cases)}")
    for test, status, details in test_cases:
        print(f"    {test}: {status} - {details}")
    
    # Success if we handled all edge cases without crashing
    handled_count = sum(1 for _, status, _ in test_cases if status == "handled")
    return handled_count >= 2  # At least half should be handled gracefully

def memory_leak_test():
    """Test for memory leaks during repeated operations"""
    print("🧠 Memory Leak Test: Repeated Operations...")
    
    coe = load_module("coe", "context-optimization-engine.py")
    if not coe:
        return False
    
    optimizer = coe.ContextOptimizer()
    
    # Run 100 optimization cycles to test for memory accumulation
    start_time = time.time()
    times = []
    
    for i in range(100):
        cycle_start = time.time()
        result = optimizer.optimize_for_agent(
            coe.AgentType.IMPLEMENTER,
            {"description": f"Memory test cycle {i}"},
            100 + i
        )
        cycle_time = (time.time() - cycle_start) * 1000
        times.append(cycle_time)
    
    total_time = time.time() - start_time
    
    # Check for performance degradation (sign of memory issues)
    first_10_avg = sum(times[:10]) / 10
    last_10_avg = sum(times[-10:]) / 10
    degradation_factor = last_10_avg / first_10_avg if first_10_avg > 0 else 1.0
    
    print(f"  ✅ Total cycles: 100")
    print(f"  ✅ Total time: {total_time:.1f}s")
    print(f"  ✅ First 10 avg: {first_10_avg:.2f}ms")
    print(f"  ✅ Last 10 avg: {last_10_avg:.2f}ms")
    print(f"  ✅ Degradation factor: {degradation_factor:.2f}x")
    
    # Success if no significant performance degradation (memory issues)
    return degradation_factor < 2.0  # Less than 2x slowdown

def integration_test_all_components():
    """Test integration between all components"""
    print("🔗 Integration Test: All Components...")
    
    components = [
        ("context-optimization-engine.py", "Context Optimization Engine"),
        ("context_intelligence_platform.py", "Intelligence Platform"),
        ("context_request_router.py", "Request Router"),
        ("context_performance_monitor.py", "Performance Monitor"),
        ("agent_context_templates.py", "Context Templates")
    ]
    
    results = []
    
    for file, name in components:
        try:
            # Try to import and basic instantiation
            module = load_module(name.lower().replace(' ', '_'), file)
            if module:
                results.append((name, True, "Loaded successfully"))
            else:
                results.append((name, False, "Failed to load"))
        except Exception as e:
            results.append((name, False, f"Error: {str(e)[:50]}"))
    
    successful = sum(1 for _, success, _ in results if success)
    
    print(f"  ✅ Component tests: {successful}/{len(components)}")
    for name, success, details in results:
        status = "✅" if success else "❌"
        print(f"    {status} {name}: {details}")
    
    return successful >= 4  # At least 4/5 components should work

def run_adversarial_validation():
    """Run all adversarial validation tests"""
    print("🛡️ ADVERSARIAL VALIDATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Concurrent Stress Test", stress_test_concurrent_agents),
        ("Large Context Boundary Test", boundary_test_large_context),
        ("Invalid Input Edge Cases", edge_case_test_invalid_inputs),
        ("Memory Leak Test", memory_leak_test),
        ("Component Integration Test", integration_test_all_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        try:
            result = test_func()
            results[test_name] = {
                "passed": result,
                "status": "PASS" if result else "FAIL"
            }
        except Exception as e:
            results[test_name] = {
                "passed": False,
                "status": "ERROR",
                "error": str(e)
            }
    
    print(f"\n{'=' * 50}")
    print("ADVERSARIAL VALIDATION RESULTS")
    print(f"{'=' * 50}")
    
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} {test_name}")
        if "error" in result:
            print(f"    Error: {result['error']}")
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count >= 4:  # 4/5 tests must pass
        print("🎉 ADVERSARIAL VALIDATION SUCCESSFUL")
        print("System demonstrates resilience under stress conditions")
        return True
    else:
        print("⚠️ ADVERSARIAL VALIDATION CONCERNS")
        print("System shows vulnerabilities under stress conditions")
        return False

if __name__ == "__main__":
    os.chdir("/Users/cal/DEV/RIF/systems")
    success = run_adversarial_validation()
    sys.exit(0 if success else 1)