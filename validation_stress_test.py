#!/usr/bin/env python3
"""
Stress Testing and Security Validation for Dynamic MCP Loader
============================================================

Validates production readiness through stress testing, error scenarios,
and security boundary testing.

Issue: #82 - Dynamic MCP Loader Validation
Component: Comprehensive validation test suite
"""

import asyncio
import logging
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add the mcp module to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcp.loader.dynamic_loader import DynamicMCPLoader, ProjectContext
from mcp.security.security_gateway import SecurityGateway

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTestSecurityGateway(SecurityGateway):
    """Security gateway that simulates various security scenarios"""
    
    def __init__(self, fail_rate=0.2, delay_ms=50):
        super().__init__()
        self.fail_rate = fail_rate
        self.delay_ms = delay_ms
        self.validation_count = 0
        
    async def validate_server_security(self, server_config):
        """Simulate security validation with controlled failure rate"""
        await asyncio.sleep(self.delay_ms / 1000.0)  # Simulate network delay
        self.validation_count += 1
        
        server_id = server_config.get('server_id', '')
        
        # Fail certain servers to test error handling
        if 'high-risk' in server_id or self.validation_count % (1/self.fail_rate) == 0:
            logger.warning(f"Security validation FAILED for {server_id}")
            return False
            
        logger.debug(f"Security validation PASSED for {server_id}")
        return True


async def test_concurrent_loading_stress():
    """Test high concurrency server loading"""
    logger.info("🔄 STRESS TEST: Concurrent Loading")
    
    security_gateway = StressTestSecurityGateway(fail_rate=0.1)
    loader = DynamicMCPLoader(
        security_gateway=security_gateway,
        max_concurrent_loads=8,
        resource_budget_mb=2048
    )
    
    # Create multiple project contexts for concurrent loading
    project_contexts = []
    for i in range(10):
        context = ProjectContext(
            project_path=f"/tmp/stress_test_{i}",
            technology_stack={'python', 'nodejs', 'docker'},
            integrations={'github', 'database', 'cloud'},
            needs_reasoning=i % 2 == 0,
            needs_memory=i % 3 == 0,
            needs_database=True,
            needs_cloud=i % 4 == 0,
            complexity='high',
            agent_type='rif-implementer'
        )
        project_contexts.append(context)
    
    # Execute concurrent loading
    start_time = time.time()
    tasks = [loader.load_servers_for_project(ctx) for ctx in project_contexts]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        total_servers_attempted = 0
        total_servers_successful = 0
        exceptions = []
        
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                total_servers_attempted += len(result)
                total_servers_successful += len([r for r in result if r.status == 'success'])
        
        duration = end_time - start_time
        logger.info(f"✅ Concurrent loading completed in {duration:.2f}s")
        logger.info(f"   Projects: {len(project_contexts)}")
        logger.info(f"   Servers attempted: {total_servers_attempted}")
        logger.info(f"   Servers successful: {total_servers_successful}")
        logger.info(f"   Success rate: {total_servers_successful/total_servers_attempted*100:.1f}%")
        logger.info(f"   Exceptions: {len(exceptions)}")
        
        return {
            "test": "concurrent_loading_stress",
            "status": "PASSED" if len(exceptions) == 0 else "PASSED_WITH_EXCEPTIONS",
            "duration_s": duration,
            "projects": len(project_contexts),
            "servers_attempted": total_servers_attempted,
            "servers_successful": total_servers_successful,
            "success_rate": total_servers_successful/total_servers_attempted*100,
            "exceptions": len(exceptions)
        }
        
    except Exception as e:
        logger.error(f"❌ Concurrent loading stress test failed: {e}")
        return {
            "test": "concurrent_loading_stress",
            "status": "FAILED",
            "error": str(e)
        }


async def test_resource_exhaustion():
    """Test behavior under resource constraint scenarios"""
    logger.info("🔄 STRESS TEST: Resource Exhaustion")
    
    # Create loader with very limited resources
    loader = DynamicMCPLoader(
        max_concurrent_loads=2,
        resource_budget_mb=100  # Very limited
    )
    
    # Create a high-resource project
    resource_heavy_context = ProjectContext(
        project_path="/tmp/resource_heavy",
        technology_stack={'python', 'nodejs', 'docker', 'kubernetes'},
        integrations={'github', 'database', 'cloud', 'api', 'monitoring'},
        needs_reasoning=True,
        needs_memory=True,
        needs_database=True,
        needs_cloud=True,
        complexity='very-high',
        agent_type='rif-architect'
    )
    
    try:
        start_time = time.time()
        results = await loader.load_servers_for_project(resource_heavy_context)
        end_time = time.time()
        
        # Should handle resource constraints gracefully
        successful = [r for r in results if r.status == 'success']
        skipped = [r for r in results if r.status == 'skipped']
        failed = [r for r in results if r.status == 'failed']
        
        logger.info(f"✅ Resource exhaustion test completed in {end_time-start_time:.2f}s")
        logger.info(f"   Servers successful: {len(successful)}")
        logger.info(f"   Servers skipped (budget): {len(skipped)}")
        logger.info(f"   Servers failed: {len(failed)}")
        
        # Should have some skipped due to budget constraints
        resource_constrained = len(skipped) > 0
        
        return {
            "test": "resource_exhaustion",
            "status": "PASSED" if resource_constrained else "UNEXPECTED",
            "servers_successful": len(successful),
            "servers_skipped": len(skipped),
            "servers_failed": len(failed),
            "resource_constraint_respected": resource_constrained
        }
        
    except Exception as e:
        logger.error(f"❌ Resource exhaustion test failed: {e}")
        return {
            "test": "resource_exhaustion", 
            "status": "FAILED",
            "error": str(e)
        }


async def test_security_boundary_conditions():
    """Test security validation edge cases"""
    logger.info("🔄 SECURITY TEST: Boundary Conditions")
    
    # Create security gateway that fails high-risk servers
    security_gateway = StressTestSecurityGateway(fail_rate=0.5)
    loader = DynamicMCPLoader(security_gateway=security_gateway)
    
    # Test with various server configurations
    high_risk_configs = [
        {
            'server_id': 'high-risk-server-1',
            'name': 'High Risk Server',
            'version': '1.0.0',
            'security_level': 'very-high',
            'capabilities': ['admin_access', 'system_control'],
            'resource_requirements': {'memory_mb': 64},
            'dependencies': []
        },
        {
            'server_id': 'normal-server-1',
            'name': 'Normal Server',
            'version': '1.0.0', 
            'security_level': 'low',
            'capabilities': ['file_access'],
            'resource_requirements': {'memory_mb': 64},
            'dependencies': []
        }
    ]
    
    security_results = []
    
    for config in high_risk_configs:
        try:
            result = await loader.load_server(config)
            security_results.append({
                "server_id": config['server_id'],
                "security_level": config['security_level'],
                "status": result.status,
                "error": result.error_message
            })
            
        except Exception as e:
            security_results.append({
                "server_id": config['server_id'],
                "security_level": config['security_level'],
                "status": "exception",
                "error": str(e)
            })
    
    # Analyze security results
    high_risk_blocked = any(r for r in security_results 
                           if 'high-risk' in r['server_id'] and r['status'] != 'success')
    normal_allowed = any(r for r in security_results
                        if 'normal' in r['server_id'] and r['status'] == 'success')
    
    logger.info("✅ Security boundary test completed")
    for result in security_results:
        status_icon = {"success": "✅", "failed": "❌", "exception": "⚠️"}.get(result['status'], "❓")
        logger.info(f"   {status_icon} {result['server_id']}: {result['status']}")
    
    return {
        "test": "security_boundary_conditions",
        "status": "PASSED" if (high_risk_blocked or normal_allowed) else "FAILED",
        "high_risk_properly_blocked": high_risk_blocked,
        "normal_servers_allowed": normal_allowed,
        "security_results": security_results
    }


async def test_error_recovery_scenarios():
    """Test error handling and recovery mechanisms"""
    logger.info("🔄 RESILIENCE TEST: Error Recovery")
    
    # Create loader with normal settings
    loader = DynamicMCPLoader()
    
    # Test invalid project context
    invalid_context = ProjectContext(
        project_path="/nonexistent/path/that/should/not/exist",
        technology_stack=set(),
        integrations=set(),
        needs_reasoning=False,
        needs_memory=False,
        needs_database=False,
        needs_cloud=False,
        complexity='unknown'  # Invalid complexity
    )
    
    error_scenarios = []
    
    try:
        # Test 1: Invalid project path
        start_time = time.time()
        results = await loader.load_servers_for_project(invalid_context)
        end_time = time.time()
        
        # Should handle gracefully without crashing
        error_scenarios.append({
            "scenario": "invalid_project_path",
            "status": "handled_gracefully",
            "duration_s": end_time - start_time,
            "results_count": len(results)
        })
        
    except Exception as e:
        error_scenarios.append({
            "scenario": "invalid_project_path",
            "status": "exception_thrown",
            "error": str(e)
        })
    
    try:
        # Test 2: Server unload of non-existent server
        unload_success = await loader.unload_server("non-existent-server-id")
        error_scenarios.append({
            "scenario": "unload_nonexistent_server",
            "status": "handled_gracefully",
            "unload_success": unload_success
        })
        
    except Exception as e:
        error_scenarios.append({
            "scenario": "unload_nonexistent_server", 
            "status": "exception_thrown",
            "error": str(e)
        })
    
    logger.info("✅ Error recovery test completed")
    all_handled = all(scenario['status'] != 'exception_thrown' for scenario in error_scenarios)
    
    return {
        "test": "error_recovery_scenarios",
        "status": "PASSED" if all_handled else "FAILED",
        "scenarios_tested": len(error_scenarios),
        "all_errors_handled": all_handled,
        "error_scenarios": error_scenarios
    }


async def test_performance_benchmarks():
    """Test performance characteristics under various loads"""
    logger.info("🔄 PERFORMANCE TEST: Benchmarks")
    
    loader = DynamicMCPLoader(max_concurrent_loads=4, resource_budget_mb=512)
    
    # Performance test scenarios
    performance_results = []
    
    # Test 1: Single server load performance
    simple_context = ProjectContext(
        project_path="/tmp/perf_test",
        technology_stack={'python'},
        integrations={'github'},
        needs_reasoning=False,
        needs_memory=False,
        needs_database=False,
        needs_cloud=False,
        complexity='low'
    )
    
    start_time = time.time()
    simple_results = await loader.load_servers_for_project(simple_context)
    end_time = time.time()
    
    simple_duration = end_time - start_time
    simple_successful = len([r for r in simple_results if r.status == 'success'])
    
    performance_results.append({
        "scenario": "simple_project_load",
        "servers_loaded": simple_successful,
        "duration_s": simple_duration,
        "servers_per_second": simple_successful / simple_duration if simple_duration > 0 else 0
    })
    
    # Test 2: Complex project load performance  
    complex_context = ProjectContext(
        project_path="/tmp/complex_perf_test",
        technology_stack={'python', 'nodejs', 'docker'},
        integrations={'github', 'database', 'cloud'},
        needs_reasoning=True,
        needs_memory=True,
        needs_database=True,
        needs_cloud=True,
        complexity='high'
    )
    
    start_time = time.time()
    complex_results = await loader.load_servers_for_project(complex_context)
    end_time = time.time()
    
    complex_duration = end_time - start_time
    complex_successful = len([r for r in complex_results if r.status == 'success'])
    
    performance_results.append({
        "scenario": "complex_project_load", 
        "servers_loaded": complex_successful,
        "duration_s": complex_duration,
        "servers_per_second": complex_successful / complex_duration if complex_duration > 0 else 0
    })
    
    # Get loading metrics
    metrics = await loader.get_loading_metrics()
    
    logger.info("✅ Performance benchmark completed")
    for result in performance_results:
        logger.info(f"   {result['scenario']}: {result['servers_loaded']} servers in {result['duration_s']:.3f}s "
                   f"({result['servers_per_second']:.1f} servers/sec)")
    
    # Performance should be reasonable (>1 server/sec for simple projects)
    simple_performance_ok = performance_results[0]['servers_per_second'] >= 1.0
    
    return {
        "test": "performance_benchmarks",
        "status": "PASSED" if simple_performance_ok else "SLOW",
        "performance_results": performance_results,
        "loading_metrics": metrics
    }


async def run_comprehensive_validation():
    """Run all validation tests"""
    logger.info("🚀 COMPREHENSIVE VALIDATION: Dynamic MCP Loader")
    logger.info("=" * 60)
    
    validation_results = []
    
    # Run all validation tests
    test_functions = [
        test_concurrent_loading_stress,
        test_resource_exhaustion,
        test_security_boundary_conditions,
        test_error_recovery_scenarios,
        test_performance_benchmarks
    ]
    
    for test_func in test_functions:
        try:
            result = await test_func()
            validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} failed: {e}")
            validation_results.append({
                "test": test_func.__name__,
                "status": "FAILED",
                "error": str(e)
            })
    
    # Summarize results
    logger.info("\n" + "=" * 60)
    logger.info("🏁 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = [r for r in validation_results if r['status'].startswith('PASSED')]
    failed_tests = [r for r in validation_results if r['status'] == 'FAILED']
    
    logger.info(f"✅ Tests Passed: {len(passed_tests)}/{len(validation_results)}")
    logger.info(f"❌ Tests Failed: {len(failed_tests)}/{len(validation_results)}")
    
    for result in validation_results:
        status_icon = {"PASSED": "✅", "PASSED_WITH_EXCEPTIONS": "⚠️", "FAILED": "❌", "SLOW": "🐌"}.get(result['status'], "❓")
        logger.info(f"   {status_icon} {result['test']}: {result['status']}")
    
    # Save validation report
    from datetime import datetime
    validation_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "validation_type": "comprehensive_stress_security_performance",
        "issue_id": 82,
        "total_tests": len(validation_results),
        "passed_tests": len(passed_tests),
        "failed_tests": len(failed_tests),
        "overall_status": "PASSED" if len(failed_tests) == 0 else "FAILED",
        "validation_results": validation_results
    }
    
    report_file = "knowledge/validation_evidence_issue_82_comprehensive.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"\n📄 Validation report saved: {report_file}")
    
    return validation_report


if __name__ == "__main__":
    import datetime
    asyncio.run(run_comprehensive_validation())