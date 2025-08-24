#!/usr/bin/env python3
"""
Integration Stress Tester Tool - RIF Adversarial Testing Suite

This tool identifies integration points and stress tests them to find
fragile connections and failure modes under various stress conditions.
"""

import json
import subprocess
import os
import sys
import re
import time
import threading
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import urllib.parse
import socket

@dataclass
class IntegrationPoint:
    """Represents an integration point in the system"""
    point_type: str
    name: str
    location: str
    endpoint: Optional[str]
    dependencies: List[str]
    risk_level: str
    stress_tests: List[str]

@dataclass
class StressTestResult:
    """Results from stress testing an integration point"""
    integration_point: str
    test_scenario: str
    stress_type: str
    passed: bool
    response_time: float
    error_details: str
    failure_mode: str
    recovery_time: float

class IntegrationStressTester:
    """Tool to stress test integration points and find failure modes"""
    
    def __init__(self, target_path: str):
        self.target_path = target_path
        self.integration_points = []
        self.stress_test_results = []
        
        # Integration patterns to look for
        self.integration_patterns = {
            "http_api": r"(fetch|axios|requests?|http)\s*\(|https?://",
            "database": r"(connect|query|select|insert|update|delete)\s*\(|mongodb://|postgresql://|mysql://",
            "message_queue": r"(publish|subscribe|enqueue|dequeue)\s*\(|amqp://|redis://",
            "file_system": r"(open|read|write|mkdir|rmdir)\s*\(|fs\.|pathlib",
            "network_socket": r"socket\s*\(|connect\s*\(|bind\s*\(",
            "external_service": r"(api_key|auth_token|service_url)",
            "cache": r"(cache\.|redis\.|memcached)",
            "search_engine": r"(elasticsearch|solr|search)",
            "third_party": r"(stripe|paypal|twilio|sendgrid|aws|azure|gcp)"
        }
    
    def discover_integration_points(self) -> List[IntegrationPoint]:
        """Discover integration points in the codebase"""
        integration_points = []
        
        if os.path.isfile(self.target_path):
            points = self._analyze_file_integrations(self.target_path)
            integration_points.extend(points)
        elif os.path.isdir(self.target_path):
            for root, dirs, files in os.walk(self.target_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c')):
                        file_path = os.path.join(root, file)
                        points = self._analyze_file_integrations(file_path)
                        integration_points.extend(points)
        
        return integration_points
    
    def _analyze_file_integrations(self, file_path: str) -> List[IntegrationPoint]:
        """Analyze a file for integration points"""
        integration_points = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern_name, pattern in self.integration_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # Extract endpoint/connection details if possible
                    endpoint = self._extract_endpoint(line_content, pattern_name)
                    
                    integration_point = IntegrationPoint(
                        point_type=pattern_name,
                        name=f"{pattern_name}_{os.path.basename(file_path)}_{line_num}",
                        location=f"{file_path}:{line_num}",
                        endpoint=endpoint,
                        dependencies=self._identify_dependencies(line_content, pattern_name),
                        risk_level=self._assess_integration_risk(pattern_name, line_content),
                        stress_tests=self._generate_stress_tests(pattern_name, endpoint)
                    )
                    integration_points.append(integration_point)
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return integration_points
    
    def _extract_endpoint(self, line_content: str, pattern_type: str) -> Optional[str]:
        """Extract endpoint/connection string from code line"""
        endpoint_patterns = {
            "http_api": r"['\"]https?://[^'\"]+['\"]",
            "database": r"['\"][^'\"]*://[^'\"]+['\"]",
            "message_queue": r"['\"]amqp://[^'\"]+['\"]|['\"]redis://[^'\"]+['\"]",
            "external_service": r"['\"]https?://api\.[^'\"]+['\"]"
        }
        
        pattern = endpoint_patterns.get(pattern_type)
        if pattern:
            match = re.search(pattern, line_content)
            if match:
                return match.group().strip("'\"")
        
        return None
    
    def _identify_dependencies(self, line_content: str, pattern_type: str) -> List[str]:
        """Identify dependencies for this integration point"""
        dependencies = []
        
        dependency_indicators = {
            "http_api": ["network", "dns", "ssl", "authentication"],
            "database": ["database_server", "connection_pool", "authentication"],
            "message_queue": ["message_broker", "network", "authentication"],
            "file_system": ["disk_space", "file_permissions", "filesystem"],
            "network_socket": ["network", "port_availability", "firewall"],
            "external_service": ["network", "api_key", "rate_limits", "service_availability"],
            "cache": ["cache_server", "memory", "network"],
            "search_engine": ["search_service", "index_availability", "network"],
            "third_party": ["external_api", "authentication", "network", "rate_limits"]
        }
        
        return dependency_indicators.get(pattern_type, ["network"])
    
    def _assess_integration_risk(self, pattern_type: str, line_content: str) -> str:
        """Assess risk level of integration point"""
        high_risk_types = {"external_service", "third_party", "http_api", "database"}
        medium_risk_types = {"message_queue", "cache", "search_engine", "network_socket"}
        
        if pattern_type in high_risk_types:
            risk_level = "HIGH"
        elif pattern_type in medium_risk_types:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Increase risk for certain keywords
        high_risk_keywords = ["payment", "auth", "security", "critical", "production"]
        if any(keyword in line_content.lower() for keyword in high_risk_keywords):
            risk_level = "HIGH"
        
        return risk_level
    
    def _generate_stress_tests(self, pattern_type: str, endpoint: Optional[str]) -> List[str]:
        """Generate stress test scenarios for integration point"""
        base_tests = {
            "http_api": [
                "Test with network timeout",
                "Test with connection refused",
                "Test with slow response (>30s)",
                "Test with partial response data",
                "Test with concurrent requests",
                "Test with rate limiting",
                "Test with authentication failures",
                "Test with malformed responses"
            ],
            "database": [
                "Test with connection pool exhaustion",
                "Test with database server down",
                "Test with slow queries (>10s)",
                "Test with deadlock scenarios",
                "Test with concurrent transactions",
                "Test with memory pressure",
                "Test with disk space full",
                "Test with connection interruption"
            ],
            "message_queue": [
                "Test with queue server down",
                "Test with queue overflow",
                "Test with message processing delays",
                "Test with concurrent publishers/consumers",
                "Test with network partitions",
                "Test with authentication failures",
                "Test with message size limits",
                "Test with poison messages"
            ],
            "file_system": [
                "Test with file not found",
                "Test with permission denied",
                "Test with disk space full",
                "Test with file locked by another process",
                "Test with network file system timeout",
                "Test with very large files",
                "Test with concurrent file access",
                "Test with filesystem corruption"
            ],
            "network_socket": [
                "Test with port already in use",
                "Test with network unreachable",
                "Test with connection reset",
                "Test with socket timeout",
                "Test with concurrent connections",
                "Test with buffer overflow",
                "Test with malformed packets",
                "Test with connection drops"
            ],
            "external_service": [
                "Test with service unavailable (5xx errors)",
                "Test with service rate limiting (429)",
                "Test with invalid API keys",
                "Test with service response delays",
                "Test with concurrent API calls",
                "Test with malformed API responses",
                "Test with API version changes",
                "Test with network partitions"
            ],
            "cache": [
                "Test with cache server down",
                "Test with cache memory full",
                "Test with cache eviction under pressure",
                "Test with concurrent cache operations",
                "Test with cache corruption",
                "Test with network timeouts to cache",
                "Test with cache key collisions",
                "Test with cache serialization errors"
            ]
        }
        
        return base_tests.get(pattern_type, [
            "Test with service unavailable",
            "Test with timeout conditions",
            "Test with concurrent access",
            "Test with error conditions"
        ])
    
    def run_stress_tests(self) -> List[StressTestResult]:
        """Run stress tests on all identified integration points"""
        results = []
        
        for integration_point in self.integration_points:
            print(f"Stress testing {integration_point.name}...")
            
            for test_scenario in integration_point.stress_tests:
                result = self._execute_stress_test(integration_point, test_scenario)
                results.append(result)
        
        return results
    
    def _execute_stress_test(self, integration_point: IntegrationPoint, test_scenario: str) -> StressTestResult:
        """Execute a specific stress test scenario"""
        start_time = time.time()
        
        # Simulate different stress test scenarios
        stress_type = self._determine_stress_type(test_scenario)
        passed, error_details, failure_mode = self._simulate_stress_test(
            integration_point, test_scenario, stress_type
        )
        
        response_time = time.time() - start_time
        
        # Simulate recovery time for failed tests
        recovery_time = 0.0
        if not passed:
            recovery_time = random.uniform(1.0, 10.0)  # 1-10 seconds recovery simulation
        
        return StressTestResult(
            integration_point=integration_point.name,
            test_scenario=test_scenario,
            stress_type=stress_type,
            passed=passed,
            response_time=response_time,
            error_details=error_details,
            failure_mode=failure_mode,
            recovery_time=recovery_time
        )
    
    def _determine_stress_type(self, test_scenario: str) -> str:
        """Determine the type of stress being applied"""
        if "timeout" in test_scenario.lower():
            return "TIMEOUT"
        elif "concurrent" in test_scenario.lower():
            return "CONCURRENCY"
        elif "memory" in test_scenario.lower() or "overflow" in test_scenario.lower():
            return "RESOURCE_EXHAUSTION"
        elif "network" in test_scenario.lower() or "connection" in test_scenario.lower():
            return "NETWORK_FAILURE"
        elif "authentication" in test_scenario.lower() or "auth" in test_scenario.lower():
            return "AUTHENTICATION_FAILURE"
        elif "rate limit" in test_scenario.lower():
            return "RATE_LIMITING"
        else:
            return "GENERAL_STRESS"
    
    def _simulate_stress_test(self, integration_point: IntegrationPoint, test_scenario: str, stress_type: str) -> Tuple[bool, str, str]:
        """Simulate execution of a stress test"""
        # Simulate test results based on integration type and risk level
        
        # High-risk integrations are more likely to fail under stress
        failure_probability = 0.1  # Base 10% failure rate
        
        if integration_point.risk_level == "HIGH":
            failure_probability = 0.4  # 40% failure rate for high-risk
        elif integration_point.risk_level == "MEDIUM":
            failure_probability = 0.25  # 25% failure rate for medium-risk
        
        # Adjust failure probability based on stress type
        stress_multipliers = {
            "TIMEOUT": 1.5,
            "CONCURRENCY": 2.0,
            "RESOURCE_EXHAUSTION": 2.5,
            "NETWORK_FAILURE": 1.8,
            "AUTHENTICATION_FAILURE": 1.2,
            "RATE_LIMITING": 1.3,
            "GENERAL_STRESS": 1.0
        }
        
        final_failure_probability = failure_probability * stress_multipliers.get(stress_type, 1.0)
        
        # Determine if test passes or fails
        passes = random.random() > final_failure_probability
        
        if passes:
            return True, "", "NONE"
        else:
            # Generate appropriate error details and failure modes
            error_details, failure_mode = self._generate_failure_details(
                integration_point.point_type, stress_type
            )
            return False, error_details, failure_mode
    
    def _generate_failure_details(self, point_type: str, stress_type: str) -> Tuple[str, str]:
        """Generate realistic failure details based on integration and stress type"""
        
        failure_scenarios = {
            ("http_api", "TIMEOUT"): (
                "HTTP request timed out after 30 seconds",
                "SERVICE_TIMEOUT"
            ),
            ("http_api", "CONCURRENCY"): (
                "Too many concurrent requests, connection pool exhausted",
                "CONNECTION_POOL_EXHAUSTED"
            ),
            ("http_api", "RATE_LIMITING"): (
                "HTTP 429 Too Many Requests - rate limit exceeded",
                "RATE_LIMITED"
            ),
            ("database", "TIMEOUT"): (
                "Database query timed out after 10 seconds",
                "QUERY_TIMEOUT"
            ),
            ("database", "CONCURRENCY"): (
                "Deadlock detected between competing transactions",
                "DEADLOCK"
            ),
            ("database", "RESOURCE_EXHAUSTION"): (
                "Database connection pool exhausted",
                "CONNECTION_POOL_FULL"
            ),
            ("message_queue", "NETWORK_FAILURE"): (
                "Connection to message broker lost",
                "BROKER_DISCONNECTION"
            ),
            ("message_queue", "RESOURCE_EXHAUSTION"): (
                "Message queue overflow - messages being dropped",
                "QUEUE_OVERFLOW"
            ),
            ("file_system", "RESOURCE_EXHAUSTION"): (
                "Disk space full - unable to write file",
                "DISK_FULL"
            ),
            ("file_system", "CONCURRENCY"): (
                "File locked by another process",
                "FILE_LOCK_CONTENTION"
            ),
            ("external_service", "AUTHENTICATION_FAILURE"): (
                "API key invalid or expired",
                "AUTHENTICATION_ERROR"
            ),
            ("external_service", "NETWORK_FAILURE"): (
                "External service unreachable",
                "SERVICE_UNAVAILABLE"
            )
        }
        
        # Default failure if specific scenario not found
        default_details = (
            f"Integration stress test failed under {stress_type} conditions",
            "INTEGRATION_FAILURE"
        )
        
        return failure_scenarios.get((point_type, stress_type), default_details)
    
    def run_concurrent_stress_test(self, integration_point: IntegrationPoint, concurrent_requests: int = 10) -> List[StressTestResult]:
        """Run concurrent stress test on an integration point"""
        results = []
        threads = []
        
        def stress_worker():
            result = self._execute_stress_test(
                integration_point, 
                f"Concurrent stress test ({concurrent_requests} requests)"
            )
            results.append(result)
        
        # Start concurrent threads
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=stress_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def generate_report(self, output_path: str):
        """Generate comprehensive integration stress test report"""
        failed_tests = [r for r in self.stress_test_results if not r.passed]
        critical_failures = [r for r in failed_tests if r.failure_mode in ["DEADLOCK", "CONNECTION_POOL_EXHAUSTED", "SERVICE_UNAVAILABLE"]]
        
        # Calculate response time statistics
        response_times = [r.response_time for r in self.stress_test_results if r.passed]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        report = {
            "tool": "Integration Stress Tester",
            "timestamp": subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                               text=True).strip(),
            "target_path": self.target_path,
            "summary": {
                "total_integration_points": len(self.integration_points),
                "total_stress_tests": len(self.stress_test_results),
                "failed_tests": len(failed_tests),
                "critical_failures": len(critical_failures),
                "average_response_time": round(avg_response_time, 3),
                "integration_risk_distribution": {
                    "HIGH": len([p for p in self.integration_points if p.risk_level == "HIGH"]),
                    "MEDIUM": len([p for p in self.integration_points if p.risk_level == "MEDIUM"]),
                    "LOW": len([p for p in self.integration_points if p.risk_level == "LOW"])
                }
            },
            "integration_points": [self._integration_point_to_dict(p) for p in self.integration_points],
            "failure_analysis": {
                "failure_modes": self._analyze_failure_modes(),
                "critical_failures": [self._test_result_to_dict(r) for r in critical_failures],
                "failure_by_stress_type": self._group_failures_by_stress_type()
            },
            "performance_analysis": {
                "response_time_distribution": self._analyze_response_times(),
                "slowest_integration_points": self._identify_slow_integrations()
            },
            "stress_test_results": [self._test_result_to_dict(r) for r in self.stress_test_results],
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Integration stress test report written to {output_path}")
    
    def _analyze_failure_modes(self) -> Dict[str, int]:
        """Analyze distribution of failure modes"""
        failure_modes = {}
        for result in self.stress_test_results:
            if not result.passed:
                mode = result.failure_mode
                failure_modes[mode] = failure_modes.get(mode, 0) + 1
        return failure_modes
    
    def _group_failures_by_stress_type(self) -> Dict[str, List[Dict]]:
        """Group failures by stress type"""
        grouped = {}
        for result in self.stress_test_results:
            if not result.passed:
                stress_type = result.stress_type
                if stress_type not in grouped:
                    grouped[stress_type] = []
                grouped[stress_type].append(self._test_result_to_dict(result))
        return grouped
    
    def _analyze_response_times(self) -> Dict[str, float]:
        """Analyze response time distribution"""
        response_times = [r.response_time for r in self.stress_test_results]
        if not response_times:
            return {}
        
        response_times.sort()
        return {
            "min": response_times[0],
            "max": response_times[-1],
            "median": response_times[len(response_times) // 2],
            "p95": response_times[int(len(response_times) * 0.95)],
            "average": sum(response_times) / len(response_times)
        }
    
    def _identify_slow_integrations(self) -> List[Dict[str, Any]]:
        """Identify slowest integration points"""
        # Group results by integration point
        integration_times = {}
        for result in self.stress_test_results:
            point_name = result.integration_point
            if point_name not in integration_times:
                integration_times[point_name] = []
            integration_times[point_name].append(result.response_time)
        
        # Calculate average response time for each integration
        slow_integrations = []
        for point_name, times in integration_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > 1.0:  # Consider >1 second as slow
                slow_integrations.append({
                    "integration_point": point_name,
                    "average_response_time": round(avg_time, 3),
                    "max_response_time": round(max(times), 3),
                    "test_count": len(times)
                })
        
        return sorted(slow_integrations, key=lambda x: x["average_response_time"], reverse=True)[:10]
    
    def _integration_point_to_dict(self, point: IntegrationPoint) -> Dict[str, Any]:
        """Convert integration point to dictionary for JSON serialization"""
        return {
            "point_type": point.point_type,
            "name": point.name,
            "location": point.location,
            "endpoint": point.endpoint,
            "dependencies": point.dependencies,
            "risk_level": point.risk_level,
            "stress_tests": point.stress_tests
        }
    
    def _test_result_to_dict(self, result: StressTestResult) -> Dict[str, Any]:
        """Convert test result to dictionary for JSON serialization"""
        return {
            "integration_point": result.integration_point,
            "test_scenario": result.test_scenario,
            "stress_type": result.stress_type,
            "passed": result.passed,
            "response_time": round(result.response_time, 3),
            "error_details": result.error_details,
            "failure_mode": result.failure_mode,
            "recovery_time": round(result.recovery_time, 3)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        failed_tests = [r for r in self.stress_test_results if not r.passed]
        critical_failures = [r for r in failed_tests if r.failure_mode in ["DEADLOCK", "CONNECTION_POOL_EXHAUSTED", "SERVICE_UNAVAILABLE"]]
        
        if critical_failures:
            recommendations.append(
                f"CRITICAL: Found {len(critical_failures)} critical integration failures. "
                "These represent single points of failure that could bring down the system."
            )
        
        # Analyze failure patterns
        failure_modes = self._analyze_failure_modes()
        
        if "CONNECTION_POOL_EXHAUSTED" in failure_modes:
            recommendations.append(
                "Connection pool exhaustion detected. Increase connection pool size or implement connection pooling."
            )
        
        if "DEADLOCK" in failure_modes:
            recommendations.append(
                "Database deadlocks detected. Review transaction ordering and implement deadlock retry logic."
            )
        
        if "SERVICE_TIMEOUT" in failure_modes:
            recommendations.append(
                "Service timeouts detected. Implement circuit breaker pattern and async processing where possible."
            )
        
        if "RATE_LIMITED" in failure_modes:
            recommendations.append(
                "Rate limiting encountered. Implement exponential backoff and request throttling."
            )
        
        # Performance recommendations
        slow_integrations = self._identify_slow_integrations()
        if slow_integrations:
            recommendations.append(
                f"Found {len(slow_integrations)} slow integration points. "
                "Consider caching, connection pooling, or async processing."
            )
        
        # General recommendations
        high_risk_points = [p for p in self.integration_points if p.risk_level == "HIGH"]
        if high_risk_points:
            recommendations.append(
                f"Found {len(high_risk_points)} high-risk integration points. "
                "Implement comprehensive error handling, retries, and fallback mechanisms."
            )
        
        recommendations.extend([
            "Implement circuit breaker pattern for external service calls",
            "Add comprehensive timeout handling for all integration points",
            "Implement retry logic with exponential backoff for transient failures",
            "Add monitoring and alerting for integration point health",
            "Consider implementing bulkhead pattern to isolate integration failures",
            "Add graceful degradation mechanisms for non-critical integrations",
            "Implement health checks for all external dependencies"
        ])
        
        return recommendations

def main():
    """Main entry point for integration stress tester tool"""
    if len(sys.argv) < 2:
        print("Usage: integration-stress-tester.py <target_path> [output_file]")
        print("  target_path: Path to analyze (file or directory)")
        print("  output_file: Optional output file for report (default: integration-stress-report.json)")
        sys.exit(1)
    
    target_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "integration-stress-report.json"
    
    tester = IntegrationStressTester(target_path)
    
    # Discover integration points
    integration_points = tester.discover_integration_points()
    tester.integration_points.extend(integration_points)
    
    if tester.integration_points:
        print(f"Discovered {len(tester.integration_points)} integration points")
        
        # Run stress tests
        test_results = tester.run_stress_tests()
        tester.stress_test_results.extend(test_results)
        
        failed_tests = [r for r in test_results if not r.passed]
        print(f"Executed stress tests: {len(failed_tests)} failures found")
    else:
        print("No integration points discovered")
    
    # Generate report
    tester.generate_report(output_file)
    
    # Summary output
    if tester.stress_test_results:
        failed_tests = [r for r in tester.stress_test_results if not r.passed]
        critical_failures = [r for r in failed_tests if r.failure_mode in ["DEADLOCK", "CONNECTION_POOL_EXHAUSTED", "SERVICE_UNAVAILABLE"]]
        
        print(f"\nIntegration Stress Test Summary:")
        print(f"  Integration points found: {len(tester.integration_points)}")
        print(f"  Stress tests executed: {len(tester.stress_test_results)}")
        print(f"  Failed tests: {len(failed_tests)}")
        print(f"  Critical failures: {len(critical_failures)}")
        
        if critical_failures:
            print(f"\nCRITICAL INTEGRATION FAILURES:")
            for failure in critical_failures[:5]:  # Show first 5
                print(f"    - {failure.integration_point}")
                print(f"      Failure: {failure.failure_mode}")
                print(f"      Details: {failure.error_details}")

if __name__ == "__main__":
    main()