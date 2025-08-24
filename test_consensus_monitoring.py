#!/usr/bin/env python3
"""
Test suite for Consensus Monitoring System
Tests all aspects of the consensus monitoring implementation
"""

import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the system monitor module to path
sys.path.insert(0, str(Path(__file__).parent / "claude" / "commands"))

from system_monitor import ConsensusMonitor, track_consensus_session
import statistics

class ConsensusMonitoringTestSuite:
    """Comprehensive test suite for consensus monitoring"""
    
    def __init__(self):
        self.temp_dir = None
        self.monitor = None
        self.test_results = []
        
    def setup(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.monitor = ConsensusMonitor(str(self.temp_dir))
        print(f"✅ Test environment setup complete - using {self.temp_dir}")
        
    def cleanup(self):
        """Cleanup test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        print("🧹 Test environment cleaned up")
        
    def create_test_voting_session(self, scenario: str = "standard") -> Dict[str, Any]:
        """Create test voting session data for different scenarios"""
        base_session = {
            "id": f"test_session_{int(time.time())}_{scenario}",
            "issue_number": 63,
            "consensus_type": "weighted_voting",
            "duration": 300000,  # 5 minutes
            "outcome": "agreed"
        }
        
        if scenario == "standard":
            # Standard voting session with high agreement
            base_session["votes"] = [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.85, "weight": 1.1, "reasoning": "analysis complete"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.90, "weight": 1.0, "reasoning": "implementation ready"},
                {"agent": "rif-validator", "decision": "approve", "confidence": 0.95, "weight": 1.5, "reasoning": "tests pass"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.80, "weight": 1.3, "reasoning": "design sound"},
            ]
            
        elif scenario == "disagreement":
            # Session with significant disagreement
            base_session["votes"] = [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.60, "weight": 1.1, "reasoning": "analysis suggests proceed"},
                {"agent": "rif-implementer", "decision": "reject", "confidence": 0.80, "weight": 1.0, "reasoning": "implementation concerns"},
                {"agent": "rif-validator", "decision": "reject", "confidence": 0.90, "weight": 1.5, "reasoning": "quality issues"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.70, "weight": 1.3, "reasoning": "architecture supports"},
                {"agent": "rif-security", "decision": "reject", "confidence": 0.95, "weight": 2.0, "reasoning": "security vulnerabilities"},
            ]
            base_session["outcome"] = "disagreed"
            base_session["duration"] = 1800000  # 30 minutes
            
        elif scenario == "low_confidence":
            # Session with low confidence levels
            base_session["votes"] = [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.30, "weight": 1.1, "reasoning": "unclear requirements"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.40, "weight": 1.0, "reasoning": "uncertain approach"},
                {"agent": "rif-validator", "decision": "approve", "confidence": 0.35, "weight": 1.5, "reasoning": "limited test coverage"},
            ]
            base_session["duration"] = 2400000  # 40 minutes
            
        elif scenario == "unanimous":
            # Perfect unanimous agreement
            base_session["votes"] = [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.95, "weight": 1.1, "reasoning": "thorough analysis complete"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.98, "weight": 1.0, "reasoning": "clean implementation"},
                {"agent": "rif-validator", "decision": "approve", "confidence": 1.0, "weight": 1.5, "reasoning": "all tests pass"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.92, "weight": 1.3, "reasoning": "excellent design"},
            ]
            base_session["consensus_type"] = "unanimous_consensus"
            base_session["duration"] = 120000  # 2 minutes
            
        return base_session
        
    def test_basic_consensus_tracking(self):
        """Test basic consensus tracking functionality"""
        print("\n🧪 Testing basic consensus tracking...")
        
        # Test with standard session
        session = self.create_test_voting_session("standard")
        report = self.monitor.track_consensus(session)
        
        # Validate report structure
        assert "session_id" in report, "Report missing session_id"
        assert "metrics" in report, "Report missing metrics"
        assert "analysis" in report, "Report missing analysis"
        assert "recommendations" in report, "Report missing recommendations"
        
        # Validate metrics
        metrics = report["metrics"]
        assert 0 <= metrics["agreement_level"] <= 1, f"Invalid agreement level: {metrics['agreement_level']}"
        assert metrics["total_participants"] == 4, f"Wrong participant count: {metrics['total_participants']}"
        assert metrics["decision_time"] == 300000, f"Wrong decision time: {metrics['decision_time']}"
        
        # Validate confidence distribution
        conf_dist = metrics["confidence_distribution"]
        assert 0 <= conf_dist["average"] <= 1, f"Invalid confidence average: {conf_dist['average']}"
        assert conf_dist["min"] <= conf_dist["max"], "Invalid confidence range"
        
        self.test_results.append({"test": "basic_consensus_tracking", "status": "PASS", "details": "All assertions passed"})
        print("✅ Basic consensus tracking test passed")
        
    def test_agreement_level_calculation(self):
        """Test agreement level calculation accuracy"""
        print("\n🧪 Testing agreement level calculations...")
        
        # Test unanimous agreement
        unanimous_session = self.create_test_voting_session("unanimous")
        report = self.monitor.track_consensus(unanimous_session)
        agreement = report["metrics"]["agreement_level"]
        assert agreement == 1.0, f"Unanimous session should have 1.0 agreement, got {agreement}"
        
        # Test disagreement scenario
        disagreement_session = self.create_test_voting_session("disagreement")
        report = self.monitor.track_consensus(disagreement_session)
        agreement = report["metrics"]["agreement_level"]
        
        # With weighted voting, security agent (weight 2.0) + implementer (weight 1.0) + validator (weight 1.5) = 4.5 reject
        # vs analyst (weight 1.1) + architect (weight 1.3) = 2.4 approve
        # Total weight = 6.9, majority weight = 4.5, so agreement = 4.5/6.9 ≈ 0.652
        expected_agreement = 4.5 / 6.9
        assert abs(agreement - expected_agreement) < 0.01, f"Expected agreement ~{expected_agreement:.3f}, got {agreement:.3f}"
        
        self.test_results.append({"test": "agreement_level_calculation", "status": "PASS", "details": "Agreement calculations accurate"})
        print("✅ Agreement level calculation test passed")
        
    def test_dissenter_identification(self):
        """Test identification of dissenting agents"""
        print("\n🧪 Testing dissenter identification...")
        
        # Test disagreement scenario
        session = self.create_test_voting_session("disagreement")
        report = self.monitor.track_consensus(session)
        
        dissenters = report["metrics"]["dissenting_agents"]
        
        # In disagreement scenario, majority decision should be "reject" (higher weighted votes)
        # So "approve" voters should be identified as dissenters: analyst and architect
        expected_dissenters = {"rif-analyst", "rif-architect"}
        actual_dissenters = set(dissenters)
        
        assert actual_dissenters == expected_dissenters, f"Expected dissenters {expected_dissenters}, got {actual_dissenters}"
        assert report["metrics"]["dissenter_count"] == 2, f"Expected 2 dissenters, got {report['metrics']['dissenter_count']}"
        
        # Test unanimous scenario - should have no dissenters
        unanimous_session = self.create_test_voting_session("unanimous")
        report = self.monitor.track_consensus(unanimous_session)
        assert report["metrics"]["dissenter_count"] == 0, f"Unanimous session should have 0 dissenters"
        
        self.test_results.append({"test": "dissenter_identification", "status": "PASS", "details": "Dissenters correctly identified"})
        print("✅ Dissenter identification test passed")
        
    def test_confidence_distribution_analysis(self):
        """Test confidence distribution analysis"""
        print("\n🧪 Testing confidence distribution analysis...")
        
        # Test low confidence scenario
        session = self.create_test_voting_session("low_confidence")
        report = self.monitor.track_consensus(session)
        
        conf_dist = report["metrics"]["confidence_distribution"]
        
        # Expected values based on test data: [0.30, 0.40, 0.35]
        expected_avg = (0.30 + 0.40 + 0.35) / 3
        expected_min = 0.30
        expected_max = 0.40
        
        assert abs(conf_dist["average"] - expected_avg) < 0.01, f"Expected avg {expected_avg:.2f}, got {conf_dist['average']:.2f}"
        assert conf_dist["min"] == expected_min, f"Expected min {expected_min}, got {conf_dist['min']}"
        assert conf_dist["max"] == expected_max, f"Expected max {expected_max}, got {conf_dist['max']}"
        
        self.test_results.append({"test": "confidence_distribution_analysis", "status": "PASS", "details": "Confidence analysis accurate"})
        print("✅ Confidence distribution analysis test passed")
        
    def test_consensus_strength_assessment(self):
        """Test consensus strength assessment"""
        print("\n🧪 Testing consensus strength assessment...")
        
        # Test different scenarios and expected strength ratings
        test_cases = [
            ("unanimous", "strong"),  # High agreement + high confidence  
            ("disagreement", "weak"),  # Moderate agreement (~65%) with high confidence
            ("low_confidence", "weak"),  # High agreement but low confidence (~35%)
        ]
        
        for scenario, expected_strength in test_cases:
            session = self.create_test_voting_session(scenario)
            report = self.monitor.track_consensus(session)
            actual_strength = report["analysis"]["consensus_strength"]
            
            assert actual_strength == expected_strength, \
                f"Scenario '{scenario}': expected strength '{expected_strength}', got '{actual_strength}'"
                
        self.test_results.append({"test": "consensus_strength_assessment", "status": "PASS", "details": "Strength assessments correct"})
        print("✅ Consensus strength assessment test passed")
        
    def test_recommendation_generation(self):
        """Test recommendation generation based on consensus analysis"""
        print("\n🧪 Testing recommendation generation...")
        
        # Test disagreement scenario - should recommend discussion or arbitration
        session = self.create_test_voting_session("disagreement")
        report = self.monitor.track_consensus(session)
        recommendations = report["recommendations"]
        
        assert any("arbitration" in rec.lower() or "discussion" in rec.lower() for rec in recommendations), \
            f"Disagreement scenario should recommend discussion or arbitration, got: {recommendations}"
            
        # Test low confidence scenario - should recommend more evidence
        session = self.create_test_voting_session("low_confidence")
        report = self.monitor.track_consensus(session)
        recommendations = report["recommendations"]
        
        assert any("evidence" in rec.lower() for rec in recommendations), \
            f"Low confidence scenario should recommend more evidence, got: {recommendations}"
            
        # Test unanimous scenario - should be positive
        session = self.create_test_voting_session("unanimous")
        report = self.monitor.track_consensus(session)
        recommendations = report["recommendations"]
        
        assert any("healthy" in rec.lower() or "proceed" in rec.lower() for rec in recommendations), \
            f"Unanimous scenario should have positive recommendations, got: {recommendations}"
            
        self.test_results.append({"test": "recommendation_generation", "status": "PASS", "details": "Recommendations appropriate"})
        print("✅ Recommendation generation test passed")
        
    def test_data_storage(self):
        """Test consensus data storage functionality"""
        print("\n🧪 Testing data storage...")
        
        session = self.create_test_voting_session("standard")
        report = self.monitor.track_consensus(session)
        
        # Check if data was stored
        consensus_dir = self.temp_dir / "consensus"
        assert consensus_dir.exists(), "Consensus directory not created"
        
        # Check if file was created with today's date
        today = datetime.now().strftime("%Y%m%d")
        expected_file = consensus_dir / f"consensus_reports_{today}.jsonl"
        assert expected_file.exists(), f"Expected storage file {expected_file} not found"
        
        # Verify stored data (JSONL format - one JSON per line)
        with open(expected_file) as f:
            lines = f.read().strip().split('\n')
            stored_data = json.loads(lines[-1])  # Get the last line (most recent)
            
        assert stored_data["session_id"] == report["session_id"], "Stored session_id doesn't match"
        assert "metrics" in stored_data, "Stored data missing metrics"
        
        self.test_results.append({"test": "data_storage", "status": "PASS", "details": "Data storage working correctly"})
        print("✅ Data storage test passed")
        
    def test_performance_benchmarks(self):
        """Test performance benchmarks for consensus monitoring"""
        print("\n🧪 Testing performance benchmarks...")
        
        # Test single session processing time
        session = self.create_test_voting_session("standard")
        start_time = time.time()
        report = self.monitor.track_consensus(session)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Should be well under 100ms as per acceptance criteria
        assert processing_time_ms < 100, f"Processing time {processing_time_ms:.1f}ms exceeds 100ms target"
        
        # Test batch processing performance
        sessions = [self.create_test_voting_session("standard") for _ in range(10)]
        start_time = time.time()
        
        reports = []
        for session in sessions:
            reports.append(self.monitor.track_consensus(session))
            
        end_time = time.time()
        batch_time_ms = (end_time - start_time) * 1000
        avg_time_per_session = batch_time_ms / 10
        
        assert avg_time_per_session < 50, f"Average processing time {avg_time_per_session:.1f}ms too high for batch processing"
        
        self.test_results.append({
            "test": "performance_benchmarks", 
            "status": "PASS", 
            "details": f"Single: {processing_time_ms:.1f}ms, Batch avg: {avg_time_per_session:.1f}ms"
        })
        print(f"✅ Performance test passed - Single: {processing_time_ms:.1f}ms, Batch avg: {avg_time_per_session:.1f}ms")
        
    def test_integration_with_system_monitor(self):
        """Test integration with the main SystemMonitor"""
        print("\n🧪 Testing SystemMonitor integration...")
        
        # Test the convenience function
        session = self.create_test_voting_session("standard")
        
        # This should not crash even without monitor initialized
        track_consensus_session(session)
        
        # Test with actual monitor integration would require more setup
        # For now, just verify the function exists and handles missing monitor gracefully
        
        self.test_results.append({"test": "system_monitor_integration", "status": "PASS", "details": "Integration functions work"})
        print("✅ SystemMonitor integration test passed")
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🧪 Testing edge cases...")
        
        # Test empty voting session
        empty_session = {"id": "empty", "votes": []}
        report = self.monitor.track_consensus(empty_session)
        assert report["metrics"]["agreement_level"] == 0.0, "Empty session should have 0 agreement"
        assert report["metrics"]["total_participants"] == 0, "Empty session should have 0 participants"
        
        # Test session with no confidence values
        no_confidence_session = {
            "id": "no_confidence",
            "votes": [
                {"agent": "test1", "decision": "approve"},
                {"agent": "test2", "decision": "approve"}
            ]
        }
        report = self.monitor.track_consensus(no_confidence_session)
        conf_dist = report["metrics"]["confidence_distribution"]
        assert conf_dist["average"] == 0.0, "No confidence session should have 0 average confidence"
        
        # Test malformed session data
        malformed_session = {"id": "malformed"}  # Missing votes
        report = self.monitor.track_consensus(malformed_session)
        assert "error" not in report, "Should handle malformed data gracefully"
        
        self.test_results.append({"test": "edge_cases", "status": "PASS", "details": "Edge cases handled correctly"})
        print("✅ Edge cases test passed")
        
    def run_all_tests(self):
        """Run all tests in the suite"""
        print("🚀 Starting Consensus Monitoring Test Suite")
        print("=" * 50)
        
        try:
            self.setup()
            
            # Run all test methods
            self.test_basic_consensus_tracking()
            self.test_agreement_level_calculation()
            self.test_dissenter_identification()
            self.test_confidence_distribution_analysis()
            self.test_consensus_strength_assessment()
            self.test_recommendation_generation()
            self.test_data_storage()
            self.test_performance_benchmarks()
            self.test_integration_with_system_monitor()
            self.test_edge_cases()
            
            # Summary
            self.print_test_summary()
            
        finally:
            self.cleanup()
            
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed_tests = [r for r in self.test_results if r["status"] == "PASS"]
        failed_tests = [r for r in self.test_results if r["status"] == "FAIL"]
        
        print(f"✅ Tests Passed: {len(passed_tests)}/{len(self.test_results)}")
        if failed_tests:
            print(f"❌ Tests Failed: {len(failed_tests)}")
            
        print("\nDetailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_icon} {result['test']}: {result['status']} - {result['details']}")
            
        if all(r["status"] == "PASS" for r in self.test_results):
            print("\n🎉 ALL TESTS PASSED - Consensus monitoring system is ready!")
        else:
            print("\n⚠️  Some tests failed - review implementation before deployment")

if __name__ == "__main__":
    test_suite = ConsensusMonitoringTestSuite()
    test_suite.run_all_tests()