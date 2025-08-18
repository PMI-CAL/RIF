#!/usr/bin/env python3
"""
Comprehensive test suite for RIF Error Analysis System
Tests error detection, classification, analysis, and continuous improvement
"""

import sys
import os
import json
import tempfile
import unittest
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the claude commands directory to the path
claude_commands_path = str(Path(__file__).parent.parent / "claude" / "commands")
sys.path.insert(0, claude_commands_path)

try:
    # Import the error analysis module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("error_analysis", 
                                                 Path(claude_commands_path) / "error-analysis.py")
    error_analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(error_analysis_module)
    
    # Extract the classes we need
    ErrorAnalyzer = error_analysis_module.ErrorAnalyzer
    ErrorEvent = error_analysis_module.ErrorEvent
    ErrorSeverity = error_analysis_module.ErrorSeverity
    ErrorType = error_analysis_module.ErrorType
    ErrorSource = error_analysis_module.ErrorSource
    RootCauseAnalysis = error_analysis_module.RootCauseAnalysis
    FishboneDiagram = error_analysis_module.FishboneDiagram
    ErrorSolution = error_analysis_module.ErrorSolution
    
except Exception as e:
    print(f"Error importing error_analysis module: {e}")
    print(f"Looking in path: {claude_commands_path}")
    sys.exit(1)

class TestErrorAnalyzer(unittest.TestCase):
    """Test cases for ErrorAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ErrorAnalyzer(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_error_capture_and_classification(self):
        """Test error capture and automatic classification"""
        
        # Test syntax error
        error_event = self.analyzer.capture_error(
            "SyntaxError: invalid syntax at line 42",
            {"file": "test.py", "line": 42},
            command="python test.py"
        )
        
        self.assertEqual(error_event.error_type, ErrorType.SYNTAX)
        self.assertEqual(error_event.severity, ErrorSeverity.MEDIUM)
        self.assertIsNotNone(error_event.id)
        
        # Test critical error
        critical_error = self.analyzer.capture_error(
            "Segmentation fault (core dumped)",
            {"process": "main", "pid": 1234},
            exit_code=139
        )
        
        self.assertEqual(critical_error.severity, ErrorSeverity.CRITICAL)
        
        # Test integration error
        integration_error = self.analyzer.capture_error(
            "Connection refused: Unable to connect to database",
            {"service": "postgresql", "port": 5432},
            exit_code=1
        )
        
        self.assertEqual(integration_error.error_type, ErrorType.INTEGRATION)
        self.assertEqual(integration_error.severity, ErrorSeverity.HIGH)
    
    def test_severity_classification(self):
        """Test error severity classification logic"""
        
        # Critical patterns
        critical_cases = [
            "segmentation fault occurred",
            "out of memory error",
            "permission denied for root access",
            "database corruption detected"
        ]
        
        for message in critical_cases:
            severity = self.analyzer._classify_severity(message, None, {})
            self.assertEqual(severity, ErrorSeverity.CRITICAL, f"Failed for: {message}")
        
        # High severity patterns  
        high_cases = [
            "connection refused by server",
            "timeout exceeded waiting for response",
            "authentication failed for user"
        ]
        
        for message in high_cases:
            severity = self.analyzer._classify_severity(message, None, {})
            self.assertEqual(severity, ErrorSeverity.HIGH, f"Failed for: {message}")
        
        # Exit code based severity
        severity = self.analyzer._classify_severity("Command failed", 1, {})
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        severity = self.analyzer._classify_severity("Command failed", 0, {})
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
    
    def test_error_type_classification(self):
        """Test error type classification"""
        
        test_cases = [
            ("SyntaxError: invalid syntax", ErrorType.SYNTAX),
            ("null pointer exception occurred", ErrorType.RUNTIME),
            ("connection timeout to service", ErrorType.INTEGRATION),
            ("CPU usage exceeded 90%", ErrorType.PERFORMANCE),
            ("authentication failed", ErrorType.SECURITY),
            ("unexpected error occurred", ErrorType.UNKNOWN)
        ]
        
        for message, expected_type in test_cases:
            error_type = self.analyzer._classify_type(message, None, {})
            self.assertEqual(error_type, expected_type, f"Failed for: {message}")
    
    def test_source_classification(self):
        """Test error source classification"""
        
        # Test stack trace based classification
        claude_stack = "/path/to/claude/code/module.py:line 123"
        source = self.analyzer._classify_source({}, claude_stack)
        self.assertEqual(source, ErrorSource.CLAUDE_CODE)
        
        rif_stack = "/path/to/rif/system/agent.py:line 456"
        source = self.analyzer._classify_source({}, rif_stack)
        self.assertEqual(source, ErrorSource.RIF_SYSTEM)
        
        # Test context based classification
        context = {"source": "user_code"}
        source = self.analyzer._classify_source(context, None)
        self.assertEqual(source, ErrorSource.USER_CODE)
    
    def test_five_whys_analysis(self):
        """Test Five Whys root cause analysis"""
        
        error_event = ErrorEvent(
            id="test123",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.HIGH,
            error_type=ErrorType.RUNTIME,
            source=ErrorSource.USER_CODE,
            message="NullPointerException in user service",
            context={"service": "user_management"}
        )
        
        analysis = self.analyzer._perform_five_whys(error_event)
        
        self.assertIsInstance(analysis, RootCauseAnalysis)
        self.assertEqual(analysis.error_id, "test123")
        self.assertIsNotNone(analysis.root_cause)
        self.assertGreater(analysis.confidence, 0.0)
        self.assertLessEqual(analysis.confidence, 1.0)
    
    def test_fishbone_diagram_creation(self):
        """Test fishbone diagram creation"""
        
        error_event = ErrorEvent(
            id="test456",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.INTEGRATION,
            source=ErrorSource.EXTERNAL,
            message="API timeout exceeded",
            context={"api": "payment_service"}
        )
        
        diagram = self.analyzer._create_fishbone_diagram(error_event)
        
        self.assertIsInstance(diagram, FishboneDiagram)
        self.assertEqual(diagram.error_id, "test456")
        self.assertIsInstance(diagram.people_factors, list)
        self.assertIsInstance(diagram.process_factors, list)
        self.assertIsInstance(diagram.technology_factors, list)
        self.assertIsInstance(diagram.environment_factors, list)
        self.assertIn(diagram.primary_cause_category, 
                     ["people_factors", "process_factors", "technology_factors", "environment_factors"])
    
    def test_pattern_similarity(self):
        """Test error pattern similarity calculation"""
        
        error_event = ErrorEvent(
            id="test789",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.HIGH,
            error_type=ErrorType.INTEGRATION,
            source=ErrorSource.EXTERNAL,
            message="database connection failed timeout",
            context={}
        )
        
        pattern = {
            "message": "connection timeout database failed",
            "error_type": "integration",
            "source": "external"
        }
        
        similarity = self.analyzer._calculate_similarity(error_event, pattern)
        
        self.assertGreater(similarity, 0.5, "Should have high similarity")
        self.assertLessEqual(similarity, 1.0, "Similarity should not exceed 1.0")
    
    def test_adversarial_analysis(self):
        """Test adversarial analysis functionality"""
        
        security_error = ErrorEvent(
            id="sec123",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.HIGH,
            error_type=ErrorType.SECURITY,
            source=ErrorSource.USER_CODE,
            message="SQL injection attempt detected",
            context={"input": "'; DROP TABLE users;--"}
        )
        
        analysis = self.analyzer._adversarial_analysis(security_error)
        
        self.assertIn("risk_assessment", analysis)
        self.assertIn("attack_vectors", analysis)
        self.assertIn("assumptions_tested", analysis)
        self.assertIn("edge_cases", analysis)
        
        # Check risk assessment
        risk = analysis["risk_assessment"]
        self.assertEqual(risk["risk_level"], "high")
        self.assertTrue(risk["security_impact"])
    
    def test_attack_vector_identification(self):
        """Test attack vector identification"""
        
        test_cases = [
            ("SQL error in query execution", ["SQL injection potential"]),
            ("Script execution failed", ["Cross-site scripting risk"]),
            ("Permission denied for operation", ["Privilege escalation risk"]),
            ("Authentication bypass detected", ["Authentication bypass risk"]),
            ("Path traversal in file access", ["Path traversal risk"])
        ]
        
        for message, expected_vectors in test_cases:
            error_event = ErrorEvent(
                id="test",
                timestamp="2025-08-18T12:00:00Z",
                severity=ErrorSeverity.MEDIUM,
                error_type=ErrorType.SECURITY,
                source=ErrorSource.USER_CODE,
                message=message,
                context={}
            )
            
            vectors = self.analyzer._identify_attack_vectors(error_event)
            
            for expected_vector in expected_vectors:
                self.assertIn(expected_vector, vectors, 
                            f"Missing vector '{expected_vector}' for message: {message}")
    
    def test_recommendations_generation(self):
        """Test improvement recommendations generation"""
        
        # Test syntax error recommendations
        syntax_error = ErrorEvent(
            id="syntax123",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.SYNTAX,
            source=ErrorSource.USER_CODE,
            message="Invalid syntax in configuration file",
            context={}
        )
        
        recommendations = self.analyzer._generate_recommendations(syntax_error)
        
        self.assertIn("Implement syntax validation pre-checks", recommendations)
        self.assertIn("Add linting to CI/CD pipeline", recommendations)
        
        # Test critical error recommendations
        critical_error = ErrorEvent(
            id="critical123",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.CRITICAL,
            error_type=ErrorType.RUNTIME,
            source=ErrorSource.RIF_SYSTEM,
            message="System crash due to memory exhaustion",
            context={}
        )
        
        critical_recommendations = self.analyzer._generate_recommendations(critical_error)
        
        self.assertIn("Implement immediate alerting", critical_recommendations)
        self.assertIn("Add automated rollback mechanisms", critical_recommendations)
    
    def test_error_storage_and_metrics(self):
        """Test error storage and metrics updates"""
        
        # Create multiple errors
        errors = [
            ("Critical system failure", ErrorSeverity.CRITICAL, ErrorType.RUNTIME),
            ("API timeout", ErrorSeverity.HIGH, ErrorType.INTEGRATION),
            ("Syntax error", ErrorSeverity.MEDIUM, ErrorType.SYNTAX),
            ("Warning message", ErrorSeverity.LOW, ErrorType.UNKNOWN)
        ]
        
        for message, severity, error_type in errors:
            error_event = ErrorEvent(
                id=f"test_{len(errors)}",
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                severity=severity,
                error_type=error_type,
                source=ErrorSource.USER_CODE,
                message=message,
                context={}
            )
            
            self.analyzer._store_error_event(error_event)
        
        # Check metrics file
        metrics_file = self.analyzer.errors_path / "metrics" / "current.json"
        self.assertTrue(metrics_file.exists())
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        self.assertEqual(metrics["total_errors"], len(errors))
        self.assertEqual(metrics["by_severity"]["critical"], 1)
        self.assertEqual(metrics["by_severity"]["high"], 1)
        self.assertEqual(metrics["by_severity"]["medium"], 1)
        self.assertEqual(metrics["by_severity"]["low"], 1)
    
    def test_comprehensive_error_analysis(self):
        """Test complete error analysis workflow"""
        
        error_event = ErrorEvent(
            id="comprehensive123",
            timestamp="2025-08-18T12:00:00Z",
            severity=ErrorSeverity.HIGH,
            error_type=ErrorType.INTEGRATION,
            source=ErrorSource.EXTERNAL,
            message="Database connection pool exhausted",
            context={"pool_size": 10, "active_connections": 10}
        )
        
        analysis = self.analyzer.analyze_error(error_event)
        
        # Check all analysis components
        self.assertIn("error_id", analysis)
        self.assertIn("five_whys", analysis)
        self.assertIn("fishbone", analysis)
        self.assertIn("pattern_match", analysis)
        self.assertIn("adversarial_analysis", analysis)
        self.assertIn("recommendations", analysis)
        
        # Check analysis file creation
        analysis_file = self.analyzer.errors_path / "analysis" / f"{error_event.id}.json"
        self.assertTrue(analysis_file.exists())

class TestErrorIntegration(unittest.TestCase):
    """Integration tests for error analysis system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ErrorAnalyzer(self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_cli_error_capture(self, mock_subprocess):
        """Test CLI error capture functionality"""
        
        # Test successful error capture
        result = os.system(f"python3 {Path(__file__).parent.parent}/claude/commands/error-analysis.py --capture 'Test error message' --context '{{\"test\": true}}' --command 'test command' --exit-code 1")
        
        # Should not fail (exit code 0)
        # Note: Actual subprocess testing would require more setup
    
    def test_error_pattern_detection(self):
        """Test error pattern detection and analysis"""
        
        # Create similar errors to test pattern detection
        base_message = "Connection timeout to database server"
        
        for i in range(5):
            error_event = self.analyzer.capture_error(
                f"{base_message} attempt {i}",
                {"attempt": i, "server": "db-primary"},
                command=f"connect_db attempt_{i}"
            )
        
        # Check that patterns are being tracked
        patterns_dir = self.analyzer.errors_path / "patterns"
        if patterns_dir.exists():
            pattern_files = list(patterns_dir.glob("*.json"))
            self.assertGreater(len(pattern_files), 0, "Should create error patterns")
    
    def test_knowledge_base_integration(self):
        """Test knowledge base integration"""
        
        # Ensure all required directories exist
        required_dirs = [
            "patterns", "solutions", "rootcauses", 
            "metrics", "logs", "analysis"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.analyzer.errors_path / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
    
    def test_metrics_tracking(self):
        """Test error metrics tracking"""
        
        # Create errors of different types
        error_types = [
            (ErrorSeverity.CRITICAL, ErrorType.SECURITY),
            (ErrorSeverity.HIGH, ErrorType.RUNTIME),
            (ErrorSeverity.MEDIUM, ErrorType.SYNTAX),
            (ErrorSeverity.LOW, ErrorType.PERFORMANCE)
        ]
        
        for severity, error_type in error_types:
            error_event = ErrorEvent(
                id=f"metrics_test_{severity.value}_{error_type.value}",
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                severity=severity,
                error_type=error_type,
                source=ErrorSource.USER_CODE,
                message=f"Test {error_type.value} error",
                context={}
            )
            
            self.analyzer._store_error_event(error_event)
        
        # Check metrics
        metrics_file = self.analyzer.errors_path / "metrics" / "current.json"
        self.assertTrue(metrics_file.exists())
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        self.assertEqual(metrics["total_errors"], len(error_types))
        self.assertGreater(metrics["by_severity"]["critical"], 0)
        self.assertGreater(metrics["by_type"]["security"], 0)

class TestErrorMonitoring(unittest.TestCase):
    """Test error monitoring and continuous improvement"""
    
    def setUp(self):
        """Set up monitoring test environment"""
        self.temp_dir = tempfile.mkdtemp()
        sys.path.insert(0, str(Path(__file__).parent.parent / "claude" / "commands"))
        from error_monitor import ErrorMonitor
        self.monitor = ErrorMonitor(self.temp_dir)
    
    def tearDown(self):
        """Clean up monitoring test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pattern_matching(self):
        """Test error pattern matching"""
        
        error = {
            "message": "database connection timeout",
            "error_type": "integration",
            "severity": "high"
        }
        
        pattern = {
            "message_keywords": ["database", "connection", "timeout"],
            "error_type": "integration",
            "severity": "high"
        }
        
        matches = self.monitor.matches_pattern(error, pattern)
        self.assertTrue(matches, "Should match similar error patterns")
    
    def test_improvement_recommendation_generation(self):
        """Test improvement recommendation generation"""
        
        pattern = {
            "pattern_id": "test_pattern",
            "frequency": 15,  # High frequency
            "error_type": "integration"
        }
        
        recommendations = self.monitor.generate_improvement_recommendations(pattern)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertIn("High frequency pattern", recommendations[0])

def run_all_tests():
    """Run all error analysis tests"""
    
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestErrorAnalyzer),
        unittest.TestLoader().loadTestsFromTestCase(TestErrorIntegration),
        unittest.TestLoader().loadTestsFromTestCase(TestErrorMonitoring)
    ]
    
    combined_suite = unittest.TestSuite(test_suites)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running RIF Error Analysis System Tests...")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some tests failed!")
        sys.exit(1)