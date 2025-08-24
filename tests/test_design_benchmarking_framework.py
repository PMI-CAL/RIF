#!/usr/bin/env python3
"""
Comprehensive test suite for RIF Design Specification Benchmarking Framework

Tests all components including:
- Design specification extraction
- Implementation analysis  
- Compliance scoring
- Overall benchmarking workflow
- Integration with GitHub issues
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Import the framework components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'systems'))

try:
    # Import with proper module name (hyphens replaced with underscores for Python)
    import importlib.util
    spec = importlib.util.spec_from_file_location("design_benchmarking_framework", 
                                                 os.path.join(os.path.dirname(__file__), '..', 'systems', 'design-benchmarking-framework.py'))
    design_benchmarking_framework = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design_benchmarking_framework)
    
    DesignSpecificationExtractor = design_benchmarking_framework.DesignSpecificationExtractor
    ImplementationAnalyzer = design_benchmarking_framework.ImplementationAnalyzer
    BenchmarkingEngine = design_benchmarking_framework.BenchmarkingEngine
    DesignSpecification = design_benchmarking_framework.DesignSpecification
    ImplementationEvidence = design_benchmarking_framework.ImplementationEvidence
    BenchmarkingResult = design_benchmarking_framework.BenchmarkingResult
    SpecificationType = design_benchmarking_framework.SpecificationType
    ComplianceLevel = design_benchmarking_framework.ComplianceLevel
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestDesignSpecificationExtractor:
    """Test the design specification extraction component"""
    
    def setup_method(self):
        self.extractor = DesignSpecificationExtractor()
    
    @patch('subprocess.run')
    def test_extract_specifications_from_issue_success(self, mock_run):
        """Test successful specification extraction from GitHub issue"""
        # Mock GitHub issue data
        mock_issue_data = {
            'body': '''
            # Implementation Requirements
            - The system must provide 90% accuracy in specification extraction
            - Implementation should complete within 2 minutes
            - Quality gates must enforce standards automatically
            
            ## Constraints
            - Cannot exceed 200MB memory usage
            - Must be compatible with existing RIF workflow
            ''',
            'title': 'Design Benchmarking Framework Implementation',
            'comments': [
                {
                    'body': 'RIF-Analyst analysis: The system shall validate compliance automatically'
                },
                {
                    'body': 'RIF-Planner recommendation: Performance should meet sub-200ms targets'
                }
            ]
        }
        
        mock_run.return_value.stdout = json.dumps(mock_issue_data)
        mock_run.return_value.check = True
        
        # Extract specifications
        specs = self.extractor.extract_specifications_from_issue(123)
        
        # Verify results
        assert len(specs) > 0
        assert any("90%" in spec.description for spec in specs)
        assert any(spec.priority == "must_have" for spec in specs)
        assert any(spec.measurable for spec in specs)
        
        # Verify subprocess was called correctly
        mock_run.assert_called_once_with(
            ["gh", "issue", "view", "123", "--json", "body,title,labels,comments"],
            capture_output=True, text=True, check=True
        )
    
    @patch('subprocess.run')
    def test_extract_specifications_github_error(self, mock_run):
        """Test handling of GitHub API errors"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'gh')
        
        specs = self.extractor.extract_specifications_from_issue(123)
        
        assert specs == []
    
    def test_extract_from_text_functional_requirements(self):
        """Test extraction of functional requirements"""
        text = """
        The system must provide real-time processing capabilities.
        Implementation should support concurrent users.
        The solution will integrate with external APIs.
        """
        
        specs = self.extractor._extract_from_text(text, 123)
        
        assert len(specs) >= 3
        functional_specs = [s for s in specs if s.type == SpecificationType.FUNCTIONAL_REQUIREMENTS]
        assert len(functional_specs) >= 3
        
        # Check priorities
        must_have_specs = [s for s in specs if s.priority == "must_have"]
        should_have_specs = [s for s in specs if s.priority == "should_have"]
        will_have_specs = [s for s in specs if s.priority == "could_have"]
        
        assert len(must_have_specs) >= 1
        assert len(should_have_specs) >= 1
    
    def test_extract_from_text_quality_gates(self):
        """Test extraction of quality gate specifications"""
        text = """
        Quality gate: Test coverage must exceed 80%
        Performance threshold should be under 100ms
        Security scanning must pass all checks
        """
        
        specs = self.extractor._extract_from_text(text, 123)
        
        quality_specs = [s for s in specs if s.type == SpecificationType.QUALITY_GATES]
        assert len(quality_specs) >= 1
        
        # Verify measurable detection
        measurable_specs = [s for s in specs if s.measurable]
        assert len(measurable_specs) >= 1
    
    def test_extract_acceptance_criteria(self):
        """Test acceptance criteria extraction"""
        text = """
        Requirements:
        - Verify system handles 1000+ concurrent requests
        - Ensure response time under 200ms
        - Confirm zero data loss under failure scenarios
        - [] Validate security compliance
        """
        
        criteria = self.extractor._extract_acceptance_criteria(text, "test requirement")
        
        assert len(criteria) >= 3
        assert any("1000+" in criterion for criterion in criteria)
        assert any("200ms" in criterion for criterion in criteria)
    
    def test_extract_success_metrics(self):
        """Test success metrics extraction"""
        text = """
        Target: 95% accuracy improvement
        Performance: under 150ms response time  
        Throughput: above 500 requests per second
        Error rate: zero failures expected
        """
        
        metrics = self.extractor._extract_success_metrics(text)
        
        assert "target_percentage" in metrics
        assert metrics["target_percentage"] == 95
        assert "max_time_ms" in metrics
        assert metrics["max_time_ms"] == 150
        assert "min_throughput" in metrics
        assert metrics["min_throughput"] == 500
    
    def test_priority_determination(self):
        """Test priority determination logic"""
        # Must have
        assert self.extractor._determine_priority("The system must provide security") == "must_have"
        assert self.extractor._determine_priority("Critical component shall validate") == "must_have"
        
        # Should have  
        assert self.extractor._determine_priority("Implementation should optimize") == "should_have"
        assert self.extractor._determine_priority("Important to consider performance") == "should_have"
        
        # Could have
        assert self.extractor._determine_priority("Nice to have feature") == "could_have"
    
    def test_measurable_detection(self):
        """Test measurable requirement detection"""
        assert self.extractor._is_measurable("90% test coverage required")
        assert self.extractor._is_measurable("Response time under 200 milliseconds")
        assert self.extractor._is_measurable("Memory usage below 1GB")
        assert not self.extractor._is_measurable("Should be user friendly")
        assert not self.extractor._is_measurable("Must be maintainable")
    
    def test_testable_detection(self):
        """Test testable requirement detection"""
        assert self.extractor._is_testable("System must validate input data")
        assert self.extractor._is_testable("Verify performance meets targets")
        assert self.extractor._is_testable("Test error handling scenarios")
        assert not self.extractor._is_testable("Code should be readable")


class TestImplementationAnalyzer:
    """Test the implementation analysis component"""
    
    def setup_method(self):
        self.analyzer = ImplementationAnalyzer("/tmp/test_repo")
        
        # Create mock specification
        self.mock_spec = DesignSpecification(
            id="spec-test-1",
            type=SpecificationType.FUNCTIONAL_REQUIREMENTS,
            description="System must provide real-time data processing",
            acceptance_criteria=["Process 1000+ events per second", "Sub-100ms latency"],
            constraints=["Memory usage under 1GB"],
            success_metrics={"min_throughput": 1000, "max_time_ms": 100},
            priority="must_have",
            measurable=True,
            testable=True,
            created_at=datetime.now(),
            issue_number=123
        )
    
    @patch('subprocess.run')
    def test_find_related_code_files_success(self, mock_run):
        """Test finding related code files"""
        mock_run.return_value.stdout = '''
        abc123 Update processing system
        def456 Add real-time functionality
        systems/real_time_processor.py
        tests/test_processor.py
        config/processing_config.yaml
        docs/processing_docs.md
        '''
        mock_run.return_value.check = True
        
        files = self.analyzer._find_related_code_files(123, self.mock_spec)
        
        assert len(files) > 0
        python_files = [f for f in files if f.endswith('.py')]
        assert len(python_files) > 0
    
    @patch('subprocess.run')
    def test_find_related_test_files_success(self, mock_run):
        """Test finding related test files"""
        mock_run.return_value.stdout = '''
        /tmp/test_repo/tests/test_processor.py
        /tmp/test_repo/tests/test_realtime.py
        /tmp/test_repo/integration_tests/test_processing.py
        '''
        mock_run.return_value.check = True
        
        test_files = self.analyzer._find_related_test_files(123, self.mock_spec)
        
        assert len(test_files) > 0
        assert any("test_processor.py" in f for f in test_files)
        assert any("test_realtime.py" in f for f in test_files)
    
    def test_extract_python_patterns(self):
        """Test Python code pattern extraction"""
        python_code = '''
        import os
        import asyncio
        from typing import Dict, List
        
        class RealtimeProcessor:
            def __init__(self):
                self.active = True
                
            def process_events(self, events: List[Dict]) -> bool:
                """Process events in real-time"""
                return True
                
            def validate_throughput(self) -> int:
                """Validate processing throughput"""
                return 1200
        
        def main():
            processor = RealtimeProcessor()
            processor.process_events([])
        '''
        
        patterns = self.analyzer._extract_python_patterns(python_code, self.mock_spec)
        
        assert len(patterns) >= 3  # Functions, classes, imports
        assert any("RealtimeProcessor" in pattern for pattern in patterns)
        assert any("process_events" in pattern for pattern in patterns)
        assert any("os, asyncio" in pattern for pattern in patterns)
    
    def test_extract_javascript_patterns(self):
        """Test JavaScript code pattern extraction"""
        js_code = '''
        const express = require('express');
        const { EventProcessor } = require('./processors');
        
        class RealtimeAPI {
            constructor() {
                this.processor = new EventProcessor();
            }
        }
        
        function processEvent(data) {
            return data.process();
        }
        
        const validateThroughput = () => {
            return processor.getMetrics();
        };
        
        export default RealtimeAPI;
        export { processEvent, validateThroughput };
        '''
        
        patterns = self.analyzer._extract_javascript_patterns(js_code, self.mock_spec)
        
        assert len(patterns) >= 2  # Functions and exports
        assert any("processEvent" in pattern for pattern in patterns)
        assert any("RealtimeAPI" in pattern for pattern in patterns)
    
    def test_calculate_compliance_score_high_compliance(self):
        """Test compliance score calculation for high compliance"""
        detailed_implementation = """
        Functions: process_events, validate_throughput, handle_realtime_data
        Classes: RealtimeProcessor, EventHandler
        Key imports: asyncio, time, threading
        LOC: ~450
        Spec keywords found: real, time, processing, system
        """
        
        code_files = ["systems/processor.py", "lib/realtime.py"]
        test_files = ["tests/test_processor.py", "tests/test_realtime.py"]
        
        score = self.analyzer._calculate_compliance_score(
            self.mock_spec, detailed_implementation, code_files, test_files
        )
        
        assert score >= 0.8  # Should be high compliance
    
    def test_calculate_compliance_score_low_compliance(self):
        """Test compliance score calculation for low compliance"""
        minimal_implementation = "LOC: ~10"
        code_files = []
        test_files = []
        
        score = self.analyzer._calculate_compliance_score(
            self.mock_spec, minimal_implementation, code_files, test_files
        )
        
        assert score <= 0.3  # Should be low compliance
    
    def test_determine_compliance_level(self):
        """Test compliance level determination"""
        assert self.analyzer._determine_compliance_level(0.95) == ComplianceLevel.FULLY_COMPLIANT
        assert self.analyzer._determine_compliance_level(0.85) == ComplianceLevel.MOSTLY_COMPLIANT
        assert self.analyzer._determine_compliance_level(0.65) == ComplianceLevel.PARTIALLY_COMPLIANT
        assert self.analyzer._determine_compliance_level(0.45) == ComplianceLevel.MINIMALLY_COMPLIANT
        assert self.analyzer._determine_compliance_level(0.25) == ComplianceLevel.NON_COMPLIANT
    
    def test_identify_issues(self):
        """Test issue identification logic"""
        # Low compliance score
        issues = self.analyzer._identify_issues(self.mock_spec, "minimal impl", 0.4)
        assert any("Low compliance score" in issue for issue in issues)
        
        # Missing tests for testable spec
        issues = self.analyzer._identify_issues(self.mock_spec, "no test mention", 0.8)
        assert any("testable but no test evidence" in issue for issue in issues)
        
        # Must-have with low compliance
        issues = self.analyzer._identify_issues(self.mock_spec, "basic impl", 0.6)
        assert any("Must-have requirement" in issue for issue in issues)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Low compliance
        recs = self.analyzer._generate_recommendations(self.mock_spec, 0.5, ["issue1", "issue2"])
        assert any("Improve implementation" in rec for rec in recs)
        
        # Missing tests
        recs = self.analyzer._generate_recommendations(self.mock_spec, 0.8, [])
        assert any("comprehensive tests" in rec for rec in recs)
        
        # Must-have low compliance
        recs = self.analyzer._generate_recommendations(self.mock_spec, 0.7, [])
        assert any("immediate attention" in rec for rec in recs)


class TestBenchmarkingEngine:
    """Test the main benchmarking engine"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = BenchmarkingEngine(self.temp_dir)
        
        # Mock dependencies
        self.engine.extractor = Mock(spec=DesignSpecificationExtractor)
        self.engine.analyzer = Mock(spec=ImplementationAnalyzer)
    
    def teardown_method(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_issue_complete_workflow(self):
        """Test complete benchmarking workflow"""
        # Mock specifications
        mock_specs = [
            DesignSpecification(
                id="spec-test-1",
                type=SpecificationType.FUNCTIONAL_REQUIREMENTS,
                description="System must process data",
                acceptance_criteria=["Process 100 items/sec"],
                constraints=["Memory < 1GB"],
                success_metrics={"min_throughput": 100},
                priority="must_have",
                measurable=True,
                testable=True,
                created_at=datetime.now(),
                issue_number=123
            )
        ]
        
        # Mock evidence
        mock_evidence = [
            ImplementationEvidence(
                spec_id="spec-test-1",
                implementation_details="Functions: process_data | Classes: DataProcessor",
                code_files=["src/processor.py"],
                test_files=["tests/test_processor.py"],
                documentation_refs=[],
                metrics_achieved={"throughput": 120},
                compliance_score=0.85,
                compliance_level=ComplianceLevel.MOSTLY_COMPLIANT,
                issues_found=[],
                recommendations=[],
                evidence_timestamp=datetime.now()
            )
        ]
        
        # Configure mocks
        self.engine.extractor.extract_specifications_from_issue.return_value = mock_specs
        self.engine.analyzer.analyze_implementation.return_value = mock_evidence
        
        # Run benchmarking
        result = self.engine.benchmark_issue(123, "Test notes")
        
        # Verify results
        assert result.issue_number == 123
        assert result.validator_notes == "Test notes"
        assert len(result.specifications) == 1
        assert len(result.evidence) == 1
        assert result.overall_adherence_score == 0.85
        assert result.overall_compliance_level == ComplianceLevel.MOSTLY_COMPLIANT
        assert result.quality_grade in ["A", "A-", "B+", "B", "B-"]
    
    def test_calculate_overall_adherence(self):
        """Test overall adherence calculation"""
        evidence = [
            Mock(compliance_score=0.9),
            Mock(compliance_score=0.8),
            Mock(compliance_score=0.7)
        ]
        
        score = self.engine._calculate_overall_adherence(evidence)
        assert score == 0.8  # Average of 0.9, 0.8, 0.7
    
    def test_calculate_overall_adherence_empty(self):
        """Test overall adherence with no evidence"""
        score = self.engine._calculate_overall_adherence([])
        assert score == 0.0
    
    def test_calculate_quality_grade(self):
        """Test quality grade calculation"""
        # High performance
        evidence = [Mock(implementation_details="high quality implementation")]
        assert self.engine._calculate_quality_grade(0.97, evidence) == "A+"
        assert self.engine._calculate_quality_grade(0.93, evidence) == "A"
        assert self.engine._calculate_quality_grade(0.87, evidence) == "B+"
        assert self.engine._calculate_quality_grade(0.75, evidence) == "C"
        assert self.engine._calculate_quality_grade(0.50, evidence) == "F"
    
    def test_identify_constraint_violations(self):
        """Test constraint violation identification"""
        specs = [
            Mock(id="spec-1", constraints=["Memory < 1GB"], priority="must_have"),
            Mock(id="spec-2", constraints=["Response < 100ms"], priority="should_have")
        ]
        
        evidence = [
            Mock(spec_id="spec-1", compliance_score=0.6),  # Low compliance
            Mock(spec_id="spec-2", compliance_score=0.8)   # Good compliance
        ]
        
        violations = self.engine._identify_constraint_violations(specs, evidence)
        
        assert len(violations) == 1  # Only spec-1 should have violations
        assert violations[0]["severity"] == "high"  # Must-have priority
        assert "Memory < 1GB" in violations[0]["constraint"]
    
    def test_generate_benchmarking_report(self):
        """Test benchmarking report generation"""
        # Create mock result
        result = BenchmarkingResult(
            issue_number=123,
            specifications=[
                Mock(id="spec-1", type=SpecificationType.FUNCTIONAL_REQUIREMENTS)
            ],
            evidence=[
                Mock(spec_id="spec-1", compliance_score=0.85, 
                     compliance_level=ComplianceLevel.MOSTLY_COMPLIANT,
                     issues_found=["Minor issue"])
            ],
            overall_adherence_score=0.85,
            overall_compliance_level=ComplianceLevel.MOSTLY_COMPLIANT,
            constraint_violations=[
                {"constraint": "Memory limit", "severity": "medium", "description": "Exceeded by 10%"}
            ],
            goal_achievement={"functional_requirements": 0.85},
            quality_grade="B+",
            recommendations=["Add more tests", "Improve documentation"],
            benchmarking_timestamp=datetime.now(),
            validator_notes="Good implementation overall"
        )
        
        report = self.engine.generate_benchmarking_report(result)
        
        # Verify report content
        assert "# Design Specification Benchmarking Report" in report
        assert "Issue**: #123" in report
        assert "Overall Grade**: B+" in report
        assert "85.0%" in report
        assert "Mostly Compliant" in report
        assert "Memory limit" in report
        assert "Add more tests" in report
        assert "Good implementation overall" in report


class TestIntegrationScenarios:
    """Integration tests with real GitHub scenarios"""
    
    @patch('subprocess.run')
    def test_real_issue_benchmarking_workflow(self, mock_run):
        """Test benchmarking workflow with realistic GitHub issue data"""
        # Mock realistic GitHub issue
        mock_issue_data = {
            'body': '''
            # DPIBS Phase 2 Implementation
            
            ## Requirements
            - System must achieve 90% accuracy in specification extraction
            - Implementation should complete benchmarking within 2 minutes
            - Grade assignments must align with human expert assessment 85%+ of cases
            
            ## Success Criteria
            - [ ] 100% core functionality tests passing
            - [ ] Performance targets achieved under realistic load
            - [ ] Seamless integration with existing RIF-Validator workflow
            
            ## Constraints
            - Memory usage cannot exceed 512MB during analysis
            - Compatible with existing GitHub issue format
            - Must support A-F grading with objective criteria
            ''',
            'title': 'DPIBS Development Phase 2: Design Specification Benchmarking',
            'comments': [
                {
                    'body': 'RIF-Analyst: Requirements analysis shows high complexity implementation needed.'
                },
                {
                    'body': 'RIF-Planner: Recommend NLP-based extraction with pattern matching validation.'
                }
            ]
        }
        
        mock_run.return_value.stdout = json.dumps(mock_issue_data)
        mock_run.return_value.check = True
        
        # Create temporary repository
        with tempfile.TemporaryDirectory() as temp_repo:
            # Create mock implementation files
            impl_file = os.path.join(temp_repo, "systems", "design_benchmarking.py")
            os.makedirs(os.path.dirname(impl_file), exist_ok=True)
            
            with open(impl_file, 'w') as f:
                f.write('''
                class DesignBenchmarkingSystem:
                    def extract_specifications(self, text):
                        """Extract design specifications with 90% accuracy"""
                        return self.nlp_processor.process(text)
                    
                    def grade_implementation(self, specs, evidence):
                        """Generate A-F grade based on compliance"""
                        score = self.calculate_adherence(specs, evidence)
                        return self.convert_to_letter_grade(score)
                ''')
            
            # Create test file
            test_file = os.path.join(temp_repo, "tests", "test_benchmarking.py")
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            
            with open(test_file, 'w') as f:
                f.write('''
                def test_specification_extraction_accuracy():
                    """Test 90% accuracy requirement"""
                    assert accuracy >= 0.9
                    
                def test_grading_alignment():
                    """Test 85% alignment with expert assessment"""
                    assert expert_alignment >= 0.85
                ''')
            
            # Mock git log for file discovery
            git_mock = Mock()
            git_mock.stdout = f"systems/design_benchmarking.py\ntests/test_benchmarking.py"
            git_mock.check = True
            
            with patch('subprocess.run', side_effect=[mock_run, git_mock, git_mock]):
                engine = BenchmarkingEngine(temp_repo)
                result = engine.benchmark_issue(124)
                
                # Verify realistic results
                assert result.issue_number == 124
                assert len(result.specifications) >= 3  # Should extract multiple specs
                assert result.overall_adherence_score > 0.0
                assert result.quality_grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
                
                # Verify accuracy requirements detected
                accuracy_specs = [s for s in result.specifications if "90%" in s.description]
                assert len(accuracy_specs) > 0
                
                # Verify performance requirements detected
                perf_specs = [s for s in result.specifications if "2 minutes" in s.description or "minutes" in str(s.constraints)]
                assert len(perf_specs) > 0


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "extractor":
            pytest.main(["-v", "-k", "TestDesignSpecificationExtractor"])
        elif test_category == "analyzer":
            pytest.main(["-v", "-k", "TestImplementationAnalyzer"])
        elif test_category == "engine":
            pytest.main(["-v", "-k", "TestBenchmarkingEngine"])
        elif test_category == "integration":
            pytest.main(["-v", "-k", "TestIntegrationScenarios"])
    else:
        # Run all tests
        pytest.main(["-v", __file__])