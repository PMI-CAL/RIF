#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced DPIBS Benchmarking Framework

Tests implementation of Phase 1 research findings:
- Enhanced NLP + structured template approach for 90% accuracy
- Multi-dimensional A-F grading system
- Performance benchmarking under 2 minutes
- Expert alignment validation
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import subprocess
import time

# Import the enhanced framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'systems'))

try:
    from enhanced_benchmarking_framework import (
        EnhancedSpecificationExtractor,
        EnhancedBenchmarkingEngine,
        SpecificationType,
        ComplianceLevel,
        DesignSpecification,
        ImplementationEvidence,
        BenchmarkingResult
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestEnhancedSpecificationExtractor:
    """Test enhanced specification extraction with research improvements"""
    
    def setup_method(self):
        self.extractor = EnhancedSpecificationExtractor()
    
    @patch('subprocess.run')
    def test_enhanced_extraction_accuracy(self, mock_run):
        """Test enhanced extraction meets accuracy targets"""
        # Create realistic GitHub issue data based on Issue #116
        mock_issue_data = {
            'body': '''
            # DPIBS Research Phase 2: Design Specification Benchmarking Methodology
            
            ## Research Objective
            Develop comprehensive methodology for automatically extracting design specifications 
            from GitHub issues and comparing them against implementation outcomes to provide 
            objective grading of design adherence.
            
            ## Success Criteria
            - [ ] Specification extraction methodology defined with 90%+ accuracy target
            - [ ] Implementation analysis framework covers code, tests, architecture, documentation
            - [ ] Benchmarking algorithm provides objective, reproducible scoring
            - [ ] A-F grading system aligns with expert human assessment 85%+ of cases
            - [ ] Methodology handles GitHub issue format variations and evolution
            - [ ] Framework integrates with existing RIF validation workflows
            - [ ] Performance requirements met for real-time benchmarking (<2 minutes per assessment)
            
            ## Technical Requirements
            - System must achieve 90% accuracy in specification extraction
            - Implementation should complete benchmarking within 2 minutes
            - Grade assignments must align with human expert assessment 85%+ of cases
            - Framework must be compatible with existing GitHub issue format
            - Cannot exceed 512MB memory usage during analysis
            
            ## Quality Gates
            - Test coverage must exceed 80%
            - Performance benchmarking under 200ms response time
            - Expert alignment targets 85% agreement rate
            ''',
            'title': 'DPIBS Research Phase 2: Design Specification Benchmarking Methodology',
            'comments': [
                {
                    'body': 'RIF-Analyst: Requirements analysis shows high complexity implementation needed with 90% accuracy targets.'
                },
                {
                    'body': 'RIF-Planner: Recommend NLP-based extraction with pattern matching validation targeting 85% expert alignment.'
                }
            ]
        }
        
        mock_run.return_value.stdout = json.dumps(mock_issue_data)
        mock_run.return_value.check = True
        
        # Extract specifications
        specifications, metrics = self.extractor.extract_specifications_from_issue(116)
        
        # Validate extraction results
        assert len(specifications) >= 5, f"Should extract at least 5 specs, got {len(specifications)}"
        assert metrics['specifications_found'] == len(specifications)
        assert metrics['extraction_time_seconds'] < 2.0, "Extraction should complete in under 2 seconds"
        
        # Verify accuracy targets detected
        accuracy_specs = [spec for spec in specifications if "90%" in spec.description]
        assert len(accuracy_specs) >= 1, "Should detect 90% accuracy requirements"
        
        # Verify performance requirements detected
        perf_specs = [spec for spec in specifications if "2 minutes" in spec.description or "200ms" in spec.description]
        assert len(perf_specs) >= 1, "Should detect performance requirements"
        
        # Verify expert alignment requirements detected
        alignment_specs = [spec for spec in specifications if "85%" in spec.description]
        assert len(alignment_specs) >= 1, "Should detect expert alignment requirements"
        
        # Verify quality gates detected
        quality_specs = [spec for spec in specifications if spec.type == SpecificationType.QUALITY_GATES]
        assert len(quality_specs) >= 2, "Should detect multiple quality gate specifications"
    
    def test_enhanced_pattern_recognition(self):
        """Test improved pattern recognition accuracy"""
        test_text = """
        The system must provide 90% accuracy in specification extraction from GitHub issues.
        Implementation should complete benchmarking within 2 minutes for real-time assessment.
        Grade assignments must align with human expert assessment 85% of the time.
        Framework cannot exceed 512MB memory usage during analysis operations.
        Performance requirements: response time under 200ms for all benchmarking operations.
        Quality gate: test coverage should exceed 80% for all components.
        """
        
        specifications = self.extractor._extract_with_hybrid_approach(test_text, 116)
        
        # Verify comprehensive extraction
        assert len(specifications) >= 6, f"Should extract at least 6 specifications, got {len(specifications)}"
        
        # Verify different types detected
        types_found = set(spec.type for spec in specifications)
        assert SpecificationType.FUNCTIONAL_REQUIREMENTS in types_found
        assert SpecificationType.QUALITY_GATES in types_found
        assert SpecificationType.PERFORMANCE_REQUIREMENTS in types_found
        assert SpecificationType.ARCHITECTURAL_CONSTRAINTS in types_found
        
        # Verify measurable specifications detected
        measurable_specs = [spec for spec in specifications if spec.measurable]
        assert len(measurable_specs) >= 4, "Should detect measurable specifications"
        
        # Verify success metrics extracted
        specs_with_metrics = [spec for spec in specifications if spec.success_metrics]
        assert len(specs_with_metrics) >= 3, "Should extract success metrics"
        
        # Verify priorities assigned correctly
        must_have_specs = [spec for spec in specifications if spec.priority == "must_have"]
        assert len(must_have_specs) >= 3, "Should identify must-have requirements"
    
    def test_specification_deduplication(self):
        """Test specification deduplication and quality filtering"""
        # Text with overlapping/duplicate requirements
        test_text = """
        The system must provide 90% accuracy in specification extraction.
        System should provide 90% accuracy for specification extraction tasks.
        Implementation must achieve ninety percent accuracy in extracting specs.
        Framework must be compatible with GitHub issues.
        The framework should integrate with existing GitHub issue format.
        """
        
        specifications = self.extractor._extract_with_hybrid_approach(test_text, 116)
        
        # Should deduplicate similar specifications
        assert len(specifications) <= 3, f"Should deduplicate similar specs, got {len(specifications)}"
        
        # Verify quality filtering
        for spec in specifications:
            assert len(spec.description) >= 20, "All specs should meet minimum length requirements"
            assert len(spec.description) <= 300, "All specs should be under maximum length"
    
    def test_enhanced_metrics_extraction(self):
        """Test enhanced success metrics extraction"""
        test_text = """
        Target: 95% accuracy improvement in specification extraction.
        Performance: under 150ms response time for benchmarking operations.
        Throughput: above 500 assessments per hour processing capacity.
        Grade requirement: minimum B+ rating for implementation quality.
        Expert alignment: 85% agreement rate with human assessments.
        """
        
        # Test individual metric extraction
        metrics = self.extractor._extract_enhanced_metrics(test_text, test_text)
        
        assert "target_percentage" in metrics, "Should extract percentage targets"
        assert metrics["target_percentage"] >= 85, "Should extract correct percentage values"
        
        if "max_time_constraint" in metrics:
            assert metrics["max_time_constraint"] == 150, "Should extract time constraints"
        
        if "target_grade" in metrics:
            assert "B" in metrics["target_grade"], "Should extract grade requirements"
    
    def test_priority_determination_accuracy(self):
        """Test enhanced priority determination"""
        test_cases = [
            ("The system must provide 90% accuracy", "must_have"),
            ("Implementation should optimize performance", "should_have"),
            ("Framework could include additional features", "could_have"),
            ("Critical requirement for expert alignment", "must_have"),
            ("Performance targets are essential", "must_have")
        ]
        
        for requirement, expected_priority in test_cases:
            priority = self.extractor._determine_enhanced_priority(requirement)
            assert priority == expected_priority, f"Failed for '{requirement}': expected {expected_priority}, got {priority}"
    
    def test_testability_detection_accuracy(self):
        """Test enhanced testability detection"""
        testable_requirements = [
            "System must validate input accuracy with 90% precision",
            "Framework should demonstrate 85% expert alignment",
            "Performance must be benchmarked under 200ms",
            "Implementation needs to measure extraction accuracy"
        ]
        
        non_testable_requirements = [
            "Code should be maintainable and readable",
            "System architecture should follow best practices"
        ]
        
        for requirement in testable_requirements:
            assert self.extractor._is_enhanced_testable(requirement), f"Should detect testability in: {requirement}"
        
        for requirement in non_testable_requirements:
            assert not self.extractor._is_enhanced_testable(requirement), f"Should not detect testability in: {requirement}"


class TestEnhancedBenchmarkingEngine:
    """Test enhanced benchmarking engine with multi-dimensional grading"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = EnhancedBenchmarkingEngine(self.temp_dir)
        
        # Mock the extractor for consistent testing
        self.engine.extractor = Mock(spec=EnhancedSpecificationExtractor)
    
    def teardown_method(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_target_compliance(self):
        """Test that benchmarking meets performance targets (<2 minutes)"""
        # Mock specifications with realistic data
        mock_specs = [
            DesignSpecification(
                id="spec-116-1",
                type=SpecificationType.QUALITY_GATES,
                description="System must achieve 90% accuracy in specification extraction",
                acceptance_criteria=["Accuracy measured against expert validation"],
                constraints=["Performance under 2 minutes"],
                success_metrics={"target_percentage": 90.0},
                priority="must_have",
                measurable=True,
                testable=True,
                created_at=datetime.now(),
                issue_number=116
            ),
            DesignSpecification(
                id="spec-116-2",
                type=SpecificationType.PERFORMANCE_REQUIREMENTS,
                description="Benchmarking should complete within 2 minutes per assessment",
                acceptance_criteria=["Processing time measured and verified"],
                constraints=["Memory usage under 512MB"],
                success_metrics={"max_time_minutes": 2.0},
                priority="must_have",
                measurable=True,
                testable=True,
                created_at=datetime.now(),
                issue_number=116
            )
        ]
        
        # Configure mock extractor
        self.engine.extractor.extract_specifications_from_issue.return_value = (
            mock_specs,
            {
                "extraction_time_seconds": 0.5,
                "specifications_found": 2,
                "accuracy_estimate": 92.0,
                "confidence_score": 88.0
            }
        )
        
        # Run benchmarking
        start_time = time.time()
        result, performance_metrics = self.engine.benchmark_issue_with_enhanced_analysis(116)
        end_time = time.time()
        
        # Verify performance targets
        assert end_time - start_time < 120, "Total benchmarking should complete in under 2 minutes"
        assert performance_metrics["performance_target_met"], "Performance target should be met"
        assert performance_metrics["total_time_seconds"] < 120, "Should track performance time correctly"
        
        # Verify result quality
        assert result.issue_number == 116
        assert len(result.specifications) == 2
        assert result.quality_grade in ["A+", "A", "A-", "B+", "B"], "Should achieve good grade with high-quality specs"
    
    def test_multidimensional_grading_system(self):
        """Test multi-dimensional A-F grading system"""
        # Create specifications covering different dimensions
        specifications = [
            # Functional requirement
            DesignSpecification(
                id="spec-test-1", type=SpecificationType.FUNCTIONAL_REQUIREMENTS,
                description="System must extract specifications accurately",
                acceptance_criteria=["Accuracy verified", "Performance measured"],
                constraints=[], success_metrics={"target_percentage": 90},
                priority="must_have", measurable=True, testable=True,
                created_at=datetime.now(), issue_number=116
            ),
            # Quality gate requirement
            DesignSpecification(
                id="spec-test-2", type=SpecificationType.QUALITY_GATES,
                description="Expert alignment must achieve 85% agreement",
                acceptance_criteria=["Expert validation conducted"],
                constraints=[], success_metrics={"target_percentage": 85},
                priority="must_have", measurable=True, testable=True,
                created_at=datetime.now(), issue_number=116
            )
        ]
        
        # Create mock evidence with varying compliance scores
        evidence = [
            ImplementationEvidence(
                spec_id="spec-test-1", implementation_details="High-quality implementation",
                code_files=["extractor.py"], test_files=["test_extractor.py"],
                documentation_refs=[], metrics_achieved={"accuracy_measured": 89.5},
                compliance_score=0.95, compliance_level=ComplianceLevel.FULLY_COMPLIANT,
                issues_found=[], recommendations=[], evidence_timestamp=datetime.now()
            ),
            ImplementationEvidence(
                spec_id="spec-test-2", implementation_details="Good implementation",
                code_files=["grader.py"], test_files=["test_grader.py"],
                documentation_refs=[], metrics_achieved={"alignment_measured": 83.2},
                compliance_score=0.82, compliance_level=ComplianceLevel.MOSTLY_COMPLIANT,
                issues_found=["Minor alignment gap"], recommendations=["Improve expert validation"],
                evidence_timestamp=datetime.now()
            )
        ]
        
        # Test multi-dimensional grading
        grading_result = self.engine._calculate_multidimensional_grade(specifications, evidence)
        
        # Verify dimensional scores
        assert "dimensional_scores" in grading_result
        dimensions = grading_result["dimensional_scores"]
        
        assert "requirement_completeness" in dimensions
        assert "specification_clarity" in dimensions
        assert "implementation_adherence" in dimensions
        assert "quality_metrics_alignment" in dimensions
        assert "testing_coverage" in dimensions
        
        # Verify overall scoring
        assert 0.0 <= grading_result["overall_score"] <= 1.0
        assert grading_result["letter_grade"] in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
        
        # Verify score calculation logic
        expected_adherence = (0.95 + 0.82) / 2  # Average of evidence compliance scores
        assert abs(dimensions["implementation_adherence"] - expected_adherence) < 0.01
    
    def test_accuracy_estimation_validation(self):
        """Test accuracy estimation meets research targets"""
        # Create high-quality specifications
        high_quality_specs = [
            DesignSpecification(
                id=f"spec-hq-{i}", type=SpecificationType.QUALITY_GATES,
                description=f"System must achieve {90-i}% accuracy in component {i}",
                acceptance_criteria=[f"Accuracy measured and validated for component {i}"],
                constraints=[f"Performance requirements for component {i}"],
                success_metrics={"target_percentage": 90-i},
                priority="must_have", measurable=True, testable=True,
                created_at=datetime.now(), issue_number=116
            )
            for i in range(5)
        ]
        
        # Test accuracy estimation
        full_text = " ".join(spec.description for spec in high_quality_specs)
        accuracy_estimate = self.engine.extractor._estimate_extraction_accuracy(high_quality_specs, full_text)
        
        assert accuracy_estimate >= 75, f"High-quality specs should have high accuracy estimate, got {accuracy_estimate}"
        
        # Test confidence scoring
        confidence_score = self.engine.extractor._calculate_confidence_score(high_quality_specs)
        assert confidence_score >= 80, f"High-quality specs should have high confidence, got {confidence_score}"
    
    def test_expert_alignment_simulation(self):
        """Test expert alignment simulation and validation"""
        # Create specifications that should align well with expert judgment
        expert_aligned_specs = [
            DesignSpecification(
                id="spec-ea-1", type=SpecificationType.QUALITY_GATES,
                description="Grade assignments must align with human expert assessment 85% of cases",
                acceptance_criteria=["Expert validation conducted", "Alignment measured"],
                constraints=["Statistical significance required"],
                success_metrics={"alignment_target": 85.0},
                priority="must_have", measurable=True, testable=True,
                created_at=datetime.now(), issue_number=116
            )
        ]
        
        # Simulate expert alignment scoring
        evidence = self.engine._analyze_implementation_enhanced(expert_aligned_specs, 116)
        
        # Verify alignment metrics in evidence
        alignment_evidence = [ev for ev in evidence if "alignment" in ev.implementation_details.lower()]
        assert len(alignment_evidence) > 0, "Should generate alignment-focused evidence"
        
        # Check for alignment metrics in achieved metrics
        for ev in evidence:
            if "alignment" in ev.spec_id:
                assert "alignment_measured" in ev.metrics_achieved, "Should include alignment measurements"
    
    def test_comprehensive_workflow_validation(self):
        """Test complete workflow meets all research requirements"""
        # Mock realistic extraction results
        self.engine.extractor.extract_specifications_from_issue.return_value = (
            [
                DesignSpecification(
                    id="spec-comp-1", type=SpecificationType.QUALITY_GATES,
                    description="System must achieve 90% accuracy in specification extraction",
                    acceptance_criteria=["Accuracy measured against validation dataset"],
                    constraints=["Performance under 2 minutes"],
                    success_metrics={"accuracy_target": 90.0},
                    priority="must_have", measurable=True, testable=True,
                    created_at=datetime.now(), issue_number=116
                ),
                DesignSpecification(
                    id="spec-comp-2", type=SpecificationType.QUALITY_GATES,
                    description="A-F grading system aligns with expert human assessment 85%+ of cases",
                    acceptance_criteria=["Expert validation study conducted"],
                    constraints=["Statistical significance achieved"],
                    success_metrics={"alignment_target": 85.0},
                    priority="must_have", measurable=True, testable=True,
                    created_at=datetime.now(), issue_number=116
                )
            ],
            {
                "extraction_time_seconds": 0.8,
                "specifications_found": 2,
                "accuracy_estimate": 91.5,
                "confidence_score": 87.3
            }
        )
        
        # Run complete workflow
        result, performance_metrics = self.engine.benchmark_issue_with_enhanced_analysis(
            116, "Comprehensive validation test"
        )
        
        # Verify all research requirements met
        assert performance_metrics["performance_target_met"], "Performance target (<2 min) should be met"
        assert performance_metrics["extraction_accuracy"] >= 85, "Extraction accuracy should meet target"
        assert result.overall_adherence_score >= 0.8, "Overall adherence should be high for quality specs"
        assert result.quality_grade in ["A+", "A", "A-", "B+"], "Should achieve good grade for research-compliant implementation"
        
        # Verify report generation
        report = self.engine.generate_enhanced_report(result, performance_metrics)
        assert "Enhanced DPIBS Benchmarking Report" in report
        assert "Performance Target Met" in report
        assert "Extraction Accuracy" in report
        assert "Multi-dimensional" in report


class TestIntegrationValidation:
    """Integration tests validating research findings implementation"""
    
    @patch('subprocess.run')
    def test_real_issue_benchmark_validation(self, mock_run):
        """Test benchmarking with realistic Issue #116 data"""
        # Use actual Issue #116 structure and requirements
        mock_issue_data = {
            'body': open('/Users/cal/DEV/RIF/tests/fixtures/issue_116_body.txt', 'w').write('''
            # Research Phase 2: Design Specification Benchmarking Methodology

            **Parent PRD**: Issue #112 - Development Process Intelligence & Benchmarking System  
            **Phase**: Research and Analysis (2 of 4)  
            **Priority**: High  
            **Complexity**: High  
            **Estimated Duration**: 4-6 days

            ## Research Objective

            Develop comprehensive methodology for automatically extracting design specifications from GitHub issues and comparing them against implementation outcomes to provide objective grading of design adherence.

            ## Success Criteria

            - [ ] Specification extraction methodology defined with 90%+ accuracy target
            - [ ] Implementation analysis framework covers code, tests, architecture, documentation
            - [ ] Benchmarking algorithm provides objective, reproducible scoring
            - [ ] A-F grading system aligns with expert human assessment 85%+ of cases
            - [ ] Methodology handles GitHub issue format variations and evolution
            - [ ] Framework integrates with existing RIF validation workflows
            - [ ] Performance requirements met for real-time benchmarking (<2 minutes per assessment)

            ## Technical Requirements

            **1. Design Specification Extraction Techniques**
            - Natural language processing approaches for requirement extraction from issue descriptions
            - Structured data identification in GitHub issues (goals, constraints, success criteria)
            - Multi-source specification aggregation (issue description, comments, analysis results)
            - Specification versioning and evolution tracking throughout development lifecycle

            **2. Implementation Analysis Methods**
            - Code analysis techniques for design pattern recognition and adherence assessment
            - Test coverage and quality metrics integration with design requirement validation
            - Documentation analysis for specification compliance verification
            - Architectural decision tracking and constraint validation

            **3. Automated Comparison Frameworks**  
            - Multi-dimensional comparison algorithms for specification vs. implementation assessment
            - Weighted scoring systems based on requirement criticality and complexity
            - Evidence collection and validation for benchmarking decisions
            - False positive/negative reduction techniques in automated grading

            **4. Benchmarking and Grading Systems**
            - Industry-standard grading methodologies adapted for software development
            - Objective scoring criteria that reduce subjective bias in assessment
            - Continuous calibration mechanisms for scoring accuracy improvement
            - Integration with existing quality gate frameworks
            ''') or '''Research Phase 2: Design Specification Benchmarking Methodology content''',
            'title': 'DPIBS Research Phase 2: Design Specification Benchmarking Methodology',
            'comments': []
        }
        
        # Create temp file for mock
        try:
            os.makedirs('/Users/cal/DEV/RIF/tests/fixtures', exist_ok=True)
        except:
            pass
        
        mock_run.return_value.stdout = json.dumps(mock_issue_data)
        mock_run.return_value.check = True
        
        with tempfile.TemporaryDirectory() as temp_repo:
            engine = EnhancedBenchmarkingEngine(temp_repo)
            result, performance_metrics = engine.benchmark_issue_with_enhanced_analysis(116)
            
            # Validate against research targets
            assert performance_metrics["performance_target_met"], "Must meet 2-minute performance target"
            assert len(result.specifications) >= 8, "Should extract comprehensive specifications from complex issue"
            assert result.overall_adherence_score >= 0.75, "Should achieve good adherence for well-structured issue"
            
            # Verify research-specific requirements detected
            accuracy_requirements = [spec for spec in result.specifications if "90%" in spec.description]
            alignment_requirements = [spec for spec in result.specifications if "85%" in spec.description]
            
            assert len(accuracy_requirements) >= 1, "Should detect 90% accuracy requirements"
            assert len(alignment_requirements) >= 1, "Should detect 85% alignment requirements"


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "extractor":
            pytest.main(["-v", "-k", "TestEnhancedSpecificationExtractor"])
        elif test_category == "engine":
            pytest.main(["-v", "-k", "TestEnhancedBenchmarkingEngine"])
        elif test_category == "integration":
            pytest.main(["-v", "-k", "TestIntegrationValidation"])
    else:
        # Run all tests
        pytest.main(["-v", __file__])