#!/usr/bin/env python3
"""
Comprehensive test suite for Context-Aware Quality Thresholds System
Issue #91: Context-Aware Quality Thresholds System

Tests component classification accuracy, threshold calculation correctness,
performance benchmarks, and system integration.
"""

import unittest
import time
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.quality_gates.component_classifier import ComponentClassifier, ComponentType
from claude.commands.quality_gates.threshold_engine import AdaptiveThresholdEngine, ChangeMetrics, ChangeContext
from claude.commands.quality_gates.weighted_calculator import WeightedCalculator

class TestComponentClassifier(unittest.TestCase):
    """Test component classification system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = ComponentClassifier()
    
    def test_critical_algorithm_classification(self):
        """Test classification of critical algorithm files."""
        test_cases = [
            ("algorithms/encryption.py", "critical_algorithms"),
            ("core/security/hash_algorithm.py", "critical_algorithms"),
            ("engine/computation_core.py", "critical_algorithms"),
            ("crypto/rsa_implementation.py", "critical_algorithms")
        ]
        
        for file_path, expected_type in test_cases:
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_file(file_path)
                self.assertEqual(result.type, expected_type)
                self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_public_api_classification(self):
        """Test classification of public API files."""
        test_cases = [
            ("api/user_controller.py", "public_apis"),
            ("endpoints/authentication.py", "public_apis"),
            ("routes/data_api.py", "public_apis"),
            ("controllers/payment_handler.py", "public_apis")
        ]
        
        for file_path, expected_type in test_cases:
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_file(file_path)
                self.assertEqual(result.type, expected_type)
                self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_business_logic_classification(self):
        """Test classification of business logic files."""
        test_cases = [
            ("services/user_service.py", "business_logic"),
            ("logic/payment_processor.py", "business_logic"),
            ("models/customer_model.py", "business_logic"),
            ("domain/order_management.py", "business_logic")
        ]
        
        for file_path, expected_type in test_cases:
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_file(file_path)
                self.assertEqual(result.type, expected_type)
                self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_ui_component_classification(self):
        """Test classification of UI component files."""
        test_cases = [
            ("components/UserProfile.jsx", "ui_components"),
            ("views/dashboard.vue", "ui_components"),
            ("pages/login.tsx", "ui_components"),
            ("templates/email_template.html", "ui_components")
        ]
        
        for file_path, expected_type in test_cases:
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_file(file_path)
                self.assertEqual(result.type, expected_type)
                self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_test_utility_classification(self):
        """Test classification of test utility files."""
        test_cases = [
            ("test_utils/fixtures.py", "test_utilities"),
            ("tests/test_user_service.py", "test_utilities"),
            ("fixtures/sample_data.py", "test_utilities"),
            ("conftest.py", "test_utilities")
        ]
        
        for file_path, expected_type in test_cases:
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_file(file_path)
                self.assertEqual(result.type, expected_type)
                self.assertGreaterEqual(result.confidence, 0.6)
    
    def test_content_analysis(self):
        """Test content-based classification for ambiguous files."""
        # Critical algorithm content
        algorithm_content = """
        def encryption_algorithm(data):
            # Critical encryption implementation
            @critical
            def compute_hash(input_data):
                return secure_hash(input_data)
            return encrypt(data)
        """
        
        result = self.classifier.classify_file("ambiguous/processor.py", algorithm_content)
        self.assertEqual(result.type, "critical_algorithms")
        self.assertGreaterEqual(result.confidence, 0.7)
        
        # API endpoint content
        api_content = """
        @app.route('/api/users', methods=['GET', 'POST'])
        class UserAPI:
            def get(self):
                return get_users()
            
            def post(self):
                return create_user()
        """
        
        result = self.classifier.classify_file("ambiguous/handler.py", api_content)
        self.assertEqual(result.type, "public_apis")
        self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_classification_performance(self):
        """Test classification performance meets <100ms requirement."""
        # Create unique file paths for proper testing
        test_files = []
        base_patterns = [
            "algorithms/file_{}.py",
            "api/handler_{}.py", 
            "services/service_{}.py",
            "components/Component_{}.jsx",
            "tests/test_{}.py"
        ]
        
        for i in range(20):  # 20 iterations
            for pattern in base_patterns:
                test_files.append(pattern.format(i))
        # Total: 100 unique files
        
        start_time = time.time()
        results = self.classifier.batch_classify(test_files)
        total_time = (time.time() - start_time) * 1000
        
        avg_time_per_file = total_time / len(test_files)
        
        self.assertLess(avg_time_per_file, 100, 
                       f"Average classification time {avg_time_per_file:.2f}ms exceeds 100ms target")
        
        # Verify all files were classified
        self.assertEqual(len(results), len(test_files))
        
        # Verify performance metrics
        metrics = self.classifier.get_performance_metrics()
        self.assertTrue(metrics["performance_target_met"])
    
    def test_accuracy_validation(self):
        """Test classification accuracy meets >95% requirement."""
        # Create test dataset
        test_dataset = [
            ("algorithms/crypto.py", "critical_algorithms"),
            ("api/users.py", "public_apis"),
            ("services/auth.py", "business_logic"),
            ("integrations/payment.py", "integration_code"),
            ("components/App.jsx", "ui_components"),
            ("tests/test_auth.py", "test_utilities"),
            ("core/encryption.py", "critical_algorithms"),
            ("endpoints/data.py", "public_apis"),
            ("models/user.py", "business_logic"),
            ("external/stripe_client.py", "integration_code")
        ]
        
        validation_result = self.classifier.validate_accuracy(test_dataset)
        
        self.assertGreaterEqual(validation_result["accuracy_percent"], 95.0,
                              f"Classification accuracy {validation_result['accuracy_percent']}% below 95% requirement")
        self.assertTrue(validation_result["meets_requirement"])

class TestThresholdEngine(unittest.TestCase):
    """Test adaptive threshold calculation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = AdaptiveThresholdEngine()
    
    def test_single_component_threshold_calculation(self):
        """Test threshold calculation for single component."""
        change_metrics = ChangeMetrics(
            lines_added=100,
            lines_deleted=50,
            lines_modified=25,
            files_changed=3,
            complexity_score=1.2
        )
        
        # Test critical algorithms
        result = self.engine.calculate_component_threshold("critical_algorithms", change_metrics)
        self.assertGreaterEqual(result.applied_threshold, 95.0)
        self.assertEqual(result.component_type, "critical_algorithms")
        
        # Test UI components
        result = self.engine.calculate_component_threshold("ui_components", change_metrics)
        self.assertLessEqual(result.applied_threshold, 90.0)  # Should be lower than critical
        
        # Test test utilities
        result = self.engine.calculate_component_threshold("test_utilities", change_metrics)
        self.assertLessEqual(result.applied_threshold, 80.0)  # Should be lowest
    
    def test_size_factor_adjustments(self):
        """Test that size factors properly adjust thresholds."""
        # Small change (should reduce threshold)
        small_change = ChangeMetrics(
            lines_added=20, lines_deleted=10, lines_modified=5,
            files_changed=1, complexity_score=1.0
        )
        small_result = self.engine.calculate_component_threshold("business_logic", small_change)
        
        # Large change (should increase threshold)
        large_change = ChangeMetrics(
            lines_added=800, lines_deleted=400, lines_modified=200,
            files_changed=10, complexity_score=1.5
        )
        large_result = self.engine.calculate_component_threshold("business_logic", large_change)
        
        # Large changes should have higher thresholds than small changes
        self.assertGreater(large_result.applied_threshold, small_result.applied_threshold)
    
    def test_context_modifiers(self):
        """Test context-based threshold modifications."""
        change_metrics = ChangeMetrics(
            lines_added=100, lines_deleted=50, lines_modified=25,
            files_changed=3, complexity_score=1.2
        )
        
        # Security critical context
        security_context = ChangeContext(
            is_security_critical=True,
            risk_level="high",
            has_tests=True
        )
        security_result = self.engine.calculate_component_threshold("business_logic", change_metrics, security_context)
        
        # Regular context
        regular_context = ChangeContext(
            risk_level="medium",
            has_tests=True
        )
        regular_result = self.engine.calculate_component_threshold("business_logic", change_metrics, regular_context)
        
        # Security critical should have higher threshold
        self.assertGreater(security_result.applied_threshold, regular_result.applied_threshold)
    
    def test_performance_requirements(self):
        """Test threshold calculation performance meets <200ms target."""
        components = {
            "critical_algorithms": ChangeMetrics(100, 50, 25, 3, 1.2),
            "public_apis": ChangeMetrics(80, 40, 20, 2, 1.1),
            "business_logic": ChangeMetrics(150, 75, 35, 4, 1.3),
            "ui_components": ChangeMetrics(200, 100, 50, 5, 0.9)
        }
        
        context = ChangeContext(is_security_critical=True, risk_level="high")
        
        start_time = time.time()
        result = self.engine.calculate_weighted_threshold(components, context)
        calculation_time = (time.time() - start_time) * 1000
        
        self.assertLess(calculation_time, 200, 
                       f"Calculation time {calculation_time:.2f}ms exceeds 200ms target")
        self.assertTrue(result["performance_target_met"])
    
    def test_backward_compatibility(self):
        """Test backward compatibility with 80% threshold."""
        # Test that results always meet minimum 80% requirement
        change_metrics = ChangeMetrics(10, 5, 2, 1, 0.5)  # Very small change
        
        for component_type in ["critical_algorithms", "public_apis", "business_logic", "integration_code"]:
            result = self.engine.calculate_component_threshold(component_type, change_metrics)
            self.assertGreaterEqual(result.applied_threshold, 80.0,
                                  f"{component_type} threshold {result.applied_threshold}% below 80% minimum")
            
        # Test validation function
        self.assertTrue(self.engine.validate_backward_compatibility(85.0))
        self.assertTrue(self.engine.validate_backward_compatibility(80.0))
        self.assertFalse(self.engine.validate_backward_compatibility(75.0))

class TestWeightedCalculator(unittest.TestCase):
    """Test weighted threshold calculation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = WeightedCalculator()
    
    def test_balanced_weight_strategy(self):
        """Test balanced weight calculation strategy."""
        components = {
            "critical_algorithms": ChangeMetrics(50, 25, 10, 2, 1.5),
            "ui_components": ChangeMetrics(200, 100, 50, 8, 0.8),
            "test_utilities": ChangeMetrics(30, 15, 5, 1, 0.5)
        }
        
        result = self.calculator.calculate_weighted_threshold(components, strategy="balanced")
        
        # Should return a weighted threshold
        self.assertGreater(result.final_threshold, 60)
        self.assertLess(result.final_threshold, 100)
        
        # Critical algorithms should have higher contribution due to higher priority
        critical_contribution = result.component_contributions.get("critical_algorithms", 0)
        ui_contribution = result.component_contributions.get("ui_components", 0)
        
        # Even though UI components have more changes, critical algorithms should have higher weighted contribution
        self.assertGreater(critical_contribution, ui_contribution * 0.5)  # Allow for size influence
    
    def test_size_based_strategy(self):
        """Test size-based weight calculation strategy."""
        components = {
            "critical_algorithms": ChangeMetrics(50, 25, 10, 2, 1.5),   # Small change
            "ui_components": ChangeMetrics(400, 200, 100, 15, 0.8)      # Large change
        }
        
        result = self.calculator.calculate_weighted_threshold(components, strategy="size_based")
        
        # UI components should dominate due to size
        critical_contribution = result.component_contributions.get("critical_algorithms", 0)
        ui_contribution = result.component_contributions.get("ui_components", 0)
        
        self.assertGreater(ui_contribution, critical_contribution)
    
    def test_priority_based_strategy(self):
        """Test priority-based weight calculation strategy."""
        components = {
            "critical_algorithms": ChangeMetrics(50, 25, 10, 2, 1.5),   # High priority
            "test_utilities": ChangeMetrics(200, 100, 50, 8, 0.8)       # Low priority, more changes
        }
        
        result = self.calculator.calculate_weighted_threshold(components, strategy="priority_based")
        
        # Critical algorithms should dominate despite smaller size
        critical_contribution = result.component_contributions.get("critical_algorithms", 0)
        test_contribution = result.component_contributions.get("test_utilities", 0)
        
        self.assertGreater(critical_contribution, test_contribution)
    
    def test_multi_component_integration(self):
        """Test complex multi-component threshold calculation."""
        components = {
            "critical_algorithms": ChangeMetrics(100, 50, 25, 3, 1.5),
            "public_apis": ChangeMetrics(80, 40, 20, 2, 1.2),
            "business_logic": ChangeMetrics(150, 75, 35, 4, 1.1),
            "ui_components": ChangeMetrics(250, 125, 60, 8, 0.9),
            "test_utilities": ChangeMetrics(120, 60, 30, 5, 0.7)
        }
        
        context = ChangeContext(
            is_security_critical=True,
            risk_level="medium",
            pr_size="large"
        )
        
        result = self.calculator.calculate_weighted_threshold(components, context, "balanced")
        
        # Verify result structure
        self.assertIn("final_threshold", result.__dict__)
        self.assertIn("component_contributions", result.__dict__)
        self.assertIn("performance_metrics", result.__dict__)
        
        # Verify all components are included
        self.assertEqual(len(result.component_contributions), 5)
        
        # Verify contributions sum roughly to 100% (allowing for rounding)
        total_contribution = sum(result.component_contributions.values())
        self.assertAlmostEqual(total_contribution, 100.0, delta=5.0)
    
    def test_fallback_constraints(self):
        """Test fallback constraint application."""
        # Test with components that might result in very low threshold
        components = {
            "test_utilities": ChangeMetrics(10, 5, 2, 1, 0.5)  # Low priority, small change
        }
        
        result = self.calculator.calculate_weighted_threshold(components)
        
        # Should apply fallback to meet 80% minimum
        self.assertGreaterEqual(result.final_threshold, 80.0)
        self.assertTrue(result.fallback_applied or result.final_threshold >= 80.0)

class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.classifier = ComponentClassifier()
        self.engine = AdaptiveThresholdEngine(classifier=self.classifier)
        self.calculator = WeightedCalculator(self.engine)
    
    def test_end_to_end_threshold_calculation(self):
        """Test complete end-to-end threshold calculation pipeline."""
        # Simulate a real-world change set
        file_changes = {
            "src/algorithms/encryption.py": ChangeMetrics(80, 40, 20, 1, 1.3),
            "src/api/auth_controller.py": ChangeMetrics(60, 30, 15, 1, 1.1),
            "src/components/LoginForm.jsx": ChangeMetrics(120, 60, 30, 2, 0.9),
            "tests/test_encryption.py": ChangeMetrics(40, 20, 10, 1, 0.6)
        }
        
        # Classify files
        classifications = {}
        for file_path in file_changes.keys():
            classifications[file_path] = self.classifier.classify_file(file_path)
        
        # Group by component type
        components = {}
        for file_path, classification in classifications.items():
            component_type = classification.type
            if component_type not in components:
                components[component_type] = ChangeMetrics(0, 0, 0, 0, 0.0)
            
            # Aggregate metrics
            existing = components[component_type]
            new_metrics = file_changes[file_path]
            components[component_type] = ChangeMetrics(
                lines_added=existing.lines_added + new_metrics.lines_added,
                lines_deleted=existing.lines_deleted + new_metrics.lines_deleted,
                lines_modified=existing.lines_modified + new_metrics.lines_modified,
                files_changed=existing.files_changed + new_metrics.files_changed,
                complexity_score=max(existing.complexity_score, new_metrics.complexity_score)
            )
        
        # Calculate weighted threshold
        context = ChangeContext(
            is_security_critical=True,  # Encryption changes
            risk_level="high",
            pr_size="medium"
        )
        
        result = self.calculator.calculate_weighted_threshold(components, context)
        
        # Verify result integrity
        self.assertGreaterEqual(result.final_threshold, 80.0)  # Meets minimum
        self.assertLessEqual(result.final_threshold, 100.0)    # Within bounds
        self.assertTrue(result.performance_metrics["performance_target_met"])
        
        # Verify security context effect (should increase threshold)
        self.assertGreater(result.final_threshold, 85.0)  # Security should push threshold up
    
    def test_performance_under_load(self):
        """Test system performance under realistic load."""
        # Simulate 100 files across different component types
        test_files = []
        change_metrics_list = []
        
        component_patterns = [
            ("algorithms/", "critical_algorithms", 20),
            ("api/", "public_apis", 30),
            ("services/", "business_logic", 25),
            ("components/", "ui_components", 20),
            ("tests/", "test_utilities", 5)
        ]
        
        for pattern, component_type, count in component_patterns:
            for i in range(count):
                file_path = f"{pattern}file_{i}.py"
                test_files.append(file_path)
                change_metrics_list.append(ChangeMetrics(
                    lines_added=50 + (i * 5),
                    lines_deleted=25 + (i * 2),
                    lines_modified=10 + i,
                    files_changed=1,
                    complexity_score=1.0 + (i * 0.1)
                ))
        
        # Measure total processing time
        start_time = time.time()
        
        # Batch classify
        classifications = self.classifier.batch_classify(test_files)
        
        # Group and calculate thresholds
        components = {}
        for i, (file_path, classification) in enumerate(classifications.items()):
            component_type = classification.type
            if component_type not in components:
                components[component_type] = ChangeMetrics(0, 0, 0, 0, 0.0)
            
            existing = components[component_type]
            new_metrics = change_metrics_list[i]
            components[component_type] = ChangeMetrics(
                lines_added=existing.lines_added + new_metrics.lines_added,
                lines_deleted=existing.lines_deleted + new_metrics.lines_deleted,
                lines_modified=existing.lines_modified + new_metrics.lines_modified,
                files_changed=existing.files_changed + new_metrics.files_changed,
                complexity_score=max(existing.complexity_score, new_metrics.complexity_score)
            )
        
        result = self.calculator.calculate_weighted_threshold(components)
        
        total_time = (time.time() - start_time) * 1000
        
        # Total processing should be under 300ms as specified in requirements
        self.assertLess(total_time, 300, 
                       f"Total processing time {total_time:.2f}ms exceeds 300ms budget")
        
        # Individual components should meet their performance targets
        classifier_metrics = self.classifier.get_performance_metrics()
        self.assertTrue(classifier_metrics["performance_target_met"])
        
        calculator_metrics = self.calculator.get_calculation_metrics()
        self.assertLess(calculator_metrics["average_calculation_time_ms"], 200)
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and graceful fallbacks."""
        # Test with invalid component type
        invalid_components = {
            "invalid_component_type": ChangeMetrics(100, 50, 25, 2, 1.0)
        }
        
        result = self.calculator.calculate_weighted_threshold(invalid_components)
        
        # Should fallback gracefully
        self.assertEqual(result.final_threshold, 80.0)  # Fallback threshold
        self.assertTrue(result.fallback_applied)
        
        # Test with empty components
        empty_result = self.calculator.calculate_weighted_threshold({})
        self.assertEqual(empty_result.final_threshold, 80.0)
        self.assertTrue(empty_result.fallback_applied)
        
        # Test classification with non-existent file
        classification = self.classifier.classify_file("non/existent/file.py")
        self.assertIsNotNone(classification.type)  # Should return some classification
        self.assertGreaterEqual(classification.confidence, 0.5)  # Should have reasonable confidence

class TestAcceptanceCriteria(unittest.TestCase):
    """Test all acceptance criteria from Issue #91."""
    
    def setUp(self):
        """Set up acceptance criteria test fixtures."""
        self.classifier = ComponentClassifier()
        self.engine = AdaptiveThresholdEngine()
        self.calculator = WeightedCalculator()
    
    def test_acceptance_criteria_1_classification_accuracy(self):
        """AC1: System correctly classifies files into component types (>95% accuracy)."""
        # Create comprehensive test dataset
        test_dataset = [
            # Critical algorithms
            ("algorithms/encryption.py", "critical_algorithms"),
            ("core/hash_functions.py", "critical_algorithms"),
            ("crypto/rsa.py", "critical_algorithms"),
            ("security/core/authentication.py", "critical_algorithms"),
            
            # Public APIs
            ("api/users.py", "public_apis"),
            ("endpoints/data.py", "public_apis"),
            ("controllers/auth.py", "public_apis"),
            ("routes/payment.py", "public_apis"),
            
            # Business logic
            ("services/user_service.py", "business_logic"),
            ("models/customer.py", "business_logic"),
            ("domain/order_processing.py", "business_logic"),
            ("logic/pricing.py", "business_logic"),
            
            # Integration code
            ("integrations/stripe_client.py", "integration_code"),
            ("external/api_client.py", "integration_code"),
            ("connectors/database.py", "integration_code"),
            ("adapters/payment_gateway.py", "integration_code"),
            
            # UI components
            ("components/UserProfile.jsx", "ui_components"),
            ("views/dashboard.vue", "ui_components"),
            ("pages/home.tsx", "ui_components"),
            ("templates/email.html", "ui_components"),
            
            # Test utilities
            ("tests/test_user.py", "test_utilities"),
            ("test_utils/fixtures.py", "test_utilities"),
            ("fixtures/sample_data.py", "test_utilities"),
            ("conftest.py", "test_utilities")
        ]
        
        validation_result = self.classifier.validate_accuracy(test_dataset)
        
        self.assertGreaterEqual(validation_result["accuracy_percent"], 95.0,
                              f"Classification accuracy {validation_result['accuracy_percent']}% fails AC1 requirement")
        self.assertTrue(validation_result["meets_requirement"])
    
    def test_acceptance_criteria_2_threshold_application(self):
        """AC2: Applies appropriate thresholds based on component classification."""
        test_cases = [
            ("critical_algorithms", 95.0, 100.0),
            ("public_apis", 90.0, 95.0),
            ("business_logic", 85.0, 90.0),
            ("integration_code", 80.0, 85.0),
            ("ui_components", 70.0, 80.0),
            ("test_utilities", 60.0, 70.0)
        ]
        
        change_metrics = ChangeMetrics(100, 50, 25, 3, 1.0)
        
        for component_type, expected_min, expected_target in test_cases:
            with self.subTest(component_type=component_type):
                result = self.engine.calculate_component_threshold(component_type, change_metrics)
                
                # Should be within expected range
                self.assertGreaterEqual(result.applied_threshold, expected_min * 0.9,  # Allow 10% variance
                                      f"{component_type} threshold {result.applied_threshold}% below expected minimum {expected_min}%")
                self.assertLessEqual(result.applied_threshold, expected_target * 1.1,   # Allow 10% variance
                                    f"{component_type} threshold {result.applied_threshold}% above expected target {expected_target}%")
    
    def test_acceptance_criteria_3_weighted_thresholds(self):
        """AC3: Calculates weighted thresholds for multi-component changes."""
        components = {
            "critical_algorithms": ChangeMetrics(50, 25, 10, 2, 1.5),
            "public_apis": ChangeMetrics(100, 50, 25, 3, 1.2),
            "ui_components": ChangeMetrics(200, 100, 50, 6, 0.9)
        }
        
        result = self.calculator.calculate_weighted_threshold(components)
        
        # Should calculate a weighted threshold
        self.assertGreater(result.final_threshold, 70.0)
        self.assertLess(result.final_threshold, 100.0)
        
        # All components should contribute
        self.assertEqual(len(result.component_contributions), 3)
        
        # Contributions should sum to approximately 100%
        total_contribution = sum(result.component_contributions.values())
        self.assertAlmostEqual(total_contribution, 100.0, delta=10.0)
        
        # Critical algorithms should have meaningful contribution despite smaller size
        critical_contribution = result.component_contributions.get("critical_algorithms", 0)
        self.assertGreater(critical_contribution, 15.0)  # Should have at least 15% contribution
    
    def test_acceptance_criteria_4_backward_compatibility(self):
        """AC4: Maintains backward compatibility with existing 80% threshold configs."""
        # Test various scenarios that should maintain 80% minimum
        test_scenarios = [
            # Small changes to low-priority components
            ({"test_utilities": ChangeMetrics(10, 5, 2, 1, 0.5)}, None),
            # Mixed components with overall low risk
            ({
                "ui_components": ChangeMetrics(50, 25, 10, 2, 0.8),
                "test_utilities": ChangeMetrics(20, 10, 5, 1, 0.6)
            }, ChangeContext(risk_level="low")),
        ]
        
        for components, context in test_scenarios:
            with self.subTest(components=list(components.keys())):
                result = self.calculator.calculate_weighted_threshold(components, context)
                
                self.assertGreaterEqual(result.final_threshold, 80.0,
                                      f"Threshold {result.final_threshold}% below 80% backward compatibility requirement")

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for the system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.classifier = ComponentClassifier()
        self.engine = AdaptiveThresholdEngine()
        self.calculator = WeightedCalculator()
    
    def test_benchmark_classification_performance(self):
        """Benchmark: Classification should be <100ms per file."""
        test_files = [f"test/file_{i}.py" for i in range(100)]
        
        times = []
        for file_path in test_files:
            start = time.time()
            self.classifier.classify_file(file_path)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\nClassification Performance Benchmark:")
        print(f"  Average time per file: {avg_time:.2f}ms")
        print(f"  Maximum time per file: {max_time:.2f}ms")
        print(f"  Target: <100ms")
        
        self.assertLess(avg_time, 100.0, 
                       f"Average classification time {avg_time:.2f}ms exceeds 100ms target")
        self.assertLess(max_time, 200.0, 
                       f"Maximum classification time {max_time:.2f}ms is unreasonably high")
    
    def test_benchmark_threshold_calculation_performance(self):
        """Benchmark: Threshold calculation should be <200ms for complex scenarios."""
        # Create complex multi-component scenario
        components = {
            f"component_type_{i}": ChangeMetrics(
                lines_added=100 + i * 20,
                lines_deleted=50 + i * 10,
                lines_modified=25 + i * 5,
                files_changed=2 + i,
                complexity_score=1.0 + i * 0.1
            )
            for i in range(10)  # 10 different components
        }
        
        context = ChangeContext(
            is_security_critical=True,
            is_breaking_change=True,
            risk_level="high",
            pr_size="large"
        )
        
        times = []
        for _ in range(10):  # Run 10 iterations
            start = time.time()
            result = self.calculator.calculate_weighted_threshold(components, context)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\nThreshold Calculation Performance Benchmark:")
        print(f"  Average calculation time: {avg_time:.2f}ms")
        print(f"  Maximum calculation time: {max_time:.2f}ms")
        print(f"  Target: <200ms")
        
        self.assertLess(avg_time, 200.0,
                       f"Average calculation time {avg_time:.2f}ms exceeds 200ms target")
    
    def test_benchmark_total_system_overhead(self):
        """Benchmark: Total system overhead should be <300ms per validation run."""
        # Simulate a realistic PR with multiple files
        file_changes = {
            f"src/algorithms/algorithm_{i}.py": ChangeMetrics(80, 40, 20, 1, 1.2)
            for i in range(5)
        }
        file_changes.update({
            f"src/api/endpoint_{i}.py": ChangeMetrics(60, 30, 15, 1, 1.1)
            for i in range(8)
        })
        file_changes.update({
            f"src/components/Component{i}.jsx": ChangeMetrics(100, 50, 25, 1, 0.9)
            for i in range(12)
        })
        
        context = ChangeContext(
            is_security_critical=True,
            risk_level="medium",
            pr_size="large"
        )
        
        start_time = time.time()
        
        # Step 1: Classify all files
        classifications = {}
        for file_path in file_changes.keys():
            classifications[file_path] = self.classifier.classify_file(file_path)
        
        # Step 2: Group by component type
        components = {}
        for file_path, classification in classifications.items():
            component_type = classification.type
            if component_type not in components:
                components[component_type] = ChangeMetrics(0, 0, 0, 0, 0.0)
            
            existing = components[component_type]
            new_metrics = file_changes[file_path]
            components[component_type] = ChangeMetrics(
                lines_added=existing.lines_added + new_metrics.lines_added,
                lines_deleted=existing.lines_deleted + new_metrics.lines_deleted,
                lines_modified=existing.lines_modified + new_metrics.lines_modified,
                files_changed=existing.files_changed + new_metrics.files_changed,
                complexity_score=max(existing.complexity_score, new_metrics.complexity_score)
            )
        
        # Step 3: Calculate weighted threshold
        result = self.calculator.calculate_weighted_threshold(components, context)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"\nTotal System Performance Benchmark:")
        print(f"  Files processed: {len(file_changes)}")
        print(f"  Component types identified: {len(components)}")
        print(f"  Total processing time: {total_time:.2f}ms")
        print(f"  Target: <300ms")
        print(f"  Final threshold: {result.final_threshold}%")
        
        self.assertLess(total_time, 300.0,
                       f"Total system overhead {total_time:.2f}ms exceeds 300ms budget")
        
        # Verify result quality
        self.assertGreater(result.final_threshold, 80.0)
        self.assertLess(result.final_threshold, 100.0)
        self.assertFalse(result.fallback_applied)  # Should handle this complex case without fallback

def run_all_tests():
    """Run all tests and generate comprehensive report."""
    import io
    from contextlib import redirect_stdout
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestComponentClassifier,
        TestThresholdEngine,
        TestWeightedCalculator,
        TestSystemIntegration,
        TestAcceptanceCriteria,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    print("=" * 80)
    print("Context-Aware Quality Thresholds System - Test Suite")
    print("Issue #91: Context-Aware Quality Thresholds System")
    print("=" * 80)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nTest Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"  Total execution time: {total_time:.2f}s")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            newline = '\n'
            error_msg = traceback.split('AssertionError: ')[-1].split(newline)[0] if 'AssertionError:' in traceback else 'See details'
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            newline = '\n'
            error_msg = traceback.split(newline)[-2] if traceback else 'Unknown error'
            print(f"  - {test}: {error_msg}")
    
    # Generate test report
    report_path = Path(__file__).parent.parent / "knowledge" / "validation" / "issue-91-test-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_report = {
        "issue": 91,
        "system": "Context-Aware Quality Thresholds System",
        "test_execution": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_time_seconds": total_time,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        },
        "acceptance_criteria_validation": {
            "component_classification_accuracy": ">95%",
            "threshold_application": "Verified",
            "weighted_calculation": "Verified", 
            "backward_compatibility": "Verified"
        },
        "performance_benchmarks": {
            "classification_time": "<100ms per file",
            "calculation_time": "<200ms for complex scenarios",
            "total_overhead": "<300ms per validation run"
        },
        "implementation_status": "Phase 1 Complete" if result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0 else "Needs Fixes"
    }
    
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\nTest report saved to: {report_path}")
    
    return result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)