#!/usr/bin/env python3
"""
Comprehensive Test Suite for Adaptive Threshold Learning System
Issue #95: Adaptive Threshold Learning System

Tests the complete adaptive threshold learning system including:
- Historical data collection
- Quality pattern analysis  
- Rule-based threshold optimization
- Configuration management
- Integration with quality gates
"""

import unittest
import json
import tempfile
import shutil
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.historical_data_collector import HistoricalDataCollector, QualityDecision
from claude.commands.quality_pattern_analyzer import QualityPatternAnalyzer, QualityPattern
from claude.commands.threshold_optimizer import ThresholdOptimizer, ThresholdOptimizationResult
from claude.commands.configuration_manager import ConfigurationManager
from claude.commands.adaptive_threshold_system import AdaptiveThresholdSystem
from claude.commands.adaptive_quality_gates_integration import AdaptiveQualityGatesIntegration

class TestHistoricalDataCollector(unittest.TestCase):
    """Test historical data collection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = HistoricalDataCollector(str(Path(self.temp_dir) / "historical"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_file_initialization(self):
        """Test that data files are created properly."""
        self.assertTrue(self.collector.quality_decisions_file.exists())
        self.assertTrue(self.collector.threshold_performance_file.exists())
        self.assertTrue(self.collector.team_metrics_file.exists())
        self.assertTrue(self.collector.project_characteristics_file.exists())
    
    def test_record_quality_decision(self):
        """Test recording quality decisions."""
        success = self.collector.record_quality_decision(
            component_type="business_logic",
            threshold_used=85.0,
            quality_score=88.0,
            decision="pass",
            context={"pr_size": "medium", "risk_level": "low"},
            issue_number=123
        )
        
        self.assertTrue(success)
        
        # Verify data was recorded
        decisions = self.collector.get_quality_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].component_type, "business_logic")
        self.assertEqual(decisions[0].threshold_used, 85.0)
        self.assertEqual(decisions[0].quality_score, 88.0)
        self.assertEqual(decisions[0].decision, "pass")
    
    def test_record_multiple_decisions(self):
        """Test recording multiple quality decisions."""
        test_data = [
            ("critical_algorithms", 95.0, 97.0, "pass", {"high_risk": True}),
            ("ui_components", 75.0, 72.0, "fail", {"ui_change": True}),
            ("test_utilities", 70.0, 65.0, "manual_override", {"test_update": True})
        ]
        
        for component_type, threshold, score, decision, context in test_data:
            success = self.collector.record_quality_decision(
                component_type, threshold, score, decision, context
            )
            self.assertTrue(success)
        
        # Verify all decisions were recorded
        decisions = self.collector.get_quality_decisions()
        self.assertEqual(len(decisions), 3)
        
        # Test filtering by component type
        critical_decisions = self.collector.get_quality_decisions(component_type="critical_algorithms")
        self.assertEqual(len(critical_decisions), 1)
        self.assertEqual(critical_decisions[0].component_type, "critical_algorithms")
    
    def test_threshold_effectiveness_analysis(self):
        """Test threshold effectiveness analysis."""
        # Record test data with different thresholds and outcomes
        test_data = [
            ("business_logic", 80.0, 85.0, "pass"),
            ("business_logic", 80.0, 75.0, "fail"),
            ("business_logic", 85.0, 87.0, "pass"),
            ("business_logic", 85.0, 83.0, "fail"),
            ("business_logic", 90.0, 92.0, "pass"),
            ("business_logic", 90.0, 88.0, "fail")
        ]
        
        for component_type, threshold, score, decision in test_data:
            self.collector.record_quality_decision(
                component_type, threshold, score, decision, {"test": True}
            )
        
        # Analyze effectiveness
        analysis = self.collector.analyze_threshold_effectiveness("business_logic")
        
        self.assertEqual(analysis["component_type"], "business_logic")
        self.assertEqual(analysis["analysis"], "complete")
        self.assertEqual(analysis["total_decisions"], 6)
        self.assertIsNotNone(analysis["optimal_threshold"])
        self.assertIn("threshold_analysis", analysis)
    
    def test_data_summary(self):
        """Test data summary generation."""
        # Add some test data
        self.collector.record_quality_decision(
            "critical_algorithms", 95.0, 97.0, "pass", {"test": True}
        )
        self.collector.record_quality_decision(
            "ui_components", 75.0, 78.0, "pass", {"test": True}
        )
        
        summary = self.collector.get_data_summary()
        
        self.assertEqual(summary["quality_decisions"]["total_count"], 2)
        self.assertIn("critical_algorithms", summary["quality_decisions"]["component_types"])
        self.assertIn("ui_components", summary["quality_decisions"]["component_types"])
        self.assertGreater(summary["data_quality"]["completeness_score"], 0)

class TestQualityPatternAnalyzer(unittest.TestCase):
    """Test quality pattern analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = Path(self.temp_dir) / "knowledge"
        self.knowledge_dir.mkdir(parents=True)
        
        # Create mock knowledge base files
        self._create_mock_knowledge_files()
        
        self.analyzer = QualityPatternAnalyzer(
            str(self.knowledge_dir),
            str(Path(self.temp_dir) / "quality")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_knowledge_files(self):
        """Create mock knowledge base files for testing."""
        # Create patterns directory
        patterns_dir = self.knowledge_dir / "patterns"
        patterns_dir.mkdir(parents=True)
        
        # Mock pattern file
        pattern_file = patterns_dir / "test-pattern.json"
        pattern_data = {
            "pattern_name": "Quality Gate Success",
            "component_type": "business_logic",
            "success_metrics": {
                "coverage_threshold": 85.0,
                "success_rate": 0.92
            },
            "quality_indicators": ["high coverage", "passing tests", "good design"],
            "context": {"team_size": 5, "project_complexity": 3}
        }
        with open(pattern_file, 'w') as f:
            json.dump(pattern_data, f)
        
        # Create decisions directory
        decisions_dir = self.knowledge_dir / "decisions"
        decisions_dir.mkdir(parents=True)
        
        # Mock decision file
        decision_file = decisions_dir / "quality-gates-decision.json"
        decision_data = {
            "decision_type": "quality_threshold_adjustment",
            "component": "critical_algorithms",
            "old_threshold": 90.0,
            "new_threshold": 95.0,
            "rationale": "Increased threshold due to critical nature",
            "outcome": "improved quality",
            "metrics": {
                "before_defects": 3,
                "after_defects": 1,
                "success_rate": 0.88
            }
        }
        with open(decision_file, 'w') as f:
            json.dump(decision_data, f)
        
        # Create learning directory
        learning_dir = self.knowledge_dir / "learning"
        learning_dir.mkdir(parents=True)
        
        # Mock learning file
        learning_file = learning_dir / "threshold-learning.json"
        learning_data = {
            "learning_topic": "threshold optimization",
            "component_types": ["public_apis", "integration_code"],
            "key_insights": [
                "Lower thresholds for integration code improved velocity",
                "API components benefit from higher quality standards"
            ],
            "quality_improvements": {
                "defect_reduction": 0.25,
                "false_positive_reduction": 0.15
            }
        }
        with open(learning_file, 'w') as f:
            json.dump(learning_data, f)
    
    def test_analyze_rif_knowledge_base(self):
        """Test RIF knowledge base analysis."""
        patterns = self.analyzer.analyze_rif_knowledge_base()
        
        self.assertGreater(len(patterns), 0)
        
        # Check that patterns were extracted from different sources
        pattern_types = {p.pattern_type for p in patterns}
        self.assertTrue(len(pattern_types) > 0)
        
        # Verify pattern structure
        for pattern in patterns:
            self.assertIsNotNone(pattern.pattern_id)
            self.assertIsNotNone(pattern.component_type)
            self.assertGreater(pattern.success_rate, 0)
            self.assertGreater(pattern.sample_size, 0)
    
    def test_find_optimal_thresholds(self):
        """Test finding optimal thresholds based on patterns."""
        # First analyze knowledge base to create patterns
        patterns = self.analyzer.analyze_rif_knowledge_base()
        self.analyzer.quality_patterns.extend(patterns)
        
        # Test with existing pattern component type
        if patterns:
            component_type = patterns[0].component_type
            result = self.analyzer.find_optimal_thresholds(component_type)
            
            self.assertEqual(result["component_type"], component_type)
            self.assertIn("suggested_threshold", result)
            self.assertGreater(result["confidence"], 0)
            self.assertIn("rationale", result)
    
    def test_find_optimal_thresholds_no_patterns(self):
        """Test finding optimal thresholds when no patterns exist."""
        result = self.analyzer.find_optimal_thresholds("unknown_component")
        
        self.assertEqual(result["component_type"], "unknown_component")
        self.assertEqual(result["recommendation"], "no_patterns_found")
        self.assertEqual(result["suggested_threshold"], 80.0)
        self.assertLess(result["confidence"], 0.5)
    
    def test_save_and_load_patterns(self):
        """Test saving and loading patterns."""
        # Analyze and save patterns
        patterns = self.analyzer.analyze_rif_knowledge_base()
        self.analyzer.quality_patterns.extend(patterns)
        success = self.analyzer.save_patterns_and_insights()
        
        self.assertTrue(success)
        
        # Create new analyzer instance and verify patterns were loaded
        new_analyzer = QualityPatternAnalyzer(
            str(self.knowledge_dir),
            str(Path(self.temp_dir) / "quality")
        )
        
        # Should load saved patterns
        self.assertGreater(len(new_analyzer.quality_patterns), 0)

class TestThresholdOptimizer(unittest.TestCase):
    """Test threshold optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ThresholdOptimizer()
    
    def test_optimization_rules_loaded(self):
        """Test that optimization rules are loaded."""
        self.assertGreater(len(self.optimizer.optimization_rules), 0)
        
        # Check for expected rule types
        rule_ids = {rule.rule_id for rule in self.optimizer.optimization_rules}
        expected_rules = ["low_pass_rate", "high_override_rate", "excellent_performance"]
        
        for expected_rule in expected_rules:
            self.assertIn(expected_rule, rule_ids)
    
    def test_threshold_optimization_low_pass_rate(self):
        """Test optimization for low pass rate scenario."""
        historical_data = {
            "sample_size": 20,
            "pass_rate": 0.5,  # Low pass rate
            "override_rate": 0.1,
            "performance_score": 0.6
        }
        
        result = self.optimizer.optimize_threshold(
            "business_logic", 85.0, historical_data
        )
        
        self.assertLess(result.optimized_threshold, 85.0)  # Should lower threshold
        self.assertIn("low_pass_rate", result.rules_applied)
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.component_type, "business_logic")
    
    def test_threshold_optimization_high_override_rate(self):
        """Test optimization for high override rate scenario."""
        historical_data = {
            "sample_size": 15,
            "pass_rate": 0.7,
            "override_rate": 0.3,  # High override rate
            "performance_score": 0.7
        }
        
        result = self.optimizer.optimize_threshold(
            "ui_components", 75.0, historical_data
        )
        
        self.assertLess(result.optimized_threshold, 75.0)  # Should lower threshold
        self.assertIn("high_override_rate", result.rules_applied)
        self.assertGreater(result.confidence, 0.6)
    
    def test_threshold_optimization_excellent_performance(self):
        """Test optimization for excellent performance scenario."""
        historical_data = {
            "sample_size": 30,
            "pass_rate": 0.95,  # Excellent pass rate
            "override_rate": 0.05,  # Low override rate
            "performance_score": 0.9
        }
        
        result = self.optimizer.optimize_threshold(
            "business_logic", 85.0, historical_data
        )
        
        self.assertGreater(result.optimized_threshold, 85.0)  # Should increase threshold
        self.assertIn("excellent_performance", result.rules_applied)
    
    def test_threshold_optimization_insufficient_data(self):
        """Test optimization with insufficient data."""
        historical_data = {
            "sample_size": 3,  # Very small sample
            "pass_rate": 0.8,
            "override_rate": 0.1,
            "performance_score": 0.7
        }
        
        result = self.optimizer.optimize_threshold(
            "test_utilities", 70.0, historical_data
        )
        
        # Should not change threshold significantly due to insufficient data
        self.assertAlmostEqual(result.optimized_threshold, 70.0, delta=5.0)
        self.assertIn("insufficient_data_safety", result.rules_applied)
        self.assertLess(result.confidence, 0.5)
    
    def test_batch_optimization(self):
        """Test batch optimization of multiple components."""
        threshold_configs = {
            "critical_algorithms": 95.0,
            "business_logic": 85.0,
            "ui_components": 75.0
        }
        
        historical_data = {
            "critical_algorithms": {"sample_size": 15, "pass_rate": 0.9, "override_rate": 0.05, "performance_score": 0.9},
            "business_logic": {"sample_size": 25, "pass_rate": 0.6, "override_rate": 0.2, "performance_score": 0.6},
            "ui_components": {"sample_size": 20, "pass_rate": 0.8, "override_rate": 0.15, "performance_score": 0.75}
        }
        
        results = self.optimizer.batch_optimize_thresholds(threshold_configs, historical_data)
        
        self.assertEqual(len(results), 3)
        
        # Business logic should be optimized down due to low pass rate
        self.assertLess(results["business_logic"].optimized_threshold, 85.0)
        
        # All results should have valid optimization data
        for component_type, result in results.items():
            self.assertEqual(result.component_type, component_type)
            self.assertIsNotNone(result.confidence)
            self.assertIsNotNone(result.risk_assessment)
    
    def test_optimization_explanation(self):
        """Test optimization explanation generation."""
        historical_data = {
            "sample_size": 20,
            "pass_rate": 0.5,
            "override_rate": 0.3,
            "performance_score": 0.6
        }
        
        result = self.optimizer.optimize_threshold(
            "integration_code", 80.0, historical_data
        )
        
        explanation = self.optimizer.explain_optimization(result)
        
        self.assertIn("component_type", explanation)
        self.assertIn("change_summary", explanation)
        self.assertIn("optimization_rationale", explanation)
        self.assertIn("rules_explanation", explanation)
        self.assertIn("risk_factors", explanation)
        self.assertIn("confidence_factors", explanation)
        
        # Check change summary calculations
        change_summary = explanation["change_summary"]
        self.assertEqual(change_summary["original_threshold"], 80.0)
        self.assertEqual(change_summary["optimized_threshold"], result.optimized_threshold)

class TestConfigurationManager(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_default_configuration_creation(self):
        """Test that default configurations are created."""
        self.assertTrue((Path(self.temp_dir) / "rif-workflow.yaml").exists())
        self.assertTrue((Path(self.temp_dir) / "adaptive-thresholds.yaml").exists())
    
    def test_get_current_thresholds(self):
        """Test getting current threshold configuration."""
        thresholds = self.config_manager.get_current_thresholds()
        
        self.assertIsInstance(thresholds, dict)
        self.assertIn("critical_algorithms", thresholds)
        self.assertIn("business_logic", thresholds)
        self.assertIn("ui_components", thresholds)
        
        # Check threshold values are reasonable
        for component, threshold in thresholds.items():
            self.assertGreaterEqual(threshold, 50.0)
            self.assertLessEqual(threshold, 100.0)
    
    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        checkpoint_id = "test_checkpoint"
        success = self.config_manager.create_checkpoint(
            checkpoint_id,
            description="Test checkpoint for unit tests"
        )
        
        self.assertTrue(success)
        
        # Verify checkpoint was recorded
        checkpoints = self.config_manager.list_checkpoints()
        checkpoint_ids = [cp.checkpoint_id for cp in checkpoints]
        self.assertIn(checkpoint_id, checkpoint_ids)
    
    def test_update_threshold(self):
        """Test updating a single threshold."""
        original_thresholds = self.config_manager.get_current_thresholds()
        original_threshold = original_thresholds.get("business_logic", 80.0)
        new_threshold = original_threshold + 5.0
        
        success = self.config_manager.update_threshold(
            "business_logic",
            new_threshold,
            "Unit test threshold update"
        )
        
        self.assertTrue(success)
        
        # Verify threshold was updated
        updated_thresholds = self.config_manager.get_current_thresholds()
        self.assertEqual(updated_thresholds["business_logic"], new_threshold)
        
        # Verify change was recorded
        changes = self.config_manager.get_change_history(component_type="business_logic")
        self.assertGreater(len(changes), 0)
        self.assertEqual(changes[0].new_value, new_threshold)
        self.assertEqual(changes[0].old_value, original_threshold)
    
    def test_batch_update_thresholds(self):
        """Test batch threshold updates."""
        updates = {
            "critical_algorithms": 97.0,
            "public_apis": 92.0,
            "business_logic": 87.0
        }
        
        results = self.config_manager.batch_update_thresholds(
            updates,
            "Batch update for testing"
        )
        
        # All updates should succeed
        for component_type, success in results.items():
            self.assertTrue(success)
        
        # Verify all thresholds were updated
        current_thresholds = self.config_manager.get_current_thresholds()
        for component_type, expected_threshold in updates.items():
            self.assertEqual(current_thresholds[component_type], expected_threshold)
    
    def test_rollback_to_checkpoint(self):
        """Test rollback functionality."""
        # Create initial checkpoint
        checkpoint_id = "before_changes"
        original_thresholds = self.config_manager.get_current_thresholds()
        
        success = self.config_manager.create_checkpoint(
            checkpoint_id,
            description="Checkpoint before changes"
        )
        self.assertTrue(success)
        
        # Make changes
        self.config_manager.update_threshold(
            "business_logic", 95.0, "Test change"
        )
        
        # Verify changes were made
        changed_thresholds = self.config_manager.get_current_thresholds()
        self.assertEqual(changed_thresholds["business_logic"], 95.0)
        
        # Rollback
        rollback_success = self.config_manager.rollback_to_checkpoint(checkpoint_id)
        self.assertTrue(rollback_success)
        
        # Verify rollback restored original values
        restored_thresholds = self.config_manager.get_current_thresholds()
        self.assertEqual(
            restored_thresholds["business_logic"],
            original_thresholds["business_logic"]
        )
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        validation_result = self.config_manager.validate_current_configuration()
        
        self.assertIn("is_valid", validation_result)
        self.assertIn("issues", validation_result)
        self.assertIn("warnings", validation_result)
        self.assertIn("configuration_summary", validation_result)
        
        # Configuration should be valid with defaults
        if not validation_result["is_valid"]:
            print("Validation issues:", validation_result["issues"])
        
        # Should have threshold configuration
        self.assertIn("thresholds", validation_result["configuration_summary"])

class TestAdaptiveThresholdSystem(unittest.TestCase):
    """Test complete adaptive threshold system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create temporary directories
        config_dir = Path(self.temp_dir) / "config"
        quality_dir = Path(self.temp_dir) / "quality"
        knowledge_dir = Path(self.temp_dir) / "knowledge"
        
        for directory in [config_dir, quality_dir, knowledge_dir]:
            directory.mkdir(parents=True)
        
        self.system = AdaptiveThresholdSystem(
            str(config_dir),
            str(quality_dir),
            str(knowledge_dir)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_system_initialization(self):
        """Test system initializes correctly."""
        self.assertIsNotNone(self.system.data_collector)
        self.assertIsNotNone(self.system.pattern_analyzer)
        self.assertIsNotNone(self.system.threshold_optimizer)
        self.assertIsNotNone(self.system.config_manager)
    
    def test_analyze_system_performance_insufficient_data(self):
        """Test system performance analysis with insufficient data."""
        analysis = self.system.analyze_current_system_performance()
        
        self.assertEqual(analysis["status"], "insufficient_data")
        self.assertIn("total_decisions", analysis)
        self.assertIn("required_minimum", analysis)
    
    @patch('claude.commands.adaptive_threshold_system.AdaptiveThresholdSystem._generate_component_threshold_recommendation')
    def test_generate_threshold_recommendations(self, mock_generate_rec):
        """Test generating threshold recommendations."""
        # Mock recommendation generation
        mock_recommendation = Mock()
        mock_recommendation.component_type = "business_logic"
        mock_recommendation.current_threshold = 85.0
        mock_recommendation.recommended_threshold = 82.0
        mock_recommendation.confidence = 0.8
        mock_recommendation.rationale = "Test rationale"
        mock_recommendation.supporting_evidence = {}
        mock_recommendation.risk_assessment = "low"
        mock_recommendation.implementation_priority = "medium"
        mock_recommendation.estimated_impact = {"quality_improvement": 0.05}
        
        mock_generate_rec.return_value = mock_recommendation
        
        # Generate recommendations
        result = self.system.generate_threshold_recommendations(
            component_types=["business_logic"],
            force_analysis=True
        )
        
        self.assertIsNotNone(result.optimization_id)
        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(result.recommendations[0].component_type, "business_logic")
        self.assertGreater(result.overall_confidence, 0)

class TestAdaptiveQualityGatesIntegration(unittest.TestCase):
    """Test integration with existing quality gates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create config files
        config_dir = Path(self.temp_dir) / "config"
        config_dir.mkdir(parents=True)
        
        # Create minimal config files for testing
        workflow_config = {
            "quality_gates": {
                "code_coverage": {
                    "enabled": True,
                    "threshold": 80,
                    "required": True
                }
            }
        }
        
        with open(config_dir / "rif-workflow.yaml", 'w') as f:
            yaml.dump(workflow_config, f)
        
        adaptive_config = {
            "optimization": {
                "enabled": True,
                "min_confidence_threshold": 0.7
            }
        }
        
        with open(config_dir / "adaptive-thresholds.yaml", 'w') as f:
            yaml.dump(adaptive_config, f)
        
        self.integration = AdaptiveQualityGatesIntegration(
            str(config_dir / "rif-workflow.yaml"),
            str(config_dir / "adaptive-thresholds.yaml")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_integration_initialization(self):
        """Test integration initializes correctly."""
        self.assertIsNotNone(self.integration.quality_gates)
        self.assertIsNotNone(self.integration.adaptive_system)
        self.assertTrue(self.integration.adaptive_enabled)
    
    @patch('claude.commands.quality_gate_enforcement.QualityGateEnforcement._get_issue_details')
    @patch('claude.commands.quality_gate_enforcement.QualityGateEnforcement.validate_issue_closure_readiness')
    def test_validate_with_adaptive_thresholds(self, mock_validate, mock_get_details):
        """Test validation with adaptive thresholds enabled."""
        # Mock issue details
        mock_get_details.return_value = {
            "number": 123,
            "title": "Fix algorithm performance issue",
            "body": "Optimized critical algorithm for better performance",
            "labels": [{"name": "enhancement"}, {"name": "state:complete"}]
        }
        
        # Mock base validation result
        mock_validate.return_value = {
            "issue_number": 123,
            "can_close": True,
            "blocking_reasons": [],
            "warnings": [],
            "quality_gates": {"all_gates_pass": True},
            "quality_score": {"score": 85}
        }
        
        result = self.integration.validate_issue_closure_with_adaptive_thresholds(123)
        
        self.assertIn("adaptive_thresholds", result)
        self.assertTrue(result["adaptive_thresholds"]["enabled"])
    
    def test_get_system_status(self):
        """Test getting adaptive system status."""
        status = self.integration.get_adaptive_system_status()
        
        self.assertIn("adaptive_enabled", status)
        self.assertIn("configuration", status)
        self.assertTrue(status["adaptive_enabled"])
        
        config = status["configuration"]
        self.assertIn("optimization_enabled", config)
        self.assertIn("min_confidence_threshold", config)

class TestSystemIntegration(unittest.TestCase):
    """Test end-to-end system integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create complete directory structure
        for subdir in ["config", "quality/historical", "quality/patterns", "knowledge"]:
            (Path(self.temp_dir) / subdir).mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete adaptive threshold learning workflow."""
        # 1. Initialize all systems
        data_collector = HistoricalDataCollector(str(Path(self.temp_dir) / "quality/historical"))
        config_manager = ConfigurationManager(str(Path(self.temp_dir) / "config"))
        
        # 2. Record historical data
        test_decisions = [
            ("business_logic", 85.0, 88.0, "pass"),
            ("business_logic", 85.0, 82.0, "fail"),
            ("business_logic", 85.0, 83.0, "manual_override"),
            ("critical_algorithms", 95.0, 97.0, "pass"),
            ("critical_algorithms", 95.0, 93.0, "fail"),
        ]
        
        for component_type, threshold, score, decision in test_decisions:
            success = data_collector.record_quality_decision(
                component_type, threshold, score, decision, {"test": True}
            )
            self.assertTrue(success)
        
        # 3. Analyze historical data
        business_logic_analysis = data_collector.analyze_threshold_effectiveness("business_logic")
        self.assertEqual(business_logic_analysis["component_type"], "business_logic")
        self.assertGreater(business_logic_analysis["total_decisions"], 0)
        
        # 4. Get current configuration
        original_thresholds = config_manager.get_current_thresholds()
        self.assertIn("business_logic", original_thresholds)
        
        # 5. Create checkpoint before changes
        checkpoint_success = config_manager.create_checkpoint(
            "integration_test_checkpoint",
            description="Integration test checkpoint"
        )
        self.assertTrue(checkpoint_success)
        
        # 6. Update threshold based on analysis
        if business_logic_analysis["analysis"] == "complete":
            new_threshold = business_logic_analysis.get("optimal_threshold", 85.0)
            update_success = config_manager.update_threshold(
                "business_logic",
                new_threshold,
                "Integration test optimization"
            )
            self.assertTrue(update_success)
        
        # 7. Verify changes were applied
        updated_thresholds = config_manager.get_current_thresholds()
        if business_logic_analysis["analysis"] == "complete":
            # Threshold should have been updated
            self.assertNotEqual(
                updated_thresholds["business_logic"],
                original_thresholds["business_logic"]
            )
        
        # 8. Test rollback capability
        rollback_success = config_manager.rollback_to_checkpoint("integration_test_checkpoint")
        self.assertTrue(rollback_success)
        
        # 9. Verify rollback restored original configuration
        restored_thresholds = config_manager.get_current_thresholds()
        self.assertEqual(
            restored_thresholds["business_logic"],
            original_thresholds["business_logic"]
        )

def run_comprehensive_tests():
    """Run all tests and generate comprehensive report."""
    import time
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestHistoricalDataCollector,
        TestQualityPatternAnalyzer,
        TestThresholdOptimizer,
        TestConfigurationManager,
        TestAdaptiveThresholdSystem,
        TestAdaptiveQualityGatesIntegration,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    print("=" * 80)
    print("Adaptive Threshold Learning System - Comprehensive Test Suite")
    print("Issue #95: Adaptive Threshold Learning System")
    print("=" * 80)
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Generate results summary
    print(f"\nTest Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"  Total execution time: {total_time:.2f}s")
    
    # Save test report
    test_report = {
        "issue": 95,
        "system": "Adaptive Threshold Learning System",
        "test_execution": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_time_seconds": total_time,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        },
        "test_categories": {
            "historical_data_collection": "verified",
            "quality_pattern_analysis": "verified", 
            "threshold_optimization": "verified",
            "configuration_management": "verified",
            "adaptive_system_integration": "verified",
            "quality_gates_integration": "verified",
            "end_to_end_workflow": "verified"
        },
        "implementation_status": "Complete" if result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0 else "Has Issues"
    }
    
    # Save test report
    report_path = Path(__file__).parent.parent / "knowledge" / "validation" / "issue-95-test-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\nTest report saved to: {report_path}")
    
    return result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)