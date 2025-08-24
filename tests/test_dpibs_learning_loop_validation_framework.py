"""
Test Suite for DPIBS Learning Loop Effectiveness Validation Framework
Issue #131: Comprehensive validation testing

This test suite validates all components of the learning loop effectiveness
validation framework including statistical validation, A/B testing, and
all 4 phases of the validation process.

Author: RIF-Implementer  
Date: 2025-08-24
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add RIF paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from systems.dpibs_learning_loop_validation_framework import (
    StatisticalValidationFramework,
    ABTestingFramework,
    LearningLoopValidationFramework,
    BaselineMetrics,
    ValidationResult
)


class TestStatisticalValidationFramework:
    """Test suite for statistical validation framework components."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.stats_framework = StatisticalValidationFramework()
        
        # Sample data for testing
        self.baseline_data = [80.0, 82.0, 78.0, 85.0, 79.0, 81.0, 83.0, 77.0, 84.0, 80.5]
        self.treatment_data = [88.0, 90.0, 86.0, 92.0, 87.0, 89.0, 91.0, 85.0, 93.0, 88.5]
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        cohens_d = self.stats_framework.calculate_cohens_d(
            baseline=self.baseline_data,
            treatment=self.treatment_data
        )
        
        # Cohen's d should be positive and substantial (> 0.8 for large effect)
        assert cohens_d > 0.5, f"Cohen's d should be substantial, got {cohens_d}"
        assert isinstance(cohens_d, float), "Cohen's d should be a float"
    
    def test_paired_ttest(self):
        """Test paired t-test functionality."""
        t_stat, p_value = self.stats_framework.perform_paired_ttest(
            baseline=self.baseline_data,
            treatment=self.treatment_data
        )
        
        # Should detect significant improvement
        assert p_value < 0.05, f"P-value should indicate significance, got {p_value}"
        assert t_stat > 0, f"T-statistic should be positive for improvement, got {t_stat}"
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        ci_lower, ci_upper = self.stats_framework.calculate_confidence_interval(
            data=self.baseline_data
        )
        
        # Confidence interval should be reasonable
        data_mean = np.mean(self.baseline_data)
        assert ci_lower < data_mean < ci_upper, "Mean should be within confidence interval"
        assert ci_upper > ci_lower, "Upper bound should be greater than lower bound"
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple testing."""
        p_values = [0.01, 0.02, 0.03, 0.04]
        corrected = self.stats_framework.bonferroni_correction(p_values)
        
        # Corrected p-values should be larger
        for original, corrected_p in zip(p_values, corrected):
            assert corrected_p >= original, "Corrected p-value should be >= original"
        
        # Should not exceed 1.0
        for p in corrected:
            assert p <= 1.0, "Corrected p-value should not exceed 1.0"
    
    def test_validate_improvement_comprehensive(self):
        """Test comprehensive improvement validation."""
        result = self.stats_framework.validate_improvement(
            baseline=self.baseline_data,
            treatment=self.treatment_data,
            metric_name="test_metric"
        )
        
        # Validate ValidationResult structure
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        assert result.metric_name == "test_metric", "Metric name should match"
        assert result.improvement_percentage > 0, "Should detect improvement"
        assert result.statistical_significance, "Should be statistically significant"
        assert result.practical_significance, "Should be practically significant"


class TestABTestingFramework:
    """Test suite for A/B testing framework components."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.ab_framework = ABTestingFramework(minimum_sample_size=5)  # Small for testing
    
    @pytest.mark.asyncio
    async def test_experiment_creation(self):
        """Test A/B experiment creation."""
        success = await self.ab_framework.create_experiment(
            experiment_id="test_experiment",
            description="Test experiment",
            validation_areas=["learning_effectiveness", "performance_improvement"]
        )
        
        assert success, "Experiment creation should succeed"
        assert "test_experiment" in self.ab_framework.experiments, "Experiment should be stored"
    
    @pytest.mark.asyncio
    async def test_measurement_addition(self):
        """Test adding measurements to A/B experiment."""
        # Create experiment first
        await self.ab_framework.create_experiment(
            experiment_id="test_measurement",
            description="Test measurements",
            validation_areas=["test_metric"]
        )
        
        # Add measurements
        success_control = await self.ab_framework.add_measurement(
            experiment_id="test_measurement",
            group="control",
            validation_area="test_metric",
            value=80.0
        )
        
        success_treatment = await self.ab_framework.add_measurement(
            experiment_id="test_measurement", 
            group="treatment",
            validation_area="test_metric",
            value=90.0
        )
        
        assert success_control, "Control measurement addition should succeed"
        assert success_treatment, "Treatment measurement addition should succeed"
    
    @pytest.mark.asyncio
    async def test_experiment_analysis(self):
        """Test A/B experiment analysis."""
        experiment_id = "test_analysis"
        
        # Create experiment
        await self.ab_framework.create_experiment(
            experiment_id=experiment_id,
            description="Test analysis",
            validation_areas=["test_metric"]
        )
        
        # Add sufficient measurements
        for i in range(6):  # Above minimum sample size
            await self.ab_framework.add_measurement(
                experiment_id=experiment_id,
                group="control",
                validation_area="test_metric",
                value=80.0 + np.random.normal(0, 2)
            )
            await self.ab_framework.add_measurement(
                experiment_id=experiment_id,
                group="treatment", 
                validation_area="test_metric",
                value=90.0 + np.random.normal(0, 2)
            )
        
        # Analyze experiment
        results = await self.ab_framework.analyze_experiment(experiment_id)
        
        assert "test_metric" in results, "Should return analysis for test_metric"
        assert isinstance(results["test_metric"], ValidationResult), "Should return ValidationResult"
    
    def test_experiment_status(self):
        """Test experiment status reporting."""
        # Test non-existent experiment
        status = self.ab_framework.get_experiment_status("non_existent")
        assert "error" in status, "Should return error for non-existent experiment"


class TestLearningLoopValidationFramework:
    """Test suite for main learning loop validation framework."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.validation_framework = LearningLoopValidationFramework()
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test validation framework initialization."""
        with patch.object(self.validation_framework.ab_testing_framework, 'create_experiment', 
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = True
            
            success = await self.validation_framework.initialize()
            assert success, "Framework initialization should succeed"
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_phase1_baseline_establishment(self):
        """Test Phase 1: Learning System Baseline Establishment."""
        with patch.object(self.validation_framework.ab_testing_framework, 'add_measurement',
                         new_callable=AsyncMock) as mock_add:
            mock_add.return_value = True
            
            baseline_metrics = await self.validation_framework.phase1_establish_baseline()
            
            # Validate baseline metrics structure
            assert isinstance(baseline_metrics, BaselineMetrics), "Should return BaselineMetrics"
            assert baseline_metrics.sample_size > 0, "Should have positive sample size"
            assert baseline_metrics.learning_extraction_accuracy > 0, "Should have positive accuracy"
            
            # Verify measurements were added to A/B framework
            assert mock_add.call_count > 0, "Should add measurements to A/B framework"
    
    @pytest.mark.asyncio 
    async def test_phase2_learning_loop_activation(self):
        """Test Phase 2: Learning Loop Activation and Measurement."""
        with patch.object(self.validation_framework.ab_testing_framework, 'add_measurement',
                         new_callable=AsyncMock) as mock_add:
            mock_add.return_value = True
            
            results = await self.validation_framework.phase2_activate_learning_loop()
            
            # Validate activation results
            assert results['activation_status'] == 'successful', "Activation should succeed"
            assert results['measurement_collection'] == 'complete', "Measurement collection should complete"
            assert results['treatment_sample_size'] > 0, "Should have positive sample size"
            
            # Validate performance requirements
            performance = results['performance_validation']
            assert performance['learning_extraction_time'] < 30.0, "Should meet time requirement"
            assert performance['context_enhancement_latency'] < 0.2, "Should meet latency requirement"
            assert performance['classification_accuracy'] > 0.9, "Should meet accuracy requirement"
    
    @pytest.mark.asyncio
    async def test_phase3_statistical_validation(self):
        """Test Phase 3: Effectiveness Analysis and Statistical Validation."""
        # Setup mock validation results
        mock_validation_result = ValidationResult(
            metric_name="test_metric",
            baseline_value=80.0,
            treatment_value=90.0,
            improvement_percentage=12.5,
            p_value=0.01,
            cohens_d=0.8,
            confidence_interval=(8.0, 17.0),
            statistical_significance=True,
            practical_significance=True
        )
        
        with patch.object(self.validation_framework.ab_testing_framework, 'analyze_experiment',
                         new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {"test_metric": mock_validation_result}
            
            results = await self.validation_framework.phase3_statistical_validation()
            
            assert "test_metric" in results, "Should return validation results"
            assert isinstance(results["test_metric"], ValidationResult), "Should return ValidationResult"
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_phase4_longterm_integration(self):
        """Test Phase 4: Long-term Validation and Integration."""
        # Setup some validation results first
        self.validation_framework.validation_results = {
            "test_metric": ValidationResult(
                metric_name="test_metric",
                baseline_value=80.0,
                treatment_value=90.0,
                improvement_percentage=12.5,
                p_value=0.01,
                cohens_d=0.8,
                confidence_interval=(8.0, 17.0),
                statistical_significance=True,
                practical_significance=True
            )
        }
        
        results = await self.validation_framework.phase4_longterm_integration()
        
        # Validate integration results
        assert results['integration_status'] == 'complete', "Integration should complete"
        assert results['long_term_tracking_enabled'], "Long-term tracking should be enabled"
        assert 'monitoring_configuration' in results, "Should include monitoring configuration"
        assert 'validation_documentation' in results, "Should include validation documentation"
    
    @pytest.mark.asyncio
    async def test_full_validation_workflow(self):
        """Test complete validation workflow integration."""
        with patch.object(self.validation_framework, 'initialize', new_callable=AsyncMock) as mock_init, \
             patch.object(self.validation_framework, 'phase1_establish_baseline', new_callable=AsyncMock) as mock_p1, \
             patch.object(self.validation_framework, 'phase2_activate_learning_loop', new_callable=AsyncMock) as mock_p2, \
             patch.object(self.validation_framework, 'phase3_statistical_validation', new_callable=AsyncMock) as mock_p3, \
             patch.object(self.validation_framework, 'phase4_longterm_integration', new_callable=AsyncMock) as mock_p4:
            
            # Setup mock returns
            mock_init.return_value = True
            mock_p1.return_value = BaselineMetrics(
                learning_extraction_accuracy=85.0,
                agent_performance_scores={'test_agent': 80.0},
                system_context_accuracy=75.0,
                development_process_efficiency=12.0,
                measurement_timestamp=datetime.now().isoformat(),
                sample_size=30,
                confidence_interval=(82.0, 88.0)
            )
            mock_p2.return_value = {'activation_status': 'successful', 'requirements_met': True}
            mock_p3.return_value = {
                'test_metric': ValidationResult(
                    metric_name="test_metric",
                    baseline_value=80.0,
                    treatment_value=90.0,
                    improvement_percentage=12.5,
                    p_value=0.01,
                    cohens_d=0.8,
                    confidence_interval=(8.0, 17.0),
                    statistical_significance=True,
                    practical_significance=True
                )
            }
            mock_p4.return_value = {'integration_status': 'complete'}
            
            # Run full validation
            results = await self.validation_framework.run_full_validation()
            
            # Validate workflow completion
            assert 'phase1_baseline' in results, "Should include Phase 1 results"
            assert 'phase2_activation' in results, "Should include Phase 2 results"
            assert 'phase3_statistical' in results, "Should include Phase 3 results"
            assert 'phase4_integration' in results, "Should include Phase 4 results"
            assert 'validation_summary' in results, "Should include validation summary"
            
            # Validate summary
            summary = results['validation_summary']
            assert summary['all_phases_complete'], "All phases should be complete"
            assert summary['overall_validation_success'], "Overall validation should succeed"


class TestValidationFrameworkIntegration:
    """Integration tests for validation framework with external systems."""
    
    @pytest.mark.asyncio
    async def test_knowledge_system_integration(self):
        """Test integration with knowledge system."""
        validation_framework = LearningLoopValidationFramework()
        
        # Test with mock knowledge system
        with patch.object(validation_framework, 'knowledge_system') as mock_knowledge:
            mock_knowledge.store_pattern = AsyncMock(return_value=True)
            
            # Run phase 4 which stores documentation
            results = await validation_framework.phase4_longterm_integration()
            
            # Should complete without knowledge system errors
            assert results['integration_status'] == 'complete'
    
    @pytest.mark.asyncio
    async def test_learning_system_integration(self):
        """Test integration with learning system."""
        validation_framework = LearningLoopValidationFramework()
        
        # Test initialization with mock learning system
        with patch.object(validation_framework, 'learning_system') as mock_learning:
            mock_learning.initialize = AsyncMock(return_value=True)
            mock_learning.shutdown = AsyncMock(return_value=True)
            
            # Initialize and shutdown should work with learning system
            success = await validation_framework.initialize()
            assert success, "Initialization with learning system should succeed"
            
            await validation_framework.shutdown()
    
    @pytest.mark.asyncio
    async def test_dpibs_infrastructure_integration(self):
        """Test comprehensive integration with DPIBS infrastructure components."""
        validation_framework = LearningLoopValidationFramework()
        
        # Mock DPIBS components that would be integrated
        mock_dpibs_components = {
            'knowledge_integration_apis': Mock(),
            'system_context_apis': Mock(), 
            'enhanced_learning_integration': Mock(),
            'benchmarking_framework': Mock()
        }
        
        # Test integration with each DPIBS component
        with patch.multiple(
            'systems.dpibs_learning_loop_validation_framework',
            **{f'get_{component}': Mock(return_value=mock_comp) 
               for component, mock_comp in mock_dpibs_components.items()}
        ):
            # Initialize framework with mocked DPIBS components
            success = await validation_framework.initialize()
            assert success, "DPIBS integration initialization should succeed"
            
            # Run full validation to test all integration points
            with patch.object(validation_framework.ab_testing_framework, 'create_experiment', 
                             new_callable=AsyncMock) as mock_create, \
                 patch.object(validation_framework.ab_testing_framework, 'add_measurement',
                             new_callable=AsyncMock) as mock_add, \
                 patch.object(validation_framework.ab_testing_framework, 'analyze_experiment',
                             new_callable=AsyncMock) as mock_analyze:
                
                mock_create.return_value = True
                mock_add.return_value = True
                mock_analyze.return_value = {
                    'learning_extraction_effectiveness': ValidationResult(
                        metric_name='learning_extraction_effectiveness',
                        baseline_value=85.0,
                        treatment_value=92.5,
                        improvement_percentage=8.8,
                        p_value=0.01,
                        cohens_d=0.8,
                        confidence_interval=(5.0, 12.0),
                        statistical_significance=True,
                        practical_significance=True
                    )
                }
                
                results = await validation_framework.run_full_validation()
                
                # Validate integration success
                assert results['validation_summary']['all_phases_complete']
                assert results['validation_summary']['integration_successful']
                assert results['validation_summary']['overall_validation_success']
    
    @pytest.mark.asyncio
    async def test_dpibs_performance_integration(self):
        """Test DPIBS performance monitoring integration."""
        validation_framework = LearningLoopValidationFramework()
        
        # Test performance monitoring integration
        with patch('systems.dpibs_learning_loop_validation_framework.get_knowledge_system') as mock_get_knowledge:
            mock_knowledge = Mock()
            mock_knowledge.store_pattern = AsyncMock(return_value=True)
            mock_get_knowledge.return_value = mock_knowledge
            
            # Run phase 4 integration which includes performance monitoring
            results = await validation_framework.phase4_longterm_integration()
            
            # Validate performance monitoring setup
            monitoring_config = results['monitoring_configuration']
            assert monitoring_config['monitoring_enabled']
            assert 'daily_metrics' in monitoring_config['validation_intervals']
            assert 'performance_degradation' in monitoring_config['alert_thresholds']
            assert 'enhanced_learning_integration_system' in monitoring_config['integration_points']


class TestPerformanceValidation:
    """Test suite for performance validation and requirements."""
    
    @pytest.mark.asyncio
    async def test_performance_requirements_validation(self):
        """Test validation of performance requirements."""
        validation_framework = LearningLoopValidationFramework()
        
        with patch.object(validation_framework.ab_testing_framework, 'add_measurement',
                         new_callable=AsyncMock) as mock_add:
            mock_add.return_value = True
            
            # Test Phase 2 which validates performance requirements
            results = await validation_framework.phase2_activate_learning_loop()
            
            performance = results['performance_validation']
            
            # Validate each performance requirement
            assert performance['learning_extraction_time'] <= 30.0, "Should meet extraction time requirement"
            assert performance['context_enhancement_latency'] <= 0.2, "Should meet latency requirement"
            assert performance['classification_accuracy'] >= 0.9, "Should meet accuracy requirement"
            assert performance['performance_improvement_correlation'] >= 0.8, "Should meet correlation requirement"
            
            assert results['requirements_met'], "All requirements should be met"


class TestEdgeCaseHandling:
    """Test suite for edge case handling in statistical functions."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.stats_framework = StatisticalValidationFramework()
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Test with empty lists
        cohens_d = self.stats_framework.calculate_cohens_d([], [])
        assert cohens_d == 0.0, "Should return 0.0 for empty data"
        
        t_stat, p_value = self.stats_framework.perform_paired_ttest([], [])
        assert t_stat == 0.0 and p_value == 1.0, "Should return safe values for empty data"
        
        ci_lower, ci_upper = self.stats_framework.calculate_confidence_interval([])
        assert ci_lower == 0.0 and ci_upper == 0.0, "Should return (0.0, 0.0) for empty data"
    
    def test_nan_data_handling(self):
        """Test handling of NaN values in datasets."""
        baseline = [80.0, float('nan'), 82.0, 85.0]
        treatment = [88.0, float('nan'), 90.0, 92.0]
        
        # Should filter out NaN values and calculate correctly
        cohens_d = self.stats_framework.calculate_cohens_d(baseline, treatment)
        assert not np.isnan(cohens_d), "Should not return NaN after filtering"
        assert cohens_d > 0, "Should calculate positive effect size"
        
        # Test validation with NaN data
        result = self.stats_framework.validate_improvement(baseline, treatment, "test_metric")
        assert not np.isnan(result.improvement_percentage), "Should not return NaN improvement"
        assert not np.isnan(result.cohens_d), "Should not return NaN Cohen's d"
        assert not np.isnan(result.p_value), "Should not return NaN p-value"
    
    def test_zero_variance_handling(self):
        """Test handling of datasets with zero variance."""
        # Identical values (zero variance)
        baseline = [80.0, 80.0, 80.0, 80.0]
        treatment = [90.0, 90.0, 90.0, 90.0]
        
        # Should handle zero variance gracefully
        cohens_d = self.stats_framework.calculate_cohens_d(baseline, treatment)
        assert not np.isnan(cohens_d), "Should handle zero variance without NaN"
        
        # Confidence interval should equal the mean for zero variance
        ci_lower, ci_upper = self.stats_framework.calculate_confidence_interval(baseline)
        assert ci_lower == ci_upper == 80.0, "CI should equal mean for zero variance"
    
    def test_single_value_handling(self):
        """Test handling of single-value datasets."""
        baseline = [80.0]
        treatment = [90.0]
        
        # Should handle insufficient data gracefully
        cohens_d = self.stats_framework.calculate_cohens_d(baseline, treatment)
        assert cohens_d == 0.0, "Should return 0.0 for insufficient data"
        
        ci_lower, ci_upper = self.stats_framework.calculate_confidence_interval(baseline)
        assert ci_lower == 0.0 and ci_upper == 0.0, "Should return (0.0, 0.0) for insufficient data"
    
    def test_division_by_zero_handling(self):
        """Test handling of division by zero scenarios."""
        # Baseline with zero mean
        baseline = [0.0, 0.0, 0.0]
        treatment = [10.0, 12.0, 8.0]
        
        result = self.stats_framework.validate_improvement(baseline, treatment, "test_metric")
        assert not np.isnan(result.improvement_percentage), "Should handle zero baseline without NaN"
        assert not np.isinf(result.improvement_percentage), "Should handle zero baseline without infinity"
    
    def test_mixed_data_types_handling(self):
        """Test handling of mixed data types and invalid values."""
        baseline = [80.0, "invalid", 82.0, None, 85.0]
        treatment = [88.0, float('inf'), 90.0, -float('inf'), 92.0]
        
        # Should filter out invalid data types and values
        result = self.stats_framework.validate_improvement(baseline, treatment, "test_metric")
        assert isinstance(result, ValidationResult), "Should return ValidationResult despite invalid data"
        assert not np.isnan(result.baseline_value), "Should calculate valid baseline after filtering"
        assert not np.isnan(result.treatment_value), "Should calculate valid treatment after filtering"
    
    def test_mismatched_length_handling(self):
        """Test handling of mismatched dataset lengths."""
        baseline = [80.0, 82.0, 85.0]
        treatment = [88.0, 90.0, 92.0, 94.0, 96.0]  # Different length
        
        # Should handle mismatched lengths for unpaired operations
        cohens_d = self.stats_framework.calculate_cohens_d(baseline, treatment)
        assert not np.isnan(cohens_d), "Should handle mismatched lengths"
        
        # Paired t-test should fail gracefully with mismatched lengths
        t_stat, p_value = self.stats_framework.perform_paired_ttest(baseline, treatment)
        assert t_stat == 0.0 and p_value == 1.0, "Should return safe values for mismatched lengths"


class TestDPIBSIntegrationValidation:
    """Comprehensive DPIBS integration validation test suite."""
    
    @pytest.mark.asyncio
    async def test_full_dpibs_workflow_integration(self):
        """Test complete DPIBS workflow integration with all components."""
        validation_framework = LearningLoopValidationFramework()
        
        # Mock all DPIBS integration points
        mocks = {
            'knowledge_system': Mock(),
            'learning_system': Mock(),
            'context_system': Mock(),
            'benchmarking_system': Mock()
        }
        
        # Setup mock behaviors
        mocks['knowledge_system'].store_pattern = AsyncMock(return_value=True)
        mocks['learning_system'].initialize = AsyncMock(return_value=True)
        mocks['learning_system'].shutdown = AsyncMock(return_value=True)
        
        with patch.multiple(
            validation_framework,
            knowledge_system=mocks['knowledge_system'],
            learning_system=mocks['learning_system']
        ):
            # Test complete workflow with DPIBS integration
            with patch.object(validation_framework.ab_testing_framework, 'create_experiment',
                             new_callable=AsyncMock) as mock_create, \
                 patch.object(validation_framework.ab_testing_framework, 'add_measurement', 
                             new_callable=AsyncMock) as mock_add, \
                 patch.object(validation_framework.ab_testing_framework, 'analyze_experiment',
                             new_callable=AsyncMock) as mock_analyze:
                
                mock_create.return_value = True
                mock_add.return_value = True
                mock_analyze.return_value = {
                    area: ValidationResult(
                        metric_name=area,
                        baseline_value=80.0,
                        treatment_value=95.0,
                        improvement_percentage=18.75,
                        p_value=0.001,
                        cohens_d=1.2,
                        confidence_interval=(12.0, 25.0),
                        statistical_significance=True,
                        practical_significance=True
                    )
                    for area in validation_framework.validation_areas
                }
                
                # Execute full workflow
                results = await validation_framework.run_full_validation()
                
                # Comprehensive validation of integration
                assert results['validation_summary']['all_phases_complete']
                assert results['validation_summary']['statistical_validation_success']
                assert results['validation_summary']['performance_requirements_met']
                assert results['validation_summary']['integration_successful']
                assert results['validation_summary']['overall_validation_success']
                
                # Verify all DPIBS integration points were exercised
                mocks['knowledge_system'].store_pattern.assert_called()
                
                # Verify all validation areas were processed
                for area in validation_framework.validation_areas:
                    assert area in results['phase3_statistical']
                    assert results['phase3_statistical'][area]['statistical_significance']
                    assert results['phase3_statistical'][area]['practical_significance']
    
    @pytest.mark.asyncio
    async def test_dpibs_failure_recovery(self):
        """Test recovery from DPIBS component failures."""
        validation_framework = LearningLoopValidationFramework()
        
        # Test recovery from knowledge system failure
        with patch.object(validation_framework, 'knowledge_system') as mock_knowledge:
            mock_knowledge.store_pattern = AsyncMock(side_effect=Exception("Knowledge system error"))
            
            # Should complete despite knowledge system failure
            results = await validation_framework.phase4_longterm_integration()
            assert results['integration_status'] == 'complete'
    
    @pytest.mark.asyncio
    async def test_dpibs_performance_compliance(self):
        """Test DPIBS performance requirement compliance."""
        validation_framework = LearningLoopValidationFramework()
        
        # Verify performance requirements align with DPIBS specifications
        requirements = validation_framework.performance_requirements
        
        assert requirements['learning_extraction_time'] <= 30.0, "DPIBS learning extraction time requirement"
        assert requirements['context_enhancement_latency'] <= 0.2, "DPIBS context latency requirement" 
        assert requirements['classification_accuracy'] >= 0.90, "DPIBS classification accuracy requirement"
        assert requirements['performance_improvement_correlation'] >= 0.80, "DPIBS correlation requirement"
        
        # Test actual performance validation
        with patch.object(validation_framework.ab_testing_framework, 'add_measurement',
                         new_callable=AsyncMock) as mock_add:
            mock_add.return_value = True
            
            results = await validation_framework.phase2_activate_learning_loop()
            performance = results['performance_validation']
            
            # All DPIBS requirements should be met
            assert all(
                performance[key] <= threshold if 'time' in key or 'latency' in key
                else performance[key] >= threshold
                for key, threshold in requirements.items()
            ), "All DPIBS performance requirements should be satisfied"


if __name__ == "__main__":
    """Run tests with pytest."""
    print("Running DPIBS Learning Loop Validation Framework Tests...")
    print("="*60)
    
    # Run tests using pytest programmatically
    import pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-k", "not slow"  # Skip slow integration tests by default
    ])
    
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
        print("\n🔍 Edge case handling validated:")
        print("   - Empty data scenarios")
        print("   - NaN value filtering")
        print("   - Division by zero protection")
        print("   - Zero variance handling")
        print("   - Mixed data type validation")
        print("\n🔗 DPIBS integration validated:")
        print("   - Full workflow integration")
        print("   - Performance compliance")
        print("   - Failure recovery")
        print("   - Component interaction")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    exit(exit_code)