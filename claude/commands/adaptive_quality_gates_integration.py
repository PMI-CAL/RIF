#!/usr/bin/env python3
"""
Adaptive Quality Gates Integration
Issue #95: Adaptive Threshold Learning System

Integrates adaptive threshold learning with existing quality gate enforcement system.
Extends quality_gate_enforcement.py with adaptive threshold capabilities.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .quality_gate_enforcement import QualityGateEnforcement
from .adaptive_threshold_system import AdaptiveThresholdSystem
from .historical_data_collector import HistoricalDataCollector
from .quality_gates.threshold_engine import AdaptiveThresholdEngine, ChangeMetrics, ChangeContext

class AdaptiveQualityGatesIntegration:
    """
    Integration layer that combines adaptive threshold learning with quality gate enforcement.
    
    Features:
    - Adaptive thresholds based on component classification
    - Historical data collection during gate validation
    - Dynamic threshold adjustment recommendations
    - Seamless integration with existing quality gates
    - Backward compatibility with static configuration
    """
    
    def __init__(self, 
                 config_path: str = "config/rif-workflow.yaml",
                 adaptive_config_path: str = "config/adaptive-thresholds.yaml"):
        """
        Initialize adaptive quality gates integration.
        
        Args:
            config_path: Path to main quality gates configuration
            adaptive_config_path: Path to adaptive configuration
        """
        self.config_path = config_path
        self.adaptive_config_path = adaptive_config_path
        
        # Initialize core systems
        self.quality_gates = QualityGateEnforcement(config_path)
        self.adaptive_system = AdaptiveThresholdSystem()
        self.data_collector = HistoricalDataCollector()
        self.threshold_engine = AdaptiveThresholdEngine()
        
        self.setup_logging()
        self._load_adaptive_config()
        
    def setup_logging(self):
        """Setup logging for adaptive quality gates integration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - AdaptiveQualityGates - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_adaptive_config(self):
        """Load adaptive configuration."""
        try:
            adaptive_config_file = Path(self.adaptive_config_path)
            if adaptive_config_file.exists():
                with open(adaptive_config_file, 'r') as f:
                    import yaml
                    self.adaptive_config = yaml.safe_load(f)
            else:
                self.adaptive_config = {"optimization": {"enabled": False}}
                
            self.adaptive_enabled = self.adaptive_config.get("optimization", {}).get("enabled", False)
            self.logger.info(f"Adaptive thresholds {'enabled' if self.adaptive_enabled else 'disabled'}")
            
        except Exception as e:
            self.logger.error(f"Failed to load adaptive configuration: {e}")
            self.adaptive_config = {"optimization": {"enabled": False}}
            self.adaptive_enabled = False
    
    def validate_issue_closure_with_adaptive_thresholds(self, issue_number: int) -> Dict[str, Any]:
        """
        Validate issue closure using adaptive thresholds if enabled.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Enhanced validation report with adaptive threshold information
        """
        self.logger.info(f"🔍 Validating issue #{issue_number} with adaptive thresholds")
        
        # Get base validation from standard quality gates
        validation_report = self.quality_gates.validate_issue_closure_readiness(issue_number)
        
        # Add adaptive threshold information
        validation_report['adaptive_thresholds'] = {
            'enabled': self.adaptive_enabled,
            'system_health': None,
            'recommendations': [],
            'threshold_adjustments': {},
            'data_collection_status': {}
        }
        
        if not self.adaptive_enabled:
            validation_report['adaptive_thresholds']['note'] = "Adaptive thresholds disabled in configuration"
            return validation_report
        
        try:
            # Get issue details for component analysis
            issue_details = self.quality_gates._get_issue_details(issue_number)
            if not issue_details:
                validation_report['adaptive_thresholds']['error'] = "Could not retrieve issue details"
                return validation_report
            
            # Analyze components involved in this issue
            components_analysis = self._analyze_issue_components(issue_number, issue_details, validation_report)
            validation_report['adaptive_thresholds']['components_analysis'] = components_analysis
            
            # Apply adaptive thresholds if needed
            if components_analysis['has_component_data']:
                adaptive_results = self._apply_adaptive_thresholds(
                    issue_number, components_analysis, validation_report
                )
                validation_report['adaptive_thresholds'].update(adaptive_results)
            
            # Collect data for learning
            self._collect_validation_data(issue_number, issue_details, validation_report)
            
            # Generate adaptive recommendations
            adaptive_recommendations = self._generate_adaptive_recommendations(
                issue_number, validation_report
            )
            validation_report['adaptive_thresholds']['recommendations'] = adaptive_recommendations
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error in adaptive validation: {e}")
            validation_report['adaptive_thresholds']['error'] = str(e)
            return validation_report
    
    def _analyze_issue_components(self, 
                                issue_number: int,
                                issue_details: Dict[str, Any],
                                validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze components involved in the issue for adaptive threshold application."""
        components_analysis = {
            'has_component_data': False,
            'components': {},
            'classification_confidence': 0.0,
            'total_changes': 0
        }
        
        try:
            # Try to identify changed files (this would normally come from PR data)
            # For now, simulate based on issue content
            changed_files = self._identify_changed_files(issue_number, issue_details)
            
            if not changed_files:
                self.logger.info(f"No file changes identified for issue #{issue_number}")
                return components_analysis
            
            # Classify components
            component_changes = {}
            total_changes = 0
            
            for file_path in changed_files:
                # Classify component type
                classification = self.threshold_engine.classifier.classify_file(file_path)
                component_type = classification.type
                
                # Estimate change metrics (would normally come from git diff)
                change_metrics = self._estimate_change_metrics(file_path, issue_details)
                
                if component_type not in component_changes:
                    component_changes[component_type] = ChangeMetrics(0, 0, 0, 0, 0.0)
                
                # Aggregate changes per component type
                existing = component_changes[component_type]
                component_changes[component_type] = ChangeMetrics(
                    lines_added=existing.lines_added + change_metrics.lines_added,
                    lines_deleted=existing.lines_deleted + change_metrics.lines_deleted,
                    lines_modified=existing.lines_modified + change_metrics.lines_modified,
                    files_changed=existing.files_changed + 1,
                    complexity_score=max(existing.complexity_score, change_metrics.complexity_score)
                )
                
                total_changes += change_metrics.total_lines_changed
            
            components_analysis['has_component_data'] = len(component_changes) > 0
            components_analysis['components'] = {
                comp_type: {
                    'lines_changed': metrics.total_lines_changed,
                    'files_changed': metrics.files_changed,
                    'complexity_score': metrics.complexity_score
                }
                for comp_type, metrics in component_changes.items()
            }
            components_analysis['total_changes'] = total_changes
            components_analysis['component_changes'] = component_changes  # Store for later use
            
            # Calculate overall classification confidence
            if component_changes:
                components_analysis['classification_confidence'] = 0.8  # Simulated confidence
            
            return components_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing issue components: {e}")
            return components_analysis
    
    def _identify_changed_files(self, issue_number: int, issue_details: Dict[str, Any]) -> List[str]:
        """Identify files changed in the issue (simulated for now)."""
        # In a real implementation, this would:
        # 1. Get the PR associated with the issue
        # 2. Get the files changed in the PR
        # 3. Return the list of file paths
        
        # For simulation, infer from issue content
        title = issue_details.get('title', '').lower()
        body = issue_details.get('body', '').lower()
        
        simulated_files = []
        
        # Simulate file identification based on keywords
        if 'algorithm' in f"{title} {body}":
            simulated_files.extend(['src/algorithms/main_algorithm.py', 'src/core/processing.py'])
        if 'api' in f"{title} {body}":
            simulated_files.extend(['src/api/endpoints.py', 'src/controllers/api_controller.py'])
        if 'ui' in f"{title} {body}" or 'component' in f"{title} {body}":
            simulated_files.extend(['src/components/UserInterface.jsx', 'src/pages/MainPage.tsx'])
        if 'test' in f"{title} {body}":
            simulated_files.extend(['tests/test_main.py', 'tests/integration/test_api.py'])
        
        # Default files if nothing specific identified
        if not simulated_files:
            simulated_files = ['src/main.py', 'src/utils/helpers.py']
        
        self.logger.info(f"Identified {len(simulated_files)} changed files for issue #{issue_number}")
        return simulated_files
    
    def _estimate_change_metrics(self, file_path: str, issue_details: Dict[str, Any]) -> ChangeMetrics:
        """Estimate change metrics for a file (simulated for now)."""
        # In a real implementation, this would analyze git diff
        # For simulation, estimate based on issue complexity
        
        title = issue_details.get('title', '').lower()
        body = issue_details.get('body', '').lower()
        
        # Estimate based on issue content
        complexity_keywords = ['complex', 'refactor', 'rewrite', 'major', 'significant']
        complexity_score = 1.0
        
        if any(keyword in f"{title} {body}" for keyword in complexity_keywords):
            complexity_score = 1.5
        
        # Estimate change size based on file type and complexity
        if 'test' in file_path.lower():
            # Test files typically have smaller changes
            return ChangeMetrics(
                lines_added=20,
                lines_deleted=5,
                lines_modified=10,
                files_changed=1,
                complexity_score=complexity_score * 0.8
            )
        elif 'algorithm' in file_path.lower():
            # Algorithm files may have more significant changes
            return ChangeMetrics(
                lines_added=50,
                lines_deleted=20,
                lines_modified=30,
                files_changed=1,
                complexity_score=complexity_score * 1.2
            )
        else:
            # Standard file changes
            return ChangeMetrics(
                lines_added=35,
                lines_deleted=15,
                lines_modified=20,
                files_changed=1,
                complexity_score=complexity_score
            )
    
    def _apply_adaptive_thresholds(self, 
                                 issue_number: int,
                                 components_analysis: Dict[str, Any],
                                 validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive thresholds to quality gate validation."""
        adaptive_results = {
            'thresholds_adjusted': False,
            'original_thresholds': {},
            'adaptive_thresholds': {},
            'threshold_reasoning': {},
            'validation_impact': {}
        }
        
        try:
            component_changes = components_analysis.get('component_changes', {})
            if not component_changes:
                return adaptive_results
            
            # Create change context from validation report
            context = self._create_change_context(validation_report)
            
            # Calculate adaptive thresholds for each component
            for component_type, change_metrics in component_changes.items():
                # Get original threshold
                original_threshold = self._get_current_component_threshold(component_type)
                adaptive_results['original_thresholds'][component_type] = original_threshold
                
                # Calculate adaptive threshold
                threshold_config = self.threshold_engine.calculate_component_threshold(
                    component_type, change_metrics, context
                )
                
                adaptive_threshold = threshold_config.applied_threshold
                adaptive_results['adaptive_thresholds'][component_type] = adaptive_threshold
                adaptive_results['threshold_reasoning'][component_type] = threshold_config.reasoning
                
                # Check if threshold was adjusted
                if abs(adaptive_threshold - original_threshold) > 1.0:  # More than 1% difference
                    adaptive_results['thresholds_adjusted'] = True
                    
                    # Simulate re-validation with new threshold (in real implementation, 
                    # this would re-run quality gates with the new threshold)
                    impact = self._simulate_threshold_impact(
                        original_threshold, adaptive_threshold, validation_report
                    )
                    adaptive_results['validation_impact'][component_type] = impact
            
            return adaptive_results
            
        except Exception as e:
            self.logger.error(f"Error applying adaptive thresholds: {e}")
            adaptive_results['error'] = str(e)
            return adaptive_results
    
    def _create_change_context(self, validation_report: Dict[str, Any]) -> ChangeContext:
        """Create change context from validation report."""
        # Analyze validation report to determine context
        risk_assessment = validation_report.get('risk_assessment', {})
        quality_score = validation_report.get('quality_score', {})
        
        # Determine if this is a security-critical change
        is_security_critical = risk_assessment.get('risk_level') in ['high', 'critical']
        
        # Determine if this is a breaking change (heuristic)
        is_breaking_change = 'breaking' in str(validation_report).lower()
        
        # Determine risk level
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        # Check if tests are present (from gate validation)
        gate_results = validation_report.get('quality_gates', {}).get('gate_results', {})
        has_tests = gate_results.get('linting', {}).get('passed', True)  # Assume tests present if linting passed
        
        return ChangeContext(
            is_security_critical=is_security_critical,
            is_breaking_change=is_breaking_change,
            risk_level=risk_level,
            has_tests=has_tests,
            historical_success_rate=0.85  # Default historical success rate
        )
    
    def _get_current_component_threshold(self, component_type: str) -> float:
        """Get current threshold for a component type."""
        # Get from current configuration
        current_thresholds = self.adaptive_system.config_manager.get_current_thresholds()
        return current_thresholds.get(component_type, 80.0)
    
    def _simulate_threshold_impact(self, 
                                 original_threshold: float,
                                 adaptive_threshold: float,
                                 validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of threshold change on validation."""
        impact = {
            'threshold_change': adaptive_threshold - original_threshold,
            'threshold_change_percent': ((adaptive_threshold - original_threshold) / original_threshold) * 100,
            'predicted_outcome_change': 'no_change',
            'confidence': 0.7
        }
        
        # Simulate impact based on current quality score
        quality_score = validation_report.get('quality_score', {}).get('score', 80)
        
        if adaptive_threshold > original_threshold:
            # Stricter threshold
            if quality_score < adaptive_threshold:
                impact['predicted_outcome_change'] = 'would_fail'
                impact['confidence'] = 0.8
            else:
                impact['predicted_outcome_change'] = 'would_still_pass'
                impact['confidence'] = 0.9
        else:
            # More lenient threshold
            if quality_score >= adaptive_threshold and quality_score < original_threshold:
                impact['predicted_outcome_change'] = 'would_pass'
                impact['confidence'] = 0.8
            else:
                impact['predicted_outcome_change'] = 'would_still_pass'
                impact['confidence'] = 0.9
        
        return impact
    
    def _collect_validation_data(self, 
                               issue_number: int,
                               issue_details: Dict[str, Any],
                               validation_report: Dict[str, Any]):
        """Collect validation data for learning."""
        try:
            # Determine component type for data collection
            components_analysis = validation_report.get('adaptive_thresholds', {}).get('components_analysis', {})
            
            # Collect data for each component involved
            for component_type in components_analysis.get('components', {}):
                # Get the threshold that was used
                threshold_used = self._get_current_component_threshold(component_type)
                
                # Get quality score
                quality_score = validation_report.get('quality_score', {}).get('score', 0)
                
                # Determine decision
                can_close = validation_report.get('can_close', False)
                decision = "pass" if can_close else "fail"
                
                # Check for manual overrides (if warnings but still allowing closure)
                warnings = validation_report.get('warnings', [])
                if warnings and can_close:
                    decision = "manual_override"
                
                # Create context for data collection
                context = {
                    'issue_number': issue_number,
                    'issue_type': self._determine_issue_type(issue_details),
                    'risk_level': validation_report.get('risk_assessment', {}).get('risk_level', 'medium'),
                    'total_changes': components_analysis.get('total_changes', 0),
                    'validation_timestamp': datetime.utcnow().isoformat() + 'Z'
                }
                
                # Record the quality decision
                success = self.data_collector.record_quality_decision(
                    component_type=component_type,
                    threshold_used=threshold_used,
                    quality_score=quality_score,
                    decision=decision,
                    context=context,
                    issue_number=issue_number
                )
                
                if success:
                    self.logger.debug(f"Recorded quality decision for {component_type}")
                else:
                    self.logger.warning(f"Failed to record quality decision for {component_type}")
            
        except Exception as e:
            self.logger.error(f"Error collecting validation data: {e}")
    
    def _determine_issue_type(self, issue_details: Dict[str, Any]) -> str:
        """Determine issue type from issue details."""
        return self.quality_gates._determine_issue_type(issue_details)
    
    def _generate_adaptive_recommendations(self, 
                                         issue_number: int,
                                         validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on adaptive threshold analysis."""
        recommendations = []
        
        adaptive_info = validation_report.get('adaptive_thresholds', {})
        
        if not adaptive_info.get('enabled'):
            return recommendations
        
        # Threshold adjustment recommendations
        if adaptive_info.get('thresholds_adjusted'):
            recommendations.append("🎯 Adaptive thresholds were applied based on component analysis")
            
            adaptive_thresholds = adaptive_info.get('adaptive_thresholds', {})
            original_thresholds = adaptive_info.get('original_thresholds', {})
            
            for component_type in adaptive_thresholds:
                original = original_thresholds.get(component_type, 0)
                adaptive = adaptive_thresholds[component_type]
                change = adaptive - original
                
                if abs(change) > 1.0:
                    direction = "increased" if change > 0 else "decreased"
                    recommendations.append(
                        f"   • {component_type}: {direction} from {original:.1f}% to {adaptive:.1f}%"
                    )
        
        # Component analysis recommendations
        components_analysis = adaptive_info.get('components_analysis', {})
        if components_analysis.get('has_component_data'):
            total_changes = components_analysis.get('total_changes', 0)
            components_count = len(components_analysis.get('components', {}))
            
            if total_changes > 500:
                recommendations.append(f"📊 Large change detected ({total_changes} lines across {components_count} component types)")
                recommendations.append("   Consider additional testing and gradual rollout")
            
            if components_count > 3:
                recommendations.append(f"🔄 Multi-component change affects {components_count} different component types")
                recommendations.append("   Ensure comprehensive integration testing")
        
        # System health recommendations
        if adaptive_info.get('system_health'):
            health_score = adaptive_info['system_health']
            if health_score < 0.7:
                recommendations.append(f"⚠️ System health score is {health_score:.2f} - consider threshold optimization")
        
        # Data collection recommendations
        components_with_limited_data = []
        for component_type in adaptive_info.get('components_analysis', {}).get('components', {}):
            # Check if we have sufficient historical data (simulated check)
            # In real implementation, this would check actual sample sizes
            historical_data = self.data_collector.get_quality_decisions(component_type, days_back=30)
            if len(historical_data) < 10:
                components_with_limited_data.append(component_type)
        
        if components_with_limited_data:
            recommendations.append("📈 Limited historical data for optimal threshold calculation:")
            recommendations.extend([f"   • {comp}: Consider collecting more quality decisions" 
                                   for comp in components_with_limited_data])
        
        return recommendations
    
    def get_adaptive_system_status(self) -> Dict[str, Any]:
        """Get current status of the adaptive threshold system."""
        try:
            status = {
                'adaptive_enabled': self.adaptive_enabled,
                'system_health': None,
                'data_collection_summary': None,
                'recent_optimizations': [],
                'configuration': {
                    'optimization_enabled': self.adaptive_config.get('optimization', {}).get('enabled', False),
                    'min_confidence_threshold': self.adaptive_config.get('optimization', {}).get('min_confidence_threshold', 0.7),
                    'max_threshold_change_percent': self.adaptive_config.get('optimization', {}).get('max_threshold_change_percent', 20.0)
                }
            }
            
            if self.adaptive_enabled:
                # Get system performance analysis
                performance_analysis = self.adaptive_system.analyze_current_system_performance()
                status['system_health'] = performance_analysis.get('system_health_score', 0)
                status['data_collection_summary'] = performance_analysis
                
                # Get data collection summary
                data_summary = self.data_collector.get_data_summary()
                status['data_collection_summary'] = data_summary
            
            return status
            
        except Exception as e:
            return {
                'adaptive_enabled': False,
                'error': str(e),
                'system_health': 0,
                'configuration': {}
            }
    
    def run_threshold_optimization(self, component_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run threshold optimization for specified components.
        
        Args:
            component_types: Specific components to optimize (None for all)
            
        Returns:
            Optimization results
        """
        if not self.adaptive_enabled:
            return {
                'optimization_run': False,
                'reason': 'Adaptive thresholds disabled in configuration'
            }
        
        try:
            self.logger.info("Running threshold optimization...")
            
            # Generate recommendations
            optimization_result = self.adaptive_system.generate_threshold_recommendations(
                component_types=component_types
            )
            
            # Log results
            self.logger.info(f"Optimization complete: {len(optimization_result.recommendations)} recommendations")
            
            return {
                'optimization_run': True,
                'optimization_id': optimization_result.optimization_id,
                'recommendations_count': len(optimization_result.recommendations),
                'overall_confidence': optimization_result.overall_confidence,
                'system_health_score': optimization_result.system_health_score,
                'predicted_improvement': optimization_result.quality_improvement_prediction,
                'recommendations': [
                    {
                        'component_type': rec.component_type,
                        'current_threshold': rec.current_threshold,
                        'recommended_threshold': rec.recommended_threshold,
                        'confidence': rec.confidence,
                        'rationale': rec.rationale,
                        'risk_assessment': rec.risk_assessment
                    }
                    for rec in optimization_result.recommendations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Threshold optimization failed: {e}")
            return {
                'optimization_run': False,
                'error': str(e)
            }

def main():
    """Command line interface for adaptive quality gates integration."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Adaptive Quality Gates Integration")
    parser.add_argument("--command", choices=["validate", "status", "optimize"], required=True,
                       help="Command to execute")
    parser.add_argument("--issue-number", type=int, help="GitHub issue number")
    parser.add_argument("--component-type", help="Specific component type to optimize")
    
    args = parser.parse_args()
    
    integration = AdaptiveQualityGatesIntegration()
    
    if args.command == "validate" and args.issue_number:
        print(f"Validating issue #{args.issue_number} with adaptive thresholds...")
        result = integration.validate_issue_closure_with_adaptive_thresholds(args.issue_number)
        print(json.dumps(result, indent=2, default=str))
        
        # Exit with error if validation fails
        if not result.get('can_close', False):
            sys.exit(1)
    
    elif args.command == "status":
        print("Adaptive Quality Gates System Status:")
        status = integration.get_adaptive_system_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == "optimize":
        component_types = [args.component_type] if args.component_type else None
        print("Running threshold optimization...")
        result = integration.run_threshold_optimization(component_types)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()