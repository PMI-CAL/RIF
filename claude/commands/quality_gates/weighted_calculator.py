#!/usr/bin/env python3
"""
Weighted Threshold Calculator for Multi-Component Changes
Issue #91: Context-Aware Quality Thresholds System

Specialized calculator for handling complex multi-component changes with intelligent
weight distribution, conflict resolution, and performance optimization.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

from .threshold_engine import AdaptiveThresholdEngine, ChangeMetrics, ChangeContext, ThresholdConfig

@dataclass
class ComponentWeight:
    """Weight configuration for a component in multi-component calculations."""
    component_type: str
    base_weight: float
    priority_adjustment: float
    size_adjustment: float
    risk_adjustment: float
    context_adjustment: float
    final_weight: float
    reasoning: str

@dataclass
class WeightedCalculationResult:
    """Result of weighted threshold calculation."""
    final_threshold: float
    fallback_applied: bool
    component_contributions: Dict[str, float]
    component_weights: Dict[str, ComponentWeight]
    calculation_method: str
    performance_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    reasoning: str
    timestamp: str

class WeightedCalculator:
    """
    Specialized calculator for multi-component threshold calculations.
    Handles complex scenarios with intelligent weight distribution.
    """
    
    def __init__(self, threshold_engine: Optional[AdaptiveThresholdEngine] = None):
        """
        Initialize weighted calculator.
        
        Args:
            threshold_engine: Optional pre-configured threshold engine
        """
        self.threshold_engine = threshold_engine or AdaptiveThresholdEngine()
        
        # Weight calculation strategies
        self.weight_strategies = {
            'balanced': self._balanced_weight_strategy,
            'size_based': self._size_based_weight_strategy,
            'priority_based': self._priority_based_weight_strategy,
            'risk_based': self._risk_based_weight_strategy
        }
        
        # Performance tracking
        self.calculation_metrics = []
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for weighted calculator."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - WeightedCalculator - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_weighted_threshold(self, 
                                   components: Dict[str, ChangeMetrics],
                                   context: Optional[ChangeContext] = None,
                                   strategy: str = 'balanced') -> WeightedCalculationResult:
        """
        Calculate weighted threshold for multi-component changes.
        
        Args:
            components: Dictionary mapping component types to change metrics
            context: Optional change context for adjustments
            strategy: Weight calculation strategy ('balanced', 'size_based', 'priority_based', 'risk_based')
            
        Returns:
            WeightedCalculationResult with detailed calculation information
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Calculating weighted threshold for {len(components)} components using {strategy} strategy")
            
            # Validate inputs
            if not components:
                return self._create_empty_result(strategy, start_time)
            
            # Get weight calculation strategy
            weight_strategy = self.weight_strategies.get(strategy, self._balanced_weight_strategy)
            
            # Calculate individual component thresholds
            component_thresholds = {}
            has_unknown_components = False
            for component_type, change_metrics in components.items():
                threshold_config = self.threshold_engine.calculate_component_threshold(
                    component_type, change_metrics, context
                )
                component_thresholds[component_type] = threshold_config
                
                # Check if this is an unknown component (using fallback reasoning)
                if "unknown component type" in threshold_config.reasoning.lower() or "fallback" in threshold_config.reasoning.lower():
                    has_unknown_components = True
            
            # Calculate component weights using selected strategy
            component_weights = weight_strategy(components, context)
            
            # Calculate weighted threshold
            weighted_result = self._calculate_final_weighted_threshold(
                component_thresholds, component_weights, context
            )
            
            # Validate and apply constraints
            validation_results = self._validate_threshold_result(
                weighted_result, component_thresholds, components, context
            )
            
            # Apply fallback if necessary
            final_threshold, fallback_applied = self._apply_fallback_constraints(
                weighted_result, validation_results, context, has_unknown_components
            )
            
            # Calculate component contributions
            component_contributions = self._calculate_component_contributions(
                component_thresholds, component_weights, final_threshold
            )
            
            # Generate reasoning
            reasoning = self._generate_calculation_reasoning(
                components, component_thresholds, component_weights, 
                final_threshold, fallback_applied, strategy
            )
            
            # Performance metrics
            calculation_time = (time.time() - start_time) * 1000
            performance_metrics = {
                'calculation_time_ms': round(calculation_time, 2),
                'performance_target_met': calculation_time < 200,
                'components_processed': len(components),
                'strategy_used': strategy
            }
            
            result = WeightedCalculationResult(
                final_threshold=round(final_threshold, 1),
                fallback_applied=fallback_applied,
                component_contributions=component_contributions,
                component_weights={ct: cw for ct, cw in component_weights.items()},
                calculation_method=strategy,
                performance_metrics=performance_metrics,
                validation_results=validation_results,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat()
            )
            
            # Track metrics
            self.calculation_metrics.append({
                'components_count': len(components),
                'strategy': strategy,
                'calculation_time_ms': calculation_time,
                'final_threshold': final_threshold,
                'fallback_applied': fallback_applied,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Weighted threshold calculated: {final_threshold:.1f}% (took {calculation_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in weighted threshold calculation: {e}")
            return self._create_error_result(str(e), strategy, start_time)
    
    def _balanced_weight_strategy(self, 
                                components: Dict[str, ChangeMetrics],
                                context: Optional[ChangeContext]) -> Dict[str, ComponentWeight]:
        """
        Balanced weight strategy considering multiple factors equally.
        
        Weight factors:
        - Component priority (25%)
        - Change size (25%)
        - Risk factor (25%)
        - Context importance (25%)
        """
        component_weights = {}
        
        # Calculate total metrics for normalization
        total_lines = sum(metrics.total_lines_changed for metrics in components.values())
        
        for component_type, change_metrics in components.items():
            component_config = self.threshold_engine.component_types.get(component_type, {})
            
            # Base weight from change size
            base_weight = change_metrics.total_lines_changed / total_lines if total_lines > 0 else 1.0 / len(components)
            
            # Priority adjustment (35% influence) - Make priority more important
            priority = component_config.get('priority', 5)
            priority_adjustment = (6 - priority) / 5 * 0.8 + 0.6  # Range: 0.6 to 1.4 - wider range
            
            # Size adjustment (20% influence) - Reduce size influence
            size_factor = min(2.0, change_metrics.total_lines_changed / 100)  # Normalize to reasonable range
            size_adjustment = 0.85 + (size_factor - 1.0) * 0.15 if size_factor > 1.0 else 0.85
            
            # Risk adjustment (30% influence) - Increase risk influence
            risk_factor = component_config.get('risk_factor', 1.0)
            risk_adjustment = 0.7 + (risk_factor - 1.0) * 0.6  # Scale risk factor influence more
            
            # Context adjustment (15% influence)
            context_adjustment = self._calculate_context_weight_adjustment(component_type, context)
            
            # Final weight calculation with enhanced priority/risk weighting
            final_weight = base_weight * priority_adjustment * size_adjustment * risk_adjustment * context_adjustment
            
            # Generate reasoning
            reasoning = f"Balanced: base={base_weight:.3f}, priority={priority_adjustment:.3f}, size={size_adjustment:.3f}, risk={risk_adjustment:.3f}, context={context_adjustment:.3f}"
            
            component_weights[component_type] = ComponentWeight(
                component_type=component_type,
                base_weight=base_weight,
                priority_adjustment=priority_adjustment,
                size_adjustment=size_adjustment,
                risk_adjustment=risk_adjustment,
                context_adjustment=context_adjustment,
                final_weight=final_weight,
                reasoning=reasoning
            )
        
        return component_weights
    
    def _size_based_weight_strategy(self, 
                                  components: Dict[str, ChangeMetrics],
                                  context: Optional[ChangeContext]) -> Dict[str, ComponentWeight]:
        """Weight strategy primarily based on change size (70% size, 30% other factors)."""
        component_weights = {}
        total_lines = sum(metrics.total_lines_changed for metrics in components.values())
        
        for component_type, change_metrics in components.items():
            component_config = self.threshold_engine.component_types.get(component_type, {})
            
            # Size-based weight (70% influence)
            base_weight = (change_metrics.total_lines_changed / total_lines) * 0.7 if total_lines > 0 else 0.7 / len(components)
            
            # Other factors (30% combined influence)
            priority = component_config.get('priority', 5)
            priority_adjustment = 1.0 + ((6 - priority) / 5) * 0.15  # 15% influence
            
            risk_factor = component_config.get('risk_factor', 1.0)
            risk_adjustment = 1.0 + (risk_factor - 1.0) * 0.15  # 15% influence
            
            context_adjustment = self._calculate_context_weight_adjustment(component_type, context)
            
            final_weight = base_weight * priority_adjustment * risk_adjustment * context_adjustment
            
            reasoning = f"Size-based: size={base_weight:.3f} (70%), priority={priority_adjustment:.3f} (15%), risk={risk_adjustment:.3f} (15%)"
            
            component_weights[component_type] = ComponentWeight(
                component_type=component_type,
                base_weight=base_weight,
                priority_adjustment=priority_adjustment,
                size_adjustment=1.0,  # Size is already in base_weight
                risk_adjustment=risk_adjustment,
                context_adjustment=context_adjustment,
                final_weight=final_weight,
                reasoning=reasoning
            )
        
        return component_weights
    
    def _priority_based_weight_strategy(self, 
                                      components: Dict[str, ChangeMetrics],
                                      context: Optional[ChangeContext]) -> Dict[str, ComponentWeight]:
        """Weight strategy primarily based on component priority (60% priority, 40% other factors)."""
        component_weights = {}
        total_lines = sum(metrics.total_lines_changed for metrics in components.values())
        
        for component_type, change_metrics in components.items():
            component_config = self.threshold_engine.component_types.get(component_type, {})
            
            # Priority-based weight (60% influence)
            priority = component_config.get('priority', 5)
            priority_weight = (6 - priority) / 5 * 0.6 + 0.4  # Range: 0.4 to 1.0
            
            # Size factor (20% influence)
            size_factor = (change_metrics.total_lines_changed / total_lines) * 0.2 if total_lines > 0 else 0.2 / len(components)
            
            # Risk factor (20% influence)
            risk_factor = component_config.get('risk_factor', 1.0)
            risk_adjustment = 1.0 + (risk_factor - 1.0) * 0.2
            
            context_adjustment = self._calculate_context_weight_adjustment(component_type, context)
            
            final_weight = (priority_weight + size_factor) * risk_adjustment * context_adjustment
            
            reasoning = f"Priority-based: priority={priority_weight:.3f} (60%), size={size_factor:.3f} (20%), risk={risk_adjustment:.3f} (20%)"
            
            component_weights[component_type] = ComponentWeight(
                component_type=component_type,
                base_weight=size_factor,
                priority_adjustment=priority_weight,
                size_adjustment=1.0,
                risk_adjustment=risk_adjustment,
                context_adjustment=context_adjustment,
                final_weight=final_weight,
                reasoning=reasoning
            )
        
        return component_weights
    
    def _risk_based_weight_strategy(self, 
                                  components: Dict[str, ChangeMetrics],
                                  context: Optional[ChangeContext]) -> Dict[str, ComponentWeight]:
        """Weight strategy primarily based on risk factors (50% risk, 50% other factors)."""
        component_weights = {}
        total_lines = sum(metrics.total_lines_changed for metrics in components.values())
        
        for component_type, change_metrics in components.items():
            component_config = self.threshold_engine.component_types.get(component_type, {})
            
            # Risk-based weight (50% influence)
            risk_factor = component_config.get('risk_factor', 1.0)
            risk_weight = risk_factor * 0.5
            
            # Size factor (25% influence)
            size_factor = (change_metrics.total_lines_changed / total_lines) * 0.25 if total_lines > 0 else 0.25 / len(components)
            
            # Priority factor (25% influence)
            priority = component_config.get('priority', 5)
            priority_adjustment = 1.0 + ((6 - priority) / 5) * 0.25
            
            context_adjustment = self._calculate_context_weight_adjustment(component_type, context)
            
            final_weight = (risk_weight + size_factor) * priority_adjustment * context_adjustment
            
            reasoning = f"Risk-based: risk={risk_weight:.3f} (50%), size={size_factor:.3f} (25%), priority={priority_adjustment:.3f} (25%)"
            
            component_weights[component_type] = ComponentWeight(
                component_type=component_type,
                base_weight=size_factor,
                priority_adjustment=priority_adjustment,
                size_adjustment=1.0,
                risk_adjustment=risk_weight,
                context_adjustment=context_adjustment,
                final_weight=final_weight,
                reasoning=reasoning
            )
        
        return component_weights
    
    def _calculate_context_weight_adjustment(self, 
                                           component_type: str, 
                                           context: Optional[ChangeContext]) -> float:
        """Calculate context-based weight adjustments for specific component types."""
        if not context:
            return 1.0
        
        adjustment = 1.0
        
        # Security-critical adjustments
        if context.is_security_critical:
            if 'security' in component_type.lower() or 'critical' in component_type.lower():
                adjustment *= 1.5  # Increase weight for security components in security context
        
        # Hotfix adjustments
        if context.is_hotfix:
            if 'critical' in component_type.lower():
                adjustment *= 1.3  # Critical components get more weight in hotfixes
        
        # Breaking change adjustments
        if context.is_breaking_change:
            if 'api' in component_type.lower():
                adjustment *= 1.4  # API components get more weight for breaking changes
        
        # Risk level adjustments
        if context.risk_level == 'high' or context.risk_level == 'critical':
            component_config = self.threshold_engine.component_types.get(component_type, {})
            priority = component_config.get('priority', 5)
            if priority <= 2:  # High priority components
                adjustment *= 1.2
        
        return adjustment
    
    def _calculate_final_weighted_threshold(self, 
                                          component_thresholds: Dict[str, ThresholdConfig],
                                          component_weights: Dict[str, ComponentWeight],
                                          context: Optional[ChangeContext]) -> float:
        """Calculate the final weighted threshold."""
        if not component_thresholds or not component_weights:
            return 80.0  # Fallback
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component_type, threshold_config in component_thresholds.items():
            if component_type in component_weights:
                weight = component_weights[component_type].final_weight
                weighted_sum += threshold_config.applied_threshold * weight
                total_weight += weight
        
        if total_weight == 0:
            return 80.0  # Fallback
        
        return weighted_sum / total_weight
    
    def _calculate_component_contributions(self, 
                                         component_thresholds: Dict[str, ThresholdConfig],
                                         component_weights: Dict[str, ComponentWeight],
                                         final_threshold: float) -> Dict[str, float]:
        """Calculate each component's contribution to the final threshold."""
        contributions = {}
        total_weight = sum(cw.final_weight for cw in component_weights.values())
        
        if total_weight == 0:
            return contributions
        
        # Calculate each component's relative contribution to the weighted sum
        for component_type, threshold_config in component_thresholds.items():
            if component_type in component_weights:
                weight = component_weights[component_type].final_weight
                # Contribution is the percentage this component's weighted value contributes
                # to the total weighted sum
                contribution = (weight / total_weight) * 100
                contributions[component_type] = round(contribution, 1)
        
        return contributions
    
    def _validate_threshold_result(self, 
                                 weighted_threshold: float,
                                 component_thresholds: Dict[str, ThresholdConfig],
                                 components: Dict[str, ChangeMetrics],
                                 context: Optional[ChangeContext]) -> Dict[str, Any]:
        """Validate the calculated threshold result."""
        validation = {
            'threshold_in_valid_range': 60 <= weighted_threshold <= 100,
            'meets_minimum_requirements': True,
            'component_validation': {},
            'warnings': [],
            'errors': []
        }
        
        # Validate individual components
        for component_type, threshold_config in component_thresholds.items():
            component_valid = 60 <= threshold_config.applied_threshold <= 100
            validation['component_validation'][component_type] = {
                'threshold_valid': component_valid,
                'threshold_value': threshold_config.applied_threshold
            }
            
            if not component_valid:
                validation['meets_minimum_requirements'] = False
                validation['errors'].append(f"Component {component_type} has invalid threshold: {threshold_config.applied_threshold}%")
        
        # Check for extreme thresholds
        if weighted_threshold < 70:
            validation['warnings'].append(f"Weighted threshold {weighted_threshold:.1f}% is unusually low")
        elif weighted_threshold > 95:
            validation['warnings'].append(f"Weighted threshold {weighted_threshold:.1f}% is unusually high")
        
        # Context-specific validations
        if context:
            if context.is_security_critical and weighted_threshold < 85:
                validation['warnings'].append("Security-critical change has threshold below 85%")
            
            if context.is_hotfix and weighted_threshold < 80:
                validation['warnings'].append("Hotfix has threshold below 80%")
        
        return validation
    
    def _apply_fallback_constraints(self, 
                                  weighted_threshold: float,
                                  validation_results: Dict[str, Any],
                                  context: Optional[ChangeContext],
                                  has_unknown_components: bool = False) -> Tuple[float, bool]:
        """Apply fallback constraints and return final threshold and whether fallback was applied."""
        fallback_threshold = self.threshold_engine.threshold_config.get('fallback_threshold', 80)
        
        # Apply fallback if validation failed
        if not validation_results['meets_minimum_requirements']:
            self.logger.warning("Validation failed, applying fallback threshold")
            return fallback_threshold, True
        
        # Apply fallback if we have unknown components
        if has_unknown_components:
            self.logger.warning("Unknown component types detected, applying fallback threshold")
            return fallback_threshold, True
        
        # Apply fallback if threshold is below minimum  
        if weighted_threshold < fallback_threshold:
            self.logger.info(f"Weighted threshold {weighted_threshold:.1f}% below fallback {fallback_threshold}%, applying fallback")
            return fallback_threshold, True
        
        # Apply fallback if threshold is unreasonably high (potential calculation error)
        if weighted_threshold > 98:
            self.logger.warning(f"Weighted threshold {weighted_threshold:.1f}% is unreasonably high, capping at 98%")
            return 98.0, True
        
        return weighted_threshold, False
    
    def _generate_calculation_reasoning(self, 
                                      components: Dict[str, ChangeMetrics],
                                      component_thresholds: Dict[str, ThresholdConfig],
                                      component_weights: Dict[str, ComponentWeight],
                                      final_threshold: float,
                                      fallback_applied: bool,
                                      strategy: str) -> str:
        """Generate human-readable reasoning for the calculation."""
        reasoning_parts = [
            f"Multi-component threshold calculation using {strategy} strategy"
        ]
        
        if fallback_applied:
            reasoning_parts.append("Fallback threshold applied due to validation constraints")
        else:
            # Component breakdown
            for component_type in components.keys():
                threshold_config = component_thresholds[component_type]
                weight = component_weights[component_type]
                reasoning_parts.append(
                    f"{component_type}: {threshold_config.applied_threshold:.1f}% (weight: {weight.final_weight:.3f})"
                )
        
        reasoning_parts.append(f"Final weighted threshold: {final_threshold:.1f}%")
        
        return " | ".join(reasoning_parts)
    
    def _create_empty_result(self, strategy: str, start_time: float) -> WeightedCalculationResult:
        """Create result for empty component list."""
        fallback_threshold = self.threshold_engine.threshold_config.get('fallback_threshold', 80)
        calculation_time = (time.time() - start_time) * 1000
        
        return WeightedCalculationResult(
            final_threshold=fallback_threshold,
            fallback_applied=True,
            component_contributions={},
            component_weights={},
            calculation_method=strategy,
            performance_metrics={
                'calculation_time_ms': round(calculation_time, 2),
                'performance_target_met': True,
                'components_processed': 0,
                'strategy_used': strategy
            },
            validation_results={'error': 'No components provided'},
            reasoning=f"No components provided, using fallback threshold: {fallback_threshold}%",
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, error: str, strategy: str, start_time: float) -> WeightedCalculationResult:
        """Create result for error cases."""
        fallback_threshold = self.threshold_engine.threshold_config.get('fallback_threshold', 80)
        calculation_time = (time.time() - start_time) * 1000
        
        return WeightedCalculationResult(
            final_threshold=fallback_threshold,
            fallback_applied=True,
            component_contributions={},
            component_weights={},
            calculation_method=f"{strategy}_error",
            performance_metrics={
                'calculation_time_ms': round(calculation_time, 2),
                'performance_target_met': True,
                'components_processed': 0,
                'strategy_used': strategy,
                'error': error
            },
            validation_results={'error': error},
            reasoning=f"Error during calculation: {error}. Using fallback threshold: {fallback_threshold}%",
            timestamp=datetime.now().isoformat()
        )
    
    def get_calculation_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics."""
        if not self.calculation_metrics:
            return {'error': 'No calculations performed yet'}
        
        total_calculations = len(self.calculation_metrics)
        avg_time = sum(m['calculation_time_ms'] for m in self.calculation_metrics) / total_calculations
        
        strategy_counts = defaultdict(int)
        for metric in self.calculation_metrics:
            strategy_counts[metric['strategy']] += 1
        
        return {
            'total_calculations': total_calculations,
            'average_calculation_time_ms': round(avg_time, 2),
            'performance_target_met': avg_time < 200,
            'strategy_usage': dict(strategy_counts),
            'fallback_rate': sum(1 for m in self.calculation_metrics if m['fallback_applied']) / total_calculations,
            'recent_calculations': self.calculation_metrics[-5:]  # Last 5 calculations
        }

def main():
    """Command line interface for weighted calculator."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python weighted_calculator.py <command> [args]")
        print("Commands:")
        print("  demo                  - Run demonstration calculation")
        print("  metrics              - Show calculation metrics")
        return
    
    command = sys.argv[1]
    calculator = WeightedCalculator()
    
    if command == "demo":
        # Demo calculation
        components = {
            'critical_algorithms': ChangeMetrics(
                lines_added=150, lines_deleted=50, lines_modified=75,
                files_changed=3, complexity_score=1.5
            ),
            'public_apis': ChangeMetrics(
                lines_added=80, lines_deleted=20, lines_modified=40,
                files_changed=2, complexity_score=1.2
            ),
            'ui_components': ChangeMetrics(
                lines_added=200, lines_deleted=100, lines_modified=50,
                files_changed=5, complexity_score=0.8
            )
        }
        
        context = ChangeContext(
            is_security_critical=True,
            risk_level='high',
            pr_size='medium'
        )
        
        result = calculator.calculate_weighted_threshold(components, context, 'balanced')
        
        print(json.dumps(asdict(result), indent=2, default=str))
    
    elif command == "metrics":
        metrics = calculator.get_calculation_metrics()
        print(json.dumps(metrics, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit(main())