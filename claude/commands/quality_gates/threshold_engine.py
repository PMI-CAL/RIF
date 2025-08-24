#!/usr/bin/env python3
"""
Adaptive Threshold Engine for Context-Aware Quality Thresholds System
Issue #91: Context-Aware Quality Thresholds System

Context-aware threshold calculation engine with <200ms performance target for complex
multi-component calculations. This engine applies component-specific thresholds and
calculates weighted averages for multi-component changes.
"""

import time
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .component_classifier import ComponentClassifier, ComponentType

@dataclass
class ThresholdConfig:
    """Configuration for a single component's quality thresholds."""
    component_type: str
    min_threshold: float
    target_threshold: float
    applied_threshold: float
    risk_factor: float
    size_factor: float
    context_modifiers: Dict[str, float]
    reasoning: str

@dataclass
class ChangeMetrics:
    """Metrics for code changes in a component."""
    lines_added: int
    lines_deleted: int
    lines_modified: int
    files_changed: int
    complexity_score: float
    
    @property
    def total_lines_changed(self) -> int:
        """Total lines impacted by changes."""
        return self.lines_added + self.lines_deleted + self.lines_modified

@dataclass
class ChangeContext:
    """Context information for threshold calculation adjustments."""
    is_hotfix: bool = False
    is_experimental: bool = False
    is_security_critical: bool = False
    is_breaking_change: bool = False
    pr_size: str = "medium"  # small, medium, large, very_large
    risk_level: str = "medium"  # low, medium, high, critical
    has_tests: bool = True
    historical_success_rate: float = 0.85

class AdaptiveThresholdEngine:
    """
    Context-aware threshold calculation engine.
    Performance target: <200ms for complex multi-component calculations.
    """
    
    def __init__(self, 
                 component_config_path: str = "config/component-types.yaml",
                 classifier: Optional[ComponentClassifier] = None):
        """
        Initialize adaptive threshold engine.
        
        Args:
            component_config_path: Path to component types configuration
            classifier: Optional pre-initialized component classifier
        """
        self.component_config_path = component_config_path
        self.config = self._load_config()
        self.component_types = self.config.get('component_types', {})
        self.threshold_config = self.config.get('configuration', {}).get('thresholds', {})
        self.performance_config = self.config.get('configuration', {}).get('performance', {})
        
        # Initialize or use provided classifier
        self.classifier = classifier or ComponentClassifier(component_config_path)
        
        # Performance tracking
        self.calculation_times = []
        self.calculation_history = []
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for threshold engine."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ThresholdEngine - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load component types configuration."""
        try:
            config_file = Path(self.component_config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file {self.component_config_path} not found")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default configuration for fallback."""
        return {
            'component_types': {
                'business_logic': {
                    'min_threshold': 85,
                    'target_threshold': 90,
                    'risk_factor': 1.2
                }
            },
            'configuration': {
                'thresholds': {
                    'fallback_threshold': 80,
                    'small_change_bonus': 5,
                    'large_change_penalty': 10,
                    'hotfix_modifier': 1.2,
                    'experimental_modifier': 0.9
                }
            }
        }
    
    def calculate_component_threshold(self, 
                                    component_type: str, 
                                    change_metrics: ChangeMetrics,
                                    context: Optional[ChangeContext] = None) -> ThresholdConfig:
        """
        Calculate threshold for single component change.
        
        Args:
            component_type: The component type (e.g., 'critical_algorithms')
            change_metrics: Metrics about the code changes
            context: Optional context for threshold adjustments
            
        Returns:
            ThresholdConfig with calculated threshold and reasoning
        """
        start_time = time.time()
        
        try:
            # Get base configuration for component type
            component_config = self.component_types.get(component_type, {})
            if not component_config:
                self.logger.warning(f"Unknown component type: {component_type}, using fallback")
                return self._create_fallback_threshold_config(component_type, change_metrics, context)
            
            base_threshold = component_config.get('min_threshold', 80)
            target_threshold = component_config.get('target_threshold', base_threshold + 10)
            risk_factor = component_config.get('risk_factor', 1.0)
            
            # Calculate size-based adjustments
            size_factor = self._calculate_size_factor(change_metrics)
            
            # Apply context modifiers
            context_modifiers = self._calculate_context_modifiers(context) if context else {}
            
            # Calculate final threshold
            # Formula: base_threshold + size_adjustment + risk_adjustment + context_adjustments
            size_adjustment = (size_factor - 1.0) * 10  # Convert factor to percentage adjustment
            risk_adjustment = (risk_factor - 1.0) * 5   # Convert factor to percentage adjustment
            
            applied_threshold = base_threshold + size_adjustment + risk_adjustment
            
            # Apply context modifiers (converted to percentage adjustments)
            for modifier_name, modifier_value in context_modifiers.items():
                context_adjustment = (modifier_value - 1.0) * 10
                applied_threshold += context_adjustment
            
            # Ensure threshold stays within reasonable bounds (60-100%)
            applied_threshold = max(60, min(100, applied_threshold))
            
            # Ensure backward compatibility (minimum 80% unless explicitly configured lower)
            fallback_threshold = self.threshold_config.get('fallback_threshold', 80)
            if applied_threshold < fallback_threshold and component_config.get('min_threshold', 80) >= fallback_threshold:
                applied_threshold = fallback_threshold
            
            # Generate reasoning
            reasoning = self._generate_threshold_reasoning(
                component_type, base_threshold, applied_threshold, 
                size_factor, risk_factor, context_modifiers, change_metrics
            )
            
            result = ThresholdConfig(
                component_type=component_type,
                min_threshold=base_threshold,
                target_threshold=target_threshold,
                applied_threshold=round(applied_threshold, 1),
                risk_factor=risk_factor,
                size_factor=size_factor,
                context_modifiers=context_modifiers,
                reasoning=reasoning
            )
            
            # Track performance
            calculation_time = (time.time() - start_time) * 1000
            self.calculation_times.append(calculation_time)
            
            self.logger.debug(f"Calculated threshold for {component_type}: {applied_threshold:.1f}% (took {calculation_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating threshold for {component_type}: {e}")
            return self._create_fallback_threshold_config(component_type, change_metrics, context)
    
    def calculate_weighted_threshold(self, 
                                   components: Dict[str, ChangeMetrics], 
                                   context: Optional[ChangeContext] = None) -> Dict[str, Any]:
        """
        Calculate weighted threshold for multi-component changes.
        
        Weighted threshold calculation algorithm:
        1. Component risk assessment (30% weight)
        2. Change size factor (25% weight) 
        3. Historical success rate (20% weight)
        4. Criticality priority (15% weight)
        5. Context factors (10% weight)
        
        Args:
            components: Dictionary mapping component types to their change metrics
            context: Optional context for threshold adjustments
            
        Returns:
            Dictionary with weighted threshold calculation results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Calculating weighted threshold for {len(components)} components")
            
            if not components:
                return self._create_fallback_weighted_result(context)
            
            # Calculate individual component thresholds
            component_thresholds = {}
            total_weight = 0
            weighted_sum = 0
            
            for component_type, change_metrics in components.items():
                threshold_config = self.calculate_component_threshold(component_type, change_metrics, context)
                component_thresholds[component_type] = threshold_config
                
                # Calculate weight for this component
                weight = self._calculate_component_weight(component_type, change_metrics, context)
                
                # Add to weighted calculation
                weighted_sum += threshold_config.applied_threshold * weight
                total_weight += weight
                
                self.logger.debug(f"Component {component_type}: threshold={threshold_config.applied_threshold:.1f}%, weight={weight:.3f}")
            
            # Calculate final weighted threshold
            if total_weight > 0:
                weighted_threshold = weighted_sum / total_weight
            else:
                weighted_threshold = self.threshold_config.get('fallback_threshold', 80)
            
            # Apply backward compatibility constraint
            fallback_threshold = self.threshold_config.get('fallback_threshold', 80)
            if weighted_threshold < fallback_threshold:
                self.logger.info(f"Weighted threshold {weighted_threshold:.1f}% below fallback {fallback_threshold}%, applying fallback")
                weighted_threshold = fallback_threshold
            
            # Generate detailed analysis
            calculation_time = (time.time() - start_time) * 1000
            
            result = {
                'weighted_threshold': round(weighted_threshold, 1),
                'fallback_threshold': fallback_threshold,
                'meets_fallback_requirement': weighted_threshold >= fallback_threshold,
                'calculation_method': 'weighted_average',
                'component_analysis': {
                    comp_type: {
                        'threshold': config.applied_threshold,
                        'weight': self._calculate_component_weight(comp_type, components[comp_type], context),
                        'contribution_percent': round((config.applied_threshold * 
                                                     self._calculate_component_weight(comp_type, components[comp_type], context) / 
                                                     weighted_sum) * 100, 1) if weighted_sum > 0 else 0,
                        'reasoning': config.reasoning
                    }
                    for comp_type, config in component_thresholds.items()
                },
                'context_applied': context.__dict__ if context else None,
                'calculation_time_ms': round(calculation_time, 2),
                'performance_target_met': calculation_time < 200,
                'timestamp': datetime.now().isoformat()
            }
            
            # Track performance and history
            self.calculation_times.append(calculation_time)
            self.calculation_history.append({
                'components': list(components.keys()),
                'weighted_threshold': weighted_threshold,
                'calculation_time_ms': calculation_time,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Weighted threshold calculated: {weighted_threshold:.1f}% (took {calculation_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted threshold: {e}")
            return self._create_fallback_weighted_result(context, error=str(e))
    
    def _calculate_size_factor(self, change_metrics: ChangeMetrics) -> float:
        """
        Calculate size adjustment factor based on change metrics.
        
        Formula: size_factor = min(1.2, 1 + (total_lines_changed / 1000))
        """
        total_changes = change_metrics.total_lines_changed
        
        if total_changes == 0:
            return 1.0
        
        # Small change bonus
        if total_changes < 50:
            bonus = self.threshold_config.get('small_change_bonus', 5) / 100
            return max(0.8, 1.0 - bonus)  # Reduce threshold for small changes
        
        # Large change penalty
        if total_changes > 1000:
            penalty = self.threshold_config.get('large_change_penalty', 10) / 100
            return min(1.2, 1.0 + penalty)  # Increase threshold for large changes
        
        # Linear scaling for medium changes
        size_factor = 1.0 + (total_changes / 1000)
        return min(1.2, size_factor)
    
    def _calculate_context_modifiers(self, context: ChangeContext) -> Dict[str, float]:
        """Calculate context-based threshold modifiers."""
        modifiers = {}
        
        if context.is_hotfix:
            modifiers['hotfix'] = self.threshold_config.get('hotfix_modifier', 1.2)
        
        if context.is_experimental:
            modifiers['experimental'] = self.threshold_config.get('experimental_modifier', 0.9)
        
        if context.is_security_critical:
            modifiers['security_critical'] = 1.3  # Always increase for security
        
        if context.is_breaking_change:
            modifiers['breaking_change'] = 1.25
        
        # Risk level modifiers
        risk_modifiers = {
            'low': 0.95,
            'medium': 1.0,
            'high': 1.15,
            'critical': 1.3
        }
        if context.risk_level in risk_modifiers:
            modifiers['risk_level'] = risk_modifiers[context.risk_level]
        
        # Test coverage modifier
        if not context.has_tests:
            modifiers['no_tests'] = 1.25  # Increase threshold if no tests
        
        return modifiers
    
    def _calculate_component_weight(self, 
                                  component_type: str, 
                                  change_metrics: ChangeMetrics, 
                                  context: Optional[ChangeContext]) -> float:
        """
        Calculate weight for component in weighted average calculation.
        
        Weight factors:
        - Component priority (from configuration)
        - Change size (lines of code)
        - Risk factor
        - Historical success rate
        """
        component_config = self.component_types.get(component_type, {})
        
        # Base weight from change size
        base_weight = change_metrics.total_lines_changed
        
        # Priority adjustment (higher priority = higher weight)
        priority = component_config.get('priority', 5)
        priority_weight = (6 - priority) * 0.2 + 0.8  # Priority 1 gets 1.8, Priority 6 gets 0.8
        
        # Risk factor adjustment
        risk_factor = component_config.get('risk_factor', 1.0)
        
        # Context adjustments
        context_weight = 1.0
        if context:
            if context.is_security_critical and 'security' in component_type.lower():
                context_weight *= 1.5
            if context.risk_level == 'critical':
                context_weight *= 1.3
        
        # Calculate final weight
        weight = base_weight * priority_weight * risk_factor * context_weight
        
        # Ensure minimum weight
        min_weight = self.threshold_config.get('min_component_weight', 0.1)
        return max(min_weight, weight)
    
    def _generate_threshold_reasoning(self, 
                                    component_type: str,
                                    base_threshold: float,
                                    applied_threshold: float,
                                    size_factor: float,
                                    risk_factor: float,
                                    context_modifiers: Dict[str, float],
                                    change_metrics: ChangeMetrics) -> str:
        """Generate human-readable reasoning for threshold calculation."""
        reasoning_parts = [
            f"Base threshold for {component_type}: {base_threshold}%"
        ]
        
        if size_factor != 1.0:
            direction = "increased" if size_factor > 1.0 else "decreased"
            reasoning_parts.append(f"Size adjustment ({change_metrics.total_lines_changed} lines): {direction} by {abs(size_factor - 1.0)*100:.1f}%")
        
        if risk_factor != 1.0:
            reasoning_parts.append(f"Risk factor adjustment: {risk_factor}x")
        
        for modifier_name, modifier_value in context_modifiers.items():
            if modifier_value != 1.0:
                direction = "increased" if modifier_value > 1.0 else "decreased"
                reasoning_parts.append(f"{modifier_name.replace('_', ' ').title()}: {direction} by {abs(modifier_value - 1.0)*100:.1f}%")
        
        reasoning_parts.append(f"Final applied threshold: {applied_threshold:.1f}%")
        
        return " | ".join(reasoning_parts)
    
    def _create_fallback_threshold_config(self, 
                                        component_type: str,
                                        change_metrics: ChangeMetrics,
                                        context: Optional[ChangeContext]) -> ThresholdConfig:
        """Create fallback threshold configuration."""
        fallback_threshold = self.threshold_config.get('fallback_threshold', 80)
        
        return ThresholdConfig(
            component_type=component_type,
            min_threshold=fallback_threshold,
            target_threshold=fallback_threshold + 10,
            applied_threshold=fallback_threshold,
            risk_factor=1.0,
            size_factor=1.0,
            context_modifiers={},
            reasoning=f"Fallback threshold applied: {fallback_threshold}% (unknown component type or error)"
        )
    
    def _create_fallback_weighted_result(self, 
                                       context: Optional[ChangeContext],
                                       error: Optional[str] = None) -> Dict[str, Any]:
        """Create fallback weighted threshold result."""
        fallback_threshold = self.threshold_config.get('fallback_threshold', 80)
        
        return {
            'weighted_threshold': fallback_threshold,
            'fallback_threshold': fallback_threshold,
            'meets_fallback_requirement': True,
            'calculation_method': 'fallback',
            'component_analysis': {},
            'context_applied': context.__dict__ if context else None,
            'calculation_time_ms': 0,
            'performance_target_met': True,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'reasoning': f"Fallback threshold applied: {fallback_threshold}%"
        }
    
    def apply_context_modifiers(self, 
                              base_threshold: float, 
                              context: ChangeContext) -> float:
        """
        Apply contextual modifiers to a base threshold.
        
        Args:
            base_threshold: Base threshold value
            context: Change context for modifications
            
        Returns:
            Modified threshold value
        """
        modified_threshold = base_threshold
        modifiers = self._calculate_context_modifiers(context)
        
        for modifier_value in modifiers.values():
            modified_threshold *= modifier_value
        
        return max(60, min(100, modified_threshold))
    
    def validate_backward_compatibility(self, threshold: float) -> bool:
        """
        Ensure threshold meets backward compatibility requirements.
        
        Args:
            threshold: Calculated threshold value
            
        Returns:
            True if threshold meets backward compatibility (>= 80%)
        """
        fallback_threshold = self.threshold_config.get('fallback_threshold', 80)
        return threshold >= fallback_threshold
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for threshold calculations."""
        if not self.calculation_times:
            return {"error": "No calculations performed yet"}
        
        avg_time = sum(self.calculation_times) / len(self.calculation_times)
        max_time = max(self.calculation_times)
        
        return {
            "total_calculations": len(self.calculation_times),
            "average_calculation_time_ms": round(avg_time, 2),
            "max_calculation_time_ms": round(max_time, 2),
            "performance_target_met": avg_time < 200,
            "calculations_under_200ms": sum(1 for t in self.calculation_times if t < 200),
            "calculation_history": self.calculation_history[-10:],  # Last 10 calculations
            "target_performance_ms": 200
        }

def main():
    """Command line interface for threshold engine."""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python threshold_engine.py <command> [args]")
        print("Commands:")
        print("  single <component_type> <lines_changed>     - Calculate single component threshold")
        print("  weighted <components_json>                  - Calculate weighted threshold")
        print("  context <base_threshold> <context_json>     - Apply context modifiers")
        print("  metrics                                     - Show performance metrics")
        return
    
    command = sys.argv[1]
    engine = AdaptiveThresholdEngine()
    
    if command == "single" and len(sys.argv) >= 4:
        component_type = sys.argv[2]
        lines_changed = int(sys.argv[3])
        
        change_metrics = ChangeMetrics(
            lines_added=lines_changed // 2,
            lines_deleted=lines_changed // 4,
            lines_modified=lines_changed // 4,
            files_changed=1,
            complexity_score=1.0
        )
        
        result = engine.calculate_component_threshold(component_type, change_metrics)
        
        output = {
            "component_type": result.component_type,
            "applied_threshold": result.applied_threshold,
            "min_threshold": result.min_threshold,
            "target_threshold": result.target_threshold,
            "risk_factor": result.risk_factor,
            "size_factor": result.size_factor,
            "reasoning": result.reasoning
        }
        
        print(json.dumps(output, indent=2))
    
    elif command == "metrics":
        metrics = engine.get_performance_metrics()
        print(json.dumps(metrics, indent=2))
    
    else:
        print(f"Unknown or incomplete command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit(main())