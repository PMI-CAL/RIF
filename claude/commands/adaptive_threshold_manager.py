#!/usr/bin/env python3
"""
Adaptive Threshold Manager - Issue #93 Phase 2
Dynamic threshold application based on component types and historical performance.

This module provides:
- Context-aware threshold calculation
- Multi-component handling strategies  
- Adaptive threshold adjustment based on historical data
- Integration with quality decision engine
"""

import os
import json
import yaml
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from statistics import mean, stdev

from component_type_classifier import ComponentTypeClassifier, BatchClassificationResult

@dataclass
class ThresholdResult:
    """Result of threshold calculation."""
    thresholds: Dict[str, Any] = field(default_factory=dict)
    component_type: str = "business_logic"
    confidence: float = 1.0
    strategy_used: str = "default"
    calculation_time_ms: float = 0.0
    historical_adjustments: Dict[str, float] = field(default_factory=dict)

@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold behavior."""
    enabled: bool = True
    learning_rate: float = 0.1
    historical_window_days: int = 90
    minimum_samples: int = 10
    max_adjustment_percent: float = 20.0
    adjustment_frequency_days: int = 30
    # Additional fields that might be in config but not used directly
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    component_learning: Dict[str, Any] = field(default_factory=dict)

class AdaptiveThresholdManager:
    """
    Manages context-aware and adaptive quality thresholds.
    
    Features:
    - Component type-specific threshold calculation
    - Multi-component change handling strategies
    - Historical performance-based threshold adjustment
    - Performance optimization with caching
    - Integration with existing quality systems
    """
    
    def __init__(self, config_path: str = "config/context-thresholds.yaml"):
        """Initialize adaptive threshold manager."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Component configurations
        self.component_thresholds = self.config.get('component_thresholds', {})
        self.quality_definitions = self.config.get('quality_definitions', {})
        self.multi_component_strategies = self.config.get('multi_component_strategies', {})
        
        # Adaptive configuration (filter out unknown fields)
        adaptive_config_data = self.config.get('adaptive_thresholds', {})
        known_fields = {f.name for f in AdaptiveThresholdConfig.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in adaptive_config_data.items() if k in known_fields}
        self.adaptive_config = AdaptiveThresholdConfig(**filtered_config)
        
        # Initialize classifier
        self.classifier = ComponentTypeClassifier(config_path)
        
        # Historical data for adaptive adjustments
        self.historical_data = self._load_historical_data()
        
        # Caching for performance
        self.threshold_cache = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Setup logging
        self.setup_logging()
        
        # Performance metrics
        self.performance_metrics = {
            'calculations': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0,
            'adaptive_adjustments': 0
        }
    
    def setup_logging(self):
        """Setup logging for adaptive threshold management."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - AdaptiveThresholdManager - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load adaptive threshold configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Config file {self.config_path} not found")
                return self._get_default_threshold_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_threshold_config()
    
    def _get_default_threshold_config(self) -> Dict[str, Any]:
        """Default threshold configuration for fallback."""
        return {
            'component_thresholds': {
                'business_logic': {
                    'quality_thresholds': {
                        'test_coverage': 80.0,
                        'security_validation': 'standard',
                        'performance_impact': 5.0,
                        'code_quality': 'standard'
                    },
                    'weight': 1.0
                }
            },
            'quality_definitions': {
                'security_levels': {
                    'standard': {'critical_vulnerabilities': 0, 'high_vulnerabilities': 3}
                },
                'code_quality_levels': {
                    'standard': {'max_critical_issues': 0, 'maintainability_threshold': 60}
                }
            },
            'multi_component_strategies': {
                'weighted_average': {'description': 'Weighted average of component thresholds'}
            },
            'adaptive_thresholds': {
                'enabled': True,
                'learning_rate': 0.1,
                'historical_window_days': 90
            }
        }
    
    def calculate_thresholds(
        self,
        file_paths: List[str],
        strategy: str = "weighted_average",
        use_adaptive: bool = True
    ) -> ThresholdResult:
        """
        Calculate context-aware thresholds for the given files.
        
        Args:
            file_paths: List of files being modified
            strategy: Multi-component handling strategy
            use_adaptive: Whether to apply adaptive adjustments
            
        Returns:
            Threshold result with calculated values and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(file_paths, strategy, use_adaptive)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.performance_metrics['cache_hits'] += 1
            return cached_result
        
        try:
            # Step 1: Classify components
            classification = self.classifier.classify_batch(file_paths, "weighted_voting")
            
            # Step 2: Get base thresholds for the primary component type
            base_thresholds = self._get_base_thresholds(classification.primary_type)
            
            # Step 3: Apply multi-component strategy if needed
            if len(classification.type_distribution) > 1:
                adjusted_thresholds = self._apply_multi_component_strategy(
                    classification, strategy, base_thresholds
                )
            else:
                adjusted_thresholds = base_thresholds.copy()
            
            # Step 4: Apply adaptive adjustments if enabled
            historical_adjustments = {}
            if use_adaptive and self.adaptive_config.enabled:
                adjusted_thresholds, historical_adjustments = self._apply_adaptive_adjustments(
                    adjusted_thresholds, classification.primary_type, file_paths
                )
            
            # Step 5: Validate and sanitize thresholds
            final_thresholds = self._validate_thresholds(adjusted_thresholds)
            
            # Create result
            calculation_time = (time.time() - start_time) * 1000
            result = ThresholdResult(
                thresholds=final_thresholds,
                component_type=classification.primary_type,
                confidence=classification.confidence,
                strategy_used=strategy,
                calculation_time_ms=calculation_time,
                historical_adjustments=historical_adjustments
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update metrics
            self.performance_metrics['calculations'] += 1
            self.performance_metrics['total_time_ms'] += calculation_time
            if historical_adjustments:
                self.performance_metrics['adaptive_adjustments'] += 1
            
            self.logger.debug(f"Calculated thresholds for {len(file_paths)} files: {classification.primary_type} (confidence: {classification.confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating thresholds: {e}")
            
            # Return safe fallback
            calculation_time = (time.time() - start_time) * 1000
            return ThresholdResult(
                thresholds=self._get_fallback_thresholds(),
                component_type="business_logic",
                confidence=0.5,
                strategy_used="fallback",
                calculation_time_ms=calculation_time
            )
    
    def _get_base_thresholds(self, component_type: str) -> Dict[str, Any]:
        """Get base thresholds for a component type."""
        component_config = self.component_thresholds.get(component_type, {})
        base_thresholds = component_config.get('quality_thresholds', {})
        
        # Resolve string threshold references to actual values
        resolved_thresholds = {}
        for threshold_name, threshold_value in base_thresholds.items():
            if isinstance(threshold_value, str):
                resolved_value = self._resolve_threshold_reference(threshold_name, threshold_value)
                resolved_thresholds[threshold_name] = resolved_value
            else:
                resolved_thresholds[threshold_name] = threshold_value
        
        return resolved_thresholds
    
    def _resolve_threshold_reference(self, threshold_name: str, reference: str) -> Any:
        """Resolve string threshold references to actual values."""
        if threshold_name == 'security_validation':
            security_levels = self.quality_definitions.get('security_levels', {})
            if reference in security_levels:
                return security_levels[reference]
            
        elif threshold_name == 'code_quality':
            quality_levels = self.quality_definitions.get('code_quality_levels', {})
            if reference in quality_levels:
                return quality_levels[reference]
        
        # If resolution fails, return the original string
        self.logger.warning(f"Could not resolve threshold reference: {threshold_name}={reference}")
        return reference
    
    def _apply_multi_component_strategy(
        self,
        classification: BatchClassificationResult,
        strategy: str,
        base_thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply multi-component handling strategy."""
        if strategy not in self.multi_component_strategies:
            self.logger.warning(f"Unknown strategy {strategy}, using weighted_average")
            strategy = "weighted_average"
        
        if strategy == "weighted_average":
            return self._weighted_average_thresholds(classification)
        
        elif strategy == "most_conservative":
            return self._most_conservative_thresholds(classification)
        
        elif strategy == "primary_component":
            return base_thresholds  # Already calculated for primary component
        
        elif strategy == "size_based":
            return self._size_based_thresholds(classification, base_thresholds)
        
        elif strategy == "risk_based":
            return self._risk_based_thresholds(classification, base_thresholds)
        
        else:
            return base_thresholds
    
    def _weighted_average_thresholds(self, classification: BatchClassificationResult) -> Dict[str, Any]:
        """Calculate weighted average thresholds across component types."""
        aggregated_thresholds = {}
        
        # Get all threshold keys
        all_keys = set()
        for component_type in classification.type_distribution.keys():
            component_thresholds = self._get_base_thresholds(component_type)
            all_keys.update(component_thresholds.keys())
        
        # Calculate weighted average for each threshold
        for key in all_keys:
            weighted_sum = 0.0
            total_weight = 0.0
            string_values = []
            
            for component_type, proportion in classification.type_distribution.items():
                component_thresholds = self._get_base_thresholds(component_type)
                
                if key in component_thresholds:
                    value = component_thresholds[key]
                    
                    if isinstance(value, (int, float)):
                        weighted_sum += value * proportion
                        total_weight += proportion
                    else:
                        string_values.append((value, proportion))
            
            if total_weight > 0:
                aggregated_thresholds[key] = weighted_sum / total_weight
            elif string_values:
                # For non-numeric values, choose the one with highest weight
                best_value = max(string_values, key=lambda x: x[1])[0]
                aggregated_thresholds[key] = best_value
        
        return aggregated_thresholds
    
    def _most_conservative_thresholds(self, classification: BatchClassificationResult) -> Dict[str, Any]:
        """Use the most conservative (strict) thresholds among component types."""
        conservative_thresholds = {}
        
        # Get all threshold keys
        all_keys = set()
        for component_type in classification.type_distribution.keys():
            component_thresholds = self._get_base_thresholds(component_type)
            all_keys.update(component_thresholds.keys())
        
        # Find most conservative value for each threshold
        for key in all_keys:
            candidate_values = []
            
            for component_type in classification.type_distribution.keys():
                component_thresholds = self._get_base_thresholds(component_type)
                if key in component_thresholds:
                    candidate_values.append(component_thresholds[key])
            
            if candidate_values:
                if key == 'test_coverage':
                    # Higher coverage is more conservative
                    conservative_thresholds[key] = max(v for v in candidate_values if isinstance(v, (int, float)))
                elif key == 'performance_impact':
                    # Lower regression tolerance is more conservative
                    conservative_thresholds[key] = min(v for v in candidate_values if isinstance(v, (int, float)))
                elif isinstance(candidate_values[0], dict):
                    # For complex thresholds, use the first one found (could be enhanced)
                    conservative_thresholds[key] = candidate_values[0]
                else:
                    # For string values, use priority order
                    priority_order = ['zero_tolerance', 'strict', 'high', 'standard', 'relaxed']
                    for priority_value in priority_order:
                        if priority_value in candidate_values:
                            conservative_thresholds[key] = priority_value
                            break
                    else:
                        conservative_thresholds[key] = candidate_values[0]
        
        return conservative_thresholds
    
    def _size_based_thresholds(self, classification: BatchClassificationResult, base_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust thresholds based on change size (larger changes get stricter thresholds)."""
        # This would ideally get change size information
        # For now, apply a conservative adjustment
        adjusted_thresholds = base_thresholds.copy()
        
        if len(classification.individual_results) > 10:  # Large change
            # Make thresholds more conservative
            for key, value in adjusted_thresholds.items():
                if key == 'test_coverage' and isinstance(value, (int, float)):
                    adjusted_thresholds[key] = min(100, value + 5)  # Increase coverage requirement
                elif key == 'performance_impact' and isinstance(value, (int, float)):
                    adjusted_thresholds[key] = max(1, value - 1)    # Stricter performance requirement
        
        return adjusted_thresholds
    
    def _risk_based_thresholds(self, classification: BatchClassificationResult, base_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust thresholds based on risk assessment."""
        adjusted_thresholds = base_thresholds.copy()
        
        # Calculate risk level based on component types
        high_risk_types = ['critical_algorithms', 'public_apis']
        risk_score = sum(
            classification.type_distribution.get(component_type, 0)
            for component_type in high_risk_types
        )
        
        if risk_score > 0.3:  # High risk scenario
            # Make thresholds more conservative
            for key, value in adjusted_thresholds.items():
                if key == 'test_coverage' and isinstance(value, (int, float)):
                    adjusted_thresholds[key] = min(100, value + 10)  # Much stricter coverage
                elif key == 'security_validation' and value == 'standard':
                    adjusted_thresholds[key] = 'high'  # Upgrade security requirements
        
        return adjusted_thresholds
    
    def _apply_adaptive_adjustments(
        self,
        thresholds: Dict[str, Any],
        component_type: str,
        file_paths: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Apply adaptive adjustments based on historical performance."""
        if not self.adaptive_config.enabled:
            return thresholds, {}
        
        adjusted_thresholds = thresholds.copy()
        adjustments = {}
        
        # Get historical performance for this component type
        historical_perf = self.historical_data.get(component_type, {})
        
        if not historical_perf or len(historical_perf.get('samples', [])) < self.adaptive_config.minimum_samples:
            return thresholds, {}
        
        try:
            # Analyze historical false positive/negative rates
            samples = historical_perf['samples']
            recent_samples = [
                s for s in samples 
                if datetime.fromisoformat(s['timestamp']) > 
                   datetime.now() - timedelta(days=self.adaptive_config.historical_window_days)
            ]
            
            if len(recent_samples) < self.adaptive_config.minimum_samples:
                return thresholds, {}
            
            # Calculate adjustment factors
            false_positive_rate = mean(s.get('false_positive', 0) for s in recent_samples)
            false_negative_rate = mean(s.get('false_negative', 0) for s in recent_samples)
            
            # Adjust thresholds based on performance
            for threshold_name, threshold_value in adjusted_thresholds.items():
                if isinstance(threshold_value, (int, float)):
                    adjustment_factor = self._calculate_adjustment_factor(
                        threshold_name, false_positive_rate, false_negative_rate
                    )
                    
                    if abs(adjustment_factor) > 0.01:  # Only apply significant adjustments
                        original_value = threshold_value
                        adjusted_value = original_value * (1 + adjustment_factor)
                        
                        # Apply bounds
                        max_change = original_value * (self.adaptive_config.max_adjustment_percent / 100)
                        adjusted_value = max(
                            original_value - max_change,
                            min(original_value + max_change, adjusted_value)
                        )
                        
                        adjusted_thresholds[threshold_name] = adjusted_value
                        adjustments[threshold_name] = adjustment_factor
                        
                        self.logger.info(f"Adaptive adjustment for {threshold_name}: {original_value:.2f} -> {adjusted_value:.2f} (factor: {adjustment_factor:.3f})")
        
        except Exception as e:
            self.logger.error(f"Error applying adaptive adjustments: {e}")
        
        return adjusted_thresholds, adjustments
    
    def _calculate_adjustment_factor(self, threshold_name: str, false_positive_rate: float, false_negative_rate: float) -> float:
        """Calculate adjustment factor based on false positive/negative rates."""
        target_fp_rate = 0.05  # Target 5% false positive rate
        target_fn_rate = 0.02  # Target 2% false negative rate
        
        if threshold_name == 'test_coverage':
            # High false positive rate means threshold too strict, lower it
            # High false negative rate means threshold too lenient, raise it
            fp_adjustment = -(false_positive_rate - target_fp_rate) * self.adaptive_config.learning_rate
            fn_adjustment = (false_negative_rate - target_fn_rate) * self.adaptive_config.learning_rate * 2
            return fp_adjustment + fn_adjustment
        
        elif threshold_name == 'performance_impact':
            # Similar logic but inverse (lower values are stricter)
            fp_adjustment = (false_positive_rate - target_fp_rate) * self.adaptive_config.learning_rate
            fn_adjustment = -(false_negative_rate - target_fn_rate) * self.adaptive_config.learning_rate * 2
            return fp_adjustment + fn_adjustment
        
        # Default adjustment
        return -(false_positive_rate - target_fp_rate) * self.adaptive_config.learning_rate * 0.5
    
    def _validate_thresholds(self, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize threshold values."""
        validated_thresholds = {}
        
        for key, value in thresholds.items():
            if key == 'test_coverage':
                # Ensure coverage is between 0-100
                if isinstance(value, (int, float)):
                    validated_thresholds[key] = max(0, min(100, float(value)))
                else:
                    validated_thresholds[key] = 80.0  # Fallback
                    
            elif key == 'performance_impact':
                # Ensure positive performance impact threshold
                if isinstance(value, (int, float)):
                    validated_thresholds[key] = max(0.1, float(value))
                else:
                    validated_thresholds[key] = 5.0  # Fallback
                    
            else:
                validated_thresholds[key] = value
        
        return validated_thresholds
    
    def _get_fallback_thresholds(self) -> Dict[str, Any]:
        """Get safe fallback thresholds."""
        return {
            'test_coverage': 80.0,
            'security_validation': {'critical_vulnerabilities': 0, 'high_vulnerabilities': 3},
            'performance_impact': 5.0,
            'code_quality': {'max_critical_issues': 0, 'maintainability_threshold': 60}
        }
    
    def _load_historical_data(self) -> Dict[str, Dict[str, Any]]:
        """Load historical threshold performance data."""
        try:
            history_file = Path("knowledge/metrics/threshold-history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.debug(f"Could not load historical data: {e}")
        
        return {}
    
    def record_threshold_performance(
        self,
        component_type: str,
        thresholds_used: Dict[str, Any],
        actual_outcome: Dict[str, Any]
    ):
        """Record threshold performance for adaptive learning."""
        try:
            # Calculate performance metrics
            performance_sample = {
                'timestamp': datetime.now().isoformat(),
                'thresholds': thresholds_used,
                'outcome': actual_outcome,
                'false_positive': actual_outcome.get('false_positive', 0),
                'false_negative': actual_outcome.get('false_negative', 0)
            }
            
            # Add to historical data
            if component_type not in self.historical_data:
                self.historical_data[component_type] = {'samples': []}
            
            self.historical_data[component_type]['samples'].append(performance_sample)
            
            # Limit sample size to prevent unbounded growth
            max_samples = 1000
            if len(self.historical_data[component_type]['samples']) > max_samples:
                self.historical_data[component_type]['samples'] = \
                    self.historical_data[component_type]['samples'][-max_samples:]
            
            # Save to file periodically
            if len(self.historical_data[component_type]['samples']) % 10 == 0:
                self._save_historical_data()
        
        except Exception as e:
            self.logger.error(f"Error recording threshold performance: {e}")
    
    def _save_historical_data(self):
        """Save historical data to file."""
        try:
            history_file = Path("knowledge/metrics/threshold-history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")
    
    def _generate_cache_key(self, file_paths: List[str], strategy: str, use_adaptive: bool) -> str:
        """Generate cache key for threshold calculation."""
        import hashlib
        key_data = f"{sorted(file_paths)}_{strategy}_{use_adaptive}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ThresholdResult]:
        """Get cached threshold result if valid."""
        if cache_key in self.threshold_cache:
            cached_result, timestamp = self.threshold_cache[cache_key]
            
            if time.time() - timestamp < self.cache_ttl_seconds:
                return cached_result
            else:
                del self.threshold_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: ThresholdResult):
        """Cache threshold calculation result."""
        self.threshold_cache[cache_key] = (result, time.time())
        
        # Limit cache size
        if len(self.threshold_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.threshold_cache.keys(),
                key=lambda k: self.threshold_cache[k][1]
            )[:20]
            for key in oldest_keys:
                del self.threshold_cache[key]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get threshold calculation performance metrics."""
        metrics = dict(self.performance_metrics)
        
        if self.performance_metrics['calculations'] > 0:
            metrics['average_time_ms'] = (
                self.performance_metrics['total_time_ms'] / 
                self.performance_metrics['calculations']
            )
            metrics['cache_hit_rate'] = (
                self.performance_metrics['cache_hits'] / 
                self.performance_metrics['calculations']
            )
            metrics['adaptive_usage_rate'] = (
                self.performance_metrics['adaptive_adjustments'] /
                self.performance_metrics['calculations']
            )
        
        return metrics

def main():
    """Command line interface for adaptive threshold manager."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Threshold Manager')
    parser.add_argument('--config', default='config/context-thresholds.yaml', help='Configuration file path')
    parser.add_argument('--files', nargs='+', required=True, help='File paths to calculate thresholds for')
    parser.add_argument('--strategy', choices=['weighted_average', 'most_conservative', 'primary_component', 'size_based', 'risk_based'],
                       default='weighted_average', help='Multi-component strategy')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive adjustments')
    parser.add_argument('--performance', action='store_true', help='Show performance metrics')
    parser.add_argument('--output', choices=['thresholds', 'summary', 'full'], default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Create manager
    manager = AdaptiveThresholdManager(config_path=args.config)
    
    if args.performance:
        metrics = manager.get_performance_metrics()
        print(json.dumps(metrics, indent=2))
        return 0
    
    # Calculate thresholds
    result = manager.calculate_thresholds(
        file_paths=args.files,
        strategy=args.strategy,
        use_adaptive=args.adaptive
    )
    
    # Output based on format
    if args.output == 'thresholds':
        print(json.dumps(result.thresholds, indent=2))
    elif args.output == 'summary':
        print(json.dumps({
            'component_type': result.component_type,
            'confidence': result.confidence,
            'strategy_used': result.strategy_used,
            'calculation_time_ms': result.calculation_time_ms
        }, indent=2))
    else:
        print(json.dumps({
            'thresholds': result.thresholds,
            'component_type': result.component_type,
            'confidence': result.confidence,
            'strategy_used': result.strategy_used,
            'calculation_time_ms': result.calculation_time_ms,
            'historical_adjustments': result.historical_adjustments
        }, indent=2))
    
    return 0

if __name__ == "__main__":
    exit(main())