#!/usr/bin/env python3
"""
Context Weight Manager - Issue #93 Phase 1
Manages component type classification and context-specific weighting for quality scoring.

This module provides:
- Automatic component type detection based on file paths and patterns
- Context-specific quality thresholds
- Weighted scoring adjustments per component criticality
- Performance optimization for rapid classification
"""

import os
import json
import yaml
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ComponentClassification:
    """Classification result for a set of components."""
    primary_type: str
    confidence: float
    type_distribution: Dict[str, float] = field(default_factory=dict)
    matched_patterns: List[str] = field(default_factory=list)
    weight: float = 1.0
    thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextWeighting:
    """Complete context weighting result."""
    overall_weight: float
    component_weights: Dict[str, float] = field(default_factory=dict)
    weighted_thresholds: Dict[str, Any] = field(default_factory=dict)
    classification_time_ms: float = 0.0
    calculation_time_ms: float = 0.0

class ContextWeightManager:
    """
    Manages context-aware weighting and thresholds for quality scoring.
    
    Features:
    - Automatic component type detection
    - Context-specific quality thresholds
    - Performance-optimized classification
    - Multi-component change handling
    - Configurable weighting strategies
    """
    
    def __init__(self, config_path: str = "config/quality-dimensions.yaml"):
        """Initialize context weight manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self.context_weights = self.config.get('context_weights', {})
        
        # Performance configuration
        self.performance_config = self.config.get('performance', {})
        self.classification_limit_ms = self.performance_config.get('classification_time_limit_ms', 50)
        
        # Caching for performance
        self.classification_cache = {}
        self.cache_duration_minutes = self.performance_config.get('cache_duration_minutes', 30)
        
        # Setup logging
        self.setup_logging()
        
        # Pre-compile patterns for performance
        self._compile_patterns()
        
        # Performance tracking
        self.classification_times = []
        self.calculation_times = []
    
    def setup_logging(self):
        """Setup logging for context weight management."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ContextWeightManager - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load context weight configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Config file {self.config_path} not found")
                return self._get_default_context_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_context_config()
    
    def _get_default_context_config(self) -> Dict[str, Any]:
        """Default context configuration for fallback."""
        return {
            'context_weights': {
                'critical_algorithms': {'weight': 1.2, 'patterns': ['**/algorithm/**', '**/core/**']},
                'public_apis': {'weight': 1.1, 'patterns': ['**/api/**', '**/rest/**']},
                'business_logic': {'weight': 1.0, 'patterns': ['**/business/**', '**/service/**']},
                'integration_code': {'weight': 0.9, 'patterns': ['**/integration/**', '**/middleware/**']},
                'ui_components': {'weight': 0.8, 'patterns': ['**/ui/**', '**/component/**']},
                'test_code': {'weight': 0.6, 'patterns': ['**/test/**', '**/tests/**']}
            },
            'performance': {
                'classification_time_limit_ms': 50,
                'cache_duration_minutes': 30
            }
        }
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.compiled_patterns = {}
        
        for context_type, config in self.context_weights.items():
            patterns = config.get('patterns', [])
            compiled = []
            
            for pattern in patterns:
                try:
                    # Convert glob-like pattern to regex
                    regex_pattern = pattern.replace('**/', '.*?/').replace('*', '[^/]*')
                    regex_pattern = f"^{regex_pattern}$"
                    compiled.append(re.compile(regex_pattern, re.IGNORECASE))
                except Exception as e:
                    self.logger.warning(f"Could not compile pattern '{pattern}': {e}")
            
            self.compiled_patterns[context_type] = compiled
    
    def classify_components(self, file_paths: List[str], strategy: str = "balanced") -> ComponentClassification:
        """
        Classify components based on file paths and determine primary type.
        
        Args:
            file_paths: List of file paths to classify
            strategy: Classification strategy (balanced, size_based, priority_based, risk_based)
            
        Returns:
            Component classification with primary type and confidence
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{hash(tuple(sorted(file_paths)))}_{strategy}"
            cached_result = self._get_cached_classification(cache_key)
            if cached_result:
                return cached_result
            
            # Perform classification
            type_scores = defaultdict(float)
            matched_patterns = []
            
            for file_path in file_paths:
                file_classifications = self._classify_single_file(file_path)
                
                for context_type, score in file_classifications.items():
                    type_scores[context_type] += score
                    if score > 0:
                        matched_patterns.extend(self.context_weights.get(context_type, {}).get('patterns', []))
            
            # Apply strategy-specific weighting
            weighted_scores = self._apply_classification_strategy(type_scores, file_paths, strategy)
            
            # Determine primary type and confidence
            if not weighted_scores:
                primary_type = 'business_logic'  # Default fallback
                confidence = 0.5
                type_distribution = {'business_logic': 1.0}
            else:
                total_score = sum(weighted_scores.values())
                if total_score > 0:
                    type_distribution = {t: s/total_score for t, s in weighted_scores.items()}
                    primary_type = max(weighted_scores, key=weighted_scores.get)
                    confidence = weighted_scores[primary_type] / total_score
                else:
                    primary_type = 'business_logic'  # Default fallback
                    confidence = 0.5
                    type_distribution = {'business_logic': 1.0}
            
            # Get context weight and thresholds for primary type
            context_config = self.context_weights.get(primary_type, {'weight': 1.0})
            weight = context_config.get('weight', 1.0)
            thresholds = context_config.get('thresholds', {})
            
            # Create classification result
            classification = ComponentClassification(
                primary_type=primary_type,
                confidence=confidence,
                type_distribution=type_distribution,
                matched_patterns=list(set(matched_patterns)),
                weight=weight,
                thresholds=thresholds
            )
            
            # Cache result
            self._cache_classification(cache_key, classification)
            
            # Track performance
            classification_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.classification_times.append(classification_time)
            
            if classification_time > self.classification_limit_ms:
                self.logger.warning(f"Component classification took {classification_time:.1f}ms (limit: {self.classification_limit_ms}ms)")
            
            self.logger.debug(f"Classified {len(file_paths)} files as '{primary_type}' (confidence: {confidence:.2f}, weight: {weight:.2f})")
            
            return classification
        
        except Exception as e:
            self.logger.error(f"Error in component classification: {e}")
            # Return safe fallback
            return ComponentClassification(
                primary_type='business_logic',
                confidence=0.5,
                weight=1.0,
                thresholds={}
            )
    
    def _classify_single_file(self, file_path: str) -> Dict[str, float]:
        """Classify a single file against all context types."""
        file_scores = {}
        
        for context_type, compiled_patterns in self.compiled_patterns.items():
            score = 0.0
            
            for pattern in compiled_patterns:
                if pattern.match(file_path):
                    score = 1.0
                    break
            
            file_scores[context_type] = score
        
        return file_scores
    
    def _apply_classification_strategy(
        self, 
        type_scores: Dict[str, float], 
        file_paths: List[str], 
        strategy: str
    ) -> Dict[str, float]:
        """Apply classification strategy to raw type scores."""
        if strategy == "balanced":
            return dict(type_scores)
        
        elif strategy == "size_based":
            # Weight by number of files
            total_files = len(file_paths)
            return {t: s * (1 + total_files / 100) for t, s in type_scores.items()}
        
        elif strategy == "priority_based":
            # Boost critical and security-related types
            priority_boost = {
                'critical_algorithms': 1.5,
                'public_apis': 1.3,
                'business_logic': 1.0,
                'integration_code': 0.8,
                'ui_components': 0.6,
                'test_code': 0.4
            }
            return {t: s * priority_boost.get(t, 1.0) for t, s in type_scores.items()}
        
        elif strategy == "risk_based":
            # Boost types that typically have higher risk
            risk_boost = {
                'critical_algorithms': 1.4,
                'public_apis': 1.2,
                'business_logic': 1.0,
                'integration_code': 1.1,  # Integration can be risky
                'ui_components': 0.7,
                'configuration': 1.3,  # Config changes can be risky
                'test_code': 0.5
            }
            return {t: s * risk_boost.get(t, 1.0) for t, s in type_scores.items()}
        
        else:
            self.logger.warning(f"Unknown classification strategy: {strategy}")
            return dict(type_scores)
    
    def calculate_context_weighting(
        self, 
        file_paths: List[str],
        strategy: str = "balanced",
        multi_component_handling: str = "weighted_average"
    ) -> ContextWeighting:
        """
        Calculate comprehensive context weighting for quality scoring.
        
        Args:
            file_paths: List of file paths being modified
            strategy: Classification strategy
            multi_component_handling: How to handle multiple component types
            
        Returns:
            Complete context weighting with all details
        """
        calc_start_time = time.time()
        
        try:
            # Step 1: Classify components
            classification = self.classify_components(file_paths, strategy)
            
            # Step 2: Handle multi-component scenarios
            if multi_component_handling == "weighted_average":
                overall_weight = self._calculate_weighted_average_weight(classification)
                component_weights = classification.type_distribution
                
            elif multi_component_handling == "highest_priority":
                overall_weight = classification.weight
                component_weights = {classification.primary_type: 1.0}
                
            elif multi_component_handling == "most_conservative":
                # Use the highest weight (most strict)
                max_weight = max(
                    self.context_weights.get(t, {}).get('weight', 1.0) 
                    for t in classification.type_distribution.keys()
                ) if classification.type_distribution else 1.0
                overall_weight = max_weight
                component_weights = classification.type_distribution
                
            else:
                overall_weight = classification.weight
                component_weights = classification.type_distribution
            
            # Step 3: Calculate weighted thresholds
            weighted_thresholds = self._calculate_weighted_thresholds(
                classification.type_distribution,
                multi_component_handling
            )
            
            # Track performance
            calculation_time = (time.time() - calc_start_time) * 1000
            self.calculation_times.append(calculation_time)
            
            # Create result
            context_weighting = ContextWeighting(
                overall_weight=overall_weight,
                component_weights=component_weights,
                weighted_thresholds=weighted_thresholds,
                classification_time_ms=self.classification_times[-1] if self.classification_times else 0.0,
                calculation_time_ms=calculation_time
            )
            
            self.logger.debug(f"Context weighting calculated: weight={overall_weight:.2f}, primary_type={classification.primary_type}")
            
            return context_weighting
        
        except Exception as e:
            self.logger.error(f"Error calculating context weighting: {e}")
            # Return safe fallback
            return ContextWeighting(
                overall_weight=1.0,
                component_weights={'business_logic': 1.0},
                weighted_thresholds={},
                calculation_time_ms=0.0
            )
    
    def _calculate_weighted_average_weight(self, classification: ComponentClassification) -> float:
        """Calculate weighted average weight across multiple component types."""
        if not classification.type_distribution:
            return 1.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for context_type, proportion in classification.type_distribution.items():
            context_weight = self.context_weights.get(context_type, {}).get('weight', 1.0)
            weighted_sum += context_weight * proportion
            total_weight += proportion
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def _calculate_weighted_thresholds(
        self, 
        type_distribution: Dict[str, float],
        handling_strategy: str
    ) -> Dict[str, Any]:
        """Calculate weighted quality thresholds across component types."""
        if not type_distribution:
            return {}
        
        weighted_thresholds = {}
        
        # Get all threshold keys from all component types
        all_threshold_keys = set()
        for context_type in type_distribution.keys():
            thresholds = self.context_weights.get(context_type, {}).get('thresholds', {})
            all_threshold_keys.update(thresholds.keys())
        
        # Calculate weighted threshold for each key
        for threshold_key in all_threshold_keys:
            if handling_strategy == "most_conservative":
                # Use the most restrictive threshold
                threshold_values = []
                for context_type in type_distribution.keys():
                    context_thresholds = self.context_weights.get(context_type, {}).get('thresholds', {})
                    if threshold_key in context_thresholds:
                        threshold_values.append(context_thresholds[threshold_key])
                
                if threshold_values:
                    if isinstance(threshold_values[0], (int, float)):
                        # For numeric thresholds, use the highest (most strict)
                        weighted_thresholds[threshold_key] = max(threshold_values)
                    else:
                        # For string thresholds, use a priority order
                        priority = {'strict': 4, 'high': 3, 'standard': 2, 'relaxed': 1, 'zero_tolerance': 5}
                        best_threshold = max(threshold_values, key=lambda x: priority.get(x, 0))
                        weighted_thresholds[threshold_key] = best_threshold
            
            else:
                # Weighted average approach
                weighted_sum = 0.0
                total_weight = 0.0
                string_values = []
                
                for context_type, proportion in type_distribution.items():
                    context_thresholds = self.context_weights.get(context_type, {}).get('thresholds', {})
                    if threshold_key in context_thresholds:
                        threshold_value = context_thresholds[threshold_key]
                        
                        if isinstance(threshold_value, (int, float)):
                            weighted_sum += threshold_value * proportion
                            total_weight += proportion
                        else:
                            string_values.append((threshold_value, proportion))
                
                if total_weight > 0:
                    weighted_thresholds[threshold_key] = weighted_sum / total_weight
                elif string_values:
                    # For string values, pick the one with highest proportion
                    best_value = max(string_values, key=lambda x: x[1])[0]
                    weighted_thresholds[threshold_key] = best_value
        
        return weighted_thresholds
    
    def _get_cached_classification(self, cache_key: str) -> Optional[ComponentClassification]:
        """Get cached classification result if valid."""
        if cache_key in self.classification_cache:
            cached_data, timestamp = self.classification_cache[cache_key]
            
            # Check if cache is still valid
            cache_age_minutes = (time.time() - timestamp) / 60
            if cache_age_minutes < self.cache_duration_minutes:
                return cached_data
            else:
                # Remove expired entry
                del self.classification_cache[cache_key]
        
        return None
    
    def _cache_classification(self, cache_key: str, classification: ComponentClassification):
        """Cache classification result."""
        self.classification_cache[cache_key] = (classification, time.time())
        
        # Clean up old cache entries periodically
        if len(self.classification_cache) > 100:  # Limit cache size
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.classification_cache.items()
                if (current_time - timestamp) / 60 > self.cache_duration_minutes
            ]
            for key in expired_keys:
                del self.classification_cache[key]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for context weight management."""
        metrics = {}
        
        if self.classification_times:
            metrics.update({
                'avg_classification_time_ms': sum(self.classification_times) / len(self.classification_times),
                'max_classification_time_ms': max(self.classification_times),
                'classification_target_ms': self.classification_limit_ms
            })
        
        if self.calculation_times:
            metrics.update({
                'avg_calculation_time_ms': sum(self.calculation_times) / len(self.calculation_times),
                'max_calculation_time_ms': max(self.calculation_times),
                'total_calculations': len(self.calculation_times)
            })
        
        metrics['cache_hit_ratio'] = len(self.classification_cache) / max(len(self.classification_times), 1)
        
        return metrics

def main():
    """Command line interface for context weight management."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Context Weight Manager')
    parser.add_argument('--config', default='config/quality-dimensions.yaml', help='Configuration file path')
    parser.add_argument('--files', nargs='+', required=True, help='File paths to classify')
    parser.add_argument('--strategy', choices=['balanced', 'size_based', 'priority_based', 'risk_based'], 
                       default='balanced', help='Classification strategy')
    parser.add_argument('--multi-component', choices=['weighted_average', 'highest_priority', 'most_conservative'],
                       default='weighted_average', help='Multi-component handling')
    parser.add_argument('--output', choices=['weight', 'classification', 'full'], default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Create manager
    manager = ContextWeightManager(config_path=args.config)
    
    # Calculate context weighting
    weighting = manager.calculate_context_weighting(
        file_paths=args.files,
        strategy=args.strategy,
        multi_component_handling=args.multi_component
    )
    
    # Output based on format
    if args.output == 'weight':
        print(f"{weighting.overall_weight:.3f}")
    elif args.output == 'classification':
        classification = manager.classify_components(args.files, args.strategy)
        print(json.dumps({
            'primary_type': classification.primary_type,
            'confidence': classification.confidence,
            'type_distribution': classification.type_distribution
        }, indent=2))
    else:
        print(json.dumps({
            'overall_weight': weighting.overall_weight,
            'component_weights': weighting.component_weights,
            'weighted_thresholds': weighting.weighted_thresholds,
            'performance_metrics': {
                'classification_time_ms': weighting.classification_time_ms,
                'calculation_time_ms': weighting.calculation_time_ms
            }
        }, indent=2))
    
    return 0

if __name__ == "__main__":
    exit(main())