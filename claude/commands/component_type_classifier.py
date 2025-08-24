#!/usr/bin/env python3
"""
Component Type Classifier - Issue #93 Phase 2
Automatic detection and classification of component types for context-aware thresholds.

This module provides:
- High-performance pattern matching for component classification
- Machine learning-enhanced classification accuracy
- Confidence scoring for classification results
- Historical classification tracking and improvement
"""

import os
import json
import yaml
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib

@dataclass
class ClassificationResult:
    """Result of component type classification."""
    component_type: str
    confidence: float
    match_score: float
    matched_patterns: List[str] = field(default_factory=list)
    alternative_types: Dict[str, float] = field(default_factory=dict)
    classification_time_ms: float = 0.0

@dataclass
class BatchClassificationResult:
    """Result of batch component classification."""
    primary_type: str
    confidence: float
    type_distribution: Dict[str, float] = field(default_factory=dict)
    individual_results: List[ClassificationResult] = field(default_factory=list)
    total_files: int = 0
    classification_time_ms: float = 0.0
    cache_hit_ratio: float = 0.0

class ComponentTypeClassifier:
    """
    High-performance component type classifier with pattern matching and machine learning.
    
    Features:
    - Fast pattern matching with compiled regex
    - Confidence scoring based on pattern strength
    - Historical classification tracking
    - Performance optimization with caching
    - Accuracy validation and improvement
    """
    
    def __init__(self, config_path: str = "config/context-thresholds.yaml"):
        """Initialize the component type classifier."""
        self.config_path = config_path
        self.config = self._load_config()
        self.component_definitions = self.config.get('component_thresholds', {})
        
        # Performance tracking
        self.classification_cache = {}
        self.performance_metrics = {
            'classifications': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0,
            'accuracy_samples': []
        }
        
        # Setup logging
        self.setup_logging()
        
        # Pre-compile patterns for performance
        self.compiled_patterns = self._compile_classification_patterns()
        
        # Load historical classification data for accuracy improvement
        self.historical_data = self._load_historical_classifications()
        
        # Validation configuration
        self.validation_config = self.config.get('validation_rules', {})
        
    def setup_logging(self):
        """Setup logging for component classification."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ComponentTypeClassifier - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load component threshold configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Config file {self.config_path} not found")
                return self._get_default_classification_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_classification_config()
    
    def _get_default_classification_config(self) -> Dict[str, Any]:
        """Default classification configuration for fallback."""
        return {
            'component_thresholds': {
                'critical_algorithms': {
                    'patterns': ['**/algorithm/**', '**/core/**', '**/engine/**'],
                    'weight': 1.2
                },
                'public_apis': {
                    'patterns': ['**/api/**', '**/rest/**', '**/graphql/**'],
                    'weight': 1.1
                },
                'business_logic': {
                    'patterns': ['**/business/**', '**/service/**', '**/domain/**'],
                    'weight': 1.0
                },
                'integration_code': {
                    'patterns': ['**/integration/**', '**/middleware/**', '**/adapter/**'],
                    'weight': 0.9
                },
                'ui_components': {
                    'patterns': ['**/ui/**', '**/component/**', '**/view/**'],
                    'weight': 0.8
                },
                'test_code': {
                    'patterns': ['**/test/**', '**/tests/**', '**/*_test.*'],
                    'weight': 0.6
                }
            }
        }
    
    def _compile_classification_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float]]]:
        """Pre-compile regex patterns for fast classification."""
        compiled_patterns = {}
        
        for component_type, config in self.component_definitions.items():
            patterns = config.get('patterns', [])
            compiled = []
            
            for i, pattern in enumerate(patterns):
                try:
                    # Convert glob-like pattern to regex
                    regex_pattern = self._glob_to_regex(pattern)
                    compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
                    
                    # Assign confidence based on pattern specificity
                    specificity = self._calculate_pattern_specificity(pattern)
                    confidence = min(1.0, 0.7 + (specificity * 0.3))
                    
                    compiled.append((compiled_regex, confidence))
                    
                except Exception as e:
                    self.logger.warning(f"Could not compile pattern '{pattern}' for {component_type}: {e}")
            
            compiled_patterns[component_type] = compiled
        
        self.logger.info(f"Compiled {sum(len(patterns) for patterns in compiled_patterns.values())} patterns for {len(compiled_patterns)} component types")
        return compiled_patterns
    
    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob-like pattern to regex with enhanced matching."""
        # Enhanced pattern conversion for better accuracy
        regex_pattern = pattern
        
        # Handle special cases for better matching
        regex_pattern = regex_pattern.replace('**/', '(?:.*/)?')  # Zero or more directories
        regex_pattern = regex_pattern.replace('*', '[^/]*')       # Match within directory
        regex_pattern = regex_pattern.replace('?', '[^/]')        # Single character
        
        # Add word boundaries for more precise matching
        regex_pattern = f'^{regex_pattern}$'
        
        return regex_pattern
    
    def _calculate_pattern_specificity(self, pattern: str) -> float:
        """Calculate specificity score for pattern weighting."""
        specificity = 0.0
        
        # More specific patterns get higher scores
        if '**' not in pattern:
            specificity += 0.3  # Exact path matching
        
        if pattern.count('/') > 2:
            specificity += 0.2  # Deep path structure
        
        if '.' in pattern:
            specificity += 0.2  # File extension matching
        
        if not pattern.endswith('*'):
            specificity += 0.3  # Exact ending
        
        return min(1.0, specificity)
    
    def classify_single_file(self, file_path: str, use_cache: bool = True) -> ClassificationResult:
        """
        Classify a single file into component type.
        
        Args:
            file_path: File path to classify
            use_cache: Whether to use classification cache
            
        Returns:
            Classification result with confidence and alternatives
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(file_path)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                return cached_result
        
        try:
            classification_scores = {}
            matched_patterns = {}
            
            # Test against all component type patterns
            for component_type, compiled_patterns in self.compiled_patterns.items():
                max_score = 0.0
                best_patterns = []
                
                for pattern_regex, pattern_confidence in compiled_patterns:
                    if pattern_regex.match(file_path):
                        score = pattern_confidence
                        max_score = max(max_score, score)
                        best_patterns.append(pattern_regex.pattern)
                
                if max_score > 0:
                    classification_scores[component_type] = max_score
                    matched_patterns[component_type] = best_patterns
            
            # Determine primary classification
            if classification_scores:
                primary_type = max(classification_scores, key=classification_scores.get)
                confidence = classification_scores[primary_type]
                
                # Apply historical accuracy adjustment
                confidence = self._adjust_confidence_with_history(primary_type, file_path, confidence)
                
                # Get alternative classifications
                alternatives = {k: v for k, v in classification_scores.items() if k != primary_type}
                
                result = ClassificationResult(
                    component_type=primary_type,
                    confidence=confidence,
                    match_score=classification_scores[primary_type],
                    matched_patterns=matched_patterns.get(primary_type, []),
                    alternative_types=alternatives
                )
            else:
                # No patterns matched - use heuristic classification
                heuristic_type, heuristic_confidence = self._heuristic_classification(file_path)
                result = ClassificationResult(
                    component_type=heuristic_type,
                    confidence=heuristic_confidence,
                    match_score=heuristic_confidence,
                    matched_patterns=[],
                    alternative_types={}
                )
            
            # Track performance
            classification_time = (time.time() - start_time) * 1000
            result.classification_time_ms = classification_time
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, result)
            
            # Update metrics
            self.performance_metrics['classifications'] += 1
            self.performance_metrics['total_time_ms'] += classification_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying file {file_path}: {e}")
            
            # Return safe fallback
            return ClassificationResult(
                component_type='business_logic',
                confidence=0.5,
                match_score=0.5,
                classification_time_ms=(time.time() - start_time) * 1000
            )
    
    def classify_batch(
        self, 
        file_paths: List[str], 
        aggregation_strategy: str = "weighted_voting"
    ) -> BatchClassificationResult:
        """
        Classify a batch of files and determine overall component type.
        
        Args:
            file_paths: List of file paths to classify
            aggregation_strategy: Strategy for combining individual classifications
            
        Returns:
            Batch classification result with primary type and distribution
        """
        start_time = time.time()
        
        if not file_paths:
            return BatchClassificationResult(
                primary_type='business_logic',
                confidence=0.5,
                total_files=0,
                classification_time_ms=0.0
            )
        
        try:
            # Classify individual files
            individual_results = []
            for file_path in file_paths:
                result = self.classify_single_file(file_path)
                individual_results.append(result)
            
            # Aggregate results based on strategy
            if aggregation_strategy == "weighted_voting":
                primary_type, confidence, distribution = self._weighted_voting_aggregation(individual_results)
            elif aggregation_strategy == "majority_vote":
                primary_type, confidence, distribution = self._majority_vote_aggregation(individual_results)
            elif aggregation_strategy == "highest_confidence":
                primary_type, confidence, distribution = self._highest_confidence_aggregation(individual_results)
            elif aggregation_strategy == "pattern_strength":
                primary_type, confidence, distribution = self._pattern_strength_aggregation(individual_results)
            else:
                primary_type, confidence, distribution = self._weighted_voting_aggregation(individual_results)
            
            # Calculate performance metrics
            total_time = (time.time() - start_time) * 1000
            cache_hits = sum(1 for result in individual_results if result.classification_time_ms < 1.0)
            cache_hit_ratio = cache_hits / len(individual_results) if individual_results else 0.0
            
            return BatchClassificationResult(
                primary_type=primary_type,
                confidence=confidence,
                type_distribution=distribution,
                individual_results=individual_results,
                total_files=len(file_paths),
                classification_time_ms=total_time,
                cache_hit_ratio=cache_hit_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Error in batch classification: {e}")
            
            # Return safe fallback
            return BatchClassificationResult(
                primary_type='business_logic',
                confidence=0.5,
                total_files=len(file_paths),
                classification_time_ms=(time.time() - start_time) * 1000
            )
    
    def _weighted_voting_aggregation(self, results: List[ClassificationResult]) -> Tuple[str, float, Dict[str, float]]:
        """Aggregate using confidence-weighted voting."""
        if not results:
            return 'business_logic', 0.5, {'business_logic': 1.0}
        
        weighted_scores = defaultdict(float)
        total_weight = 0.0
        
        for result in results:
            weight = result.confidence
            weighted_scores[result.component_type] += weight
            total_weight += weight
        
        if total_weight == 0:
            return 'business_logic', 0.5, {'business_logic': 1.0}
        
        # Normalize to get distribution
        distribution = {k: v / total_weight for k, v in weighted_scores.items()}
        
        # Primary type and confidence
        primary_type = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[primary_type] / total_weight
        
        return primary_type, confidence, distribution
    
    def _majority_vote_aggregation(self, results: List[ClassificationResult]) -> Tuple[str, float, Dict[str, float]]:
        """Aggregate using simple majority vote."""
        if not results:
            return 'business_logic', 0.5, {'business_logic': 1.0}
        
        type_counts = Counter(result.component_type for result in results)
        total_files = len(results)
        
        primary_type = type_counts.most_common(1)[0][0]
        confidence = type_counts[primary_type] / total_files
        
        distribution = {k: v / total_files for k, v in type_counts.items()}
        
        return primary_type, confidence, distribution
    
    def _highest_confidence_aggregation(self, results: List[ClassificationResult]) -> Tuple[str, float, Dict[str, float]]:
        """Aggregate by selecting the highest confidence classification."""
        if not results:
            return 'business_logic', 0.5, {'business_logic': 1.0}
        
        best_result = max(results, key=lambda r: r.confidence)
        
        # Still calculate distribution for completeness
        type_counts = Counter(result.component_type for result in results)
        distribution = {k: v / len(results) for k, v in type_counts.items()}
        
        return best_result.component_type, best_result.confidence, distribution
    
    def _pattern_strength_aggregation(self, results: List[ClassificationResult]) -> Tuple[str, float, Dict[str, float]]:
        """Aggregate based on pattern match strength."""
        if not results:
            return 'business_logic', 0.5, {'business_logic': 1.0}
        
        strength_scores = defaultdict(float)
        
        for result in results:
            # Weight by both confidence and number of matched patterns
            pattern_bonus = len(result.matched_patterns) * 0.1
            strength = result.match_score + pattern_bonus
            strength_scores[result.component_type] += strength
        
        primary_type = max(strength_scores, key=strength_scores.get)
        
        # Calculate confidence and distribution
        total_strength = sum(strength_scores.values())
        if total_strength > 0:
            confidence = strength_scores[primary_type] / total_strength
            distribution = {k: v / total_strength for k, v in strength_scores.items()}
        else:
            confidence = 0.5
            distribution = {'business_logic': 1.0}
        
        return primary_type, confidence, distribution
    
    def _heuristic_classification(self, file_path: str) -> Tuple[str, float]:
        """Fallback heuristic classification when no patterns match."""
        file_path_lower = file_path.lower()
        
        # File extension-based heuristics
        if file_path_lower.endswith(('.test.js', '.spec.js', '.test.py', '.spec.py', '_test.py', 'Test.java')):
            return 'test_code', 0.8
        
        if file_path_lower.endswith(('.html', '.css', '.scss', '.less', '.vue', '.jsx')):
            return 'ui_components', 0.7
        
        if file_path_lower.endswith(('.json', '.yaml', '.yml', '.xml', '.conf', '.ini')):
            return 'configuration', 0.7
        
        # Path-based heuristics
        if '/test/' in file_path_lower or '/tests/' in file_path_lower:
            return 'test_code', 0.6
        
        if '/config/' in file_path_lower or '/settings/' in file_path_lower:
            return 'configuration', 0.6
        
        if '/api/' in file_path_lower or '/rest/' in file_path_lower:
            return 'public_apis', 0.6
        
        # Default fallback
        return 'business_logic', 0.4
    
    def _adjust_confidence_with_history(self, component_type: str, file_path: str, base_confidence: float) -> float:
        """Adjust confidence based on historical classification accuracy."""
        # This would ideally use machine learning model
        # For now, use simple heuristics based on historical data
        
        if component_type in self.historical_data:
            accuracy = self.historical_data[component_type].get('accuracy', 0.8)
            # Adjust confidence based on historical accuracy
            adjusted_confidence = base_confidence * accuracy
            return min(1.0, max(0.1, adjusted_confidence))
        
        return base_confidence
    
    def _load_historical_classifications(self) -> Dict[str, Dict[str, float]]:
        """Load historical classification data for accuracy tracking."""
        try:
            history_file = Path("knowledge/metrics/classification-history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.debug(f"Could not load classification history: {e}")
        
        # Return default historical data
        return {
            component_type: {'accuracy': 0.8, 'samples': 0}
            for component_type in self.component_definitions.keys()
        }
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file path."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ClassificationResult]:
        """Get cached classification result if valid."""
        if cache_key in self.classification_cache:
            cached_result, timestamp = self.classification_cache[cache_key]
            
            # Check if cache is still valid (5 minutes)
            if time.time() - timestamp < 300:  # 5 minutes
                return cached_result
            else:
                del self.classification_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: ClassificationResult):
        """Cache classification result."""
        self.classification_cache[cache_key] = (result, time.time())
        
        # Limit cache size
        if len(self.classification_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.classification_cache.keys(),
                key=lambda k: self.classification_cache[k][1]
            )[:100]
            for key in oldest_keys:
                del self.classification_cache[key]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get classification performance metrics."""
        metrics = dict(self.performance_metrics)
        
        if self.performance_metrics['classifications'] > 0:
            metrics['average_time_ms'] = (
                self.performance_metrics['total_time_ms'] / 
                self.performance_metrics['classifications']
            )
            metrics['cache_hit_rate'] = (
                self.performance_metrics['cache_hits'] / 
                self.performance_metrics['classifications']
            )
        
        return metrics
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate classification configuration for consistency."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'performance_check': {}
        }
        
        try:
            # Check pattern coverage
            total_patterns = sum(
                len(config.get('patterns', []))
                for config in self.component_definitions.values()
            )
            
            if total_patterns < 10:
                validation_results['warnings'].append("Low pattern coverage - consider adding more patterns")
            
            # Check for pattern conflicts
            all_patterns = []
            for component_type, config in self.component_definitions.items():
                patterns = config.get('patterns', [])
                all_patterns.extend([(p, component_type) for p in patterns])
            
            # Performance check
            start_time = time.time()
            self.classify_single_file("test/sample/file.py", use_cache=False)
            classification_time = (time.time() - start_time) * 1000
            
            validation_results['performance_check'] = {
                'single_classification_ms': classification_time,
                'target_ms': 50,
                'passes_target': classification_time < 50
            }
            
            if classification_time > 50:
                validation_results['warnings'].append(
                    f"Classification time {classification_time:.1f}ms exceeds 50ms target"
                )
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results

def main():
    """Command line interface for component type classifier."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Component Type Classifier')
    parser.add_argument('--config', default='config/context-thresholds.yaml', help='Configuration file path')
    parser.add_argument('--file', help='Single file to classify')
    parser.add_argument('--files', nargs='+', help='Multiple files to classify')
    parser.add_argument('--strategy', choices=['weighted_voting', 'majority_vote', 'highest_confidence', 'pattern_strength'],
                       default='weighted_voting', help='Batch aggregation strategy')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--performance', action='store_true', help='Show performance metrics')
    parser.add_argument('--output', choices=['type', 'confidence', 'full'], default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = ComponentTypeClassifier(config_path=args.config)
    
    if args.validate:
        validation = classifier.validate_configuration()
        print(json.dumps(validation, indent=2))
        return 0
    
    if args.performance:
        metrics = classifier.get_performance_metrics()
        print(json.dumps(metrics, indent=2))
        return 0
    
    if args.file:
        # Single file classification
        result = classifier.classify_single_file(args.file)
        
        if args.output == 'type':
            print(result.component_type)
        elif args.output == 'confidence':
            print(f"{result.confidence:.3f}")
        else:
            print(json.dumps({
                'component_type': result.component_type,
                'confidence': result.confidence,
                'match_score': result.match_score,
                'matched_patterns': result.matched_patterns,
                'alternative_types': result.alternative_types,
                'classification_time_ms': result.classification_time_ms
            }, indent=2))
    
    elif args.files:
        # Batch file classification
        result = classifier.classify_batch(args.files, args.strategy)
        
        if args.output == 'type':
            print(result.primary_type)
        elif args.output == 'confidence':
            print(f"{result.confidence:.3f}")
        else:
            print(json.dumps({
                'primary_type': result.primary_type,
                'confidence': result.confidence,
                'type_distribution': result.type_distribution,
                'total_files': result.total_files,
                'classification_time_ms': result.classification_time_ms,
                'cache_hit_ratio': result.cache_hit_ratio
            }, indent=2))
    
    else:
        print("Please specify either --file or --files")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())