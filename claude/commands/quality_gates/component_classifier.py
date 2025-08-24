#!/usr/bin/env python3
"""
Component Classifier for Context-Aware Quality Thresholds System
Issue #91: Context-Aware Quality Thresholds System

Multi-stage file classification system achieving >95% accuracy.
Performance target: <100ms per file classification.

This module implements a three-stage classification pipeline:
1. Pattern matching (80% accuracy, <10ms) - Fast path-based classification
2. Content analysis (15% edge cases, <50ms) - Handle ambiguous files  
3. Heuristic fallback (5% uncertain, <10ms) - Ensure 100% classification coverage
"""

import re
import time
import yaml
import fnmatch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

@dataclass
class ComponentType:
    """Represents a classified component type with confidence score."""
    type: str
    confidence: float
    classification_method: str
    patterns_matched: List[str]
    content_patterns_matched: List[str]
    processing_time_ms: float

@dataclass
class ClassificationMetrics:
    """Metrics for classification performance monitoring."""
    total_files_processed: int
    pattern_match_count: int
    content_analysis_count: int
    fallback_count: int
    average_confidence: float
    total_processing_time_ms: float
    accuracy_rate: Optional[float] = None

class ClassificationMethod(Enum):
    """Classification method types for performance tracking."""
    PATTERN_MATCHING = "pattern_matching"
    CONTENT_ANALYSIS = "content_analysis"
    HEURISTIC_FALLBACK = "heuristic_fallback"

class ComponentClassifier:
    """
    Multi-stage file classification system achieving >95% accuracy.
    Performance target: <100ms per file classification.
    """
    
    def __init__(self, config_path: str = "config/component-types.yaml"):
        """
        Initialize component classifier.
        
        Args:
            config_path: Path to component types configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.component_types = self.config.get('component_types', {})
        self.classification_config = self.config.get('configuration', {}).get('classification', {})
        self.performance_config = self.config.get('configuration', {}).get('performance', {})
        
        # Performance tracking
        self.metrics = ClassificationMetrics(
            total_files_processed=0,
            pattern_match_count=0,
            content_analysis_count=0,
            fallback_count=0,
            average_confidence=0.0,
            total_processing_time_ms=0.0
        )
        
        # Setup logging first
        self.setup_logging()
        
        # Compiled patterns for performance
        self._compiled_patterns = {}
        self._compiled_content_patterns = {}
        self._setup_compiled_patterns()
    
    def setup_logging(self):
        """Setup logging for component classification."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ComponentClassifier - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load component types configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
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
                    'risk_factor': 1.2,
                    'patterns': ['**/*.py', '**/*.js', '**/*.ts'],
                    'content_patterns': []
                }
            },
            'configuration': {
                'classification': {
                    'pattern_matching_weight': 0.6,
                    'content_analysis_weight': 0.3,
                    'heuristic_fallback_weight': 0.1,
                    'max_classification_time_ms': 100,
                    'high_confidence_threshold': 0.9,
                    'medium_confidence_threshold': 0.7,
                    'low_confidence_threshold': 0.5
                }
            }
        }
    
    def _setup_compiled_patterns(self):
        """Pre-compile regex patterns for performance optimization."""
        self.logger.info("Compiling patterns for optimized performance...")
        
        for component_type, config in self.component_types.items():
            # Compile content patterns as regex
            content_patterns = config.get('content_patterns', [])
            compiled_content = []
            
            for pattern in content_patterns:
                try:
                    compiled_content.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{pattern}' for {component_type}: {e}")
            
            self._compiled_content_patterns[component_type] = compiled_content
        
        self.logger.info(f"Compiled patterns for {len(self.component_types)} component types")
    
    def classify_file(self, file_path: str, content: Optional[str] = None) -> ComponentType:
        """
        Classify a single file using three-stage classification pipeline.
        
        Args:
            file_path: Path to the file to classify
            content: Optional file content for content analysis
            
        Returns:
            ComponentType with classification results
        """
        start_time = time.time()
        
        try:
            # Stage 1: Pattern matching (fast path, ~80% accuracy)
            pattern_result = self._pattern_matching_stage(file_path)
            
            if pattern_result.confidence >= self.classification_config.get('high_confidence_threshold', 0.9):
                pattern_result.processing_time_ms = (time.time() - start_time) * 1000
                self._update_metrics(pattern_result, ClassificationMethod.PATTERN_MATCHING)
                return pattern_result
            
            # Stage 2: Content analysis (edge cases, ~15% of files)
            if content and pattern_result.confidence < self.classification_config.get('high_confidence_threshold', 0.9):
                content_result = self._content_analysis_stage(file_path, content, pattern_result)
                
                if content_result.confidence >= self.classification_config.get('medium_confidence_threshold', 0.7):
                    content_result.processing_time_ms = (time.time() - start_time) * 1000
                    self._update_metrics(content_result, ClassificationMethod.CONTENT_ANALYSIS)
                    return content_result
            
            # Stage 3: Heuristic fallback (~5% uncertain cases)
            fallback_result = self._heuristic_fallback_stage(file_path, pattern_result)
            fallback_result.processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(fallback_result, ClassificationMethod.HEURISTIC_FALLBACK)
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Error classifying file {file_path}: {e}")
            # Return fallback classification
            fallback = ComponentType(
                type="business_logic",
                confidence=0.6,
                classification_method="error_fallback",
                patterns_matched=[],
                content_patterns_matched=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
            self._update_metrics(fallback, ClassificationMethod.HEURISTIC_FALLBACK)
            return fallback
    
    def _pattern_matching_stage(self, file_path: str) -> ComponentType:
        """
        Stage 1: Fast path-based classification using fnmatch patterns.
        Target: <10ms processing time, ~80% accuracy.
        """
        best_match = None
        best_confidence = 0.0
        best_patterns_matched = []
        
        file_path_lower = file_path.lower()
        file_extension = Path(file_path).suffix.lower()
        
        for component_type, config in self.component_types.items():
            patterns = config.get('patterns', [])
            confidence = 0.0
            patterns_matched = []
            
            # Check path patterns
            for pattern in patterns:
                if fnmatch.fnmatch(file_path_lower, pattern.lower()):
                    confidence += 0.8  # Strong pattern match
                    patterns_matched.append(pattern)
            
            # Check critical extensions
            critical_extensions = config.get('critical_extensions', [])
            if file_extension in critical_extensions:
                confidence += 0.3  # Extension bonus
            
            # Special handling for test files - they have highest priority
            if component_type == 'test_utilities' and (
                'test' in file_path_lower or 
                file_path_lower.startswith('test') or
                '/test' in file_path_lower or
                'spec' in file_path_lower
            ):
                confidence += 1.0  # Very strong boost for test files
                # If this is clearly a test file, set very high confidence
                if ('test' in file_path_lower and ('test_' in file_path_lower or '/test' in file_path_lower)):
                    confidence = 2.0  # Override other patterns
            
            # Special handling for UI components - boost confidence for templates and views
            elif component_type == 'ui_components' and (
                'template' in file_path_lower or
                'view' in file_path_lower or
                'component' in file_path_lower or
                file_extension in ['.html', '.htm', '.jsx', '.vue', '.tsx', '.svelte']
            ):
                confidence += 0.4  # Strong boost for UI files
            
            # Apply priority weighting
            priority = config.get('priority', 5)
            priority_weight = 1.0 + (6 - priority) * 0.1  # Higher priority = higher weight
            confidence *= priority_weight
            
            # Normalize confidence (but allow test files to exceed 1.0 for priority)
            if component_type == 'test_utilities' and confidence > 1.5:
                confidence = min(2.0, confidence)  # Allow test files higher confidence
            else:
                confidence = min(1.0, confidence)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = component_type
                best_patterns_matched = patterns_matched
        
        # Default to business_logic if no patterns match
        if not best_match:
            best_match = "business_logic"
            best_confidence = 0.6
        
        return ComponentType(
            type=best_match,
            confidence=best_confidence,
            classification_method="pattern_matching",
            patterns_matched=best_patterns_matched,
            content_patterns_matched=[],
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def _content_analysis_stage(self, file_path: str, content: str, pattern_result: ComponentType) -> ComponentType:
        """
        Stage 2: Content-based classification for ambiguous files.
        Target: <50ms processing time for edge cases.
        """
        best_match = pattern_result.type
        best_confidence = pattern_result.confidence
        best_content_patterns = []
        
        # Limit content analysis to first 10KB for performance
        content_sample = content[:10240] if len(content) > 10240 else content
        
        for component_type, compiled_patterns in self._compiled_content_patterns.items():
            if not compiled_patterns:
                continue
                
            content_confidence = 0.0
            content_patterns_matched = []
            
            # Check compiled content patterns
            for pattern in compiled_patterns:
                matches = pattern.findall(content_sample)
                if matches:
                    # Weight matches based on component type importance
                    match_weight = 0.6 if component_type in ['critical_algorithms', 'public_apis'] else 0.4
                    content_confidence += match_weight * len(matches)
                    content_patterns_matched.append(pattern.pattern)
            
            # Special boost for API detection in content
            if component_type == 'public_apis' and any(
                keyword in content_sample.lower() for keyword in 
                ['@app.route', 'class.*api', 'def get(', 'def post(', 'def put(', 'def delete(', 'endpoint']
            ):
                content_confidence += 0.7  # Strong API content indicator
                content_patterns_matched.append("api_keyword_detection")
            
            # Combine with pattern-based confidence
            config = self.component_types.get(component_type, {})
            priority_weight = 1.0 + (6 - config.get('priority', 5)) * 0.1
            
            # Weighted combination of pattern and content confidence
            pattern_weight = self.classification_config.get('pattern_matching_weight', 0.6)
            content_weight = self.classification_config.get('content_analysis_weight', 0.3)
            
            # If content analysis found strong indicators, give it more weight
            if content_confidence > 0.7:
                pattern_weight = 0.4
                content_weight = 0.6
            
            combined_confidence = (
                (pattern_result.confidence if component_type == pattern_result.type else 0.2) * pattern_weight +
                content_confidence * content_weight
            ) * priority_weight
            
            combined_confidence = min(1.0, combined_confidence)
            
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_match = component_type
                best_content_patterns = content_patterns_matched
        
        return ComponentType(
            type=best_match,
            confidence=best_confidence,
            classification_method="content_analysis",
            patterns_matched=pattern_result.patterns_matched,
            content_patterns_matched=best_content_patterns,
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def _heuristic_fallback_stage(self, file_path: str, pattern_result: ComponentType) -> ComponentType:
        """
        Stage 3: Heuristic fallback for uncertain classifications.
        Target: <10ms processing time, ensures 100% classification coverage.
        """
        file_path_lower = file_path.lower()
        
        # Heuristic rules based on common patterns
        heuristic_rules = [
            # High-priority patterns
            (['algorithm', 'crypto', 'security', 'core'], 'critical_algorithms', 0.8),
            (['api', 'endpoint', 'route', 'controller'], 'public_apis', 0.7),
            (['service', 'business', 'logic', 'model'], 'business_logic', 0.75),
            (['integration', 'client', 'external', 'connector'], 'integration_code', 0.7),
            (['component', 'view', 'page', 'template'], 'ui_components', 0.65),
            (['test', 'spec', 'fixture', 'mock'], 'test_utilities', 0.6),
        ]
        
        best_match = pattern_result.type
        best_confidence = pattern_result.confidence
        
        for keywords, component_type, confidence in heuristic_rules:
            if any(keyword in file_path_lower for keyword in keywords):
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = component_type
                break  # Use first matching rule
        
        # Ensure minimum confidence for fallback
        if best_confidence < self.classification_config.get('low_confidence_threshold', 0.5):
            best_confidence = 0.6
        
        return ComponentType(
            type=best_match,
            confidence=best_confidence,
            classification_method="heuristic_fallback",
            patterns_matched=pattern_result.patterns_matched,
            content_patterns_matched=[],
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def batch_classify(self, files: List[str], contents: Optional[Dict[str, str]] = None) -> Dict[str, ComponentType]:
        """
        Optimized batch classification for performance.
        
        Args:
            files: List of file paths to classify
            contents: Optional mapping of file paths to their content
            
        Returns:
            Dictionary mapping file paths to their classifications
        """
        self.logger.info(f"Starting batch classification of {len(files)} files")
        start_time = time.time()
        
        results = {}
        
        # Process files in batches for memory efficiency
        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file_path in batch:
                content = contents.get(file_path) if contents else None
                results[file_path] = self.classify_file(file_path, content)
        
        total_time = (time.time() - start_time) * 1000
        avg_time_per_file = total_time / len(files) if files else 0
        
        self.logger.info(f"Batch classification completed in {total_time:.2f}ms")
        self.logger.info(f"Average time per file: {avg_time_per_file:.2f}ms")
        
        return results
    
    def get_confidence_score(self, classification: ComponentType) -> float:
        """
        Calculate classification confidence [0.0, 1.0].
        
        Args:
            classification: ComponentType result
            
        Returns:
            Normalized confidence score
        """
        return min(1.0, max(0.0, classification.confidence))
    
    def _update_metrics(self, classification: ComponentType, method: ClassificationMethod):
        """Update performance metrics."""
        self.metrics.total_files_processed += 1
        self.metrics.total_processing_time_ms += classification.processing_time_ms
        
        # Update confidence running average
        old_avg = self.metrics.average_confidence
        new_avg = (old_avg * (self.metrics.total_files_processed - 1) + classification.confidence) / self.metrics.total_files_processed
        self.metrics.average_confidence = new_avg
        
        # Update method counters
        if method == ClassificationMethod.PATTERN_MATCHING:
            self.metrics.pattern_match_count += 1
        elif method == ClassificationMethod.CONTENT_ANALYSIS:
            self.metrics.content_analysis_count += 1
        elif method == ClassificationMethod.HEURISTIC_FALLBACK:
            self.metrics.fallback_count += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if self.metrics.total_files_processed == 0:
            return {"error": "No files processed yet"}
        
        avg_time = self.metrics.total_processing_time_ms / self.metrics.total_files_processed
        
        return {
            "total_files_processed": self.metrics.total_files_processed,
            "average_processing_time_ms": round(avg_time, 2),
            "average_confidence": round(self.metrics.average_confidence, 3),
            "method_distribution": {
                "pattern_matching": self.metrics.pattern_match_count,
                "content_analysis": self.metrics.content_analysis_count,
                "heuristic_fallback": self.metrics.fallback_count
            },
            "performance_target_met": avg_time < self.classification_config.get('max_classification_time_ms', 100),
            "total_processing_time_ms": round(self.metrics.total_processing_time_ms, 2)
        }
    
    def validate_accuracy(self, test_dataset: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Validate classification accuracy against test dataset.
        
        Args:
            test_dataset: List of (file_path, expected_component_type) tuples
            
        Returns:
            Accuracy validation results
        """
        self.logger.info(f"Validating accuracy on {len(test_dataset)} test cases")
        
        correct_classifications = 0
        total_classifications = len(test_dataset)
        classification_results = []
        
        for file_path, expected_type in test_dataset:
            predicted_type = self.classify_file(file_path)
            is_correct = predicted_type.type == expected_type
            
            if is_correct:
                correct_classifications += 1
            
            classification_results.append({
                "file_path": file_path,
                "expected": expected_type,
                "predicted": predicted_type.type,
                "confidence": predicted_type.confidence,
                "correct": is_correct,
                "processing_time_ms": predicted_type.processing_time_ms
            })
        
        accuracy = (correct_classifications / total_classifications) * 100 if total_classifications > 0 else 0
        self.metrics.accuracy_rate = accuracy
        
        validation_result = {
            "accuracy_percent": round(accuracy, 2),
            "correct_classifications": correct_classifications,
            "total_classifications": total_classifications,
            "meets_requirement": accuracy >= 95.0,
            "classification_details": classification_results,
            "performance_metrics": self.get_performance_metrics()
        }
        
        self.logger.info(f"Accuracy validation completed: {accuracy:.2f}% (target: 95%)")
        return validation_result

def main():
    """Command line interface for component classification."""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python component_classifier.py <command> [args]")
        print("Commands:")
        print("  classify <file_path>           - Classify a single file")
        print("  batch <file_list_path>         - Batch classify files from list")
        print("  validate <test_dataset_path>   - Validate accuracy on test dataset")
        print("  metrics                        - Show performance metrics")
        return
    
    command = sys.argv[1]
    classifier = ComponentClassifier()
    
    if command == "classify" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        result = classifier.classify_file(file_path)
        
        output = {
            "file_path": file_path,
            "component_type": result.type,
            "confidence": round(result.confidence, 3),
            "classification_method": result.classification_method,
            "patterns_matched": result.patterns_matched,
            "content_patterns_matched": result.content_patterns_matched,
            "processing_time_ms": round(result.processing_time_ms, 2)
        }
        
        print(json.dumps(output, indent=2))
        
    elif command == "batch" and len(sys.argv) >= 3:
        file_list_path = sys.argv[2]
        
        try:
            with open(file_list_path, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
            
            results = classifier.batch_classify(files)
            
            output = {
                "batch_results": {
                    file_path: {
                        "component_type": result.type,
                        "confidence": round(result.confidence, 3),
                        "classification_method": result.classification_method,
                        "processing_time_ms": round(result.processing_time_ms, 2)
                    }
                    for file_path, result in results.items()
                },
                "performance_metrics": classifier.get_performance_metrics()
            }
            
            print(json.dumps(output, indent=2))
            
        except Exception as e:
            print(f"Error processing file list: {e}", file=sys.stderr)
            return 1
    
    elif command == "metrics":
        metrics = classifier.get_performance_metrics()
        print(json.dumps(metrics, indent=2))
    
    elif command == "validate" and len(sys.argv) >= 3:
        test_dataset_path = sys.argv[2]
        
        try:
            with open(test_dataset_path, 'r') as f:
                test_data = json.load(f)
            
            test_dataset = [(item["file_path"], item["expected_type"]) for item in test_data]
            
            validation_result = classifier.validate_accuracy(test_dataset)
            print(json.dumps(validation_result, indent=2))
            
            # Exit with error code if accuracy requirement not met
            if not validation_result["meets_requirement"]:
                return 1
            
        except Exception as e:
            print(f"Error validating accuracy: {e}", file=sys.stderr)
            return 1
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit(main())