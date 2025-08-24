#!/usr/bin/env python3
"""
Performance Risk Calculator - Issue #93 Phase 3
Advanced performance risk assessment for the Multi-Dimensional Quality Scoring System.

This module provides:
- Performance regression pattern analysis
- Resource utilization risk assessment
- Scalability impact evaluation
- Performance-specific risk mitigation strategies
"""

import os
import json
import yaml
import subprocess
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import re
import statistics

@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    io_operations: int = 0
    network_calls: int = 0
    database_queries: int = 0
    cache_hit_ratio: float = 1.0

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    avg_execution_time: float = 0.0
    max_memory_usage: float = 0.0
    typical_cpu_usage: float = 0.0
    baseline_timestamp: datetime = None
    confidence: float = 1.0

@dataclass
class PerformanceRiskProfile:
    """Comprehensive performance risk assessment."""
    overall_risk_score: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    regression_percentage: float = 0.0
    risk_factors: Dict[str, float] = field(default_factory=dict)
    performance_concerns: List[str] = field(default_factory=list)
    scalability_impact: str = "minimal"  # minimal, moderate, significant, severe
    mitigation_strategies: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class FilePerformanceAnalysis:
    """Performance analysis for individual file."""
    file_path: str
    complexity_score: float = 0.0
    performance_patterns: Dict[str, int] = field(default_factory=dict)
    resource_risks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0

class PerformanceRiskCalculator:
    """
    Advanced performance risk calculator for multi-dimensional quality scoring.
    
    Analyzes code changes for performance impact including:
    - Algorithmic complexity analysis
    - Resource utilization patterns
    - Database and I/O operation risks
    - Scalability impact assessment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize performance risk calculator."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/performance-risk-patterns.yaml"
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.logger = logging.getLogger(__name__)
        
        # Load performance patterns and thresholds
        self._load_performance_config()
        
        # Performance-sensitive patterns
        self.performance_critical_patterns = {
            'loop_complexity': r'for.*for.*for|while.*while.*while',
            'nested_loops': r'for[^{]*{[^}]*for[^{]*{[^}]*for',
            'recursive_calls': r'def\s+(\w+).*:\s*.*\1\s*\(',
            'database_queries': r'(SELECT|INSERT|UPDATE|DELETE|execute|query)',
            'file_operations': r'(open|read|write|seek|file)',
            'network_calls': r'(requests\.|urllib|http|socket)',
            'memory_allocations': r'(malloc|new|list|dict|set)\s*\(',
            'string_concatenation': r'\+\s*["\'].*["\']|\+.*str\(',
            'regex_compilation': r're\.(compile|match|search|findall)',
            'json_operations': r'json\.(loads|dumps|encode|decode)',
            'synchronous_io': r'(time\.sleep|input\(|raw_input)',
            'inefficient_sorting': r'\.sort\(\)|sorted\(',
            'global_variables': r'global\s+\w+',
            'exception_handling': r'try:.*except.*:',
            'lambda_functions': r'lambda.*:',
        }
        
        # Algorithm complexity patterns
        self.complexity_patterns = {
            'o_n_squared': r'for.*in.*:.*for.*in.*:',
            'o_n_cubed': r'for.*in.*:.*for.*in.*:.*for.*in.*:',
            'nested_iterations': r'\.join\(.*for.*in.*for.*in',
            'cartesian_product': r'itertools\.product|for.*in.*for.*in.*if',
            'recursive_enumeration': r'def.*recursive.*:.*recursive\(',
        }
        
        # Resource utilization risks
        self.resource_risk_patterns = {
            'memory_leaks': r'while\s+True:|for.*in.*range\(.*,.*\):',
            'unbounded_growth': r'\.append\(|\.extend\(|list\(\)|dict\(\)',
            'large_collections': r'range\(\s*\d{4,}\)|list\(range\(',
            'inefficient_data_structures': r'list.*index\(|in\s+list',
            'blocking_operations': r'sleep\(|wait\(|join\(',
        }
    
    def _load_performance_config(self):
        """Load performance risk configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.performance_weights = config.get('performance_weights', {})
                    self.regression_thresholds = config.get('regression_thresholds', {})
                    self.complexity_weights = config.get('complexity_weights', {})
            else:
                self._set_default_performance_config()
        except Exception as e:
            self.logger.warning(f"Failed to load performance configuration: {e}")
            self._set_default_performance_config()
    
    def _set_default_performance_config(self):
        """Set default performance configuration."""
        self.performance_weights = {
            'algorithmic_complexity': 0.35,
            'resource_utilization': 0.25,
            'io_operations': 0.20,
            'memory_patterns': 0.20
        }
        self.regression_thresholds = {
            'acceptable': 5.0,    # 5% regression acceptable
            'concerning': 15.0,   # 15% regression concerning
            'critical': 30.0      # 30% regression critical
        }
        self.complexity_weights = {
            'o_n_squared': 0.8,
            'o_n_cubed': 1.0,
            'recursive_calls': 0.6,
            'nested_loops': 0.7
        }
    
    def calculate_performance_risk(self, 
                                 files: List[str],
                                 performance_data: Optional[Dict] = None,
                                 baseline: Optional[PerformanceBaseline] = None,
                                 context: Optional[str] = None) -> PerformanceRiskProfile:
        """
        Calculate comprehensive performance risk assessment.
        
        Args:
            files: List of files to analyze
            performance_data: Optional performance measurement data
            baseline: Optional performance baseline for comparison
            context: Optional component context
            
        Returns:
            PerformanceRiskProfile with detailed risk assessment
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"perf_risk:{hash(tuple(sorted(files)))}"
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_result
        
        try:
            # Analyze each file for performance risks
            file_analyses = []
            for file_path in files:
                if os.path.exists(file_path):
                    analysis = self._analyze_file_performance_risk(file_path)
                    file_analyses.append(analysis)
            
            # Calculate overall performance risk profile
            profile = self._calculate_performance_risk_profile(
                file_analyses, performance_data, baseline, context
            )
            
            # Cache result
            self.cache[cache_key] = (profile, datetime.now())
            
            # Clean old cache entries
            self._cleanup_cache()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Performance risk calculation failed: {e}")
            return PerformanceRiskProfile(
                overall_risk_score=0.3,
                risk_level="medium",
                confidence=0.5
            )
    
    def _analyze_file_performance_risk(self, file_path: str) -> FilePerformanceAnalysis:
        """Analyze performance risk for individual file."""
        start_time = time.time()
        
        analysis = FilePerformanceAnalysis(file_path=file_path)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Analyze algorithmic complexity
            complexity_score = 0.0
            for pattern_name, pattern in self.complexity_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    weight = self.complexity_weights.get(pattern_name, 0.5)
                    pattern_score = len(matches) * weight
                    complexity_score += pattern_score
                    analysis.performance_patterns[pattern_name] = len(matches)
            
            # Analyze general performance patterns  
            for pattern_name, pattern in self.performance_critical_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    analysis.performance_patterns[pattern_name] = len(matches)
                    
                    # Generate specific concerns
                    if pattern_name == 'database_queries' and len(matches) > 5:
                        analysis.resource_risks.append("High database query volume detected")
                    elif pattern_name == 'nested_loops' and len(matches) > 2:
                        analysis.resource_risks.append("Multiple nested loop patterns found")
                    elif pattern_name == 'memory_allocations' and len(matches) > 10:
                        analysis.resource_risks.append("Frequent memory allocation patterns")
                    elif pattern_name == 'synchronous_io' and len(matches) > 3:
                        analysis.resource_risks.append("Blocking I/O operations detected")
            
            # Analyze resource utilization risks
            for pattern_name, pattern in self.resource_risk_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    analysis.performance_patterns[f"resource_{pattern_name}"] = len(matches)
                    if len(matches) > 1:
                        analysis.resource_risks.append(f"Resource risk: {pattern_name}")
            
            # Calculate file complexity score
            total_patterns = sum(analysis.performance_patterns.values())
            line_count = len(lines)
            
            # Complexity based on pattern density and total patterns
            if line_count > 0:
                pattern_density = total_patterns / line_count
                analysis.complexity_score = min(complexity_score + (pattern_density * 100), 10.0)
            else:
                analysis.complexity_score = complexity_score
            
            # Generate optimization opportunities
            analysis.optimization_opportunities = self._generate_optimization_suggestions(
                analysis.performance_patterns, analysis.resource_risks
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze performance for {file_path}: {e}")
            analysis.complexity_score = 1.0  # Default low complexity
        
        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        return analysis
    
    def _calculate_performance_risk_profile(self,
                                          file_analyses: List[FilePerformanceAnalysis],
                                          performance_data: Optional[Dict],
                                          baseline: Optional[PerformanceBaseline],
                                          context: Optional[str]) -> PerformanceRiskProfile:
        """Calculate overall performance risk profile."""
        profile = PerformanceRiskProfile()
        
        if not file_analyses:
            return profile
        
        # Aggregate complexity scores
        complexity_scores = [analysis.complexity_score for analysis in file_analyses]
        avg_complexity = statistics.mean(complexity_scores)
        max_complexity = max(complexity_scores)
        
        # Aggregate performance patterns
        all_patterns = {}
        all_risks = []
        for analysis in file_analyses:
            for pattern, count in analysis.performance_patterns.items():
                all_patterns[pattern] = all_patterns.get(pattern, 0) + count
            all_risks.extend(analysis.resource_risks)
        
        # Calculate risk components
        risk_components = {}
        
        # Algorithmic complexity risk
        complexity_risk = min(avg_complexity * 0.1, 1.0)
        risk_components['algorithmic_complexity'] = complexity_risk
        
        # Resource utilization risk
        resource_pattern_count = sum(1 for pattern in all_patterns.keys() 
                                   if pattern.startswith('resource_'))
        resource_risk = min(resource_pattern_count * 0.2, 1.0)
        risk_components['resource_utilization'] = resource_risk
        
        # I/O operation risk
        io_patterns = ['database_queries', 'file_operations', 'network_calls']
        io_risk = sum(all_patterns.get(pattern, 0) for pattern in io_patterns) * 0.05
        risk_components['io_operations'] = min(io_risk, 1.0)
        
        # Memory pattern risk
        memory_patterns = ['memory_allocations', 'string_concatenation', 'large_collections']
        memory_risk = sum(all_patterns.get(pattern, 0) for pattern in memory_patterns) * 0.03
        risk_components['memory_patterns'] = min(memory_risk, 1.0)
        
        # Calculate regression percentage if performance data available
        regression_percentage = 0.0
        if performance_data and baseline:
            regression_percentage = self._calculate_regression(performance_data, baseline)
            profile.regression_percentage = regression_percentage
            
            # Add regression risk component
            if regression_percentage > self.regression_thresholds['critical']:
                risk_components['performance_regression'] = 1.0
            elif regression_percentage > self.regression_thresholds['concerning']:
                risk_components['performance_regression'] = 0.7
            elif regression_percentage > self.regression_thresholds['acceptable']:
                risk_components['performance_regression'] = 0.4
            else:
                risk_components['performance_regression'] = 0.1
        
        # Apply context-specific weighting
        context_multiplier = self._get_context_performance_multiplier(context)
        
        # Calculate overall risk score
        profile.overall_risk_score = min(
            sum(risk_components[comp] * self.performance_weights.get(comp, 0.25) 
                for comp in risk_components) * context_multiplier,
            1.0
        )
        
        profile.risk_factors = risk_components
        profile.performance_concerns = list(set(all_risks))
        
        # Determine risk level
        if profile.overall_risk_score >= 0.8:
            profile.risk_level = "critical"
        elif profile.overall_risk_score >= 0.6:
            profile.risk_level = "high"
        elif profile.overall_risk_score >= 0.3:
            profile.risk_level = "medium"
        else:
            profile.risk_level = "low"
        
        # Assess scalability impact
        profile.scalability_impact = self._assess_scalability_impact(
            max_complexity, all_patterns, len(file_analyses)
        )
        
        # Generate mitigation strategies
        profile.mitigation_strategies = self._generate_performance_mitigations(
            risk_components, all_patterns, regression_percentage
        )
        
        # Set confidence
        profile.confidence = min(len(file_analyses) / 5.0, 1.0)
        
        return profile
    
    def _calculate_regression(self, 
                            performance_data: Dict, 
                            baseline: PerformanceBaseline) -> float:
        """Calculate performance regression percentage."""
        if not baseline or not performance_data:
            return 0.0
        
        current_time = performance_data.get('execution_time_ms', 0)
        current_memory = performance_data.get('memory_usage_mb', 0)
        
        # Calculate time regression
        time_regression = 0.0
        if baseline.avg_execution_time > 0:
            time_regression = ((current_time - baseline.avg_execution_time) / 
                             baseline.avg_execution_time) * 100
        
        # Calculate memory regression
        memory_regression = 0.0
        if baseline.max_memory_usage > 0:
            memory_regression = ((current_memory - baseline.max_memory_usage) / 
                               baseline.max_memory_usage) * 100
        
        # Return worst regression
        return max(time_regression, memory_regression, 0.0)
    
    def _get_context_performance_multiplier(self, context: Optional[str]) -> float:
        """Get performance risk multiplier based on context."""
        if not context:
            return 1.0
        
        context_multipliers = {
            'critical_algorithms': 1.5,   # Performance is critical
            'public_apis': 1.3,           # Public APIs need performance
            'business_logic': 1.0,        # Standard performance requirements
            'integration_code': 1.2,      # Integration points can be bottlenecks
            'ui_components': 0.9,         # UI performance is important but not critical
            'test_code': 0.6              # Test performance is less critical
        }
        
        return context_multipliers.get(context, 1.0)
    
    def _assess_scalability_impact(self, 
                                 max_complexity: float, 
                                 patterns: Dict[str, int],
                                 file_count: int) -> str:
        """Assess scalability impact of performance risks."""
        # Consider complexity and pattern density
        complexity_impact = max_complexity / 10.0
        pattern_density = sum(patterns.values()) / max(file_count, 1)
        
        # High-impact patterns for scalability
        scalability_patterns = [
            'o_n_squared', 'o_n_cubed', 'nested_loops', 
            'database_queries', 'memory_allocations'
        ]
        
        scalability_risk = sum(patterns.get(pattern, 0) for pattern in scalability_patterns)
        
        total_impact = complexity_impact + (pattern_density * 0.1) + (scalability_risk * 0.05)
        
        if total_impact >= 2.0:
            return "severe"
        elif total_impact >= 1.0:
            return "significant"
        elif total_impact >= 0.5:
            return "moderate"
        else:
            return "minimal"
    
    def _generate_optimization_suggestions(self, 
                                         patterns: Dict[str, int],
                                         risks: List[str]) -> List[str]:
        """Generate optimization suggestions based on patterns."""
        suggestions = []
        
        # Pattern-specific suggestions
        if patterns.get('nested_loops', 0) > 2:
            suggestions.append("Consider algorithmic optimization to reduce nested loops")
        
        if patterns.get('database_queries', 0) > 5:
            suggestions.append("Implement query batching or caching to reduce database load")
        
        if patterns.get('memory_allocations', 0) > 10:
            suggestions.append("Review memory allocation patterns for optimization opportunities")
        
        if patterns.get('string_concatenation', 0) > 5:
            suggestions.append("Use string builders or join operations instead of concatenation")
        
        if patterns.get('synchronous_io', 0) > 0:
            suggestions.append("Consider asynchronous I/O operations for better performance")
        
        # Risk-specific suggestions
        if "High database query volume detected" in risks:
            suggestions.append("Implement database connection pooling and query optimization")
        
        if "Blocking I/O operations detected" in risks:
            suggestions.append("Use non-blocking I/O patterns or background processing")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _generate_performance_mitigations(self,
                                        risk_components: Dict[str, float],
                                        patterns: Dict[str, int],
                                        regression_percentage: float) -> List[str]:
        """Generate performance mitigation strategies."""
        strategies = []
        
        # Risk component-specific strategies
        if risk_components.get('algorithmic_complexity', 0) > 0.5:
            strategies.append("Review and optimize algorithm complexity in critical paths")
        
        if risk_components.get('resource_utilization', 0) > 0.5:
            strategies.append("Implement resource pooling and efficient data structures")
        
        if risk_components.get('io_operations', 0) > 0.5:
            strategies.append("Optimize I/O operations with caching and batching")
        
        if risk_components.get('memory_patterns', 0) > 0.5:
            strategies.append("Implement memory management optimizations")
        
        # Regression-specific strategies
        if regression_percentage > 15:
            strategies.append("Conduct performance profiling to identify regression sources")
            strategies.append("Implement performance monitoring and alerting")
        
        # Pattern-specific strategies
        if patterns.get('o_n_squared', 0) > 0 or patterns.get('o_n_cubed', 0) > 0:
            strategies.append("Replace quadratic/cubic algorithms with more efficient alternatives")
        
        return strategies[:4]  # Limit to top 4 strategies
    
    def _cleanup_cache(self):
        """Clean expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, cache_time) in self.cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

def main():
    """CLI interface for performance risk calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Risk Calculator')
    parser.add_argument('--files', nargs='+', required=True, help='Files to analyze')
    parser.add_argument('--context', help='Component context')
    parser.add_argument('--performance-data', help='JSON file with performance measurements')
    parser.add_argument('--baseline', help='JSON file with performance baseline')
    parser.add_argument('--output', choices=['score', 'level', 'regression', 'full'], 
                       default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = PerformanceRiskCalculator()
    
    # Load performance data if provided
    performance_data = None
    if args.performance_data and os.path.exists(args.performance_data):
        with open(args.performance_data, 'r') as f:
            performance_data = json.load(f)
    
    # Load baseline if provided
    baseline = None
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, 'r') as f:
            baseline_data = json.load(f)
            baseline = PerformanceBaseline(**baseline_data)
    
    # Perform analysis
    profile = calculator.calculate_performance_risk(
        files=args.files,
        performance_data=performance_data,
        baseline=baseline,
        context=args.context
    )
    
    # Output results
    if args.output == 'score':
        print(f"{profile.overall_risk_score:.3f}")
    elif args.output == 'level':
        print(profile.risk_level)
    elif args.output == 'regression':
        print(f"{profile.regression_percentage:.1f}")
    else:
        result = {
            'overall_risk_score': profile.overall_risk_score,
            'risk_level': profile.risk_level,
            'regression_percentage': profile.regression_percentage,
            'risk_factors': profile.risk_factors,
            'scalability_impact': profile.scalability_impact,
            'mitigation_strategies': profile.mitigation_strategies,
            'confidence': profile.confidence
        }
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()