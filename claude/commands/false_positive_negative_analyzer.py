#!/usr/bin/env python3
"""
False Positive/Negative Analysis UI

Advanced analysis tool for identifying, categorizing, and visualizing false positive
and false negative patterns in quality gate decisions.

Part of RIF Issue #94: Quality Gate Effectiveness Monitoring - Phase 2
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FalsePositivePattern:
    """Pattern representing common false positive scenarios"""
    pattern_id: str
    description: str
    frequency: int
    component_types: List[str]
    typical_conditions: List[str]
    impact_score: float  # 1-10 scale
    recommended_actions: List[str]
    examples: List[str]

@dataclass
class FalseNegativePattern:
    """Pattern representing common false negative scenarios"""
    pattern_id: str
    description: str
    frequency: int
    severity_levels: List[str]
    missed_defect_types: List[str]
    risk_score: float  # 1-10 scale
    prevention_strategies: List[str]
    examples: List[str]

@dataclass
class AnalysisResult:
    """Complete false positive/negative analysis results"""
    analysis_period: str
    total_decisions: int
    false_positive_count: int
    false_negative_count: int
    fp_rate: float
    fn_rate: float
    fp_patterns: List[FalsePositivePattern]
    fn_patterns: List[FalseNegativePattern]
    cost_analysis: Dict[str, float]
    recommendations: List[str]
    timestamp: str

class FalsePositiveNegativeAnalyzer:
    """
    Advanced analyzer for false positive and false negative patterns.
    Provides detailed insights into quality gate accuracy issues.
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize analyzer with knowledge base path"""
        self.knowledge_path = Path(knowledge_base_path)
        self.metrics_path = self.knowledge_path / "quality_metrics"
        self.analysis_path = self.knowledge_path / "analysis"
        
        # Create analysis directory
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        
        # Pattern matching rules
        self.fp_pattern_rules = self._initialize_fp_pattern_rules()
        self.fn_pattern_rules = self._initialize_fn_pattern_rules()
    
    def analyze_false_positives_negatives(self, days: int = 30) -> AnalysisResult:
        """
        Perform comprehensive false positive/negative analysis
        
        Args:
            days: Number of days to analyze
            
        Returns:
            AnalysisResult: Complete analysis results
        """
        self.start_time = datetime.now()
        
        try:
            logger.info(f"Analyzing false positives/negatives for last {days} days...")
            
            # Load quality gate decisions
            decisions = self._load_quality_decisions(days)
            
            # Separate false positives and negatives
            false_positives = [d for d in decisions if d.get('outcome') == 'false_positive']
            false_negatives = [d for d in decisions if d.get('outcome') == 'false_negative']
            
            logger.info(f"Found {len(false_positives)} false positives, {len(false_negatives)} false negatives")
            
            # Analyze patterns
            fp_patterns = self._analyze_fp_patterns(false_positives)
            fn_patterns = self._analyze_fn_patterns(false_negatives)
            
            # Cost analysis
            cost_analysis = self._calculate_cost_impact(false_positives, false_negatives, len(decisions))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(fp_patterns, fn_patterns, cost_analysis)
            
            # Create analysis result
            result = AnalysisResult(
                analysis_period=f"{days} days",
                total_decisions=len(decisions),
                false_positive_count=len(false_positives),
                false_negative_count=len(false_negatives),
                fp_rate=len(false_positives) / len(decisions) if decisions else 0.0,
                fn_rate=len(false_negatives) / len(decisions) if decisions else 0.0,
                fp_patterns=fp_patterns,
                fn_patterns=fn_patterns,
                cost_analysis=cost_analysis,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
            # Save analysis results
            self._save_analysis_result(result)
            
            processing_time = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.info(f"Analysis completed in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in false positive/negative analysis: {str(e)}")
            raise
    
    def _load_quality_decisions(self, days: int) -> List[Dict[str, Any]]:
        """Load quality gate decisions from specified time period"""
        decisions = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load from recent metrics
        recent_dir = self.metrics_path / "recent"
        if recent_dir.exists():
            for metrics_file in recent_dir.glob("*.json"):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check timestamp
                    if 'timestamp' in data:
                        file_date = datetime.fromisoformat(data['timestamp'])
                        if file_date >= cutoff_date:
                            decisions.append(data)
                            
                except Exception as e:
                    logger.warning(f"Error loading metrics file {metrics_file}: {str(e)}")
        
        # Load from real-time sessions
        realtime_dir = self.metrics_path / "realtime"
        if realtime_dir.exists():
            for session_file in realtime_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    for decision in session_data.get('decisions', []):
                        if decision.get('timestamp'):
                            decision_date = datetime.fromisoformat(decision['timestamp'])
                            if decision_date >= cutoff_date:
                                decisions.append(decision)
                                
                except Exception as e:
                    logger.warning(f"Error loading session file {session_file}: {str(e)}")
        
        # Load from correlation data
        correlation_dir = self.metrics_path / "correlations"
        if correlation_dir.exists():
            for corr_file in correlation_dir.glob("*.json"):
                try:
                    with open(corr_file, 'r') as f:
                        corr_data = json.load(f)
                    
                    for decision in corr_data.get('decisions', []):
                        if decision.get('timestamp'):
                            decision_date = datetime.fromisoformat(decision['timestamp'])
                            if decision_date >= cutoff_date and decision.get('production_outcome'):
                                # These have validated outcomes
                                decisions.append(decision)
                                
                except Exception as e:
                    logger.warning(f"Error loading correlation file {corr_file}: {str(e)}")
        
        logger.info(f"Loaded {len(decisions)} quality gate decisions")
        return decisions
    
    def _analyze_fp_patterns(self, false_positives: List[Dict[str, Any]]) -> List[FalsePositivePattern]:
        """Analyze patterns in false positive decisions"""
        if not false_positives:
            return []
        
        patterns = []
        
        # Group by various attributes for pattern detection
        component_groups = defaultdict(list)
        condition_groups = defaultdict(list)
        threshold_groups = defaultdict(list)
        
        for fp in false_positives:
            # Group by component type
            component_type = fp.get('component_type', 'unknown')
            component_groups[component_type].append(fp)
            
            # Group by quality conditions
            conditions = fp.get('quality_conditions', [])
            for condition in conditions:
                condition_groups[condition].append(fp)
            
            # Group by threshold proximity
            score = fp.get('quality_score', 0)
            threshold = fp.get('threshold', 1)
            if score and threshold:
                proximity = abs(score - threshold) / threshold
                if proximity < 0.1:  # Within 10% of threshold
                    threshold_groups['near_threshold'].append(fp)
                elif proximity < 0.2:  # Within 20% of threshold
                    threshold_groups['close_threshold'].append(fp)
        
        # Generate component-based patterns
        for component, fps in component_groups.items():
            if len(fps) >= 3:  # Need at least 3 instances for a pattern
                
                # Analyze typical conditions
                all_conditions = []
                for fp in fps:
                    all_conditions.extend(fp.get('quality_conditions', []))
                
                common_conditions = [cond for cond, count in Counter(all_conditions).most_common(3)]
                
                # Calculate impact score
                impact_score = min(10, len(fps) * 2)  # Scale based on frequency
                
                # Generate recommendations
                recommendations = self._generate_fp_recommendations(component, fps)
                
                # Get examples
                examples = [f"Score: {fp.get('quality_score', 0):.2f}, Threshold: {fp.get('threshold', 1):.2f}" 
                          for fp in fps[:3]]
                
                pattern = FalsePositivePattern(
                    pattern_id=f"fp_component_{component}",
                    description=f"False positives in {component} components",
                    frequency=len(fps),
                    component_types=[component],
                    typical_conditions=common_conditions,
                    impact_score=impact_score,
                    recommended_actions=recommendations,
                    examples=examples
                )
                patterns.append(pattern)
        
        # Generate threshold-based patterns
        for threshold_type, fps in threshold_groups.items():
            if len(fps) >= 5:  # Need more instances for threshold patterns
                
                components = list(set(fp.get('component_type', 'unknown') for fp in fps))
                
                pattern = FalsePositivePattern(
                    pattern_id=f"fp_threshold_{threshold_type}",
                    description=f"False positives {threshold_type.replace('_', ' ')}",
                    frequency=len(fps),
                    component_types=components,
                    typical_conditions=["threshold_proximity", "borderline_quality"],
                    impact_score=8.0,  # High impact due to threshold issues
                    recommended_actions=[
                        "Review threshold calibration",
                        "Consider dynamic thresholds",
                        "Analyze threshold sensitivity"
                    ],
                    examples=[f"Score: {fp.get('quality_score', 0):.2f}, Threshold: {fp.get('threshold', 1):.2f}" 
                            for fp in fps[:3]]
                )
                patterns.append(pattern)
        
        # Sort patterns by impact and frequency
        patterns.sort(key=lambda p: (p.impact_score, p.frequency), reverse=True)
        
        return patterns[:10]  # Return top 10 patterns
    
    def _analyze_fn_patterns(self, false_negatives: List[Dict[str, Any]]) -> List[FalseNegativePattern]:
        """Analyze patterns in false negative decisions"""
        if not false_negatives:
            return []
        
        patterns = []
        
        # Group by various attributes
        severity_groups = defaultdict(list)
        defect_type_groups = defaultdict(list)
        component_groups = defaultdict(list)
        
        for fn in false_negatives:
            # Group by defect severity
            severity = fn.get('production_defect_severity', 'unknown')
            severity_groups[severity].append(fn)
            
            # Group by defect type
            defect_type = fn.get('defect_type', 'unknown')
            defect_type_groups[defect_type].append(fn)
            
            # Group by component
            component = fn.get('component_type', 'unknown')
            component_groups[component].append(fn)
        
        # Generate severity-based patterns
        for severity, fns in severity_groups.items():
            if len(fns) >= 2 and severity != 'unknown':  # Lower threshold for FNs due to importance
                
                defect_types = list(set(fn.get('defect_type', 'unknown') for fn in fns))
                
                # Calculate risk score
                severity_weights = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}
                risk_score = severity_weights.get(severity.lower(), 5) * min(len(fns), 3)
                
                # Generate prevention strategies
                strategies = self._generate_fn_prevention_strategies(severity, fns)
                
                examples = [f"Missed {fn.get('defect_type', 'defect')} with severity {severity}" 
                          for fn in fns[:3]]
                
                pattern = FalseNegativePattern(
                    pattern_id=f"fn_severity_{severity}",
                    description=f"False negatives leading to {severity} severity defects",
                    frequency=len(fns),
                    severity_levels=[severity],
                    missed_defect_types=defect_types,
                    risk_score=risk_score,
                    prevention_strategies=strategies,
                    examples=examples
                )
                patterns.append(pattern)
        
        # Generate defect type patterns
        for defect_type, fns in defect_type_groups.items():
            if len(fns) >= 3 and defect_type != 'unknown':
                
                severities = list(set(fn.get('production_defect_severity', 'unknown') for fn in fns))
                
                # Calculate risk based on defect type and frequency
                defect_risk_weights = {
                    'security': 10, 'performance': 7, 'functionality': 6,
                    'ui': 3, 'documentation': 1
                }
                base_risk = defect_risk_weights.get(defect_type.lower(), 5)
                risk_score = base_risk + min(len(fns) * 2, 5)
                
                strategies = self._generate_defect_type_strategies(defect_type, fns)
                
                examples = [f"Missed {defect_type} defect, severity: {fn.get('production_defect_severity', 'unknown')}" 
                          for fn in fns[:3]]
                
                pattern = FalseNegativePattern(
                    pattern_id=f"fn_defect_{defect_type}",
                    description=f"Missed {defect_type} defects",
                    frequency=len(fns),
                    severity_levels=severities,
                    missed_defect_types=[defect_type],
                    risk_score=risk_score,
                    prevention_strategies=strategies,
                    examples=examples
                )
                patterns.append(pattern)
        
        # Sort patterns by risk score and frequency
        patterns.sort(key=lambda p: (p.risk_score, p.frequency), reverse=True)
        
        return patterns[:10]  # Return top 10 patterns
    
    def _generate_fp_recommendations(self, component: str, false_positives: List[Dict[str, Any]]) -> List[str]:
        """Generate specific recommendations for false positive patterns"""
        recommendations = []
        
        # Analyze score distributions
        scores = [fp.get('quality_score', 0) for fp in false_positives if fp.get('quality_score')]
        if scores:
            avg_score = sum(scores) / len(scores)
            recommendations.append(f"Consider raising {component} threshold (avg FP score: {avg_score:.2f})")
        
        # Analyze conditions
        conditions = []
        for fp in false_positives:
            conditions.extend(fp.get('quality_conditions', []))
        
        common_conditions = Counter(conditions).most_common(3)
        for condition, count in common_conditions:
            if count > len(false_positives) * 0.5:  # >50% of FPs have this condition
                recommendations.append(f"Review '{condition}' condition for {component} components")
        
        # Component-specific recommendations
        component_recommendations = {
            'function': ["Review complexity thresholds", "Consider context-aware analysis"],
            'class': ["Analyze inheritance impact", "Review coupling metrics"],
            'file': ["Consider file-specific patterns", "Review aggregation logic"],
            'module': ["Analyze inter-module dependencies", "Review module-level metrics"]
        }
        
        if component.lower() in component_recommendations:
            recommendations.extend(component_recommendations[component.lower()])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_fn_prevention_strategies(self, severity: str, false_negatives: List[Dict[str, Any]]) -> List[str]:
        """Generate prevention strategies for false negative patterns"""
        strategies = []
        
        # Severity-specific strategies
        severity_strategies = {
            'critical': [
                "Implement mandatory security review for critical components",
                "Add automated vulnerability scanning",
                "Require senior developer approval"
            ],
            'high': [
                "Enhance automated testing coverage",
                "Implement stricter quality gates",
                "Add performance regression testing"
            ],
            'medium': [
                "Improve code review guidelines",
                "Add automated style checking",
                "Enhance documentation requirements"
            ],
            'low': [
                "Add linting rules",
                "Improve developer training",
                "Enhance feedback loops"
            ]
        }
        
        base_strategies = severity_strategies.get(severity.lower(), [])
        strategies.extend(base_strategies)
        
        # Analyze patterns in the false negatives
        defect_types = [fn.get('defect_type', '') for fn in false_negatives]
        if 'security' in defect_types:
            strategies.append("Implement security-focused quality gates")
        if 'performance' in defect_types:
            strategies.append("Add performance profiling to quality checks")
        
        return strategies[:5]  # Return top 5 strategies
    
    def _generate_defect_type_strategies(self, defect_type: str, false_negatives: List[Dict[str, Any]]) -> List[str]:
        """Generate strategies specific to defect types"""
        type_strategies = {
            'security': [
                "Implement SAST/DAST scanning",
                "Add security-focused code review checklist",
                "Require security expert review for sensitive code"
            ],
            'performance': [
                "Add performance benchmarking",
                "Implement resource usage monitoring",
                "Add load testing requirements"
            ],
            'functionality': [
                "Increase unit test coverage requirements",
                "Add integration testing",
                "Implement behavior-driven testing"
            ],
            'ui': [
                "Add visual regression testing",
                "Implement accessibility scanning",
                "Add cross-browser testing"
            ],
            'documentation': [
                "Require inline documentation",
                "Add API documentation validation",
                "Implement documentation coverage metrics"
            ]
        }
        
        return type_strategies.get(defect_type.lower(), [
            "Enhance relevant quality checks",
            "Improve domain-specific training",
            "Add specialized review processes"
        ])
    
    def _calculate_cost_impact(self, false_positives: List[Dict[str, Any]], 
                             false_negatives: List[Dict[str, Any]], 
                             total_decisions: int) -> Dict[str, float]:
        """Calculate cost impact of false positives and negatives"""
        
        # Estimated costs (in developer hours)
        fp_investigation_time = 0.5  # 30 minutes per false positive
        fn_fix_time = 8.0  # 8 hours per false negative (including debugging, fixing, testing)
        blocked_pipeline_cost = 0.25  # 15 minutes per blocked pipeline
        
        fp_cost = len(false_positives) * fp_investigation_time
        fn_cost = len(false_negatives) * fn_fix_time
        
        # Calculate blocked pipeline impact (assuming some FPs block pipelines)
        blocked_fps = [fp for fp in false_positives if fp.get('blocked_pipeline', False)]
        pipeline_cost = len(blocked_fps) * blocked_pipeline_cost
        
        total_cost = fp_cost + fn_cost + pipeline_cost
        
        # Calculate rates and efficiency
        fp_rate = len(false_positives) / total_decisions if total_decisions > 0 else 0
        fn_rate = len(false_negatives) / total_decisions if total_decisions > 0 else 0
        
        return {
            'false_positive_cost_hours': fp_cost,
            'false_negative_cost_hours': fn_cost,
            'pipeline_disruption_cost_hours': pipeline_cost,
            'total_cost_hours': total_cost,
            'fp_rate_percent': fp_rate * 100,
            'fn_rate_percent': fn_rate * 100,
            'cost_per_decision_minutes': (total_cost * 60) / total_decisions if total_decisions > 0 else 0,
            'efficiency_score': max(0, 100 - (fp_rate * 100 + fn_rate * 200))  # FNs weighted 2x
        }
    
    def _generate_recommendations(self, fp_patterns: List[FalsePositivePattern],
                                fn_patterns: List[FalseNegativePattern],
                                cost_analysis: Dict[str, float]) -> List[str]:
        """Generate overall recommendations based on analysis"""
        recommendations = []
        
        # High-level recommendations based on rates
        fp_rate = cost_analysis.get('fp_rate_percent', 0)
        fn_rate = cost_analysis.get('fn_rate_percent', 0)
        
        if fp_rate > 15:  # >15% false positive rate
            recommendations.append("URGENT: High false positive rate requires immediate threshold review and calibration")
        elif fp_rate > 10:
            recommendations.append("Moderate false positive rate suggests threshold optimization needed")
        
        if fn_rate > 3:  # >3% false negative rate
            recommendations.append("CRITICAL: Elevated false negative rate poses quality risk - strengthen quality gates")
        elif fn_rate > 1:
            recommendations.append("Monitor false negative rate - consider additional quality checks")
        
        # Pattern-based recommendations
        if fp_patterns:
            top_fp_pattern = fp_patterns[0]
            recommendations.append(f"Address top FP pattern: {top_fp_pattern.description} ({top_fp_pattern.frequency} occurrences)")
        
        if fn_patterns:
            top_fn_pattern = fn_patterns[0]
            recommendations.append(f"Prioritize FN pattern: {top_fn_pattern.description} (risk score: {top_fn_pattern.risk_score})")
        
        # Cost-based recommendations
        total_cost = cost_analysis.get('total_cost_hours', 0)
        if total_cost > 40:  # >40 hours per analysis period
            recommendations.append(f"High cost impact ({total_cost:.1f} hours) - ROI positive for quality gate optimization")
        
        # Efficiency recommendations
        efficiency = cost_analysis.get('efficiency_score', 100)
        if efficiency < 85:
            recommendations.append("Quality gate efficiency below target (85%) - comprehensive review recommended")
        
        return recommendations[:7]  # Return top 7 recommendations
    
    def _save_analysis_result(self, result: AnalysisResult):
        """Save analysis results to file system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save current analysis
            current_file = self.analysis_path / "current_fp_fn_analysis.json"
            with open(current_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            # Save timestamped version
            historical_file = self.analysis_path / f"fp_fn_analysis_{timestamp}.json"
            with open(historical_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            logger.info(f"Analysis results saved to {current_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def _initialize_fp_pattern_rules(self) -> Dict[str, Any]:
        """Initialize false positive pattern matching rules"""
        return {
            'threshold_proximity': {
                'condition': lambda score, threshold: abs(score - threshold) / threshold < 0.1,
                'weight': 0.8
            },
            'component_specific': {
                'condition': lambda component: component in ['test', 'example', 'demo'],
                'weight': 0.6
            },
            'score_clustering': {
                'condition': 'statistical_clustering',
                'weight': 0.7
            }
        }
    
    def _initialize_fn_pattern_rules(self) -> Dict[str, Any]:
        """Initialize false negative pattern matching rules"""
        return {
            'severity_clustering': {
                'condition': 'severity_grouping',
                'weight': 0.9
            },
            'defect_type_patterns': {
                'condition': 'defect_type_grouping', 
                'weight': 0.8
            },
            'component_weakness': {
                'condition': 'component_vulnerability',
                'weight': 0.7
            }
        }
    
    def generate_html_report(self, result: AnalysisResult) -> str:
        """Generate HTML report for false positive/negative analysis"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>False Positive/Negative Analysis Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .report {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; color: #2c3e50; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .fp-value {{ color: #e74c3c; }}
        .fn-value {{ color: #c0392b; }}
        .cost-value {{ color: #8e44ad; }}
        .efficiency-value {{ color: #27ae60; }}
        .patterns-section {{ background: white; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .pattern-item {{ border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; background: #f8f9fa; }}
        .fp-pattern {{ border-left-color: #e74c3c; }}
        .fn-pattern {{ border-left-color: #c0392b; }}
        .pattern-header {{ font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
        .pattern-stats {{ display: flex; gap: 15px; font-size: 0.9em; color: #7f8c8d; margin-bottom: 8px; }}
        .pattern-details {{ margin-top: 10px; }}
        .pattern-list {{ margin: 5px 0; padding-left: 20px; }}
        .recommendations-section {{ background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; margin-bottom: 20px; }}
        .recommendation-item {{ padding: 8px 0; }}
        .cost-breakdown {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .cost-item {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #ecf0f1; }}
        .footer {{ text-align: center; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>False Positive/Negative Analysis Report</h1>
            <p>Analysis Period: {result.analysis_period} | Generated: {result.timestamp}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="fp-value metric-value">{result.fp_rate:.1%}</div>
                <div>False Positive Rate</div>
                <div style="font-size: 0.8em; color: #7f8c8d;">{result.false_positive_count} of {result.total_decisions} decisions</div>
            </div>
            <div class="summary-card">
                <div class="fn-value metric-value">{result.fn_rate:.1%}</div>
                <div>False Negative Rate</div>
                <div style="font-size: 0.8em; color: #7f8c8d;">{result.false_negative_count} of {result.total_decisions} decisions</div>
            </div>
            <div class="summary-card">
                <div class="cost-value metric-value">{result.cost_analysis.get('total_cost_hours', 0):.1f}h</div>
                <div>Total Cost Impact</div>
                <div style="font-size: 0.8em; color: #7f8c8d;">Development hours lost</div>
            </div>
            <div class="summary-card">
                <div class="efficiency-value metric-value">{result.cost_analysis.get('efficiency_score', 0):.0f}%</div>
                <div>Quality Gate Efficiency</div>
                <div style="font-size: 0.8em; color: #7f8c8d;">Target: >85%</div>
            </div>
        </div>
        
        <div class="patterns-section">
            <h2>False Positive Patterns</h2>
            {self._generate_fp_patterns_html(result.fp_patterns)}
        </div>
        
        <div class="patterns-section">
            <h2>False Negative Patterns</h2>
            {self._generate_fn_patterns_html(result.fn_patterns)}
        </div>
        
        <div class="recommendations-section">
            <h2>Recommendations</h2>
            {''.join([f'<div class="recommendation-item">• {rec}</div>' for rec in result.recommendations])}
        </div>
        
        <div class="cost-breakdown">
            <h2>Cost Analysis Breakdown</h2>
            <div class="cost-item">
                <span>False Positive Investigation Time:</span>
                <span>{result.cost_analysis.get('false_positive_cost_hours', 0):.1f} hours</span>
            </div>
            <div class="cost-item">
                <span>False Negative Fix Time:</span>
                <span>{result.cost_analysis.get('false_negative_cost_hours', 0):.1f} hours</span>
            </div>
            <div class="cost-item">
                <span>Pipeline Disruption Time:</span>
                <span>{result.cost_analysis.get('pipeline_disruption_cost_hours', 0):.1f} hours</span>
            </div>
            <div class="cost-item">
                <span>Cost Per Decision:</span>
                <span>{result.cost_analysis.get('cost_per_decision_minutes', 0):.1f} minutes</span>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by RIF False Positive/Negative Analysis System</p>
            <p>Issue #94 - Phase 2: Analytics Dashboard Implementation</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_fp_patterns_html(self, patterns: List[FalsePositivePattern]) -> str:
        """Generate HTML for false positive patterns"""
        if not patterns:
            return "<p>No significant false positive patterns detected.</p>"
        
        html = ""
        for pattern in patterns:
            html += f"""
            <div class="pattern-item fp-pattern">
                <div class="pattern-header">{pattern.description}</div>
                <div class="pattern-stats">
                    <span>Frequency: {pattern.frequency}</span>
                    <span>Impact Score: {pattern.impact_score:.1f}/10</span>
                    <span>Components: {', '.join(pattern.component_types)}</span>
                </div>
                <div class="pattern-details">
                    <div><strong>Typical Conditions:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{cond}</li>' for cond in pattern.typical_conditions])}
                    </ul>
                    <div><strong>Recommended Actions:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{action}</li>' for action in pattern.recommended_actions])}
                    </ul>
                    <div><strong>Examples:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{example}</li>' for example in pattern.examples[:2]])}
                    </ul>
                </div>
            </div>
            """
        return html
    
    def _generate_fn_patterns_html(self, patterns: List[FalseNegativePattern]) -> str:
        """Generate HTML for false negative patterns"""
        if not patterns:
            return "<p>No significant false negative patterns detected.</p>"
        
        html = ""
        for pattern in patterns:
            html += f"""
            <div class="pattern-item fn-pattern">
                <div class="pattern-header">{pattern.description}</div>
                <div class="pattern-stats">
                    <span>Frequency: {pattern.frequency}</span>
                    <span>Risk Score: {pattern.risk_score:.1f}/10</span>
                    <span>Severity Levels: {', '.join(pattern.severity_levels)}</span>
                </div>
                <div class="pattern-details">
                    <div><strong>Missed Defect Types:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{defect}</li>' for defect in pattern.missed_defect_types])}
                    </ul>
                    <div><strong>Prevention Strategies:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{strategy}</li>' for strategy in pattern.prevention_strategies])}
                    </ul>
                    <div><strong>Examples:</strong></div>
                    <ul class="pattern-list">
                        {''.join([f'<li>{example}</li>' for example in pattern.examples[:2]])}
                    </ul>
                </div>
            </div>
            """
        return html

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Analyze False Positive/Negative Patterns')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--output', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--knowledge-path', default='/Users/cal/DEV/RIF/knowledge', help='Path to knowledge base')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FalsePositiveNegativeAnalyzer(args.knowledge_path)
    
    # Run analysis
    print(f"Analyzing false positive/negative patterns for last {args.days} days...")
    result = analyzer.analyze_false_positives_negatives(args.days)
    
    # Output results
    if args.output in ['json', 'both']:
        print("\n=== ANALYSIS JSON RESULTS ===")
        print(json.dumps(asdict(result), indent=2))
    
    if args.output in ['html', 'both']:
        html_report = analyzer.generate_html_report(result)
        html_file = Path(args.knowledge_path) / "analysis" / "fp_fn_analysis_report.html"
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"\n=== HTML REPORT GENERATED ===")
        print(f"Report saved to: {html_file}")
        print(f"Open in browser: file://{html_file}")
    
    # Display summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Analysis Period: {result.analysis_period}")
    print(f"Total Decisions: {result.total_decisions}")
    print(f"False Positive Rate: {result.fp_rate:.1%} ({result.false_positive_count} decisions)")
    print(f"False Negative Rate: {result.fn_rate:.1%} ({result.false_negative_count} decisions)")
    print(f"Total Cost Impact: {result.cost_analysis.get('total_cost_hours', 0):.1f} hours")
    print(f"Quality Gate Efficiency: {result.cost_analysis.get('efficiency_score', 0):.0f}%")
    
    print(f"\nTop False Positive Patterns:")
    for i, pattern in enumerate(result.fp_patterns[:3], 1):
        print(f"  {i}. {pattern.description} (freq: {pattern.frequency}, impact: {pattern.impact_score:.1f})")
    
    print(f"\nTop False Negative Patterns:")
    for i, pattern in enumerate(result.fn_patterns[:3], 1):
        print(f"  {i}. {pattern.description} (freq: {pattern.frequency}, risk: {pattern.risk_score:.1f})")
    
    print(f"\nKey Recommendations:")
    for i, rec in enumerate(result.recommendations[:3], 1):
        print(f"  {i}. {rec}")

if __name__ == "__main__":
    main()