#!/usr/bin/env python3
"""
Component Performance Breakdown Analyzer

Detailed performance analysis tool for quality gate effectiveness by component type,
providing granular insights into component-specific quality patterns and performance metrics.

Part of RIF Issue #94: Quality Gate Effectiveness Monitoring - Phase 2
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentMetrics:
    """Performance metrics for a specific component type"""
    component_type: str
    total_decisions: int
    effectiveness_score: float
    false_positive_rate: float
    false_negative_rate: float
    avg_processing_time_ms: float
    success_rate: float
    error_rate: float
    quality_score_distribution: Dict[str, float]  # percentiles
    threshold_hit_rate: float
    trend_direction: str
    performance_grade: str  # A, B, C, D, F

@dataclass
class QualityConditionAnalysis:
    """Analysis of quality conditions for a component"""
    condition_name: str
    frequency: int
    avg_score: float
    effectiveness: float
    false_positive_contribution: float
    recommendation: str

@dataclass
class ComponentComparison:
    """Comparison between component types"""
    compared_components: List[str]
    best_performer: str
    worst_performer: str
    performance_gaps: Dict[str, float]
    improvement_opportunities: List[str]

@dataclass
class ComponentBreakdownResult:
    """Complete component performance analysis result"""
    analysis_period: str
    total_components: int
    component_metrics: List[ComponentMetrics]
    condition_analysis: Dict[str, List[QualityConditionAnalysis]]
    component_comparison: ComponentComparison
    performance_insights: List[str]
    optimization_recommendations: List[str]
    timestamp: str

class ComponentPerformanceAnalyzer:
    """
    Advanced analyzer for component-specific quality gate performance.
    Provides detailed breakdown and comparison across component types.
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize analyzer with knowledge base path"""
        self.knowledge_path = Path(knowledge_base_path)
        self.metrics_path = self.knowledge_path / "quality_metrics"
        self.analysis_path = self.knowledge_path / "analysis"
        
        # Create analysis directory
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        
        # Performance grading thresholds
        self.grading_thresholds = {
            'A': {'effectiveness': 0.95, 'fp_rate': 0.05, 'fn_rate': 0.01, 'processing_time': 30},
            'B': {'effectiveness': 0.90, 'fp_rate': 0.10, 'fn_rate': 0.02, 'processing_time': 50},
            'C': {'effectiveness': 0.85, 'fp_rate': 0.15, 'fn_rate': 0.03, 'processing_time': 75},
            'D': {'effectiveness': 0.75, 'fp_rate': 0.20, 'fn_rate': 0.05, 'processing_time': 100}
        }
        
        self.start_time = None
    
    def analyze_component_performance(self, days: int = 30) -> ComponentBreakdownResult:
        """
        Perform comprehensive component performance analysis
        
        Args:
            days: Number of days to analyze
            
        Returns:
            ComponentBreakdownResult: Complete component analysis
        """
        self.start_time = datetime.now()
        
        try:
            logger.info(f"Analyzing component performance for last {days} days...")
            
            # Load quality gate decisions
            decisions = self._load_quality_decisions(days)
            
            # Group decisions by component type
            component_data = self._group_by_component(decisions)
            
            logger.info(f"Analyzing {len(component_data)} component types with {len(decisions)} total decisions")
            
            # Analyze each component type
            component_metrics = []
            for component_type, component_decisions in component_data.items():
                metrics = self._analyze_component_metrics(component_type, component_decisions)
                component_metrics.append(metrics)
            
            # Analyze quality conditions by component
            condition_analysis = self._analyze_quality_conditions(component_data)
            
            # Compare components
            component_comparison = self._compare_components(component_metrics)
            
            # Generate insights
            insights = self._generate_performance_insights(component_metrics, component_comparison)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(component_metrics, condition_analysis)
            
            # Create result
            result = ComponentBreakdownResult(
                analysis_period=f"{days} days",
                total_components=len(component_metrics),
                component_metrics=sorted(component_metrics, key=lambda x: x.effectiveness_score, reverse=True),
                condition_analysis=condition_analysis,
                component_comparison=component_comparison,
                performance_insights=insights,
                optimization_recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
            # Save results
            self._save_analysis_result(result)
            
            processing_time = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.info(f"Component analysis completed in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in component performance analysis: {str(e)}")
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
        
        # Load from analytics outputs
        analytics_dir = self.metrics_path / "analytics"
        if analytics_dir.exists():
            for analytics_file in analytics_dir.glob("*.json"):
                try:
                    with open(analytics_file, 'r') as f:
                        analytics_data = json.load(f)
                    
                    for decision in analytics_data.get('quality_decisions', []):
                        if decision.get('timestamp'):
                            decision_date = datetime.fromisoformat(decision['timestamp'])
                            if decision_date >= cutoff_date:
                                decisions.append(decision)
                                
                except Exception as e:
                    logger.warning(f"Error loading analytics file {analytics_file}: {str(e)}")
        
        logger.info(f"Loaded {len(decisions)} quality gate decisions")
        return decisions
    
    def _group_by_component(self, decisions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group decisions by component type"""
        component_data = defaultdict(list)
        
        for decision in decisions:
            component_type = decision.get('component_type', 'unknown')
            
            # Normalize component type names
            component_type = self._normalize_component_type(component_type)
            
            component_data[component_type].append(decision)
        
        # Remove components with too few decisions for meaningful analysis
        filtered_data = {k: v for k, v in component_data.items() if len(v) >= 3}
        
        logger.info(f"Grouped into {len(filtered_data)} component types (filtered from {len(component_data)})")
        return filtered_data
    
    def _normalize_component_type(self, component_type: str) -> str:
        """Normalize component type names for consistent grouping"""
        component_type = component_type.lower().strip()
        
        # Mapping common variations
        normalizations = {
            'func': 'function',
            'method': 'function',
            'procedure': 'function',
            'class': 'class',
            'object': 'class',
            'interface': 'interface',
            'module': 'module',
            'package': 'module',
            'file': 'file',
            'script': 'file',
            'test': 'test',
            'spec': 'test',
            'config': 'configuration',
            'configuration': 'configuration',
            'api': 'api',
            'endpoint': 'api',
            'service': 'service',
            'component': 'component',
            'widget': 'component',
            'util': 'utility',
            'utility': 'utility',
            'helper': 'utility'
        }
        
        return normalizations.get(component_type, component_type)
    
    def _analyze_component_metrics(self, component_type: str, decisions: List[Dict[str, Any]]) -> ComponentMetrics:
        """Analyze performance metrics for a specific component type"""
        
        total_decisions = len(decisions)
        
        # Outcome analysis
        true_positives = sum(1 for d in decisions if d.get('outcome') == 'true_positive')
        false_positives = sum(1 for d in decisions if d.get('outcome') == 'false_positive')
        false_negatives = sum(1 for d in decisions if d.get('outcome') == 'false_negative')
        true_negatives = sum(1 for d in decisions if d.get('outcome') == 'true_negative')
        
        # Calculate effectiveness
        if (true_positives + false_positives) > 0:
            effectiveness_score = true_positives / (true_positives + false_positives)
        else:
            effectiveness_score = 1.0 if total_decisions > 0 else 0.0
        
        # Calculate rates
        false_positive_rate = false_positives / total_decisions if total_decisions > 0 else 0.0
        false_negative_rate = false_negatives / total_decisions if total_decisions > 0 else 0.0
        
        # Processing time analysis
        processing_times = [d.get('processing_time_ms', 0) for d in decisions if d.get('processing_time_ms')]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        
        # Success/error rates
        successful_decisions = sum(1 for d in decisions if not d.get('error', False))
        success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0.0
        error_rate = 1.0 - success_rate
        
        # Quality score distribution
        quality_scores = [d.get('quality_score', 0) for d in decisions if d.get('quality_score') is not None]
        quality_distribution = self._calculate_score_distribution(quality_scores)
        
        # Threshold hit rate (decisions that pass quality threshold)
        passed_decisions = sum(1 for d in decisions if d.get('quality_score', 0) >= d.get('threshold', 1))
        threshold_hit_rate = passed_decisions / total_decisions if total_decisions > 0 else 0.0
        
        # Trend analysis
        trend_direction = self._calculate_component_trend(decisions)
        
        # Performance grade
        performance_grade = self._calculate_performance_grade(
            effectiveness_score, false_positive_rate, false_negative_rate, avg_processing_time
        )
        
        return ComponentMetrics(
            component_type=component_type,
            total_decisions=total_decisions,
            effectiveness_score=effectiveness_score,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            avg_processing_time_ms=avg_processing_time,
            success_rate=success_rate,
            error_rate=error_rate,
            quality_score_distribution=quality_distribution,
            threshold_hit_rate=threshold_hit_rate,
            trend_direction=trend_direction,
            performance_grade=performance_grade
        )
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Calculate percentile distribution of quality scores"""
        if not scores:
            return {'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0}
        
        sorted_scores = sorted(scores)
        
        return {
            'p25': statistics.quantiles(sorted_scores, n=4)[0] if len(sorted_scores) > 1 else sorted_scores[0],
            'p50': statistics.median(sorted_scores),
            'p75': statistics.quantiles(sorted_scores, n=4)[2] if len(sorted_scores) > 1 else sorted_scores[-1],
            'p90': sorted_scores[int(len(sorted_scores) * 0.9)] if len(sorted_scores) > 1 else sorted_scores[-1],
            'p95': sorted_scores[int(len(sorted_scores) * 0.95)] if len(sorted_scores) > 1 else sorted_scores[-1]
        }
    
    def _calculate_component_trend(self, decisions: List[Dict[str, Any]]) -> str:
        """Calculate trend direction for component performance"""
        if len(decisions) < 6:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_decisions = sorted(decisions, key=lambda d: d.get('timestamp', ''))
        
        # Split into two halves
        mid_point = len(sorted_decisions) // 2
        older_decisions = sorted_decisions[:mid_point]
        recent_decisions = sorted_decisions[mid_point:]
        
        # Calculate effectiveness for each period
        def calc_effectiveness(decision_set):
            tp = sum(1 for d in decision_set if d.get('outcome') == 'true_positive')
            fp = sum(1 for d in decision_set if d.get('outcome') == 'false_positive')
            return tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        older_effectiveness = calc_effectiveness(older_decisions)
        recent_effectiveness = calc_effectiveness(recent_decisions)
        
        if recent_effectiveness > older_effectiveness * 1.05:  # >5% improvement
            return 'improving'
        elif recent_effectiveness < older_effectiveness * 0.95:  # >5% decline
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_performance_grade(self, effectiveness: float, fp_rate: float, 
                                   fn_rate: float, processing_time: float) -> str:
        """Calculate overall performance grade for component"""
        
        for grade, thresholds in self.grading_thresholds.items():
            if (effectiveness >= thresholds['effectiveness'] and 
                fp_rate <= thresholds['fp_rate'] and
                fn_rate <= thresholds['fn_rate'] and
                processing_time <= thresholds['processing_time']):
                return grade
        
        return 'F'  # Failing grade if all thresholds missed
    
    def _analyze_quality_conditions(self, component_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[QualityConditionAnalysis]]:
        """Analyze quality conditions by component type"""
        condition_analysis = {}
        
        for component_type, decisions in component_data.items():
            component_conditions = []
            
            # Extract all conditions for this component
            condition_data = defaultdict(list)
            
            for decision in decisions:
                conditions = decision.get('quality_conditions', [])
                quality_score = decision.get('quality_score', 0)
                outcome = decision.get('outcome', 'unknown')
                
                for condition in conditions:
                    condition_data[condition].append({
                        'score': quality_score,
                        'outcome': outcome,
                        'decision': decision
                    })
            
            # Analyze each condition
            for condition, condition_decisions in condition_data.items():
                if len(condition_decisions) >= 3:  # Need minimum data
                    
                    frequency = len(condition_decisions)
                    avg_score = statistics.mean([cd['score'] for cd in condition_decisions if cd['score']])
                    
                    # Calculate effectiveness for this condition
                    tp = sum(1 for cd in condition_decisions if cd['outcome'] == 'true_positive')
                    fp = sum(1 for cd in condition_decisions if cd['outcome'] == 'false_positive')
                    effectiveness = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    
                    # Calculate false positive contribution
                    total_fps_component = sum(1 for d in decisions if d.get('outcome') == 'false_positive')
                    condition_fps = fp
                    fp_contribution = condition_fps / total_fps_component if total_fps_component > 0 else 0.0
                    
                    # Generate recommendation
                    recommendation = self._generate_condition_recommendation(
                        condition, effectiveness, fp_contribution, avg_score
                    )
                    
                    analysis = QualityConditionAnalysis(
                        condition_name=condition,
                        frequency=frequency,
                        avg_score=avg_score,
                        effectiveness=effectiveness,
                        false_positive_contribution=fp_contribution,
                        recommendation=recommendation
                    )
                    
                    component_conditions.append(analysis)
            
            # Sort by frequency and effectiveness
            component_conditions.sort(key=lambda c: (c.frequency, c.effectiveness), reverse=True)
            condition_analysis[component_type] = component_conditions[:10]  # Top 10 conditions
        
        return condition_analysis
    
    def _generate_condition_recommendation(self, condition: str, effectiveness: float, 
                                         fp_contribution: float, avg_score: float) -> str:
        """Generate recommendations for specific quality conditions"""
        
        if effectiveness < 0.7:
            return f"Low effectiveness ({effectiveness:.1%}) - review '{condition}' condition logic"
        elif fp_contribution > 0.3:
            return f"High FP contribution ({fp_contribution:.1%}) - consider relaxing '{condition}' thresholds"
        elif avg_score < 0.5:
            return f"Low average scores ({avg_score:.2f}) - '{condition}' may be too strict"
        elif effectiveness > 0.95 and fp_contribution < 0.05:
            return f"Excellent performance - use '{condition}' as best practice template"
        else:
            return f"Performing within acceptable ranges - monitor '{condition}' for changes"
    
    def _compare_components(self, component_metrics: List[ComponentMetrics]) -> ComponentComparison:
        """Compare performance across component types"""
        
        if len(component_metrics) < 2:
            return ComponentComparison(
                compared_components=[m.component_type for m in component_metrics],
                best_performer=component_metrics[0].component_type if component_metrics else 'none',
                worst_performer=component_metrics[0].component_type if component_metrics else 'none',
                performance_gaps={},
                improvement_opportunities=[]
            )
        
        # Sort by effectiveness
        sorted_by_effectiveness = sorted(component_metrics, key=lambda m: m.effectiveness_score, reverse=True)
        best_performer = sorted_by_effectiveness[0]
        worst_performer = sorted_by_effectiveness[-1]
        
        # Calculate performance gaps
        performance_gaps = {
            'effectiveness_gap': best_performer.effectiveness_score - worst_performer.effectiveness_score,
            'fp_rate_gap': worst_performer.false_positive_rate - best_performer.false_positive_rate,
            'fn_rate_gap': worst_performer.false_negative_rate - best_performer.false_negative_rate,
            'processing_time_gap': worst_performer.avg_processing_time_ms - best_performer.avg_processing_time_ms
        }
        
        # Identify improvement opportunities
        improvement_opportunities = []
        
        # Find components with high FP rates
        high_fp_components = [m for m in component_metrics if m.false_positive_rate > 0.15]
        if high_fp_components:
            improvement_opportunities.append(
                f"High FP rates in: {', '.join([m.component_type for m in high_fp_components])}"
            )
        
        # Find components with high processing times
        slow_components = [m for m in component_metrics if m.avg_processing_time_ms > 75]
        if slow_components:
            improvement_opportunities.append(
                f"Performance optimization needed for: {', '.join([m.component_type for m in slow_components])}"
            )
        
        # Find components with low effectiveness
        low_effectiveness = [m for m in component_metrics if m.effectiveness_score < 0.8]
        if low_effectiveness:
            improvement_opportunities.append(
                f"Effectiveness improvements needed for: {', '.join([m.component_type for m in low_effectiveness])}"
            )
        
        return ComponentComparison(
            compared_components=[m.component_type for m in component_metrics],
            best_performer=best_performer.component_type,
            worst_performer=worst_performer.component_type,
            performance_gaps=performance_gaps,
            improvement_opportunities=improvement_opportunities
        )
    
    def _generate_performance_insights(self, component_metrics: List[ComponentMetrics], 
                                     comparison: ComponentComparison) -> List[str]:
        """Generate performance insights from component analysis"""
        insights = []
        
        if not component_metrics:
            return ["No component data available for analysis"]
        
        # Overall performance insights
        avg_effectiveness = statistics.mean([m.effectiveness_score for m in component_metrics])
        insights.append(f"Average component effectiveness: {avg_effectiveness:.1%}")
        
        # Grade distribution
        grade_counts = Counter([m.performance_grade for m in component_metrics])
        top_grade = grade_counts.most_common(1)[0]
        insights.append(f"Most common performance grade: {top_grade[0]} ({top_grade[1]} components)")
        
        # Best and worst performers
        if len(component_metrics) > 1:
            insights.append(f"Best performer: {comparison.best_performer}")
            insights.append(f"Worst performer: {comparison.worst_performer}")
            
            effectiveness_gap = comparison.performance_gaps.get('effectiveness_gap', 0)
            if effectiveness_gap > 0.2:  # >20% gap
                insights.append(f"Significant effectiveness gap ({effectiveness_gap:.1%}) indicates optimization potential")
        
        # Trend insights
        improving_components = [m for m in component_metrics if m.trend_direction == 'improving']
        declining_components = [m for m in component_metrics if m.trend_direction == 'declining']
        
        if improving_components:
            insights.append(f"{len(improving_components)} components showing improvement trends")
        
        if declining_components:
            insights.append(f"ALERT: {len(declining_components)} components showing declining performance")
        
        # Processing time insights
        fast_components = [m for m in component_metrics if m.avg_processing_time_ms < 30]
        if fast_components:
            insights.append(f"{len(fast_components)} components meet performance targets (<30ms)")
        
        # Threshold hit rate insights
        avg_hit_rate = statistics.mean([m.threshold_hit_rate for m in component_metrics])
        if avg_hit_rate < 0.7:
            insights.append(f"Low average threshold hit rate ({avg_hit_rate:.1%}) suggests strict thresholds")
        
        return insights[:8]  # Return top 8 insights
    
    def _generate_optimization_recommendations(self, component_metrics: List[ComponentMetrics], 
                                             condition_analysis: Dict[str, List[QualityConditionAnalysis]]) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        if not component_metrics:
            return ["No data available for recommendations"]
        
        # Component-specific recommendations
        for metrics in component_metrics:
            if metrics.performance_grade in ['D', 'F']:
                recommendations.append(
                    f"URGENT: {metrics.component_type} components need immediate attention "
                    f"(effectiveness: {metrics.effectiveness_score:.1%}, grade: {metrics.performance_grade})"
                )
            elif metrics.false_positive_rate > 0.2:
                recommendations.append(
                    f"High FP rate in {metrics.component_type} components ({metrics.false_positive_rate:.1%}) - "
                    f"review thresholds and conditions"
                )
            elif metrics.avg_processing_time_ms > 100:
                recommendations.append(
                    f"Performance optimization needed for {metrics.component_type} components "
                    f"({metrics.avg_processing_time_ms:.0f}ms average)"
                )
        
        # Condition-based recommendations
        for component_type, conditions in condition_analysis.items():
            high_impact_conditions = [c for c in conditions if c.false_positive_contribution > 0.25]
            if high_impact_conditions:
                condition_names = [c.condition_name for c in high_impact_conditions[:2]]
                recommendations.append(
                    f"Review quality conditions for {component_type}: {', '.join(condition_names)} "
                    f"(high FP contribution)"
                )
        
        # Best practice propagation
        a_grade_components = [m for m in component_metrics if m.performance_grade == 'A']
        if a_grade_components and len(component_metrics) > len(a_grade_components):
            best_component = a_grade_components[0].component_type
            recommendations.append(
                f"Apply {best_component} component patterns to improve other component types"
            )
        
        # Threshold optimization
        low_hit_rate_components = [m for m in component_metrics if m.threshold_hit_rate < 0.5]
        if low_hit_rate_components:
            component_names = [m.component_type for m in low_hit_rate_components[:3]]
            recommendations.append(
                f"Consider threshold relaxation for: {', '.join(component_names)} "
                f"(low pass rates suggest over-strict thresholds)"
            )
        
        # Trend-based recommendations
        declining_components = [m for m in component_metrics if m.trend_direction == 'declining']
        if declining_components:
            component_names = [m.component_type for m in declining_components]
            recommendations.append(
                f"Investigate declining performance in: {', '.join(component_names)} "
                f"(trend analysis shows degradation)"
            )
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _save_analysis_result(self, result: ComponentBreakdownResult):
        """Save analysis results to file system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save current analysis
            current_file = self.analysis_path / "current_component_analysis.json"
            with open(current_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            # Save timestamped version
            historical_file = self.analysis_path / f"component_analysis_{timestamp}.json"
            with open(historical_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            logger.info(f"Component analysis results saved to {current_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def generate_html_report(self, result: ComponentBreakdownResult) -> str:
        """Generate HTML report for component performance analysis"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Component Performance Breakdown</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .report {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; color: #2c3e50; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .components-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .component-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .component-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .component-name {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
        .grade {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; color: white; }}
        .grade-A {{ background: #27ae60; }}
        .grade-B {{ background: #2ecc71; }}
        .grade-C {{ background: #f39c12; }}
        .grade-D {{ background: #e67e22; }}
        .grade-F {{ background: #e74c3c; }}
        .metrics-row {{ display: flex; justify-content: space-between; margin: 8px 0; }}
        .metric-label {{ color: #7f8c8d; }}
        .metric-value {{ font-weight: bold; }}
        .effectiveness-high {{ color: #27ae60; }}
        .effectiveness-medium {{ color: #f39c12; }}
        .effectiveness-low {{ color: #e74c3c; }}
        .trend-improving {{ color: #27ae60; }}
        .trend-declining {{ color: #e74c3c; }}
        .trend-stable {{ color: #3498db; }}
        .comparison-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .insights-section {{ background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; margin-bottom: 20px; }}
        .recommendations-section {{ background: #fff5f5; padding: 20px; border-radius: 8px; border-left: 4px solid #e74c3c; margin-bottom: 20px; }}
        .insight-item, .recommendation-item {{ padding: 8px 0; }}
        .conditions-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .condition-item {{ border-left: 3px solid #3498db; padding: 10px; margin: 8px 0; background: #f8f9fa; }}
        .footer {{ text-align: center; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>Component Performance Breakdown</h1>
            <p>Analysis Period: {result.analysis_period} | Generated: {result.timestamp}</p>
        </div>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <div class="metrics-row">
                <span class="metric-label">Total Component Types:</span>
                <span class="metric-value">{result.total_components}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-label">Best Performer:</span>
                <span class="metric-value">{result.component_comparison.best_performer}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-label">Needs Attention:</span>
                <span class="metric-value">{result.component_comparison.worst_performer}</span>
            </div>
        </div>
        
        <div class="components-grid">
            {self._generate_component_cards_html(result.component_metrics)}
        </div>
        
        <div class="comparison-section">
            <h2>Component Comparison</h2>
            {self._generate_comparison_html(result.component_comparison)}
        </div>
        
        <div class="conditions-section">
            <h2>Quality Conditions Analysis</h2>
            {self._generate_conditions_html(result.condition_analysis)}
        </div>
        
        <div class="insights-section">
            <h2>Performance Insights</h2>
            {''.join([f'<div class="insight-item">• {insight}</div>' for insight in result.performance_insights])}
        </div>
        
        <div class="recommendations-section">
            <h2>Optimization Recommendations</h2>
            {''.join([f'<div class="recommendation-item">• {rec}</div>' for rec in result.optimization_recommendations])}
        </div>
        
        <div class="footer">
            <p>Generated by RIF Component Performance Analyzer</p>
            <p>Issue #94 - Phase 2: Analytics Dashboard Implementation</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_component_cards_html(self, components: List[ComponentMetrics]) -> str:
        """Generate HTML cards for component metrics"""
        html = ""
        
        for component in components:
            effectiveness_class = (
                'effectiveness-high' if component.effectiveness_score >= 0.9 
                else 'effectiveness-medium' if component.effectiveness_score >= 0.8 
                else 'effectiveness-low'
            )
            
            trend_class = f"trend-{component.trend_direction.replace('_', '-')}"
            
            html += f"""
            <div class="component-card">
                <div class="component-header">
                    <div class="component-name">{component.component_type.title()}</div>
                    <div class="grade grade-{component.performance_grade}">{component.performance_grade}</div>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Effectiveness:</span>
                    <span class="metric-value {effectiveness_class}">{component.effectiveness_score:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">False Positive Rate:</span>
                    <span class="metric-value">{component.false_positive_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">False Negative Rate:</span>
                    <span class="metric-value">{component.false_negative_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Avg Processing Time:</span>
                    <span class="metric-value">{component.avg_processing_time_ms:.0f}ms</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{component.success_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Total Decisions:</span>
                    <span class="metric-value">{component.total_decisions}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Trend:</span>
                    <span class="metric-value {trend_class}">{component.trend_direction.replace('_', ' ').title()}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Median Quality Score:</span>
                    <span class="metric-value">{component.quality_score_distribution.get('p50', 0):.2f}</span>
                </div>
            </div>
            """
        
        return html
    
    def _generate_comparison_html(self, comparison: ComponentComparison) -> str:
        """Generate HTML for component comparison"""
        html = f"""
        <div class="metrics-row">
            <span class="metric-label">Components Analyzed:</span>
            <span class="metric-value">{len(comparison.compared_components)}</span>
        </div>
        
        <div class="metrics-row">
            <span class="metric-label">Effectiveness Gap:</span>
            <span class="metric-value">{comparison.performance_gaps.get('effectiveness_gap', 0):.1%}</span>
        </div>
        
        <div class="metrics-row">
            <span class="metric-label">FP Rate Gap:</span>
            <span class="metric-value">{comparison.performance_gaps.get('fp_rate_gap', 0):.1%}</span>
        </div>
        
        <div class="metrics-row">
            <span class="metric-label">Processing Time Gap:</span>
            <span class="metric-value">{comparison.performance_gaps.get('processing_time_gap', 0):.0f}ms</span>
        </div>
        
        <h3>Improvement Opportunities:</h3>
        {''.join([f'<div>• {opp}</div>' for opp in comparison.improvement_opportunities])}
        """
        
        return html
    
    def _generate_conditions_html(self, condition_analysis: Dict[str, List[QualityConditionAnalysis]]) -> str:
        """Generate HTML for quality conditions analysis"""
        if not condition_analysis:
            return "<p>No quality condition data available</p>"
        
        html = ""
        
        for component_type, conditions in condition_analysis.items():
            if conditions:
                html += f"<h3>{component_type.title()} Components</h3>"
                
                for condition in conditions[:5]:  # Top 5 conditions per component
                    effectiveness_class = (
                        'effectiveness-high' if condition.effectiveness >= 0.9
                        else 'effectiveness-medium' if condition.effectiveness >= 0.8
                        else 'effectiveness-low'
                    )
                    
                    html += f"""
                    <div class="condition-item">
                        <strong>{condition.condition_name}</strong>
                        <div style="margin-top: 5px; font-size: 0.9em;">
                            Frequency: {condition.frequency} | 
                            Effectiveness: <span class="{effectiveness_class}">{condition.effectiveness:.1%}</span> | 
                            FP Contribution: {condition.false_positive_contribution:.1%}
                        </div>
                        <div style="margin-top: 5px; font-style: italic; color: #7f8c8d;">
                            {condition.recommendation}
                        </div>
                    </div>
                    """
        
        return html

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Analyze Component Performance Breakdown')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--output', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--knowledge-path', default='/Users/cal/DEV/RIF/knowledge', help='Path to knowledge base')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComponentPerformanceAnalyzer(args.knowledge_path)
    
    # Run analysis
    print(f"Analyzing component performance for last {args.days} days...")
    result = analyzer.analyze_component_performance(args.days)
    
    # Output results
    if args.output in ['json', 'both']:
        print("\n=== COMPONENT ANALYSIS JSON RESULTS ===")
        print(json.dumps(asdict(result), indent=2))
    
    if args.output in ['html', 'both']:
        html_report = analyzer.generate_html_report(result)
        html_file = Path(args.knowledge_path) / "analysis" / "component_performance_report.html"
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"\n=== HTML REPORT GENERATED ===")
        print(f"Report saved to: {html_file}")
        print(f"Open in browser: file://{html_file}")
    
    # Display summary
    print(f"\n=== COMPONENT ANALYSIS SUMMARY ===")
    print(f"Analysis Period: {result.analysis_period}")
    print(f"Total Component Types: {result.total_components}")
    print(f"Best Performer: {result.component_comparison.best_performer}")
    print(f"Worst Performer: {result.component_comparison.worst_performer}")
    
    print(f"\nComponent Grades:")
    grade_counts = Counter([m.performance_grade for m in result.component_metrics])
    for grade, count in sorted(grade_counts.items()):
        print(f"  Grade {grade}: {count} components")
    
    print(f"\nTop Performance Issues:")
    for i, rec in enumerate(result.optimization_recommendations[:3], 1):
        if 'URGENT' in rec or 'High' in rec:
            print(f"  {i}. {rec}")
    
    print(f"\nKey Insights:")
    for i, insight in enumerate(result.performance_insights[:3], 1):
        print(f"  {i}. {insight}")

if __name__ == "__main__":
    main()