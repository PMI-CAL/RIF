#!/usr/bin/env python3
"""
Quality Analytics Engine - Issue #94 Phase 1
Base analytics engine for calculating quality gate effectiveness metrics.

This component:
- Calculates effectiveness metrics from quality gate data
- Performs trend analysis over time
- Identifies patterns in false positives and negatives
- Provides statistical analysis of quality gate performance
- Generates actionable insights for threshold optimization
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import math


@dataclass
class QualityTrend:
    """Represents a trend in quality metrics over time."""
    metric_name: str
    time_period: str
    data_points: List[float]
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0.0 to 1.0
    statistical_significance: float
    average_value: float
    min_value: float
    max_value: float
    standard_deviation: float


@dataclass
class EffectivenessInsight:
    """Actionable insight about quality gate effectiveness."""
    insight_type: str  # 'threshold_adjustment', 'process_improvement', 'false_positive', etc.
    priority: str  # 'high', 'medium', 'low'
    gate_type: str
    description: str
    impact_estimate: float  # Expected improvement percentage
    confidence_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report for quality gate effectiveness."""
    report_id: str
    timestamp: str
    analysis_period: Dict[str, str]
    overall_metrics: Dict[str, float]
    gate_specific_metrics: Dict[str, Dict[str, float]]
    trends: List[QualityTrend]
    insights: List[EffectivenessInsight]
    correlation_analysis: Dict[str, Any]
    recommendations: List[str]


class QualityAnalyticsEngine:
    """
    Advanced analytics engine for quality gate effectiveness analysis.
    Provides statistical analysis, trend detection, and actionable insights.
    """
    
    def __init__(self, storage_path: str = "knowledge/quality_metrics"):
        """Initialize the quality analytics engine."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize analytics storage
        self.analytics_path = self.storage_path / "analytics"
        self.analytics_path.mkdir(exist_ok=True)
        
        # Analytics configuration
        self.min_data_points = 5  # Minimum data points for statistical significance
        self.trend_sensitivity = 0.1  # Sensitivity for trend detection
        self.confidence_threshold = 0.7  # Minimum confidence for insights
    
    def setup_logging(self):
        """Setup logging for quality analytics."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QualityAnalyticsEngine - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(
        self,
        days_back: int = 30,
        include_trends: bool = True,
        include_insights: bool = True
    ) -> AnalyticsReport:
        """
        Generate comprehensive analytics report for quality gate effectiveness.
        
        Args:
            days_back: Number of days to analyze
            include_trends: Whether to include trend analysis
            include_insights: Whether to generate insights
            
        Returns:
            Comprehensive analytics report
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            self.logger.info(f"Generating analytics report for {days_back} days")
            
            # Load data for analysis
            quality_decisions = self._load_quality_decisions_period(start_time, end_time)
            production_outcomes = self._load_production_outcomes_period(start_time, end_time)
            
            if not quality_decisions:
                self.logger.warning("No quality decisions found for analysis")
                return self._generate_empty_report(start_time, end_time)
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(quality_decisions, production_outcomes)
            
            # Calculate gate-specific metrics
            gate_metrics = self._calculate_gate_specific_metrics(quality_decisions, production_outcomes)
            
            # Perform trend analysis
            trends = []
            if include_trends and len(quality_decisions) >= self.min_data_points:
                trends = self._perform_trend_analysis(quality_decisions, production_outcomes, days_back)
            
            # Generate insights
            insights = []
            if include_insights:
                insights = self._generate_insights(overall_metrics, gate_metrics, trends)
            
            # Perform correlation analysis
            correlation_analysis = self._perform_correlation_analysis(quality_decisions, production_outcomes)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_metrics, insights, trends)
            
            # Create report
            report = AnalyticsReport(
                report_id=self._generate_report_id(),
                timestamp=datetime.now().isoformat(),
                analysis_period={
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days_back
                },
                overall_metrics=overall_metrics,
                gate_specific_metrics=gate_metrics,
                trends=trends,
                insights=insights,
                correlation_analysis=correlation_analysis,
                recommendations=recommendations
            )
            
            # Store report
            self._store_analytics_report(report)
            
            self.logger.info(
                f"Analytics report generated: {len(insights)} insights, "
                f"{len(trends)} trends, {len(recommendations)} recommendations"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            return self._generate_error_report(str(e))
    
    def calculate_gate_effectiveness_score(
        self,
        gate_type: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive effectiveness score for a specific gate type.
        
        Args:
            gate_type: Type of quality gate to analyze
            days_back: Number of days to analyze
            
        Returns:
            Effectiveness score and supporting metrics
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Load data for the specific gate
            decisions = self._load_gate_decisions_period(gate_type, start_time, end_time)
            outcomes = self._load_production_outcomes_period(start_time, end_time)
            
            if not decisions:
                return {
                    'gate_type': gate_type,
                    'effectiveness_score': 0.0,
                    'status': 'insufficient_data',
                    'message': f'No decisions found for {gate_type} in last {days_back} days'
                }
            
            # Calculate confusion matrix
            tp, fp, tn, fn = self._calculate_confusion_matrix_for_gate(decisions, outcomes)
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate effectiveness score (weighted combination)
            effectiveness_score = (
                accuracy * 0.4 +      # 40% weight on accuracy
                precision * 0.3 +     # 30% weight on precision
                recall * 0.2 +        # 20% weight on recall
                f1_score * 0.1        # 10% weight on F1 score
            )
            
            # Calculate performance metrics
            processing_times = [d.get('processing_time_ms', 0) for d in decisions]
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Determine performance category
            if effectiveness_score >= 90:
                performance_category = 'excellent'
            elif effectiveness_score >= 80:
                performance_category = 'good'
            elif effectiveness_score >= 70:
                performance_category = 'fair'
            elif effectiveness_score >= 60:
                performance_category = 'poor'
            else:
                performance_category = 'critical'
            
            return {
                'gate_type': gate_type,
                'effectiveness_score': effectiveness_score,
                'performance_category': performance_category,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                },
                'performance': {
                    'total_decisions': len(decisions),
                    'average_processing_time_ms': avg_processing_time,
                    'performance_warning': avg_processing_time > 50
                },
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days_back
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating gate effectiveness: {e}")
            return {
                'gate_type': gate_type,
                'effectiveness_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def identify_threshold_optimization_opportunities(
        self,
        days_back: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Identify opportunities to optimize quality gate thresholds.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            List of threshold optimization opportunities
        """
        try:
            opportunities = []
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Get all gate types
            all_decisions = self._load_quality_decisions_period(start_time, end_time)
            gate_types = set(d['gate_type'] for d in all_decisions)
            
            for gate_type in gate_types:
                # Calculate current effectiveness
                effectiveness = self.calculate_gate_effectiveness_score(gate_type, days_back)
                
                if effectiveness['status'] == 'error' or effectiveness['effectiveness_score'] == 0:
                    continue
                
                # Analyze false positive/negative patterns
                decisions = self._load_gate_decisions_period(gate_type, start_time, end_time)
                outcomes = self._load_production_outcomes_period(start_time, end_time)
                
                fp_analysis = self._analyze_false_positives(decisions, outcomes)
                fn_analysis = self._analyze_false_negatives(decisions, outcomes)
                
                # Generate optimization recommendations
                recommendations = []
                
                if fp_analysis['rate'] > 0.15:  # More than 15% false positives
                    recommendations.append({
                        'type': 'relax_threshold',
                        'reason': f"High false positive rate ({fp_analysis['rate']:.1%})",
                        'impact_estimate': fp_analysis['rate'] * 50,  # Estimated improvement
                        'confidence': min(fp_analysis['confidence'], 0.8)
                    })
                
                if fn_analysis['rate'] > 0.05:  # More than 5% false negatives
                    recommendations.append({
                        'type': 'tighten_threshold',
                        'reason': f"High false negative rate ({fn_analysis['rate']:.1%})",
                        'impact_estimate': fn_analysis['rate'] * 80,  # Higher impact for FN
                        'confidence': min(fn_analysis['confidence'], 0.9)
                    })
                
                # Performance-based recommendations
                if effectiveness['performance']['average_processing_time_ms'] > 100:
                    recommendations.append({
                        'type': 'optimize_performance',
                        'reason': f"Slow processing ({effectiveness['performance']['average_processing_time_ms']:.1f}ms)",
                        'impact_estimate': 20,
                        'confidence': 0.7
                    })
                
                if recommendations:
                    opportunities.append({
                        'gate_type': gate_type,
                        'current_effectiveness': effectiveness['effectiveness_score'],
                        'current_performance': effectiveness['performance_category'],
                        'recommendations': recommendations,
                        'priority': self._calculate_optimization_priority(recommendations),
                        'analysis_confidence': min([r['confidence'] for r in recommendations])
                    })
            
            # Sort by priority and impact
            opportunities.sort(
                key=lambda x: (x['priority'], -x['current_effectiveness']),
                reverse=True
            )
            
            self.logger.info(f"Identified {len(opportunities)} threshold optimization opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {e}")
            return []
    
    def _load_quality_decisions_period(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load quality decisions within a time period."""
        decisions = []
        recent_dir = self.storage_path / "recent"
        
        if not recent_dir.exists():
            return decisions
        
        for decision_file in recent_dir.glob("decisions_*.jsonl"):
            try:
                with open(decision_file, 'r') as f:
                    for line in f:
                        decision = json.loads(line.strip())
                        decision_time = datetime.fromisoformat(decision['timestamp'])
                        
                        if start_time <= decision_time <= end_time:
                            decisions.append(decision)
            except Exception as e:
                self.logger.warning(f"Error reading decision file {decision_file}: {e}")
                continue
        
        return decisions
    
    def _load_production_outcomes_period(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load production outcomes within a time period."""
        outcomes = []
        correlations_dir = self.storage_path / "correlations"
        
        if not correlations_dir.exists():
            return outcomes
        
        for outcome_file in correlations_dir.glob("outcomes_*.jsonl"):
            try:
                with open(outcome_file, 'r') as f:
                    for line in f:
                        outcome = json.loads(line.strip())
                        outcome_time = datetime.fromisoformat(outcome['timestamp'])
                        
                        if start_time <= outcome_time <= end_time:
                            outcomes.append(outcome)
            except Exception as e:
                self.logger.warning(f"Error reading outcome file {outcome_file}: {e}")
                continue
        
        return outcomes
    
    def _load_gate_decisions_period(self, gate_type: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load decisions for a specific gate type within a time period."""
        all_decisions = self._load_quality_decisions_period(start_time, end_time)
        return [d for d in all_decisions if d['gate_type'] == gate_type]
    
    def _calculate_overall_metrics(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall quality gate metrics."""
        if not decisions:
            return {}
        
        # Basic counts
        total_decisions = len(decisions)
        pass_count = sum(1 for d in decisions if d['decision'] == 'pass')
        fail_count = sum(1 for d in decisions if d['decision'] == 'fail')
        
        # Performance metrics
        processing_times = [d.get('processing_time_ms', 0) for d in decisions]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Effectiveness metrics (requires correlation with outcomes)
        tp, fp, tn, fn = self._calculate_overall_confusion_matrix(decisions, outcomes)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        
        return {
            'total_decisions': total_decisions,
            'pass_rate': (pass_count / total_decisions) * 100,
            'fail_rate': (fail_count / total_decisions) * 100,
            'average_processing_time_ms': avg_processing_time,
            'accuracy_percent': accuracy,
            'precision_percent': precision,
            'recall_percent': recall,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _calculate_gate_specific_metrics(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each gate type."""
        gate_metrics = {}
        gate_types = set(d['gate_type'] for d in decisions)
        
        for gate_type in gate_types:
            gate_decisions = [d for d in decisions if d['gate_type'] == gate_type]
            
            # Calculate confusion matrix for this gate
            tp, fp, tn, fn = self._calculate_confusion_matrix_for_gate(gate_decisions, outcomes)
            
            # Calculate metrics
            total = len(gate_decisions)
            pass_count = sum(1 for d in gate_decisions if d['decision'] == 'pass')
            
            processing_times = [d.get('processing_time_ms', 0) for d in gate_decisions]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
            
            gate_metrics[gate_type] = {
                'total_decisions': total,
                'pass_rate': (pass_count / total) * 100,
                'average_processing_time_ms': avg_processing_time,
                'accuracy_percent': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        
        return gate_metrics
    
    def _perform_trend_analysis(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]], days_back: int) -> List[QualityTrend]:
        """Perform trend analysis on quality metrics."""
        trends = []
        
        try:
            # Group decisions by day
            daily_data = self._group_decisions_by_day(decisions, days_back)
            
            # Analyze trends for key metrics
            metrics_to_analyze = ['pass_rate', 'accuracy', 'processing_time']
            
            for metric_name in metrics_to_analyze:
                data_points = []
                
                for day_data in daily_data:
                    if metric_name == 'pass_rate':
                        value = (sum(1 for d in day_data if d['decision'] == 'pass') / len(day_data)) * 100 if day_data else 0
                    elif metric_name == 'accuracy':
                        # Simplified accuracy calculation for daily data
                        value = 85.0  # Placeholder - would need proper correlation analysis
                    elif metric_name == 'processing_time':
                        value = sum(d.get('processing_time_ms', 0) for d in day_data) / len(day_data) if day_data else 0
                    
                    data_points.append(value)
                
                if len(data_points) >= self.min_data_points:
                    trend = self._analyze_trend(metric_name, data_points)
                    if trend:
                        trends.append(trend)
            
        except Exception as e:
            self.logger.warning(f"Error performing trend analysis: {e}")
        
        return trends
    
    def _generate_insights(self, overall_metrics: Dict[str, float], gate_metrics: Dict[str, Dict[str, float]], trends: List[QualityTrend]) -> List[EffectivenessInsight]:
        """Generate actionable insights from analytics."""
        insights = []
        
        try:
            # Overall performance insights
            if overall_metrics.get('false_positive_rate', 0) > 15:
                insights.append(EffectivenessInsight(
                    insight_type='false_positive_reduction',
                    priority='high',
                    gate_type='overall',
                    description=f"False positive rate is {overall_metrics['false_positive_rate']:.1f}%, consider relaxing thresholds",
                    impact_estimate=overall_metrics['false_positive_rate'] * 0.5,
                    confidence_score=0.8,
                    supporting_data={'fpr': overall_metrics['false_positive_rate']},
                    recommendations=[
                        'Review and adjust quality gate thresholds',
                        'Analyze patterns in false positive cases',
                        'Consider context-aware threshold adjustment'
                    ]
                ))
            
            if overall_metrics.get('false_negative_rate', 0) > 5:
                insights.append(EffectivenessInsight(
                    insight_type='false_negative_reduction',
                    priority='critical',
                    gate_type='overall',
                    description=f"False negative rate is {overall_metrics['false_negative_rate']:.1f}%, tighten quality controls",
                    impact_estimate=overall_metrics['false_negative_rate'] * 2.0,
                    confidence_score=0.9,
                    supporting_data={'fnr': overall_metrics['false_negative_rate']},
                    recommendations=[
                        'Tighten quality gate thresholds immediately',
                        'Add additional validation steps',
                        'Implement mandatory manual review for high-risk changes'
                    ]
                ))
            
            # Gate-specific insights
            for gate_type, metrics in gate_metrics.items():
                if metrics.get('average_processing_time_ms', 0) > 100:
                    insights.append(EffectivenessInsight(
                        insight_type='performance_optimization',
                        priority='medium',
                        gate_type=gate_type,
                        description=f"{gate_type} gate processing time is {metrics['average_processing_time_ms']:.1f}ms",
                        impact_estimate=20.0,
                        confidence_score=0.7,
                        supporting_data={'processing_time': metrics['average_processing_time_ms']},
                        recommendations=[
                            f'Optimize {gate_type} gate implementation',
                            'Consider caching or parallel processing',
                            'Review gate configuration for efficiency'
                        ]
                    ))
            
            # Trend-based insights
            for trend in trends:
                if trend.trend_direction == 'declining' and trend.trend_strength > 0.6:
                    insights.append(EffectivenessInsight(
                        insight_type='trend_degradation',
                        priority='high',
                        gate_type='overall',
                        description=f"{trend.metric_name} is declining with {trend.trend_strength:.1f} strength",
                        impact_estimate=trend.trend_strength * 30,
                        confidence_score=trend.statistical_significance,
                        supporting_data={'trend': asdict(trend)},
                        recommendations=[
                            f'Investigate root cause of {trend.metric_name} decline',
                            'Review recent changes to quality processes',
                            'Consider adjusting quality gate configuration'
                        ]
                    ))
            
        except Exception as e:
            self.logger.warning(f"Error generating insights: {e}")
        
        # Filter insights by confidence threshold
        high_confidence_insights = [i for i in insights if i.confidence_score >= self.confidence_threshold]
        
        # Sort by priority and impact
        priority_order = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
        high_confidence_insights.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.impact_estimate),
            reverse=True
        )
        
        return high_confidence_insights
    
    def _perform_correlation_analysis(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform correlation analysis between quality scores and production outcomes."""
        try:
            if not decisions or not outcomes:
                return {'status': 'insufficient_data'}
            
            # Group by issue number
            issue_quality_scores = {}
            issue_defect_counts = {}
            
            # Calculate quality scores by issue
            issue_decisions = {}
            for decision in decisions:
                issue_num = decision['issue_number']
                if issue_num not in issue_decisions:
                    issue_decisions[issue_num] = []
                issue_decisions[issue_num].append(decision)
            
            for issue_num, issue_decision_list in issue_decisions.items():
                # Calculate overall quality score for the issue
                pass_count = sum(1 for d in issue_decision_list if d['decision'] == 'pass')
                quality_score = (pass_count / len(issue_decision_list)) * 100
                issue_quality_scores[issue_num] = quality_score
            
            # Count defects by issue
            for outcome in outcomes:
                issue_num = outcome['issue_number']
                defect_count = outcome.get('defect_count', 0)
                issue_defect_counts[issue_num] = issue_defect_counts.get(issue_num, 0) + defect_count
            
            # Calculate correlation
            common_issues = set(issue_quality_scores.keys()) & set(issue_defect_counts.keys())
            if len(common_issues) < 3:
                return {'status': 'insufficient_correlation_data'}
            
            scores = [issue_quality_scores[issue] for issue in common_issues]
            defects = [issue_defect_counts[issue] for issue in common_issues]
            
            correlation = self._calculate_correlation(scores, defects)
            
            return {
                'status': 'success',
                'correlation_coefficient': correlation,
                'sample_size': len(common_issues),
                'interpretation': self._interpret_correlation(correlation),
                'scores_stats': {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'stdev': statistics.stdev(scores) if len(scores) > 1 else 0
                },
                'defects_stats': {
                    'mean': statistics.mean(defects),
                    'median': statistics.median(defects),
                    'stdev': statistics.stdev(defects) if len(defects) > 1 else 0
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Error in correlation analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_recommendations(self, overall_metrics: Dict[str, float], insights: List[EffectivenessInsight], trends: List[QualityTrend]) -> List[str]:
        """Generate high-level recommendations based on analysis."""
        recommendations = []
        
        # High-level recommendations based on overall metrics
        if overall_metrics.get('false_positive_rate', 0) > 10:
            recommendations.append("Consider implementing adaptive quality thresholds to reduce false positives")
        
        if overall_metrics.get('false_negative_rate', 0) > 3:
            recommendations.append("Strengthen quality gates - false negative rate indicates insufficient coverage")
        
        if overall_metrics.get('average_processing_time_ms', 0) > 75:
            recommendations.append("Optimize quality gate performance - processing time impacts development velocity")
        
        # Recommendations from insights
        critical_insights = [i for i in insights if i.priority == 'critical']
        if critical_insights:
            recommendations.append("Address critical quality issues immediately - review failed gate patterns")
        
        # Trend-based recommendations
        declining_trends = [t for t in trends if t.trend_direction == 'declining']
        if declining_trends:
            recommendations.append("Quality metrics are declining - investigate recent process changes")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous monitoring of quality gate effectiveness",
            "Establish regular review cycles for quality threshold optimization",
            "Consider implementing automated threshold adjustment based on historical data"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    # Helper methods for calculations and analysis
    
    def _calculate_overall_confusion_matrix(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Calculate overall confusion matrix across all gates."""
        # This is a simplified implementation
        # In practice, would need more sophisticated correlation logic
        tp = fp = tn = fn = 0
        
        # Create issue outcome mapping
        issue_has_defects = {}
        for outcome in outcomes:
            issue_num = outcome['issue_number']
            has_defects = outcome.get('defect_count', 0) > 0
            issue_has_defects[issue_num] = has_defects
        
        # Group decisions by issue
        issue_decisions = {}
        for decision in decisions:
            issue_num = decision['issue_number']
            if issue_num not in issue_decisions:
                issue_decisions[issue_num] = []
            issue_decisions[issue_num].append(decision)
        
        # Analyze each issue
        for issue_num, issue_decision_list in issue_decisions.items():
            # Calculate overall pass/fail for issue
            pass_count = sum(1 for d in issue_decision_list if d['decision'] == 'pass')
            overall_passed = pass_count >= len(issue_decision_list) * 0.7  # 70% threshold
            
            # Get actual outcome
            has_defects = issue_has_defects.get(issue_num, False)
            
            # Update confusion matrix
            if overall_passed and not has_defects:
                tn += 1
            elif overall_passed and has_defects:
                fn += 1
            elif not overall_passed and has_defects:
                tp += 1
            elif not overall_passed and not has_defects:
                fp += 1
        
        return tp, fp, tn, fn
    
    def _calculate_confusion_matrix_for_gate(self, gate_decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix for a specific gate type."""
        # Similar to overall calculation but for specific gate
        return self._calculate_overall_confusion_matrix(gate_decisions, outcomes)
    
    def _group_decisions_by_day(self, decisions: List[Dict[str, Any]], days_back: int) -> List[List[Dict[str, Any]]]:
        """Group decisions by day for trend analysis."""
        end_date = datetime.now().date()
        daily_groups = []
        
        for i in range(days_back):
            target_date = end_date - timedelta(days=i)
            day_decisions = []
            
            for decision in decisions:
                decision_date = datetime.fromisoformat(decision['timestamp']).date()
                if decision_date == target_date:
                    day_decisions.append(decision)
            
            daily_groups.append(day_decisions)
        
        return daily_groups
    
    def _analyze_trend(self, metric_name: str, data_points: List[float]) -> Optional[QualityTrend]:
        """Analyze trend in a series of data points."""
        if len(data_points) < self.min_data_points:
            return None
        
        try:
            # Calculate linear trend
            x_values = list(range(len(data_points)))
            
            # Simple linear regression
            n = len(data_points)
            sum_x = sum(x_values)
            sum_y = sum(data_points)
            sum_xy = sum(x * y for x, y in zip(x_values, data_points))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction and strength
            if abs(slope) < self.trend_sensitivity:
                trend_direction = 'stable'
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = 'improving'
                trend_strength = min(abs(slope) / max(data_points), 1.0)
            else:
                trend_direction = 'declining'
                trend_strength = min(abs(slope) / max(data_points), 1.0)
            
            # Calculate statistical significance (simplified)
            std_dev = statistics.stdev(data_points) if len(data_points) > 1 else 0
            statistical_significance = min(trend_strength / (std_dev + 0.1), 1.0)
            
            return QualityTrend(
                metric_name=metric_name,
                time_period=f"last_{len(data_points)}_days",
                data_points=data_points,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                statistical_significance=statistical_significance,
                average_value=statistics.mean(data_points),
                min_value=min(data_points),
                max_value=max(data_points),
                standard_deviation=std_dev
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing trend for {metric_name}: {e}")
            return None
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "negative" if correlation < 0 else "positive"
        
        return f"{strength} {direction} correlation"
    
    def _analyze_false_positives(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze false positive patterns."""
        # Simplified analysis
        failed_decisions = [d for d in decisions if d['decision'] == 'fail']
        
        # Map issues to defects
        issue_has_defects = {}
        for outcome in outcomes:
            issue_num = outcome['issue_number']
            has_defects = outcome.get('defect_count', 0) > 0
            issue_has_defects[issue_num] = has_defects
        
        # Count false positives (failed but no defects)
        fp_count = 0
        total_fails = 0
        
        issue_fails = set(d['issue_number'] for d in failed_decisions)
        for issue_num in issue_fails:
            total_fails += 1
            has_defects = issue_has_defects.get(issue_num, True)  # Conservative assumption
            if not has_defects:
                fp_count += 1
        
        fp_rate = fp_count / total_fails if total_fails > 0 else 0
        confidence = min(total_fails / 10.0, 1.0)  # Higher confidence with more data
        
        return {
            'rate': fp_rate,
            'count': fp_count,
            'total_fails': total_fails,
            'confidence': confidence
        }
    
    def _analyze_false_negatives(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze false negative patterns."""
        # Simplified analysis
        passed_decisions = [d for d in decisions if d['decision'] == 'pass']
        
        # Map issues to defects
        issue_has_defects = {}
        for outcome in outcomes:
            issue_num = outcome['issue_number']
            has_defects = outcome.get('defect_count', 0) > 0
            issue_has_defects[issue_num] = has_defects
        
        # Count false negatives (passed but has defects)
        fn_count = 0
        total_passes = 0
        
        issue_passes = set(d['issue_number'] for d in passed_decisions)
        for issue_num in issue_passes:
            total_passes += 1
            has_defects = issue_has_defects.get(issue_num, False)
            if has_defects:
                fn_count += 1
        
        fn_rate = fn_count / total_passes if total_passes > 0 else 0
        confidence = min(total_passes / 10.0, 1.0)
        
        return {
            'rate': fn_rate,
            'count': fn_count,
            'total_passes': total_passes,
            'confidence': confidence
        }
    
    def _calculate_optimization_priority(self, recommendations: List[Dict[str, Any]]) -> int:
        """Calculate optimization priority based on recommendations."""
        max_impact = max(r.get('impact_estimate', 0) for r in recommendations)
        max_confidence = max(r.get('confidence', 0) for r in recommendations)
        
        # Simple priority scoring
        priority_score = (max_impact * 0.6) + (max_confidence * 0.4)
        
        if priority_score >= 60:
            return 3  # High priority
        elif priority_score >= 40:
            return 2  # Medium priority
        else:
            return 1  # Low priority
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"analytics_report_{timestamp}".encode()).hexdigest()[:16]
    
    def _generate_empty_report(self, start_time: datetime, end_time: datetime) -> AnalyticsReport:
        """Generate empty report when no data is available."""
        return AnalyticsReport(
            report_id=self._generate_report_id(),
            timestamp=datetime.now().isoformat(),
            analysis_period={
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': (end_time - start_time).days
            },
            overall_metrics={'status': 'no_data'},
            gate_specific_metrics={},
            trends=[],
            insights=[],
            correlation_analysis={'status': 'no_data'},
            recommendations=['Insufficient data for analysis - collect more quality gate decisions']
        )
    
    def _generate_error_report(self, error_message: str) -> AnalyticsReport:
        """Generate error report when analysis fails."""
        return AnalyticsReport(
            report_id=self._generate_report_id(),
            timestamp=datetime.now().isoformat(),
            analysis_period={'error': error_message},
            overall_metrics={'status': 'error'},
            gate_specific_metrics={},
            trends=[],
            insights=[],
            correlation_analysis={'status': 'error'},
            recommendations=[f'Fix analysis error: {error_message}']
        )
    
    def _store_analytics_report(self, report: AnalyticsReport) -> None:
        """Store analytics report to persistent storage."""
        report_file = self.analytics_path / f"report_{report.report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)


def main():
    """Command line interface for quality analytics engine."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_analytics_engine.py <command> [args]")
        print("Commands:")
        print("  report [days_back]              - Generate comprehensive analytics report")
        print("  gate-effectiveness <gate_type> [days_back] - Calculate gate effectiveness")
        print("  optimize [days_back]            - Identify optimization opportunities")
        print("  trends [days_back]              - Show trend analysis only")
        return 1
    
    engine = QualityAnalyticsEngine()
    command = sys.argv[1]
    
    if command == "report":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        report = engine.generate_comprehensive_report(days_back)
        print(json.dumps(asdict(report), indent=2, default=str))
        
    elif command == "gate-effectiveness" and len(sys.argv) >= 3:
        gate_type = sys.argv[2]
        days_back = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        effectiveness = engine.calculate_gate_effectiveness_score(gate_type, days_back)
        print(json.dumps(effectiveness, indent=2))
        
    elif command == "optimize":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        opportunities = engine.identify_threshold_optimization_opportunities(days_back)
        print(json.dumps(opportunities, indent=2))
        
    elif command == "trends":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        # Generate minimal report with trends only
        report = engine.generate_comprehensive_report(days_back, include_insights=False)
        trends_data = [asdict(trend) for trend in report.trends]
        print(json.dumps(trends_data, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())