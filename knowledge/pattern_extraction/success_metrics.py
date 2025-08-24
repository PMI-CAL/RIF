"""
Success Metrics Calculator - Statistical analysis for pattern success rates.

This module calculates comprehensive success metrics for extracted patterns,
including success rates, applicability scores, confidence intervals,
reusability indices, and evolution tracking.
"""

import json
import logging
import statistics
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from .discovery_engine import ExtractedPattern


@dataclass
class SuccessMetrics:
    """Comprehensive success metrics for a pattern."""
    pattern_id: str
    success_rate: float
    confidence_interval: Tuple[float, float]
    applicability_score: float
    reusability_index: float
    evolution_score: float
    reliability_score: float
    adoption_rate: float
    performance_impact: float
    sample_size: int
    calculation_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['confidence_interval'] = list(data['confidence_interval'])
        data['calculation_date'] = data['calculation_date'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuccessMetrics':
        """Create from dictionary."""
        data['confidence_interval'] = tuple(data['confidence_interval'])
        data['calculation_date'] = datetime.fromisoformat(data['calculation_date'])
        return cls(**data)


@dataclass
class PatternApplication:
    """Record of a pattern application instance."""
    pattern_id: str
    application_id: str
    context: Dict[str, Any]
    success: bool
    performance_metrics: Dict[str, float]
    timestamp: datetime
    feedback_score: Optional[float] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class PatternEvolutionData:
    """Data tracking pattern evolution over time."""
    pattern_id: str
    version_history: List[Dict[str, Any]]
    success_trend: List[float]
    usage_trend: List[int]
    improvement_indicators: List[str]
    adaptation_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SuccessMetricsCalculator:
    """
    Calculates comprehensive success metrics for patterns.
    
    This calculator provides multi-dimensional analysis of pattern effectiveness,
    including statistical confidence measures, applicability assessment,
    and evolution tracking over time.
    """
    
    def __init__(self, knowledge_system=None):
        self.knowledge = knowledge_system
        self.logger = logging.getLogger(__name__)
        
        # Statistical confidence thresholds
        self.confidence_levels = {
            'high': 0.95,
            'medium': 0.80,
            'low': 0.65
        }
        
        # Minimum sample sizes for reliable statistics
        self.min_sample_sizes = {
            'success_rate': 5,
            'applicability_score': 3,
            'reusability_index': 2,
            'evolution_score': 4
        }
        
        # Weighting factors for composite metrics
        self.metric_weights = {
            'success_rate': 0.30,
            'applicability_score': 0.25,
            'reusability_index': 0.20,
            'evolution_score': 0.15,
            'performance_impact': 0.10
        }
        
        # Pattern application history (would be loaded from knowledge base)
        self.application_history: Dict[str, List[PatternApplication]] = defaultdict(list)
        
        # Pattern evolution tracking
        self.evolution_data: Dict[str, PatternEvolutionData] = {}
    
    def calculate_pattern_metrics(self, pattern: ExtractedPattern, 
                                application_data: Optional[List[PatternApplication]] = None) -> SuccessMetrics:
        """
        Calculate comprehensive success metrics for a pattern.
        
        Args:
            pattern: The pattern to calculate metrics for
            application_data: Optional list of application instances
            
        Returns:
            SuccessMetrics object with calculated values
        """
        try:
            pattern_id = pattern.signature.combined_hash
            
            # Get application data
            if application_data is None:
                application_data = self._load_application_data(pattern_id)
            
            # Calculate individual metrics
            success_rate, confidence_interval = self._calculate_success_rate(application_data)
            applicability_score = self._calculate_applicability_score(pattern, application_data)
            reusability_index = self._calculate_reusability_index(pattern, application_data)
            evolution_score = self._calculate_evolution_score(pattern_id)
            reliability_score = self._calculate_reliability_score(application_data)
            adoption_rate = self._calculate_adoption_rate(pattern_id, application_data)
            performance_impact = self._calculate_performance_impact(application_data)
            
            metrics = SuccessMetrics(
                pattern_id=pattern_id,
                success_rate=success_rate,
                confidence_interval=confidence_interval,
                applicability_score=applicability_score,
                reusability_index=reusability_index,
                evolution_score=evolution_score,
                reliability_score=reliability_score,
                adoption_rate=adoption_rate,
                performance_impact=performance_impact,
                sample_size=len(application_data),
                calculation_date=datetime.now()
            )
            
            self.logger.info(
                f"Calculated metrics for pattern {pattern_id}: "
                f"success_rate={success_rate:.3f}, applicability={applicability_score:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern metrics: {e}")
            return self._create_default_metrics(pattern.signature.combined_hash)
    
    def calculate_batch_metrics(self, patterns: List[ExtractedPattern]) -> Dict[str, SuccessMetrics]:
        """Calculate metrics for multiple patterns efficiently."""
        metrics_batch = {}
        
        try:
            # Load all application data at once for efficiency
            all_application_data = self._load_batch_application_data(
                [p.signature.combined_hash for p in patterns]
            )
            
            for pattern in patterns:
                pattern_id = pattern.signature.combined_hash
                application_data = all_application_data.get(pattern_id, [])
                
                metrics = self.calculate_pattern_metrics(pattern, application_data)
                metrics_batch[pattern_id] = metrics
            
            self.logger.info(f"Calculated batch metrics for {len(patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error in batch metrics calculation: {e}")
        
        return metrics_batch
    
    def _calculate_success_rate(self, applications: List[PatternApplication]) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate success rate with confidence interval.
        
        Returns:
            Tuple of (success_rate, confidence_interval)
        """
        if not applications:
            return 0.5, (0.0, 1.0)  # Default with maximum uncertainty
        
        successes = sum(1 for app in applications if app.success)
        total = len(applications)
        success_rate = successes / total
        
        # Calculate confidence interval using Wilson score interval
        confidence_interval = self._wilson_confidence_interval(successes, total, 0.95)
        
        return success_rate, confidence_interval
    
    def _wilson_confidence_interval(self, successes: int, total: int, confidence: float) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if total == 0:
            return (0.0, 1.0)
        
        z = self._get_z_score(confidence)
        p = successes / total
        n = total
        
        center = p + (z * z) / (2 * n)
        spread = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
        denominator = 1 + (z * z) / n
        
        lower = (center - spread) / denominator
        upper = (center + spread) / denominator
        
        return (max(0.0, lower), min(1.0, upper))
    
    def _get_z_score(self, confidence: float) -> float:
        """Get z-score for confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return z_scores.get(confidence, 1.960)  # Default to 95%
    
    def _calculate_applicability_score(self, pattern: ExtractedPattern, 
                                     applications: List[PatternApplication]) -> float:
        """
        Calculate applicability score based on successful contexts.
        
        Measures how broadly applicable a pattern is across different contexts.
        """
        if not applications:
            return 0.5  # Default moderate applicability
        
        # Analyze context diversity
        contexts = [app.context for app in applications if app.success]
        if not contexts:
            return 0.0
        
        # Calculate context diversity metrics
        diversity_score = self._calculate_context_diversity(contexts)
        success_consistency = self._calculate_success_consistency(applications)
        cross_domain_applicability = self._calculate_cross_domain_applicability(contexts)
        
        # Weighted combination
        applicability_score = (
            diversity_score * 0.4 +
            success_consistency * 0.4 +
            cross_domain_applicability * 0.2
        )
        
        return min(1.0, max(0.0, applicability_score))
    
    def _calculate_context_diversity(self, contexts: List[Dict[str, Any]]) -> float:
        """Calculate diversity of successful application contexts."""
        if not contexts:
            return 0.0
        
        # Collect context dimensions
        dimensions = defaultdict(set)
        for context in contexts:
            for key, value in context.items():
                if isinstance(value, (str, int, float)):
                    dimensions[key].add(str(value))
        
        # Calculate diversity for each dimension
        diversity_scores = []
        for dimension, values in dimensions.items():
            unique_values = len(values)
            total_contexts = len(contexts)
            
            # Normalize by theoretical maximum diversity
            diversity = min(1.0, unique_values / min(total_contexts, 5))  # Cap at 5 for diminishing returns
            diversity_scores.append(diversity)
        
        return statistics.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_success_consistency(self, applications: List[PatternApplication]) -> float:
        """Calculate consistency of success across different applications."""
        if not applications:
            return 0.0
        
        # Group by context similarity
        context_groups = self._group_by_context_similarity(applications)
        
        group_success_rates = []
        for group in context_groups:
            if len(group) >= 2:  # Only consider groups with multiple applications
                successes = sum(1 for app in group if app.success)
                group_success_rate = successes / len(group)
                group_success_rates.append(group_success_rate)
        
        if not group_success_rates:
            # Fall back to overall consistency measure
            successes = [1 if app.success else 0 for app in applications]
            return 1.0 - (statistics.stdev(successes) if len(successes) > 1 else 0.0)
        
        # Calculate consistency as inverse of variance in success rates
        if len(group_success_rates) == 1:
            return 1.0
        
        variance = statistics.variance(group_success_rates)
        return max(0.0, 1.0 - variance * 4)  # Scale variance for 0-1 range
    
    def _calculate_cross_domain_applicability(self, contexts: List[Dict[str, Any]]) -> float:
        """Calculate cross-domain applicability."""
        if not contexts:
            return 0.0
        
        # Extract domains from contexts
        domains = set()
        for context in contexts:
            domain = context.get('domain', 'unknown')
            if domain and domain != 'unknown':
                domains.add(domain)
        
        # Score based on number of different domains
        domain_count = len(domains)
        if domain_count <= 1:
            return 0.3  # Limited to single domain
        elif domain_count <= 3:
            return 0.6  # Good cross-domain applicability
        else:
            return 1.0  # Excellent cross-domain applicability
    
    def _group_by_context_similarity(self, applications: List[PatternApplication]) -> List[List[PatternApplication]]:
        """Group applications by context similarity."""
        # Simple grouping by domain for now
        # Could be enhanced with more sophisticated similarity measures
        
        domain_groups = defaultdict(list)
        for app in applications:
            domain = app.context.get('domain', 'unknown')
            domain_groups[domain].append(app)
        
        return list(domain_groups.values())
    
    def _calculate_reusability_index(self, pattern: ExtractedPattern,
                                   applications: List[PatternApplication]) -> float:
        """
        Calculate reusability index based on cross-project usage.
        
        Measures how well a pattern transfers across different projects and contexts.
        """
        if not applications:
            return 0.5  # Default moderate reusability
        
        # Analyze project diversity
        projects = set(app.context.get('project_id', 'unknown') for app in applications)
        project_diversity = min(1.0, len(projects) / max(len(applications), 5))
        
        # Analyze adaptation requirements
        adaptation_score = self._calculate_adaptation_score(applications)
        
        # Analyze technology independence
        tech_independence = self._calculate_technology_independence(applications)
        
        # Analyze size scalability
        size_scalability = self._calculate_size_scalability(applications)
        
        # Weighted combination
        reusability_index = (
            project_diversity * 0.3 +
            adaptation_score * 0.3 +
            tech_independence * 0.2 +
            size_scalability * 0.2
        )
        
        return min(1.0, max(0.0, reusability_index))
    
    def _calculate_adaptation_score(self, applications: List[PatternApplication]) -> float:
        """Calculate how easily a pattern adapts to different contexts."""
        if not applications:
            return 0.5
        
        # Look for adaptation indicators in application notes
        adaptation_indicators = ['adapted', 'modified', 'customized', 'adjusted']
        
        adaptations = 0
        for app in applications:
            notes = app.notes or ''
            if any(indicator in notes.lower() for indicator in adaptation_indicators):
                adaptations += 1
        
        # Higher adaptation rate might indicate flexibility, but also complexity
        adaptation_rate = adaptations / len(applications)
        
        # Optimal adaptation rate is moderate (shows flexibility without excessive complexity)
        if adaptation_rate < 0.2:
            return 0.7  # Low adaptation needed - good reusability
        elif adaptation_rate < 0.6:
            return 1.0  # Moderate adaptation - excellent flexibility
        else:
            return 0.5  # High adaptation needed - may be too complex
    
    def _calculate_technology_independence(self, applications: List[PatternApplication]) -> float:
        """Calculate technology independence score."""
        if not applications:
            return 0.5
        
        # Extract technology stack information from contexts
        technologies = set()
        for app in applications:
            tech_stack = app.context.get('technology_stack', [])
            if isinstance(tech_stack, list):
                technologies.update(tech_stack)
            elif isinstance(tech_stack, str):
                technologies.add(tech_stack)
        
        # More diverse technology usage indicates better independence
        tech_diversity = min(1.0, len(technologies) / 5)  # Normalize to max 5 technologies
        
        return tech_diversity
    
    def _calculate_size_scalability(self, applications: List[PatternApplication]) -> float:
        """Calculate scalability across different project sizes."""
        if not applications:
            return 0.5
        
        # Extract project size indicators
        sizes = []
        for app in applications:
            size_indicator = app.context.get('project_size', 'unknown')
            if size_indicator in ['small', 'medium', 'large', 'enterprise']:
                sizes.append(size_indicator)
        
        if not sizes:
            return 0.5
        
        # Score based on size diversity
        unique_sizes = len(set(sizes))
        if unique_sizes >= 3:
            return 1.0
        elif unique_sizes == 2:
            return 0.7
        else:
            return 0.4
    
    def _calculate_evolution_score(self, pattern_id: str) -> float:
        """Calculate pattern evolution score showing improvement over time."""
        evolution_data = self.evolution_data.get(pattern_id)
        if not evolution_data or len(evolution_data.success_trend) < 2:
            return 0.5  # Default neutral evolution
        
        trend = evolution_data.success_trend
        
        # Calculate trend slope
        x = list(range(len(trend)))
        slope = self._calculate_trend_slope(x, trend)
        
        # Normalize slope to 0-1 range
        # Positive slope (improvement) -> higher score
        # Negative slope (degradation) -> lower score
        evolution_score = 0.5 + (slope * 2)  # Scale slope for reasonable range
        
        return min(1.0, max(0.0, evolution_score))
    
    def _calculate_trend_slope(self, x: List[int], y: List[float]) -> float:
        """Calculate linear trend slope using least squares."""
        n = len(x)
        if n < 2:
            return 0.0
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_reliability_score(self, applications: List[PatternApplication]) -> float:
        """Calculate reliability score based on consistent performance."""
        if not applications:
            return 0.5
        
        # Analyze performance consistency
        performance_scores = []
        for app in applications:
            if app.feedback_score is not None:
                performance_scores.append(app.feedback_score)
        
        if not performance_scores:
            # Use success rate as proxy
            success_rate = sum(1 for app in applications if app.success) / len(applications)
            return success_rate
        
        # Calculate reliability as inverse of performance variance
        mean_performance = statistics.mean(performance_scores)
        
        if len(performance_scores) == 1:
            return mean_performance
        
        variance = statistics.variance(performance_scores)
        consistency = max(0.0, 1.0 - variance / max(mean_performance, 0.1))
        
        # Combine mean performance and consistency
        reliability = (mean_performance * 0.6) + (consistency * 0.4)
        
        return min(1.0, max(0.0, reliability))
    
    def _calculate_adoption_rate(self, pattern_id: str, applications: List[PatternApplication]) -> float:
        """Calculate adoption rate showing how frequently pattern is used."""
        if not applications:
            return 0.0
        
        # Calculate adoption rate over time
        now = datetime.now()
        recent_applications = [
            app for app in applications
            if (now - app.timestamp).days <= 90  # Last 90 days
        ]
        
        if not recent_applications:
            return 0.1  # Low recent adoption
        
        # Normalize by total time period and expected usage
        days_in_period = 90
        adoption_rate = len(recent_applications) / days_in_period
        
        # Normalize to reasonable scale (0-1)
        return min(1.0, adoption_rate * 30)  # Scale assuming max ~1 usage per 3 days
    
    def _calculate_performance_impact(self, applications: List[PatternApplication]) -> float:
        """Calculate performance impact of pattern usage."""
        if not applications:
            return 0.5
        
        # Extract performance metrics
        performance_improvements = []
        for app in applications:
            metrics = app.performance_metrics
            
            # Look for common performance indicators
            for metric_name in ['execution_time', 'memory_usage', 'throughput', 'error_rate']:
                if metric_name in metrics:
                    # Assume positive values are improvements for most metrics
                    # (except error_rate where negative is better)
                    value = metrics[metric_name]
                    if metric_name == 'error_rate':
                        value = -value  # Invert error rate
                    performance_improvements.append(value)
        
        if not performance_improvements:
            return 0.5  # Neutral impact
        
        # Calculate average performance impact
        avg_impact = statistics.mean(performance_improvements)
        
        # Normalize to 0-1 range (assuming impacts are typically between -1 and 1)
        normalized_impact = (avg_impact + 1) / 2
        
        return min(1.0, max(0.0, normalized_impact))
    
    def _load_application_data(self, pattern_id: str) -> List[PatternApplication]:
        """Load application data for a pattern from knowledge base."""
        # This would normally query the knowledge base
        # For now, return cached data or generate sample data
        
        if pattern_id in self.application_history:
            return self.application_history[pattern_id]
        
        # Generate sample application data for demonstration
        return self._generate_sample_application_data(pattern_id)
    
    def _load_batch_application_data(self, pattern_ids: List[str]) -> Dict[str, List[PatternApplication]]:
        """Load application data for multiple patterns efficiently."""
        batch_data = {}
        for pattern_id in pattern_ids:
            batch_data[pattern_id] = self._load_application_data(pattern_id)
        return batch_data
    
    def _generate_sample_application_data(self, pattern_id: str) -> List[PatternApplication]:
        """Generate sample application data for testing."""
        # This would be removed in production - data would come from knowledge base
        sample_data = []
        
        # Generate 3-8 sample applications
        import random
        num_applications = random.randint(3, 8)
        
        domains = ['web', 'api', 'data', 'ml', 'security', 'infrastructure']
        project_sizes = ['small', 'medium', 'large']
        technologies = ['python', 'javascript', 'java', 'go', 'rust']
        
        for i in range(num_applications):
            success = random.random() > 0.2  # 80% success rate
            
            application = PatternApplication(
                pattern_id=pattern_id,
                application_id=f"{pattern_id}_app_{i}",
                context={
                    'domain': random.choice(domains),
                    'project_size': random.choice(project_sizes),
                    'technology_stack': [random.choice(technologies)],
                    'project_id': f"project_{random.randint(1, 10)}"
                },
                success=success,
                performance_metrics={
                    'execution_time': random.uniform(-0.5, 1.0),
                    'memory_usage': random.uniform(-0.3, 0.8),
                    'error_rate': random.uniform(-0.9, 0.1)
                },
                timestamp=datetime.now() - timedelta(days=random.randint(1, 180)),
                feedback_score=random.uniform(0.3, 1.0) if success else random.uniform(0.0, 0.6),
                notes=f"Application {i} notes"
            )
            sample_data.append(application)
        
        return sample_data
    
    def _create_default_metrics(self, pattern_id: str) -> SuccessMetrics:
        """Create default metrics when calculation fails."""
        return SuccessMetrics(
            pattern_id=pattern_id,
            success_rate=0.5,
            confidence_interval=(0.0, 1.0),
            applicability_score=0.5,
            reusability_index=0.5,
            evolution_score=0.5,
            reliability_score=0.5,
            adoption_rate=0.1,
            performance_impact=0.5,
            sample_size=0,
            calculation_date=datetime.now()
        )
    
    def update_pattern_application(self, pattern_id: str, application: PatternApplication):
        """Update pattern application history."""
        self.application_history[pattern_id].append(application)
        
        # Trigger metrics recalculation if significant number of new applications
        if len(self.application_history[pattern_id]) % 5 == 0:
            self.logger.info(f"Pattern {pattern_id} has {len(self.application_history[pattern_id])} applications - consider metrics recalculation")
    
    def get_pattern_ranking(self, patterns: List[ExtractedPattern]) -> List[Tuple[ExtractedPattern, float]]:
        """Rank patterns by overall quality score."""
        try:
            pattern_scores = []
            
            for pattern in patterns:
                metrics = self.calculate_pattern_metrics(pattern)
                
                # Calculate composite quality score
                quality_score = (
                    metrics.success_rate * self.metric_weights['success_rate'] +
                    metrics.applicability_score * self.metric_weights['applicability_score'] +
                    metrics.reusability_index * self.metric_weights['reusability_index'] +
                    metrics.evolution_score * self.metric_weights['evolution_score'] +
                    metrics.performance_impact * self.metric_weights['performance_impact']
                )
                
                # Adjust for confidence (lower sample sizes get penalized)
                confidence_adjustment = min(1.0, metrics.sample_size / 10)  # Full confidence at 10+ samples
                adjusted_score = quality_score * (0.5 + 0.5 * confidence_adjustment)
                
                pattern_scores.append((pattern, adjusted_score))
            
            # Sort by score (descending)
            pattern_scores.sort(key=lambda x: x[1], reverse=True)
            
            return pattern_scores
            
        except Exception as e:
            self.logger.error(f"Error ranking patterns: {e}")
            return [(pattern, 0.5) for pattern in patterns]
    
    def analyze_success_factors(self, high_performing_patterns: List[ExtractedPattern]) -> Dict[str, Any]:
        """Analyze common factors in high-performing patterns."""
        if not high_performing_patterns:
            return {}
        
        try:
            analysis = {
                'pattern_types': Counter(),
                'domains': Counter(),
                'complexity_levels': Counter(),
                'extraction_methods': Counter(),
                'common_characteristics': [],
                'success_drivers': []
            }
            
            for pattern in high_performing_patterns:
                analysis['pattern_types'][pattern.pattern_type] += 1
                analysis['domains'][pattern.context.get('domain', 'unknown')] += 1
                analysis['complexity_levels'][pattern.context.get('complexity', 'medium')] += 1
                analysis['extraction_methods'][pattern.extraction_method] += 1
            
            # Identify common characteristics
            if analysis['pattern_types'].most_common(1):
                most_common_type = analysis['pattern_types'].most_common(1)[0][0]
                analysis['common_characteristics'].append(f"Most successful pattern type: {most_common_type}")
            
            if analysis['domains'].most_common(1):
                most_common_domain = analysis['domains'].most_common(1)[0][0]
                analysis['common_characteristics'].append(f"Most successful domain: {most_common_domain}")
            
            # Identify success drivers
            analysis['success_drivers'] = [
                "Clear problem definition",
                "Multiple application contexts",
                "Consistent positive feedback",
                "Low adaptation requirements",
                "Strong performance impact"
            ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing success factors: {e}")
            return {}
    
    def export_metrics_report(self, patterns: List[ExtractedPattern], output_path: str) -> bool:
        """Export comprehensive metrics report."""
        try:
            # Calculate metrics for all patterns
            all_metrics = self.calculate_batch_metrics(patterns)
            
            # Rank patterns
            pattern_rankings = self.get_pattern_ranking(patterns)
            
            # Analyze success factors for top patterns
            top_patterns = [p for p, score in pattern_rankings[:10]]  # Top 10
            success_analysis = self.analyze_success_factors(top_patterns)
            
            # Create report
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_patterns': len(patterns),
                    'patterns_with_data': len([m for m in all_metrics.values() if m.sample_size > 0]),
                    'analysis_period': '180 days'
                },
                'summary_statistics': {
                    'average_success_rate': statistics.mean([m.success_rate for m in all_metrics.values()]),
                    'average_applicability': statistics.mean([m.applicability_score for m in all_metrics.values()]),
                    'average_reusability': statistics.mean([m.reusability_index for m in all_metrics.values()]),
                    'total_applications': sum(m.sample_size for m in all_metrics.values())
                },
                'pattern_rankings': [
                    {
                        'pattern_title': pattern.title,
                        'pattern_type': pattern.pattern_type,
                        'quality_score': score,
                        'metrics': all_metrics[pattern.signature.combined_hash].to_dict()
                    }
                    for pattern, score in pattern_rankings[:20]  # Top 20
                ],
                'success_analysis': success_analysis,
                'detailed_metrics': {
                    pattern_id: metrics.to_dict()
                    for pattern_id, metrics in all_metrics.items()
                }
            }
            
            # Write report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Exported metrics report to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics report: {e}")
            return False