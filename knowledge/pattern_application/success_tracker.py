"""
Success Tracker - Pattern Application Success Measurement

This module tracks and measures the success of pattern applications,
providing comprehensive metrics and learning feedback for continuous
improvement of the pattern application system.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import asdict

from .core import (
    ApplicationRecord, Pattern, ImplementationPlan,
    PatternApplicationStatus
)

logger = logging.getLogger(__name__)


class SuccessTracker:
    """
    Success tracking and measurement system for pattern applications.
    
    This class provides comprehensive success measurement including:
    - Implementation quality assessment
    - Timeline adherence tracking
    - Pattern effectiveness measurement
    - Outcome validation
    - Learning extraction for improvement
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize success tracker."""
        self.knowledge_system = knowledge_system
        self._load_success_metrics()
        self._load_quality_indicators()
        self._load_benchmark_data()
    
    def calculate_success_score(self, application: ApplicationRecord) -> float:
        """
        Calculate comprehensive success score for a pattern application.
        
        This method evaluates success across multiple dimensions:
        - Implementation completeness
        - Quality metrics
        - Timeline adherence
        - Pattern adaptation accuracy
        - Outcome validation
        
        Args:
            application: ApplicationRecord to measure
            
        Returns:
            Success score (0.0 to 1.0, higher is better)
        """
        logger.info(f"Calculating success score for application {application.application_id}")
        
        if application.status != PatternApplicationStatus.COMPLETED:
            logger.warning(f"Application {application.application_id} not completed, returning 0.0")
            return 0.0
        
        # Component scores
        completion_score = self._calculate_completion_score(application)
        quality_score = self._calculate_quality_score(application)
        timeline_score = self._calculate_timeline_score(application)
        adaptation_score = self._calculate_adaptation_score(application)
        outcome_score = self._calculate_outcome_score(application)
        
        # Weighted combination
        weights = self.success_weights
        final_score = (
            completion_score * weights['completion'] +
            quality_score * weights['quality'] +
            timeline_score * weights['timeline'] +
            adaptation_score * weights['adaptation'] +
            outcome_score * weights['outcome']
        )
        
        # Store detailed metrics for learning
        detailed_metrics = {
            'completion_score': completion_score,
            'quality_score': quality_score,
            'timeline_score': timeline_score,
            'adaptation_score': adaptation_score,
            'outcome_score': outcome_score,
            'final_score': final_score,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        application.execution_metrics.update(detailed_metrics)
        
        logger.info(f"Success score calculated: {final_score:.3f} for {application.application_id}")
        
        return final_score
    
    def track_application_metrics(self, application: ApplicationRecord, 
                                metrics: Dict[str, Any]) -> bool:
        """
        Track ongoing metrics for a pattern application.
        
        Args:
            application: Application to track
            metrics: Metrics data to record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Merge with existing metrics
            application.execution_metrics.update(metrics)
            
            # Add timestamp
            application.execution_metrics['last_updated'] = datetime.utcnow().isoformat()
            
            # Store in knowledge system if configured
            if self.knowledge_system:
                self._store_metrics(application)
            
            logger.info(f"Updated metrics for application {application.application_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track metrics for {application.application_id}: {str(e)}")
            return False
    
    def generate_success_report(self, application: ApplicationRecord) -> Dict[str, Any]:
        """
        Generate comprehensive success report for an application.
        
        Args:
            application: Application to report on
            
        Returns:
            Detailed success report dictionary
        """
        if not application.success_score:
            success_score = self.calculate_success_score(application)
        else:
            success_score = application.success_score
        
        report = {
            'application_id': application.application_id,
            'pattern_id': application.pattern_id,
            'issue_id': application.issue_id,
            'overall_success_score': success_score,
            'success_grade': self._score_to_grade(success_score),
            'completion_status': application.status.value,
            'duration': self._calculate_duration(application),
            
            'detailed_scores': application.execution_metrics.get('detailed_scores', {}),
            
            'strengths': self._identify_strengths(application),
            'weaknesses': self._identify_weaknesses(application),
            'lessons_learned': application.lessons_learned,
            'recommendations': self._generate_recommendations(application),
            
            'pattern_performance': self._analyze_pattern_performance(application),
            'adaptation_effectiveness': self._analyze_adaptation_effectiveness(application),
            
            'benchmark_comparison': self._compare_to_benchmarks(application),
            
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return report
    
    def extract_learnings(self, application: ApplicationRecord) -> List[str]:
        """
        Extract learnings and insights from a completed application.
        
        Args:
            application: Completed application to analyze
            
        Returns:
            List of learning insights
        """
        learnings = []
        
        # Success-based learnings
        if application.success_score and application.success_score > 0.8:
            learnings.append("High success pattern application - approach validated")
            
            if application.adaptation_result:
                adaptation_strategy = application.adaptation_result.adaptation_strategy.value
                learnings.append(f"Successful adaptation strategy: {adaptation_strategy}")
        
        # Challenge-based learnings
        if application.execution_metrics.get('timeline_score', 1.0) < 0.7:
            learnings.append("Timeline challenges encountered - consider better estimation")
        
        if application.execution_metrics.get('quality_score', 1.0) < 0.7:
            learnings.append("Quality issues found - enhance validation processes")
        
        # Pattern-specific learnings
        if application.adaptation_result and len(application.adaptation_result.changes_made) > 3:
            learnings.append("Extensive pattern adaptation required - consider pattern refinement")
        
        # Implementation plan learnings
        if application.implementation_plan:
            if len(application.implementation_plan.risk_factors) > 5:
                learnings.append("High-risk implementation - risk assessment was accurate")
        
        # Store learnings in application record
        application.lessons_learned.extend(learnings)
        application.lessons_learned = list(set(application.lessons_learned))  # Remove duplicates
        
        return learnings
    
    def update_pattern_success_metrics(self, application: ApplicationRecord) -> bool:
        """
        Update success metrics for the pattern based on application results.
        
        Args:
            application: Completed application
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not application.success_score:
                return False
            
            # Update pattern success rate and usage count
            pattern_update_data = {
                'pattern_id': application.pattern_id,
                'application_success': application.success_score,
                'usage_increment': 1,
                'last_used': datetime.utcnow().isoformat(),
                'application_id': application.application_id
            }
            
            # Store pattern update in knowledge system
            if self.knowledge_system:
                self.knowledge_system.store_knowledge(
                    collection="pattern_usage_updates",
                    content=pattern_update_data,
                    metadata={
                        "type": "pattern_usage_update",
                        "pattern_id": application.pattern_id,
                        "success_score": application.success_score
                    }
                )
            
            logger.info(f"Updated pattern {application.pattern_id} success metrics")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pattern metrics: {str(e)}")
            return False
    
    def _load_success_metrics(self):
        """Load success measurement configuration."""
        self.success_weights = {
            'completion': 0.25,    # 25% - Did we complete what was planned?
            'quality': 0.30,       # 30% - How good is the implementation?
            'timeline': 0.15,      # 15% - Did we meet timeline expectations?
            'adaptation': 0.15,    # 15% - How well did pattern adaptation work?
            'outcome': 0.15        # 15% - Did we achieve the desired outcome?
        }
        
        self.grade_thresholds = {
            'A': 0.9,   # Excellent
            'B': 0.8,   # Good
            'C': 0.7,   # Satisfactory
            'D': 0.6,   # Poor
            'F': 0.0    # Failure
        }
    
    def _load_quality_indicators(self):
        """Load quality assessment indicators."""
        self.quality_indicators = {
            'tests_passing': 0.3,
            'code_review_approved': 0.2,
            'security_scan_clean': 0.2,
            'performance_benchmarks_met': 0.15,
            'documentation_complete': 0.1,
            'standards_compliance': 0.05
        }
        
        self.quality_benchmarks = {
            'test_coverage_minimum': 0.8,
            'code_quality_minimum': 0.7,
            'security_score_minimum': 0.9,
            'performance_degradation_maximum': 0.1
        }
    
    def _load_benchmark_data(self):
        """Load benchmark data for comparisons."""
        self.complexity_benchmarks = {
            'low': {
                'expected_duration_hours': 2,
                'expected_success_rate': 0.9,
                'typical_task_count': 3
            },
            'medium': {
                'expected_duration_hours': 6,
                'expected_success_rate': 0.8,
                'typical_task_count': 6
            },
            'high': {
                'expected_duration_hours': 16,
                'expected_success_rate': 0.7,
                'typical_task_count': 12
            },
            'very-high': {
                'expected_duration_hours': 40,
                'expected_success_rate': 0.6,
                'typical_task_count': 20
            }
        }
    
    def _calculate_completion_score(self, application: ApplicationRecord) -> float:
        """Calculate completion score based on implementation plan execution."""
        if not application.implementation_plan:
            return 0.5  # Neutral score if no plan
        
        plan = application.implementation_plan
        metrics = application.execution_metrics
        
        # Check if all tasks were completed
        tasks_completed = metrics.get('tasks_completed', 0)
        total_tasks = len(plan.tasks) if plan.tasks else 1
        
        completion_ratio = min(tasks_completed / total_tasks, 1.0)
        
        # Check if quality gates were passed
        quality_gates_passed = metrics.get('quality_gates_passed', 0)
        total_quality_gates = len(plan.quality_gates) if plan.quality_gates else 1
        
        quality_gate_ratio = min(quality_gates_passed / total_quality_gates, 1.0)
        
        # Weighted combination
        completion_score = (completion_ratio * 0.7) + (quality_gate_ratio * 0.3)
        
        return completion_score
    
    def _calculate_quality_score(self, application: ApplicationRecord) -> float:
        """Calculate quality score based on implementation quality metrics."""
        metrics = application.execution_metrics
        quality_score = 0.0
        
        # Check each quality indicator
        for indicator, weight in self.quality_indicators.items():
            if indicator in metrics:
                # Normalize different metric types
                if isinstance(metrics[indicator], bool):
                    indicator_score = 1.0 if metrics[indicator] else 0.0
                elif isinstance(metrics[indicator], (int, float)):
                    indicator_score = min(float(metrics[indicator]), 1.0)
                else:
                    indicator_score = 0.5  # Unknown format
                
                quality_score += indicator_score * weight
        
        # If no quality metrics available, use pattern confidence as proxy
        if quality_score == 0.0 and application.adaptation_result:
            quality_score = application.adaptation_result.confidence_score
        
        return quality_score
    
    def _calculate_timeline_score(self, application: ApplicationRecord) -> float:
        """Calculate timeline adherence score."""
        if not application.started_at or not application.completed_at:
            return 0.5  # Neutral if timing data missing
        
        actual_duration = application.completed_at - application.started_at
        actual_hours = actual_duration.total_seconds() / 3600
        
        # Get expected duration from implementation plan
        expected_hours = 8.0  # Default
        if application.implementation_plan:
            estimated_time = application.implementation_plan.estimated_total_time
            expected_hours = self._parse_time_estimate(estimated_time)
        
        # Calculate timeline score (closer to expected = better)
        if actual_hours <= expected_hours:
            # Under or on time is good
            timeline_score = 1.0
        else:
            # Over time decreases score
            overrun_ratio = actual_hours / expected_hours
            timeline_score = max(0.0, 1.0 - (overrun_ratio - 1.0))
        
        return timeline_score
    
    def _calculate_adaptation_score(self, application: ApplicationRecord) -> float:
        """Calculate pattern adaptation effectiveness score."""
        if not application.adaptation_result:
            return 0.5  # Neutral if no adaptation data
        
        adaptation = application.adaptation_result
        
        # Base score from adaptation confidence
        base_score = adaptation.confidence_score
        
        # Bonus for successful adaptations with multiple changes
        if len(adaptation.changes_made) > 1 and base_score > 0.7:
            base_score += 0.1  # Bonus for complex successful adaptation
        
        # Penalty for low confidence adaptations
        if base_score < 0.5:
            base_score *= 0.8
        
        return min(1.0, base_score)
    
    def _calculate_outcome_score(self, application: ApplicationRecord) -> float:
        """Calculate outcome achievement score."""
        metrics = application.execution_metrics
        
        # Check success criteria if available
        if application.implementation_plan and application.implementation_plan.success_criteria:
            criteria_met = metrics.get('success_criteria_met', 0)
            total_criteria = len(application.implementation_plan.success_criteria)
            outcome_score = min(criteria_met / total_criteria, 1.0) if total_criteria > 0 else 0.5
        else:
            # Use general outcome indicators
            outcome_indicators = [
                'functionality_working',
                'requirements_satisfied',
                'stakeholder_approval',
                'integration_successful'
            ]
            
            met_indicators = sum(1 for indicator in outcome_indicators 
                               if metrics.get(indicator, False))
            outcome_score = met_indicators / len(outcome_indicators)
        
        return outcome_score
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        for grade, threshold in sorted(self.grade_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return 'F'
    
    def _calculate_duration(self, application: ApplicationRecord) -> str:
        """Calculate human-readable duration."""
        if not application.started_at or not application.completed_at:
            return "Unknown"
        
        duration = application.completed_at - application.started_at
        
        if duration.total_seconds() < 3600:
            return f"{int(duration.total_seconds() / 60)} minutes"
        elif duration.total_seconds() < 86400:
            return f"{duration.total_seconds() / 3600:.1f} hours"
        else:
            return f"{duration.days} days, {duration.seconds // 3600} hours"
    
    def _identify_strengths(self, application: ApplicationRecord) -> List[str]:
        """Identify strengths from the application execution."""
        strengths = []
        metrics = application.execution_metrics
        
        if metrics.get('completion_score', 0) > 0.8:
            strengths.append("Excellent task completion rate")
        
        if metrics.get('quality_score', 0) > 0.8:
            strengths.append("High implementation quality")
        
        if metrics.get('timeline_score', 0) > 0.8:
            strengths.append("Good timeline adherence")
        
        if application.adaptation_result and application.adaptation_result.confidence_score > 0.8:
            strengths.append("Effective pattern adaptation")
        
        if metrics.get('tests_passing', False):
            strengths.append("Comprehensive testing")
        
        return strengths
    
    def _identify_weaknesses(self, application: ApplicationRecord) -> List[str]:
        """Identify weaknesses from the application execution."""
        weaknesses = []
        metrics = application.execution_metrics
        
        if metrics.get('completion_score', 1) < 0.6:
            weaknesses.append("Low task completion rate")
        
        if metrics.get('quality_score', 1) < 0.6:
            weaknesses.append("Quality issues in implementation")
        
        if metrics.get('timeline_score', 1) < 0.6:
            weaknesses.append("Timeline overruns")
        
        if application.adaptation_result and application.adaptation_result.confidence_score < 0.5:
            weaknesses.append("Pattern adaptation challenges")
        
        if not metrics.get('tests_passing', True):
            weaknesses.append("Testing deficiencies")
        
        return weaknesses
    
    def _generate_recommendations(self, application: ApplicationRecord) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        weaknesses = self._identify_weaknesses(application)
        
        if "Low task completion rate" in weaknesses:
            recommendations.append("Improve task breakdown and dependency management")
        
        if "Quality issues in implementation" in weaknesses:
            recommendations.append("Enhance code review and validation processes")
        
        if "Timeline overruns" in weaknesses:
            recommendations.append("Improve time estimation and planning")
        
        if "Pattern adaptation challenges" in weaknesses:
            recommendations.append("Refine pattern adaptation algorithms")
        
        if "Testing deficiencies" in weaknesses:
            recommendations.append("Strengthen testing requirements and automation")
        
        return recommendations
    
    def _analyze_pattern_performance(self, application: ApplicationRecord) -> Dict[str, Any]:
        """Analyze how well the pattern performed."""
        analysis = {
            'pattern_id': application.pattern_id,
            'adaptation_required': bool(application.adaptation_result),
            'adaptation_changes': len(application.adaptation_result.changes_made) if application.adaptation_result else 0,
            'pattern_confidence': application.adaptation_result.confidence_score if application.adaptation_result else 0.0,
            'overall_effectiveness': application.success_score or 0.0
        }
        
        return analysis
    
    def _analyze_adaptation_effectiveness(self, application: ApplicationRecord) -> Dict[str, Any]:
        """Analyze effectiveness of pattern adaptation."""
        if not application.adaptation_result:
            return {'adaptation_used': False}
        
        adaptation = application.adaptation_result
        
        analysis = {
            'adaptation_used': True,
            'strategy': adaptation.adaptation_strategy.value,
            'changes_count': len(adaptation.changes_made),
            'confidence': adaptation.confidence_score,
            'changes_list': adaptation.changes_made,
            'effectiveness_score': min(adaptation.confidence_score * (application.success_score or 0.5), 1.0)
        }
        
        return analysis
    
    def _compare_to_benchmarks(self, application: ApplicationRecord) -> Dict[str, Any]:
        """Compare application performance to benchmarks."""
        comparison = {}
        
        # Get complexity benchmarks
        complexity = 'medium'  # default
        if application.implementation_plan:
            complexity = application.implementation_plan.complexity_assessment
        
        benchmarks = self.complexity_benchmarks.get(complexity, self.complexity_benchmarks['medium'])
        
        # Duration comparison
        actual_duration = self._calculate_duration_hours(application)
        expected_duration = benchmarks['expected_duration_hours']
        comparison['duration_vs_benchmark'] = {
            'actual_hours': actual_duration,
            'expected_hours': expected_duration,
            'performance': 'better' if actual_duration <= expected_duration else 'worse'
        }
        
        # Success rate comparison
        success_rate = application.success_score or 0.0
        expected_success = benchmarks['expected_success_rate']
        comparison['success_vs_benchmark'] = {
            'actual_success': success_rate,
            'expected_success': expected_success,
            'performance': 'better' if success_rate >= expected_success else 'worse'
        }
        
        return comparison
    
    def _calculate_duration_hours(self, application: ApplicationRecord) -> float:
        """Calculate actual duration in hours."""
        if not application.started_at or not application.completed_at:
            return 0.0
        
        duration = application.completed_at - application.started_at
        return duration.total_seconds() / 3600
    
    def _parse_time_estimate(self, time_str: str) -> float:
        """Parse time estimate string to hours."""
        try:
            time_lower = time_str.lower()
            
            if 'hour' in time_lower:
                # Extract number before 'hour'
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', time_lower)
                if match:
                    return float(match.group(1))
            elif 'day' in time_lower:
                # Extract number before 'day' and multiply by 8
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', time_lower)
                if match:
                    return float(match.group(1)) * 8
            elif 'minute' in time_lower:
                # Extract number before 'minute' and divide by 60
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', time_lower)
                if match:
                    return float(match.group(1)) / 60
            
            return 8.0  # Default 8 hours
            
        except Exception:
            return 8.0  # Default fallback
    
    def _store_metrics(self, application: ApplicationRecord):
        """Store metrics in knowledge system."""
        try:
            self.knowledge_system.store_knowledge(
                collection="application_metrics",
                content=application.execution_metrics,
                metadata={
                    "type": "application_metrics",
                    "application_id": application.application_id,
                    "pattern_id": application.pattern_id,
                    "issue_id": application.issue_id,
                    "success_score": application.success_score
                }
            )
        except Exception as e:
            logger.error(f"Failed to store metrics in knowledge system: {str(e)}")