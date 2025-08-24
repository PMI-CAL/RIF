#!/usr/bin/env python3
"""
Specialist Assignment Accuracy Tracking

Advanced tracking system for monitoring and analyzing the accuracy and effectiveness
of specialist agent assignments within the RIF framework.

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
class SpecialistMetrics:
    """Performance metrics for a specific specialist type"""
    specialist_type: str
    total_assignments: int
    appropriate_assignments: int
    inappropriate_assignments: int
    accuracy_rate: float
    avg_response_time_hours: float
    success_rate: float
    escalation_rate: float
    quality_improvement_score: float
    assignment_confidence: float
    workload_efficiency: float

@dataclass
class AssignmentPattern:
    """Pattern analysis for specialist assignments"""
    pattern_id: str
    description: str
    frequency: int
    accuracy_rate: float
    typical_contexts: List[str]
    success_factors: List[str]
    failure_factors: List[str]
    recommendations: List[str]

@dataclass
class WorkloadAnalysis:
    """Analysis of specialist workload distribution"""
    specialist_type: str
    current_load: int
    optimal_load_range: Tuple[int, int]
    load_balance_score: float
    peak_periods: List[str]
    underutilized_periods: List[str]
    capacity_recommendations: List[str]

@dataclass
class SpecialistAssignmentResult:
    """Complete specialist assignment analysis result"""
    analysis_period: str
    total_assignments: int
    overall_accuracy_rate: float
    specialist_metrics: List[SpecialistMetrics]
    assignment_patterns: List[AssignmentPattern]
    workload_analysis: List[WorkloadAnalysis]
    optimization_opportunities: List[str]
    training_recommendations: List[str]
    system_improvements: List[str]
    timestamp: str

class SpecialistAssignmentTracker:
    """
    Advanced tracker for specialist assignment accuracy and optimization.
    Provides comprehensive insights into specialist utilization and effectiveness.
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize tracker with knowledge base path"""
        self.knowledge_path = Path(knowledge_base_path)
        self.metrics_path = self.knowledge_path / "quality_metrics"
        self.specialists_path = self.knowledge_path / "specialists"
        self.analysis_path = self.knowledge_path / "analysis"
        
        # Create directories
        self.specialists_path.mkdir(parents=True, exist_ok=True)
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        
        # Specialist type definitions
        self.specialist_types = [
            'rif-analyst', 'rif-planner', 'rif-architect', 'rif-implementer',
            'rif-validator', 'rif-learner', 'security-specialist', 'performance-specialist',
            'ui-specialist', 'data-specialist', 'integration-specialist'
        ]
        
        # Assignment context mapping
        self.context_specialist_mapping = {
            'security_issue': ['security-specialist', 'rif-validator'],
            'performance_problem': ['performance-specialist', 'rif-implementer'],
            'ui_enhancement': ['ui-specialist', 'rif-implementer'],
            'data_analysis': ['data-specialist', 'rif-analyst'],
            'integration_task': ['integration-specialist', 'rif-architect'],
            'code_review': ['rif-validator', 'security-specialist'],
            'architecture_design': ['rif-architect', 'rif-planner'],
            'testing': ['rif-validator', 'performance-specialist'],
            'documentation': ['rif-learner', 'rif-analyst'],
            'planning': ['rif-planner', 'rif-analyst']
        }
        
        self.start_time = None
    
    def analyze_specialist_assignments(self, days: int = 30) -> SpecialistAssignmentResult:
        """
        Perform comprehensive specialist assignment analysis
        
        Args:
            days: Number of days to analyze
            
        Returns:
            SpecialistAssignmentResult: Complete assignment analysis
        """
        self.start_time = datetime.now()
        
        try:
            logger.info(f"Analyzing specialist assignments for last {days} days...")
            
            # Load assignment data
            assignment_data = self._load_assignment_data(days)
            
            # Load specialist performance data
            performance_data = self._load_performance_data(days)
            
            logger.info(f"Analyzing {len(assignment_data)} assignments with {len(performance_data)} performance records")
            
            # Calculate specialist metrics
            specialist_metrics = self._calculate_specialist_metrics(assignment_data, performance_data)
            
            # Analyze assignment patterns
            assignment_patterns = self._analyze_assignment_patterns(assignment_data)
            
            # Analyze workload distribution
            workload_analysis = self._analyze_workload_distribution(assignment_data)
            
            # Generate optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                specialist_metrics, assignment_patterns, workload_analysis
            )
            
            # Generate training recommendations
            training_recommendations = self._generate_training_recommendations(
                specialist_metrics, assignment_patterns
            )
            
            # Generate system improvements
            system_improvements = self._generate_system_improvements(
                assignment_patterns, workload_analysis
            )
            
            # Calculate overall accuracy
            total_assignments = len(assignment_data)
            appropriate_assignments = sum(1 for a in assignment_data if a.get('assignment_appropriate', True))
            overall_accuracy = appropriate_assignments / total_assignments if total_assignments > 0 else 0.0
            
            # Create result
            result = SpecialistAssignmentResult(
                analysis_period=f"{days} days",
                total_assignments=total_assignments,
                overall_accuracy_rate=overall_accuracy,
                specialist_metrics=specialist_metrics,
                assignment_patterns=assignment_patterns,
                workload_analysis=workload_analysis,
                optimization_opportunities=optimization_opportunities,
                training_recommendations=training_recommendations,
                system_improvements=system_improvements,
                timestamp=datetime.now().isoformat()
            )
            
            # Save results
            self._save_analysis_result(result)
            
            processing_time = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.info(f"Specialist assignment analysis completed in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in specialist assignment analysis: {str(e)}")
            raise
    
    def _load_assignment_data(self, days: int) -> List[Dict[str, Any]]:
        """Load specialist assignment data from specified time period"""
        assignment_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load from specialist assignments directory
        if self.specialists_path.exists():
            for assignment_file in self.specialists_path.glob("assignments_*.json"):
                try:
                    with open(assignment_file, 'r') as f:
                        data = json.load(f)
                    
                    for assignment in data.get('assignments', []):
                        if assignment.get('timestamp'):
                            assignment_date = datetime.fromisoformat(assignment['timestamp'])
                            if assignment_date >= cutoff_date:
                                assignment_data.append(assignment)
                                
                except Exception as e:
                    logger.warning(f"Error loading assignment file {assignment_file}: {str(e)}")
        
        # Load from quality metrics (assignments embedded in decisions)
        realtime_dir = self.metrics_path / "realtime"
        if realtime_dir.exists():
            for session_file in realtime_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    for decision in session_data.get('decisions', []):
                        if decision.get('specialist_assignment'):
                            assignment = decision['specialist_assignment']
                            if assignment.get('timestamp'):
                                assignment_date = datetime.fromisoformat(assignment['timestamp'])
                                if assignment_date >= cutoff_date:
                                    # Enrich with decision context
                                    assignment['decision_context'] = {
                                        'quality_score': decision.get('quality_score'),
                                        'component_type': decision.get('component_type'),
                                        'outcome': decision.get('outcome')
                                    }
                                    assignment_data.append(assignment)
                                    
                except Exception as e:
                    logger.warning(f"Error loading session file {session_file}: {str(e)}")
        
        # Generate synthetic assignment data if no real data exists (for testing/demo)
        if not assignment_data:
            assignment_data = self._generate_synthetic_assignments(days)
        
        logger.info(f"Loaded {len(assignment_data)} specialist assignments")
        return assignment_data
    
    def _load_performance_data(self, days: int) -> List[Dict[str, Any]]:
        """Load specialist performance data"""
        performance_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load from specialist performance directory
        performance_dir = self.specialists_path / "performance"
        if performance_dir.exists():
            for perf_file in performance_dir.glob("*.json"):
                try:
                    with open(perf_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('timestamp'):
                        perf_date = datetime.fromisoformat(data['timestamp'])
                        if perf_date >= cutoff_date:
                            performance_data.append(data)
                            
                except Exception as e:
                    logger.warning(f"Error loading performance file {perf_file}: {str(e)}")
        
        # Load from quality metrics (specialist outcomes)
        analytics_dir = self.metrics_path / "analytics"
        if analytics_dir.exists():
            for analytics_file in analytics_dir.glob("*.json"):
                try:
                    with open(analytics_file, 'r') as f:
                        data = json.load(f)
                    
                    for outcome in data.get('specialist_outcomes', []):
                        if outcome.get('timestamp'):
                            outcome_date = datetime.fromisoformat(outcome['timestamp'])
                            if outcome_date >= cutoff_date:
                                performance_data.append(outcome)
                                
                except Exception as e:
                    logger.warning(f"Error loading analytics file {analytics_file}: {str(e)}")
        
        logger.info(f"Loaded {len(performance_data)} performance records")
        return performance_data
    
    def _generate_synthetic_assignments(self, days: int) -> List[Dict[str, Any]]:
        """Generate synthetic assignment data for testing purposes"""
        import random
        
        synthetic_data = []
        contexts = list(self.context_specialist_mapping.keys())
        
        # Generate assignments for the specified period
        for day in range(days):
            date = datetime.now() - timedelta(days=day)
            daily_assignments = random.randint(2, 8)  # 2-8 assignments per day
            
            for _ in range(daily_assignments):
                context = random.choice(contexts)
                suitable_specialists = self.context_specialist_mapping[context]
                assigned_specialist = random.choice(suitable_specialists)
                
                # Determine if assignment was appropriate
                is_appropriate = random.random() > 0.15  # 85% accuracy baseline
                
                # Generate assignment record
                assignment = {
                    'assignment_id': f"assign_{date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}",
                    'timestamp': date.isoformat(),
                    'context': context,
                    'assigned_specialist': assigned_specialist,
                    'assignment_appropriate': is_appropriate,
                    'response_time_hours': random.uniform(0.5, 24.0),
                    'completion_status': random.choice(['completed', 'in_progress', 'escalated']),
                    'quality_outcome': random.uniform(0.6, 1.0),
                    'confidence_score': random.uniform(0.7, 1.0),
                    'workload_factor': random.uniform(0.3, 1.0)
                }
                
                synthetic_data.append(assignment)
        
        logger.info(f"Generated {len(synthetic_data)} synthetic assignments for testing")
        return synthetic_data
    
    def _calculate_specialist_metrics(self, assignment_data: List[Dict[str, Any]], 
                                    performance_data: List[Dict[str, Any]]) -> List[SpecialistMetrics]:
        """Calculate performance metrics for each specialist type"""
        specialist_metrics = []
        
        # Group assignments by specialist type
        specialist_assignments = defaultdict(list)
        for assignment in assignment_data:
            specialist_type = assignment.get('assigned_specialist', 'unknown')
            specialist_assignments[specialist_type].append(assignment)
        
        # Group performance data by specialist
        specialist_performance = defaultdict(list)
        for performance in performance_data:
            specialist_type = performance.get('specialist_type', 'unknown')
            specialist_performance[specialist_type].append(performance)
        
        # Calculate metrics for each specialist
        for specialist_type in self.specialist_types:
            assignments = specialist_assignments.get(specialist_type, [])
            performance_records = specialist_performance.get(specialist_type, [])
            
            if not assignments and not performance_records:
                continue  # Skip specialists with no data
            
            # Assignment accuracy metrics
            total_assignments = len(assignments)
            appropriate_assignments = sum(1 for a in assignments if a.get('assignment_appropriate', True))
            inappropriate_assignments = total_assignments - appropriate_assignments
            accuracy_rate = appropriate_assignments / total_assignments if total_assignments > 0 else 0.0
            
            # Response time metrics
            response_times = [a.get('response_time_hours', 0) for a in assignments if a.get('response_time_hours')]
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            
            # Success rate
            completed_assignments = sum(1 for a in assignments if a.get('completion_status') == 'completed')
            success_rate = completed_assignments / total_assignments if total_assignments > 0 else 0.0
            
            # Escalation rate
            escalated_assignments = sum(1 for a in assignments if a.get('completion_status') == 'escalated')
            escalation_rate = escalated_assignments / total_assignments if total_assignments > 0 else 0.0
            
            # Quality improvement score
            quality_outcomes = [a.get('quality_outcome', 0) for a in assignments if a.get('quality_outcome')]
            quality_improvement_score = statistics.mean(quality_outcomes) if quality_outcomes else 0.0
            
            # Assignment confidence
            confidence_scores = [a.get('confidence_score', 0) for a in assignments if a.get('confidence_score')]
            assignment_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            
            # Workload efficiency
            workload_factors = [a.get('workload_factor', 0) for a in assignments if a.get('workload_factor')]
            workload_efficiency = statistics.mean(workload_factors) if workload_factors else 0.0
            
            metrics = SpecialistMetrics(
                specialist_type=specialist_type,
                total_assignments=total_assignments,
                appropriate_assignments=appropriate_assignments,
                inappropriate_assignments=inappropriate_assignments,
                accuracy_rate=accuracy_rate,
                avg_response_time_hours=avg_response_time,
                success_rate=success_rate,
                escalation_rate=escalation_rate,
                quality_improvement_score=quality_improvement_score,
                assignment_confidence=assignment_confidence,
                workload_efficiency=workload_efficiency
            )
            
            specialist_metrics.append(metrics)
        
        # Sort by total assignments (most active first)
        specialist_metrics.sort(key=lambda m: m.total_assignments, reverse=True)
        
        return specialist_metrics
    
    def _analyze_assignment_patterns(self, assignment_data: List[Dict[str, Any]]) -> List[AssignmentPattern]:
        """Analyze patterns in specialist assignments"""
        patterns = []
        
        # Group by context
        context_assignments = defaultdict(list)
        for assignment in assignment_data:
            context = assignment.get('context', 'unknown')
            context_assignments[context].append(assignment)
        
        # Analyze each context pattern
        for context, assignments in context_assignments.items():
            if len(assignments) < 3:  # Need minimum data
                continue
            
            # Calculate pattern metrics
            total = len(assignments)
            appropriate = sum(1 for a in assignments if a.get('assignment_appropriate', True))
            accuracy_rate = appropriate / total
            
            # Analyze typical contexts
            specialists_used = [a.get('assigned_specialist', 'unknown') for a in assignments]
            common_specialists = [spec for spec, count in Counter(specialists_used).most_common(3)]
            
            # Success factors
            successful_assignments = [a for a in assignments if a.get('quality_outcome', 0) > 0.8]
            success_factors = self._identify_success_factors(successful_assignments)
            
            # Failure factors
            failed_assignments = [a for a in assignments if a.get('quality_outcome', 0) < 0.6]
            failure_factors = self._identify_failure_factors(failed_assignments)
            
            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(
                context, accuracy_rate, success_factors, failure_factors
            )
            
            pattern = AssignmentPattern(
                pattern_id=f"pattern_{context}",
                description=f"Assignments for {context.replace('_', ' ')} context",
                frequency=total,
                accuracy_rate=accuracy_rate,
                typical_contexts=common_specialists,
                success_factors=success_factors,
                failure_factors=failure_factors,
                recommendations=recommendations
            )
            
            patterns.append(pattern)
        
        # Sort by frequency and accuracy
        patterns.sort(key=lambda p: (p.frequency, p.accuracy_rate), reverse=True)
        
        return patterns[:10]  # Return top 10 patterns
    
    def _identify_success_factors(self, successful_assignments: List[Dict[str, Any]]) -> List[str]:
        """Identify common factors in successful assignments"""
        if not successful_assignments:
            return []
        
        factors = []
        
        # Response time factor
        response_times = [a.get('response_time_hours', 0) for a in successful_assignments]
        if response_times:
            avg_response = statistics.mean(response_times)
            if avg_response < 4:  # Fast response
                factors.append("Fast response time (<4 hours)")
        
        # Confidence factor
        confidence_scores = [a.get('confidence_score', 0) for a in successful_assignments]
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence > 0.8:
                factors.append("High assignment confidence (>80%)")
        
        # Workload factor
        workload_factors = [a.get('workload_factor', 0) for a in successful_assignments]
        if workload_factors:
            avg_workload = statistics.mean(workload_factors)
            if avg_workload < 0.7:
                factors.append("Optimal workload level (<70%)")
        
        # Specialist appropriateness
        appropriate_rate = sum(1 for a in successful_assignments if a.get('assignment_appropriate', True))
        if appropriate_rate / len(successful_assignments) > 0.95:
            factors.append("Highly appropriate specialist selection")
        
        return factors[:5]  # Return top 5 factors
    
    def _identify_failure_factors(self, failed_assignments: List[Dict[str, Any]]) -> List[str]:
        """Identify common factors in failed assignments"""
        if not failed_assignments:
            return []
        
        factors = []
        
        # Response time factor
        response_times = [a.get('response_time_hours', 0) for a in failed_assignments]
        if response_times:
            avg_response = statistics.mean(response_times)
            if avg_response > 12:  # Slow response
                factors.append("Slow response time (>12 hours)")
        
        # Inappropriate assignments
        inappropriate_rate = sum(1 for a in failed_assignments if not a.get('assignment_appropriate', True))
        if inappropriate_rate / len(failed_assignments) > 0.3:
            factors.append("High inappropriate assignment rate (>30%)")
        
        # High escalation rate
        escalated_rate = sum(1 for a in failed_assignments if a.get('completion_status') == 'escalated')
        if escalated_rate / len(failed_assignments) > 0.2:
            factors.append("Frequent escalations (>20%)")
        
        # Low confidence
        confidence_scores = [a.get('confidence_score', 0) for a in failed_assignments]
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence < 0.6:
                factors.append("Low assignment confidence (<60%)")
        
        # High workload
        workload_factors = [a.get('workload_factor', 0) for a in failed_assignments]
        if workload_factors:
            avg_workload = statistics.mean(workload_factors)
            if avg_workload > 0.9:
                factors.append("Overloaded specialists (>90% capacity)")
        
        return factors[:5]  # Return top 5 factors
    
    def _generate_pattern_recommendations(self, context: str, accuracy_rate: float, 
                                        success_factors: List[str], failure_factors: List[str]) -> List[str]:
        """Generate recommendations based on assignment patterns"""
        recommendations = []
        
        if accuracy_rate < 0.7:
            recommendations.append(f"URGENT: Review specialist selection criteria for {context} - accuracy below 70%")
        elif accuracy_rate < 0.85:
            recommendations.append(f"Improve specialist matching for {context} - current accuracy {accuracy_rate:.1%}")
        
        # Context-specific recommendations
        if context == 'security_issue' and accuracy_rate < 0.9:
            recommendations.append("Security issues require high accuracy - consider mandatory security specialist assignment")
        
        if context == 'performance_problem':
            recommendations.append("Performance issues benefit from early performance specialist involvement")
        
        # Success factor based recommendations
        if "Fast response time" in success_factors:
            recommendations.append(f"Prioritize quick response for {context} assignments")
        
        if "High assignment confidence" in success_factors:
            recommendations.append(f"Use confidence scoring to validate {context} assignments")
        
        # Failure factor based recommendations  
        if "Slow response time" in failure_factors:
            recommendations.append(f"Implement response time monitoring for {context} assignments")
        
        if "High inappropriate assignment rate" in failure_factors:
            recommendations.append(f"Review and refine specialist selection algorithm for {context}")
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def _analyze_workload_distribution(self, assignment_data: List[Dict[str, Any]]) -> List[WorkloadAnalysis]:
        """Analyze workload distribution across specialists"""
        workload_analyses = []
        
        # Group assignments by specialist and time
        specialist_loads = defaultdict(list)
        for assignment in assignment_data:
            specialist = assignment.get('assigned_specialist', 'unknown')
            timestamp = assignment.get('timestamp', '')
            if timestamp:
                specialist_loads[specialist].append(assignment)
        
        for specialist_type in self.specialist_types:
            assignments = specialist_loads.get(specialist_type, [])
            
            if not assignments:
                continue
            
            current_load = len(assignments)
            
            # Calculate optimal load range based on specialist type
            load_multipliers = {
                'rif-implementer': 1.2,  # Can handle more load
                'rif-validator': 1.0,    # Standard load
                'security-specialist': 0.8,  # Needs more focus per task
                'performance-specialist': 0.9,
                'ui-specialist': 1.1
            }
            
            base_optimal = 20  # Base optimal load per analysis period
            multiplier = load_multipliers.get(specialist_type, 1.0)
            optimal_min = int(base_optimal * multiplier * 0.7)
            optimal_max = int(base_optimal * multiplier * 1.3)
            optimal_load_range = (optimal_min, optimal_max)
            
            # Calculate load balance score
            if current_load < optimal_min:
                load_balance_score = current_load / optimal_min
            elif current_load > optimal_max:
                load_balance_score = optimal_max / current_load
            else:
                load_balance_score = 1.0
            
            # Analyze time patterns
            peak_periods, underutilized_periods = self._analyze_time_patterns(assignments)
            
            # Generate capacity recommendations
            capacity_recommendations = self._generate_capacity_recommendations(
                specialist_type, current_load, optimal_load_range, load_balance_score
            )
            
            analysis = WorkloadAnalysis(
                specialist_type=specialist_type,
                current_load=current_load,
                optimal_load_range=optimal_load_range,
                load_balance_score=load_balance_score,
                peak_periods=peak_periods,
                underutilized_periods=underutilized_periods,
                capacity_recommendations=capacity_recommendations
            )
            
            workload_analyses.append(analysis)
        
        # Sort by load imbalance (most problematic first)
        workload_analyses.sort(key=lambda w: abs(1.0 - w.load_balance_score), reverse=True)
        
        return workload_analyses
    
    def _analyze_time_patterns(self, assignments: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Analyze time-based patterns in assignments"""
        # Group by day of week and hour
        daily_counts = defaultdict(int)
        hourly_counts = defaultdict(int)
        
        for assignment in assignments:
            timestamp = assignment.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    daily_counts[dt.strftime('%A')] += 1
                    hourly_counts[dt.hour] += 1
                except:
                    continue
        
        # Identify peak periods
        peak_periods = []
        if daily_counts:
            avg_daily = sum(daily_counts.values()) / len(daily_counts)
            for day, count in daily_counts.items():
                if count > avg_daily * 1.3:  # 30% above average
                    peak_periods.append(f"{day}s")
        
        # Identify underutilized periods
        underutilized_periods = []
        if daily_counts:
            for day, count in daily_counts.items():
                if count < avg_daily * 0.5:  # 50% below average
                    underutilized_periods.append(f"{day}s")
        
        return peak_periods[:3], underutilized_periods[:3]  # Return top 3 each
    
    def _generate_capacity_recommendations(self, specialist_type: str, current_load: int, 
                                         optimal_range: Tuple[int, int], balance_score: float) -> List[str]:
        """Generate capacity recommendations for specialist"""
        recommendations = []
        optimal_min, optimal_max = optimal_range
        
        if current_load < optimal_min:
            recommendations.append(f"Underutilized: Consider cross-training {specialist_type} for related tasks")
            recommendations.append(f"Current load ({current_load}) below optimal minimum ({optimal_min})")
            
        elif current_load > optimal_max:
            recommendations.append(f"OVERLOADED: {specialist_type} exceeds optimal capacity ({current_load} > {optimal_max})")
            recommendations.append(f"Consider adding additional {specialist_type} capacity or load redistribution")
            
        if balance_score < 0.8:
            recommendations.append(f"Load balancing needed for {specialist_type} (score: {balance_score:.1%})")
        
        # Specialist-specific recommendations
        if specialist_type == 'security-specialist' and current_load > optimal_min:
            recommendations.append("Security specialist load should be kept low for thorough analysis")
        
        if specialist_type == 'rif-implementer' and current_load < optimal_max * 0.8:
            recommendations.append("Implementer capacity available - consider assigning more development tasks")
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def _identify_optimization_opportunities(self, specialist_metrics: List[SpecialistMetrics],
                                           assignment_patterns: List[AssignmentPattern],
                                           workload_analysis: List[WorkloadAnalysis]) -> List[str]:
        """Identify optimization opportunities across all analyses"""
        opportunities = []
        
        # Accuracy-based opportunities
        low_accuracy_specialists = [m for m in specialist_metrics if m.accuracy_rate < 0.8]
        if low_accuracy_specialists:
            names = [m.specialist_type for m in low_accuracy_specialists]
            opportunities.append(f"Improve assignment accuracy for: {', '.join(names)}")
        
        # Response time opportunities
        slow_specialists = [m for m in specialist_metrics if m.avg_response_time_hours > 8]
        if slow_specialists:
            names = [m.specialist_type for m in slow_specialists]
            opportunities.append(f"Optimize response times for: {', '.join(names)}")
        
        # Workload balancing opportunities
        imbalanced_workloads = [w for w in workload_analysis if w.load_balance_score < 0.7]
        if imbalanced_workloads:
            names = [w.specialist_type for w in imbalanced_workloads]
            opportunities.append(f"Rebalance workloads for: {', '.join(names)}")
        
        # Pattern-based opportunities
        low_accuracy_patterns = [p for p in assignment_patterns if p.accuracy_rate < 0.8]
        if low_accuracy_patterns:
            contexts = [p.description for p in low_accuracy_patterns[:3]]
            opportunities.append(f"Improve assignment patterns for: {', '.join(contexts)}")
        
        # High-impact opportunities
        if any(m.escalation_rate > 0.2 for m in specialist_metrics):
            opportunities.append("HIGH IMPACT: Reduce escalation rates through better initial assignments")
        
        if any(m.success_rate < 0.7 for m in specialist_metrics):
            opportunities.append("HIGH IMPACT: Address low success rates affecting overall quality")
        
        return opportunities[:8]  # Return top 8 opportunities
    
    def _generate_training_recommendations(self, specialist_metrics: List[SpecialistMetrics],
                                         assignment_patterns: List[AssignmentPattern]) -> List[str]:
        """Generate training recommendations based on analysis"""
        recommendations = []
        
        # Specialist-specific training needs
        for metrics in specialist_metrics:
            if metrics.accuracy_rate < 0.8:
                recommendations.append(
                    f"{metrics.specialist_type}: Assignment accuracy training needed "
                    f"(current: {metrics.accuracy_rate:.1%})"
                )
            
            if metrics.quality_improvement_score < 0.7:
                recommendations.append(
                    f"{metrics.specialist_type}: Quality improvement techniques training "
                    f"(score: {metrics.quality_improvement_score:.1%})"
                )
            
            if metrics.escalation_rate > 0.15:
                recommendations.append(
                    f"{metrics.specialist_type}: Escalation management and problem-solving training "
                    f"(escalation rate: {metrics.escalation_rate:.1%})"
                )
        
        # Pattern-based training
        for pattern in assignment_patterns:
            if pattern.accuracy_rate < 0.8 and pattern.failure_factors:
                recommendations.append(
                    f"Context-specific training for {pattern.description}: "
                    f"Focus on {pattern.failure_factors[0]}"
                )
        
        # General training recommendations
        low_confidence_specialists = [m for m in specialist_metrics if m.assignment_confidence < 0.7]
        if low_confidence_specialists:
            recommendations.append("Confidence building and decision-making training for low-confidence specialists")
        
        high_response_time_specialists = [m for m in specialist_metrics if m.avg_response_time_hours > 12]
        if high_response_time_specialists:
            recommendations.append("Time management and prioritization training for slow-response specialists")
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def _generate_system_improvements(self, assignment_patterns: List[AssignmentPattern],
                                    workload_analysis: List[WorkloadAnalysis]) -> List[str]:
        """Generate system-level improvements"""
        improvements = []
        
        # Assignment algorithm improvements
        low_accuracy_patterns = [p for p in assignment_patterns if p.accuracy_rate < 0.85]
        if low_accuracy_patterns:
            improvements.append("Enhance assignment algorithm with context-specific rules and confidence scoring")
        
        # Load balancing improvements
        imbalanced_loads = [w for w in workload_analysis if w.load_balance_score < 0.8]
        if imbalanced_loads:
            improvements.append("Implement dynamic load balancing with real-time capacity monitoring")
        
        # Response time improvements
        improvements.append("Add response time SLA monitoring with automated escalation triggers")
        
        # Quality feedback loop
        improvements.append("Implement closed-loop feedback system for assignment accuracy improvement")
        
        # Predictive assignment
        improvements.append("Develop predictive assignment model based on historical success patterns")
        
        # Capacity planning
        improvements.append("Add capacity planning dashboard with workload forecasting")
        
        # Performance monitoring
        improvements.append("Real-time specialist performance monitoring with alerting")
        
        # Cross-training support
        improvements.append("Automated cross-training recommendations based on workload patterns")
        
        return improvements[:10]  # Return top 10 improvements
    
    def _save_analysis_result(self, result: SpecialistAssignmentResult):
        """Save analysis results to file system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save current analysis
            current_file = self.analysis_path / "current_specialist_analysis.json"
            with open(current_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            # Save timestamped version
            historical_file = self.analysis_path / f"specialist_analysis_{timestamp}.json"
            with open(historical_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            logger.info(f"Specialist analysis results saved to {current_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def generate_html_report(self, result: SpecialistAssignmentResult) -> str:
        """Generate HTML report for specialist assignment analysis"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Specialist Assignment Accuracy Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .report {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; color: #2c3e50; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .specialists-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .specialist-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .specialist-name {{ font-size: 1.1em; font-weight: bold; color: #2c3e50; margin-bottom: 15px; }}
        .metrics-row {{ display: flex; justify-content: space-between; margin: 6px 0; font-size: 0.9em; }}
        .metric-label {{ color: #7f8c8d; }}
        .metric-value {{ font-weight: bold; }}
        .accuracy-high {{ color: #27ae60; }}
        .accuracy-medium {{ color: #f39c12; }}
        .accuracy-low {{ color: #e74c3c; }}
        .patterns-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .pattern-item {{ border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; background: #f8f9fa; }}
        .pattern-header {{ font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
        .workload-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .workload-item {{ display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px; }}
        .load-balanced {{ border-left: 4px solid #27ae60; }}
        .load-imbalanced {{ border-left: 4px solid #e74c3c; }}
        .recommendations-section {{ background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; margin-bottom: 20px; }}
        .training-section {{ background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 20px; }}
        .improvements-section {{ background: #d1ecf1; padding: 20px; border-radius: 8px; border-left: 4px solid #17a2b8; margin-bottom: 20px; }}
        .recommendation-item {{ padding: 8px 0; }}
        .footer {{ text-align: center; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>Specialist Assignment Accuracy Analysis</h1>
            <p>Analysis Period: {result.analysis_period} | Generated: {result.timestamp}</p>
        </div>
        
        <div class="summary">
            <h2>Overall Assignment Performance</h2>
            <div class="metrics-row">
                <span class="metric-label">Total Assignments:</span>
                <span class="metric-value">{result.total_assignments}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-label">Overall Accuracy Rate:</span>
                <span class="metric-value accuracy-{'high' if result.overall_accuracy_rate >= 0.9 else 'medium' if result.overall_accuracy_rate >= 0.8 else 'low'}">{result.overall_accuracy_rate:.1%}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-label">Active Specialists:</span>
                <span class="metric-value">{len(result.specialist_metrics)}</span>
            </div>
        </div>
        
        <div class="specialists-grid">
            {self._generate_specialist_cards_html(result.specialist_metrics)}
        </div>
        
        <div class="patterns-section">
            <h2>Assignment Patterns</h2>
            {self._generate_patterns_html(result.assignment_patterns)}
        </div>
        
        <div class="workload-section">
            <h2>Workload Analysis</h2>
            {self._generate_workload_html(result.workload_analysis)}
        </div>
        
        <div class="recommendations-section">
            <h2>Optimization Opportunities</h2>
            {''.join([f'<div class="recommendation-item">• {opp}</div>' for opp in result.optimization_opportunities])}
        </div>
        
        <div class="training-section">
            <h2>Training Recommendations</h2>
            {''.join([f'<div class="recommendation-item">• {rec}</div>' for rec in result.training_recommendations])}
        </div>
        
        <div class="improvements-section">
            <h2>System Improvements</h2>
            {''.join([f'<div class="recommendation-item">• {imp}</div>' for imp in result.system_improvements])}
        </div>
        
        <div class="footer">
            <p>Generated by RIF Specialist Assignment Tracker</p>
            <p>Issue #94 - Phase 2: Analytics Dashboard Implementation</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_specialist_cards_html(self, specialists: List[SpecialistMetrics]) -> str:
        """Generate HTML cards for specialist metrics"""
        html = ""
        
        for specialist in specialists:
            accuracy_class = (
                'accuracy-high' if specialist.accuracy_rate >= 0.9 
                else 'accuracy-medium' if specialist.accuracy_rate >= 0.8 
                else 'accuracy-low'
            )
            
            html += f"""
            <div class="specialist-card">
                <div class="specialist-name">{specialist.specialist_type.replace('-', ' ').title()}</div>
                
                <div class="metrics-row">
                    <span class="metric-label">Assignments:</span>
                    <span class="metric-value">{specialist.total_assignments}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Accuracy Rate:</span>
                    <span class="metric-value {accuracy_class}">{specialist.accuracy_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{specialist.success_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Avg Response Time:</span>
                    <span class="metric-value">{specialist.avg_response_time_hours:.1f}h</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Escalation Rate:</span>
                    <span class="metric-value">{specialist.escalation_rate:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Quality Score:</span>
                    <span class="metric-value">{specialist.quality_improvement_score:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Assignment Confidence:</span>
                    <span class="metric-value">{specialist.assignment_confidence:.1%}</span>
                </div>
                
                <div class="metrics-row">
                    <span class="metric-label">Workload Efficiency:</span>
                    <span class="metric-value">{specialist.workload_efficiency:.1%}</span>
                </div>
            </div>
            """
        
        return html
    
    def _generate_patterns_html(self, patterns: List[AssignmentPattern]) -> str:
        """Generate HTML for assignment patterns"""
        if not patterns:
            return "<p>No significant assignment patterns detected.</p>"
        
        html = ""
        for pattern in patterns:
            accuracy_class = (
                'accuracy-high' if pattern.accuracy_rate >= 0.9
                else 'accuracy-medium' if pattern.accuracy_rate >= 0.8
                else 'accuracy-low'
            )
            
            html += f"""
            <div class="pattern-item">
                <div class="pattern-header">{pattern.description}</div>
                <div style="display: flex; gap: 20px; margin: 8px 0; font-size: 0.9em;">
                    <span>Frequency: {pattern.frequency}</span>
                    <span>Accuracy: <span class="{accuracy_class}">{pattern.accuracy_rate:.1%}</span></span>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Common Specialists:</strong> {', '.join(pattern.typical_contexts)}
                </div>
                {f'<div style="margin-top: 5px;"><strong>Success Factors:</strong> {", ".join(pattern.success_factors)}</div>' if pattern.success_factors else ''}
                {f'<div style="margin-top: 5px;"><strong>Failure Factors:</strong> {", ".join(pattern.failure_factors)}</div>' if pattern.failure_factors else ''}
                <div style="margin-top: 5px;"><strong>Recommendations:</strong></div>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    {''.join([f'<li>{rec}</li>' for rec in pattern.recommendations])}
                </ul>
            </div>
            """
        
        return html
    
    def _generate_workload_html(self, workload_analyses: List[WorkloadAnalysis]) -> str:
        """Generate HTML for workload analysis"""
        if not workload_analyses:
            return "<p>No workload data available.</p>"
        
        html = ""
        for workload in workload_analyses:
            balance_class = 'load-balanced' if workload.load_balance_score >= 0.8 else 'load-imbalanced'
            
            html += f"""
            <div class="workload-item {balance_class}">
                <div>
                    <strong>{workload.specialist_type.replace('-', ' ').title()}</strong><br>
                    <span style="font-size: 0.9em; color: #7f8c8d;">
                        Load: {workload.current_load} | Optimal: {workload.optimal_load_range[0]}-{workload.optimal_load_range[1]} | 
                        Balance Score: {workload.load_balance_score:.1%}
                    </span>
                    {f'<br><span style="font-size: 0.8em;">Peak: {", ".join(workload.peak_periods)}</span>' if workload.peak_periods else ''}
                </div>
                <div style="text-align: right; font-size: 0.8em;">
                    {''.join([f'<div>{rec}</div>' for rec in workload.capacity_recommendations[:2]])}
                </div>
            </div>
            """
        
        return html

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Analyze Specialist Assignment Accuracy')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--output', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--knowledge-path', default='/Users/cal/DEV/RIF/knowledge', help='Path to knowledge base')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SpecialistAssignmentTracker(args.knowledge_path)
    
    # Run analysis
    print(f"Analyzing specialist assignment accuracy for last {args.days} days...")
    result = tracker.analyze_specialist_assignments(args.days)
    
    # Output results
    if args.output in ['json', 'both']:
        print("\n=== SPECIALIST ASSIGNMENT ANALYSIS JSON ===")
        print(json.dumps(asdict(result), indent=2))
    
    if args.output in ['html', 'both']:
        html_report = tracker.generate_html_report(result)
        html_file = Path(args.knowledge_path) / "analysis" / "specialist_assignment_report.html"
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"\n=== HTML REPORT GENERATED ===")
        print(f"Report saved to: {html_file}")
        print(f"Open in browser: file://{html_file}")
    
    # Display summary
    print(f"\n=== SPECIALIST ASSIGNMENT SUMMARY ===")
    print(f"Analysis Period: {result.analysis_period}")
    print(f"Total Assignments: {result.total_assignments}")
    print(f"Overall Accuracy Rate: {result.overall_accuracy_rate:.1%}")
    print(f"Active Specialists: {len(result.specialist_metrics)}")
    
    print(f"\nTop Performing Specialists:")
    for i, specialist in enumerate(sorted(result.specialist_metrics, key=lambda x: x.accuracy_rate, reverse=True)[:3], 1):
        print(f"  {i}. {specialist.specialist_type}: {specialist.accuracy_rate:.1%} accuracy, {specialist.total_assignments} assignments")
    
    print(f"\nKey Optimization Opportunities:")
    for i, opp in enumerate(result.optimization_opportunities[:3], 1):
        print(f"  {i}. {opp}")
    
    print(f"\nTop Training Needs:")
    for i, training in enumerate(result.training_recommendations[:3], 1):
        print(f"  {i}. {training}")

if __name__ == "__main__":
    main()