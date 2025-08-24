#!/usr/bin/env python3
"""
RIF Dynamic Orchestrator Core System
Issue #52-56: Dynamic Orchestration Implementation

This module provides the core dynamic orchestration system with state transitions,
adaptive agent selection, and workflow loop-back mechanisms.
"""

import json
import time
import uuid
import copy
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from orchestrator_integration import IntegratedOrchestratorSystem
except ImportError:
    # Fallback for testing
    IntegratedOrchestratorSystem = None

try:
    from github_state_manager import GitHubStateManager
except ImportError:
    # Fallback for testing
    GitHubStateManager = None

try:
    import sqlite3
    import duckdb
except ImportError:
    # Fallback for testing without database dependencies
    sqlite3 = None
    duckdb = None

try:
    from workflow_loopback_manager import WorkflowLoopbackManager, ValidationResult
except ImportError:
    # Fallback for testing
    WorkflowLoopbackManager = None
    ValidationResult = None

@dataclass
class ContextModel:
    """Rich context model for intelligent decision making."""
    github_issues: List[int] = field(default_factory=list)
    complexity: str = 'medium'
    validation_results: Optional[Any] = None
    retry_count: int = 0
    error_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_level: str = 'standard'
    priority: int = 1
    workflow_type: str = 'standard'
    agent_performance_history: Dict[str, List[float]] = field(default_factory=dict)
    pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'github_issues': self.github_issues,
            'complexity': self.complexity,
            'validation_results': str(self.validation_results) if self.validation_results else None,
            'retry_count': self.retry_count,
            'error_history': self.error_history,
            'performance_metrics': self.performance_metrics,
            'security_level': self.security_level,
            'priority': self.priority,
            'workflow_type': self.workflow_type,
            'agent_performance_history': self.agent_performance_history,
            'pattern_matches': self.pattern_matches,
            'confidence_scores': self.confidence_scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextModel':
        """Create from dictionary data."""
        return cls(
            github_issues=data.get('github_issues', []),
            complexity=data.get('complexity', 'medium'),
            validation_results=data.get('validation_results'),
            retry_count=data.get('retry_count', 0),
            error_history=data.get('error_history', []),
            performance_metrics=data.get('performance_metrics', {}),
            security_level=data.get('security_level', 'standard'),
            priority=data.get('priority', 1),
            workflow_type=data.get('workflow_type', 'standard'),
            agent_performance_history=data.get('agent_performance_history', {}),
            pattern_matches=data.get('pattern_matches', []),
            confidence_scores=data.get('confidence_scores', {})
        )


class ContextModelingEngine:
    """Engine for modeling and understanding workflow context."""
    
    def __init__(self):
        """Initialize the context modeling engine."""
        self.pattern_cache = {}
        self.semantic_analyzers = {
            'complexity': self._analyze_complexity_factors,
            'security': self._analyze_security_factors,
            'performance': self._analyze_performance_factors,
            'risk': self._analyze_risk_factors
        }
    
    def model_context(self, context_data: Dict[str, Any]) -> ContextModel:
        """
        Create rich context model from basic context data.
        
        Args:
            context_data: Raw context information
            
        Returns:
            Enhanced context model
        """
        # Create base model
        context_model = ContextModel.from_dict(context_data)
        
        # Enhance with semantic analysis
        context_model = self._enhance_with_semantic_analysis(context_model, context_data)
        
        # Add pattern matching
        context_model.pattern_matches = self._find_pattern_matches(context_model)
        
        # Calculate confidence scores
        context_model.confidence_scores = self._calculate_confidence_scores(context_model)
        
        return context_model
    
    def _enhance_with_semantic_analysis(self, context_model: ContextModel, raw_data: Dict[str, Any]) -> ContextModel:
        """Enhance context model with semantic analysis."""
        for analyzer_name, analyzer_func in self.semantic_analyzers.items():
            try:
                analysis_result = analyzer_func(context_model, raw_data)
                if analysis_result:
                    setattr(context_model, f'{analyzer_name}_analysis', analysis_result)
            except Exception as e:
                context_model.error_history.append(f'Semantic analysis error in {analyzer_name}: {str(e)}')
        
        return context_model
    
    def _analyze_complexity_factors(self, context: ContextModel, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complexity indicators in the context."""
        factors = {
            'issue_count': len(context.github_issues),
            'error_frequency': len(context.error_history),
            'retry_rate': context.retry_count,
            'workflow_complexity': self._assess_workflow_complexity(context.workflow_type)
        }
        
        # Calculate overall complexity score
        complexity_score = (
            min(factors['issue_count'] / 10, 1.0) * 0.3 +
            min(factors['error_frequency'] / 5, 1.0) * 0.3 +
            min(factors['retry_rate'] / 3, 1.0) * 0.2 +
            factors['workflow_complexity'] * 0.2
        )
        
        factors['complexity_score'] = complexity_score
        factors['recommended_complexity'] = self._score_to_complexity(complexity_score)
        
        return factors
    
    def _analyze_security_factors(self, context: ContextModel, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security implications in the context."""
        security_indicators = {
            'security_critical': raw_data.get('security_critical', False),
            'compliance_required': raw_data.get('compliance_required', False),
            'external_dependencies': raw_data.get('external_dependencies', []),
            'data_sensitivity': raw_data.get('data_sensitivity', 'low')
        }
        
        # Calculate security risk score
        risk_score = 0.0
        if security_indicators['security_critical']:
            risk_score += 0.4
        if security_indicators['compliance_required']:
            risk_score += 0.3
        if len(security_indicators['external_dependencies']) > 5:
            risk_score += 0.2
        if security_indicators['data_sensitivity'] in ['high', 'critical']:
            risk_score += 0.1
        
        security_indicators['security_risk_score'] = min(risk_score, 1.0)
        security_indicators['requires_security_agent'] = risk_score > 0.3
        
        return security_indicators
    
    def _analyze_performance_factors(self, context: ContextModel, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance requirements and constraints."""
        performance_factors = {
            'urgency': context.priority,
            'estimated_duration': raw_data.get('estimated_time_hours', 8),
            'parallel_processing': raw_data.get('parallel_processing', False),
            'resource_constraints': raw_data.get('resource_constraints', {})
        }
        
        # Analyze historical performance
        if context.agent_performance_history:
            avg_performance = {}
            for agent, scores in context.agent_performance_history.items():
                if scores:
                    avg_performance[agent] = sum(scores) / len(scores)
            performance_factors['historical_performance'] = avg_performance
        
        return performance_factors
    
    def _analyze_risk_factors(self, context: ContextModel, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk factors in the workflow context."""
        risk_factors = {
            'failure_probability': min(context.retry_count / 5.0, 1.0),
            'error_patterns': self._identify_error_patterns(context.error_history),
            'complexity_risk': self._assess_complexity_risk(context.complexity),
            'timeline_risk': raw_data.get('timeline_pressure', 'normal')
        }
        
        # Calculate overall risk score
        risk_score = (
            risk_factors['failure_probability'] * 0.4 +
            len(risk_factors['error_patterns']) / 10.0 * 0.3 +
            risk_factors['complexity_risk'] * 0.3
        )
        
        risk_factors['overall_risk_score'] = min(risk_score, 1.0)
        risk_factors['risk_level'] = self._score_to_risk_level(risk_score)
        
        return risk_factors
    
    def _assess_workflow_complexity(self, workflow_type: str) -> float:
        """Assess workflow complexity based on type."""
        complexity_mapping = {
            'simple': 0.1,
            'standard': 0.3,
            'complex': 0.6,
            'enterprise': 0.8,
            'experimental': 0.9
        }
        return complexity_mapping.get(workflow_type.lower(), 0.3)
    
    def _score_to_complexity(self, score: float) -> str:
        """Convert complexity score to complexity level."""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _assess_complexity_risk(self, complexity: str) -> float:
        """Assess risk based on complexity level."""
        risk_mapping = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'very_high': 0.8
        }
        return risk_mapping.get(complexity, 0.3)
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _identify_error_patterns(self, error_history: List[str]) -> List[str]:
        """Identify common patterns in error history."""
        patterns = []
        
        # Common error patterns
        pattern_keywords = {
            'timeout': r'timeout|timed out|exceeded',
            'memory': r'memory|out of memory|oom',
            'network': r'network|connection|connectivity',
            'authentication': r'auth|authentication|permission|unauthorized',
            'validation': r'validation|invalid|failed.*test',
            'dependency': r'dependency|import|module.*not.*found'
        }
        
        for pattern_name, pattern_regex in pattern_keywords.items():
            if any(re.search(pattern_regex, error.lower()) for error in error_history):
                patterns.append(pattern_name)
        
        return patterns
    
    def _find_pattern_matches(self, context: ContextModel) -> List[Dict[str, Any]]:
        """Find matching patterns from knowledge base."""
        # This is a simplified pattern matching - in production would query actual knowledge base
        matches = []
        
        # Mock pattern matching based on context characteristics
        if context.complexity == 'high' and context.retry_count > 2:
            matches.append({
                'pattern_name': 'High Complexity Retry Pattern',
                'confidence': 0.8,
                'recommendations': ['Add architect agent', 'Increase planning depth']
            })
        
        if hasattr(context, 'security_analysis') and getattr(context, 'security_analysis', {}).get('requires_security_agent'):
            matches.append({
                'pattern_name': 'Security Critical Workflow Pattern',
                'confidence': 0.9,
                'recommendations': ['Include security agent', 'Add compliance checks']
            })
        
        return matches
    
    def _calculate_confidence_scores(self, context: ContextModel) -> Dict[str, float]:
        """Calculate confidence scores for various decisions."""
        scores = {}
        
        # Base confidence from data completeness
        data_completeness = self._assess_data_completeness(context)
        scores['data_quality'] = data_completeness
        
        # Pattern matching confidence
        if context.pattern_matches:
            avg_pattern_confidence = sum(p['confidence'] for p in context.pattern_matches) / len(context.pattern_matches)
            scores['pattern_matching'] = avg_pattern_confidence
        else:
            scores['pattern_matching'] = 0.5
        
        # Historical performance confidence
        if context.agent_performance_history:
            scores['historical_performance'] = 0.8
        else:
            scores['historical_performance'] = 0.5
        
        # Overall decision confidence
        scores['overall'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _assess_data_completeness(self, context: ContextModel) -> float:
        """Assess completeness of context data."""
        required_fields = ['github_issues', 'complexity', 'workflow_type']
        optional_fields = ['validation_results', 'performance_metrics', 'priority']
        
        required_score = sum(1 for field in required_fields if getattr(context, field, None)) / len(required_fields)
        optional_score = sum(1 for field in optional_fields if getattr(context, field, None)) / len(optional_fields)
        
        return required_score * 0.7 + optional_score * 0.3


class ValidationResultAnalyzer:
    """Analyzer for validation results to determine optimal next actions."""
    
    def __init__(self):
        """Initialize the validation result analyzer."""
        self.failure_patterns = {
            'architectural': ['architecture', 'design', 'structure', 'coupling'],
            'requirements': ['requirement', 'specification', 'acceptance', 'criteria'],
            'implementation': ['bug', 'error', 'implementation', 'logic', 'syntax'],
            'performance': ['performance', 'timeout', 'slow', 'memory', 'resource'],
            'security': ['security', 'vulnerability', 'authorization', 'authentication'],
            'integration': ['integration', 'interface', 'compatibility', 'dependency']
        }
    
    def analyze_validation_results(self, validation_results: Any, context: ContextModel) -> Dict[str, Any]:
        """
        Analyze validation results to determine optimal recovery strategy.
        
        Args:
            validation_results: Validation result object or data
            context: Current workflow context
            
        Returns:
            Analysis results with recommendations
        """
        if not validation_results:
            return {'status': 'no_results', 'recommended_action': 'continue'}
        
        # Extract failure information
        failure_info = self._extract_failure_info(validation_results)
        
        # Categorize failures
        failure_categories = self._categorize_failures(failure_info)
        
        # Determine optimal next state
        recommended_state = self._determine_optimal_state(failure_categories, context)
        
        # Generate recovery strategy
        recovery_strategy = self._generate_recovery_strategy(failure_categories, context)
        
        return {
            'status': 'analyzed',
            'failure_info': failure_info,
            'failure_categories': failure_categories,
            'recommended_state': recommended_state,
            'recovery_strategy': recovery_strategy,
            'confidence': self._calculate_analysis_confidence(failure_info, context)
        }
    
    def _extract_failure_info(self, validation_results: Any) -> Dict[str, Any]:
        """Extract failure information from validation results."""
        failure_info = {
            'success': True,
            'errors': [],
            'warnings': [],
            'details': 'No specific details available'
        }
        
        # Handle different validation result types
        if hasattr(validation_results, 'success'):
            failure_info['success'] = validation_results.success
        
        if hasattr(validation_results, 'errors'):
            failure_info['errors'] = validation_results.errors if validation_results.errors else []
        
        if hasattr(validation_results, 'warnings'):
            failure_info['warnings'] = validation_results.warnings if validation_results.warnings else []
        
        if hasattr(validation_results, 'details'):
            failure_info['details'] = validation_results.details
        
        # Handle dictionary-like validation results
        if isinstance(validation_results, dict):
            failure_info.update(validation_results)
        
        # Handle string validation results
        if isinstance(validation_results, str):
            failure_info['details'] = validation_results
            failure_info['success'] = 'fail' not in validation_results.lower() and 'error' not in validation_results.lower()
        
        return failure_info
    
    def _categorize_failures(self, failure_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize failures by type for optimal recovery strategy."""
        categories = {category: [] for category in self.failure_patterns.keys()}
        
        # Analyze error messages
        all_messages = []
        all_messages.extend(failure_info.get('errors', []))
        all_messages.extend(failure_info.get('warnings', []))
        if failure_info.get('details'):
            all_messages.append(failure_info['details'])
        
        for message in all_messages:
            message_lower = str(message).lower()
            for category, keywords in self.failure_patterns.items():
                if any(keyword in message_lower for keyword in keywords):
                    categories[category].append(str(message))
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _determine_optimal_state(self, failure_categories: Dict[str, List[str]], context: ContextModel) -> str:
        """Determine optimal next state based on failure analysis."""
        if not failure_categories:
            return 'learning'  # No failures, proceed to learning
        
        # Prioritize recovery states based on failure types
        state_priorities = {
            'architectural': 'architecting',
            'requirements': 'analyzing', 
            'implementation': 'implementing',
            'performance': 'implementing',
            'security': 'implementing',
            'integration': 'implementing'
        }
        
        # Find highest priority failure category
        priority_order = ['architectural', 'requirements', 'implementation', 'performance', 'security', 'integration']
        
        for category in priority_order:
            if category in failure_categories:
                return state_priorities[category]
        
        return 'implementing'  # Default fallback
    
    def _generate_recovery_strategy(self, failure_categories: Dict[str, List[str]], context: ContextModel) -> Dict[str, Any]:
        """Generate recovery strategy based on failure analysis."""
        strategy = {
            'primary_actions': [],
            'secondary_actions': [],
            'risk_mitigation': [],
            'estimated_effort': 'medium'
        }
        
        # Generate actions based on failure categories
        if 'architectural' in failure_categories:
            strategy['primary_actions'].append('Redesign system architecture')
            strategy['secondary_actions'].append('Include RIF-Architect agent')
            strategy['estimated_effort'] = 'high'
        
        if 'requirements' in failure_categories:
            strategy['primary_actions'].append('Re-analyze requirements')
            strategy['secondary_actions'].append('Stakeholder consultation')
        
        if 'implementation' in failure_categories:
            strategy['primary_actions'].append('Fix implementation issues')
            strategy['secondary_actions'].append('Code review and refactoring')
        
        if 'performance' in failure_categories:
            strategy['primary_actions'].append('Performance optimization')
            strategy['secondary_actions'].append('Resource allocation review')
        
        if 'security' in failure_categories:
            strategy['primary_actions'].append('Security remediation')
            strategy['secondary_actions'].append('Include RIF-Security agent')
            strategy['risk_mitigation'].append('Security audit')
        
        # Add context-based adjustments
        if context.retry_count > 2:
            strategy['risk_mitigation'].append('Consider alternative approach')
            strategy['estimated_effort'] = 'high'
        
        return strategy
    
    def _calculate_analysis_confidence(self, failure_info: Dict[str, Any], context: ContextModel) -> float:
        """Calculate confidence in the analysis results."""
        confidence = 0.7  # Base confidence
        
        # Adjust based on information quality
        if failure_info.get('details') and len(str(failure_info['details'])) > 50:
            confidence += 0.1
        
        if failure_info.get('errors') and len(failure_info['errors']) > 0:
            confidence += 0.1
        
        # Adjust based on context quality
        if context.retry_count > 0:
            confidence += 0.05  # More data points improve confidence
        
        if len(context.error_history) > 3:
            confidence += 0.05  # Pattern recognition improves with history
        
        return min(confidence, 1.0)


class EnhancedStateAnalyzer:
    """Enhanced state analyzer with intelligent decision making capabilities."""
    
    def __init__(self, context_engine: ContextModelingEngine, validation_analyzer: ValidationResultAnalyzer):
        """Initialize enhanced state analyzer."""
        self.context_engine = context_engine
        self.validation_analyzer = validation_analyzer
        self.decision_cache = {}
        self.pattern_weights = {
            'complexity': 0.3,
            'security': 0.2,
            'performance': 0.2,
            'risk': 0.2,
            'history': 0.1
        }
    
    def analyze_and_determine_next_state(self, current_state: str, context_data: Dict[str, Any], 
                                       workflow_graph: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Perform intelligent analysis to determine optimal next state.
        
        Args:
            current_state: Current workflow state
            context_data: Raw context data
            workflow_graph: Workflow state graph
            
        Returns:
            Tuple of (next_state, analysis_details)
        """
        # Create rich context model
        context_model = self.context_engine.model_context(context_data)
        
        # Get possible transitions
        current_state_config = workflow_graph['states'].get(current_state, {})
        possible_transitions = current_state_config.get('transitions', [])
        
        if not possible_transitions:
            return 'completed', {'reason': 'No valid transitions available'}
        
        # Analyze validation results if available
        validation_analysis = None
        if context_model.validation_results:
            validation_analysis = self.validation_analyzer.analyze_validation_results(
                context_model.validation_results, context_model
            )
        
        # Apply intelligent decision logic
        next_state, analysis_details = self._apply_intelligent_decision_logic(
            current_state, possible_transitions, context_model, validation_analysis, workflow_graph
        )
        
        return next_state, analysis_details
    
    def _apply_intelligent_decision_logic(self, current_state: str, possible_transitions: List[str], 
                                        context_model: ContextModel, validation_analysis: Optional[Dict[str, Any]], 
                                        workflow_graph: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Apply sophisticated decision logic with pattern recognition."""
        decision_factors = {
            'validation_driven': self._evaluate_validation_factor(validation_analysis, possible_transitions),
            'complexity_driven': self._evaluate_complexity_factor(context_model, possible_transitions, workflow_graph),
            'pattern_driven': self._evaluate_pattern_factor(context_model, possible_transitions),
            'risk_driven': self._evaluate_risk_factor(context_model, possible_transitions),
            'performance_driven': self._evaluate_performance_factor(context_model, possible_transitions)
        }
        
        # Weight and combine factors
        state_scores = {}
        for state in possible_transitions:
            score = 0.0
            for factor_name, factor_result in decision_factors.items():
                if state in factor_result['recommendations']:
                    weight = self.pattern_weights.get(factor_name.replace('_driven', ''), 0.1)
                    score += factor_result['recommendations'][state] * weight
            state_scores[state] = score
        
        # Select highest scoring state
        best_state = max(state_scores.keys(), key=lambda s: state_scores[s]) if state_scores else possible_transitions[0]
        
        # Prepare analysis details
        analysis_details = {
            'decision_factors': decision_factors,
            'state_scores': state_scores,
            'selected_state': best_state,
            'confidence': context_model.confidence_scores.get('overall', 0.7),
            'reasoning': self._generate_decision_reasoning(best_state, decision_factors, context_model)
        }
        
        return best_state, analysis_details
    
    def _evaluate_validation_factor(self, validation_analysis: Optional[Dict[str, Any]], 
                                  possible_transitions: List[str]) -> Dict[str, Any]:
        """Evaluate validation-driven decision factor."""
        if not validation_analysis or validation_analysis.get('status') != 'analyzed':
            return {'recommendations': {state: 0.5 for state in possible_transitions}, 'confidence': 0.3}
        
        recommended_state = validation_analysis.get('recommended_state')
        recommendations = {state: 0.3 for state in possible_transitions}
        
        if recommended_state in possible_transitions:
            recommendations[recommended_state] = 0.9
        
        return {
            'recommendations': recommendations,
            'confidence': validation_analysis.get('confidence', 0.7),
            'details': validation_analysis
        }
    
    def _evaluate_complexity_factor(self, context_model: ContextModel, possible_transitions: List[str], 
                                  workflow_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate complexity-driven decision factor."""
        complexity_rules = workflow_graph.get('decision_rules', {}).get('complexity_based', {})
        complexity_path = complexity_rules.get(context_model.complexity, [])
        
        recommendations = {state: 0.3 for state in possible_transitions}
        
        # Favor states in the complexity path
        for state in possible_transitions:
            if state in complexity_path:
                # Find position in path to determine priority
                try:
                    current_index = complexity_path.index(context_model.complexity if hasattr(context_model, 'current_state') else 'analyzing')
                    state_index = complexity_path.index(state)
                    if state_index == current_index + 1:  # Next logical step
                        recommendations[state] = 0.9
                    elif state_index > current_index:
                        recommendations[state] = 0.7
                except ValueError:
                    recommendations[state] = 0.5
        
        return {
            'recommendations': recommendations,
            'confidence': 0.8,
            'complexity_path': complexity_path
        }
    
    def _evaluate_pattern_factor(self, context_model: ContextModel, possible_transitions: List[str]) -> Dict[str, Any]:
        """Evaluate pattern-driven decision factor."""
        recommendations = {state: 0.4 for state in possible_transitions}
        
        # Apply pattern-based recommendations
        for pattern in context_model.pattern_matches:
            pattern_recommendations = pattern.get('recommendations', [])
            for rec in pattern_recommendations:
                if 'architect' in rec.lower() and 'architecting' in possible_transitions:
                    recommendations['architecting'] = 0.8
                elif 'security' in rec.lower() and 'implementing' in possible_transitions:
                    recommendations['implementing'] = 0.7
                elif 'planning' in rec.lower() and 'planning' in possible_transitions:
                    recommendations['planning'] = 0.8
        
        return {
            'recommendations': recommendations,
            'confidence': sum(p['confidence'] for p in context_model.pattern_matches) / len(context_model.pattern_matches) if context_model.pattern_matches else 0.5,
            'patterns_applied': len(context_model.pattern_matches)
        }
    
    def _evaluate_risk_factor(self, context_model: ContextModel, possible_transitions: List[str]) -> Dict[str, Any]:
        """Evaluate risk-driven decision factor."""
        recommendations = {state: 0.5 for state in possible_transitions}
        
        # High retry count suggests need for different approach
        if context_model.retry_count > 2:
            if 'analyzing' in possible_transitions:
                recommendations['analyzing'] = 0.8  # Go back to analysis
            elif 'planning' in possible_transitions:
                recommendations['planning'] = 0.7  # Re-plan approach
        
        # Error patterns suggest specific actions
        if hasattr(context_model, 'risk_analysis'):
            risk_analysis = getattr(context_model, 'risk_analysis', {})
            error_patterns = risk_analysis.get('error_patterns', [])
            
            if 'timeout' in error_patterns and 'implementing' in possible_transitions:
                recommendations['implementing'] = 0.3  # Avoid implementation if timeout issues
            
            if 'dependency' in error_patterns and 'planning' in possible_transitions:
                recommendations['planning'] = 0.8  # Re-plan dependencies
        
        return {
            'recommendations': recommendations,
            'confidence': 0.6,
            'retry_count': context_model.retry_count
        }
    
    def _evaluate_performance_factor(self, context_model: ContextModel, possible_transitions: List[str]) -> Dict[str, Any]:
        """Evaluate performance-driven decision factor."""
        recommendations = {state: 0.5 for state in possible_transitions}
        
        # High priority suggests fast path
        if context_model.priority > 3:
            # Prefer direct paths, avoid extra planning
            if 'implementing' in possible_transitions:
                recommendations['implementing'] = 0.8
            if 'validating' in possible_transitions:
                recommendations['validating'] = 0.7
        
        # Low priority allows thorough approach
        elif context_model.priority < 2:
            if 'planning' in possible_transitions:
                recommendations['planning'] = 0.7
            if 'architecting' in possible_transitions:
                recommendations['architecting'] = 0.6
        
        return {
            'recommendations': recommendations,
            'confidence': 0.7,
            'priority_level': context_model.priority
        }
    
    def _generate_decision_reasoning(self, selected_state: str, decision_factors: Dict[str, Any], 
                                   context_model: ContextModel) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []
        
        # Find dominant factors
        for factor_name, factor_data in decision_factors.items():
            if selected_state in factor_data['recommendations']:
                score = factor_data['recommendations'][selected_state]
                if score > 0.7:
                    factor_readable = factor_name.replace('_driven', '').replace('_', ' ').title()
                    reasons.append(f"{factor_readable} analysis strongly supports {selected_state}")
                elif score > 0.6:
                    factor_readable = factor_name.replace('_driven', '').replace('_', ' ').title()
                    reasons.append(f"{factor_readable} analysis moderately supports {selected_state}")
        
        if not reasons:
            reasons.append(f"Default selection: {selected_state}")
        
        # Add context-specific reasoning
        if context_model.retry_count > 1:
            reasons.append(f"Retry count ({context_model.retry_count}) suggests need for different approach")
        
        if context_model.complexity in ['high', 'very_high']:
            reasons.append(f"High complexity ({context_model.complexity}) requires careful state management")
        
        return "; ".join(reasons)


class DynamicOrchestrator:
    """
    Core dynamic orchestration system that manages workflow state transitions,
    agent selection, and decision-making with adaptive capabilities.
    """
    
    def __init__(self, workflow_graph: Optional[Dict] = None, db_path: str = 'knowledge/orchestration.duckdb'):
        """
        Initialize the dynamic orchestrator.
        
        Args:
            workflow_graph: Custom workflow state graph
            db_path: Database path for persistence
        """
        self.workflow_graph = workflow_graph or self._get_default_workflow_graph()
        self.current_state = 'initialized'
        self.context = {}
        self.history = []
        self.agent_assignments = {}
        self.confidence_threshold = 0.7
        self.max_retries = 3
        
        # Initialize integrated system if available
        if IntegratedOrchestratorSystem:
            self.integration = IntegratedOrchestratorSystem(db_path)
        else:
            self.integration = None
            
        # Initialize GitHub state manager for label synchronization
        if GitHubStateManager:
            self.github_state_manager = GitHubStateManager()
        else:
            self.github_state_manager = None
            
        # Initialize enhanced analysis components
        self.context_engine = ContextModelingEngine()
        self.validation_analyzer = ValidationResultAnalyzer()
        self.state_analyzer = EnhancedStateAnalyzer(self.context_engine, self.validation_analyzer)
        
        # Initialize enhanced agent selection system
        self.learning_agent_selector = LearningAgentSelector()
        self.adaptive_agent_selector = AdaptiveAgentSelector()  # Backward compatibility
        
        # Initialize loop-back intelligence components
        self.failure_pattern_analyzer = FailurePatternAnalyzer()
        self.loop_back_decision_engine = LoopBackDecisionEngine(self.failure_pattern_analyzer)
        self.recovery_strategy_selector = RecoveryStrategySelector()
        
        # Initialize workflow loop-back manager
        try:
            from workflow_loopback_manager import WorkflowLoopbackManager
            self.loopback_manager = WorkflowLoopbackManager(
                max_loops=self.max_retries,
                persistence_system=getattr(self.integration, 'persistence', None) if self.integration else None
            )
        except ImportError:
            self.loopback_manager = None
            
        # Performance tracking with enhanced metrics
        self.metrics = {
            'transitions': 0,
            'decisions': 0,
            'retries': 0,
            'errors': 0,
            'intelligent_decisions': 0,
            'pattern_matches': 0,
            'avg_decision_time': 0.0,
            'decision_confidence_avg': 0.0
        }
    
    def _get_default_workflow_graph(self) -> Dict[str, Any]:
        """Get the default RIF workflow state graph."""
        return {
            'states': {
                'initialized': {
                    'transitions': ['analyzing'],
                    'agents': ['RIF-Analyst'],
                    'requirements': []
                },
                'analyzing': {
                    'transitions': ['planning', 'implementing', 'failed'],
                    'agents': ['RIF-Analyst'],
                    'requirements': ['github_issues']
                },
                'planning': {
                    'transitions': ['architecting', 'implementing', 'analyzing'],
                    'agents': ['RIF-Planner'],
                    'requirements': ['requirements_analysis']
                },
                'architecting': {
                    'transitions': ['implementing', 'planning'],
                    'agents': ['RIF-Architect'],
                    'requirements': ['high_complexity_detected']
                },
                'implementing': {
                    'transitions': ['validating', 'analyzing'],
                    'agents': ['RIF-Implementer'],
                    'requirements': ['implementation_plan']
                },
                'validating': {
                    'transitions': ['learning', 'implementing'],
                    'agents': ['RIF-Validator'],
                    'requirements': ['code_complete']
                },
                'learning': {
                    'transitions': ['completed'],
                    'agents': ['RIF-Learner'],
                    'requirements': ['validation_passed']
                },
                'completed': {
                    'transitions': [],
                    'agents': [],
                    'requirements': []
                },
                'failed': {
                    'transitions': ['analyzing'],
                    'agents': ['RIF-Analyst'],
                    'requirements': []
                }
            },
            'decision_rules': {
                'complexity_based': {
                    'low': ['analyzing', 'implementing', 'validating'],
                    'medium': ['analyzing', 'planning', 'implementing', 'validating'],
                    'high': ['analyzing', 'planning', 'architecting', 'implementing', 'validating'],
                    'very_high': ['analyzing', 'planning', 'architecting', 'implementing', 'validating', 'learning']
                },
                'retry_logic': {
                    'max_attempts': 3,
                    'backoff_states': ['analyzing', 'planning']
                }
            }
        }
    
    def analyze_current_state(self) -> str:
        """
        Analyze the current state and determine the next appropriate state using enhanced intelligence.
        
        Returns:
            Next state to transition to
        """
        import time
        start_time = time.time()
        
        current_state_config = self.workflow_graph['states'].get(self.current_state, {})
        possible_transitions = current_state_config.get('transitions', [])
        
        if not possible_transitions:
            return 'completed'
        
        # Use enhanced state analyzer if available
        if hasattr(self, 'state_analyzer'):
            try:
                next_state, analysis_details = self.state_analyzer.analyze_and_determine_next_state(
                    self.current_state, self.context, self.workflow_graph
                )
                
                # Record enhanced decision metrics
                decision_time = time.time() - start_time
                self.metrics['intelligent_decisions'] += 1
                self.metrics['avg_decision_time'] = (
                    (self.metrics['avg_decision_time'] * self.metrics['decisions'] + decision_time) /
                    (self.metrics['decisions'] + 1)
                )
                
                confidence = analysis_details.get('confidence', 0.7)
                self.metrics['decision_confidence_avg'] = (
                    (self.metrics['decision_confidence_avg'] * self.metrics['decisions'] + confidence) /
                    (self.metrics['decisions'] + 1)
                )
                
                # Store analysis details for debugging/monitoring
                self.context['last_analysis'] = analysis_details
                
                # Record pattern matches
                if 'pattern_matches' in self.context:
                    self.metrics['pattern_matches'] += len(self.context.get('pattern_matches', []))
                
            except Exception as e:
                # Fallback to basic decision logic on error
                self.metrics['errors'] += 1
                self.context['analysis_error'] = str(e)
                next_state = self._apply_decision_logic(possible_transitions)
        else:
            # Fallback to basic decision logic
            next_state = self._apply_decision_logic(possible_transitions)
        
        # Record the decision
        self.metrics['decisions'] += 1
        
        return next_state
    
    def _apply_decision_logic(self, possible_transitions: List[str]) -> str:
        """
        Apply intelligent decision logic to select the best next state.
        
        Args:
            possible_transitions: List of possible next states
            
        Returns:
            Selected next state
        """
        # Enhanced validation results handling with intelligent loop-back detection
        validation_results = self.context.get('validation_results')
        error_history = self.context.get('error_history', [])
        retry_count = self.context.get('retry_count', 0)
        
        # Determine if this is a loop-back scenario
        is_loop_back = (
            (validation_results and hasattr(validation_results, 'success') and not validation_results.success) or
            (error_history and len(error_history) > 0) or
            (retry_count > 0)
        )
        
        # Use enhanced loop-back decision engine for intelligent decisions
        if is_loop_back and hasattr(self, 'loop_back_decision_engine'):
            try:
                recommended_state, decision_details = self.loop_back_decision_engine.make_loop_back_decision(
                    self.current_state, self.context, possible_transitions
                )
                
                # Store decision details for monitoring
                self.context['enhanced_loop_back_decision'] = decision_details
                self.metrics['intelligent_decisions'] += 1
                
                return recommended_state
                
            except Exception as e:
                # Fallback to basic logic on error
                self.context['loop_back_decision_error'] = str(e)
                self.metrics['errors'] += 1
        
        # Use WorkflowLoopbackManager if available (backward compatibility)
        if validation_results and hasattr(self, 'loopback_manager') and self.loopback_manager:
            try:
                loop_back_decision = self.loopback_manager.should_loop_back(validation_results, self.current_state)
                if loop_back_decision:
                    target_state, reason = loop_back_decision
                    if target_state in possible_transitions:
                        # Record the loop-back decision reason
                        self.context['basic_loop_back_decision'] = {
                            'target_state': target_state,
                            'reason': reason,
                            'timestamp': datetime.now().isoformat()
                        }
                        return target_state
            except Exception as e:
                # Continue to basic validation handling
                self.context['loopback_manager_error'] = str(e)
        
        # Fallback to simple validation result handling for backward compatibility
        if validation_results:
            if hasattr(validation_results, 'success') and not validation_results.success:
                if 'implementing' in possible_transitions and self.current_state == 'validating':
                    return 'implementing'  # Loop back to fix issues
                elif 'analyzing' in possible_transitions and self.current_state == 'implementing':
                    return 'analyzing'  # Loop back for re-analysis
        
        # Check complexity-based routing
        complexity = self.context.get('complexity', 'medium')
        complexity_rules = self.workflow_graph['decision_rules']['complexity_based'].get(complexity, [])
        
        # Find the next logical step based on complexity
        current_index = -1
        if self.current_state in complexity_rules:
            current_index = complexity_rules.index(self.current_state)
        
        if current_index >= 0 and current_index + 1 < len(complexity_rules):
            next_logical_state = complexity_rules[current_index + 1]
            if next_logical_state in possible_transitions:
                return next_logical_state
        
        # Check retry logic
        retry_count = self.context.get('retry_count', 0)
        retry_logic = self.workflow_graph['decision_rules'].get('retry_logic', {})
        max_retries = retry_logic.get('max_attempts', 3)
        
        if retry_count >= max_retries and 'failed' in possible_transitions:
            return 'failed'
        
        # Default to first available transition
        return possible_transitions[0]
    
    def transition_state(self, new_state: str, reason: str = "", 
                        context_updates: Optional[Dict] = None,
                        agents_selected: Optional[List[str]] = None,
                        issue_number: Optional[int] = None,
                        execute_loopback: bool = False) -> bool:
        """
        Transition to a new state with validation, persistence, and GitHub synchronization.
        
        Args:
            new_state: Target state
            reason: Reason for transition
            context_updates: Updates to apply to context
            agents_selected: Agents selected for the new state
            issue_number: GitHub issue number to update (if available)
            
        Returns:
            Success status
        """
        try:
            # Validate transition
            current_state_config = self.workflow_graph['states'].get(self.current_state, {})
            valid_transitions = current_state_config.get('transitions', [])
            
            if new_state not in valid_transitions and new_state != 'failed':
                raise ValueError(f"Invalid transition from {self.current_state} to {new_state}")
            
            # Get agents for the new state if not specified
            if agents_selected is None:
                new_state_config = self.workflow_graph['states'].get(new_state, {})
                agents_selected = new_state_config.get('agents', [])
            
            # Calculate confidence score
            confidence_score = self._calculate_transition_confidence(new_state, context_updates)
            
            # Enhanced agent selection using learning capabilities
            if agents_selected is None and hasattr(self, 'learning_agent_selector'):
                try:
                    # Create enhanced context for agent selection
                    selection_context = self.context.copy()
                    if context_updates:
                        selection_context.update(context_updates)
                    
                    agents_selected = self.learning_agent_selector.compose_dynamic_team(selection_context)
                    
                    # Record agent selection metrics
                    self.metrics['intelligent_agent_selections'] = self.metrics.get('intelligent_agent_selections', 0) + 1
                    
                except Exception as e:
                    # Fallback to basic agent selection
                    self.context['agent_selection_error'] = str(e)
                    new_state_config = self.workflow_graph['states'].get(new_state, {})
                    agents_selected = new_state_config.get('agents', [])
            
            # Generate recovery strategy if this is a failure recovery transition
            recovery_strategy = None
            if self.current_state in ['validating', 'implementing'] and new_state in ['implementing', 'analyzing', 'planning']:
                error_history = self.context.get('error_history', [])
                if error_history and hasattr(self, 'recovery_strategy_selector'):
                    try:
                        # Analyze failures and select recovery strategy
                        failure_analysis = self.failure_pattern_analyzer.analyze_failure_patterns(error_history, self.context)
                        recovery_strategy = self.recovery_strategy_selector.select_recovery_strategy(
                            failure_analysis, self.context, new_state
                        )
                        
                        # Store recovery strategy in context
                        self.context['recovery_strategy'] = recovery_strategy
                        
                    except Exception as e:
                        self.context['recovery_strategy_error'] = str(e)
            
            # Handle loop-back execution if requested
            if execute_loopback and self.loopback_manager and 'loop_back_decision' in self.context:
                loop_decision = self.context['loop_back_decision']
                try:
                    # Execute intelligent loop-back with context preservation
                    self.context = self.loopback_manager.execute_loopback(
                        self.current_state, new_state, self.context, reason
                    )
                    # Clear the loop-back decision marker
                    self.context.pop('loop_back_decision', None)
                except Exception as e:
                    # Log loop-back failure but continue with normal transition
                    self.context['loopback_error'] = str(e)
                    self.metrics['errors'] += 1
            
            # Update context
            if context_updates:
                self.context.update(context_updates)
            
            # Synchronize with GitHub if issue number is available
            github_success = True
            github_message = "No GitHub sync (issue number not provided)"
            
            if issue_number and self.github_state_manager:
                github_success, github_message = self.github_state_manager.transition_state(
                    issue_number, new_state, reason
                )
                
                if not github_success:
                    # Log GitHub sync failure but don't fail the entire transition
                    self.context['github_sync_error'] = github_message
                    self.metrics['errors'] += 1
            
            # Record transition in history
            transition_record = {
                'from_state': self.current_state,
                'to_state': new_state,
                'reason': reason,
                'agents_selected': agents_selected,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat(),
                'context_snapshot': self.context.copy(),
                'github_sync': {
                    'success': github_success,
                    'message': github_message,
                    'issue_number': issue_number
                }
            }
            
            self.history.append(transition_record)
            
            # Update agent assignments
            for agent in agents_selected:
                self.agent_assignments[agent] = 'active'
            
            # Clear previous assignments
            for agent, status in list(self.agent_assignments.items()):
                if agent not in agents_selected and status == 'active':
                    self.agent_assignments[agent] = 'completed'
            
            # Use integration system if available
            if self.integration and hasattr(self.integration, 'transition_state'):
                self.integration.transition_state(
                    self.current_state, new_state, reason, agents_selected,
                    context_updates, confidence_score
                )
            
            # Update current state
            self.current_state = new_state
            self.metrics['transitions'] += 1
            
            return True
            
        except Exception as e:
            self.metrics['errors'] += 1
            self.context['last_error'] = str(e)
            return False
    
    def _calculate_transition_confidence(self, new_state: str, 
                                       context_updates: Optional[Dict] = None) -> float:
        """
        Calculate confidence score for a state transition.
        
        Args:
            new_state: Target state
            context_updates: Context updates being applied
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.8
        
        # Adjust based on context completeness
        new_state_config = self.workflow_graph['states'].get(new_state, {})
        requirements = new_state_config.get('requirements', [])
        
        fulfilled_requirements = 0
        for requirement in requirements:
            if requirement in self.context or (context_updates and requirement in context_updates):
                fulfilled_requirements += 1
        
        if requirements:
            requirement_score = fulfilled_requirements / len(requirements)
        else:
            requirement_score = 1.0
        
        # Adjust based on retry count
        retry_count = self.context.get('retry_count', 0)
        retry_penalty = min(retry_count * 0.1, 0.3)
        
        # Calculate final confidence
        confidence = base_confidence * requirement_score - retry_penalty
        
        return max(0.1, min(1.0, confidence))
    
    def run_workflow(self, initial_context: Dict[str, Any], 
                    max_iterations: int = 50, 
                    enable_intelligent_loopback: bool = True) -> Dict[str, Any]:
        """
        Run a complete workflow from start to finish.
        
        Args:
            initial_context: Initial context data
            max_iterations: Maximum number of state transitions
            
        Returns:
            Final workflow result
        """
        start_time = time.time()
        
        try:
            # Initialize workflow
            self.context.update(initial_context)
            
            if self.integration:
                session_id = self.integration.start_orchestration_session(
                    workflow_type=initial_context.get('workflow_type', 'standard'),
                    priority=initial_context.get('priority', 0),
                    context=initial_context
                )
                self.context['session_id'] = session_id
            
            iteration = 0
            while iteration < max_iterations and self.current_state not in ['completed', 'failed']:
                # Analyze current state
                next_state = self.analyze_current_state()
                
                if next_state == self.current_state:
                    break  # No valid transitions available
                
                # Check if this is a loop-back transition and execute accordingly
                is_loopback = 'loop_back_decision' in self.context
                
                # Transition to next state
                success = self.transition_state(
                    next_state,
                    reason=f"Workflow progression iteration {iteration + 1}" + 
                           (f" (Loop-back)" if is_loopback else ""),
                    execute_loopback=is_loopback and enable_intelligent_loopback
                )
                
                if not success:
                    self.context['retry_count'] = self.context.get('retry_count', 0) + 1
                    self.metrics['retries'] += 1
                    
                    if self.context['retry_count'] >= self.max_retries:
                        self.transition_state('failed', 'Max retries exceeded')
                        break
                
                iteration += 1
            
            # Calculate final metrics
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                'final_state': self.current_state,
                'iterations': iteration,
                'duration_seconds': duration,
                'success': self.current_state == 'completed',
                'metrics': self.metrics.copy(),
                'history': self.history.copy(),
                'context': self.context.copy()
            }
            
            # Add loop-back statistics if available
            if self.loopback_manager:
                result['loopback_statistics'] = self.loopback_manager.get_loop_statistics()
            
            # Complete session if using integration
            if self.integration and 'session_id' in self.context:
                outcome = 'completed' if result['success'] else 'failed'
                self.integration.complete_orchestration_session(outcome, result)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            return {
                'final_state': 'failed',
                'error': str(e),
                'metrics': self.metrics.copy(),
                'success': False
            }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get enhanced state summary for monitoring with intelligence insights.
        
        Returns:
            Enhanced state summary information
        """
        summary = {
            'current_state': self.current_state,
            'context_keys': list(self.context.keys()),
            'active_agents': [agent for agent, status in self.agent_assignments.items() 
                            if status == 'active'],
            'transition_count': len(self.history),
            'metrics': self.metrics.copy(),
            'confidence_threshold': self.confidence_threshold,
            'last_transition': self.history[-1] if self.history else None
        }
        
        # Add enhanced intelligence insights
        if hasattr(self, 'state_analyzer'):
            try:
                context_model = self.context_engine.model_context(self.context)
                summary['intelligence_insights'] = {
                    'context_quality': context_model.confidence_scores.get('data_quality', 0.5),
                    'pattern_matches': len(context_model.pattern_matches),
                    'complexity_analysis': getattr(context_model, 'complexity_analysis', {}),
                    'risk_analysis': getattr(context_model, 'risk_analysis', {}),
                    'decision_confidence': self.metrics.get('decision_confidence_avg', 0.7)
                }
            except Exception as e:
                summary['intelligence_insights'] = {'error': str(e)}
        
        # Add loop-back intelligence status
        if hasattr(self, 'loop_back_decision_engine'):
            try:
                summary['loop_back_intelligence'] = {
                    'decision_history_count': len(self.loop_back_decision_engine.decision_history),
                    'failure_patterns_known': len(self.failure_pattern_analyzer.failure_patterns),
                    'recovery_strategies_available': len(self.recovery_strategy_selector.recovery_strategies),
                    'recent_decisions': [
                        {
                            'from_state': d['from_state'],
                            'to_state': d['to_state'],
                            'timestamp': d['timestamp']
                        }
                        for d in self.loop_back_decision_engine.decision_history[-3:]
                    ]
                }
            except Exception as e:
                summary['loop_back_intelligence'] = {'error': str(e)}
        
        # Add agent performance insights
        if hasattr(self, 'learning_agent_selector'):
            try:
                performance_summary = {}
                for agent, scores in self.learning_agent_selector.performance_history.items():
                    if scores:
                        performance_summary[agent] = {
                            'recent_score': scores[-1],
                            'average_score': sum(scores) / len(scores),
                            'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[-2] else 'stable'
                        }
                
                summary['agent_performance'] = performance_summary
            except Exception as e:
                summary['agent_performance'] = {'error': str(e)}
        
        return summary
    
    def record_transition_performance(self, agents: List[str], success: bool, 
                                    performance_metrics: Optional[Dict[str, float]] = None):
        """
        Record performance metrics for agents after a transition.
        
        Args:
            agents: List of agents involved in the transition
            success: Whether the transition was successful
            performance_metrics: Optional detailed performance metrics
        """
        if not hasattr(self, 'learning_agent_selector'):
            return
        
        try:
            # Calculate base performance score from success
            base_score = 0.8 if success else 0.3
            
            # Enhance with detailed metrics if available
            if performance_metrics:
                # Use provided metrics
                for agent in agents:
                    self.learning_agent_selector.performance_tracking_system.record_performance(
                        agent, self.context, performance_metrics
                    )
            else:
                # Use simple success-based scoring
                simple_metrics = {'reliability': base_score}
                for agent in agents:
                    self.learning_agent_selector.performance_tracking_system.record_performance(
                        agent, self.context, simple_metrics
                    )
            
            # Update loop-back decision engine with transition outcome
            if hasattr(self, 'loop_back_decision_engine') and len(self.history) > 1:
                last_transition = self.history[-1]
                previous_state = last_transition.get('from_state', 'unknown')
                current_state = last_transition.get('to_state', self.current_state)
                
                self.loop_back_decision_engine.record_transition_outcome(
                    previous_state, current_state, success
                )
        
        except Exception as e:
            self.context['performance_recording_error'] = str(e)
            self.metrics['errors'] += 1
    
    def validate_github_sync(self, issue_number: int) -> Dict[str, Any]:
        """
        Validate that GitHub issue state is synchronized with orchestrator state.
        
        Args:
            issue_number: GitHub issue number to validate
            
        Returns:
            Validation report
        """
        if not self.github_state_manager:
            return {
                'synchronized': False,
                'error': 'GitHub state manager not available',
                'orchestrator_state': self.current_state,
                'github_state': None
            }
        
        try:
            github_state = self.github_state_manager.get_current_state(issue_number)
            
            return {
                'synchronized': self.current_state == github_state,
                'orchestrator_state': self.current_state,
                'github_state': github_state,
                'issue_number': issue_number,
                'validation_report': self.github_state_manager.validate_issue_state(issue_number)
            }
            
        except Exception as e:
            return {
                'synchronized': False,
                'error': str(e),
                'orchestrator_state': self.current_state,
                'github_state': None,
                'issue_number': issue_number
            }
    
    def sync_with_github(self, issue_number: int) -> bool:
        """
        Force synchronization of orchestrator state with GitHub.
        
        Args:
            issue_number: GitHub issue number to sync
            
        Returns:
            Success status
        """
        if not self.github_state_manager:
            return False
        
        try:
            success, message = self.github_state_manager.transition_state(
                issue_number, self.current_state, "Orchestrator state synchronization"
            )
            
            if success:
                self.context['last_github_sync'] = {
                    'timestamp': datetime.now().isoformat(),
                    'issue_number': issue_number,
                    'state': self.current_state,
                    'message': message
                }
            
            return success
            
        except Exception as e:
            self.context['github_sync_error'] = str(e)
            return False


class LearningAgentSelector:
    """
    Enhanced adaptive agent selection system with learning capabilities and performance optimization.
    """
    
    def __init__(self):
        """Initialize the learning agent selector."""
        self.agent_capabilities = self._get_agent_capabilities()
        self.performance_history = defaultdict(list)
        self.specialization_matrix = self._build_specialization_matrix()
        self.team_optimization_engine = TeamOptimizationEngine()
        self.performance_tracking_system = PerformanceTrackingSystem()
        self.learning_weights = {
            'historical_performance': 0.4,
            'capability_match': 0.3,
            'workload_balance': 0.2,
            'team_synergy': 0.1
        }
    
    def _get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Define agent capabilities and specializations."""
        return {
            'RIF-Analyst': {
                'primary': ['requirements_analysis', 'pattern_recognition'],
                'secondary': ['complexity_assessment', 'risk_analysis'],
                'complexity_preference': ['low', 'medium', 'high'],
                'contexts': ['new_issue', 'requirements_unclear']
            },
            'RIF-Planner': {
                'primary': ['strategic_planning', 'workflow_design'],
                'secondary': ['resource_allocation', 'timeline_planning'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['complex_project', 'multi_phase']
            },
            'RIF-Architect': {
                'primary': ['system_design', 'architecture_planning'],
                'secondary': ['technology_selection', 'scalability_design'],
                'complexity_preference': ['high', 'very_high'],
                'contexts': ['high_complexity', 'system_design']
            },
            'RIF-Implementer': {
                'primary': ['code_implementation', 'feature_development'],
                'secondary': ['testing', 'debugging'],
                'complexity_preference': ['low', 'medium', 'high'],
                'contexts': ['implementation_ready', 'coding_required']
            },
            'RIF-Validator': {
                'primary': ['quality_assurance', 'testing'],
                'secondary': ['performance_testing', 'security_testing'],
                'complexity_preference': ['low', 'medium', 'high', 'very_high'],
                'contexts': ['validation_required', 'quality_gates']
            },
            'RIF-Learner': {
                'primary': ['knowledge_extraction', 'pattern_learning'],
                'secondary': ['process_improvement', 'metrics_analysis'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['learning_phase', 'improvement_needed']
            },
            'RIF-Security': {
                'primary': ['security_analysis', 'vulnerability_assessment'],
                'secondary': ['compliance_checking', 'threat_modeling'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['security_critical', 'compliance_required']
            }
        }
    
    def _build_specialization_matrix(self) -> Dict[str, List[str]]:
        """Build specialization matrix for agent selection."""
        matrix = {}
        for agent, capabilities in self.agent_capabilities.items():
            for primary_skill in capabilities['primary']:
                if primary_skill not in matrix:
                    matrix[primary_skill] = []
                matrix[primary_skill].append(agent)
        return matrix
    
    def compose_dynamic_team(self, context: Dict[str, Any]) -> List[str]:
        """
        Compose a dynamic agent team with enhanced learning capabilities.
        
        Args:
            context: Context information for team composition
            
        Returns:
            List of selected agent names
        """
        selected_agents = []
        
        # Extract context requirements
        complexity = context.get('complexity', 'medium')
        security_critical = context.get('security_critical', False)
        required_skills = context.get('required_skills', [])
        max_team_size = context.get('max_team_size', 4)
        
        # Always include security agent for critical systems
        if security_critical and 'RIF-Security' not in selected_agents:
            selected_agents.append('RIF-Security')
        
        # Select agents based on complexity
        complexity_agents = self._get_agents_for_complexity(complexity)
        selected_agents.extend(complexity_agents)
        
        # Select agents based on required skills
        for skill in required_skills:
            if skill in self.specialization_matrix:
                skill_agents = self.specialization_matrix[skill]
                for agent in skill_agents:
                    if agent not in selected_agents:
                        selected_agents.append(agent)
                        break
        
        # Use team optimization engine for enhanced selection
        if hasattr(self, 'team_optimization_engine'):
            try:
                # Convert performance history to format expected by optimizer
                performance_data = {}
                for agent, history in self.performance_history.items():
                    performance_data[agent] = history
                
                optimized_team = self.team_optimization_engine.optimize_team_composition(
                    selected_agents, context, performance_data
                )
                selected_agents = optimized_team
            except Exception:
                # Fallback to basic selection on optimization error
                pass
        
        # Apply performance-based optimization
        selected_agents = self._optimize_team_by_performance(selected_agents, context)
        
        # Limit team size
        return selected_agents[:max_team_size]
    
    def _get_agents_for_complexity(self, complexity: str) -> List[str]:
        """Get agents suitable for given complexity level."""
        suitable_agents = []
        for agent, capabilities in self.agent_capabilities.items():
            if complexity in capabilities['complexity_preference']:
                suitable_agents.append(agent)
        return suitable_agents
    
    def _optimize_team_by_performance(self, agents: List[str], 
                                    context: Dict[str, Any]) -> List[str]:
        """
        Optimize agent team based on performance history.
        
        Args:
            agents: Initial agent selection
            context: Context for optimization
            
        Returns:
            Optimized agent list
        """
        # If no performance history, return original selection
        if not any(self.performance_history.values()):
            return agents
        
        # Score agents based on performance history
        agent_scores = {}
        for agent in agents:
            if agent in self.performance_history:
                scores = self.performance_history[agent]
                agent_scores[agent] = sum(scores) / len(scores) if scores else 0.5
            else:
                agent_scores[agent] = 0.5  # Default score for new agents
        
        # Sort by performance score
        optimized_agents = sorted(agents, key=lambda a: agent_scores.get(a, 0.5), reverse=True)
        
        return optimized_agents
    
    def record_agent_performance(self, agent: str, performance_score: float):
        """
        Record agent performance for future optimization.
        
        Args:
            agent: Agent name
            performance_score: Performance score between 0.0 and 1.0
        """
        self.performance_history[agent].append(performance_score)
        
        # Keep only recent performance data
        if len(self.performance_history[agent]) > 10:
            self.performance_history[agent] = self.performance_history[agent][-10:]


class TeamOptimizationEngine:
    """Engine for optimizing agent team composition based on multiple factors."""
    
    def __init__(self):
        """Initialize team optimization engine."""
        self.optimization_strategies = {
            'performance_based': self._optimize_by_performance,
            'capability_based': self._optimize_by_capabilities,
            'workload_based': self._optimize_by_workload,
            'synergy_based': self._optimize_by_synergy
        }
        self.team_cache = {}
        self.synergy_matrix = self._build_synergy_matrix()
    
    def optimize_team_composition(self, initial_team: List[str], context: Dict[str, Any], 
                                 performance_data: Dict[str, List[float]]) -> List[str]:
        """
        Optimize team composition using multiple strategies.
        
        Args:
            initial_team: Initial agent selection
            context: Context for optimization
            performance_data: Historical performance data
            
        Returns:
            Optimized agent team
        """
        optimization_results = {}
        
        # Apply each optimization strategy
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                optimized_team = strategy_func(initial_team, context, performance_data)
                optimization_results[strategy_name] = optimized_team
            except Exception as e:
                optimization_results[strategy_name] = initial_team  # Fallback
        
        # Combine optimization results using weighted scoring
        final_team = self._combine_optimization_results(optimization_results, context)
        
        # Ensure team size constraints
        max_team_size = context.get('max_team_size', 4)
        return final_team[:max_team_size]
    
    def _optimize_by_performance(self, team: List[str], context: Dict[str, Any], 
                               performance_data: Dict[str, List[float]]) -> List[str]:
        """Optimize team based on historical performance."""
        agent_scores = {}
        
        for agent in team:
            if agent in performance_data and performance_data[agent]:
                # Calculate weighted performance score (recent performance weighted higher)
                scores = performance_data[agent]
                weighted_score = 0.0
                weight_sum = 0.0
                
                for i, score in enumerate(reversed(scores)):
                    weight = (i + 1) ** 0.5  # Recent scores have higher weight
                    weighted_score += score * weight
                    weight_sum += weight
                
                agent_scores[agent] = weighted_score / weight_sum if weight_sum > 0 else 0.5
            else:
                agent_scores[agent] = 0.5  # Default score for agents without history
        
        # Sort by performance score
        return sorted(team, key=lambda a: agent_scores.get(a, 0.5), reverse=True)
    
    def _optimize_by_capabilities(self, team: List[str], context: Dict[str, Any], 
                                performance_data: Dict[str, List[float]]) -> List[str]:
        """Optimize team based on capability matching."""
        required_skills = context.get('required_skills', [])
        complexity = context.get('complexity', 'medium')
        
        agent_capability_scores = {}
        
        for agent in team:
            score = 0.0
            
            # Check skill matching
            agent_capabilities = self._get_agent_capabilities_for_agent(agent)
            primary_skills = agent_capabilities.get('primary', [])
            secondary_skills = agent_capabilities.get('secondary', [])
            
            for skill in required_skills:
                if skill in primary_skills:
                    score += 1.0
                elif skill in secondary_skills:
                    score += 0.5
            
            # Check complexity preference
            complexity_preferences = agent_capabilities.get('complexity_preference', [])
            if complexity in complexity_preferences:
                score += 0.5
            
            agent_capability_scores[agent] = score
        
        # Sort by capability score
        return sorted(team, key=lambda a: agent_capability_scores.get(a, 0.0), reverse=True)
    
    def _optimize_by_workload(self, team: List[str], context: Dict[str, Any], 
                            performance_data: Dict[str, List[float]]) -> List[str]:
        """Optimize team based on current workload balance."""
        # This is a simplified workload analysis - in production would check actual workloads
        workload_scores = {}
        
        for agent in team:
            # Simulate workload based on recent activity
            recent_activity = len(performance_data.get(agent, [])[-5:])  # Last 5 activities
            workload_score = max(0, 1.0 - (recent_activity / 10.0))  # Lower workload = higher score
            workload_scores[agent] = workload_score
        
        return sorted(team, key=lambda a: workload_scores.get(a, 1.0), reverse=True)
    
    def _optimize_by_synergy(self, team: List[str], context: Dict[str, Any], 
                           performance_data: Dict[str, List[float]]) -> List[str]:
        """Optimize team based on agent synergy."""
        if len(team) < 2:
            return team
        
        # Calculate team synergy scores for different combinations
        best_combination = team
        best_synergy_score = 0.0
        
        # For small teams, try different orderings
        from itertools import permutations
        
        if len(team) <= 4:  # Only for small teams to avoid computational explosion
            for combination in permutations(team):
                synergy_score = self._calculate_team_synergy(list(combination))
                if synergy_score > best_synergy_score:
                    best_synergy_score = synergy_score
                    best_combination = list(combination)
        
        return best_combination
    
    def _calculate_team_synergy(self, team: List[str]) -> float:
        """Calculate synergy score for a team composition."""
        if len(team) < 2:
            return 1.0
        
        total_synergy = 0.0
        pair_count = 0
        
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                agent1, agent2 = team[i], team[j]
                synergy = self.synergy_matrix.get((agent1, agent2), 0.5)
                total_synergy += synergy
                pair_count += 1
        
        return total_synergy / pair_count if pair_count > 0 else 1.0
    
    def _build_synergy_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build synergy matrix for agent pairs."""
        # This is a simplified synergy matrix - in production would be learned from data
        synergy_pairs = {
            ('RIF-Analyst', 'RIF-Planner'): 0.9,
            ('RIF-Planner', 'RIF-Architect'): 0.8,
            ('RIF-Architect', 'RIF-Implementer'): 0.7,
            ('RIF-Implementer', 'RIF-Validator'): 0.8,
            ('RIF-Validator', 'RIF-Learner'): 0.6,
            ('RIF-Security', 'RIF-Architect'): 0.7,
            ('RIF-Security', 'RIF-Validator'): 0.8,
        }
        
        # Make symmetric
        symmetric_matrix = {}
        for (agent1, agent2), synergy in synergy_pairs.items():
            symmetric_matrix[(agent1, agent2)] = synergy
            symmetric_matrix[(agent2, agent1)] = synergy
        
        return symmetric_matrix
    
    def _get_agent_capabilities_for_agent(self, agent: str) -> Dict[str, Any]:
        """Get capabilities for a specific agent."""
        # This would typically load from a configuration or knowledge base
        capabilities_map = {
            'RIF-Analyst': {
                'primary': ['requirements_analysis', 'pattern_recognition'],
                'secondary': ['complexity_assessment', 'risk_analysis'],
                'complexity_preference': ['low', 'medium', 'high']
            },
            'RIF-Planner': {
                'primary': ['strategic_planning', 'workflow_design'],
                'secondary': ['resource_allocation', 'timeline_planning'],
                'complexity_preference': ['medium', 'high', 'very_high']
            },
            'RIF-Architect': {
                'primary': ['system_design', 'architecture_planning'],
                'secondary': ['technology_selection', 'scalability_design'],
                'complexity_preference': ['high', 'very_high']
            },
            'RIF-Implementer': {
                'primary': ['code_implementation', 'feature_development'],
                'secondary': ['testing', 'debugging'],
                'complexity_preference': ['low', 'medium', 'high']
            },
            'RIF-Validator': {
                'primary': ['quality_assurance', 'testing'],
                'secondary': ['performance_testing', 'security_testing'],
                'complexity_preference': ['low', 'medium', 'high', 'very_high']
            },
            'RIF-Learner': {
                'primary': ['knowledge_extraction', 'pattern_learning'],
                'secondary': ['process_improvement', 'metrics_analysis'],
                'complexity_preference': ['medium', 'high', 'very_high']
            },
            'RIF-Security': {
                'primary': ['security_analysis', 'vulnerability_assessment'],
                'secondary': ['compliance_checking', 'threat_modeling'],
                'complexity_preference': ['medium', 'high', 'very_high']
            }
        }
        
        return capabilities_map.get(agent, {'primary': [], 'secondary': [], 'complexity_preference': ['medium']})
    
    def _combine_optimization_results(self, optimization_results: Dict[str, List[str]], 
                                    context: Dict[str, Any]) -> List[str]:
        """Combine results from different optimization strategies."""
        # Weight different optimization strategies based on context
        strategy_weights = {
            'performance_based': 0.4,
            'capability_based': 0.3,
            'workload_based': 0.2,
            'synergy_based': 0.1
        }
        
        # Adjust weights based on context
        if context.get('complexity') in ['high', 'very_high']:
            strategy_weights['capability_based'] = 0.4
            strategy_weights['performance_based'] = 0.3
        
        if context.get('time_pressure', False):
            strategy_weights['performance_based'] = 0.5
            strategy_weights['workload_based'] = 0.3
        
        # Score each agent based on weighted rankings
        all_agents = set()
        for team in optimization_results.values():
            all_agents.update(team)
        
        agent_scores = {agent: 0.0 for agent in all_agents}
        
        for strategy_name, team in optimization_results.items():
            weight = strategy_weights.get(strategy_name, 0.1)
            
            for i, agent in enumerate(team):
                # Higher ranking (lower index) gets higher score
                position_score = (len(team) - i) / len(team)
                agent_scores[agent] += position_score * weight
        
        # Return agents sorted by combined score
        return sorted(all_agents, key=lambda a: agent_scores[a], reverse=True)


class PerformanceTrackingSystem:
    """System for tracking and analyzing agent performance over time."""
    
    def __init__(self):
        """Initialize performance tracking system."""
        self.performance_metrics = defaultdict(list)
        self.context_performance = defaultdict(lambda: defaultdict(list))  # agent -> context_type -> scores
        self.trend_analysis_window = 10
        self.performance_categories = {
            'speed': 'Task completion speed',
            'quality': 'Output quality score',
            'reliability': 'Success rate',
            'collaboration': 'Team collaboration score'
        }
    
    def record_performance(self, agent: str, context: Dict[str, Any], performance_data: Dict[str, float]):
        """
        Record performance data for an agent in a specific context.
        
        Args:
            agent: Agent identifier
            context: Context in which performance was measured
            performance_data: Dictionary of performance metrics
        """
        # Record overall performance
        overall_score = self._calculate_overall_score(performance_data)
        self.performance_metrics[agent].append({
            'timestamp': datetime.now().isoformat(),
            'context': context.copy(),
            'metrics': performance_data.copy(),
            'overall_score': overall_score
        })
        
        # Record context-specific performance
        context_type = self._get_context_type(context)
        self.context_performance[agent][context_type].append(overall_score)
        
        # Limit history size
        if len(self.performance_metrics[agent]) > 50:
            self.performance_metrics[agent] = self.performance_metrics[agent][-50:]
        
        if len(self.context_performance[agent][context_type]) > 20:
            self.context_performance[agent][context_type] = self.context_performance[agent][context_type][-20:]
    
    def get_performance_prediction(self, agent: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict agent performance for a given context.
        
        Args:
            agent: Agent identifier
            context: Context for prediction
            
        Returns:
            Predicted performance metrics
        """
        context_type = self._get_context_type(context)
        
        # Get context-specific performance history
        context_scores = self.context_performance[agent][context_type]
        
        if not context_scores:
            # Fallback to overall performance
            if agent in self.performance_metrics:
                recent_scores = [entry['overall_score'] for entry in self.performance_metrics[agent][-10:]]
                if recent_scores:
                    predicted_score = sum(recent_scores) / len(recent_scores)
                else:
                    predicted_score = 0.5
            else:
                predicted_score = 0.5
        else:
            # Use context-specific trend analysis
            predicted_score = self._predict_from_trend(context_scores)
        
        # Calculate confidence based on data availability
        confidence = min(len(context_scores) / 10.0, 1.0) if context_scores else 0.3
        
        return {
            'predicted_score': predicted_score,
            'confidence': confidence,
            'context_type': context_type,
            'historical_data_points': len(context_scores)
        }
    
    def get_performance_trends(self, agent: str, window_size: Optional[int] = None) -> Dict[str, Any]:
        """Get performance trends for an agent."""
        window_size = window_size or self.trend_analysis_window
        
        if agent not in self.performance_metrics:
            return {'trend': 'no_data', 'direction': 'unknown', 'confidence': 0.0}
        
        recent_entries = self.performance_metrics[agent][-window_size:]
        if len(recent_entries) < 3:
            return {'trend': 'insufficient_data', 'direction': 'unknown', 'confidence': 0.3}
        
        scores = [entry['overall_score'] for entry in recent_entries]
        
        # Calculate trend direction
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        trend_direction = 'improving' if second_avg > first_avg else 'declining' if second_avg < first_avg else 'stable'
        trend_strength = abs(second_avg - first_avg)
        
        return {
            'trend': trend_direction,
            'strength': trend_strength,
            'confidence': min(len(recent_entries) / window_size, 1.0),
            'recent_average': second_avg,
            'data_points': len(recent_entries)
        }
    
    def _calculate_overall_score(self, performance_data: Dict[str, float]) -> float:
        """Calculate overall performance score from individual metrics."""
        if not performance_data:
            return 0.5
        
        # Weight different performance categories
        weights = {'speed': 0.3, 'quality': 0.4, 'reliability': 0.2, 'collaboration': 0.1}
        
        weighted_score = 0.0
        weight_sum = 0.0
        
        for metric, value in performance_data.items():
            weight = weights.get(metric, 0.1)
            weighted_score += value * weight
            weight_sum += weight
        
        return weighted_score / weight_sum if weight_sum > 0 else sum(performance_data.values()) / len(performance_data)
    
    def _get_context_type(self, context: Dict[str, Any]) -> str:
        """Extract context type for performance categorization."""
        complexity = context.get('complexity', 'medium')
        workflow_type = context.get('workflow_type', 'standard')
        security_critical = context.get('security_critical', False)
        
        if security_critical:
            return f'security_{complexity}'
        else:
            return f'{workflow_type}_{complexity}'
    
    def _predict_from_trend(self, scores: List[float]) -> float:
        """Predict next score based on trend analysis."""
        if len(scores) < 2:
            return scores[0] if scores else 0.5
        
        # Simple linear trend prediction
        recent_scores = scores[-5:]  # Use last 5 scores for prediction
        
        if len(recent_scores) < 2:
            return recent_scores[-1]
        
        # Calculate simple moving average with trend
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Calculate trend
        if len(recent_scores) >= 3:
            trend = (recent_scores[-1] - recent_scores[0]) / (len(recent_scores) - 1)
            predicted = avg_score + trend
        else:
            predicted = avg_score
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, predicted))


class AdaptiveAgentSelector(LearningAgentSelector):
    """
    Backward-compatible adaptive agent selector that extends LearningAgentSelector.
    """
    
    def compose_dynamic_team(self, context: Dict[str, Any]) -> List[str]:
        """
        Compose a dynamic agent team using enhanced learning capabilities.
        
        Args:
            context: Context information for team composition
            
        Returns:
            List of selected agent names
        """
        # Get initial team using base capability matching
        initial_team = self._get_base_team_selection(context)
        
        # Use team optimization engine for enhanced selection
        if hasattr(self, 'team_optimization_engine'):
            try:
                # Convert performance history to format expected by optimizer
                performance_data = {}
                for agent, history in self.performance_history.items():
                    performance_data[agent] = history
                
                optimized_team = self.team_optimization_engine.optimize_team_composition(
                    initial_team, context, performance_data
                )
                return optimized_team
            except Exception as e:
                # Fallback to basic selection on optimization error
                return initial_team
        else:
            return initial_team
    
    def _get_base_team_selection(self, context: Dict[str, Any]) -> List[str]:
        """Get base team selection using original logic."""
        selected_agents = []
        
        # Extract context requirements
        complexity = context.get('complexity', 'medium')
        security_critical = context.get('security_critical', False)
        required_skills = context.get('required_skills', [])
        max_team_size = context.get('max_team_size', 4)
        
        # Always include security agent for critical systems
        if security_critical and 'RIF-Security' not in selected_agents:
            selected_agents.append('RIF-Security')
        
        # Select agents based on complexity
        complexity_agents = self._get_agents_for_complexity(complexity)
        selected_agents.extend(complexity_agents)
        
        # Select agents based on required skills
        for skill in required_skills:
            if skill in self.specialization_matrix:
                skill_agents = self.specialization_matrix[skill]
                for agent in skill_agents:
                    if agent not in selected_agents:
                        selected_agents.append(agent)
                        break
        
        # Apply performance-based optimization
        selected_agents = self._optimize_team_by_performance(selected_agents, context)
        
        # Limit team size
        return selected_agents[:max_team_size]
    
    def _get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Define agent capabilities and specializations."""
        return {
            'RIF-Analyst': {
                'primary': ['requirements_analysis', 'pattern_recognition'],
                'secondary': ['complexity_assessment', 'risk_analysis'],
                'complexity_preference': ['low', 'medium', 'high'],
                'contexts': ['new_issue', 'requirements_unclear']
            },
            'RIF-Planner': {
                'primary': ['strategic_planning', 'workflow_design'],
                'secondary': ['resource_allocation', 'timeline_planning'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['complex_project', 'multi_phase']
            },
            'RIF-Architect': {
                'primary': ['system_design', 'architecture_planning'],
                'secondary': ['technology_selection', 'scalability_design'],
                'complexity_preference': ['high', 'very_high'],
                'contexts': ['high_complexity', 'system_design']
            },
            'RIF-Implementer': {
                'primary': ['code_implementation', 'feature_development'],
                'secondary': ['testing', 'debugging'],
                'complexity_preference': ['low', 'medium', 'high'],
                'contexts': ['implementation_ready', 'coding_required']
            },
            'RIF-Validator': {
                'primary': ['quality_assurance', 'testing'],
                'secondary': ['performance_testing', 'security_testing'],
                'complexity_preference': ['low', 'medium', 'high', 'very_high'],
                'contexts': ['validation_required', 'quality_gates']
            },
            'RIF-Learner': {
                'primary': ['knowledge_extraction', 'pattern_learning'],
                'secondary': ['process_improvement', 'metrics_analysis'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['learning_phase', 'improvement_needed']
            },
            'RIF-Security': {
                'primary': ['security_analysis', 'vulnerability_assessment'],
                'secondary': ['compliance_checking', 'threat_modeling'],
                'complexity_preference': ['medium', 'high', 'very_high'],
                'contexts': ['security_critical', 'compliance_required']
            }
        }
    
    def _build_specialization_matrix(self) -> Dict[str, List[str]]:
        """Build specialization matrix for agent selection."""
        matrix = {}
        for agent, capabilities in self.agent_capabilities.items():
            for primary_skill in capabilities['primary']:
                if primary_skill not in matrix:
                    matrix[primary_skill] = []
                matrix[primary_skill].append(agent)
        return matrix
    
    def compose_dynamic_team(self, context: Dict[str, Any]) -> List[str]:
        """
        Compose a dynamic agent team based on context requirements.
        
        Args:
            context: Context information for team composition
            
        Returns:
            List of selected agent names
        """
        selected_agents = []
        
        # Extract context requirements
        complexity = context.get('complexity', 'medium')
        security_critical = context.get('security_critical', False)
        required_skills = context.get('required_skills', [])
        max_team_size = context.get('max_team_size', 4)
        
        # Always include security agent for critical systems
        if security_critical and 'RIF-Security' not in selected_agents:
            selected_agents.append('RIF-Security')
        
        # Select agents based on complexity
        complexity_agents = self._get_agents_for_complexity(complexity)
        selected_agents.extend(complexity_agents)
        
        # Select agents based on required skills
        for skill in required_skills:
            if skill in self.specialization_matrix:
                skill_agents = self.specialization_matrix[skill]
                for agent in skill_agents:
                    if agent not in selected_agents:
                        selected_agents.append(agent)
                        break
        
        # Apply performance-based optimization
        selected_agents = self._optimize_team_by_performance(selected_agents, context)
        
        # Limit team size
        return selected_agents[:max_team_size]
    
    def _get_agents_for_complexity(self, complexity: str) -> List[str]:
        """Get agents suitable for given complexity level."""
        suitable_agents = []
        for agent, capabilities in self.agent_capabilities.items():
            if complexity in capabilities['complexity_preference']:
                suitable_agents.append(agent)
        return suitable_agents
    
    def _optimize_team_by_performance(self, agents: List[str], 
                                    context: Dict[str, Any]) -> List[str]:
        """
        Optimize agent team based on performance history.
        
        Args:
            agents: Initial agent selection
            context: Context for optimization
            
        Returns:
            Optimized agent list
        """
        # If no performance history, return original selection
        if not any(self.performance_history.values()):
            return agents
        
        # Score agents based on performance history
        agent_scores = {}
        for agent in agents:
            if agent in self.performance_history:
                scores = self.performance_history[agent]
                agent_scores[agent] = sum(scores) / len(scores) if scores else 0.5
            else:
                agent_scores[agent] = 0.5  # Default score for new agents
        
        # Sort by performance score
        optimized_agents = sorted(agents, key=lambda a: agent_scores.get(a, 0.5), reverse=True)
        
        return optimized_agents
    
    def record_agent_performance(self, agent: str, performance_score: float):
        """
        Record agent performance for future optimization.
        
        Args:
            agent: Agent name
            performance_score: Performance score between 0.0 and 1.0
        """
        self.performance_history[agent].append(performance_score)
        
        # Keep only recent performance data
        if len(self.performance_history[agent]) > 10:
            self.performance_history[agent] = self.performance_history[agent][-10:]


class FailurePatternAnalyzer:
    """Analyzer for identifying patterns in workflow failures to enable intelligent recovery."""
    
    def __init__(self):
        """Initialize failure pattern analyzer."""
        self.failure_patterns = {
            'timeout_pattern': {
                'keywords': ['timeout', 'timed out', 'deadline', 'duration'],
                'recovery_strategy': 'extend_timeout_and_simplify',
                'recommended_state': 'implementing',
                'confidence_threshold': 0.7
            },
            'dependency_pattern': {
                'keywords': ['dependency', 'import', 'module', 'package', 'library'],
                'recovery_strategy': 'resolve_dependencies',
                'recommended_state': 'planning',
                'confidence_threshold': 0.8
            },
            'architecture_pattern': {
                'keywords': ['architecture', 'design', 'structure', 'coupling', 'cohesion'],
                'recovery_strategy': 'redesign_architecture',
                'recommended_state': 'architecting',
                'confidence_threshold': 0.9
            },
            'requirements_pattern': {
                'keywords': ['requirement', 'specification', 'unclear', 'ambiguous', 'missing'],
                'recovery_strategy': 'clarify_requirements',
                'recommended_state': 'analyzing',
                'confidence_threshold': 0.8
            },
            'resource_pattern': {
                'keywords': ['memory', 'cpu', 'disk', 'resource', 'capacity', 'limit'],
                'recovery_strategy': 'optimize_resources',
                'recommended_state': 'implementing',
                'confidence_threshold': 0.7
            },
            'integration_pattern': {
                'keywords': ['integration', 'interface', 'api', 'compatibility', 'version'],
                'recovery_strategy': 'fix_integration',
                'recommended_state': 'implementing',
                'confidence_threshold': 0.8
            }
        }
        self.failure_history = []
        self.pattern_cache = {}
    
    def analyze_failure_patterns(self, error_history: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze failure patterns in error history.
        
        Args:
            error_history: List of error messages/descriptions
            context: Current workflow context
            
        Returns:
            Analysis results with identified patterns and recovery recommendations
        """
        if not error_history:
            return {'patterns_found': [], 'confidence': 0.0, 'recommended_strategy': 'continue'}
        
        # Identify patterns in error history
        identified_patterns = self._identify_patterns(error_history)
        
        # Analyze pattern frequency and recency
        pattern_analysis = self._analyze_pattern_significance(identified_patterns, error_history)
        
        # Generate recovery recommendations
        recovery_recommendations = self._generate_recovery_recommendations(pattern_analysis, context)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_analysis_confidence(pattern_analysis, len(error_history))
        
        return {
            'patterns_found': list(pattern_analysis.keys()),
            'pattern_details': pattern_analysis,
            'recovery_recommendations': recovery_recommendations,
            'confidence': overall_confidence,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _identify_patterns(self, error_history: List[str]) -> Dict[str, List[int]]:
        """Identify which patterns appear in the error history."""
        identified_patterns = defaultdict(list)
        
        for i, error_msg in enumerate(error_history):
            error_lower = error_msg.lower()
            
            for pattern_name, pattern_config in self.failure_patterns.items():
                keywords = pattern_config['keywords']
                
                # Check if any keywords match
                matches = [keyword for keyword in keywords if keyword in error_lower]
                if matches:
                    identified_patterns[pattern_name].append(i)
        
        return dict(identified_patterns)
    
    def _analyze_pattern_significance(self, identified_patterns: Dict[str, List[int]], 
                                    error_history: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze the significance of identified patterns."""
        pattern_analysis = {}
        total_errors = len(error_history)
        
        for pattern_name, error_indices in identified_patterns.items():
            frequency = len(error_indices)
            frequency_ratio = frequency / total_errors
            
            # Calculate recency score (recent errors weighted higher)
            recency_score = 0.0
            for idx in error_indices:
                position_weight = (total_errors - idx) / total_errors  # More recent = higher weight
                recency_score += position_weight
            recency_score = recency_score / frequency if frequency > 0 else 0.0
            
            # Calculate clustering (are errors clustered together?)
            clustering_score = self._calculate_clustering_score(error_indices, total_errors)
            
            # Overall significance score
            significance = (frequency_ratio * 0.4 + recency_score * 0.4 + clustering_score * 0.2)
            
            pattern_analysis[pattern_name] = {
                'frequency': frequency,
                'frequency_ratio': frequency_ratio,
                'recency_score': recency_score,
                'clustering_score': clustering_score,
                'significance': significance,
                'error_indices': error_indices,
                'recovery_strategy': self.failure_patterns[pattern_name]['recovery_strategy'],
                'recommended_state': self.failure_patterns[pattern_name]['recommended_state']
            }
        
        return pattern_analysis
    
    def _calculate_clustering_score(self, error_indices: List[int], total_errors: int) -> float:
        """Calculate how clustered the errors are (clustered errors suggest systematic issues)."""
        if len(error_indices) < 2:
            return 0.5
        
        # Calculate average distance between consecutive error occurrences
        distances = []
        for i in range(1, len(error_indices)):
            distances.append(error_indices[i] - error_indices[i-1])
        
        if not distances:
            return 0.5
        
        avg_distance = sum(distances) / len(distances)
        max_possible_distance = total_errors / len(error_indices)
        
        # Lower average distance = higher clustering = higher score
        clustering_score = 1.0 - min(avg_distance / max_possible_distance, 1.0)
        return clustering_score
    
    def _generate_recovery_recommendations(self, pattern_analysis: Dict[str, Dict[str, Any]], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recovery recommendations based on pattern analysis."""
        if not pattern_analysis:
            return {'primary_strategy': 'continue', 'confidence': 0.3}
        
        # Find the most significant pattern
        most_significant_pattern = max(pattern_analysis.keys(), 
                                     key=lambda p: pattern_analysis[p]['significance'])
        
        pattern_info = pattern_analysis[most_significant_pattern]
        
        # Generate primary recommendation
        primary_strategy = pattern_info['recovery_strategy']
        recommended_state = pattern_info['recommended_state']
        confidence = pattern_info['significance']
        
        # Generate alternative strategies
        alternative_strategies = []
        for pattern_name, info in pattern_analysis.items():
            if pattern_name != most_significant_pattern and info['significance'] > 0.3:
                alternative_strategies.append({
                    'strategy': info['recovery_strategy'],
                    'state': info['recommended_state'],
                    'confidence': info['significance']
                })
        
        # Context-based adjustments
        if context.get('retry_count', 0) > 3:
            # Many retries suggest need for more drastic change
            if 'redesign' not in primary_strategy:
                primary_strategy = 'comprehensive_redesign'
                recommended_state = 'analyzing'
        
        return {
            'primary_strategy': primary_strategy,
            'recommended_state': recommended_state,
            'confidence': confidence,
            'alternative_strategies': alternative_strategies,
            'dominant_pattern': most_significant_pattern
        }
    
    def _calculate_analysis_confidence(self, pattern_analysis: Dict[str, Dict[str, Any]], 
                                     total_errors: int) -> float:
        """Calculate confidence in the analysis based on data quality and pattern clarity."""
        if not pattern_analysis:
            return 0.3
        
        # Base confidence from data quantity
        data_confidence = min(total_errors / 5.0, 1.0)  # More errors = more confidence (up to 5)
        
        # Pattern clarity confidence (how significant are the patterns)
        if pattern_analysis:
            max_significance = max(info['significance'] for info in pattern_analysis.values())
            pattern_confidence = max_significance
        else:
            pattern_confidence = 0.3
        
        # Combined confidence
        overall_confidence = (data_confidence * 0.4 + pattern_confidence * 0.6)
        
        return min(overall_confidence, 1.0)


class LoopBackDecisionEngine:
    """Engine for making intelligent loop-back decisions based on failure analysis."""
    
    def __init__(self, failure_analyzer: FailurePatternAnalyzer):
        """Initialize loop-back decision engine."""
        self.failure_analyzer = failure_analyzer
        self.decision_history = []
        self.state_success_rates = defaultdict(lambda: {'successes': 0, 'failures': 0})
        self.transition_preferences = {
            'from_validating': {
                'implementing': 0.7,  # Most common loop-back
                'planning': 0.2,      # Major redesign needed
                'analyzing': 0.1      # Requirements issues
            },
            'from_implementing': {
                'analyzing': 0.4,     # Requirements clarification
                'planning': 0.3,      # Strategy adjustment  
                'architecting': 0.3   # Design issues
            },
            'from_planning': {
                'analyzing': 0.8,     # Back to requirements
                'implementing': 0.2   # Skip architecture if simple
            }
        }
    
    def make_loop_back_decision(self, current_state: str, context: Dict[str, Any], 
                              possible_transitions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Make intelligent loop-back decision based on failure analysis.
        
        Args:
            current_state: Current workflow state
            context: Current context including error history
            possible_transitions: Available state transitions
            
        Returns:
            Tuple of (recommended_state, decision_details)
        """
        # Analyze failure patterns if error history available
        error_history = context.get('error_history', [])
        validation_results = context.get('validation_results')
        
        failure_analysis = None
        if error_history:
            failure_analysis = self.failure_analyzer.analyze_failure_patterns(error_history, context)
        
        # Make decision based on multiple factors
        decision_factors = self._evaluate_loop_back_factors(
            current_state, context, possible_transitions, failure_analysis
        )
        
        # Select best transition
        best_transition = self._select_optimal_transition(
            decision_factors, possible_transitions, current_state
        )
        
        # Generate decision details
        decision_details = {
            'decision_factors': decision_factors,
            'failure_analysis': failure_analysis,
            'selected_transition': best_transition,
            'confidence': decision_factors.get(best_transition, {}).get('confidence', 0.5),
            'reasoning': self._generate_decision_reasoning(best_transition, decision_factors, context)
        }
        
        # Record decision for learning
        self._record_decision(current_state, best_transition, decision_details, context)
        
        return best_transition, decision_details
    
    def _evaluate_loop_back_factors(self, current_state: str, context: Dict[str, Any], 
                                  possible_transitions: List[str], 
                                  failure_analysis: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Evaluate various factors for loop-back decision."""
        decision_factors = {}
        
        for transition in possible_transitions:
            factors = {
                'pattern_match': self._evaluate_pattern_match_factor(transition, failure_analysis),
                'historical_success': self._evaluate_historical_success_factor(current_state, transition),
                'retry_analysis': self._evaluate_retry_factor(transition, context),
                'complexity_consideration': self._evaluate_complexity_factor(transition, context),
                'preference_weight': self._evaluate_preference_factor(current_state, transition)
            }
            
            # Calculate weighted score
            weights = {
                'pattern_match': 0.3,
                'historical_success': 0.25,
                'retry_analysis': 0.2,
                'complexity_consideration': 0.15,
                'preference_weight': 0.1
            }
            
            weighted_score = sum(factors[factor] * weights[factor] for factor in factors)
            
            factors['weighted_score'] = weighted_score
            factors['confidence'] = self._calculate_factor_confidence(factors)
            
            decision_factors[transition] = factors
        
        return decision_factors
    
    def _evaluate_pattern_match_factor(self, transition: str, 
                                     failure_analysis: Optional[Dict[str, Any]]) -> float:
        """Evaluate how well transition matches identified failure patterns."""
        if not failure_analysis or not failure_analysis.get('recovery_recommendations'):
            return 0.5  # Neutral score when no pattern analysis available
        
        recommended_state = failure_analysis['recovery_recommendations'].get('recommended_state')
        confidence = failure_analysis['recovery_recommendations'].get('confidence', 0.5)
        
        if transition == recommended_state:
            return confidence
        
        # Check alternative strategies
        alternatives = failure_analysis['recovery_recommendations'].get('alternative_strategies', [])
        for alt in alternatives:
            if alt['state'] == transition:
                return alt['confidence'] * 0.8  # Slightly lower weight for alternatives
        
        return 0.3  # Low score if transition doesn't match any patterns
    
    def _evaluate_historical_success_factor(self, current_state: str, transition: str) -> float:
        """Evaluate based on historical success rates for this transition."""
        state_key = f"{current_state}_to_{transition}"
        
        if state_key not in self.state_success_rates:
            return 0.5  # No historical data
        
        success_data = self.state_success_rates[state_key]
        total_attempts = success_data['successes'] + success_data['failures']
        
        if total_attempts == 0:
            return 0.5
        
        success_rate = success_data['successes'] / total_attempts
        
        # Weight by number of attempts (more data = more confidence)
        confidence_weight = min(total_attempts / 10.0, 1.0)
        
        return success_rate * confidence_weight + 0.5 * (1 - confidence_weight)
    
    def _evaluate_retry_factor(self, transition: str, context: Dict[str, Any]) -> float:
        """Evaluate based on retry count and patterns."""
        retry_count = context.get('retry_count', 0)
        
        if retry_count == 0:
            return 0.5
        
        # Higher retry count suggests need for more significant change
        if retry_count > 3:
            # Prefer states that represent bigger changes
            big_change_states = ['analyzing', 'planning', 'architecting']
            if transition in big_change_states:
                return 0.8
            else:
                return 0.2
        elif retry_count > 1:
            # Moderate changes preferred
            moderate_change_states = ['planning', 'implementing']
            if transition in moderate_change_states:
                return 0.7
            else:
                return 0.4
        else:
            # Small retry count, modest changes acceptable
            return 0.6
    
    def _evaluate_complexity_factor(self, transition: str, context: Dict[str, Any]) -> float:
        """Evaluate based on complexity considerations."""
        complexity = context.get('complexity', 'medium')
        
        # Different complexities prefer different loop-back strategies
        complexity_preferences = {
            'low': {
                'implementing': 0.8,  # Keep it simple
                'validating': 0.7,
                'analyzing': 0.5,
                'planning': 0.3,
                'architecting': 0.2
            },
            'medium': {
                'implementing': 0.7,
                'planning': 0.6,
                'analyzing': 0.6,
                'validating': 0.5,
                'architecting': 0.4
            },
            'high': {
                'planning': 0.8,      # More planning needed
                'architecting': 0.7,
                'analyzing': 0.6,
                'implementing': 0.5,
                'validating': 0.4
            },
            'very_high': {
                'analyzing': 0.8,     # Back to fundamentals
                'architecting': 0.7,
                'planning': 0.6,
                'implementing': 0.4,
                'validating': 0.3
            }
        }
        
        return complexity_preferences.get(complexity, {}).get(transition, 0.5)
    
    def _evaluate_preference_factor(self, current_state: str, transition: str) -> float:
        """Evaluate based on general transition preferences."""
        transition_key = f"from_{current_state}"
        
        if transition_key in self.transition_preferences:
            return self.transition_preferences[transition_key].get(transition, 0.3)
        
        return 0.5
    
    def _calculate_factor_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence in the factor evaluation."""
        # Confidence based on how consistent the factors are
        scores = [v for k, v in factors.items() if k not in ['weighted_score', 'confidence']]
        
        if not scores:
            return 0.5
        
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Low variance = high confidence (factors agree)
        confidence = max(0.3, 1.0 - variance)
        
        return confidence
    
    def _select_optimal_transition(self, decision_factors: Dict[str, Dict[str, Any]], 
                                 possible_transitions: List[str], current_state: str) -> str:
        """Select the optimal transition based on decision factors."""
        if not decision_factors:
            return possible_transitions[0] if possible_transitions else 'completed'
        
        # Select transition with highest weighted score
        best_transition = max(
            decision_factors.keys(),
            key=lambda t: decision_factors[t]['weighted_score']
        )
        
        return best_transition
    
    def _generate_decision_reasoning(self, selected_transition: str, 
                                   decision_factors: Dict[str, Dict[str, Any]], 
                                   context: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the decision."""
        if selected_transition not in decision_factors:
            return f"Default selection: {selected_transition}"
        
        factors = decision_factors[selected_transition]
        reasons = []
        
        # Identify strongest factors
        factor_names = {
            'pattern_match': 'Failure pattern analysis',
            'historical_success': 'Historical success rate',
            'retry_analysis': 'Retry count analysis', 
            'complexity_consideration': 'Complexity assessment',
            'preference_weight': 'General preferences'
        }
        
        for factor_key, factor_name in factor_names.items():
            if factors.get(factor_key, 0) > 0.7:
                reasons.append(f"{factor_name} strongly supports {selected_transition}")
            elif factors.get(factor_key, 0) > 0.6:
                reasons.append(f"{factor_name} supports {selected_transition}")
        
        # Add context-specific reasoning
        retry_count = context.get('retry_count', 0)
        if retry_count > 2:
            reasons.append(f"High retry count ({retry_count}) suggests need for significant change")
        
        if not reasons:
            reasons.append(f"Balanced evaluation favors {selected_transition}")
        
        return "; ".join(reasons)
    
    def _record_decision(self, from_state: str, to_state: str, 
                        decision_details: Dict[str, Any], context: Dict[str, Any]):
        """Record decision for future learning."""
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'from_state': from_state,
            'to_state': to_state,
            'context': context.copy(),
            'decision_details': decision_details.copy()
        }
        
        self.decision_history.append(decision_record)
        
        # Limit history size
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
    
    def record_transition_outcome(self, from_state: str, to_state: str, success: bool):
        """Record whether a transition was ultimately successful."""
        state_key = f"{from_state}_to_{to_state}"
        
        if success:
            self.state_success_rates[state_key]['successes'] += 1
        else:
            self.state_success_rates[state_key]['failures'] += 1


class RecoveryStrategySelector:
    """Selector for choosing optimal recovery strategies based on failure analysis."""
    
    def __init__(self):
        """Initialize recovery strategy selector."""
        self.recovery_strategies = {
            'extend_timeout_and_simplify': {
                'actions': ['Increase timeout limits', 'Simplify implementation', 'Add progress monitoring'],
                'effort': 'low',
                'success_rate': 0.7,
                'applicable_states': ['implementing', 'validating']
            },
            'resolve_dependencies': {
                'actions': ['Update dependency versions', 'Add missing dependencies', 'Resolve conflicts'],
                'effort': 'medium',
                'success_rate': 0.8,
                'applicable_states': ['planning', 'implementing']
            },
            'redesign_architecture': {
                'actions': ['Rethink system architecture', 'Improve design patterns', 'Reduce coupling'],
                'effort': 'high',
                'success_rate': 0.9,
                'applicable_states': ['architecting', 'planning']
            },
            'clarify_requirements': {
                'actions': ['Gather additional requirements', 'Clarify ambiguities', 'Update specifications'],
                'effort': 'medium',
                'success_rate': 0.8,
                'applicable_states': ['analyzing', 'planning']
            },
            'optimize_resources': {
                'actions': ['Memory optimization', 'Performance tuning', 'Resource pooling'],
                'effort': 'medium',
                'success_rate': 0.7,
                'applicable_states': ['implementing', 'validating']
            },
            'fix_integration': {
                'actions': ['Update integration points', 'Fix API compatibility', 'Version alignment'],
                'effort': 'medium',
                'success_rate': 0.8,
                'applicable_states': ['implementing', 'validating']
            },
            'comprehensive_redesign': {
                'actions': ['Complete requirements review', 'Full system redesign', 'Phased implementation'],
                'effort': 'very_high',
                'success_rate': 0.95,
                'applicable_states': ['analyzing']
            }
        }
        self.strategy_history = defaultdict(list)
    
    def select_recovery_strategy(self, failure_analysis: Dict[str, Any], 
                               context: Dict[str, Any], current_state: str) -> Dict[str, Any]:
        """
        Select optimal recovery strategy based on failure analysis.
        
        Args:
            failure_analysis: Results from failure pattern analysis
            context: Current workflow context
            current_state: Current workflow state
            
        Returns:
            Selected recovery strategy with implementation details
        """
        # Get candidate strategies from failure analysis
        recommended_strategies = self._get_candidate_strategies(failure_analysis, current_state)
        
        # Evaluate strategies based on multiple criteria
        strategy_scores = self._evaluate_strategies(recommended_strategies, context, current_state)
        
        # Select best strategy
        best_strategy = self._select_best_strategy(strategy_scores, context)
        
        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(best_strategy, context)
        
        return {
            'selected_strategy': best_strategy,
            'implementation_plan': implementation_plan,
            'alternative_strategies': [s for s in strategy_scores.keys() if s != best_strategy],
            'confidence': strategy_scores.get(best_strategy, {}).get('overall_score', 0.5)
        }
    
    def _get_candidate_strategies(self, failure_analysis: Dict[str, Any], 
                                current_state: str) -> List[str]:
        """Get candidate recovery strategies from failure analysis."""
        candidates = []
        
        if failure_analysis and 'recovery_recommendations' in failure_analysis:
            # Primary strategy from failure analysis
            primary_strategy = failure_analysis['recovery_recommendations'].get('primary_strategy')
            if primary_strategy and primary_strategy in self.recovery_strategies:
                candidates.append(primary_strategy)
            
            # Alternative strategies
            alternatives = failure_analysis['recovery_recommendations'].get('alternative_strategies', [])
            for alt in alternatives:
                strategy = alt.get('strategy')
                if strategy and strategy in self.recovery_strategies:
                    candidates.append(strategy)
        
        # Add applicable strategies for current state
        for strategy_name, strategy_info in self.recovery_strategies.items():
            if current_state in strategy_info['applicable_states'] and strategy_name not in candidates:
                candidates.append(strategy_name)
        
        return candidates[:5]  # Limit to top 5 candidates
    
    def _evaluate_strategies(self, candidate_strategies: List[str], 
                           context: Dict[str, Any], current_state: str) -> Dict[str, Dict[str, Any]]:
        """Evaluate candidate strategies based on multiple criteria."""
        strategy_evaluations = {}
        
        for strategy_name in candidate_strategies:
            if strategy_name not in self.recovery_strategies:
                continue
            
            strategy_info = self.recovery_strategies[strategy_name]
            
            evaluation = {
                'effort_score': self._evaluate_effort_appropriateness(strategy_info['effort'], context),
                'success_rate_score': strategy_info['success_rate'],
                'state_compatibility_score': self._evaluate_state_compatibility(strategy_info, current_state),
                'historical_performance_score': self._evaluate_historical_performance(strategy_name, context),
                'urgency_compatibility_score': self._evaluate_urgency_compatibility(strategy_info, context)
            }
            
            # Calculate weighted overall score
            weights = {
                'effort_score': 0.2,
                'success_rate_score': 0.3,
                'state_compatibility_score': 0.2,
                'historical_performance_score': 0.15,
                'urgency_compatibility_score': 0.15
            }
            
            overall_score = sum(evaluation[criterion] * weights[criterion] 
                              for criterion in evaluation)
            
            evaluation['overall_score'] = overall_score
            strategy_evaluations[strategy_name] = evaluation
        
        return strategy_evaluations
    
    def _evaluate_effort_appropriateness(self, effort_level: str, context: Dict[str, Any]) -> float:
        """Evaluate if effort level is appropriate for context."""
        retry_count = context.get('retry_count', 0)
        complexity = context.get('complexity', 'medium')
        priority = context.get('priority', 1)
        
        effort_scores = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'very_high': 1.0
        }
        
        base_score = effort_scores.get(effort_level, 0.5)
        
        # Adjust based on retry count (more retries justify higher effort)
        if retry_count > 3:
            base_score = min(base_score + 0.3, 1.0)
        elif retry_count > 1:
            base_score = min(base_score + 0.1, 1.0)
        
        # Adjust based on complexity
        if complexity in ['high', 'very_high'] and effort_level in ['high', 'very_high']:
            base_score = min(base_score + 0.2, 1.0)
        
        # Adjust based on priority (high priority might prefer lower effort)
        if priority > 3 and effort_level in ['very_high', 'high']:
            base_score = max(base_score - 0.2, 0.0)
        
        return base_score
    
    def _evaluate_state_compatibility(self, strategy_info: Dict[str, Any], current_state: str) -> float:
        """Evaluate compatibility of strategy with current state."""
        applicable_states = strategy_info.get('applicable_states', [])
        
        if current_state in applicable_states:
            return 1.0
        else:
            return 0.3  # Can still be applicable with state transition
    
    def _evaluate_historical_performance(self, strategy_name: str, context: Dict[str, Any]) -> float:
        """Evaluate based on historical performance of this strategy."""
        if strategy_name not in self.strategy_history:
            return 0.5  # No historical data
        
        history = self.strategy_history[strategy_name]
        if not history:
            return 0.5
        
        # Calculate success rate from recent history
        recent_history = history[-10:]  # Last 10 attempts
        successes = sum(1 for outcome in recent_history if outcome.get('success', False))
        success_rate = successes / len(recent_history)
        
        # Weight by amount of data
        data_weight = min(len(recent_history) / 10.0, 1.0)
        
        return success_rate * data_weight + 0.5 * (1 - data_weight)
    
    def _evaluate_urgency_compatibility(self, strategy_info: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate if strategy matches urgency requirements."""
        priority = context.get('priority', 1)
        effort_level = strategy_info.get('effort', 'medium')
        
        # High priority tasks prefer lower effort strategies
        effort_penalty = {
            'low': 0.0,
            'medium': 0.1,
            'high': 0.3,
            'very_high': 0.5
        }
        
        base_score = 1.0
        if priority > 3:
            base_score -= effort_penalty.get(effort_level, 0.2)
        
        return max(base_score, 0.3)
    
    def _select_best_strategy(self, strategy_scores: Dict[str, Dict[str, Any]], 
                            context: Dict[str, Any]) -> str:
        """Select the best strategy based on evaluation scores."""
        if not strategy_scores:
            return 'comprehensive_redesign'  # Default fallback
        
        # Select strategy with highest overall score
        best_strategy = max(strategy_scores.keys(), 
                           key=lambda s: strategy_scores[s]['overall_score'])
        
        return best_strategy
    
    def _generate_implementation_plan(self, strategy_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed implementation plan for selected strategy."""
        if strategy_name not in self.recovery_strategies:
            return {'actions': ['Continue with current approach'], 'estimated_effort': 'unknown'}
        
        strategy_info = self.recovery_strategies[strategy_name]
        
        # Basic implementation plan
        implementation_plan = {
            'strategy_name': strategy_name,
            'actions': strategy_info['actions'].copy(),
            'estimated_effort': strategy_info['effort'],
            'expected_success_rate': strategy_info['success_rate'],
            'applicable_states': strategy_info['applicable_states']
        }
        
        # Add context-specific adjustments
        retry_count = context.get('retry_count', 0)
        if retry_count > 2:
            implementation_plan['actions'].insert(0, 'Conduct root cause analysis of previous failures')
        
        complexity = context.get('complexity', 'medium')
        if complexity in ['high', 'very_high']:
            implementation_plan['actions'].append('Add additional validation and testing phases')
        
        return implementation_plan
    
    def record_strategy_outcome(self, strategy_name: str, context: Dict[str, Any], 
                              success: bool, details: Dict[str, Any]):
        """Record the outcome of a recovery strategy for future learning."""
        outcome_record = {
            'timestamp': datetime.now().isoformat(),
            'context': context.copy(),
            'success': success,
            'details': details.copy()
        }
        
        self.strategy_history[strategy_name].append(outcome_record)
        
        # Limit history size
        if len(self.strategy_history[strategy_name]) > 50:
            self.strategy_history[strategy_name] = self.strategy_history[strategy_name][-50:]


# Mock validation classes for testing
class MockValidationSuccess:
    """Mock validation result for successful validation."""
    def __init__(self):
        self.success = True
        self.details = "All tests passed"
        self.coverage = 95.0


class MockValidationFailure:
    """Mock validation result for failed validation."""
    def __init__(self):
        self.success = False
        self.details = "Tests failed - implementation needs fixes"
        self.errors = ["Unit test failure in module X", "Integration test timeout"]


# Utility functions
def create_test_workflow_graph() -> Dict[str, Any]:
    """Create a simplified workflow graph for testing."""
    return {
        'states': {
            'initialized': {
                'transitions': ['analyzing'],
                'agents': ['RIF-Analyst'],
                'requirements': []
            },
            'analyzing': {
                'transitions': ['implementing', 'failed'],
                'agents': ['RIF-Analyst'],
                'requirements': ['github_issues']
            },
            'implementing': {
                'transitions': ['validating', 'analyzing'],
                'agents': ['RIF-Implementer'],
                'requirements': ['requirements']
            },
            'validating': {
                'transitions': ['completed', 'implementing'],
                'agents': ['RIF-Validator'],
                'requirements': ['implementation']
            },
            'completed': {
                'transitions': [],
                'agents': [],
                'requirements': []
            },
            'failed': {
                'transitions': ['analyzing'],
                'agents': ['RIF-Analyst'],
                'requirements': []
            }
        },
        'decision_rules': {
            'complexity_based': {
                'low': ['analyzing', 'implementing', 'validating'],
                'medium': ['analyzing', 'implementing', 'validating'],
                'high': ['analyzing', 'implementing', 'validating']
            },
            'retry_logic': {
                'max_attempts': 2,
                'backoff_states': ['analyzing']
            }
        }
    }


def demo_dynamic_orchestration():
    """Demonstrate dynamic orchestration functionality."""
    print("🚀 Starting Dynamic Orchestration Demo")
    
    # Create orchestrator with test workflow
    test_workflow = create_test_workflow_graph()
    orchestrator = DynamicOrchestrator(test_workflow)
    
    # Create agent selector
    selector = AdaptiveAgentSelector()
    
    print("1. Testing agent selection...")
    # Test complexity-based selection
    agents = selector.compose_dynamic_team({'complexity': 'high'})
    print(f"   High complexity team: {agents}")
    
    # Test security-critical selection
    agents = selector.compose_dynamic_team({'security_critical': True})
    print(f"   Security-critical team: {agents}")
    
    print("2. Running workflow simulation...")
    # Run workflow
    initial_context = {
        'github_issues': [1, 2, 3],
        'complexity': 'medium',
        'workflow_type': 'demo'
    }
    
    result = orchestrator.run_workflow(initial_context, max_iterations=10)
    
    print(f"   Final state: {result['final_state']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Duration: {result['duration_seconds']:.2f}s")
    print(f"   Success: {result['success']}")
    
    print("3. Testing loop-back mechanism...")
    # Test validation failure scenario
    orchestrator.current_state = 'validating'
    orchestrator.context['validation_results'] = MockValidationFailure()
    
    next_state = orchestrator.analyze_current_state()
    print(f"   Validation failure leads to: {next_state}")
    
    # Test validation success scenario
    orchestrator.context['validation_results'] = MockValidationSuccess()
    next_state = orchestrator.analyze_current_state()
    print(f"   Validation success leads to: {next_state}")
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_dynamic_orchestration()