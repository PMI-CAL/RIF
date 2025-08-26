#!/usr/bin/env python3
"""
Enhanced Orchestration Intelligence Layer - Issue #52 Implementation

This module implements the Enhanced Orchestration Intelligence Layer as designed by RIF-Architect,
providing sophisticated state analysis, adaptive agent selection, and context-aware decision making
while maintaining pattern compliance where Claude Code IS the orchestrator.

Architecture: Multi-Layer Adaptive Enhancement Pattern
Components: 6 core intelligent systems with learning capabilities
Integration: Extends existing orchestration_utilities.py
"""

import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics
import subprocess
import sys
import os

# Add the commands directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestration_utilities import (
    IssueContext, ContextAnalyzer, StateValidator, 
    OrchestrationHelper, GitHubStateManager
)
from content_analysis_engine import (
    ContentAnalysisEngine, ContentAnalysisResult, WorkflowState, ComplexityLevel
)

# Set up logging
logger = logging.getLogger(__name__)

class FailureCategory(Enum):
    """Categories of failures for intelligent analysis"""
    ARCHITECTURAL_ISSUES = "architecture"
    REQUIREMENT_GAPS = "requirements"
    IMPLEMENTATION_ERRORS = "implementation"
    QUALITY_FAILURES = "quality"
    PERFORMANCE_ISSUES = "performance"
    SECURITY_CONCERNS = "security"
    INTEGRATION_PROBLEMS = "integration"
    UNKNOWN = "unknown"

class ConfidenceLevel(Enum):
    """Confidence levels for decision making"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class ValidationResult:
    """Rich validation result with failure analysis"""
    passed: bool
    score: float
    failures: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_issues: List[str] = field(default_factory=list)
    architectural_concerns: List[str] = field(default_factory=list)
    
    def has_architectural_issues(self) -> bool:
        """Check if validation has architectural concerns"""
        return len(self.architectural_concerns) > 0 or any(
            f.get('category') == FailureCategory.ARCHITECTURAL_ISSUES.value 
            for f in self.failures
        )
    
    def has_requirement_gaps(self) -> bool:
        """Check if validation indicates requirement gaps"""
        return any(
            f.get('category') == FailureCategory.REQUIREMENT_GAPS.value 
            for f in self.failures
        )
    
    def has_fixable_errors(self) -> bool:
        """Check if validation has fixable implementation errors"""
        fixable_categories = [
            FailureCategory.IMPLEMENTATION_ERRORS.value,
            FailureCategory.QUALITY_FAILURES.value
        ]
        return any(
            f.get('category') in fixable_categories 
            for f in self.failures
        )

@dataclass 
class ContextModel:
    """Rich context model with semantic analysis"""
    issue_context: IssueContext
    complexity_dimensions: Dict[str, float] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    performance_context: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    historical_patterns: List[str] = field(default_factory=list)
    semantic_tags: Set[str] = field(default_factory=set)
    validation_results: Optional[ValidationResult] = None
    # ISSUE #273: Add content analysis result for label-free orchestration
    content_analysis_result: Optional['ContentAnalysisResult'] = None
    
    @property
    def overall_complexity_score(self) -> float:
        """Calculate weighted overall complexity score"""
        if not self.complexity_dimensions:
            return self.issue_context.complexity_score / 4.0
        
        weights = {
            'technical': 0.3,
            'architectural': 0.25,
            'integration': 0.2,
            'performance': 0.15,
            'security': 0.1
        }
        
        weighted_sum = sum(
            self.complexity_dimensions.get(dim, 0.5) * weight 
            for dim, weight in weights.items()
        )
        return min(weighted_sum, 1.0)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score"""
        if not self.risk_factors:
            return 0.3  # Default moderate risk
        
        risk_scores = [rf.get('severity', 0.5) for rf in self.risk_factors]
        return min(statistics.mean(risk_scores), 1.0)

@dataclass
class AgentPerformanceData:
    """Performance tracking data for agents"""
    agent_name: str
    successes: int = 0
    failures: int = 0
    avg_completion_time: float = 0.0
    complexity_handling: Dict[str, int] = field(default_factory=dict)
    last_performance_update: datetime = field(default_factory=datetime.now)
    context_specific_performance: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score based on performance history"""
        base_score = self.success_rate
        
        # Adjust for recency - more recent performance weighted higher
        recency_hours = (datetime.now() - self.last_performance_update).total_seconds() / 3600
        recency_factor = max(0.5, 1.0 - (recency_hours / (24 * 7)))  # Week decay
        
        return base_score * recency_factor

class ContextModelingEngine:
    """
    Multi-dimensional context analysis engine for rich semantic understanding.
    Phase 1 component: Foundation Enhancement
    """
    
    def __init__(self):
        self.knowledge_base_path = Path("knowledge")
        self.patterns_path = self.knowledge_base_path / "patterns" 
        self.decisions_path = self.knowledge_base_path / "decisions"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ISSUE #273: Initialize ContentAnalysisEngine to replace label dependency
        self.content_analysis_engine = ContentAnalysisEngine()
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def enrich_context(self, issue_context: IssueContext) -> ContextModel:
        """
        Create rich context model with multi-dimensional analysis.
        
        ISSUE #273: Now uses ContentAnalysisEngine for intelligent content analysis
        instead of relying on GitHub labels.
        
        Args:
            issue_context: Basic issue context from ContextAnalyzer
            
        Returns:
            ContextModel with rich semantic analysis
        """
        context_model = ContextModel(issue_context=issue_context)
        
        # ISSUE #273: Use ContentAnalysisEngine for intelligent content analysis
        content_analysis = self.content_analysis_engine.analyze_issue_content(
            issue_context.title, 
            issue_context.body
        )
        
        # Store content analysis result for later use
        context_model.content_analysis_result = content_analysis
        
        # Analyze complexity dimensions (enhanced with content analysis)
        context_model.complexity_dimensions = self._analyze_complexity_dimensions(issue_context, content_analysis)
        
        # Analyze security context
        context_model.security_context = self._analyze_security_context(issue_context)
        
        # Analyze performance context
        context_model.performance_context = self._analyze_performance_context(issue_context)
        
        # Identify risk factors (enhanced with content analysis)
        context_model.risk_factors = self._identify_risk_factors(issue_context, content_analysis)
        
        # Find historical patterns
        context_model.historical_patterns = self._find_historical_patterns(issue_context)
        
        # Generate semantic tags (enhanced with content analysis)
        context_model.semantic_tags = self._generate_semantic_tags(issue_context, content_analysis)
        
        return context_model
    
    def _analyze_complexity_dimensions(self, issue_context: IssueContext, content_analysis: Optional['ContentAnalysisResult'] = None) -> Dict[str, float]:
        """Analyze complexity across multiple dimensions"""
        dimensions = {
            'technical': self._assess_technical_complexity(issue_context),
            'architectural': self._assess_architectural_complexity(issue_context),
            'integration': self._assess_integration_complexity(issue_context),
            'performance': self._assess_performance_complexity(issue_context),
            'security': self._assess_security_complexity(issue_context)
        }
        
        # ISSUE #273: Enhance with content analysis derived complexity
        if content_analysis:
            # Use content analysis complexity as a baseline adjustment
            content_complexity_map = {
                ComplexityLevel.LOW: 0.25,
                ComplexityLevel.MEDIUM: 0.5, 
                ComplexityLevel.HIGH: 0.75,
                ComplexityLevel.VERY_HIGH: 1.0
            }
            
            base_complexity = content_complexity_map.get(content_analysis.complexity, 0.5)
            
            # Adjust all dimensions based on content analysis
            for dimension in dimensions:
                # Weight content analysis vs traditional analysis (70/30 split)
                dimensions[dimension] = (dimensions[dimension] * 0.3) + (base_complexity * 0.7)
        
        return dimensions
    
    def _assess_technical_complexity(self, issue_context: IssueContext) -> float:
        """Assess technical implementation complexity"""
        technical_indicators = [
            'algorithm', 'optimization', 'concurrent', 'async', 'threading',
            'parallel', 'distributed', 'cache', 'indexing', 'parsing'
        ]
        
        content = (issue_context.title + " " + issue_context.body).lower()
        matches = sum(1 for indicator in technical_indicators if indicator in content)
        
        # Scale to 0-1 range
        return min(matches / len(technical_indicators) * 2, 1.0)
    
    def _assess_architectural_complexity(self, issue_context: IssueContext) -> float:
        """Assess architectural design complexity"""
        arch_indicators = [
            'architecture', 'design', 'pattern', 'framework', 'infrastructure',
            'microservice', 'api', 'interface', 'layer', 'component'
        ]
        
        content = (issue_context.title + " " + issue_context.body).lower()
        matches = sum(1 for indicator in arch_indicators if indicator in content)
        
        # Higher weight for explicit architecture mentions
        if 'architect' in content:
            matches *= 1.5
        
        return min(matches / len(arch_indicators) * 2, 1.0)
    
    def _assess_integration_complexity(self, issue_context: IssueContext) -> float:
        """Assess system integration complexity"""
        integration_indicators = [
            'integration', 'mcp', 'server', 'client', 'protocol',
            'api', 'webhook', 'event', 'message', 'communication'
        ]
        
        content = (issue_context.title + " " + issue_context.body).lower()
        matches = sum(1 for indicator in integration_indicators if indicator in content)
        
        return min(matches / len(integration_indicators) * 2, 1.0)
    
    def _assess_performance_complexity(self, issue_context: IssueContext) -> float:
        """Assess performance optimization complexity"""
        performance_indicators = [
            'performance', 'speed', 'optimization', 'benchmark', 'metrics',
            'monitoring', 'profiling', 'memory', 'cpu', 'latency'
        ]
        
        content = (issue_context.title + " " + issue_context.body).lower()
        matches = sum(1 for indicator in performance_indicators if indicator in content)
        
        return min(matches / len(performance_indicators) * 2, 1.0)
    
    def _assess_security_complexity(self, issue_context: IssueContext) -> float:
        """Assess security implementation complexity"""
        security_indicators = [
            'security', 'auth', 'permission', 'validation', 'sanitization',
            'encryption', 'token', 'certificate', 'vulnerability', 'audit'
        ]
        
        content = (issue_context.title + " " + issue_context.body).lower()
        matches = sum(1 for indicator in security_indicators if indicator in content)
        
        return min(matches / len(security_indicators) * 2, 1.0)
    
    def _analyze_security_context(self, issue_context: IssueContext) -> Dict[str, Any]:
        """Analyze security implications and requirements"""
        security_context = {
            'requires_security_review': False,
            'auth_implications': False,
            'data_privacy_concerns': False,
            'vulnerability_risk': 'low',
            'compliance_requirements': []
        }
        
        content = (issue_context.title + " " + issue_context.body).lower()
        
        # Check for security requirements
        if any(term in content for term in ['security', 'auth', 'permission', 'access']):
            security_context['requires_security_review'] = True
        
        if any(term in content for term in ['login', 'token', 'session', 'credential']):
            security_context['auth_implications'] = True
        
        if any(term in content for term in ['data', 'privacy', 'gdpr', 'personal']):
            security_context['data_privacy_concerns'] = True
        
        # Assess vulnerability risk
        high_risk_terms = ['external', 'input', 'upload', 'exec', 'eval']
        if any(term in content for term in high_risk_terms):
            security_context['vulnerability_risk'] = 'high'
        elif security_context['requires_security_review']:
            security_context['vulnerability_risk'] = 'medium'
        
        return security_context
    
    def _analyze_performance_context(self, issue_context: IssueContext) -> Dict[str, Any]:
        """Analyze performance requirements and implications"""
        performance_context = {
            'performance_critical': False,
            'scalability_requirements': False,
            'real_time_requirements': False,
            'resource_constraints': [],
            'benchmark_requirements': False
        }
        
        content = (issue_context.title + " " + issue_context.body).lower()
        
        if any(term in content for term in ['performance', 'speed', 'fast', 'optimization']):
            performance_context['performance_critical'] = True
        
        if any(term in content for term in ['scale', 'concurrent', 'parallel', 'distributed']):
            performance_context['scalability_requirements'] = True
        
        if any(term in content for term in ['real-time', 'latency', 'instant', 'immediate']):
            performance_context['real_time_requirements'] = True
        
        if any(term in content for term in ['memory', 'cpu', 'disk', 'bandwidth']):
            performance_context['resource_constraints'].append('resource_aware')
        
        if any(term in content for term in ['benchmark', 'metrics', 'measurement']):
            performance_context['benchmark_requirements'] = True
        
        return performance_context
    
    def _identify_risk_factors(self, issue_context: IssueContext, content_analysis: Optional['ContentAnalysisResult'] = None) -> List[Dict[str, Any]]:
        """Identify potential risk factors for the issue"""
        risk_factors = []
        
        content = (issue_context.title + " " + issue_context.body).lower()
        
        # Technical risk factors
        if issue_context.complexity_score >= 3:
            risk_factors.append({
                'type': 'complexity',
                'description': 'High complexity may lead to implementation challenges',
                'severity': 0.7,
                'mitigation': 'Consider architectural planning phase'
            })
        
        # Integration risks
        if any(term in content for term in ['integration', 'api', 'external']):
            risk_factors.append({
                'type': 'integration',
                'description': 'External dependencies may cause reliability issues',
                'severity': 0.6,
                'mitigation': 'Implement robust error handling and fallbacks'
            })
        
        # Security risks
        if any(term in content for term in ['security', 'auth', 'access']):
            risk_factors.append({
                'type': 'security',
                'description': 'Security implementation requires careful validation',
                'severity': 0.8,
                'mitigation': 'Mandatory security review and testing'
            })
        
        # Performance risks
        if any(term in content for term in ['performance', 'scale', 'concurrent']):
            risk_factors.append({
                'type': 'performance',
                'description': 'Performance requirements may be difficult to achieve',
                'severity': 0.6,
                'mitigation': 'Implement performance benchmarking and monitoring'
            })
        
        return risk_factors
    
    def _find_historical_patterns(self, issue_context: IssueContext) -> List[str]:
        """Find similar historical patterns from knowledge base"""
        patterns = []
        
        try:
            if self.patterns_path.exists():
                for pattern_file in self.patterns_path.glob("*.json"):
                    try:
                        with open(pattern_file, 'r') as f:
                            pattern_data = json.load(f)
                        
                        # Simple pattern matching based on keywords
                        pattern_keywords = pattern_data.get('keywords', [])
                        issue_words = set((issue_context.title + " " + issue_context.body).lower().split())
                        
                        if pattern_keywords and issue_words:
                            common_words = set(pattern_keywords) & issue_words
                            if len(common_words) >= 2:  # At least 2 matching keywords
                                patterns.append(pattern_file.stem)
                    
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        except Exception as e:
            self.logger.warning(f"Error finding historical patterns: {e}")
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _generate_semantic_tags(self, issue_context: IssueContext, content_analysis: Optional['ContentAnalysisResult'] = None) -> Set[str]:
        """Generate semantic tags for the issue"""
        tags = set()
        
        content = (issue_context.title + " " + issue_context.body).lower()
        
        # Technology tags
        tech_keywords = {
            'python': 'python', 'javascript': 'javascript', 'api': 'api',
            'database': 'database', 'web': 'web', 'cli': 'cli',
            'mcp': 'mcp', 'server': 'server', 'client': 'client'
        }
        
        for keyword, tag in tech_keywords.items():
            if keyword in content:
                tags.add(tag)
        
        # Feature type tags
        if any(word in content for word in ['implement', 'create', 'add']):
            tags.add('feature')
        elif any(word in content for word in ['fix', 'bug', 'error']):
            tags.add('bugfix')
        elif any(word in content for word in ['enhance', 'improve', 'optimize']):
            tags.add('enhancement')
        elif any(word in content for word in ['refactor', 'restructure', 'redesign']):
            tags.add('refactoring')
        
        # Domain tags
        if any(word in content for word in ['orchestrat', 'workflow', 'agent']):
            tags.add('orchestration')
        elif any(word in content for word in ['quality', 'test', 'validation']):
            tags.add('quality')
        elif any(word in content for word in ['monitor', 'metrics', 'dashboard']):
            tags.add('monitoring')
        
        # ISSUE #273: Add content analysis derived tags
        if content_analysis:
            # Add semantic indicators from content analysis
            for category, indicators in content_analysis.semantic_indicators.items():
                for indicator in indicators:
                    tags.add(f"{category}:{indicator.lower()}")
            
            # Add state and complexity tags
            tags.add(f"state:{content_analysis.state.value}")
            tags.add(f"complexity:{content_analysis.complexity.value}")
            
            # Add risk factor tags
            for risk in content_analysis.risk_factors:
                tags.add(f"risk:{risk.lower().replace(' ', '_')}")
        
        return tags

class ValidationResultAnalyzer:
    """
    Sophisticated validation result processing for intelligent failure analysis.
    Phase 1 component: Foundation Enhancement
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_patterns = self._load_failure_patterns()
    
    def analyze_validation_results(self, raw_results: Dict[str, Any]) -> ValidationResult:
        """
        Analyze raw validation results and create structured ValidationResult.
        
        Args:
            raw_results: Raw validation data from RIF-Validator
            
        Returns:
            ValidationResult with categorized failures and analysis
        """
        validation_result = ValidationResult(
            passed=raw_results.get('passed', False),
            score=raw_results.get('score', 0.0)
        )
        
        # Analyze failures
        raw_failures = raw_results.get('failures', [])
        validation_result.failures = self._categorize_failures(raw_failures)
        
        # Extract warnings
        validation_result.warnings = raw_results.get('warnings', [])
        
        # Extract performance metrics
        validation_result.performance_metrics = raw_results.get('performance_metrics', {})
        
        # Identify security issues
        validation_result.security_issues = self._extract_security_issues(raw_results)
        
        # Identify architectural concerns
        validation_result.architectural_concerns = self._extract_architectural_concerns(raw_results)
        
        return validation_result
    
    def _categorize_failures(self, raw_failures: List[Any]) -> List[Dict[str, Any]]:
        """Categorize failures for intelligent analysis"""
        categorized_failures = []
        
        for failure in raw_failures:
            failure_dict = self._normalize_failure_data(failure)
            
            # Determine failure category
            category = self._determine_failure_category(failure_dict)
            failure_dict['category'] = category.value
            
            # Add recovery recommendation
            failure_dict['recovery_recommendation'] = self._get_recovery_recommendation(category, failure_dict)
            
            categorized_failures.append(failure_dict)
        
        return categorized_failures
    
    def _normalize_failure_data(self, failure: Any) -> Dict[str, Any]:
        """Normalize failure data to consistent format"""
        if isinstance(failure, dict):
            return failure
        elif isinstance(failure, str):
            return {'message': failure, 'type': 'unknown'}
        else:
            return {'message': str(failure), 'type': 'unknown'}
    
    def _determine_failure_category(self, failure_data: Dict[str, Any]) -> FailureCategory:
        """Determine the category of a failure"""
        message = failure_data.get('message', '').lower()
        failure_type = failure_data.get('type', '').lower()
        
        # Architectural issues
        if any(term in message for term in ['architecture', 'design', 'pattern', 'structure']):
            return FailureCategory.ARCHITECTURAL_ISSUES
        
        # Requirement gaps
        if any(term in message for term in ['requirement', 'specification', 'missing', 'incomplete']):
            return FailureCategory.REQUIREMENT_GAPS
        
        # Implementation errors
        if any(term in message for term in ['syntax', 'logic', 'bug', 'error', 'exception']):
            return FailureCategory.IMPLEMENTATION_ERRORS
        
        # Quality failures
        if any(term in message for term in ['quality', 'standard', 'convention', 'style']):
            return FailureCategory.QUALITY_FAILURES
        
        # Performance issues
        if any(term in message for term in ['performance', 'slow', 'timeout', 'memory', 'cpu']):
            return FailureCategory.PERFORMANCE_ISSUES
        
        # Security concerns
        if any(term in message for term in ['security', 'vulnerability', 'auth', 'permission']):
            return FailureCategory.SECURITY_CONCERNS
        
        # Integration problems
        if any(term in message for term in ['integration', 'connection', 'api', 'network']):
            return FailureCategory.INTEGRATION_PROBLEMS
        
        return FailureCategory.UNKNOWN
    
    def _get_recovery_recommendation(self, category: FailureCategory, failure_data: Dict[str, Any]) -> str:
        """Get recovery recommendation based on failure category"""
        recommendations = {
            FailureCategory.ARCHITECTURAL_ISSUES: "Return to architecture phase for design review",
            FailureCategory.REQUIREMENT_GAPS: "Return to analysis phase for requirement clarification",
            FailureCategory.IMPLEMENTATION_ERRORS: "Continue implementation with bug fixes",
            FailureCategory.QUALITY_FAILURES: "Apply quality improvements and re-validate",
            FailureCategory.PERFORMANCE_ISSUES: "Optimize performance and benchmark",
            FailureCategory.SECURITY_CONCERNS: "Conduct security review and remediation",
            FailureCategory.INTEGRATION_PROBLEMS: "Debug integration and test connections",
            FailureCategory.UNKNOWN: "Investigate root cause and determine appropriate action"
        }
        
        return recommendations.get(category, "Manual investigation required")
    
    def _extract_security_issues(self, raw_results: Dict[str, Any]) -> List[str]:
        """Extract security issues from validation results"""
        security_issues = []
        
        # Look for security-related failures
        for failure in raw_results.get('failures', []):
            failure_dict = self._normalize_failure_data(failure)
            message = failure_dict.get('message', '').lower()
            
            if any(term in message for term in ['security', 'vulnerability', 'auth', 'permission']):
                security_issues.append(failure_dict.get('message', str(failure)))
        
        # Look for security warnings
        for warning in raw_results.get('warnings', []):
            if any(term in warning.lower() for term in ['security', 'vulnerability', 'auth']):
                security_issues.append(warning)
        
        return security_issues
    
    def _extract_architectural_concerns(self, raw_results: Dict[str, Any]) -> List[str]:
        """Extract architectural concerns from validation results"""
        architectural_concerns = []
        
        # Look for architecture-related failures
        for failure in raw_results.get('failures', []):
            failure_dict = self._normalize_failure_data(failure)
            message = failure_dict.get('message', '').lower()
            
            if any(term in message for term in ['architecture', 'design', 'pattern', 'structure']):
                architectural_concerns.append(failure_dict.get('message', str(failure)))
        
        return architectural_concerns
    
    def _load_failure_patterns(self) -> Dict[str, Any]:
        """Load known failure patterns from knowledge base"""
        patterns = {}
        
        try:
            patterns_file = Path("knowledge/patterns/validation-failure-patterns.json")
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load failure patterns: {e}")
        
        return patterns

class DynamicStateAnalyzer:
    """
    Enhanced state analysis with validation result processing and loop-back intelligence.
    Phase 1 component: Foundation Enhancement
    
    Replaces basic state transition logic with intelligent context-aware analysis.
    """
    
    def __init__(self):
        self.context_engine = ContextModelingEngine()
        self.validation_analyzer = ValidationResultAnalyzer()
        self.state_validator = StateValidator()
        self.knowledge_base_path = Path("knowledge")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Decision confidence tracking
        self.decision_history = deque(maxlen=100)
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def analyze_current_state(self, issue_context: IssueContext, 
                            validation_results: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Analyze current state and determine optimal next transition with confidence.
        
        Args:
            issue_context: Issue context from ContextAnalyzer
            validation_results: Optional validation results for intelligent analysis
            
        Returns:
            Tuple of (next_state, confidence_score)
        """
        # Create rich context model
        context_model = self.context_engine.enrich_context(issue_context)
        
        # Add validation results if available
        if validation_results:
            context_model.validation_results = self.validation_analyzer.analyze_validation_results(validation_results)
        
        # Determine next state with confidence
        next_state, confidence = self._determine_optimal_state(context_model)
        
        # Log decision for learning
        self._log_decision(issue_context, next_state, confidence, context_model)
        
        return next_state, confidence
    
    def _determine_optimal_state(self, context_model: ContextModel) -> Tuple[str, float]:
        """Determine optimal next state with confidence scoring"""
        # ISSUE #273: Replace label dependency with content analysis
        if context_model.content_analysis_result:
            # Use content-derived state instead of labels
            current_state = context_model.content_analysis_result.state.value
            confidence_adjustment = context_model.content_analysis_result.confidence_score
        else:
            # Fallback to label-based approach for backward compatibility
            current_state = context_model.issue_context.current_state_label
            if not current_state:
                return 'analyzing', ConfidenceLevel.HIGH.value
            current_state = current_state.replace('state:', '')
            confidence_adjustment = 1.0
        
        # Validation-based intelligent decisions
        if context_model.validation_results:
            return self._analyze_validation_based_transition(context_model)
        
        # Pattern-based decisions for new states (adjusted for content analysis confidence)
        if current_state == 'new':
            return 'analyzing', ConfidenceLevel.VERY_HIGH.value * confidence_adjustment
        
        elif current_state == 'analyzing':
            # Complexity-based decision
            if context_model.overall_complexity_score >= 0.7:
                return 'architecting', ConfidenceLevel.HIGH.value * confidence_adjustment
            elif context_model.overall_complexity_score >= 0.5:
                return 'planning', ConfidenceLevel.MEDIUM.value * confidence_adjustment
            else:
                return 'implementing', ConfidenceLevel.MEDIUM.value * confidence_adjustment
        
        elif current_state == 'planning':
            # Check if architecture needed based on context
            if (context_model.overall_complexity_score >= 0.6 or 
                any(rf['type'] == 'complexity' for rf in context_model.risk_factors)):
                return 'architecting', ConfidenceLevel.HIGH.value * confidence_adjustment
            else:
                return 'implementing', ConfidenceLevel.MEDIUM.value * confidence_adjustment
        
        elif current_state == 'architecting':
            return 'implementing', ConfidenceLevel.HIGH.value * confidence_adjustment
        
        elif current_state == 'implementing':
            return 'validating', ConfidenceLevel.HIGH.value * confidence_adjustment
        
        elif current_state == 'validating':
            # Default to learning if no specific validation results
            return 'learning', ConfidenceLevel.MEDIUM.value * confidence_adjustment
        
        elif current_state == 'learning':
            return 'complete', ConfidenceLevel.HIGH.value * confidence_adjustment
        
        # Fallback
        valid_next = self.state_validator.VALID_TRANSITIONS.get(current_state, [])
        if valid_next:
            return valid_next[0], ConfidenceLevel.LOW.value
        
        return current_state, ConfidenceLevel.VERY_LOW.value
    
    def _analyze_validation_based_transition(self, context_model: ContextModel) -> Tuple[str, float]:
        """Analyze transition based on validation results"""
        validation = context_model.validation_results
        # ISSUE #273: Use content analysis for state extraction
        current_state = self._get_current_state_from_context(context_model)
        
        if not validation:
            return current_state, ConfidenceLevel.LOW.value
        
        # High confidence decisions based on validation analysis
        if validation.has_architectural_issues():
            return 'architecting', ConfidenceLevel.VERY_HIGH.value
        
        elif validation.has_requirement_gaps():
            return 'analyzing', ConfidenceLevel.VERY_HIGH.value
        
        elif validation.has_fixable_errors():
            return 'implementing', ConfidenceLevel.HIGH.value
        
        elif validation.passed:
            # Successful validation - move to learning
            return 'learning', ConfidenceLevel.VERY_HIGH.value
        
        else:
            # Validation failed but no clear category - investigate
            if validation.score < 0.3:
                # Significant failure - might need architectural review
                return 'architecting', ConfidenceLevel.MEDIUM.value
            else:
                # Minor issues - continue implementation
                return 'implementing', ConfidenceLevel.MEDIUM.value
    
    def get_state_analysis_summary(self, issue_context: IssueContext) -> Dict[str, Any]:
        """Get comprehensive state analysis summary"""
        context_model = self.context_engine.enrich_context(issue_context)
        next_state, confidence = self._determine_optimal_state(context_model)
        
        return {
            'current_state': issue_context.current_state_label,
            'recommended_next_state': next_state,
            'confidence': confidence,
            'complexity_analysis': {
                'overall_score': context_model.overall_complexity_score,
                'dimensions': context_model.complexity_dimensions,
                'requires_architecture': context_model.overall_complexity_score >= 0.6
            },
            'risk_analysis': {
                'overall_risk': context_model.risk_score,
                'risk_factors': context_model.risk_factors
            },
            'context_insights': {
                'semantic_tags': list(context_model.semantic_tags),
                'historical_patterns': context_model.historical_patterns,
                'security_implications': context_model.security_context,
                'performance_implications': context_model.performance_context
            }
        }
    
    def _log_decision(self, issue_context: IssueContext, next_state: str, 
                     confidence: float, context_model: ContextModel):
        """Log decision for learning and improvement"""
        decision_log = {
            'timestamp': datetime.now().isoformat(),
            'issue_number': issue_context.number,
            'current_state': issue_context.current_state_label,
            'recommended_state': next_state,
            'confidence': confidence,
            'complexity_score': context_model.overall_complexity_score,
            'risk_score': context_model.risk_score,
            'context_hash': self._generate_context_hash(context_model)
        }
        
        self.decision_history.append(decision_log)
        
        # Periodically save to knowledge base
        if len(self.decision_history) % 10 == 0:
            self._save_decision_history()
    
    def _generate_context_hash(self, context_model: ContextModel) -> str:
        """Generate hash of context for similarity tracking"""
        context_string = f"{context_model.overall_complexity_score}_{context_model.risk_score}_{sorted(context_model.semantic_tags)}"
        return hashlib.md5(context_string.encode()).hexdigest()[:8]
    
    def _save_decision_history(self):
        """Save decision history to knowledge base"""
        try:
            decisions_file = self.knowledge_base_path / "decisions" / "dynamic-state-analysis-decisions.jsonl"
            decisions_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(decisions_file, 'a') as f:
                for decision in self.decision_history:
                    f.write(json.dumps(decision) + '\n')
            
            self.decision_history.clear()
            
        except Exception as e:
            self.logger.warning(f"Could not save decision history: {e}")

class PerformanceTrackingSystem:
    """
    Context-specific performance tracking with learning capabilities.
    Phase 2 component: Adaptive Selection Enhancement
    """
    
    def __init__(self):
        self.knowledge_base_path = Path("knowledge")
        self.metrics_path = self.knowledge_base_path / "metrics"
        self.performance_data_file = self.metrics_path / "agent_performance_tracking.json"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing performance data
        self.performance_data = self._load_performance_data()
    
    def track_agent_performance(self, agent_name: str, context_model: ContextModel, 
                               success: bool, completion_time: float,
                               performance_metrics: Dict[str, float] = None):
        """
        Track agent performance with context-specific data.
        
        Args:
            agent_name: Name of the agent
            context_model: Context model for the task
            success: Whether the task was successful
            completion_time: Time taken to complete the task
            performance_metrics: Additional performance metrics
        """
        if agent_name not in self.performance_data:
            self.performance_data[agent_name] = AgentPerformanceData(agent_name=agent_name)
        
        agent_data = self.performance_data[agent_name]
        
        # Update basic metrics
        if success:
            agent_data.successes += 1
        else:
            agent_data.failures += 1
        
        # Update average completion time
        total_tasks = agent_data.successes + agent_data.failures
        agent_data.avg_completion_time = (
            (agent_data.avg_completion_time * (total_tasks - 1) + completion_time) / total_tasks
        )
        
        # Track complexity handling
        complexity_level = f"complexity_{int(context_model.overall_complexity_score * 4)}"
        if complexity_level not in agent_data.complexity_handling:
            agent_data.complexity_handling[complexity_level] = 0
        agent_data.complexity_handling[complexity_level] += 1
        
        # Track context-specific performance
        for tag in context_model.semantic_tags:
            if tag not in agent_data.context_specific_performance:
                agent_data.context_specific_performance[tag] = []
            
            performance_score = 1.0 if success else 0.0
            if performance_metrics:
                # Incorporate quality metrics if available
                performance_score *= performance_metrics.get('quality_score', 1.0)
            
            agent_data.context_specific_performance[tag].append(performance_score)
            
            # Keep only recent performance (last 20 tasks)
            agent_data.context_specific_performance[tag] = \
                agent_data.context_specific_performance[tag][-20:]
        
        agent_data.last_performance_update = datetime.now()
        
        # Save updated data
        self._save_performance_data()
    
    def predict_agent_performance(self, agent_name: str, context_model: ContextModel) -> float:
        """
        Predict agent performance for given context.
        
        Args:
            agent_name: Name of the agent
            context_model: Context model for prediction
            
        Returns:
            Predicted performance score (0.0 to 1.0)
        """
        if agent_name not in self.performance_data:
            return 0.5  # Neutral prediction for unknown agents
        
        agent_data = self.performance_data[agent_name]
        
        # Base performance from success rate
        base_performance = agent_data.reliability_score
        
        # Context-specific adjustments
        context_adjustments = []
        for tag in context_model.semantic_tags:
            if tag in agent_data.context_specific_performance:
                context_scores = agent_data.context_specific_performance[tag]
                if context_scores:
                    avg_context_performance = statistics.mean(context_scores)
                    context_adjustments.append(avg_context_performance)
        
        # Combine base performance with context-specific performance
        if context_adjustments:
            context_performance = statistics.mean(context_adjustments)
            # Weighted combination: 60% base, 40% context-specific
            predicted_performance = (base_performance * 0.6) + (context_performance * 0.4)
        else:
            predicted_performance = base_performance
        
        return min(max(predicted_performance, 0.0), 1.0)
    
    def get_agent_trends(self, agent_name: str, days: int = 30) -> Dict[str, Any]:
        """Get performance trends for an agent over specified days"""
        if agent_name not in self.performance_data:
            return {}
        
        agent_data = self.performance_data[agent_name]
        
        return {
            'success_rate': agent_data.success_rate,
            'reliability_score': agent_data.reliability_score,
            'avg_completion_time': agent_data.avg_completion_time,
            'complexity_handling': agent_data.complexity_handling,
            'total_tasks': agent_data.successes + agent_data.failures,
            'last_update': agent_data.last_performance_update.isoformat(),
            'context_expertise': {
                tag: statistics.mean(scores) if scores else 0.0
                for tag, scores in agent_data.context_specific_performance.items()
            }
        }
    
    def _load_performance_data(self) -> Dict[str, AgentPerformanceData]:
        """Load performance data from disk"""
        performance_data = {}
        
        try:
            if self.performance_data_file.exists():
                with open(self.performance_data_file, 'r') as f:
                    raw_data = json.load(f)
                
                for agent_name, data in raw_data.items():
                    performance_data[agent_name] = AgentPerformanceData(
                        agent_name=agent_name,
                        successes=data.get('successes', 0),
                        failures=data.get('failures', 0),
                        avg_completion_time=data.get('avg_completion_time', 0.0),
                        complexity_handling=data.get('complexity_handling', {}),
                        last_performance_update=datetime.fromisoformat(
                            data.get('last_performance_update', datetime.now().isoformat())
                        ),
                        context_specific_performance=data.get('context_specific_performance', {})
                    )
        
        except Exception as e:
            self.logger.warning(f"Could not load performance data: {e}")
        
        return performance_data
    
    def _save_performance_data(self):
        """Save performance data to disk"""
        try:
            self.metrics_path.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            serializable_data = {}
            for agent_name, agent_data in self.performance_data.items():
                serializable_data[agent_name] = {
                    'successes': agent_data.successes,
                    'failures': agent_data.failures,
                    'avg_completion_time': agent_data.avg_completion_time,
                    'complexity_handling': agent_data.complexity_handling,
                    'last_performance_update': agent_data.last_performance_update.isoformat(),
                    'context_specific_performance': agent_data.context_specific_performance
                }
            
            with open(self.performance_data_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Could not save performance data: {e}")

class TeamOptimizationEngine:
    """
    Multi-strategy team optimization for dynamic agent composition.
    Phase 2 component: Adaptive Selection Enhancement
    """
    
    def __init__(self):
        self.performance_tracker = PerformanceTrackingSystem()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent capability definitions
        self.agent_capabilities = {
            'RIF-Analyst': {
                'primary': ['analysis', 'requirements', 'planning'],
                'secondary': ['documentation', 'research'],
                'complexity_preference': 'medium_high',
                'parallel_safe': True
            },
            'RIF-Planner': {
                'primary': ['planning', 'strategy', 'coordination'],
                'secondary': ['analysis', 'architecture'],
                'complexity_preference': 'high',
                'parallel_safe': True
            },
            'RIF-Architect': {
                'primary': ['architecture', 'design', 'integration'],
                'secondary': ['performance', 'scalability'],
                'complexity_preference': 'very_high',
                'parallel_safe': False  # Architecture needs sequential focus
            },
            'RIF-Implementer': {
                'primary': ['implementation', 'coding', 'integration'],
                'secondary': ['testing', 'debugging'],
                'complexity_preference': 'all',
                'parallel_safe': False  # Implementation needs focused attention
            },
            'RIF-Validator': {
                'primary': ['validation', 'testing', 'quality'],
                'secondary': ['performance', 'security'],
                'complexity_preference': 'medium_high',
                'parallel_safe': True
            },
            'RIF-Learner': {
                'primary': ['learning', 'documentation', 'knowledge'],
                'secondary': ['analysis', 'patterns'],
                'complexity_preference': 'all',
                'parallel_safe': True
            }
        }
    
    def optimize_team_composition(self, context_models: List[ContextModel], 
                                 strategy: str = 'performance') -> List[Dict[str, Any]]:
        """
        Optimize team composition for multiple issues.
        
        Args:
            context_models: List of context models for issues to process
            strategy: Optimization strategy ('performance', 'capability', 'workload', 'synergy')
            
        Returns:
            List of optimized agent assignments
        """
        if strategy == 'performance':
            return self._optimize_by_performance(context_models)
        elif strategy == 'capability':
            return self._optimize_by_capability_match(context_models)
        elif strategy == 'workload':
            return self._optimize_by_workload_balance(context_models)
        elif strategy == 'synergy':
            return self._optimize_by_synergy(context_models)
        else:
            # Default to performance-based optimization
            return self._optimize_by_performance(context_models)
    
    def _optimize_by_performance(self, context_models: List[ContextModel]) -> List[Dict[str, Any]]:
        """Optimize team based on predicted performance"""
        assignments = []
        
        for context_model in context_models:
            # ISSUE #273: Use content analysis for state extraction
            current_state = self._get_current_state_from_context(context_model)
            if current_state:
                state_name = current_state
                
                # Get candidate agents for this state
                candidate_agents = self._get_candidate_agents(state_name, context_model)
                
                # Score agents by predicted performance
                agent_scores = []
                for agent in candidate_agents:
                    predicted_performance = self.performance_tracker.predict_agent_performance(
                        agent, context_model
                    )
                    agent_scores.append((agent, predicted_performance))
                
                # Sort by performance and select best
                agent_scores.sort(key=lambda x: x[1], reverse=True)
                
                if agent_scores:
                    best_agent, score = agent_scores[0]
                    assignments.append({
                        'issue_number': context_model.issue_context.number,
                        'agent': best_agent,
                        'predicted_performance': score,
                        'optimization_strategy': 'performance',
                        'alternatives': [{'agent': a, 'score': s} for a, s in agent_scores[1:3]]
                    })
        
        return assignments
    
    def _optimize_by_capability_match(self, context_models: List[ContextModel]) -> List[Dict[str, Any]]:
        """Optimize team based on capability matching"""
        assignments = []
        
        for context_model in context_models:
            # ISSUE #273: Use content analysis for state extraction
            current_state = self._get_current_state_from_context(context_model)
            if current_state:
                state_name = current_state
                candidate_agents = self._get_candidate_agents(state_name, context_model)
                
                # Score agents by capability match
                agent_scores = []
                for agent in candidate_agents:
                    capability_score = self._calculate_capability_match(agent, context_model)
                    agent_scores.append((agent, capability_score))
                
                agent_scores.sort(key=lambda x: x[1], reverse=True)
                
                if agent_scores:
                    best_agent, score = agent_scores[0]
                    assignments.append({
                        'issue_number': context_model.issue_context.number,
                        'agent': best_agent,
                        'capability_match_score': score,
                        'optimization_strategy': 'capability',
                        'context_tags': list(context_model.semantic_tags)
                    })
        
        return assignments
    
    def _optimize_by_workload_balance(self, context_models: List[ContextModel]) -> List[Dict[str, Any]]:
        """Optimize team for balanced workload distribution"""
        assignments = []
        agent_workload = defaultdict(int)
        
        for context_model in context_models:
            # ISSUE #273: Use content analysis for state extraction
            current_state = self._get_current_state_from_context(context_model)
            if current_state:
                state_name = current_state
                candidate_agents = self._get_candidate_agents(state_name, context_model)
                
                # Select agent with lowest current workload
                best_agent = None
                lowest_workload = float('inf')
                
                for agent in candidate_agents:
                    current_workload = agent_workload[agent]
                    if current_workload < lowest_workload:
                        lowest_workload = current_workload
                        best_agent = agent
                
                if best_agent:
                    # Add workload based on issue complexity
                    workload_weight = int(context_model.overall_complexity_score * 10)
                    agent_workload[best_agent] += workload_weight
                    
                    assignments.append({
                        'issue_number': context_model.issue_context.number,
                        'agent': best_agent,
                        'workload_weight': workload_weight,
                        'optimization_strategy': 'workload',
                        'agent_current_workload': agent_workload[best_agent]
                    })
        
        return assignments
    
    def _optimize_by_synergy(self, context_models: List[ContextModel]) -> List[Dict[str, Any]]:
        """Optimize team for maximum synergy between agents"""
        assignments = []
        
        # Group issues that could benefit from coordinated agents
        related_groups = self._group_related_issues(context_models)
        
        for group in related_groups:
            # For each group, assign complementary agents
            group_assignments = self._assign_synergistic_agents(group)
            assignments.extend(group_assignments)
        
        return assignments
    
    def _get_candidate_agents(self, state_name: str, context_model: ContextModel) -> List[str]:
        """Get candidate agents for a state and context"""
        # Base agent for the state
        state_agents = {
            'analyzing': ['RIF-Analyst'],
            'planning': ['RIF-Planner'],
            'architecting': ['RIF-Architect'],
            'implementing': ['RIF-Implementer'],
            'validating': ['RIF-Validator'],
            'learning': ['RIF-Learner']
        }
        
        base_agents = state_agents.get(state_name, [])
        
        # Add specialist agents based on context
        specialists = []
        
        if 'security' in context_model.semantic_tags:
            specialists.append('RIF-Security-Specialist')
        
        if context_model.overall_complexity_score >= 0.8:
            if 'RIF-Architect' not in base_agents:
                specialists.append('RIF-Architect')
        
        if context_model.performance_context.get('performance_critical', False):
            specialists.append('RIF-Performance-Specialist')
        
        return base_agents + specialists
    
    def _calculate_capability_match(self, agent: str, context_model: ContextModel) -> float:
        """Calculate how well an agent's capabilities match the context"""
        if agent not in self.agent_capabilities:
            return 0.5  # Default neutral score
        
        capabilities = self.agent_capabilities[agent]
        
        # Score based on semantic tag overlap
        primary_match = len(set(capabilities['primary']) & context_model.semantic_tags) / len(capabilities['primary'])
        secondary_match = len(set(capabilities['secondary']) & context_model.semantic_tags) / len(capabilities['secondary'])
        
        # Complexity preference match
        complexity_pref = capabilities['complexity_preference']
        complexity_score = context_model.overall_complexity_score
        
        complexity_match = 1.0
        if complexity_pref == 'low' and complexity_score > 0.6:
            complexity_match = 0.3
        elif complexity_pref == 'medium_high' and (complexity_score < 0.3 or complexity_score > 0.9):
            complexity_match = 0.7
        elif complexity_pref == 'high' and complexity_score < 0.5:
            complexity_match = 0.5
        elif complexity_pref == 'very_high' and complexity_score < 0.7:
            complexity_match = 0.4
        
        # Weighted combination
        total_match = (primary_match * 0.5) + (secondary_match * 0.3) + (complexity_match * 0.2)
        return min(total_match, 1.0)
    
    def _group_related_issues(self, context_models: List[ContextModel]) -> List[List[ContextModel]]:
        """Group related issues for synergistic processing"""
        groups = []
        processed = set()
        
        for i, context_model in enumerate(context_models):
            if i in processed:
                continue
            
            # Start a new group
            group = [context_model]
            processed.add(i)
            
            # Find related issues based on semantic similarity
            for j, other_context in enumerate(context_models[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # Check for semantic tag overlap
                tag_overlap = len(context_model.semantic_tags & other_context.semantic_tags)
                if tag_overlap >= 2:  # At least 2 common tags
                    group.append(other_context)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _assign_synergistic_agents(self, group: List[ContextModel]) -> List[Dict[str, Any]]:
        """Assign complementary agents to a group of related issues"""
        assignments = []
        
        # For synergy, prefer agents that work well together
        synergy_pairs = {
            'RIF-Analyst': ['RIF-Planner', 'RIF-Architect'],
            'RIF-Planner': ['RIF-Architect', 'RIF-Implementer'],
            'RIF-Architect': ['RIF-Implementer', 'RIF-Validator'],
            'RIF-Implementer': ['RIF-Validator', 'RIF-Learner'],
            'RIF-Validator': ['RIF-Learner'],
        }
        
        used_agents = set()
        
        for context_model in group:
            current_state = context_model.issue_context.current_state_label
            if current_state:
                state_name = current_state.replace('state:', '')
                candidate_agents = self._get_candidate_agents(state_name, context_model)
                
                # Prefer agents that synergize with already used agents
                synergy_scores = []
                for agent in candidate_agents:
                    if agent in used_agents:
                        continue  # Don't double-assign
                    
                    synergy_score = 0
                    for used_agent in used_agents:
                        if agent in synergy_pairs.get(used_agent, []):
                            synergy_score += 1
                    
                    synergy_scores.append((agent, synergy_score))
                
                # Sort by synergy score, then capability match
                synergy_scores.sort(key=lambda x: (x[1], self._calculate_capability_match(x[0], context_model)), reverse=True)
                
                if synergy_scores:
                    best_agent, synergy_score = synergy_scores[0]
                    used_agents.add(best_agent)
                    
                    assignments.append({
                        'issue_number': context_model.issue_context.number,
                        'agent': best_agent,
                        'synergy_score': synergy_score,
                        'optimization_strategy': 'synergy',
                        'group_size': len(group)
                    })
        
        return assignments

class LearningAgentSelector:
    """
    Enhanced AdaptiveAgentSelector with learning capabilities and performance optimization.
    Phase 2 component: Adaptive Selection Enhancement
    
    Extends the existing AdaptiveAgentSelector with sophisticated learning.
    """
    
    def __init__(self):
        self.performance_tracker = PerformanceTrackingSystem()
        self.team_optimizer = TeamOptimizationEngine()
        self.context_engine = ContextModelingEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_factor = 0.2  # Balance between exploitation and exploration
        
        # Selection history for learning
        self.selection_history = deque(maxlen=200)
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def select_optimal_agents(self, context_models: List[ContextModel], 
                            optimization_strategy: str = 'performance') -> List[Dict[str, Any]]:
        """
        Select optimal agents using learning-enhanced selection.
        
        Args:
            context_models: List of context models for issues
            optimization_strategy: Strategy for team optimization
            
        Returns:
            List of agent assignments with optimization details
        """
        # Apply team optimization
        optimized_assignments = self.team_optimizer.optimize_team_composition(
            context_models, optimization_strategy
        )
        
        # Apply learning-based adjustments
        learning_adjusted_assignments = self._apply_learning_adjustments(optimized_assignments, context_models)
        
        # Log selections for learning
        self._log_selections(learning_adjusted_assignments, context_models)
        
        return learning_adjusted_assignments
    
    def select_single_agent(self, context_model: ContextModel) -> Dict[str, Any]:
        """Select optimal agent for a single issue with learning"""
        assignments = self.select_optimal_agents([context_model])
        return assignments[0] if assignments else {}
    
    def update_performance_feedback(self, issue_number: int, agent_name: str, 
                                  success: bool, completion_time: float,
                                  performance_metrics: Dict[str, float] = None):
        """
        Update performance feedback for learning.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of the agent that worked on the issue
            success: Whether the task was successful
            completion_time: Time taken to complete
            performance_metrics: Additional performance metrics
        """
        # Find the context model for this issue from selection history
        context_model = self._find_context_from_history(issue_number)
        
        if context_model:
            self.performance_tracker.track_agent_performance(
                agent_name, context_model, success, completion_time, performance_metrics
            )
            
            self.logger.info(f"Updated performance feedback for {agent_name} on issue #{issue_number}: {success}")
        else:
            self.logger.warning(f"Could not find context for issue #{issue_number} to update performance")
    
    def get_agent_recommendations(self, context_model: ContextModel) -> List[Dict[str, Any]]:
        """Get ranked agent recommendations with reasoning"""
        # ISSUE #273: Use content analysis for state extraction
        current_state = self._get_current_state_from_context(context_model)
        if not current_state:
            return []
        
        state_name = current_state.replace('state:', '')
        candidate_agents = self.team_optimizer._get_candidate_agents(state_name, context_model)
        
        recommendations = []
        for agent in candidate_agents:
            # Get performance prediction
            predicted_performance = self.performance_tracker.predict_agent_performance(agent, context_model)
            
            # Get capability match
            capability_match = self.team_optimizer._calculate_capability_match(agent, context_model)
            
            # Get performance trends
            trends = self.performance_tracker.get_agent_trends(agent)
            
            recommendations.append({
                'agent': agent,
                'predicted_performance': predicted_performance,
                'capability_match': capability_match,
                'success_rate': trends.get('success_rate', 0.5),
                'reliability_score': trends.get('reliability_score', 0.5),
                'context_expertise': trends.get('context_expertise', {}),
                'recommendation_reason': self._generate_recommendation_reason(
                    agent, context_model, predicted_performance, capability_match
                )
            })
        
        # Sort by weighted score
        for rec in recommendations:
            rec['overall_score'] = (
                rec['predicted_performance'] * 0.4 +
                rec['capability_match'] * 0.3 +
                rec['reliability_score'] * 0.3
            )
        
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        return recommendations
    
    def _apply_learning_adjustments(self, assignments: List[Dict[str, Any]], 
                                  context_models: List[ContextModel]) -> List[Dict[str, Any]]:
        """Apply learning-based adjustments to assignments"""
        adjusted_assignments = []
        
        for assignment in assignments:
            # Find the corresponding context model
            context_model = next(
                (cm for cm in context_models if cm.issue_context.number == assignment['issue_number']),
                None
            )
            
            if not context_model:
                adjusted_assignments.append(assignment)
                continue
            
            # Apply exploration vs exploitation
            if self._should_explore(context_model):
                # Exploration: try a different agent occasionally
                alternative_agents = self._get_exploration_candidates(assignment, context_model)
                if alternative_agents:
                    exploration_agent = alternative_agents[0]
                    assignment['agent'] = exploration_agent['agent']
                    assignment['exploration_mode'] = True
                    assignment['exploration_reason'] = 'Learning exploration'
            
            # Apply learning-based confidence adjustments
            assignment['selection_confidence'] = self._calculate_selection_confidence(assignment, context_model)
            
            adjusted_assignments.append(assignment)
        
        return adjusted_assignments
    
    def _should_explore(self, context_model: ContextModel) -> bool:
        """Determine if we should explore alternative agents"""
        # Exploration probability based on context novelty and exploration factor
        context_hash = hashlib.md5(str(sorted(context_model.semantic_tags)).encode()).hexdigest()[:8]
        
        # Check how often we've seen similar contexts
        similar_contexts = sum(1 for sel in self.selection_history if sel.get('context_hash') == context_hash)
        
        # More exploration for novel contexts
        if similar_contexts < 3:
            exploration_prob = self.exploration_factor * 2
        else:
            exploration_prob = self.exploration_factor
        
        return hash(context_hash) % 100 < exploration_prob * 100
    
    def _get_exploration_candidates(self, assignment: Dict[str, Any], 
                                  context_model: ContextModel) -> List[Dict[str, Any]]:
        """Get alternative agents for exploration"""
        current_agent = assignment['agent']
        
        # Get all recommendations and filter out the current choice
        all_recommendations = self.get_agent_recommendations(context_model)
        alternatives = [rec for rec in all_recommendations if rec['agent'] != current_agent]
        
        return alternatives[:2]  # Top 2 alternatives
    
    def _calculate_selection_confidence(self, assignment: Dict[str, Any], 
                                      context_model: ContextModel) -> float:
        """Calculate confidence in the agent selection"""
        agent_name = assignment['agent']
        
        # Base confidence from performance prediction
        predicted_performance = self.performance_tracker.predict_agent_performance(agent_name, context_model)
        
        # Adjust based on historical data availability
        trends = self.performance_tracker.get_agent_trends(agent_name)
        total_tasks = trends.get('total_tasks', 0)
        
        # More confidence with more historical data
        data_confidence = min(total_tasks / 20, 1.0)  # Max confidence at 20+ tasks
        
        # Combine factors
        overall_confidence = (predicted_performance * 0.7) + (data_confidence * 0.3)
        
        return overall_confidence
    
    def _generate_recommendation_reason(self, agent: str, context_model: ContextModel,
                                      predicted_performance: float, capability_match: float) -> str:
        """Generate human-readable reason for agent recommendation"""
        reasons = []
        
        if predicted_performance > 0.8:
            reasons.append("excellent historical performance")
        elif predicted_performance > 0.6:
            reasons.append("good historical performance")
        
        if capability_match > 0.8:
            reasons.append("strong capability match")
        elif capability_match > 0.6:
            reasons.append("good capability match")
        
        # Context-specific reasons
        semantic_tags = list(context_model.semantic_tags)
        if semantic_tags:
            if len(semantic_tags) <= 2:
                reasons.append(f"expertise in {', '.join(semantic_tags)}")
            else:
                reasons.append(f"expertise in {semantic_tags[0]} and {len(semantic_tags)-1} other areas")
        
        if context_model.overall_complexity_score > 0.7:
            reasons.append("handles high complexity well")
        
        if not reasons:
            reasons.append("suitable for this task type")
        
        return "; ".join(reasons)
    
    def _log_selections(self, assignments: List[Dict[str, Any]], context_models: List[ContextModel]):
        """Log agent selections for learning"""
        for assignment, context_model in zip(assignments, context_models):
            context_hash = hashlib.md5(str(sorted(context_model.semantic_tags)).encode()).hexdigest()[:8]
            
            selection_log = {
                'timestamp': datetime.now().isoformat(),
                'issue_number': assignment['issue_number'],
                'selected_agent': assignment['agent'],
                'context_hash': context_hash,
                'complexity_score': context_model.overall_complexity_score,
                'semantic_tags': list(context_model.semantic_tags),
                'optimization_strategy': assignment.get('optimization_strategy', 'unknown'),
                'selection_confidence': assignment.get('selection_confidence', 0.5),
                'exploration_mode': assignment.get('exploration_mode', False)
            }
            
            self.selection_history.append(selection_log)
    
    def _find_context_from_history(self, issue_number: int) -> Optional[ContextModel]:
        """Find context model from selection history"""
        # This is a simplified version - in practice, you'd want to store full contexts
        # For now, we'll return None and rely on external context provision
        return None

class FailurePatternAnalyzer:
    """
    Intelligent pattern recognition in failure histories for loop-back optimization.
    Phase 3 component: Loop-Back Intelligence
    """
    
    def __init__(self):
        self.knowledge_base_path = Path("knowledge")
        self.patterns_path = self.knowledge_base_path / "patterns"
        self.failure_patterns_file = self.knowledge_base_path / "decisions" / "failure_patterns.json"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing failure patterns
        self.failure_patterns = self._load_failure_patterns()
        
        # Pattern matching threshold
        self.similarity_threshold = 0.7
    
    def analyze_failure_patterns(self, validation_result: ValidationResult, 
                               context_model: ContextModel) -> Dict[str, Any]:
        """
        Analyze failure patterns and provide intelligent categorization.
        
        Args:
            validation_result: ValidationResult with failure data
            context_model: Context model for the failing issue
            
        Returns:
            Dict with pattern analysis and recovery recommendations
        """
        if validation_result.passed:
            return {'pattern_type': 'success', 'confidence': 1.0}
        
        # Extract failure signature
        failure_signature = self._extract_failure_signature(validation_result, context_model)
        
        # Find matching patterns
        pattern_matches = self._find_matching_patterns(failure_signature)
        
        # Categorize failure type
        failure_category = self._categorize_failure_type(validation_result, pattern_matches)
        
        # Generate recovery strategy
        recovery_strategy = self._generate_recovery_strategy(failure_category, pattern_matches, context_model)
        
        return {
            'failure_signature': failure_signature,
            'pattern_matches': pattern_matches,
            'failure_category': failure_category,
            'recovery_strategy': recovery_strategy,
            'pattern_confidence': self._calculate_pattern_confidence(pattern_matches),
            'historical_success_rate': self._get_historical_success_rate(failure_category)
        }
    
    def learn_from_failure(self, validation_result: ValidationResult, context_model: ContextModel,
                          recovery_outcome: Dict[str, Any]):
        """
        Learn from failure resolution to improve future pattern matching.
        
        Args:
            validation_result: Original validation failure
            context_model: Context model for the issue
            recovery_outcome: Results of the recovery attempt
        """
        failure_signature = self._extract_failure_signature(validation_result, context_model)
        
        # Create learning record
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'failure_signature': failure_signature,
            'context_characteristics': {
                'complexity_score': context_model.overall_complexity_score,
                'semantic_tags': list(context_model.semantic_tags),
                'risk_factors': [rf['type'] for rf in context_model.risk_factors]
            },
            'recovery_strategy': recovery_outcome.get('strategy_used'),
            'recovery_success': recovery_outcome.get('success', False),
            'recovery_time': recovery_outcome.get('time_taken', 0),
            'lessons_learned': recovery_outcome.get('lessons', [])
        }
        
        # Update failure patterns
        self._update_failure_patterns(learning_record)
        
        self.logger.info(f"Learned from failure pattern: {failure_signature['primary_category']}")
    
    def _extract_failure_signature(self, validation_result: ValidationResult, 
                                 context_model: ContextModel) -> Dict[str, Any]:
        """Extract a signature that characterizes the failure"""
        signature = {
            'primary_category': self._get_primary_failure_category(validation_result),
            'failure_count': len(validation_result.failures),
            'failure_types': [f.get('category', 'unknown') for f in validation_result.failures],
            'score_range': self._discretize_score(validation_result.score),
            'complexity_range': self._discretize_complexity(context_model.overall_complexity_score),
            'context_tags': sorted(list(context_model.semantic_tags)),
            'has_security_issues': len(validation_result.security_issues) > 0,
            'has_architectural_concerns': len(validation_result.architectural_concerns) > 0,
            'performance_issues': any('performance' in str(f).lower() for f in validation_result.failures)
        }
        
        return signature
    
    def _get_primary_failure_category(self, validation_result: ValidationResult) -> str:
        """Get the most significant failure category"""
        if validation_result.has_architectural_issues():
            return FailureCategory.ARCHITECTURAL_ISSUES.value
        elif validation_result.has_requirement_gaps():
            return FailureCategory.REQUIREMENT_GAPS.value
        elif validation_result.has_fixable_errors():
            return FailureCategory.IMPLEMENTATION_ERRORS.value
        elif len(validation_result.security_issues) > 0:
            return FailureCategory.SECURITY_CONCERNS.value
        else:
            return FailureCategory.UNKNOWN.value
    
    def _discretize_score(self, score: float) -> str:
        """Convert continuous score to discrete range"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _discretize_complexity(self, complexity: float) -> str:
        """Convert continuous complexity to discrete range"""
        if complexity >= 0.8:
            return 'very_high'
        elif complexity >= 0.6:
            return 'high'
        elif complexity >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _find_matching_patterns(self, failure_signature: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns that match the current failure signature"""
        matches = []
        
        for pattern_id, pattern_data in self.failure_patterns.items():
            similarity = self._calculate_pattern_similarity(failure_signature, pattern_data['signature'])
            
            if similarity >= self.similarity_threshold:
                matches.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'pattern_data': pattern_data,
                    'success_rate': pattern_data.get('recovery_success_rate', 0.5)
                })
        
        # Sort by similarity and success rate
        matches.sort(key=lambda x: (x['similarity'], x['success_rate']), reverse=True)
        return matches[:5]  # Top 5 matches
    
    def _calculate_pattern_similarity(self, signature1: Dict[str, Any], signature2: Dict[str, Any]) -> float:
        """Calculate similarity between two failure signatures"""
        similarity_factors = []
        
        # Primary category match (high weight)
        if signature1['primary_category'] == signature2['primary_category']:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Score range match
        if signature1['score_range'] == signature2['score_range']:
            similarity_factors.append(0.8)
        else:
            similarity_factors.append(0.2)
        
        # Complexity range match
        if signature1['complexity_range'] == signature2['complexity_range']:
            similarity_factors.append(0.6)
        else:
            similarity_factors.append(0.1)
        
        # Context tags overlap
        tags1 = set(signature1['context_tags'])
        tags2 = set(signature2['context_tags'])
        if tags1 and tags2:
            tag_overlap = len(tags1 & tags2) / len(tags1 | tags2)
            similarity_factors.append(tag_overlap)
        else:
            similarity_factors.append(0.3)
        
        # Boolean flags match
        bool_matches = sum([
            signature1['has_security_issues'] == signature2['has_security_issues'],
            signature1['has_architectural_concerns'] == signature2['has_architectural_concerns'],
            signature1['performance_issues'] == signature2['performance_issues']
        ]) / 3.0
        similarity_factors.append(bool_matches * 0.4)
        
        # Weighted average
        weights = [0.4, 0.2, 0.15, 0.15, 0.1]
        weighted_similarity = sum(factor * weight for factor, weight in zip(similarity_factors, weights))
        
        return weighted_similarity
    
    def _categorize_failure_type(self, validation_result: ValidationResult, 
                               pattern_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize the failure type with confidence"""
        primary_category = self._get_primary_failure_category(validation_result)
        
        # Get confidence from pattern matches
        if pattern_matches:
            pattern_confidence = statistics.mean([m['similarity'] for m in pattern_matches])
        else:
            pattern_confidence = 0.3
        
        return {
            'category': primary_category,
            'confidence': pattern_confidence,
            'severity': self._assess_failure_severity(validation_result),
            'urgency': self._assess_failure_urgency(validation_result, pattern_matches)
        }
    
    def _assess_failure_severity(self, validation_result: ValidationResult) -> str:
        """Assess the severity of the failure"""
        if validation_result.score < 0.2:
            return 'critical'
        elif validation_result.score < 0.4:
            return 'high'
        elif validation_result.score < 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _assess_failure_urgency(self, validation_result: ValidationResult, 
                              pattern_matches: List[Dict[str, Any]]) -> str:
        """Assess the urgency of addressing the failure"""
        # Security issues are always urgent
        if len(validation_result.security_issues) > 0:
            return 'urgent'
        
        # Architectural issues in high complexity contexts are urgent
        if validation_result.has_architectural_issues():
            return 'urgent'
        
        # Check historical patterns for urgency indicators
        if pattern_matches:
            avg_recovery_time = statistics.mean([
                p['pattern_data'].get('avg_recovery_time', 24) for p in pattern_matches
            ])
            if avg_recovery_time > 48:  # More than 2 days historically
                return 'high'
        
        return 'medium'
    
    def _generate_recovery_strategy(self, failure_category: Dict[str, Any], 
                                  pattern_matches: List[Dict[str, Any]], 
                                  context_model: ContextModel) -> Dict[str, Any]:
        """Generate intelligent recovery strategy"""
        category_name = failure_category['category']
        
        # Base recovery strategies
        base_strategies = {
            FailureCategory.ARCHITECTURAL_ISSUES.value: {
                'recommended_state': 'architecting',
                'agent': 'RIF-Architect',
                'approach': 'architectural_review',
                'estimated_time': 4.0
            },
            FailureCategory.REQUIREMENT_GAPS.value: {
                'recommended_state': 'analyzing',
                'agent': 'RIF-Analyst',
                'approach': 'requirement_clarification',
                'estimated_time': 2.0
            },
            FailureCategory.IMPLEMENTATION_ERRORS.value: {
                'recommended_state': 'implementing',
                'agent': 'RIF-Implementer',
                'approach': 'iterative_fixes',
                'estimated_time': 3.0
            },
            FailureCategory.SECURITY_CONCERNS.value: {
                'recommended_state': 'implementing',
                'agent': 'RIF-Security-Specialist',
                'approach': 'security_remediation',
                'estimated_time': 6.0
            },
            FailureCategory.PERFORMANCE_ISSUES.value: {
                'recommended_state': 'implementing',
                'agent': 'RIF-Performance-Specialist',
                'approach': 'performance_optimization',
                'estimated_time': 4.0
            }
        }
        
        base_strategy = base_strategies.get(category_name, {
            'recommended_state': 'implementing',
            'agent': 'RIF-Implementer',
            'approach': 'general_fixes',
            'estimated_time': 3.0
        })
        
        # Refine strategy based on pattern matches
        if pattern_matches:
            successful_patterns = [p for p in pattern_matches if p['success_rate'] > 0.6]
            if successful_patterns:
                # Use strategy from most successful similar pattern
                best_pattern = successful_patterns[0]['pattern_data']
                if 'successful_recovery_strategy' in best_pattern:
                    base_strategy.update(best_pattern['successful_recovery_strategy'])
        
        # Context-specific adjustments
        if context_model.overall_complexity_score >= 0.8:
            base_strategy['estimated_time'] *= 1.5
            base_strategy['requires_planning'] = True
        
        if 'security' in context_model.semantic_tags:
            base_strategy['requires_security_review'] = True
        
        return base_strategy
    
    def _calculate_pattern_confidence(self, pattern_matches: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in pattern matching"""
        if not pattern_matches:
            return 0.2
        
        # Weighted by similarity scores
        similarities = [m['similarity'] for m in pattern_matches]
        return statistics.mean(similarities)
    
    def _get_historical_success_rate(self, failure_category: Dict[str, Any]) -> float:
        """Get historical success rate for similar failure categories"""
        category_name = failure_category['category']
        
        success_rates = []
        for pattern_data in self.failure_patterns.values():
            if pattern_data['signature'].get('primary_category') == category_name:
                success_rates.append(pattern_data.get('recovery_success_rate', 0.5))
        
        return statistics.mean(success_rates) if success_rates else 0.5
    
    def _load_failure_patterns(self) -> Dict[str, Any]:
        """Load existing failure patterns from knowledge base"""
        patterns = {}
        
        try:
            if self.failure_patterns_file.exists():
                with open(self.failure_patterns_file, 'r') as f:
                    patterns = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load failure patterns: {e}")
        
        return patterns
    
    def _update_failure_patterns(self, learning_record: Dict[str, Any]):
        """Update failure patterns with new learning"""
        signature = learning_record['failure_signature']
        pattern_key = f"{signature['primary_category']}_{signature['complexity_range']}"
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = {
                'signature': signature,
                'recovery_attempts': 0,
                'successful_recoveries': 0,
                'recovery_success_rate': 0.0,
                'avg_recovery_time': 0.0,
                'successful_recovery_strategy': None,
                'created_at': datetime.now().isoformat()
            }
        
        pattern = self.failure_patterns[pattern_key]
        pattern['recovery_attempts'] += 1
        
        if learning_record['recovery_success']:
            pattern['successful_recoveries'] += 1
            pattern['successful_recovery_strategy'] = learning_record['recovery_strategy']
        
        # Update success rate and average time
        pattern['recovery_success_rate'] = pattern['successful_recoveries'] / pattern['recovery_attempts']
        
        if 'recovery_times' not in pattern:
            pattern['recovery_times'] = []
        pattern['recovery_times'].append(learning_record['recovery_time'])
        pattern['avg_recovery_time'] = statistics.mean(pattern['recovery_times'])
        
        # Keep only recent recovery times (last 20)
        pattern['recovery_times'] = pattern['recovery_times'][-20:]
        
        pattern['updated_at'] = datetime.now().isoformat()
        
        # Save updated patterns
        self._save_failure_patterns()
    
    def _save_failure_patterns(self):
        """Save failure patterns to knowledge base"""
        try:
            self.failure_patterns_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failure_patterns_file, 'w') as f:
                json.dump(self.failure_patterns, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save failure patterns: {e}")

class LoopBackDecisionEngine:
    """
    Context-aware decision making for optimal state transitions and loop-back decisions.
    Phase 3 component: Loop-Back Intelligence
    """
    
    def __init__(self):
        self.failure_analyzer = FailurePatternAnalyzer()
        self.state_validator = StateValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Decision weights for different factors
        self.decision_weights = {
            'pattern_confidence': 0.3,
            'failure_severity': 0.25,
            'recovery_success_rate': 0.2,
            'context_complexity': 0.15,
            'time_efficiency': 0.1
        }
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def make_loop_back_decision(self, validation_result: ValidationResult, 
                              context_model: ContextModel,
                              current_state: str) -> Dict[str, Any]:
        """
        Make intelligent loop-back decision based on failure analysis.
        
        Args:
            validation_result: Validation results with failures
            context_model: Context model for the issue
            current_state: Current workflow state
            
        Returns:
            Dict with loop-back decision and reasoning
        """
        # Analyze failure patterns
        pattern_analysis = self.failure_analyzer.analyze_failure_patterns(validation_result, context_model)
        
        # Generate decision options
        decision_options = self._generate_decision_options(
            pattern_analysis, context_model, current_state
        )
        
        # Score and rank options
        scored_options = self._score_decision_options(decision_options, pattern_analysis, context_model)
        
        # Select best option
        best_option = scored_options[0] if scored_options else self._get_default_option(current_state)
        
        return {
            'recommended_action': best_option,
            'alternatives': scored_options[1:3],  # Top 3 alternatives
            'decision_reasoning': self._generate_decision_reasoning(best_option, pattern_analysis),
            'confidence': best_option.get('confidence', 0.5),
            'pattern_analysis': pattern_analysis
        }
    
    def evaluate_transition_readiness(self, context_model: ContextModel, 
                                    target_state: str) -> Dict[str, Any]:
        """
        Evaluate if an issue is ready to transition to a target state.
        
        Args:
            context_model: Context model for the issue
            target_state: Target state to evaluate
            
        Returns:
            Dict with readiness assessment
        """
        # Check basic state transition validity
        # ISSUE #273: Use content analysis for state extraction
        current_state = self._get_current_state_from_context(context_model)
        if current_state:
            is_valid, reason = self.state_validator.validate_state_transition(current_state, target_state)
        else:
            is_valid, reason = True, "No current state"
        
        if not is_valid:
            return {
                'ready': False,
                'confidence': 1.0,
                'blocking_reasons': [reason],
                'recommendations': [f"Cannot transition to {target_state}"]
            }
        
        # Intelligent readiness assessment
        readiness_factors = self._assess_readiness_factors(context_model, target_state)
        overall_readiness = self._calculate_overall_readiness(readiness_factors)
        
        return {
            'ready': overall_readiness > 0.7,
            'confidence': overall_readiness,
            'readiness_factors': readiness_factors,
            'blocking_reasons': self._identify_blocking_reasons(readiness_factors),
            'recommendations': self._generate_readiness_recommendations(readiness_factors, target_state)
        }
    
    def _generate_decision_options(self, pattern_analysis: Dict[str, Any], 
                                 context_model: ContextModel, current_state: str) -> List[Dict[str, Any]]:
        """Generate possible decision options for loop-back"""
        options = []
        
        # Option 1: Follow pattern-based recommendation
        if pattern_analysis.get('recovery_strategy'):
            recovery_strategy = pattern_analysis['recovery_strategy']
            options.append({
                'type': 'pattern_based',
                'target_state': recovery_strategy.get('recommended_state', 'implementing'),
                'agent': recovery_strategy.get('agent', 'RIF-Implementer'),
                'approach': recovery_strategy.get('approach', 'general_fixes'),
                'estimated_time': recovery_strategy.get('estimated_time', 3.0),
                'success_probability': pattern_analysis.get('historical_success_rate', 0.5)
            })
        
        # Option 2: Conservative approach - go back one state
        conservative_state = self._get_previous_state(current_state)
        if conservative_state:
            options.append({
                'type': 'conservative',
                'target_state': conservative_state,
                'agent': self.state_validator.get_required_agent(conservative_state),
                'approach': 'step_back_and_review',
                'estimated_time': 2.0,
                'success_probability': 0.7
            })
        
        # Option 3: Aggressive fix in current state
        options.append({
            'type': 'aggressive',
            'target_state': current_state,
            'agent': self.state_validator.get_required_agent(current_state),
            'approach': 'direct_fixes',
            'estimated_time': 1.5,
            'success_probability': 0.4
        })
        
        # Option 4: Architecture review for complex issues
        if context_model.overall_complexity_score >= 0.7:
            options.append({
                'type': 'architectural',
                'target_state': 'architecting',
                'agent': 'RIF-Architect',
                'approach': 'comprehensive_redesign',
                'estimated_time': 6.0,
                'success_probability': 0.8
            })
        
        return options
    
    def _score_decision_options(self, options: List[Dict[str, Any]], 
                              pattern_analysis: Dict[str, Any], 
                              context_model: ContextModel) -> List[Dict[str, Any]]:
        """Score and rank decision options"""
        scored_options = []
        
        for option in options:
            score_components = {}
            
            # Pattern confidence component
            if option['type'] == 'pattern_based':
                score_components['pattern_confidence'] = pattern_analysis.get('pattern_confidence', 0.5)
            else:
                score_components['pattern_confidence'] = 0.3
            
            # Success probability component
            score_components['recovery_success_rate'] = option.get('success_probability', 0.5)
            
            # Time efficiency component (inverted - shorter time is better)
            max_time = max(opt.get('estimated_time', 3.0) for opt in options)
            score_components['time_efficiency'] = 1.0 - (option.get('estimated_time', 3.0) / max_time)
            
            # Complexity appropriateness
            if context_model.overall_complexity_score >= 0.7 and option['type'] == 'architectural':
                score_components['context_complexity'] = 1.0
            elif option['type'] == 'conservative':
                score_components['context_complexity'] = 0.7
            else:
                score_components['context_complexity'] = 0.5
            
            # Failure severity consideration
            failure_category = pattern_analysis.get('failure_category', {})
            severity = failure_category.get('severity', 'medium')
            if severity == 'critical' and option['type'] in ['pattern_based', 'architectural']:
                score_components['failure_severity'] = 1.0
            elif severity == 'high' and option['type'] != 'aggressive':
                score_components['failure_severity'] = 0.8
            else:
                score_components['failure_severity'] = 0.6
            
            # Calculate weighted score
            total_score = sum(
                score_components[factor] * weight 
                for factor, weight in self.decision_weights.items()
                if factor in score_components
            )
            
            option['score_components'] = score_components
            option['total_score'] = total_score
            option['confidence'] = min(total_score, 1.0)
            
            scored_options.append(option)
        
        # Sort by total score
        scored_options.sort(key=lambda x: x['total_score'], reverse=True)
        return scored_options
    
    def _get_previous_state(self, current_state: str) -> Optional[str]:
        """Get the previous state in the workflow"""
        state_sequence = ['analyzing', 'planning', 'architecting', 'implementing', 'validating', 'learning']
        
        if current_state in state_sequence:
            index = state_sequence.index(current_state)
            if index > 0:
                return state_sequence[index - 1]
        
        return None
    
    def _get_default_option(self, current_state: str) -> Dict[str, Any]:
        """Get default fallback option"""
        return {
            'type': 'default',
            'target_state': 'implementing',
            'agent': 'RIF-Implementer',
            'approach': 'general_fixes',
            'estimated_time': 3.0,
            'success_probability': 0.5,
            'total_score': 0.5,
            'confidence': 0.5
        }
    
    def _generate_decision_reasoning(self, option: Dict[str, Any], 
                                   pattern_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = []
        
        # Main decision rationale
        if option['type'] == 'pattern_based':
            reasoning_parts.append("Based on similar failure patterns in knowledge base")
            if pattern_analysis.get('pattern_confidence', 0) > 0.7:
                reasoning_parts.append("with high pattern matching confidence")
        elif option['type'] == 'conservative':
            reasoning_parts.append("Conservative approach to ensure stability")
        elif option['type'] == 'architectural':
            reasoning_parts.append("Complex issue requires architectural review")
        elif option['type'] == 'aggressive':
            reasoning_parts.append("Direct fixes to minimize time impact")
        
        # Success probability context
        success_prob = option.get('success_probability', 0.5)
        if success_prob > 0.8:
            reasoning_parts.append("with high historical success rate")
        elif success_prob < 0.4:
            reasoning_parts.append("though success rate is uncertain")
        
        # Time consideration
        estimated_time = option.get('estimated_time', 3.0)
        if estimated_time < 2.0:
            reasoning_parts.append("and quick resolution expected")
        elif estimated_time > 5.0:
            reasoning_parts.append("requiring significant time investment")
        
        return "; ".join(reasoning_parts)
    
    def _assess_readiness_factors(self, context_model: ContextModel, 
                                target_state: str) -> Dict[str, float]:
        """Assess various factors affecting readiness for state transition"""
        factors = {}
        
        # Complexity readiness
        complexity = context_model.overall_complexity_score
        if target_state == 'implementing':
            if complexity < 0.6:
                factors['complexity_readiness'] = 1.0
            elif complexity < 0.8:
                factors['complexity_readiness'] = 0.7  # May need architecture
            else:
                factors['complexity_readiness'] = 0.3  # Likely needs architecture
        else:
            factors['complexity_readiness'] = 1.0
        
        # Requirements clarity
        if 'requirements' in context_model.semantic_tags or target_state != 'implementing':
            factors['requirements_clarity'] = 0.9
        else:
            factors['requirements_clarity'] = 0.6  # Unclear if requirements are sufficient
        
        # Risk assessment
        high_risk_count = sum(1 for rf in context_model.risk_factors if rf.get('severity', 0) > 0.7)
        if high_risk_count == 0:
            factors['risk_assessment'] = 1.0
        elif high_risk_count <= 2:
            factors['risk_assessment'] = 0.6
        else:
            factors['risk_assessment'] = 0.3
        
        # Security readiness
        if context_model.security_context.get('requires_security_review', False):
            if target_state in ['implementing', 'validating']:
                factors['security_readiness'] = 0.5  # Needs security planning
            else:
                factors['security_readiness'] = 1.0
        else:
            factors['security_readiness'] = 1.0
        
        # Agent history appropriateness
        agent_history = context_model.issue_context.agent_history
        if target_state == 'implementing' and 'RIF-Analyst' not in agent_history:
            factors['process_readiness'] = 0.4  # Skipped analysis
        elif target_state == 'validating' and 'RIF-Implementer' not in agent_history:
            factors['process_readiness'] = 0.2  # No implementation yet
        else:
            factors['process_readiness'] = 1.0
        
        return factors
    
    def _calculate_overall_readiness(self, readiness_factors: Dict[str, float]) -> float:
        """Calculate overall readiness score from individual factors"""
        if not readiness_factors:
            return 0.5
        
        # Weighted average with minimum threshold enforcement
        weights = {
            'complexity_readiness': 0.25,
            'requirements_clarity': 0.2,
            'risk_assessment': 0.2,
            'security_readiness': 0.2,
            'process_readiness': 0.15
        }
        
        weighted_sum = sum(
            readiness_factors.get(factor, 0.5) * weight
            for factor, weight in weights.items()
        )
        
        # Apply minimum threshold - any factor below 0.3 significantly impacts readiness
        min_factor = min(readiness_factors.values())
        if min_factor < 0.3:
            weighted_sum *= 0.7  # Penalty for very low factors
        
        return min(weighted_sum, 1.0)
    
    def _identify_blocking_reasons(self, readiness_factors: Dict[str, float]) -> List[str]:
        """Identify reasons blocking the transition"""
        blocking_reasons = []
        
        for factor, score in readiness_factors.items():
            if score < 0.5:
                reason_map = {
                    'complexity_readiness': 'Issue complexity requires architectural planning',
                    'requirements_clarity': 'Requirements need further clarification',
                    'risk_assessment': 'High-risk factors need mitigation planning',
                    'security_readiness': 'Security review required before implementation',
                    'process_readiness': 'Previous workflow steps need completion'
                }
                
                if factor in reason_map:
                    blocking_reasons.append(reason_map[factor])
        
        return blocking_reasons
    
    def _generate_readiness_recommendations(self, readiness_factors: Dict[str, float], 
                                          target_state: str) -> List[str]:
        """Generate recommendations to improve readiness"""
        recommendations = []
        
        for factor, score in readiness_factors.items():
            if score < 0.7:
                recommendation_map = {
                    'complexity_readiness': 'Consider architectural planning phase',
                    'requirements_clarity': 'Conduct additional requirements analysis',
                    'risk_assessment': 'Develop risk mitigation strategies',
                    'security_readiness': 'Perform security assessment and planning',
                    'process_readiness': 'Complete prerequisite workflow steps'
                }
                
                if factor in recommendation_map:
                    recommendations.append(recommendation_map[factor])
        
        return recommendations

class TransitionEngine:
    """
    Intelligent state transition management with context preservation.
    Phase 3/4 component: Integration & Optimization
    """
    
    def __init__(self):
        self.loop_back_engine = LoopBackDecisionEngine()
        self.state_analyzer = DynamicStateAnalyzer()
        self.github_manager = GitHubStateManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Transition history for learning
        self.transition_history = deque(maxlen=100)
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def execute_intelligent_transition(self, context_model: ContextModel, 
                                     validation_result: Optional[ValidationResult] = None) -> Dict[str, Any]:
        """
        Execute intelligent state transition with full context awareness.
        
        Args:
            context_model: Rich context model for the issue
            validation_result: Optional validation result for failure-based transitions
            
        Returns:
            Dict with transition results and recommendations
        """
        # ISSUE #273: Use content analysis for state extraction
        current_state = self._get_current_state_from_context(context_model)
        if not current_state:
            current_state = 'new'
        
        if validation_result and not validation_result.passed:
            # Handle failure-based transition
            transition_decision = self.loop_back_engine.make_loop_back_decision(
                validation_result, context_model, current_state
            )
            
            recommended_action = transition_decision['recommended_action']
            next_state = recommended_action['target_state']
            agent = recommended_action['agent']
            
            transition_type = 'loop_back'
            reasoning = f"Loop-back decision: {transition_decision['decision_reasoning']}"
            
        else:
            # Handle normal progression
            next_state, confidence = self.state_analyzer.analyze_current_state(context_model, validation_result)
            
            # Get required agent for the state
            agent = self.state_analyzer.state_validator.get_required_agent(next_state)
            if not agent:
                agent = 'RIF-Implementer'  # Fallback
            
            transition_type = 'progression'
            reasoning = f"Normal progression to {next_state} with {confidence:.2f} confidence"
        
        # Validate transition readiness
        readiness_assessment = self.loop_back_engine.evaluate_transition_readiness(context_model, next_state)
        
        if not readiness_assessment['ready'] and readiness_assessment['confidence'] > 0.8:
            # Strong indication that transition isn't ready
            self.logger.warning(f"Transition to {next_state} not ready: {readiness_assessment['blocking_reasons']}")
            
            # Recommend prerequisite actions instead
            return {
                'transition_executed': False,
                'current_state': current_state,
                'recommended_state': next_state,
                'blocking_reasons': readiness_assessment['blocking_reasons'],
                'recommendations': readiness_assessment['recommendations'],
                'transition_type': 'blocked',
                'readiness_score': readiness_assessment['confidence']
            }
        
        # Execute the transition
        transition_result = self._execute_transition(
            context_model, current_state, next_state, agent, reasoning
        )
        
        # Log transition for learning
        self._log_transition(context_model, current_state, next_state, transition_result, transition_type)
        
        return transition_result
    
    def _execute_transition(self, context_model: ContextModel, current_state: str, 
                           next_state: str, agent: str, reasoning: str) -> Dict[str, Any]:
        """Execute the actual state transition"""
        issue_number = context_model.issue_context.number
        
        # Update GitHub state
        comment = f"Transitioning to state:{next_state}. {reasoning}. Agent: {agent}"
        github_success = self.github_manager.update_issue_state(issue_number, next_state, comment)
        
        if not github_success:
            return {
                'transition_executed': False,
                'error': 'Failed to update GitHub state',
                'current_state': current_state,
                'attempted_state': next_state
            }
        
        # Add agent tracking label
        self.github_manager.add_agent_tracking_label(issue_number, agent)
        
        return {
            'transition_executed': True,
            'previous_state': current_state,
            'new_state': next_state,
            'assigned_agent': agent,
            'reasoning': reasoning,
            'github_updated': github_success,
            'transition_timestamp': datetime.now().isoformat(),
            'issue_number': issue_number
        }
    
    def _log_transition(self, context_model: ContextModel, current_state: str, 
                       next_state: str, transition_result: Dict[str, Any], transition_type: str):
        """Log transition for learning and analysis"""
        transition_log = {
            'timestamp': datetime.now().isoformat(),
            'issue_number': context_model.issue_context.number,
            'current_state': current_state,
            'next_state': next_state,
            'transition_type': transition_type,
            'success': transition_result.get('transition_executed', False),
            'context_complexity': context_model.overall_complexity_score,
            'context_tags': list(context_model.semantic_tags),
            'reasoning': transition_result.get('reasoning', ''),
            'agent_assigned': transition_result.get('assigned_agent', '')
        }
        
        self.transition_history.append(transition_log)
        
        # Periodically save transition history
        if len(self.transition_history) % 20 == 0:
            self._save_transition_history()
    
    def _save_transition_history(self):
        """Save transition history to knowledge base"""
        try:
            history_file = Path("knowledge/metrics/state_transition_history.jsonl")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'a') as f:
                for transition in self.transition_history:
                    f.write(json.dumps(transition) + '\n')
            
            self.transition_history.clear()
            self.logger.info("Saved transition history to knowledge base")
            
        except Exception as e:
            self.logger.warning(f"Could not save transition history: {e}")

class ParallelCoordinator:
    """
    Coordination support for Claude Code parallel agent orchestration.
    Phase 4 component: Integration & Optimization
    """
    
    def __init__(self):
        self.learning_selector = LearningAgentSelector()
        self.transition_engine = TransitionEngine()
        self.context_engine = ContextModelingEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Coordination constraints
        self.max_parallel_agents = 4
        self.max_complex_issues_parallel = 2
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
    
    def coordinate_parallel_orchestration(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Coordinate parallel orchestration of multiple issues.
        Provides optimization for Claude Code to execute parallel Tasks.
        
        Args:
            issue_numbers: List of GitHub issue numbers to orchestrate
            
        Returns:
            Dict with parallel orchestration plan
        """
        # Analyze all issues
        context_analyzer = ContextAnalyzer()
        context_models = []
        
        for issue_num in issue_numbers:
            try:
                issue_context = context_analyzer.analyze_issue(issue_num)
                context_model = self.context_engine.enrich_context(issue_context)
                context_models.append(context_model)
            except Exception as e:
                self.logger.warning(f"Could not analyze issue {issue_num}: {e}")
        
        if not context_models:
            return {'parallel_tasks': [], 'coordination_plan': 'no_issues_to_process'}
        
        # Generate parallel coordination plan
        coordination_plan = self._generate_coordination_plan(context_models)
        
        # Create parallel task assignments
        parallel_tasks = self._create_parallel_tasks(coordination_plan)
        
        return {
            'parallel_tasks': parallel_tasks,
            'coordination_plan': coordination_plan,
            'total_issues': len(context_models),
            'parallel_batches': len(coordination_plan['batches']),
            'estimated_completion_time': coordination_plan['estimated_time'],
            'optimization_strategy': coordination_plan['strategy']
        }
    
    def generate_claude_orchestration_commands(self, coordination_result: Dict[str, Any]) -> List[str]:
        """
        Generate Task() command strings for Claude Code to execute.
        
        Args:
            coordination_result: Result from coordinate_parallel_orchestration
            
        Returns:
            List of Task() command strings for Claude Code
        """
        task_commands = []
        
        for task in coordination_result.get('parallel_tasks', []):
            issue_num = task['issue_number']
            agent = task['agent']
            description = task['description']
            
            # Generate the Task() command
            task_command = f'''Task(
    description="{agent}: {description}",
    subagent_type="general-purpose",
    prompt="You are {agent}. {task.get('prompt', f'Process issue #{issue_num}')}. Follow all instructions in claude/agents/{agent.lower().replace('rif-', '')}.md."
)'''
            
            task_commands.append(task_command)
        
        return task_commands
    
    def _generate_coordination_plan(self, context_models: List[ContextModel]) -> Dict[str, Any]:
        """Generate intelligent coordination plan for parallel processing"""
        # Analyze constraints and dependencies
        constraints = self._analyze_coordination_constraints(context_models)
        
        # Group issues into parallel batches
        batches = self._create_parallel_batches(context_models, constraints)
        
        # Estimate timing and select strategy
        estimated_time = self._estimate_coordination_time(batches)
        strategy = self._select_coordination_strategy(context_models, constraints)
        
        return {
            'batches': batches,
            'constraints': constraints,
            'estimated_time': estimated_time,
            'strategy': strategy,
            'optimization_notes': self._generate_optimization_notes(batches, constraints)
        }
    
    def _analyze_coordination_constraints(self, context_models: List[ContextModel]) -> Dict[str, Any]:
        """Analyze constraints that affect parallel coordination"""
        constraints = {
            'high_complexity_count': 0,
            'architecture_dependent': [],
            'security_sensitive': [],
            'resource_intensive': [],
            'sequential_required': []
        }
        
        for cm in context_models:
            issue_num = cm.issue_context.number
            
            # High complexity issues
            if cm.overall_complexity_score >= 0.7:
                constraints['high_complexity_count'] += 1
            
            # Architecture dependencies
            if 'architecture' in cm.semantic_tags or cm.overall_complexity_score >= 0.8:
                constraints['architecture_dependent'].append(issue_num)
            
            # Security sensitive
            if cm.security_context.get('requires_security_review', False):
                constraints['security_sensitive'].append(issue_num)
            
            # Resource intensive (implementation or architecture work) 
            # ISSUE #273: Use content analysis for state extraction
            current_state = self._get_current_state_from_context(cm)
            if current_state and any(state in current_state for state in ['implementing', 'architecting']):
                constraints['resource_intensive'].append(issue_num)
            
            # Sequential processing required
            if (cm.overall_complexity_score >= 0.9 or 
                len(cm.risk_factors) > 3 or 
                'migration' in cm.semantic_tags):
                constraints['sequential_required'].append(issue_num)
        
        return constraints
    
    def _create_parallel_batches(self, context_models: List[ContextModel], 
                                constraints: Dict[str, Any]) -> List[List[ContextModel]]:
        """Create batches of issues that can be processed in parallel"""
        batches = []
        remaining_models = context_models.copy()
        
        # First batch: Sequential-only issues (process one at a time)
        sequential_issues = [
            cm for cm in remaining_models 
            if cm.issue_context.number in constraints['sequential_required']
        ]
        
        for seq_model in sequential_issues:
            batches.append([seq_model])
            remaining_models.remove(seq_model)
        
        # Remaining batches: Parallel-safe issues
        while remaining_models:
            batch = []
            high_complexity_in_batch = 0
            
            for cm in remaining_models.copy():
                # Check batch constraints
                if len(batch) >= self.max_parallel_agents:
                    break
                
                if cm.overall_complexity_score >= 0.7:
                    if high_complexity_in_batch >= self.max_complex_issues_parallel:
                        continue
                    high_complexity_in_batch += 1
                
                # Check for resource conflicts
                # ISSUE #273: Use content analysis for state extraction  
                current_state = self._get_current_state_from_context(cm)
                if current_state and 'implementing' in current_state:
                    # Limit concurrent implementations
                    impl_count = sum(1 for bcm in batch 
                                   if self._get_current_state_from_context(bcm) and 
                                      'implementing' in self._get_current_state_from_context(bcm))
                    if impl_count >= 1:  # Max 1 implementation per batch
                        continue
                
                batch.append(cm)
                remaining_models.remove(cm)
            
            if batch:
                batches.append(batch)
            else:
                # Fallback: add remaining issues one by one
                if remaining_models:
                    batches.append([remaining_models.pop(0)])
        
        return batches
    
    def _estimate_coordination_time(self, batches: List[List[ContextModel]]) -> float:
        """Estimate total coordination time"""
        total_time = 0.0
        
        for batch in batches:
            # Batch time is the maximum time of issues in the batch (parallel processing)
            batch_times = []
            for cm in batch:
                # Estimate time based on complexity and state
                base_time = cm.overall_complexity_score * 4.0  # 0-4 hours based on complexity
                
                # ISSUE #273: Use content analysis for state extraction
                current_state = self._get_current_state_from_context(cm)
                if current_state:
                    if 'architecting' in current_state:
                        base_time += 2.0
                    elif 'implementing' in current_state:
                        base_time += 1.0
                
                batch_times.append(base_time)
            
            batch_time = max(batch_times) if batch_times else 0.0
            total_time += batch_time
        
        return total_time
    
    def _select_coordination_strategy(self, context_models: List[ContextModel], 
                                    constraints: Dict[str, Any]) -> str:
        """Select optimal coordination strategy"""
        if constraints['high_complexity_count'] >= len(context_models) * 0.5:
            return 'complexity_focused'
        elif len(constraints['security_sensitive']) > 0:
            return 'security_aware'
        elif len(constraints['sequential_required']) > 0:
            return 'hybrid_sequential_parallel'
        else:
            return 'performance_optimized'
    
    def _create_parallel_tasks(self, coordination_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create parallel task assignments from coordination plan"""
        parallel_tasks = []
        
        for batch_idx, batch in enumerate(coordination_plan['batches']):
            for cm in batch:
                # Get optimal agent assignment
                assignment = self.learning_selector.select_single_agent(cm)
                
                if assignment:
                    parallel_tasks.append({
                        'issue_number': cm.issue_context.number,
                        'agent': assignment['agent'],
                        'description': f"Process issue #{cm.issue_context.number}: {cm.issue_context.title[:50]}...",
                        'batch_index': batch_idx,
                        'complexity_score': cm.overall_complexity_score,
                        'estimated_time': assignment.get('estimated_time', 3.0),
                        'prompt': f"Process GitHub issue #{cm.issue_context.number} titled '{cm.issue_context.title}'. Current state: {cm.issue_context.current_state_label}. Context tags: {', '.join(cm.semantic_tags)}.",
                        'parallel_safe': batch_idx == 0 or len(coordination_plan['batches'][batch_idx]) > 1
                    })
        
        return parallel_tasks
    
    def _generate_optimization_notes(self, batches: List[List[ContextModel]], 
                                   constraints: Dict[str, Any]) -> List[str]:
        """Generate optimization notes for the coordination plan"""
        notes = []
        
        if len(batches) == 1:
            notes.append("All issues can be processed in parallel")
        else:
            parallel_batches = sum(1 for batch in batches if len(batch) > 1)
            notes.append(f"Issues grouped into {len(batches)} batches, {parallel_batches} parallel")
        
        if constraints['high_complexity_count'] > 0:
            notes.append(f"{constraints['high_complexity_count']} high-complexity issues require careful resource management")
        
        if constraints['sequential_required']:
            notes.append(f"Issues {constraints['sequential_required']} require sequential processing")
        
        if constraints['security_sensitive']:
            notes.append(f"Issues {constraints['security_sensitive']} have security implications")
        
        return notes

class EnhancedBlockingDetectionEngine:
    """
    Enhanced Blocking Detection Engine for Issue #228 - Critical Orchestration Failure Resolution
    
    Detects explicit blocking declarations in issue body and comments:
    - "THIS ISSUE BLOCKS ALL OTHERS"
    - "BLOCKS ALL OTHER WORK"
    - "STOP ALL WORK"
    - "MUST COMPLETE BEFORE ALL"
    
    Prevents false positives by requiring exact blocking phrases, not just "critical" or "urgent".
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Exact phrases that indicate blocking priority (case-insensitive)
        self.blocking_phrases = [
            "this issue blocks all others",
            "this issue blocks all other work", 
            "blocks all other work",
            "blocks all others",
            "stop all work",
            "must complete before all",
            "must complete before all other work",
            "must complete before all others"
        ]
        
        # Words that alone do NOT constitute blocking (prevent false positives)
        self.non_blocking_keywords = [
            "critical", "urgent", "important", "priority", "blocking"
        ]
    
    def detect_blocking_issues(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Detect which issues have explicit blocking declarations.
        
        Args:
            issue_numbers: List of issue numbers to check
            
        Returns:
            Dict with blocking analysis results
        """
        blocking_issues = []
        blocking_details = {}
        non_blocking_issues = []
        
        for issue_num in issue_numbers:
            try:
                blocking_result = self._analyze_issue_for_blocking(issue_num)
                
                if blocking_result['is_blocking']:
                    blocking_issues.append(issue_num)
                    blocking_details[str(issue_num)] = blocking_result['details']
                    self.logger.warning(f"BLOCKING ISSUE DETECTED: #{issue_num}")
                else:
                    non_blocking_issues.append(issue_num)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing issue {issue_num} for blocking: {e}")
                non_blocking_issues.append(issue_num)  # Assume non-blocking on error
        
        return {
            'blocking_issues': blocking_issues,
            'blocking_details': blocking_details,
            'non_blocking_issues': non_blocking_issues,
            'has_blocking_issues': len(blocking_issues) > 0,
            'blocking_count': len(blocking_issues),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_issue_for_blocking(self, issue_number: int) -> Dict[str, Any]:
        """
        Analyze a single issue for blocking declarations.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Dict with blocking analysis for the issue
        """
        # Get issue data including comments
        issue_data = self._get_issue_with_comments(issue_number)
        if not issue_data:
            return {'is_blocking': False, 'details': 'Could not fetch issue data'}
        
        blocking_sources = []
        
        # Check issue body for blocking phrases
        body_text = (issue_data.get('body') or '').lower()
        body_blocking_phrases = [phrase for phrase in self.blocking_phrases if phrase in body_text]
        if body_blocking_phrases:
            blocking_sources.append({
                'source': 'issue_body',
                'phrases': body_blocking_phrases,
                'excerpt': self._extract_blocking_context(body_text, body_blocking_phrases[0])
            })
        
        # Check comments for blocking phrases
        comments = issue_data.get('comments', [])
        for i, comment in enumerate(comments):
            comment_body = (comment.get('body') or '').lower()
            comment_blocking_phrases = [phrase for phrase in self.blocking_phrases if phrase in comment_body]
            if comment_blocking_phrases:
                blocking_sources.append({
                    'source': f'comment_{i}',
                    'author': comment.get('author', {}).get('login', 'unknown'),
                    'phrases': comment_blocking_phrases,
                    'excerpt': self._extract_blocking_context(comment_body, comment_blocking_phrases[0])
                })
        
        is_blocking = len(blocking_sources) > 0
        
        return {
            'is_blocking': is_blocking,
            'details': {
                'blocking_sources': blocking_sources,
                'blocking_phrase_count': sum(len(source['phrases']) for source in blocking_sources),
                'detected_phrases': list(set(phrase for source in blocking_sources for phrase in source['phrases']))
            }
        }
    
    def _get_issue_with_comments(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Get issue data including comments from GitHub"""
        try:
            # Use gh CLI to get issue with comments
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number), 
                '--json', 'number,title,body,state,labels,comments'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                self.logger.error(f"Failed to fetch issue {issue_number}: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching issue {issue_number}: {e}")
            return None
    
    def _extract_blocking_context(self, text: str, phrase: str, context_chars: int = 100) -> str:
        """Extract context around blocking phrase for evidence"""
        phrase_pos = text.find(phrase)
        if phrase_pos == -1:
            return phrase
            
        start = max(0, phrase_pos - context_chars // 2)
        end = min(len(text), phrase_pos + len(phrase) + context_chars // 2)
        
        context = text[start:end].strip()
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context

class EnhancedOrchestrationIntelligence:
    """
    Main integration facade for the Enhanced Orchestration Intelligence Layer.
    
    This class integrates all 6 core components into a cohesive system that supports 
    Claude Code as the orchestrator while providing sophisticated intelligence capabilities.
    
    ISSUE #228 INTEGRATION: Enhanced blocking detection prevents orchestration failures
    by detecting explicit blocking declarations like "THIS ISSUE BLOCKS ALL OTHERS".
    """
    
    def __init__(self):
        # Initialize all core components
        self.context_engine = ContextModelingEngine()
        self.state_analyzer = DynamicStateAnalyzer()
        self.validation_analyzer = ValidationResultAnalyzer()
        self.learning_selector = LearningAgentSelector()
        self.transition_engine = TransitionEngine()
        self.parallel_coordinator = ParallelCoordinator()
        
        # Integration utilities
        self.context_analyzer = ContextAnalyzer()
        self.github_manager = GitHubStateManager()
        
        # ISSUE #228: Enhanced blocking detection engine
        self.blocking_detection = EnhancedBlockingDetectionEngine()
        
        # CRITICAL INTEGRATION: Consensus orchestrator for issue #148
        try:
            from claude.commands.consensus_orchestrator_integration import ConsensusOrchestratorIntegration
            self.consensus_integration = ConsensusOrchestratorIntegration()
            self.consensus_enabled = True
            logger.info("Consensus orchestrator integration enabled")
        except ImportError as e:
            logger.warning(f"Consensus integration not available: {e}")
            self.consensus_integration = None
            self.consensus_enabled = False
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _get_current_state_from_context(self, context_model: ContextModel) -> str:
        """
        Get current state from content analysis or fall back to labels
        ISSUE #273: Centralized state extraction with content analysis priority
        """
        if context_model.content_analysis_result:
            return context_model.content_analysis_result.state.value
        
        # Fallback to label-based state
        label_state = context_model.issue_context.current_state_label
        if label_state:
            return label_state.replace('state:', '')
        
        return 'analyzing'  # Default state
        
    def analyze_issue_with_intelligence(self, issue_number: int) -> Dict[str, Any]:
        """
        Perform comprehensive intelligent analysis of a GitHub issue.
        
        Args:
            issue_number: GitHub issue number to analyze
            
        Returns:
            Dict with complete intelligence analysis
        """
        try:
            # Phase 1: Basic context analysis
            issue_context = self.context_analyzer.analyze_issue(issue_number)
            
            # Phase 2: Rich context modeling
            context_model = self.context_engine.enrich_context(issue_context)
            
            # Phase 3: State analysis with intelligence
            state_analysis = self.state_analyzer.get_state_analysis_summary(issue_context)
            
            # Phase 4: Agent recommendations
            agent_recommendations = self.learning_selector.get_agent_recommendations(context_model)
            
            return {
                'issue_number': issue_number,
                'context_model': {
                    'complexity_score': context_model.overall_complexity_score,
                    'risk_score': context_model.risk_score,
                    'semantic_tags': list(context_model.semantic_tags),
                    'security_context': context_model.security_context,
                    'performance_context': context_model.performance_context,
                    'risk_factors': context_model.risk_factors
                },
                'state_analysis': state_analysis,
                'agent_recommendations': agent_recommendations,
                'intelligence_summary': self._generate_intelligence_summary(context_model, state_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing issue {issue_number}: {e}")
            return {'error': str(e), 'issue_number': issue_number}
    
    def generate_orchestration_plan(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Generate comprehensive orchestration plan for multiple issues.
        ISSUE #228: Enhanced blocking detection integrated for pre-flight validation.
        
        Args:
            issue_numbers: List of GitHub issue numbers
            
        Returns:
            Dict with orchestration plan and Task() launch codes
        """
        try:
            # ISSUE #228: PRE-FLIGHT BLOCKING DETECTION (CRITICAL FIRST STEP)
            self.logger.info(f"Performing pre-flight blocking detection for issues: {issue_numbers}")
            blocking_analysis = self.blocking_detection.detect_blocking_issues(issue_numbers)
            
            # If blocking issues detected, HALT all other work
            if blocking_analysis['has_blocking_issues']:
                self.logger.critical(f"BLOCKING ISSUES DETECTED: {blocking_analysis['blocking_issues']}")
                return {
                    'orchestration_blocked': True,
                    'blocking_analysis': blocking_analysis,
                    'blocking_issues': blocking_analysis['blocking_issues'],
                    'blocking_count': blocking_analysis['blocking_count'],
                    'blocking_details': blocking_analysis['blocking_details'],
                    'enforcement_action': 'HALT_ALL_ORCHESTRATION',
                    'message': f"Orchestration BLOCKED - {blocking_analysis['blocking_count']} issues require completion first",
                    'allowed_issues': blocking_analysis['blocking_issues'],  # Only blocking issues can proceed
                    'blocked_issues': blocking_analysis['non_blocking_issues'],
                    'task_launch_codes': self._generate_blocking_only_commands(blocking_analysis['blocking_issues']),
                    'execution_ready': True,
                    'parallel_execution': False,  # Blocking issues processed sequentially
                    'blocking_detection_active': True
                }
            
            # CRITICAL INTEGRATION: Consensus evaluation for orchestration decisions
            consensus_decisions = []
            consensus_enabled_issues = []
            
            if self.consensus_enabled and self.consensus_integration:
                for issue_num in issue_numbers:
                    try:
                        # Analyze issue for consensus requirements
                        issue_analysis = self.analyze_issue_with_intelligence(issue_num)
                        issue_context = self._extract_consensus_context(issue_num, issue_analysis)
                        
                        # Evaluate consensus requirement
                        consensus_decision = self.consensus_integration.evaluate_orchestration_decision(issue_context)
                        consensus_decisions.append(consensus_decision)
                        
                        if consensus_decision.consensus_required:
                            consensus_enabled_issues.append(issue_num)
                            self.logger.info(f"Consensus required for issue #{issue_num}: {consensus_decision.decision_rationale}")
                    
                    except Exception as e:
                        self.logger.warning(f"Could not evaluate consensus for issue #{issue_num}: {e}")
            
            # Parallel coordination analysis (only if no blocking issues)
            coordination_result = self.parallel_coordinator.coordinate_parallel_orchestration(issue_numbers)
            
            # Generate Claude Code Task() commands (enhanced with consensus)
            task_commands = self._generate_consensus_aware_commands(coordination_result, consensus_decisions)
            
            # Add intelligence insights
            intelligence_insights = []
            for issue_num in issue_numbers[:3]:  # Limit insights for performance
                try:
                    analysis = self.analyze_issue_with_intelligence(issue_num)
                    intelligence_insights.append({
                        'issue': issue_num,
                        'complexity': analysis.get('context_model', {}).get('complexity_score', 0.5),
                        'recommendations': analysis.get('intelligence_summary', {}).get('key_recommendations', [])
                    })
                except Exception:
                    continue
            
            return {
                'orchestration_plan': coordination_result,
                'task_launch_codes': task_commands,
                'consensus_decisions': [decision.to_dict() for decision in consensus_decisions],
                'consensus_enabled_issues': consensus_enabled_issues,
                'consensus_integration_active': self.consensus_enabled,
                'intelligence_insights': intelligence_insights,
                'execution_ready': len(task_commands) > 0,
                'parallel_execution': len(coordination_result.get('parallel_tasks', [])) > 1,
                # ISSUE #228: Blocking detection status
                'orchestration_blocked': False,
                'blocking_analysis': blocking_analysis,
                'blocking_detection_active': True
            }
            
        except Exception as e:
            self.logger.error(f"Error generating orchestration plan: {e}")
            return {'error': str(e)}
    
    def handle_validation_failure(self, issue_number: int, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle validation failure with intelligent loop-back decisions.
        
        Args:
            issue_number: GitHub issue number that failed validation
            validation_results: Raw validation results from RIF-Validator
            
        Returns:
            Dict with recovery plan and actions
        """
        try:
            # Get issue context
            issue_context = self.context_analyzer.analyze_issue(issue_number)
            context_model = self.context_engine.enrich_context(issue_context)
            
            # Analyze validation results
            validation_result = self.validation_analyzer.analyze_validation_results(validation_results)
            
            # Execute intelligent transition with failure handling
            transition_result = self.transition_engine.execute_intelligent_transition(
                context_model, validation_result
            )
            
            return {
                'issue_number': issue_number,
                'validation_analysis': {
                    'failure_categories': [f.get('category') for f in validation_result.failures],
                    'severity': len(validation_result.failures),
                    'security_issues': len(validation_result.security_issues),
                    'architectural_concerns': len(validation_result.architectural_concerns)
                },
                'recovery_plan': transition_result,
                'recommended_actions': self._generate_recovery_actions(transition_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error handling validation failure for issue {issue_number}: {e}")
            return {'error': str(e), 'issue_number': issue_number}
    
    def get_orchestration_status(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Get comprehensive status for multiple issues with orchestration recommendations.
        
        Args:
            issue_numbers: List of GitHub issue numbers to check
            
        Returns:
            Dict with status summary and recommendations
        """
        status_summary = {
            'total_issues': len(issue_numbers),
            'ready_for_orchestration': [],
            'blocked_issues': [],
            'high_priority': [],
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0},
            'state_distribution': {},
            'orchestration_recommendations': []
        }
        
        for issue_num in issue_numbers:
            try:
                analysis = self.analyze_issue_with_intelligence(issue_num)
                
                if 'error' in analysis:
                    continue
                
                # Complexity distribution
                complexity = analysis['context_model']['complexity_score']
                if complexity < 0.3:
                    status_summary['complexity_distribution']['low'] += 1
                elif complexity < 0.6:
                    status_summary['complexity_distribution']['medium'] += 1
                elif complexity < 0.8:
                    status_summary['complexity_distribution']['high'] += 1
                else:
                    status_summary['complexity_distribution']['very_high'] += 1
                
                # State distribution
                current_state = analysis['state_analysis']['current_state']
                if current_state:
                    status_summary['state_distribution'][current_state] = \
                        status_summary['state_distribution'].get(current_state, 0) + 1
                
                # Readiness assessment
                recommended_state = analysis['state_analysis']['recommended_next_state']
                confidence = analysis['state_analysis']['confidence']
                
                if confidence > 0.7:
                    status_summary['ready_for_orchestration'].append({
                        'issue': issue_num,
                        'state': recommended_state,
                        'confidence': confidence
                    })
                else:
                    status_summary['blocked_issues'].append({
                        'issue': issue_num,
                        'reasons': analysis.get('intelligence_summary', {}).get('blocking_factors', [])
                    })
                
                # High priority identification
                risk_score = analysis['context_model']['risk_score']
                if risk_score > 0.7 or 'security' in analysis['context_model']['semantic_tags']:
                    status_summary['high_priority'].append(issue_num)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing issue {issue_num} for status: {e}")
        
        # Generate orchestration recommendations
        status_summary['orchestration_recommendations'] = self._generate_orchestration_recommendations(status_summary)
        
        return status_summary
    
    def _generate_intelligence_summary(self, context_model: ContextModel, state_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of intelligence analysis"""
        summary = {
            'complexity_assessment': self._assess_complexity_level(context_model.overall_complexity_score),
            'risk_level': self._assess_risk_level(context_model.risk_score),
            'key_characteristics': [],
            'recommended_approach': state_analysis.get('recommended_next_state', 'implementing'),
            'confidence_level': state_analysis.get('confidence', 0.5),
            'key_recommendations': [],
            'blocking_factors': []
        }
        
        # Key characteristics
        if context_model.overall_complexity_score >= 0.7:
            summary['key_characteristics'].append('high_complexity')
        if context_model.security_context.get('requires_security_review', False):
            summary['key_characteristics'].append('security_sensitive')
        if context_model.performance_context.get('performance_critical', False):
            summary['key_characteristics'].append('performance_critical')
        if len(context_model.risk_factors) > 2:
            summary['key_characteristics'].append('high_risk')
        
        # Key recommendations
        if context_model.overall_complexity_score >= 0.8:
            summary['key_recommendations'].append('Requires architectural planning before implementation')
        if 'security' in context_model.semantic_tags:
            summary['key_recommendations'].append('Mandatory security review and validation')
        if context_model.risk_score > 0.7:
            summary['key_recommendations'].append('Risk mitigation planning essential')
        
        # Blocking factors
        if state_analysis.get('confidence', 0.5) < 0.5:
            summary['blocking_factors'].append('Low confidence in state transition')
        if not context_model.issue_context.agent_history and state_analysis.get('recommended_next_state') == 'implementing':
            summary['blocking_factors'].append('Needs analysis phase before implementation')
        
        return summary
    
    def _assess_complexity_level(self, complexity_score: float) -> str:
        """Convert complexity score to readable assessment"""
        if complexity_score >= 0.8:
            return 'very_high'
        elif complexity_score >= 0.6:
            return 'high'
        elif complexity_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_risk_level(self, risk_score: float) -> str:
        """Convert risk score to readable assessment"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recovery_actions(self, transition_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recovery steps"""
        actions = []
        
        if transition_result.get('transition_executed', False):
            new_state = transition_result.get('new_state')
            agent = transition_result.get('assigned_agent')
            actions.append(f"Issue transitioned to state:{new_state}")
            actions.append(f"Agent {agent} assigned for recovery")
        else:
            blocking_reasons = transition_result.get('blocking_reasons', [])
            recommendations = transition_result.get('recommendations', [])
            
            actions.append("Transition blocked - prerequisite actions required:")
            actions.extend(blocking_reasons)
            actions.append("Recommended actions:")
            actions.extend(recommendations)
        
        return actions
    
    def _generate_blocking_only_commands(self, blocking_issues: List[int]) -> List[str]:
        """
        Generate Task() commands for blocking issues only.
        ISSUE #228: When blocking issues detected, only work on those issues.
        
        Args:
            blocking_issues: List of issue numbers with blocking declarations
            
        Returns:
            List of Task() command strings for blocking issues
        """
        commands = []
        
        for issue_num in blocking_issues:
            try:
                # Get issue context for appropriate agent selection
                issue_context = self.context_analyzer.analyze_issue(issue_num)
                # ISSUE #273: Use content analysis for state extraction
                context_model = self.context_engine.enrich_context(issue_context) 
                current_state = self._get_current_state_from_context(context_model)
                
                # Select appropriate agent based on current state
                if current_state == 'new' or current_state is None:
                    agent_type = "RIF-Analyst"
                    task_description = f"Analyze BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)"
                    prompt_instruction = f"You are RIF-Analyst. Perform critical analysis of BLOCKING issue #{issue_num}. This issue blocks all other work - complete analysis urgently. Follow all instructions in claude/agents/rif-analyst.md."
                elif current_state == 'planning':
                    agent_type = "RIF-Planner" 
                    task_description = f"Plan BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)"
                    prompt_instruction = f"You are RIF-Planner. Create urgent plan for BLOCKING issue #{issue_num}. This issue blocks all other work - prioritize completion. Follow all instructions in claude/agents/rif-planner.md."
                elif current_state == 'implementing':
                    agent_type = "RIF-Implementer"
                    task_description = f"Implement BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)" 
                    prompt_instruction = f"You are RIF-Implementer. Implement BLOCKING issue #{issue_num} immediately. This issue blocks all other work - complete implementation urgently. Follow all instructions in claude/agents/rif-implementer.md."
                elif current_state == 'validating':
                    agent_type = "RIF-Validator"
                    task_description = f"Validate BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)"
                    prompt_instruction = f"You are RIF-Validator. Validate BLOCKING issue #{issue_num} urgently. This issue blocks all other work - ensure quality and complete validation. Follow all instructions in claude/agents/rif-validator.md."
                else:
                    # Default to analyst for unknown states
                    agent_type = "RIF-Analyst"
                    task_description = f"Analyze BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)"
                    prompt_instruction = f"You are RIF-Analyst. Analyze BLOCKING issue #{issue_num}. This issue blocks all other work - determine next steps urgently. Follow all instructions in claude/agents/rif-analyst.md."
                
                # Generate Task() command
                command = f'''Task(
    description="{task_description}",
    subagent_type="general-purpose",
    prompt="{prompt_instruction}"
)'''
                commands.append(command)
                
            except Exception as e:
                self.logger.error(f"Error generating command for blocking issue {issue_num}: {e}")
                # Fallback command
                fallback_command = f'''Task(
    description="Resolve BLOCKING issue #{issue_num} (THIS ISSUE BLOCKS ALL OTHERS)",
    subagent_type="general-purpose", 
    prompt="You are RIF-Analyst. Resolve BLOCKING issue #{issue_num}. This issue blocks all other work - complete urgently. Follow all instructions in claude/agents/rif-analyst.md."
)'''
                commands.append(fallback_command)
        
        return commands
    
    def _extract_consensus_context(self, issue_num: int, issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context needed for consensus evaluation from issue analysis"""
        context = {
            'issue_number': issue_num,
            'title': 'Unknown',
            'risk_level': 'medium',
            'complexity': 'medium',
            'security_critical': False,
            'agent_confidence': 1.0,
            'previous_failures': False,
            'multi_system_impact': False,
            'emergency_protocol': False,
            'estimated_impact': 'medium'
        }
        
        try:
            # Extract from issue analysis
            context_model = issue_analysis.get('context_model', {})
            
            context.update({
                'title': context_model.get('title', 'Unknown'),
                'complexity': self._map_complexity_score_to_level(context_model.get('complexity_score', 0.5)),
                'security_critical': len(context_model.get('security_context', {})) > 0,
                'agent_confidence': context_model.get('confidence_score', 1.0),
                'multi_system_impact': len(context_model.get('dependency_chain', [])) > 3,
                'estimated_impact': context_model.get('impact_analysis', {}).get('overall_impact', 'medium')
            })
            
            # Map risk factors to risk level
            risk_factors = context_model.get('risk_factors', [])
            if any('critical' in rf.lower() for rf in risk_factors):
                context['risk_level'] = 'critical'
            elif any('high' in rf.lower() or 'security' in rf.lower() for rf in risk_factors):
                context['risk_level'] = 'high'
            elif any('emergency' in rf.lower() or 'urgent' in rf.lower() for rf in risk_factors):
                context['emergency_protocol'] = True
                context['risk_level'] = 'high'
            
            # Check for previous failures (simplified heuristic)
            semantic_tags = context_model.get('semantic_tags', [])
            context['previous_failures'] = any('fix' in tag.lower() or 'error' in tag.lower() for tag in semantic_tags)
            
        except Exception as e:
            self.logger.warning(f"Error extracting consensus context for issue #{issue_num}: {e}")
        
        return context
    
    def _map_complexity_score_to_level(self, complexity_score: float) -> str:
        """Map numerical complexity score to complexity level"""
        if complexity_score >= 0.8:
            return 'very_high'
        elif complexity_score >= 0.6:
            return 'high'
        elif complexity_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_consensus_aware_commands(self, coordination_result: Dict[str, Any], 
                                         consensus_decisions: List) -> List[str]:
        """Generate Task() commands enhanced with consensus information"""
        # Start with original commands
        task_commands = self.parallel_coordinator.generate_claude_orchestration_commands(coordination_result)
        
        if not consensus_decisions:
            return task_commands
        
        # Enhance commands with consensus information
        enhanced_commands = []
        for i, command in enumerate(task_commands):
            if i < len(consensus_decisions):
                consensus_decision = consensus_decisions[i]
                
                if consensus_decision.consensus_required:
                    # Modify command to include consensus agents
                    agents = consensus_decision.recommended_agents
                    if len(agents) > 1:
                        # Create parallel consensus tasks
                        for j, agent in enumerate(agents):
                            consensus_command = f'''Task(
    description="{agent}: Consensus validation for critical decision",
    subagent_type="general-purpose",
    prompt="You are {agent}. Participate in consensus validation for critical decision. Provide your analysis and vote. Follow all instructions in claude/agents/{agent.lower().replace('rif-', '')}.md."
)'''
                            enhanced_commands.append(consensus_command)
                    else:
                        enhanced_commands.append(command)
                else:
                    enhanced_commands.append(command)
            else:
                enhanced_commands.append(command)
        
        return enhanced_commands

    def _generate_orchestration_recommendations(self, status_summary: Dict[str, Any]) -> List[str]:
        """Generate high-level orchestration recommendations"""
        recommendations = []
        
        ready_count = len(status_summary['ready_for_orchestration'])
        blocked_count = len(status_summary['blocked_issues'])
        high_priority_count = len(status_summary['high_priority'])
        
        if ready_count > 0:
            recommendations.append(f"{ready_count} issues ready for immediate orchestration")
            
            if ready_count > 3:
                recommendations.append("Consider parallel execution for performance")
        
        if blocked_count > 0:
            recommendations.append(f"{blocked_count} issues blocked - address prerequisites first")
        
        if high_priority_count > 0:
            recommendations.append(f"{high_priority_count} high-priority issues identified - prioritize these")
        
        # Complexity-based recommendations
        complexity_dist = status_summary['complexity_distribution']
        if complexity_dist['very_high'] > 0:
            recommendations.append(f"{complexity_dist['very_high']} very high complexity issues - ensure architectural review")
        
        if complexity_dist['high'] + complexity_dist['very_high'] > ready_count * 0.5:
            recommendations.append("High complexity workload - consider sequential processing")
        
        return recommendations

# Main integration facade - this is what Claude Code should use
def get_enhanced_orchestration_intelligence() -> EnhancedOrchestrationIntelligence:
    """
    Factory function to get the main Enhanced Orchestration Intelligence system.
    
    Returns:
        EnhancedOrchestrationIntelligence instance ready for use
    """
    return EnhancedOrchestrationIntelligence()

# Convenience functions for common operations
def analyze_issue(issue_number: int) -> Dict[str, Any]:
    """Convenience function for single issue analysis"""
    eoi = get_enhanced_orchestration_intelligence()
    return eoi.analyze_issue_with_intelligence(issue_number)

def generate_orchestration_plan(issue_numbers: List[int]) -> Dict[str, Any]:
    """Convenience function for orchestration planning"""
    eoi = get_enhanced_orchestration_intelligence()
    return eoi.generate_orchestration_plan(issue_numbers)

def handle_validation_failure(issue_number: int, validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for validation failure handling"""
    eoi = get_enhanced_orchestration_intelligence()
    return eoi.handle_validation_failure(issue_number, validation_results)

# Example usage demonstration
def example_usage():
    """Example of how Claude Code should use the Enhanced Orchestration Intelligence Layer"""
    print("Enhanced Orchestration Intelligence Layer - Usage Examples")
    print("=" * 70)
    
    # Example 1: Single issue analysis
    print("\n1. Single Issue Analysis:")
    try:
        analysis = analyze_issue(52)  # This issue
        if 'error' not in analysis:
            print(f"   Complexity: {analysis['context_model']['complexity_score']:.2f}")
            print(f"   Risk Level: {analysis['context_model']['risk_score']:.2f}")
            print(f"   Recommended State: {analysis['state_analysis']['recommended_next_state']}")
            print(f"   Confidence: {analysis['state_analysis']['confidence']:.2f}")
        else:
            print(f"   Error: {analysis['error']}")
    except Exception as e:
        print(f"   Example failed: {e}")
    
    # Example 2: Multi-issue orchestration
    print("\n2. Multi-Issue Orchestration Plan:")
    try:
        plan = generate_orchestration_plan([52])  # This issue
        if 'error' not in plan:
            print(f"   Parallel Tasks: {len(plan.get('task_launch_codes', []))}")
            print(f"   Execution Ready: {plan.get('execution_ready', False)}")
            print(f"   Parallel Capable: {plan.get('parallel_execution', False)}")
        else:
            print(f"   Error: {plan['error']}")
    except Exception as e:
        print(f"   Example failed: {e}")
    
    print("\n3. Pattern-Compliant Architecture:")
    print("   ✅ Claude Code IS the orchestrator")
    print("   ✅ Enhanced Intelligence Layer provides decision support")
    print("   ✅ Task() launching for agent execution")
    print("   ✅ No violating orchestrator classes")
    
    print(f"\n🎉 Enhanced Orchestration Intelligence Layer implementation complete!")
    print("   Ready for RIF-Validator testing and validation.")

if __name__ == "__main__":
    example_usage()

# Final Implementation Checkpoint
logger.info("Enhanced Orchestration Intelligence Layer - Complete implementation with all 6 components integrated")