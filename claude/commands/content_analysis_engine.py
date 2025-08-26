#!/usr/bin/env python3
"""
Content Analysis Engine - Replace Label Dependency System

This module implements intelligent content analysis to derive issue state, complexity,
and dependencies from issue text rather than relying on GitHub labels.

CRITICAL: This addresses Issue #273 by removing the label dependency from 
orchestration logic and implementing semantic content analysis.

Architecture:
- ContentAnalysisEngine: Main analysis engine
- StateAnalyzer: Derives workflow state from issue content
- ComplexityAnalyzer: Assesses task complexity from requirements
- DependencyAnalyzer: Identifies blocking and prerequisite relationships
- PerformanceOptimized: Sub-100ms response time required
"""

import json
import re
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set, Pattern
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import statistics
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    """Workflow states derived from content analysis"""
    NEW = "new"
    ANALYZING = "analyzing" 
    PLANNING = "planning"
    ARCHITECTING = "architecting"
    IMPLEMENTING = "implementing"
    VALIDATING = "validating"
    LEARNING = "learning"
    COMPLETE = "complete"
    BLOCKED = "blocked"

class ComplexityLevel(Enum):
    """Complexity levels derived from content analysis"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    VERY_HIGH = "very-high"

@dataclass
class ContentAnalysisResult:
    """Result of content analysis with confidence scoring"""
    state: WorkflowState
    complexity: ComplexityLevel
    dependencies: List[int] = field(default_factory=list)
    blocking_issues: List[int] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_time_ms: float = 0.0
    semantic_indicators: Dict[str, List[str]] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)

class StateAnalyzer:
    """Analyzes issue content to derive workflow state"""
    
    def __init__(self):
        self.state_patterns = self._initialize_state_patterns()
        
    def _initialize_state_patterns(self) -> Dict[WorkflowState, List[Pattern]]:
        """Initialize regex patterns for state detection"""
        return {
            WorkflowState.NEW: [
                re.compile(r'(?i)\b(new|created|initial|needs?\s+analysis|requires?\s+investigation)', re.IGNORECASE),
                re.compile(r'(?i)\b(analyze|investigate|understand|research)', re.IGNORECASE),
                re.compile(r'(?i)\b(what|how|why|when)\b.*\?', re.IGNORECASE)
            ],
            WorkflowState.ANALYZING: [
                re.compile(r'(?i)\b(analyzing|investigation|research|analysis)', re.IGNORECASE),
                re.compile(r'(?i)\b(understanding|exploring|examining)', re.IGNORECASE),
                re.compile(r'(?i)\b(requirements?\s+analysis|feasibility)', re.IGNORECASE)
            ],
            WorkflowState.PLANNING: [
                re.compile(r'(?i)\b(plan|planning|strategy|approach)', re.IGNORECASE),
                re.compile(r'(?i)\b(design|architecture|workflow)', re.IGNORECASE),
                re.compile(r'(?i)\b(steps|phases|milestones)', re.IGNORECASE)
            ],
            WorkflowState.ARCHITECTING: [
                re.compile(r'(?i)\b(architect|architecture|design)', re.IGNORECASE),
                re.compile(r'(?i)\b(structure|framework|system)', re.IGNORECASE),
                re.compile(r'(?i)\b(components?|modules?|interfaces?)', re.IGNORECASE)
            ],
            WorkflowState.IMPLEMENTING: [
                re.compile(r'(?i)\b(implement|implementation|code|coding)', re.IGNORECASE),
                re.compile(r'(?i)\b(develop|build|create|write)', re.IGNORECASE),
                re.compile(r'(?i)\b(feature|function|method|class)', re.IGNORECASE),
                re.compile(r'(?i)\b(ready\s+to\s+implement|start\s+implementation)', re.IGNORECASE)
            ],
            WorkflowState.VALIDATING: [
                re.compile(r'(?i)\b(test|testing|validate|validation)', re.IGNORECASE),
                re.compile(r'(?i)\b(verify|check|quality|review)', re.IGNORECASE),
                re.compile(r'(?i)\b(complete.*test|ready.*test|needs?\s+testing)', re.IGNORECASE)
            ],
            WorkflowState.LEARNING: [
                re.compile(r'(?i)\b(learn|learning|knowledge|document)', re.IGNORECASE),
                re.compile(r'(?i)\b(extract.*pattern|update.*knowledge)', re.IGNORECASE),
                re.compile(r'(?i)\b(retrospective|lessons|insights)', re.IGNORECASE)
            ],
            WorkflowState.COMPLETE: [
                re.compile(r'(?i)\b(complete|completed|done|finished)', re.IGNORECASE),
                re.compile(r'(?i)\b(resolved|fixed|implemented)', re.IGNORECASE),
                re.compile(r'(?i)\b(success|working|deployed)', re.IGNORECASE)
            ],
            WorkflowState.BLOCKED: [
                re.compile(r'(?i)\b(block|blocked|blocking|stuck)', re.IGNORECASE),
                re.compile(r'(?i)\b(dependency|depends\s+on|waiting\s+for)', re.IGNORECASE),
                re.compile(r'(?i)\b(cannot\s+proceed|prerequisite)', re.IGNORECASE)
            ]
        }
    
    def analyze_state(self, issue_text: str) -> Tuple[WorkflowState, float]:
        """
        Analyze issue content to derive workflow state with confidence
        
        Args:
            issue_text: Combined title and body text of issue
            
        Returns:
            Tuple of (WorkflowState, confidence_score)
        """
        state_scores = {}
        
        for state, patterns in self.state_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                pattern_matches = pattern.findall(issue_text)
                if pattern_matches:
                    # Weight by frequency and position
                    frequency_score = len(pattern_matches) * 0.1
                    position_score = self._calculate_position_weight(issue_text, pattern)
                    score += frequency_score + position_score
                    matches.extend(pattern_matches)
            
            if matches:
                state_scores[state] = {
                    'score': score,
                    'matches': matches
                }
        
        if not state_scores:
            return WorkflowState.NEW, 0.5  # Default to new with moderate confidence
        
        # Find highest scoring state
        best_state = max(state_scores.keys(), key=lambda s: state_scores[s]['score'])
        max_score = state_scores[best_state]['score']
        
        # Calculate confidence as normalized score
        confidence = min(max_score / 2.0, 1.0)  # Normalize to 0-1 range
        
        return best_state, confidence
    
    def _calculate_position_weight(self, text: str, pattern: Pattern) -> float:
        """Calculate weight based on pattern position in text"""
        match = pattern.search(text)
        if not match:
            return 0.0
        
        position = match.start()
        text_length = len(text)
        
        # Higher weight for patterns appearing earlier
        position_ratio = position / text_length if text_length > 0 else 0
        return 1.0 - (position_ratio * 0.5)

class ComplexityAnalyzer:
    """Analyzes issue content to determine task complexity"""
    
    def __init__(self):
        self.complexity_indicators = self._initialize_complexity_indicators()
        
    def _initialize_complexity_indicators(self) -> Dict[ComplexityLevel, Dict[str, List[Pattern]]]:
        """Initialize complexity indicators"""
        return {
            ComplexityLevel.LOW: {
                'simple_tasks': [
                    re.compile(r'(?i)\b(fix|update|change|modify)\s+\w+', re.IGNORECASE),
                    re.compile(r'(?i)\b(simple|quick|small|minor)', re.IGNORECASE),
                    re.compile(r'(?i)\b(one|single|few)\s+\w+', re.IGNORECASE)
                ],
                'small_scope': [
                    re.compile(r'(?i)\b(file|function|method|variable)', re.IGNORECASE),
                    re.compile(r'(?i)\b(text|message|label|button)', re.IGNORECASE)
                ]
            },
            ComplexityLevel.MEDIUM: {
                'moderate_tasks': [
                    re.compile(r'(?i)\b(implement|add|create|develop)', re.IGNORECASE),
                    re.compile(r'(?i)\b(feature|component|module)', re.IGNORECASE),
                    re.compile(r'(?i)\b(integrate|connect|link)', re.IGNORECASE)
                ],
                'multi_step': [
                    re.compile(r'(?i)\b(steps?|phases?|stages?)', re.IGNORECASE),
                    re.compile(r'(?i)\b(multiple|several|various)', re.IGNORECASE)
                ]
            },
            ComplexityLevel.HIGH: {
                'complex_tasks': [
                    re.compile(r'(?i)\b(architecture|framework|system)', re.IGNORECASE),
                    re.compile(r'(?i)\b(refactor|redesign|migrate)', re.IGNORECASE),
                    re.compile(r'(?i)\b(optimization|performance|security)', re.IGNORECASE)
                ],
                'large_scope': [
                    re.compile(r'(?i)\b(application|platform|infrastructure)', re.IGNORECASE),
                    re.compile(r'(?i)\b(database|api|service)', re.IGNORECASE),
                    re.compile(r'(?i)\b(workflow|pipeline|process)', re.IGNORECASE)
                ]
            },
            ComplexityLevel.VERY_HIGH: {
                'very_complex': [
                    re.compile(r'(?i)\b(distributed|microservices|scalability)', re.IGNORECASE),
                    re.compile(r'(?i)\b(machine\s+learning|ai|algorithm)', re.IGNORECASE),
                    re.compile(r'(?i)\b(real[- ]?time|concurrent|parallel)', re.IGNORECASE)
                ],
                'enterprise_scope': [
                    re.compile(r'(?i)\b(enterprise|production|deployment)', re.IGNORECASE),
                    re.compile(r'(?i)\b(compliance|audit|governance)', re.IGNORECASE),
                    re.compile(r'(?i)\b(monitoring|observability|telemetry)', re.IGNORECASE)
                ]
            }
        }
    
    def analyze_complexity(self, issue_text: str) -> Tuple[ComplexityLevel, float]:
        """
        Analyze issue content to determine complexity level
        
        Args:
            issue_text: Combined title and body text of issue
            
        Returns:
            Tuple of (ComplexityLevel, confidence_score)
        """
        complexity_scores = {}
        
        for level, categories in self.complexity_indicators.items():
            total_score = 0.0
            total_matches = []
            
            for category, patterns in categories.items():
                for pattern in patterns:
                    matches = pattern.findall(issue_text)
                    if matches:
                        # Score based on match frequency and pattern specificity
                        score = len(matches) * self._get_pattern_weight(category)
                        total_score += score
                        total_matches.extend(matches)
            
            if total_matches:
                complexity_scores[level] = {
                    'score': total_score,
                    'matches': total_matches
                }
        
        if not complexity_scores:
            return ComplexityLevel.MEDIUM, 0.5  # Default to medium
        
        # Find highest scoring complexity level
        best_complexity = max(complexity_scores.keys(), 
                            key=lambda c: complexity_scores[c]['score'])
        max_score = complexity_scores[best_complexity]['score']
        
        # Calculate confidence
        confidence = min(max_score / 5.0, 1.0)  # Normalize to 0-1 range
        
        return best_complexity, confidence
    
    def _get_pattern_weight(self, category: str) -> float:
        """Get weight for pattern category"""
        weights = {
            'simple_tasks': 0.5,
            'small_scope': 0.3,
            'moderate_tasks': 1.0,
            'multi_step': 0.7,
            'complex_tasks': 1.5,
            'large_scope': 1.2,
            'very_complex': 2.0,
            'enterprise_scope': 1.8
        }
        return weights.get(category, 1.0)

class DependencyAnalyzer:
    """Analyzes issue content to identify dependencies"""
    
    def __init__(self):
        self.dependency_patterns = self._initialize_dependency_patterns()
        
    def _initialize_dependency_patterns(self) -> Dict[str, List[Pattern]]:
        """Initialize dependency detection patterns"""
        return {
            'blocking': [
                re.compile(r'(?i)\b(blocks?|blocking)\s+(?:issue\s*)?#?(\d+)', re.IGNORECASE),
                re.compile(r'(?i)(?:this\s+)?(?:issue\s+)?blocks?\s+(?:all\s+)?others?', re.IGNORECASE),
                re.compile(r'(?i)(?:must\s+complete\s+before|prerequisite)', re.IGNORECASE)
            ],
            'depends_on': [
                re.compile(r'(?i)\b(?:depends?\s+on|requires?)\s+(?:issue\s*)?#?(\d+)', re.IGNORECASE),
                re.compile(r'(?i)\b(?:blocked\s+by|waiting\s+for)\s+(?:issue\s*)?#?(\d+)', re.IGNORECASE),
                re.compile(r'(?i)\b(?:needs?\s+|requires?\s+)(?:issue\s*)?#?(\d+)', re.IGNORECASE)
            ],
            'issue_refs': [
                re.compile(r'(?i)(?:see\s+|ref\s+|related\s+to\s+)?(?:issue\s*)?#(\d+)', re.IGNORECASE),
                re.compile(r'(?i)(?:fixes?|resolves?|closes?)\s+(?:issue\s*)?#(\d+)', re.IGNORECASE)
            ]
        }
    
    def analyze_dependencies(self, issue_text: str) -> Tuple[List[int], List[int]]:
        """
        Analyze issue content to identify dependencies and blocking relationships
        
        Args:
            issue_text: Combined title and body text of issue
            
        Returns:
            Tuple of (depends_on_issues, blocks_issues)
        """
        depends_on = set()
        blocks = set()
        
        # Extract blocking relationships
        for pattern in self.dependency_patterns['blocking']:
            matches = pattern.findall(issue_text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # Pattern with multiple capture groups - use first non-empty
                        for group in match:
                            if group and group.isdigit():
                                blocks.add(int(group))
                                break
                    elif isinstance(match, str) and match.isdigit():
                        # Pattern with single capture group
                        blocks.add(int(match))
        
        # Extract dependency relationships
        for pattern in self.dependency_patterns['depends_on']:
            matches = pattern.findall(issue_text)
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with multiple capture groups - use first non-empty
                    for group in match:
                        if group and group.isdigit():
                            depends_on.add(int(group))
                            break
                elif isinstance(match, str) and match.isdigit():
                    # Pattern with single capture group
                    depends_on.add(int(match))
        
        return list(depends_on), list(blocks)

class ContentAnalysisEngine:
    """
    Main Content Analysis Engine to replace label dependency system
    
    This engine analyzes issue content to derive:
    - Workflow state (replaces state: labels)
    - Task complexity (replaces complexity: labels) 
    - Dependencies and blocking relationships
    - Risk factors and semantic indicators
    
    Performance target: Sub-100ms analysis time
    Accuracy target: 90%+ state determination accuracy
    """
    
    def __init__(self):
        self.state_analyzer = StateAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer() 
        self.dependency_analyzer = DependencyAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        
    def analyze_issue_content(self, issue_title: str, issue_body: str = "") -> ContentAnalysisResult:
        """
        Analyze issue content to derive state, complexity, and dependencies
        
        This method replaces the label dependency at line 668 in enhanced_orchestration_intelligence.py
        
        Args:
            issue_title: GitHub issue title
            issue_body: GitHub issue body text
            
        Returns:
            ContentAnalysisResult with derived state, complexity, dependencies
        """
        start_time = time.time()
        
        # Combine title and body for analysis
        combined_text = f"{issue_title}\n\n{issue_body}"
        
        # Perform multi-dimensional analysis
        state, state_confidence = self.state_analyzer.analyze_state(combined_text)
        complexity, complexity_confidence = self.complexity_analyzer.analyze_complexity(combined_text)
        depends_on, blocks = self.dependency_analyzer.analyze_dependencies(combined_text)
        
        # Calculate overall confidence score
        overall_confidence = (state_confidence + complexity_confidence) / 2.0
        
        # Extract semantic indicators
        semantic_indicators = self._extract_semantic_indicators(combined_text)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(combined_text)
        
        # Calculate analysis time
        analysis_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Update performance tracking
        self.analysis_count += 1
        self.total_analysis_time += analysis_time
        
        # Log performance warning if over 100ms
        if analysis_time > 100:
            self.logger.warning(f"Analysis time {analysis_time:.2f}ms exceeds 100ms target")
        
        return ContentAnalysisResult(
            state=state,
            complexity=complexity,
            dependencies=depends_on,
            blocking_issues=blocks,
            confidence_score=overall_confidence,
            analysis_time_ms=analysis_time,
            semantic_indicators=semantic_indicators,
            risk_factors=risk_factors
        )
    
    def _extract_semantic_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract semantic indicators from issue text"""
        indicators = {
            'technologies': [],
            'actions': [],
            'domains': [],
            'urgency': []
        }
        
        # Technology indicators
        tech_patterns = [
            re.compile(r'(?i)\b(python|javascript|java|go|rust|node\.?js|react|vue|angular)', re.IGNORECASE),
            re.compile(r'(?i)\b(api|rest|graphql|database|sql|nosql|redis|mongodb)', re.IGNORECASE),
            re.compile(r'(?i)\b(docker|kubernetes|aws|gcp|azure|terraform)', re.IGNORECASE)
        ]
        
        for pattern in tech_patterns:
            matches = pattern.findall(text)
            indicators['technologies'].extend([match.lower() for match in matches])
        
        # Action indicators  
        action_patterns = [
            re.compile(r'(?i)\b(implement|create|build|develop|design|architect)', re.IGNORECASE),
            re.compile(r'(?i)\b(fix|resolve|debug|troubleshoot|optimize)', re.IGNORECASE),
            re.compile(r'(?i)\b(test|validate|verify|review|audit)', re.IGNORECASE)
        ]
        
        for pattern in action_patterns:
            matches = pattern.findall(text)
            indicators['actions'].extend([match.lower() for match in matches])
        
        # Domain indicators
        domain_patterns = [
            re.compile(r'(?i)\b(authentication|authorization|security|encryption)', re.IGNORECASE),
            re.compile(r'(?i)\b(performance|optimization|caching|scalability)', re.IGNORECASE),
            re.compile(r'(?i)\b(ui|ux|frontend|backend|fullstack)', re.IGNORECASE)
        ]
        
        for pattern in domain_patterns:
            matches = pattern.findall(text)
            indicators['domains'].extend([match.lower() for match in matches])
        
        # Urgency indicators
        urgency_patterns = [
            re.compile(r'(?i)\b(urgent|critical|emergency|asap|immediately)', re.IGNORECASE),
            re.compile(r'(?i)\b(high\s+priority|blocking|blocker)', re.IGNORECASE),
            re.compile(r'(?i)\b(deadline|due\s+date|time\s+sensitive)', re.IGNORECASE)
        ]
        
        for pattern in urgency_patterns:
            matches = pattern.findall(text)
            indicators['urgency'].extend([match.lower() for match in matches])
        
        return indicators
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify risk factors from issue content"""
        risk_factors = []
        
        risk_patterns = {
            'Breaking Change': re.compile(r'(?i)\b(breaking|break|backward\s+compatible)', re.IGNORECASE),
            'Security Risk': re.compile(r'(?i)\b(security|vulnerability|exploit|breach)', re.IGNORECASE),
            'Performance Risk': re.compile(r'(?i)\b(performance|slow|timeout|memory|cpu)', re.IGNORECASE),
            'Data Risk': re.compile(r'(?i)\b(database|data\s+loss|migration|backup)', re.IGNORECASE),
            'Integration Risk': re.compile(r'(?i)\b(integration|third[- ]?party|external|api)', re.IGNORECASE),
            'Deployment Risk': re.compile(r'(?i)\b(deploy|production|release|rollback)', re.IGNORECASE)
        }
        
        for risk_type, pattern in risk_patterns.items():
            if pattern.search(text):
                risk_factors.append(risk_type)
        
        return risk_factors
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the analysis engine"""
        if self.analysis_count == 0:
            return {'avg_analysis_time_ms': 0.0, 'total_analyses': 0}
        
        avg_time = self.total_analysis_time / self.analysis_count
        return {
            'avg_analysis_time_ms': avg_time,
            'total_analyses': self.analysis_count,
            'performance_target_met': avg_time < 100.0
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics"""
        self.analysis_count = 0
        self.total_analysis_time = 0.0

# Export main classes for import
__all__ = [
    'ContentAnalysisEngine',
    'ContentAnalysisResult', 
    'WorkflowState',
    'ComplexityLevel',
    'StateAnalyzer',
    'ComplexityAnalyzer', 
    'DependencyAnalyzer'
]