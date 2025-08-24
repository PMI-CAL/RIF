"""
Pattern Application Engine - Core Components

This module contains the core interfaces and data models for the Pattern Application Engine,
which applies learned patterns to new issues with context adaptation and success tracking.

The system is designed with abstraction layers to handle the dependency on Issue #76
(Pattern Matching System) while enabling parallel development.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import uuid
from enum import Enum


class PatternApplicationStatus(Enum):
    """Status of pattern application process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AdaptationStrategy(Enum):
    """Strategy for pattern adaptation to context."""
    EXACT_MATCH = "exact_match"
    TECH_STACK_ADAPTATION = "tech_stack_adaptation"
    COMPLEXITY_SCALING = "complexity_scaling"
    CONSTRAINT_INTEGRATION = "constraint_integration"
    FULL_ADAPTATION = "full_adaptation"


@dataclass
class TechStack:
    """Technology stack information for context adaptation."""
    primary_language: str
    frameworks: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    architecture_pattern: Optional[str] = None
    deployment_target: Optional[str] = None


@dataclass
class IssueConstraints:
    """Constraints and requirements for issue implementation."""
    timeline: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None
    quality_gates: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    performance_requirements: Optional[Dict[str, Any]] = None
    compatibility_requirements: List[str] = field(default_factory=list)


@dataclass
class IssueContext:
    """Comprehensive context information for an issue."""
    issue_id: str
    title: str
    description: str
    complexity: str  # low, medium, high, very-high
    tech_stack: TechStack
    constraints: IssueConstraints
    domain: str
    labels: List[str] = field(default_factory=list)
    similar_issues: List[str] = field(default_factory=list)
    existing_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "description": self.description,
            "complexity": self.complexity,
            "tech_stack": {
                "primary_language": self.tech_stack.primary_language,
                "frameworks": self.tech_stack.frameworks,
                "databases": self.tech_stack.databases,
                "tools": self.tech_stack.tools,
                "architecture_pattern": self.tech_stack.architecture_pattern,
                "deployment_target": self.tech_stack.deployment_target
            },
            "constraints": {
                "timeline": self.constraints.timeline,
                "resource_limits": self.constraints.resource_limits,
                "quality_gates": self.constraints.quality_gates,
                "compliance_requirements": self.constraints.compliance_requirements,
                "performance_requirements": self.constraints.performance_requirements,
                "compatibility_requirements": self.constraints.compatibility_requirements
            },
            "domain": self.domain,
            "labels": self.labels,
            "similar_issues": self.similar_issues,
            "existing_patterns": self.existing_patterns
        }


@dataclass
class Pattern:
    """Pattern definition with metadata and implementation details."""
    pattern_id: str
    name: str
    description: str
    complexity: str
    tech_stack: Optional[TechStack] = None
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0
    implementation_steps: List[Dict[str, Any]] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity,
            "tech_stack": {
                "primary_language": self.tech_stack.primary_language,
                "frameworks": self.tech_stack.frameworks,
                "databases": self.tech_stack.databases,
                "tools": self.tech_stack.tools,
                "architecture_pattern": self.tech_stack.architecture_pattern,
                "deployment_target": self.tech_stack.deployment_target
            } if self.tech_stack else None,
            "domain": self.domain,
            "tags": self.tags,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "implementation_steps": self.implementation_steps,
            "code_examples": self.code_examples,
            "validation_criteria": self.validation_criteria
        }


@dataclass
class AdaptationResult:
    """Result of pattern adaptation to context."""
    adapted_pattern: Pattern
    adaptation_strategy: AdaptationStrategy
    changes_made: List[str]
    confidence_score: float
    adaptation_notes: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapted_pattern": self.adapted_pattern.to_dict(),
            "adaptation_strategy": self.adaptation_strategy.value,
            "changes_made": self.changes_made,
            "confidence_score": self.confidence_score,
            "adaptation_notes": self.adaptation_notes,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ImplementationTask:
    """Individual task within an implementation plan."""
    task_id: str
    title: str
    description: str
    estimated_time: str
    complexity: str
    dependencies: List[str] = field(default_factory=list)
    agent_assignment: Optional[str] = None
    validation_criteria: List[str] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ImplementationPlan:
    """Comprehensive implementation plan generated from adapted pattern."""
    plan_id: str
    issue_id: str
    pattern_id: str
    tasks: List[ImplementationTask]
    estimated_total_time: str
    complexity_assessment: str
    risk_factors: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plan_id": self.plan_id,
            "issue_id": self.issue_id,
            "pattern_id": self.pattern_id,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "description": task.description,
                    "estimated_time": task.estimated_time,
                    "complexity": task.complexity,
                    "dependencies": task.dependencies,
                    "agent_assignment": task.agent_assignment,
                    "validation_criteria": task.validation_criteria,
                    "code_examples": task.code_examples
                }
                for task in self.tasks
            ],
            "estimated_total_time": self.estimated_total_time,
            "complexity_assessment": self.complexity_assessment,
            "risk_factors": self.risk_factors,
            "quality_gates": self.quality_gates,
            "checkpoints": self.checkpoints,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ApplicationRecord:
    """Record of pattern application with tracking information."""
    application_id: str
    pattern_id: str
    issue_id: str
    status: PatternApplicationStatus
    adaptation_result: Optional[AdaptationResult] = None
    implementation_plan: Optional[ImplementationPlan] = None
    success_score: Optional[float] = None
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "application_id": self.application_id,
            "pattern_id": self.pattern_id,
            "issue_id": self.issue_id,
            "status": self.status.value,
            "adaptation_result": self.adaptation_result.to_dict() if self.adaptation_result else None,
            "implementation_plan": self.implementation_plan.to_dict() if self.implementation_plan else None,
            "success_score": self.success_score,
            "execution_metrics": self.execution_metrics,
            "lessons_learned": self.lessons_learned,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class PatternMatchingInterface(ABC):
    """
    Abstract interface for pattern matching functionality.
    
    This interface provides abstraction over the pattern matching system
    from Issue #76, enabling parallel development while maintaining
    loose coupling.
    """
    
    @abstractmethod
    def find_applicable_patterns(self, issue_context: IssueContext, 
                               limit: int = 10) -> List[Pattern]:
        """
        Find patterns applicable to the given issue context.
        
        Args:
            issue_context: Context information for the issue
            limit: Maximum number of patterns to return
            
        Returns:
            List of applicable patterns ranked by relevance
        """
        pass
    
    @abstractmethod
    def rank_patterns(self, patterns: List[Pattern], 
                     issue_context: IssueContext) -> List[Pattern]:
        """
        Rank patterns by applicability to the given context.
        
        Args:
            patterns: List of patterns to rank
            issue_context: Context for ranking
            
        Returns:
            List of patterns ranked by relevance (highest first)
        """
        pass
    
    @abstractmethod
    def calculate_pattern_relevance(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """
        Calculate relevance score for a pattern-context pair.
        
        Args:
            pattern: Pattern to evaluate
            issue_context: Context to match against
            
        Returns:
            Relevance score (0.0 to 1.0, higher is more relevant)
        """
        pass


class PatternApplicationInterface(ABC):
    """
    Abstract interface for pattern application functionality.
    
    This is the main interface for the Pattern Application Engine,
    providing methods for applying patterns to new issues with
    context adaptation and success tracking.
    """
    
    @abstractmethod
    def apply_pattern(self, pattern_id: str, 
                     issue_context: IssueContext) -> ApplicationRecord:
        """
        Apply a pattern to an issue context.
        
        Args:
            pattern_id: ID of pattern to apply
            issue_context: Context of issue to apply pattern to
            
        Returns:
            ApplicationRecord tracking the application process
        """
        pass
    
    @abstractmethod
    def adapt_pattern_to_context(self, pattern: Pattern, 
                               issue_context: IssueContext) -> AdaptationResult:
        """
        Adapt a pattern to a specific issue context.
        
        Args:
            pattern: Pattern to adapt
            issue_context: Context to adapt to
            
        Returns:
            AdaptationResult containing adapted pattern and metadata
        """
        pass
    
    @abstractmethod
    def generate_implementation_plan(self, adapted_pattern: Pattern, 
                                   issue_context: IssueContext) -> ImplementationPlan:
        """
        Generate implementation plan from adapted pattern.
        
        Args:
            adapted_pattern: Pattern adapted to context
            issue_context: Context for implementation
            
        Returns:
            ImplementationPlan with detailed tasks and guidance
        """
        pass
    
    @abstractmethod
    def track_application_progress(self, application_id: str, 
                                 status: PatternApplicationStatus,
                                 metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update tracking information for pattern application.
        
        Args:
            application_id: ID of application to update
            status: New status
            metrics: Optional metrics to record
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def measure_application_success(self, application_id: str) -> float:
        """
        Measure success score for a completed pattern application.
        
        Args:
            application_id: ID of application to measure
            
        Returns:
            Success score (0.0 to 1.0, higher is better)
        """
        pass


class ContextExtractionInterface(ABC):
    """
    Abstract interface for extracting context from GitHub issues.
    
    This interface handles parsing GitHub issues and extracting
    structured context information for pattern matching and adaptation.
    """
    
    @abstractmethod
    def extract_issue_context(self, issue_data: Dict[str, Any]) -> IssueContext:
        """
        Extract structured context from GitHub issue data.
        
        Args:
            issue_data: GitHub issue data (from gh CLI)
            
        Returns:
            IssueContext with extracted information
        """
        pass
    
    @abstractmethod
    def detect_tech_stack(self, issue_content: str, 
                         repository_data: Optional[Dict[str, Any]] = None) -> TechStack:
        """
        Detect technology stack from issue content and repository.
        
        Args:
            issue_content: Combined issue title and body text
            repository_data: Optional repository information
            
        Returns:
            TechStack with detected technologies
        """
        pass
    
    @abstractmethod
    def assess_complexity(self, issue_context: IssueContext) -> str:
        """
        Assess complexity level for an issue.
        
        Args:
            issue_context: Context to assess
            
        Returns:
            Complexity level: 'low', 'medium', 'high', 'very-high'
        """
        pass
    
    @abstractmethod
    def extract_constraints(self, issue_content: str) -> IssueConstraints:
        """
        Extract constraints and requirements from issue content.
        
        Args:
            issue_content: Issue description and comments
            
        Returns:
            IssueConstraints with extracted requirements
        """
        pass


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}{unique_id}" if prefix else unique_id


def load_pattern_from_json(file_path: str) -> Pattern:
    """
    Load pattern from JSON file.
    
    Args:
        file_path: Path to JSON pattern file
        
    Returns:
        Pattern object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract tech stack if present
    tech_stack = None
    if 'tech_stack' in data and data['tech_stack']:
        ts_data = data['tech_stack']
        tech_stack = TechStack(
            primary_language=ts_data.get('primary_language', ''),
            frameworks=ts_data.get('frameworks', []),
            databases=ts_data.get('databases', []),
            tools=ts_data.get('tools', []),
            architecture_pattern=ts_data.get('architecture_pattern'),
            deployment_target=ts_data.get('deployment_target')
        )
    
    return Pattern(
        pattern_id=data.get('pattern_id', generate_id('pattern_')),
        name=data.get('pattern_name', data.get('name', 'Unknown Pattern')),
        description=data.get('description', ''),
        complexity=data.get('complexity', 'medium'),
        tech_stack=tech_stack,
        domain=data.get('domain', 'general'),
        tags=data.get('tags', []),
        confidence=data.get('confidence', 0.0),
        success_rate=data.get('success_rate', 0.0),
        usage_count=data.get('usage_count', 0),
        implementation_steps=data.get('implementation_steps', []),
        code_examples=data.get('code_examples', []),
        validation_criteria=data.get('validation_criteria', [])
    )


class PatternApplicationError(Exception):
    """Base exception for pattern application errors."""
    pass


class PatternNotFoundError(PatternApplicationError):
    """Exception raised when pattern is not found."""
    pass


class ContextExtractionError(PatternApplicationError):
    """Exception raised during context extraction."""
    pass


class PatternAdaptationError(PatternApplicationError):
    """Exception raised during pattern adaptation."""
    pass