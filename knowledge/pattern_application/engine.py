"""
Pattern Application Engine - Main Implementation

This module contains the main PatternApplicationEngine class that orchestrates
the entire pattern application process, from context extraction to success tracking.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json
import os

from .core import (
    Pattern, IssueContext, TechStack, IssueConstraints,
    ApplicationRecord, AdaptationResult, 
    ImplementationPlan, PatternApplicationStatus, AdaptationStrategy,
    PatternApplicationInterface, PatternMatchingInterface,
    ContextExtractionInterface, generate_id,
    PatternApplicationError, PatternNotFoundError
)

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system
except ImportError:
    # Fallback for testing
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")

logger = logging.getLogger(__name__)


class PatternApplicationEngine(PatternApplicationInterface):
    """
    Main Pattern Application Engine implementation.
    
    This class orchestrates the entire pattern application process:
    1. Context extraction from GitHub issues
    2. Pattern matching and ranking
    3. Pattern adaptation to context
    4. Implementation plan generation
    5. Application tracking and success measurement
    
    The engine is designed with dependency injection to support different
    implementations of pattern matching, context extraction, etc.
    """
    
    def __init__(self, 
                 pattern_matcher: PatternMatchingInterface,
                 context_extractor: ContextExtractionInterface,
                 knowledge_system=None):
        """
        Initialize Pattern Application Engine.
        
        Args:
            pattern_matcher: Pattern matching implementation
            context_extractor: Context extraction implementation
            knowledge_system: Knowledge system interface (optional)
        """
        self.pattern_matcher = pattern_matcher
        self.context_extractor = context_extractor
        self.knowledge_system = knowledge_system or get_knowledge_system()
        
        # Storage for tracking applications
        self.applications: Dict[str, ApplicationRecord] = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Pattern Application Engine initialized")
    
    def _load_configuration(self):
        """Load configuration from environment or defaults."""
        self.config = {
            'max_patterns_to_consider': int(os.getenv('PA_MAX_PATTERNS', '20')),
            'min_confidence_threshold': float(os.getenv('PA_MIN_CONFIDENCE', '0.5')),
            'adaptation_strategies': ['full_adaptation', 'tech_stack_adaptation'],
            'success_tracking_enabled': True,
            'store_applications': True
        }
    
    def apply_pattern(self, pattern_id: str, 
                     issue_context: IssueContext) -> ApplicationRecord:
        """
        Apply a pattern to an issue context.
        
        This is the main entry point for pattern application. It orchestrates
        the entire process from pattern retrieval to success tracking.
        
        Args:
            pattern_id: ID of pattern to apply
            issue_context: Context of issue to apply pattern to
            
        Returns:
            ApplicationRecord tracking the application process
        """
        logger.info(f"Starting pattern application: {pattern_id} -> {issue_context.issue_id}")
        
        # Create application record
        application_id = generate_id('app_')
        application = ApplicationRecord(
            application_id=application_id,
            pattern_id=pattern_id,
            issue_id=issue_context.issue_id,
            status=PatternApplicationStatus.IN_PROGRESS
        )
        
        self.applications[application_id] = application
        
        try:
            # Step 1: Retrieve pattern from knowledge system
            pattern = self._retrieve_pattern(pattern_id)
            if not pattern:
                raise PatternNotFoundError(f"Pattern {pattern_id} not found")
            
            # Step 2: Adapt pattern to context
            logger.info(f"Adapting pattern {pattern_id} to context")
            adaptation_result = self.adapt_pattern_to_context(pattern, issue_context)
            application.adaptation_result = adaptation_result
            
            # Step 3: Generate implementation plan
            logger.info(f"Generating implementation plan")
            implementation_plan = self.generate_implementation_plan(
                adaptation_result.adapted_pattern, issue_context)
            application.implementation_plan = implementation_plan
            
            # Step 4: Update status and store results
            application.status = PatternApplicationStatus.COMPLETED
            application.completed_at = datetime.utcnow()
            
            # Store application record in knowledge system
            if self.config['store_applications']:
                self._store_application_record(application)
            
            logger.info(f"Pattern application completed: {application_id}")
            
            return application
            
        except Exception as e:
            logger.error(f"Pattern application failed: {str(e)}")
            application.status = PatternApplicationStatus.FAILED
            application.execution_metrics['error'] = str(e)
            application.completed_at = datetime.utcnow()
            raise PatternApplicationError(f"Pattern application failed: {str(e)}") from e
    
    def apply_best_pattern(self, issue_context: IssueContext) -> Optional[ApplicationRecord]:
        """
        Find and apply the best matching pattern for an issue context.
        
        This method uses the pattern matcher to find applicable patterns,
        ranks them, and applies the best one.
        
        Args:
            issue_context: Context of issue to find pattern for
            
        Returns:
            ApplicationRecord if successful, None if no suitable pattern found
        """
        logger.info(f"Finding best pattern for issue {issue_context.issue_id}")
        
        try:
            # Find applicable patterns
            applicable_patterns = self.pattern_matcher.find_applicable_patterns(
                issue_context, limit=self.config['max_patterns_to_consider'])
            
            if not applicable_patterns:
                logger.info("No applicable patterns found")
                return None
            
            # Find best pattern (first in ranked list)
            best_pattern = applicable_patterns[0]
            
            # Check confidence threshold
            relevance_score = self.pattern_matcher.calculate_pattern_relevance(
                best_pattern, issue_context)
            
            if relevance_score < self.config['min_confidence_threshold']:
                logger.info(f"Best pattern relevance {relevance_score} below threshold")
                return None
            
            # Apply the best pattern
            return self.apply_pattern(best_pattern.pattern_id, issue_context)
            
        except Exception as e:
            logger.error(f"Failed to find and apply best pattern: {str(e)}")
            raise PatternApplicationError(f"Best pattern application failed: {str(e)}") from e
    
    def adapt_pattern_to_context(self, pattern: Pattern, 
                               issue_context: IssueContext) -> AdaptationResult:
        """
        Adapt a pattern to a specific issue context.
        
        This method implements multi-factor adaptation considering:
        - Technology stack differences
        - Complexity scaling
        - Constraint integration
        - Domain-specific modifications
        
        Args:
            pattern: Pattern to adapt
            issue_context: Context to adapt to
            
        Returns:
            AdaptationResult containing adapted pattern and metadata
        """
        logger.info(f"Adapting pattern {pattern.pattern_id} for context")
        
        # Create a copy of the pattern for adaptation
        adapted_pattern = self._deep_copy_pattern(pattern)
        changes_made = []
        adaptation_strategy = AdaptationStrategy.EXACT_MATCH
        
        # Step 1: Technology stack adaptation
        if self._needs_tech_stack_adaptation(pattern, issue_context):
            adapted_pattern = self._adapt_tech_stack(adapted_pattern, issue_context.tech_stack)
            changes_made.append("Technology stack adaptation")
            adaptation_strategy = AdaptationStrategy.TECH_STACK_ADAPTATION
        
        # Step 2: Complexity scaling
        if pattern.complexity != issue_context.complexity:
            adapted_pattern = self._adapt_complexity(adapted_pattern, issue_context.complexity)
            changes_made.append(f"Complexity scaling: {pattern.complexity} -> {issue_context.complexity}")
            if adaptation_strategy == AdaptationStrategy.EXACT_MATCH:
                adaptation_strategy = AdaptationStrategy.COMPLEXITY_SCALING
        
        # Step 3: Constraint integration
        if self._has_significant_constraints(issue_context.constraints):
            adapted_pattern = self._adapt_constraints(adapted_pattern, issue_context.constraints)
            changes_made.append("Constraint integration")
            if adaptation_strategy in [AdaptationStrategy.EXACT_MATCH]:
                adaptation_strategy = AdaptationStrategy.CONSTRAINT_INTEGRATION
        
        # Set full adaptation if multiple adaptations were made
        if len(changes_made) > 1:
            adaptation_strategy = AdaptationStrategy.FULL_ADAPTATION
        
        # Calculate confidence score based on changes made
        confidence_score = self._calculate_adaptation_confidence(pattern, adapted_pattern, changes_made)
        
        # Create adaptation result
        adaptation_result = AdaptationResult(
            adapted_pattern=adapted_pattern,
            adaptation_strategy=adaptation_strategy,
            changes_made=changes_made,
            confidence_score=confidence_score,
            adaptation_notes=self._generate_adaptation_notes(changes_made, pattern, issue_context)
        )
        
        logger.info(f"Pattern adaptation completed with confidence {confidence_score:.2f}")
        
        return adaptation_result
    
    def generate_implementation_plan(self, adapted_pattern: Pattern, 
                                   issue_context: IssueContext) -> ImplementationPlan:
        """
        Generate implementation plan from adapted pattern.
        
        This method converts the adapted pattern into a detailed implementation
        plan with tasks, timelines, and validation criteria.
        
        Args:
            adapted_pattern: Pattern adapted to context
            issue_context: Context for implementation
            
        Returns:
            ImplementationPlan with detailed tasks and guidance
        """
        from .plan_generator import PlanGenerator
        
        # Create plan generator (could be injected in future)
        plan_generator = PlanGenerator(self.knowledge_system)
        
        # Generate the plan
        plan = plan_generator.generate_plan(adapted_pattern, issue_context)
        
        logger.info(f"Generated implementation plan with {len(plan.tasks)} tasks")
        
        return plan
    
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
        if application_id not in self.applications:
            logger.error(f"Application {application_id} not found for tracking")
            return False
        
        application = self.applications[application_id]
        application.status = status
        
        if metrics:
            application.execution_metrics.update(metrics)
        
        if status in [PatternApplicationStatus.COMPLETED, PatternApplicationStatus.FAILED]:
            application.completed_at = datetime.utcnow()
        
        logger.info(f"Updated application {application_id} status to {status.value}")
        
        return True
    
    def measure_application_success(self, application_id: str) -> float:
        """
        Measure success score for a completed pattern application.
        
        This method analyzes the application results and generates a success
        score based on multiple factors including implementation quality,
        timeline adherence, and outcome validation.
        
        Args:
            application_id: ID of application to measure
            
        Returns:
            Success score (0.0 to 1.0, higher is better)
        """
        from .success_tracker import SuccessTracker
        
        if application_id not in self.applications:
            logger.error(f"Application {application_id} not found for success measurement")
            return 0.0
        
        application = self.applications[application_id]
        
        # Create success tracker (could be injected in future)
        success_tracker = SuccessTracker(self.knowledge_system)
        
        # Calculate success score
        success_score = success_tracker.calculate_success_score(application)
        
        # Update application record
        application.success_score = success_score
        
        logger.info(f"Measured success score {success_score:.2f} for application {application_id}")
        
        return success_score
    
    def get_application_record(self, application_id: str) -> Optional[ApplicationRecord]:
        """Get application record by ID."""
        return self.applications.get(application_id)
    
    def list_applications(self, issue_id: Optional[str] = None) -> List[ApplicationRecord]:
        """
        List application records, optionally filtered by issue ID.
        
        Args:
            issue_id: Optional issue ID to filter by
            
        Returns:
            List of application records
        """
        applications = list(self.applications.values())
        
        if issue_id:
            applications = [app for app in applications if app.issue_id == issue_id]
        
        return applications
    
    # Private helper methods
    
    def _retrieve_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve pattern from knowledge system."""
        try:
            # Search for pattern by ID
            results = self.knowledge_system.retrieve_knowledge(
                pattern_id, collection="patterns", n_results=1)
            
            if not results:
                return None
            
            # Convert result to Pattern object
            pattern_data = results[0]
            return self._dict_to_pattern(pattern_data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve pattern {pattern_id}: {str(e)}")
            return None
    
    def _dict_to_pattern(self, pattern_dict: Dict[str, Any]) -> Pattern:
        """Convert dictionary to Pattern object."""
        from .core import Pattern, TechStack
        
        # Handle nested content if present
        if 'content' in pattern_dict and isinstance(pattern_dict['content'], (str, dict)):
            if isinstance(pattern_dict['content'], str):
                try:
                    content = json.loads(pattern_dict['content'])
                except json.JSONDecodeError:
                    content = pattern_dict
            else:
                content = pattern_dict['content']
        else:
            content = pattern_dict
        
        # Extract tech stack
        tech_stack = None
        if 'tech_stack' in content and content['tech_stack']:
            ts_data = content['tech_stack']
            tech_stack = TechStack(
                primary_language=ts_data.get('primary_language', ''),
                frameworks=ts_data.get('frameworks', []),
                databases=ts_data.get('databases', []),
                tools=ts_data.get('tools', []),
                architecture_pattern=ts_data.get('architecture_pattern'),
                deployment_target=ts_data.get('deployment_target')
            )
        
        return Pattern(
            pattern_id=content.get('pattern_id', generate_id('pattern_')),
            name=content.get('pattern_name', content.get('name', 'Unknown Pattern')),
            description=content.get('description', ''),
            complexity=content.get('complexity', 'medium'),
            tech_stack=tech_stack,
            domain=content.get('domain', 'general'),
            tags=content.get('tags', []),
            confidence=content.get('confidence', 0.0),
            success_rate=content.get('success_rate', 0.0),
            usage_count=content.get('usage_count', 0),
            implementation_steps=content.get('implementation_steps', []),
            code_examples=content.get('code_examples', []),
            validation_criteria=content.get('validation_criteria', [])
        )
    
    def _deep_copy_pattern(self, pattern: Pattern) -> Pattern:
        """Create a deep copy of a pattern for adaptation."""
        import copy
        return copy.deepcopy(pattern)
    
    def _needs_tech_stack_adaptation(self, pattern: Pattern, context: IssueContext) -> bool:
        """Check if pattern needs technology stack adaptation."""
        if not pattern.tech_stack or not context.tech_stack:
            return False
        
        return (pattern.tech_stack.primary_language != context.tech_stack.primary_language or
                set(pattern.tech_stack.frameworks) != set(context.tech_stack.frameworks))
    
    def _adapt_tech_stack(self, pattern: Pattern, target_tech_stack: TechStack) -> Pattern:
        """Adapt pattern for different technology stack."""
        # Update pattern's tech stack
        pattern.tech_stack = target_tech_stack
        
        # TODO: Implement more sophisticated tech stack adaptation
        # - Convert code examples to target language
        # - Adapt framework-specific recommendations
        # - Update tool suggestions
        
        return pattern
    
    def _adapt_complexity(self, pattern: Pattern, target_complexity: str) -> Pattern:
        """Adapt pattern for different complexity level."""
        original_complexity = pattern.complexity
        pattern.complexity = target_complexity
        
        # Adjust implementation steps based on complexity
        complexity_scale = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'very-high': 2.0
        }
        
        original_scale = complexity_scale.get(original_complexity, 1.0)
        target_scale = complexity_scale.get(target_complexity, 1.0)
        adjustment_ratio = target_scale / original_scale
        
        # TODO: Implement more sophisticated complexity scaling
        # - Adjust number of implementation steps
        # - Scale validation criteria
        # - Modify risk assessments
        
        return pattern
    
    def _has_significant_constraints(self, constraints: IssueConstraints) -> bool:
        """Check if issue has significant constraints requiring adaptation."""
        return (constraints.timeline is not None or
                constraints.resource_limits is not None or
                len(constraints.quality_gates) > 0 or
                constraints.performance_requirements is not None)
    
    def _adapt_constraints(self, pattern: Pattern, constraints: IssueConstraints) -> Pattern:
        """Adapt pattern for specific constraints."""
        # TODO: Implement constraint adaptation
        # - Adjust timeline recommendations
        # - Add constraint-specific validation steps
        # - Modify resource allocation suggestions
        
        return pattern
    
    def _calculate_adaptation_confidence(self, original: Pattern, adapted: Pattern, 
                                       changes: List[str]) -> float:
        """Calculate confidence score for pattern adaptation."""
        base_confidence = original.confidence
        
        # Reduce confidence based on number and type of changes
        reduction_factors = {
            'Technology stack adaptation': 0.1,
            'Constraint integration': 0.05,
        }
        
        total_reduction = 0.0
        for change in changes:
            for change_type, factor in reduction_factors.items():
                if change_type in change:
                    total_reduction += factor
                    break
        
        # Apply complexity change penalty
        if original.complexity != adapted.complexity:
            complexity_diff = abs(['low', 'medium', 'high', 'very-high'].index(adapted.complexity) -
                                ['low', 'medium', 'high', 'very-high'].index(original.complexity))
            total_reduction += complexity_diff * 0.05
        
        # Ensure confidence doesn't go below 0.1
        return max(0.1, base_confidence - total_reduction)
    
    def _generate_adaptation_notes(self, changes: List[str], original: Pattern, 
                                 context: IssueContext) -> str:
        """Generate human-readable notes about adaptation changes."""
        if not changes:
            return "Pattern applied without modifications."
        
        notes = f"Pattern '{original.name}' adapted for issue {context.issue_id}:\n"
        for i, change in enumerate(changes, 1):
            notes += f"{i}. {change}\n"
        
        notes += f"\nOriginal pattern complexity: {original.complexity}"
        notes += f"\nTarget context complexity: {context.complexity}"
        
        if original.tech_stack and context.tech_stack:
            notes += f"\nTech stack adaptation: {original.tech_stack.primary_language} -> {context.tech_stack.primary_language}"
        
        return notes
    
    def _store_application_record(self, application: ApplicationRecord):
        """Store application record in knowledge system."""
        try:
            self.knowledge_system.store_knowledge(
                collection="pattern_applications",
                content=application.to_dict(),
                metadata={
                    "type": "pattern_application",
                    "status": application.status.value,
                    "pattern_id": application.pattern_id,
                    "issue_id": application.issue_id,
                    "application_id": application.application_id
                },
                doc_id=application.application_id
            )
            logger.info(f"Stored application record {application.application_id}")
        except Exception as e:
            logger.error(f"Failed to store application record: {str(e)}")