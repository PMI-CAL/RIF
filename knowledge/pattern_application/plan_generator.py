"""
Implementation Plan Generator

This module generates detailed implementation plans from adapted patterns,
breaking down the work into specific tasks with timelines, dependencies,
and validation criteria.
"""

from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime, timedelta
import logging

from .core import (
    Pattern, IssueContext, ImplementationPlan, ImplementationTask,
    generate_id
)

logger = logging.getLogger(__name__)


class PlanGenerator:
    """
    Implementation plan generator that converts adapted patterns into
    detailed, actionable implementation plans with tasks, timelines,
    and validation criteria.
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize plan generator."""
        self.knowledge_system = knowledge_system
        self._load_planning_templates()
        self._load_estimation_rules()
        self._load_agent_assignment_rules()
    
    def generate_plan(self, adapted_pattern: Pattern, 
                     issue_context: IssueContext) -> ImplementationPlan:
        """
        Generate detailed implementation plan from adapted pattern.
        
        This method converts the adapted pattern into a comprehensive
        implementation plan with specific tasks, timelines, dependencies,
        and validation criteria.
        
        Args:
            adapted_pattern: Pattern adapted to context
            issue_context: Context for implementation
            
        Returns:
            ImplementationPlan with detailed tasks and guidance
        """
        logger.info(f"Generating implementation plan for pattern {adapted_pattern.pattern_id}")
        
        # Generate plan ID
        plan_id = generate_id('plan_')
        
        # Extract tasks from pattern
        tasks = self._extract_tasks_from_pattern(adapted_pattern, issue_context)
        
        # Add context-specific tasks
        context_tasks = self._generate_context_specific_tasks(issue_context)
        tasks.extend(context_tasks)
        
        # Add validation tasks
        validation_tasks = self._generate_validation_tasks(adapted_pattern, issue_context)
        tasks.extend(validation_tasks)
        
        # Establish dependencies between tasks
        tasks = self._establish_task_dependencies(tasks, adapted_pattern, issue_context)
        
        # Assign agents to tasks
        tasks = self._assign_agents_to_tasks(tasks, issue_context)
        
        # Calculate time estimates
        total_time = self._calculate_total_time(tasks)
        
        # Generate checkpoints
        checkpoints = self._generate_checkpoints(tasks, adapted_pattern)
        
        # Assess risks
        risk_factors = self._assess_risk_factors(adapted_pattern, issue_context, tasks)
        
        # Generate quality gates
        quality_gates = self._generate_quality_gates(adapted_pattern, issue_context)
        
        # Define success criteria
        success_criteria = self._generate_success_criteria(adapted_pattern, issue_context)
        
        # Create implementation plan
        plan = ImplementationPlan(
            plan_id=plan_id,
            issue_id=issue_context.issue_id,
            pattern_id=adapted_pattern.pattern_id,
            tasks=tasks,
            estimated_total_time=total_time,
            complexity_assessment=issue_context.complexity,
            risk_factors=risk_factors,
            quality_gates=quality_gates,
            checkpoints=checkpoints,
            success_criteria=success_criteria
        )
        
        logger.info(f"Generated plan with {len(tasks)} tasks, estimated time: {total_time}")
        
        return plan
    
    def _load_planning_templates(self):
        """Load planning templates for different pattern types."""
        self.task_templates = {
            'analysis': {
                'template': "Analyze {component} for {purpose}",
                'estimated_time': "30-60 minutes",
                'complexity': 'low',
                'agent': 'rif-analyst'
            },
            'design': {
                'template': "Design {component} architecture for {purpose}",
                'estimated_time': "1-2 hours",
                'complexity': 'medium',
                'agent': 'rif-architect'
            },
            'implementation': {
                'template': "Implement {component} with {requirements}",
                'estimated_time': "2-4 hours",
                'complexity': 'medium',
                'agent': 'rif-implementer'
            },
            'testing': {
                'template': "Test {component} for {criteria}",
                'estimated_time': "1-2 hours",
                'complexity': 'low',
                'agent': 'rif-validator'
            },
            'validation': {
                'template': "Validate {component} meets {requirements}",
                'estimated_time': "30-90 minutes",
                'complexity': 'low',
                'agent': 'rif-validator'
            },
            'integration': {
                'template': "Integrate {component} with {system}",
                'estimated_time': "1-3 hours",
                'complexity': 'medium',
                'agent': 'rif-implementer'
            }
        }
        
        self.complexity_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5,
            'very-high': 2.0
        }
    
    def _load_estimation_rules(self):
        """Load time estimation rules."""
        self.base_estimates = {
            'file_creation': 15,  # minutes
            'simple_function': 30,
            'complex_function': 60,
            'class_design': 45,
            'database_schema': 60,
            'api_endpoint': 45,
            'test_suite': 90,
            'documentation': 30,
            'integration': 120,
            'validation': 60
        }
        
        self.tech_stack_multipliers = {
            'javascript': 1.0,
            'python': 0.9,
            'java': 1.2,
            'go': 1.1,
            'rust': 1.3,
            'unknown': 1.2
        }
    
    def _load_agent_assignment_rules(self):
        """Load agent assignment rules."""
        self.agent_capabilities = {
            'rif-analyst': [
                'requirements analysis', 'complexity assessment', 
                'pattern matching', 'risk analysis'
            ],
            'rif-architect': [
                'system design', 'architecture planning',
                'dependency analysis', 'scalability planning'
            ],
            'rif-implementer': [
                'code implementation', 'feature development',
                'bug fixing', 'integration', 'refactoring'
            ],
            'rif-validator': [
                'testing', 'quality assurance', 'validation',
                'performance testing', 'security testing'
            ],
            'rif-learner': [
                'pattern extraction', 'knowledge storage',
                'learning analysis', 'improvement recommendations'
            ]
        }
        
        self.task_agent_mapping = {
            'analysis': 'rif-analyst',
            'design': 'rif-architect',
            'architecture': 'rif-architect',
            'implementation': 'rif-implementer',
            'coding': 'rif-implementer',
            'testing': 'rif-validator',
            'validation': 'rif-validator',
            'quality': 'rif-validator',
            'learning': 'rif-learner',
            'documentation': 'rif-implementer'  # Can be shared
        }
    
    def _extract_tasks_from_pattern(self, pattern: Pattern, 
                                   context: IssueContext) -> List[ImplementationTask]:
        """Extract tasks from pattern implementation steps."""
        tasks = []
        
        for i, step in enumerate(pattern.implementation_steps):
            # Handle different step formats
            if isinstance(step, dict):
                title = step.get('title', f"Step {i+1}")
                description = step.get('description', str(step))
                estimated_time = step.get('estimated_time', self._estimate_step_time(description, context))
            elif isinstance(step, str):
                title = f"Step {i+1}: {step[:50]}..."
                description = step
                estimated_time = self._estimate_step_time(description, context)
            else:
                title = f"Step {i+1}"
                description = str(step)
                estimated_time = "1 hour"
            
            # Determine task complexity
            task_complexity = self._determine_task_complexity(description, context.complexity)
            
            # Create task
            task = ImplementationTask(
                task_id=generate_id('task_'),
                title=title,
                description=description,
                estimated_time=estimated_time,
                complexity=task_complexity,
                dependencies=[],
                agent_assignment=None,  # Will be assigned later
                validation_criteria=self._extract_task_validation(step, pattern),
                code_examples=self._extract_task_code_examples(step, pattern)
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_context_specific_tasks(self, context: IssueContext) -> List[ImplementationTask]:
        """Generate tasks specific to the issue context."""
        tasks = []
        
        # Technology stack setup tasks
        if context.tech_stack and context.tech_stack.primary_language:
            setup_task = ImplementationTask(
                task_id=generate_id('task_'),
                title=f"Setup {context.tech_stack.primary_language} development environment",
                description=f"Configure development environment for {context.tech_stack.primary_language} "
                           f"with required frameworks: {', '.join(context.tech_stack.frameworks or [])}",
                estimated_time="30-60 minutes",
                complexity='low',
                dependencies=[],
                agent_assignment='rif-implementer',
                validation_criteria=['Environment successfully configured', 'All dependencies installed'],
                code_examples=[]
            )
            tasks.append(setup_task)
        
        # Constraint-specific tasks
        if context.constraints.performance_requirements:
            perf_task = ImplementationTask(
                task_id=generate_id('task_'),
                title="Implement performance monitoring",
                description="Add performance monitoring and optimization for specified requirements",
                estimated_time="1-2 hours",
                complexity='medium',
                dependencies=[],
                agent_assignment='rif-implementer',
                validation_criteria=['Performance requirements met', 'Monitoring in place'],
                code_examples=[]
            )
            tasks.append(perf_task)
        
        if context.constraints.compliance_requirements:
            compliance_task = ImplementationTask(
                task_id=generate_id('task_'),
                title="Ensure compliance requirements",
                description=f"Implement compliance measures for: {', '.join(context.constraints.compliance_requirements)}",
                estimated_time="2-4 hours",
                complexity='high',
                dependencies=[],
                agent_assignment='rif-validator',
                validation_criteria=['Compliance requirements validated', 'Audit trail established'],
                code_examples=[]
            )
            tasks.append(compliance_task)
        
        return tasks
    
    def _generate_validation_tasks(self, pattern: Pattern, 
                                  context: IssueContext) -> List[ImplementationTask]:
        """Generate validation and testing tasks."""
        tasks = []
        
        # Unit testing task
        unit_test_task = ImplementationTask(
            task_id=generate_id('task_'),
            title="Create unit tests",
            description="Implement comprehensive unit tests for all components",
            estimated_time=self._estimate_testing_time(pattern, context, 'unit'),
            complexity='medium',
            dependencies=[],
            agent_assignment='rif-validator',
            validation_criteria=['All functions have unit tests', 'Test coverage > 80%'],
            code_examples=[]
        )
        tasks.append(unit_test_task)
        
        # Integration testing task
        if context.complexity in ['high', 'very-high']:
            integration_test_task = ImplementationTask(
                task_id=generate_id('task_'),
                title="Create integration tests",
                description="Implement integration tests for system components",
                estimated_time=self._estimate_testing_time(pattern, context, 'integration'),
                complexity='medium',
                dependencies=[],
                agent_assignment='rif-validator',
                validation_criteria=['Integration points tested', 'End-to-end workflows validated'],
                code_examples=[]
            )
            tasks.append(integration_test_task)
        
        # Pattern validation task
        pattern_validation_task = ImplementationTask(
            task_id=generate_id('task_'),
            title="Validate pattern implementation",
            description=f"Ensure implementation follows pattern '{pattern.name}' correctly",
            estimated_time="1-2 hours",
            complexity='medium',
            dependencies=[],
            agent_assignment='rif-validator',
            validation_criteria=pattern.validation_criteria or ['Pattern correctly implemented'],
            code_examples=[]
        )
        tasks.append(pattern_validation_task)
        
        return tasks
    
    def _establish_task_dependencies(self, tasks: List[ImplementationTask], 
                                   pattern: Pattern, context: IssueContext) -> List[ImplementationTask]:
        """Establish dependencies between tasks."""
        if len(tasks) <= 1:
            return tasks
        
        # Basic sequential dependencies for pattern steps
        pattern_tasks = [task for task in tasks if task.title.startswith('Step')]
        for i in range(1, len(pattern_tasks)):
            pattern_tasks[i].dependencies.append(pattern_tasks[i-1].task_id)
        
        # Setup tasks should come first
        setup_tasks = [task for task in tasks if 'setup' in task.title.lower() or 'environment' in task.title.lower()]
        implementation_tasks = [task for task in tasks if task.agent_assignment == 'rif-implementer' and task not in setup_tasks]
        
        for impl_task in implementation_tasks:
            for setup_task in setup_tasks:
                if setup_task.task_id not in impl_task.dependencies:
                    impl_task.dependencies.append(setup_task.task_id)
        
        # Testing tasks depend on implementation tasks
        test_tasks = [task for task in tasks if 'test' in task.title.lower() or task.agent_assignment == 'rif-validator']
        
        for test_task in test_tasks:
            for impl_task in implementation_tasks:
                if impl_task.task_id not in test_task.dependencies:
                    test_task.dependencies.append(impl_task.task_id)
        
        return tasks
    
    def _assign_agents_to_tasks(self, tasks: List[ImplementationTask], 
                               context: IssueContext) -> List[ImplementationTask]:
        """Assign appropriate agents to tasks."""
        for task in tasks:
            if task.agent_assignment:
                continue  # Already assigned
            
            # Try to match based on task content
            task_content = f"{task.title} {task.description}".lower()
            
            best_agent = 'rif-implementer'  # Default
            best_score = 0
            
            for agent, capabilities in self.agent_capabilities.items():
                score = sum(1 for capability in capabilities if capability in task_content)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            task.agent_assignment = best_agent
        
        return tasks
    
    def _calculate_total_time(self, tasks: List[ImplementationTask]) -> str:
        """Calculate total estimated time for all tasks."""
        total_minutes = 0
        
        for task in tasks:
            # Parse time estimate
            time_str = task.estimated_time.lower()
            
            # Extract time ranges and convert to minutes
            if 'hour' in time_str:
                if '-' in time_str:
                    # Range like "1-2 hours"
                    parts = time_str.split('-')
                    try:
                        min_hours = float(parts[0].strip())
                        max_hours = float(parts[1].split()[0].strip())
                        avg_hours = (min_hours + max_hours) / 2
                        total_minutes += avg_hours * 60
                    except (ValueError, IndexError):
                        total_minutes += 60  # Default 1 hour
                else:
                    # Single value like "2 hours"
                    try:
                        hours = float(time_str.split()[0])
                        total_minutes += hours * 60
                    except (ValueError, IndexError):
                        total_minutes += 60
            elif 'minute' in time_str:
                if '-' in time_str:
                    # Range like "30-60 minutes"
                    parts = time_str.split('-')
                    try:
                        min_mins = float(parts[0].strip())
                        max_mins = float(parts[1].split()[0].strip())
                        avg_mins = (min_mins + max_mins) / 2
                        total_minutes += avg_mins
                    except (ValueError, IndexError):
                        total_minutes += 30  # Default 30 minutes
                else:
                    # Single value like "45 minutes"
                    try:
                        minutes = float(time_str.split()[0])
                        total_minutes += minutes
                    except (ValueError, IndexError):
                        total_minutes += 30
            else:
                total_minutes += 60  # Default 1 hour
        
        # Convert back to human-readable format
        if total_minutes < 60:
            return f"{int(total_minutes)} minutes"
        elif total_minutes < 480:  # Less than 8 hours
            hours = total_minutes / 60
            return f"{hours:.1f} hours"
        else:
            days = total_minutes / (60 * 8)  # 8 hour work day
            return f"{days:.1f} days"
    
    def _generate_checkpoints(self, tasks: List[ImplementationTask], 
                            pattern: Pattern) -> List[str]:
        """Generate checkpoints for progress tracking."""
        checkpoints = []
        
        # Major phase checkpoints
        if any(task.agent_assignment == 'rif-architect' for task in tasks):
            checkpoints.append("Architecture design complete")
        
        if any(task.agent_assignment == 'rif-implementer' for task in tasks):
            checkpoints.append("Core implementation complete")
        
        if any('test' in task.title.lower() for task in tasks):
            checkpoints.append("Testing phase complete")
        
        checkpoints.append("Pattern validation complete")
        checkpoints.append("All tasks completed and validated")
        
        return checkpoints
    
    def _assess_risk_factors(self, pattern: Pattern, context: IssueContext, 
                           tasks: List[ImplementationTask]) -> List[str]:
        """Assess and list risk factors for the implementation."""
        risks = []
        
        # Complexity-based risks
        if context.complexity == 'very-high':
            risks.append("Very high complexity may lead to scope creep")
            risks.append("Integration challenges with multiple components")
        elif context.complexity == 'high':
            risks.append("High complexity requires careful planning")
        
        # Technology stack risks
        if context.tech_stack and context.tech_stack.primary_language == 'unknown':
            risks.append("Unknown technology stack increases implementation uncertainty")
        
        # Pattern adaptation risks
        if pattern.confidence < 0.7:
            risks.append("Pattern has lower confidence, may require adjustments")
        
        # Constraint risks
        if context.constraints.timeline:
            risks.append("Timeline constraints may impact quality")
        
        if context.constraints.performance_requirements:
            risks.append("Performance requirements may require optimization iterations")
        
        # Task dependency risks
        max_dependencies = max(len(task.dependencies) for task in tasks) if tasks else 0
        if max_dependencies > 3:
            risks.append("Complex task dependencies may cause delays")
        
        return risks
    
    def _generate_quality_gates(self, pattern: Pattern, context: IssueContext) -> List[str]:
        """Generate quality gates for the implementation."""
        gates = []
        
        # Standard quality gates
        gates.append("All unit tests passing")
        gates.append("Code review completed")
        gates.append("No critical security vulnerabilities")
        
        # Complexity-based gates
        if context.complexity in ['high', 'very-high']:
            gates.append("Integration tests passing")
            gates.append("Performance benchmarks met")
            gates.append("Architecture review approved")
        
        # Context-specific gates
        if context.constraints.quality_gates:
            gates.extend(context.constraints.quality_gates)
        
        # Pattern-specific gates
        if pattern.validation_criteria:
            gates.extend(pattern.validation_criteria)
        
        return list(set(gates))  # Remove duplicates
    
    def _generate_success_criteria(self, pattern: Pattern, context: IssueContext) -> List[str]:
        """Generate success criteria for the implementation."""
        criteria = []
        
        # Basic success criteria
        criteria.append(f"Issue {context.issue_id} requirements fully implemented")
        criteria.append(f"Pattern '{pattern.name}' correctly applied")
        criteria.append("All quality gates passed")
        criteria.append("Implementation validated and tested")
        
        # Context-specific criteria
        if context.constraints.performance_requirements:
            criteria.append("Performance requirements met or exceeded")
        
        if context.constraints.compliance_requirements:
            criteria.append("All compliance requirements satisfied")
        
        # Pattern-specific criteria
        if pattern.success_rate > 0.8:
            criteria.append("Implementation follows proven high-success pattern")
        
        return criteria
    
    # Helper methods
    
    def _estimate_step_time(self, description: str, context: IssueContext) -> str:
        """Estimate time for a single step."""
        description_lower = description.lower()
        
        # Base estimate based on keywords
        base_time = 60  # minutes
        
        if any(word in description_lower for word in ['setup', 'configure', 'install']):
            base_time = 30
        elif any(word in description_lower for word in ['implement', 'create', 'build']):
            base_time = 120
        elif any(word in description_lower for word in ['test', 'validate', 'verify']):
            base_time = 60
        elif any(word in description_lower for word in ['design', 'plan', 'architect']):
            base_time = 90
        
        # Apply complexity multiplier
        multiplier = self.complexity_multipliers.get(context.complexity, 1.0)
        base_time = int(base_time * multiplier)
        
        # Apply tech stack multiplier
        if context.tech_stack and context.tech_stack.primary_language:
            tech_multiplier = self.tech_stack_multipliers.get(
                context.tech_stack.primary_language, 1.0)
            base_time = int(base_time * tech_multiplier)
        
        # Convert to human-readable format
        if base_time < 60:
            return f"{base_time} minutes"
        else:
            hours = base_time / 60
            return f"{hours:.1f} hours"
    
    def _determine_task_complexity(self, description: str, context_complexity: str) -> str:
        """Determine complexity of individual task."""
        description_lower = description.lower()
        
        # Simple tasks
        if any(word in description_lower for word in ['setup', 'configure', 'install', 'update']):
            return 'low'
        
        # Complex tasks
        elif any(word in description_lower for word in ['architect', 'design system', 'integrate', 'optimize']):
            return 'high'
        
        # Medium complexity by default, but consider context
        elif context_complexity in ['high', 'very-high']:
            return 'medium'
        else:
            return 'low'
    
    def _extract_task_validation(self, step: Any, pattern: Pattern) -> List[str]:
        """Extract validation criteria for a task from step or pattern."""
        criteria = []
        
        if isinstance(step, dict) and 'validation' in step:
            criteria.extend(step['validation'])
        
        # Add default validation based on task type
        step_str = str(step).lower()
        if 'implement' in step_str:
            criteria.append("Implementation complete and functional")
        elif 'test' in step_str:
            criteria.append("All tests passing")
        elif 'design' in step_str:
            criteria.append("Design reviewed and approved")
        
        return criteria
    
    def _extract_task_code_examples(self, step: Any, pattern: Pattern) -> List[Dict[str, str]]:
        """Extract code examples for a task from step or pattern."""
        examples = []
        
        if isinstance(step, dict) and 'code_examples' in step:
            examples.extend(step['code_examples'])
        
        # Use pattern code examples as reference
        if pattern.code_examples:
            examples.extend(pattern.code_examples[:2])  # Limit to 2 examples
        
        return examples
    
    def _estimate_testing_time(self, pattern: Pattern, context: IssueContext, test_type: str) -> str:
        """Estimate time needed for testing based on pattern and context."""
        base_times = {
            'unit': 90,      # minutes
            'integration': 150,
            'validation': 60
        }
        
        base_time = base_times.get(test_type, 90)
        
        # Scale based on complexity
        multiplier = self.complexity_multipliers.get(context.complexity, 1.0)
        total_time = int(base_time * multiplier)
        
        # Scale based on number of implementation steps
        if pattern.implementation_steps:
            step_multiplier = min(len(pattern.implementation_steps) / 5, 2.0)  # Cap at 2x
            total_time = int(total_time * step_multiplier)
        
        # Convert to human-readable format
        if total_time < 60:
            return f"{total_time} minutes"
        else:
            hours = total_time / 60
            return f"{hours:.1f} hours"