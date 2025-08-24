"""
Pattern Adapter - Context-Aware Pattern Adaptation System

This module implements sophisticated pattern adaptation algorithms that modify
patterns to match specific issue contexts, considering technology stack,
complexity, constraints, and domain requirements.
"""

from typing import Dict, List, Any, Optional, Tuple
import copy
import re
import logging
from datetime import datetime

from .core import (
    Pattern, IssueContext, TechStack, IssueConstraints,
    AdaptationResult, AdaptationStrategy
)

logger = logging.getLogger(__name__)


class PatternAdapter:
    """
    Pattern adaptation engine with multi-factor adaptation strategies.
    
    This class implements sophisticated adaptation algorithms that can:
    - Adapt patterns across different technology stacks
    - Scale patterns for different complexity levels
    - Integrate constraints into pattern recommendations
    - Maintain pattern quality during adaptation
    """
    
    def __init__(self):
        """Initialize pattern adapter with adaptation configurations."""
        self._load_adaptation_rules()
        self._load_tech_stack_mappings()
        self._load_complexity_scaling_rules()
    
    def adapt_pattern(self, pattern: Pattern, 
                     issue_context: IssueContext) -> AdaptationResult:
        """
        Adapt a pattern to a specific issue context.
        
        This is the main entry point for pattern adaptation, orchestrating
        all adaptation strategies based on the differences between pattern
        and context requirements.
        
        Args:
            pattern: Pattern to adapt
            issue_context: Context to adapt to
            
        Returns:
            AdaptationResult with adapted pattern and metadata
        """
        logger.info(f"Adapting pattern {pattern.pattern_id} for context {issue_context.issue_id}")
        
        # Create deep copy for adaptation
        adapted_pattern = copy.deepcopy(pattern)
        changes_made = []
        adaptation_strategy = AdaptationStrategy.EXACT_MATCH
        
        # Step 1: Technology stack adaptation
        if self._needs_tech_stack_adaptation(pattern, issue_context):
            adapted_pattern, tech_changes = self._adapt_technology_stack(
                adapted_pattern, issue_context.tech_stack)
            changes_made.extend(tech_changes)
            adaptation_strategy = AdaptationStrategy.TECH_STACK_ADAPTATION
        
        # Step 2: Complexity scaling
        if pattern.complexity != issue_context.complexity:
            adapted_pattern, complexity_changes = self._adapt_complexity(
                adapted_pattern, issue_context.complexity)
            changes_made.extend(complexity_changes)
            if adaptation_strategy == AdaptationStrategy.EXACT_MATCH:
                adaptation_strategy = AdaptationStrategy.COMPLEXITY_SCALING
        
        # Step 3: Constraint integration
        if self._has_significant_constraints(issue_context.constraints):
            adapted_pattern, constraint_changes = self._adapt_constraints(
                adapted_pattern, issue_context.constraints)
            changes_made.extend(constraint_changes)
            if adaptation_strategy == AdaptationStrategy.EXACT_MATCH:
                adaptation_strategy = AdaptationStrategy.CONSTRAINT_INTEGRATION
        
        # Step 4: Domain-specific adaptations
        if pattern.domain != issue_context.domain and issue_context.domain != 'general':
            adapted_pattern, domain_changes = self._adapt_domain(
                adapted_pattern, issue_context.domain)
            changes_made.extend(domain_changes)
        
        # Set full adaptation if multiple strategies were applied
        if len([change for change in changes_made if 'Technology' in change or 
                'Complexity' in change or 'Constraint' in change]) > 1:
            adaptation_strategy = AdaptationStrategy.FULL_ADAPTATION
        
        # Calculate adaptation confidence
        confidence_score = self._calculate_adaptation_confidence(
            pattern, adapted_pattern, changes_made)
        
        # Generate adaptation notes
        adaptation_notes = self._generate_adaptation_notes(
            changes_made, pattern, issue_context)
        
        # Create adaptation result
        result = AdaptationResult(
            adapted_pattern=adapted_pattern,
            adaptation_strategy=adaptation_strategy,
            changes_made=changes_made,
            confidence_score=confidence_score,
            adaptation_notes=adaptation_notes
        )
        
        logger.info(f"Pattern adaptation completed: {len(changes_made)} changes, "
                   f"confidence: {confidence_score:.2f}")
        
        return result
    
    def _load_adaptation_rules(self):
        """Load adaptation rules and configurations."""
        self.adaptation_rules = {
            'preserve_core_logic': True,
            'maintain_pattern_intent': True,
            'prefer_conservative_changes': True,
            'validate_after_adaptation': True
        }
        
        self.change_impact_weights = {
            'language_change': 0.3,
            'framework_change': 0.2,
            'complexity_increase': 0.15,
            'constraint_addition': 0.1,
            'domain_shift': 0.1
        }
    
    def _load_tech_stack_mappings(self):
        """Load technology stack adaptation mappings."""
        self.language_mappings = {
            ('javascript', 'python'): {
                'concepts': {
                    'async/await': 'asyncio',
                    'Promise': 'Future',
                    'npm': 'pip',
                    'package.json': 'requirements.txt',
                    'jest': 'pytest',
                    'express': 'fastapi'
                },
                'patterns': {
                    'middleware': 'decorator',
                    'callback': 'generator',
                    'closure': 'nonlocal'
                }
            },
            ('python', 'javascript'): {
                'concepts': {
                    'asyncio': 'async/await',
                    'pip': 'npm',
                    'requirements.txt': 'package.json',
                    'pytest': 'jest',
                    'fastapi': 'express',
                    'django': 'express'
                },
                'patterns': {
                    'decorator': 'middleware',
                    'generator': 'callback',
                    'nonlocal': 'closure'
                }
            },
            ('javascript', 'java'): {
                'concepts': {
                    'npm': 'maven',
                    'package.json': 'pom.xml',
                    'jest': 'junit',
                    'express': 'spring boot'
                },
                'patterns': {
                    'prototype': 'interface',
                    'closure': 'inner class'
                }
            },
            ('java', 'javascript'): {
                'concepts': {
                    'maven': 'npm',
                    'pom.xml': 'package.json',
                    'junit': 'jest',
                    'spring boot': 'express'
                },
                'patterns': {
                    'interface': 'prototype',
                    'inner class': 'closure'
                }
            }
        }
        
        self.framework_adaptations = {
            'react': {
                'vue': ['component structure', 'state management', 'lifecycle methods'],
                'angular': ['component architecture', 'dependency injection']
            },
            'django': {
                'fastapi': ['async support', 'automatic documentation', 'pydantic models'],
                'flask': ['lightweight structure', 'manual configuration']
            },
            'spring boot': {
                'express': ['middleware concept', 'route handling'],
                'fastapi': ['automatic validation', 'async support']
            }
        }
    
    def _load_complexity_scaling_rules(self):
        """Load complexity scaling adaptation rules."""
        self.complexity_scaling = {
            'low': {
                'max_steps': 3,
                'max_files': 2,
                'max_dependencies': 1,
                'validation_requirements': ['basic testing'],
                'documentation_level': 'minimal'
            },
            'medium': {
                'max_steps': 6,
                'max_files': 5,
                'max_dependencies': 3,
                'validation_requirements': ['unit tests', 'integration tests'],
                'documentation_level': 'standard'
            },
            'high': {
                'max_steps': 12,
                'max_files': 15,
                'max_dependencies': 8,
                'validation_requirements': ['comprehensive testing', 'code review'],
                'documentation_level': 'detailed'
            },
            'very-high': {
                'max_steps': 20,
                'max_files': 30,
                'max_dependencies': 15,
                'validation_requirements': [
                    'extensive testing', 'architecture review', 'security review'
                ],
                'documentation_level': 'comprehensive'
            }
        }
    
    def _needs_tech_stack_adaptation(self, pattern: Pattern, 
                                   context: IssueContext) -> bool:
        """Check if pattern needs technology stack adaptation."""
        if not pattern.tech_stack or not context.tech_stack:
            return False
        
        needs_adaptation = (
            pattern.tech_stack.primary_language != context.tech_stack.primary_language or
            set(pattern.tech_stack.frameworks or []) != set(context.tech_stack.frameworks or []) or
            set(pattern.tech_stack.databases or []) != set(context.tech_stack.databases or [])
        )
        
        return needs_adaptation
    
    def _adapt_technology_stack(self, pattern: Pattern, 
                              target_tech_stack: TechStack) -> Tuple[Pattern, List[str]]:
        """
        Adapt pattern for different technology stack.
        
        This method performs sophisticated technology adaptation including:
        - Language-specific concept mapping
        - Framework pattern translation
        - Tool and dependency updates
        - Code example adaptation
        """
        changes_made = []
        
        original_tech = pattern.tech_stack
        if not original_tech:
            pattern.tech_stack = target_tech_stack
            changes_made.append("Technology stack: none -> complete stack assignment")
            return pattern, changes_made
        
        # Primary language adaptation
        if original_tech.primary_language != target_tech_stack.primary_language:
            old_lang = original_tech.primary_language
            new_lang = target_tech_stack.primary_language
            
            # Adapt implementation steps
            pattern.implementation_steps = self._adapt_implementation_steps_for_language(
                pattern.implementation_steps, old_lang, new_lang)
            
            # Adapt code examples
            pattern.code_examples = self._adapt_code_examples_for_language(
                pattern.code_examples, old_lang, new_lang)
            
            # Update validation criteria
            pattern.validation_criteria = self._adapt_validation_for_language(
                pattern.validation_criteria, old_lang, new_lang)
            
            changes_made.append(f"Technology stack: {old_lang} -> {new_lang}")
        
        # Framework adaptation
        if set(original_tech.frameworks or []) != set(target_tech_stack.frameworks or []):
            old_frameworks = original_tech.frameworks or []
            new_frameworks = target_tech_stack.frameworks or []
            
            # Adapt framework-specific recommendations
            pattern.implementation_steps = self._adapt_implementation_steps_for_frameworks(
                pattern.implementation_steps, old_frameworks, new_frameworks)
            
            changes_made.append(f"Frameworks: {old_frameworks} -> {new_frameworks}")
        
        # Database adaptation
        if set(original_tech.databases or []) != set(target_tech_stack.databases or []):
            old_dbs = original_tech.databases or []
            new_dbs = target_tech_stack.databases or []
            
            # Adapt database-specific steps
            pattern.implementation_steps = self._adapt_implementation_steps_for_databases(
                pattern.implementation_steps, old_dbs, new_dbs)
            
            changes_made.append(f"Databases: {old_dbs} -> {new_dbs}")
        
        # Update pattern tech stack
        pattern.tech_stack = target_tech_stack
        
        return pattern, changes_made
    
    def _adapt_complexity(self, pattern: Pattern, 
                         target_complexity: str) -> Tuple[Pattern, List[str]]:
        """
        Adapt pattern for different complexity level.
        
        This method scales patterns up or down based on complexity requirements,
        adjusting implementation steps, validation requirements, and documentation.
        """
        changes_made = []
        original_complexity = pattern.complexity
        
        if original_complexity == target_complexity:
            return pattern, changes_made
        
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            original_idx = complexity_levels.index(original_complexity)
            target_idx = complexity_levels.index(target_complexity)
            
            if target_idx > original_idx:
                # Scaling up complexity
                pattern, scale_up_changes = self._scale_up_complexity(
                    pattern, original_complexity, target_complexity)
                changes_made.extend(scale_up_changes)
            else:
                # Scaling down complexity
                pattern, scale_down_changes = self._scale_down_complexity(
                    pattern, original_complexity, target_complexity)
                changes_made.extend(scale_down_changes)
            
            pattern.complexity = target_complexity
            changes_made.append(f"Complexity scaling: {original_complexity} -> {target_complexity}")
            
        except ValueError:
            changes_made.append(f"Unknown complexity level, applied default scaling")
        
        return pattern, changes_made
    
    def _adapt_constraints(self, pattern: Pattern, 
                          constraints: IssueConstraints) -> Tuple[Pattern, List[str]]:
        """
        Adapt pattern for specific constraints and requirements.
        
        This method modifies patterns to accommodate timeline, resource,
        quality, and compliance constraints.
        """
        changes_made = []
        
        # Timeline constraints
        if constraints.timeline:
            pattern.implementation_steps = self._adapt_for_timeline_constraints(
                pattern.implementation_steps, constraints.timeline)
            changes_made.append(f"Timeline constraint: {constraints.timeline}")
        
        # Resource constraints
        if constraints.resource_limits:
            pattern.implementation_steps = self._adapt_for_resource_constraints(
                pattern.implementation_steps, constraints.resource_limits)
            changes_made.append("Resource constraints applied")
        
        # Quality gates
        if constraints.quality_gates:
            pattern.validation_criteria.extend(constraints.quality_gates)
            pattern.validation_criteria = list(set(pattern.validation_criteria))  # Remove duplicates
            changes_made.append(f"Quality gates: {len(constraints.quality_gates)} added")
        
        # Performance requirements
        if constraints.performance_requirements:
            performance_steps = self._generate_performance_steps(
                constraints.performance_requirements)
            pattern.implementation_steps.extend(performance_steps)
            changes_made.append("Performance requirements integrated")
        
        # Compliance requirements
        if constraints.compliance_requirements:
            compliance_steps = self._generate_compliance_steps(
                constraints.compliance_requirements)
            pattern.implementation_steps.extend(compliance_steps)
            changes_made.append(f"Compliance: {', '.join(constraints.compliance_requirements)}")
        
        # Compatibility requirements
        if constraints.compatibility_requirements:
            compat_steps = self._generate_compatibility_steps(
                constraints.compatibility_requirements)
            pattern.implementation_steps.extend(compat_steps)
            changes_made.append("Compatibility requirements added")
        
        return pattern, changes_made
    
    def _adapt_domain(self, pattern: Pattern, target_domain: str) -> Tuple[Pattern, List[str]]:
        """Adapt pattern for different domain requirements."""
        changes_made = []
        
        domain_specific_adaptations = {
            'backend': {
                'focus': ['API design', 'database optimization', 'scalability'],
                'validation': ['load testing', 'security testing', 'data validation']
            },
            'frontend': {
                'focus': ['user experience', 'responsive design', 'performance'],
                'validation': ['browser testing', 'accessibility testing', 'UI testing']
            },
            'security': {
                'focus': ['threat modeling', 'secure coding', 'compliance'],
                'validation': ['security scanning', 'penetration testing', 'compliance audit']
            },
            'devops': {
                'focus': ['automation', 'monitoring', 'reliability'],
                'validation': ['infrastructure testing', 'deployment validation', 'monitoring setup']
            }
        }
        
        if target_domain in domain_specific_adaptations:
            adaptations = domain_specific_adaptations[target_domain]
            
            # Add domain-specific focus areas
            domain_steps = [f"Focus on {focus}" for focus in adaptations['focus']]
            pattern.implementation_steps.extend(domain_steps)
            
            # Add domain-specific validation
            pattern.validation_criteria.extend(adaptations['validation'])
            pattern.validation_criteria = list(set(pattern.validation_criteria))
            
            changes_made.append(f"Domain adaptation: {pattern.domain} -> {target_domain}")
            pattern.domain = target_domain
        
        return pattern, changes_made
    
    def _has_significant_constraints(self, constraints: IssueConstraints) -> bool:
        """Check if issue has significant constraints requiring adaptation."""
        return (
            constraints.timeline is not None or
            constraints.resource_limits is not None or
            len(constraints.quality_gates) > 0 or
            constraints.performance_requirements is not None or
            len(constraints.compliance_requirements) > 0 or
            len(constraints.compatibility_requirements) > 0
        )
    
    def _calculate_adaptation_confidence(self, original: Pattern, 
                                       adapted: Pattern, changes: List[str]) -> float:
        """Calculate confidence score for pattern adaptation."""
        base_confidence = original.confidence if original.confidence > 0 else 0.7
        
        # Calculate impact of changes
        total_impact = 0.0
        
        for change in changes:
            for change_type, impact in self.change_impact_weights.items():
                if change_type.replace('_', ' ').lower() in change.lower():
                    total_impact += impact
                    break
        
        # Reduce confidence based on total impact
        confidence_reduction = min(total_impact, 0.6)  # Cap reduction at 60%
        adapted_confidence = base_confidence * (1 - confidence_reduction)
        
        # Ensure minimum confidence
        return max(0.1, adapted_confidence)
    
    def _generate_adaptation_notes(self, changes: List[str], 
                                 original: Pattern, context: IssueContext) -> str:
        """Generate detailed adaptation notes."""
        if not changes:
            return f"Pattern '{original.name}' applied without modifications to issue {context.issue_id}."
        
        notes = f"Pattern '{original.name}' adapted for issue {context.issue_id}:\n\n"
        
        for i, change in enumerate(changes, 1):
            notes += f"{i}. {change}\n"
        
        notes += f"\nOriginal characteristics:\n"
        notes += f"- Complexity: {original.complexity}\n"
        notes += f"- Domain: {original.domain}\n"
        
        if original.tech_stack:
            notes += f"- Language: {original.tech_stack.primary_language}\n"
            if original.tech_stack.frameworks:
                notes += f"- Frameworks: {', '.join(original.tech_stack.frameworks)}\n"
        
        notes += f"\nTarget characteristics:\n"
        notes += f"- Complexity: {context.complexity}\n"
        notes += f"- Domain: {context.domain}\n"
        
        if context.tech_stack:
            notes += f"- Language: {context.tech_stack.primary_language}\n"
            if context.tech_stack.frameworks:
                notes += f"- Frameworks: {', '.join(context.tech_stack.frameworks)}\n"
        
        notes += f"\nAdaptation preserved the core pattern logic while adjusting for context-specific requirements."
        
        return notes
    
    # Helper methods for specific adaptations
    
    def _adapt_implementation_steps_for_language(self, steps: List[Dict[str, Any]], 
                                               old_lang: str, new_lang: str) -> List[Dict[str, Any]]:
        """Adapt implementation steps for language change."""
        adapted_steps = []
        
        mapping_key = (old_lang, new_lang)
        mappings = self.language_mappings.get(mapping_key, {})
        
        for step in steps:
            adapted_step = copy.deepcopy(step)
            
            # Adapt step description
            if isinstance(adapted_step, dict) and 'description' in adapted_step:
                description = adapted_step['description']
                
                # Apply concept mappings
                for old_concept, new_concept in mappings.get('concepts', {}).items():
                    description = re.sub(
                        r'\b' + re.escape(old_concept) + r'\b',
                        new_concept,
                        description,
                        flags=re.IGNORECASE
                    )
                
                adapted_step['description'] = description
            elif isinstance(adapted_step, str):
                description = adapted_step
                for old_concept, new_concept in mappings.get('concepts', {}).items():
                    description = re.sub(
                        r'\b' + re.escape(old_concept) + r'\b',
                        new_concept,
                        description,
                        flags=re.IGNORECASE
                    )
                adapted_step = description
            
            adapted_steps.append(adapted_step)
        
        return adapted_steps
    
    def _adapt_code_examples_for_language(self, examples: List[Dict[str, str]], 
                                        old_lang: str, new_lang: str) -> List[Dict[str, str]]:
        """Adapt code examples for language change."""
        adapted_examples = []
        
        for example in examples:
            adapted_example = copy.deepcopy(example)
            
            # Update language field
            if 'language' in adapted_example:
                adapted_example['language'] = new_lang
            
            # Add adaptation note
            adapted_example['adaptation_note'] = f"Adapted from {old_lang} to {new_lang}"
            
            adapted_examples.append(adapted_example)
        
        return adapted_examples
    
    def _adapt_validation_for_language(self, criteria: List[str], 
                                     old_lang: str, new_lang: str) -> List[str]:
        """Adapt validation criteria for language change."""
        adapted_criteria = []
        
        language_test_mappings = {
            'javascript': 'jest',
            'python': 'pytest',
            'java': 'junit',
            'go': 'go test'
        }
        
        for criterion in criteria:
            adapted_criterion = criterion
            
            # Update test framework references
            if old_lang in language_test_mappings and new_lang in language_test_mappings:
                old_framework = language_test_mappings[old_lang]
                new_framework = language_test_mappings[new_lang]
                adapted_criterion = criterion.replace(old_framework, new_framework)
            
            adapted_criteria.append(adapted_criterion)
        
        return adapted_criteria
    
    def _scale_up_complexity(self, pattern: Pattern, 
                           from_complexity: str, to_complexity: str) -> Tuple[Pattern, List[str]]:
        """Scale pattern up to higher complexity."""
        changes = []
        
        target_rules = self.complexity_scaling[to_complexity]
        
        # Add more implementation steps if needed
        current_steps = len(pattern.implementation_steps)
        target_steps = target_rules['max_steps']
        
        if current_steps < target_steps:
            additional_steps = [
                {"description": "Add comprehensive error handling"},
                {"description": "Implement logging and monitoring"},
                {"description": "Add performance optimization"},
                {"description": "Implement scalability considerations"},
                {"description": "Add security measures"},
                {"description": "Create comprehensive documentation"}
            ]
            
            needed_steps = min(target_steps - current_steps, len(additional_steps))
            pattern.implementation_steps.extend(additional_steps[:needed_steps])
            changes.append(f"Added {needed_steps} implementation steps for higher complexity")
        
        # Enhance validation criteria
        enhanced_validation = target_rules['validation_requirements']
        for validation in enhanced_validation:
            if validation not in pattern.validation_criteria:
                pattern.validation_criteria.append(validation)
        
        changes.append("Enhanced validation requirements")
        
        return pattern, changes
    
    def _scale_down_complexity(self, pattern: Pattern, 
                             from_complexity: str, to_complexity: str) -> Tuple[Pattern, List[str]]:
        """Scale pattern down to lower complexity."""
        changes = []
        
        target_rules = self.complexity_scaling[to_complexity]
        
        # Reduce implementation steps if needed
        current_steps = len(pattern.implementation_steps)
        target_steps = target_rules['max_steps']
        
        if current_steps > target_steps:
            # Keep the most essential steps
            pattern.implementation_steps = pattern.implementation_steps[:target_steps]
            removed_steps = current_steps - target_steps
            changes.append(f"Simplified implementation: removed {removed_steps} steps")
        
        # Simplify validation criteria
        simplified_validation = target_rules['validation_requirements']
        pattern.validation_criteria = [
            criterion for criterion in pattern.validation_criteria 
            if any(simple in criterion.lower() for simple in simplified_validation)
        ]
        changes.append("Simplified validation requirements")
        
        return pattern, changes