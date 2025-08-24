"""
Recommendation Generator - Intelligent Pattern Recommendation System

This module generates useful, actionable recommendations from applicable patterns
with detailed implementation guidance, adaptation instructions, and success metrics.

Key Features:
- Context-aware recommendation generation
- Implementation step generation
- Adaptation requirement analysis
- Risk assessment and mitigation
- Success criteria definition
- Resource requirement estimation
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import knowledge system interfaces
try:
    from knowledge.interface import get_knowledge_system
    from knowledge.database.database_interface import RIFDatabase
except ImportError:
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")
    
    class RIFDatabase:
        pass

# Import pattern application core components
from knowledge.pattern_application.core import (
    Pattern, IssueContext, TechStack, ImplementationTask
)


@dataclass
class PatternRecommendation:
    """Comprehensive pattern recommendation with implementation guidance."""
    pattern_id: str
    pattern_name: str
    recommendation_strength: str  # 'strong', 'moderate', 'weak'
    confidence_score: float
    estimated_success_rate: float
    
    # Implementation guidance
    implementation_steps: List[Dict[str, Any]]
    adaptation_requirements: List[str]
    code_examples: List[Dict[str, str]]
    
    # Context analysis
    applicability_explanation: str
    technology_fit: str
    complexity_assessment: str
    
    # Risk and resource analysis
    risk_factors: List[str]
    mitigation_strategies: List[str]
    estimated_effort: str
    required_expertise: List[str]
    
    # Success metrics
    success_criteria: List[str]
    validation_steps: List[str]
    quality_gates: List[str]
    
    # Historical context
    similar_applications: List[str]
    lessons_learned: List[str]


class RecommendationGenerator:
    """
    Intelligent recommendation generator that creates actionable guidance
    from applicable patterns.
    
    This system generates comprehensive recommendations that include:
    - Detailed implementation steps
    - Context-specific adaptations
    - Risk assessments and mitigation strategies
    - Resource and effort estimations
    - Success criteria and validation steps
    - Historical precedents and lessons learned
    """
    
    def __init__(self, knowledge_system=None, database: Optional[RIFDatabase] = None):
        """Initialize the recommendation generator."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.database = database
        
        # Recommendation strength thresholds
        self.strength_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.3
        }
        
        # Effort estimation factors
        self.effort_factors = {
            'complexity_multipliers': {
                'low': 1.0,
                'medium': 1.5,
                'high': 2.5,
                'very-high': 4.0
            },
            'adaptation_multipliers': {
                'minimal': 1.0,
                'moderate': 1.3,
                'significant': 1.8,
                'extensive': 2.5
            },
            'tech_stack_multipliers': {
                'perfect_match': 1.0,
                'compatible': 1.2,
                'different': 1.8,
                'incompatible': 3.0
            }
        }
        
        # Common risk patterns
        self.risk_patterns = {
            'technology_mismatch': {
                'description': 'Technology stack differences',
                'probability': 'medium',
                'impact': 'medium'
            },
            'complexity_underestimation': {
                'description': 'Underestimating implementation complexity',
                'probability': 'high',
                'impact': 'high'
            },
            'incomplete_requirements': {
                'description': 'Missing or unclear requirements',
                'probability': 'medium',
                'impact': 'high'
            },
            'integration_challenges': {
                'description': 'Integration with existing systems',
                'probability': 'medium',
                'impact': 'medium'
            }
        }
        
        self.logger.info("Recommendation Generator initialized")
    
    def generate_recommendations(self, patterns: List[Pattern], 
                               issue_context: IssueContext) -> List[Dict[str, Any]]:
        """
        Generate useful recommendations from applicable patterns.
        
        This implements the core requirement from Issue #76 acceptance criteria:
        "Generates useful recommendations"
        
        Args:
            patterns: List of applicable patterns
            issue_context: Context for generating recommendations
            
        Returns:
            List of recommendation dictionaries with detailed guidance
        """
        self.logger.info(f"Generating recommendations for {len(patterns)} patterns")
        
        if not patterns:
            return []
        
        try:
            recommendations = []
            
            for pattern in patterns:
                recommendation = self.generate_pattern_recommendation(pattern, issue_context)
                
                # Only include recommendations that meet minimum quality standards
                if self._meets_recommendation_standards(recommendation):
                    recommendations.append(recommendation.to_dict())
            
            # Sort recommendations by strength and confidence
            recommendations.sort(
                key=lambda x: (
                    self._strength_to_numeric(x['recommendation_strength']),
                    x['confidence_score']
                ),
                reverse=True
            )
            
            self.logger.info(f"Generated {len(recommendations)} quality recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def generate_pattern_recommendation(self, pattern: Pattern, 
                                      issue_context: IssueContext) -> PatternRecommendation:
        """
        Generate comprehensive recommendation for a specific pattern.
        
        Args:
            pattern: Pattern to generate recommendation for
            issue_context: Context for the recommendation
            
        Returns:
            PatternRecommendation with detailed guidance
        """
        # Calculate recommendation strength and confidence
        strength, confidence = self._calculate_recommendation_metrics(pattern, issue_context)
        
        # Generate implementation guidance
        implementation_steps = self._generate_implementation_steps(pattern, issue_context)
        adaptation_requirements = self._identify_adaptation_requirements(pattern, issue_context)
        code_examples = self._adapt_code_examples(pattern, issue_context)
        
        # Analyze context fit
        applicability_explanation = self._generate_applicability_explanation(pattern, issue_context)
        technology_fit = self._assess_technology_fit(pattern, issue_context)
        complexity_assessment = self._assess_complexity_fit(pattern, issue_context)
        
        # Risk and resource analysis
        risk_factors = self._identify_risk_factors(pattern, issue_context)
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors)
        estimated_effort = self._estimate_implementation_effort(pattern, issue_context)
        required_expertise = self._identify_required_expertise(pattern, issue_context)
        
        # Success metrics
        success_criteria = self._define_success_criteria(pattern, issue_context)
        validation_steps = self._generate_validation_steps(pattern, issue_context)
        quality_gates = self._define_quality_gates(pattern, issue_context)
        
        # Historical context
        similar_applications = self._find_similar_applications(pattern, issue_context)
        lessons_learned = self._extract_lessons_learned(pattern, issue_context)
        
        return PatternRecommendation(
            pattern_id=pattern.pattern_id,
            pattern_name=pattern.name,
            recommendation_strength=strength,
            confidence_score=confidence,
            estimated_success_rate=self._estimate_success_rate(pattern, issue_context),
            implementation_steps=implementation_steps,
            adaptation_requirements=adaptation_requirements,
            code_examples=code_examples,
            applicability_explanation=applicability_explanation,
            technology_fit=technology_fit,
            complexity_assessment=complexity_assessment,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            estimated_effort=estimated_effort,
            required_expertise=required_expertise,
            success_criteria=success_criteria,
            validation_steps=validation_steps,
            quality_gates=quality_gates,
            similar_applications=similar_applications,
            lessons_learned=lessons_learned
        )
    
    def _calculate_recommendation_metrics(self, pattern: Pattern, 
                                        issue_context: IssueContext) -> Tuple[str, float]:
        """Calculate recommendation strength and confidence."""
        # Import pattern ranker for scoring
        try:
            from .pattern_ranker import PatternRanker
            ranker = PatternRanker(self.knowledge_system, self.database)
            ranking_result = ranker.calculate_pattern_ranking(pattern, issue_context)
            
            overall_score = ranking_result.overall_score
            confidence = ranking_result.confidence_level
            
            # Determine strength based on score
            if overall_score >= self.strength_thresholds['strong']:
                strength = 'strong'
            elif overall_score >= self.strength_thresholds['moderate']:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            # Convert confidence level to numeric score
            confidence_numeric = {
                'high': 0.9,
                'medium': 0.7,
                'low': 0.5
            }.get(confidence, 0.5)
            
            return strength, confidence_numeric
            
        except Exception as e:
            self.logger.debug(f"Metrics calculation failed: {str(e)}")
            return 'moderate', 0.5
    
    def _generate_implementation_steps(self, pattern: Pattern, 
                                     issue_context: IssueContext) -> List[Dict[str, Any]]:
        """Generate detailed implementation steps adapted to context."""
        steps = []
        
        # Start with pattern's existing steps if available
        base_steps = pattern.implementation_steps or []
        
        # If no existing steps, generate basic ones
        if not base_steps:
            base_steps = self._generate_default_implementation_steps(pattern, issue_context)
        
        # Adapt steps to context
        for i, step in enumerate(base_steps):
            adapted_step = self._adapt_implementation_step(step, pattern, issue_context, i + 1)
            steps.append(adapted_step)
        
        # Add context-specific steps
        context_steps = self._generate_context_specific_steps(pattern, issue_context)
        steps.extend(context_steps)
        
        return steps
    
    def _identify_adaptation_requirements(self, pattern: Pattern, 
                                        issue_context: IssueContext) -> List[str]:
        """Identify what adaptations are needed for the pattern."""
        adaptations = []
        
        # Technology adaptations
        if pattern.tech_stack and issue_context.tech_stack:
            tech_adaptations = self._identify_tech_adaptations(
                pattern.tech_stack, issue_context.tech_stack
            )
            adaptations.extend(tech_adaptations)
        
        # Complexity adaptations
        complexity_adaptations = self._identify_complexity_adaptations(
            pattern.complexity, issue_context.complexity
        )
        adaptations.extend(complexity_adaptations)
        
        # Domain adaptations
        domain_adaptations = self._identify_domain_adaptations(
            pattern.domain, issue_context.domain
        )
        adaptations.extend(domain_adaptations)
        
        # Constraint adaptations
        constraint_adaptations = self._identify_constraint_adaptations(
            pattern, issue_context
        )
        adaptations.extend(constraint_adaptations)
        
        return adaptations
    
    def _adapt_code_examples(self, pattern: Pattern, 
                           issue_context: IssueContext) -> List[Dict[str, str]]:
        """Adapt code examples to the current context."""
        adapted_examples = []
        
        for example in pattern.code_examples:
            adapted_example = self._adapt_single_code_example(example, pattern, issue_context)
            adapted_examples.append(adapted_example)
        
        # Generate additional examples if none exist
        if not adapted_examples:
            additional_examples = self._generate_context_code_examples(pattern, issue_context)
            adapted_examples.extend(additional_examples)
        
        return adapted_examples
    
    def _generate_applicability_explanation(self, pattern: Pattern, 
                                          issue_context: IssueContext) -> str:
        """Generate explanation of why this pattern is applicable."""
        explanations = []
        
        # Semantic relevance
        if self._has_semantic_relevance(pattern, issue_context):
            explanations.append(f"Pattern '{pattern.name}' addresses similar requirements")
        
        # Technology compatibility
        tech_explanation = self._explain_technology_compatibility(pattern, issue_context)
        if tech_explanation:
            explanations.append(tech_explanation)
        
        # Historical success
        if pattern.success_rate > 0.7:
            explanations.append(f"High historical success rate ({pattern.success_rate:.1%})")
        
        # Domain relevance
        if pattern.domain == issue_context.domain:
            explanations.append(f"Perfect domain match ({pattern.domain})")
        
        # Usage frequency
        if pattern.usage_count > 5:
            explanations.append(f"Well-proven pattern (used {pattern.usage_count} times)")
        
        return ". ".join(explanations) + "." if explanations else "General applicability."
    
    def _assess_technology_fit(self, pattern: Pattern, 
                             issue_context: IssueContext) -> str:
        """Assess how well pattern fits the technology context."""
        if not pattern.tech_stack or not issue_context.tech_stack:
            return "neutral"
        
        # Language compatibility
        lang_match = (pattern.tech_stack.primary_language.lower() == 
                     issue_context.tech_stack.primary_language.lower())
        
        # Framework overlap
        pattern_frameworks = set(fw.lower() for fw in pattern.tech_stack.frameworks)
        context_frameworks = set(fw.lower() for fw in issue_context.tech_stack.frameworks)
        framework_overlap = bool(pattern_frameworks.intersection(context_frameworks))
        
        if lang_match and framework_overlap:
            return "excellent"
        elif lang_match:
            return "good"
        elif framework_overlap:
            return "fair"
        else:
            return "requires_adaptation"
    
    def _assess_complexity_fit(self, pattern: Pattern, 
                             issue_context: IssueContext) -> str:
        """Assess complexity alignment."""
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            pattern_idx = complexity_levels.index(pattern.complexity)
            issue_idx = complexity_levels.index(issue_context.complexity)
            
            diff = abs(pattern_idx - issue_idx)
            
            if diff == 0:
                return "perfect_match"
            elif diff == 1:
                return "good_match"
            elif diff == 2:
                return "moderate_mismatch"
            else:
                return "significant_mismatch"
                
        except ValueError:
            return "unknown"
    
    def _identify_risk_factors(self, pattern: Pattern, 
                             issue_context: IssueContext) -> List[str]:
        """Identify potential risks in applying this pattern."""
        risks = []
        
        # Technology mismatch risks
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        if tech_fit in ["requires_adaptation", "fair"]:
            risks.append("Technology stack adaptation required")
        
        # Complexity risks
        complexity_fit = self._assess_complexity_fit(pattern, issue_context)
        if complexity_fit in ["moderate_mismatch", "significant_mismatch"]:
            risks.append("Complexity mismatch may require significant modifications")
        
        # Low success rate risks
        if pattern.success_rate < 0.5:
            risks.append("Pattern has limited historical success data")
        
        # Incomplete pattern risks
        if not pattern.validation_criteria:
            risks.append("Pattern lacks validation criteria")
        
        if not pattern.code_examples:
            risks.append("Pattern lacks concrete implementation examples")
        
        # Domain mismatch risks
        if pattern.domain != issue_context.domain and pattern.domain != 'general':
            risks.append(f"Pattern designed for {pattern.domain} domain, applying to {issue_context.domain}")
        
        return risks
    
    def _generate_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Generate mitigation strategies for identified risks."""
        mitigations = []
        
        for risk in risk_factors:
            if "technology stack" in risk.lower():
                mitigations.append("Create technology stack adaptation plan before implementation")
                mitigations.append("Set up proof-of-concept to validate technical feasibility")
            
            elif "complexity mismatch" in risk.lower():
                mitigations.append("Break down implementation into smaller, manageable phases")
                mitigations.append("Include additional complexity assessment and planning time")
            
            elif "success data" in risk.lower():
                mitigations.append("Implement comprehensive testing and validation")
                mitigations.append("Create detailed rollback plan")
            
            elif "validation criteria" in risk.lower():
                mitigations.append("Define custom validation criteria for this implementation")
            
            elif "implementation examples" in risk.lower():
                mitigations.append("Research and create implementation examples before starting")
            
            elif "domain" in risk.lower():
                mitigations.append("Engage domain experts for pattern adaptation review")
        
        # Add general mitigations
        mitigations.extend([
            "Regular progress reviews and adjustment points",
            "Maintain comprehensive documentation throughout implementation"
        ])
        
        return list(set(mitigations))  # Remove duplicates
    
    def _estimate_implementation_effort(self, pattern: Pattern, 
                                      issue_context: IssueContext) -> str:
        """Estimate implementation effort."""
        base_effort = 1.0
        
        # Complexity factor
        complexity_multiplier = self.effort_factors['complexity_multipliers'].get(
            issue_context.complexity, 1.5
        )
        base_effort *= complexity_multiplier
        
        # Adaptation factor
        adaptation_complexity = self._assess_adaptation_complexity(pattern, issue_context)
        adaptation_multiplier = self.effort_factors['adaptation_multipliers'].get(
            adaptation_complexity, 1.3
        )
        base_effort *= adaptation_multiplier
        
        # Technology factor
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        tech_multiplier = self.effort_factors['tech_stack_multipliers'].get(
            tech_fit, 1.2
        )
        base_effort *= tech_multiplier
        
        # Convert to effort categories
        if base_effort <= 1.5:
            return "low (1-3 days)"
        elif base_effort <= 3.0:
            return "medium (3-7 days)"
        elif base_effort <= 6.0:
            return "high (1-2 weeks)"
        else:
            return "very_high (2+ weeks)"
    
    def _identify_required_expertise(self, pattern: Pattern, 
                                   issue_context: IssueContext) -> List[str]:
        """Identify required expertise for implementation."""
        expertise = []
        
        # Technology expertise
        if issue_context.tech_stack:
            if issue_context.tech_stack.primary_language:
                expertise.append(f"{issue_context.tech_stack.primary_language} development")
            
            for framework in issue_context.tech_stack.frameworks:
                expertise.append(f"{framework} framework")
        
        # Domain expertise
        if issue_context.domain != 'general':
            expertise.append(f"{issue_context.domain} domain knowledge")
        
        # Pattern-specific expertise
        if pattern.complexity in ['high', 'very-high']:
            expertise.append("Senior development experience")
        
        if pattern.domain in ['security', 'performance', 'scalability']:
            expertise.append(f"{pattern.domain} specialization")
        
        return list(set(expertise))
    
    def _define_success_criteria(self, pattern: Pattern, 
                               issue_context: IssueContext) -> List[str]:
        """Define success criteria for the implementation."""
        criteria = []
        
        # Use pattern's existing criteria if available
        if pattern.validation_criteria:
            criteria.extend(pattern.validation_criteria)
        
        # Add context-specific criteria
        criteria.extend([
            "Implementation passes all automated tests",
            "Code quality meets project standards",
            "Performance requirements are satisfied",
            "Security requirements are addressed"
        ])
        
        # Add complexity-specific criteria
        if issue_context.complexity in ['high', 'very-high']:
            criteria.extend([
                "Implementation is properly documented",
                "Error handling and edge cases are covered",
                "Monitoring and logging are implemented"
            ])
        
        return list(set(criteria))
    
    def _generate_validation_steps(self, pattern: Pattern, 
                                 issue_context: IssueContext) -> List[str]:
        """Generate validation steps for the implementation."""
        steps = [
            "Run comprehensive test suite",
            "Perform code quality analysis",
            "Review implementation against requirements",
            "Validate error handling and edge cases"
        ]
        
        # Add technology-specific validation
        if issue_context.tech_stack:
            if 'web' in issue_context.labels:
                steps.append("Test cross-browser compatibility")
            if 'api' in issue_context.labels:
                steps.append("Validate API contract compliance")
            if 'database' in issue_context.labels:
                steps.append("Verify database schema and migrations")
        
        # Add complexity-specific validation
        if issue_context.complexity in ['high', 'very-high']:
            steps.extend([
                "Conduct performance testing",
                "Perform security assessment",
                "Review scalability considerations"
            ])
        
        return steps
    
    def _define_quality_gates(self, pattern: Pattern, 
                            issue_context: IssueContext) -> List[str]:
        """Define quality gates for the implementation."""
        gates = [
            "Code coverage > 80%",
            "No critical security vulnerabilities",
            "All tests pass",
            "Code review approval"
        ]
        
        # Add pattern-specific gates
        if pattern.success_rate > 0.8:
            gates.append(f"Implementation matches proven pattern (success rate: {pattern.success_rate:.1%})")
        
        # Add complexity-specific gates
        if issue_context.complexity in ['high', 'very-high']:
            gates.extend([
                "Performance benchmarks met",
                "Documentation completeness review",
                "Architectural review approval"
            ])
        
        return gates
    
    def _find_similar_applications(self, pattern: Pattern, 
                                 issue_context: IssueContext) -> List[str]:
        """Find similar applications of this pattern."""
        # This would query historical data for similar pattern applications
        # For now, return placeholder based on usage count
        applications = []
        
        if pattern.usage_count > 0:
            applications.append(f"Pattern successfully used {pattern.usage_count} times")
        
        if pattern.success_rate > 0.7:
            applications.append(f"High success rate ({pattern.success_rate:.1%}) in similar contexts")
        
        return applications
    
    def _extract_lessons_learned(self, pattern: Pattern, 
                               issue_context: IssueContext) -> List[str]:
        """Extract lessons learned from similar applications."""
        lessons = []
        
        # General lessons based on pattern characteristics
        if pattern.complexity in ['high', 'very-high']:
            lessons.append("Invest time in thorough planning and design")
            lessons.append("Consider implementing in phases")
        
        if pattern.success_rate < 0.6:
            lessons.append("Pay extra attention to validation and testing")
        
        # Technology-specific lessons
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        if tech_fit == "requires_adaptation":
            lessons.append("Technology adaptation may take longer than expected")
        
        return lessons
    
    def _meets_recommendation_standards(self, recommendation: PatternRecommendation) -> bool:
        """Check if recommendation meets quality standards."""
        # Minimum confidence threshold
        if recommendation.confidence_score < 0.3:
            return False
        
        # Must have implementation steps
        if not recommendation.implementation_steps:
            return False
        
        # Must have success criteria
        if not recommendation.success_criteria:
            return False
        
        return True
    
    def _strength_to_numeric(self, strength: str) -> float:
        """Convert strength string to numeric for sorting."""
        return {
            'strong': 3.0,
            'moderate': 2.0,
            'weak': 1.0
        }.get(strength, 0.0)
    
    # Helper methods for implementation step generation
    def _generate_default_implementation_steps(self, pattern: Pattern, 
                                             issue_context: IssueContext) -> List[Dict[str, Any]]:
        """Generate default implementation steps when pattern has none."""
        return [
            {
                "title": "Analysis and Planning",
                "description": "Analyze requirements and plan implementation approach",
                "estimated_time": "2-4 hours"
            },
            {
                "title": "Setup and Configuration",
                "description": "Set up development environment and configuration",
                "estimated_time": "1-2 hours"
            },
            {
                "title": "Core Implementation",
                "description": f"Implement {pattern.name} pattern",
                "estimated_time": "4-8 hours"
            },
            {
                "title": "Testing and Validation",
                "description": "Test implementation and validate against requirements",
                "estimated_time": "2-4 hours"
            }
        ]
    
    def _adapt_implementation_step(self, step: Dict[str, Any], pattern: Pattern, 
                                 issue_context: IssueContext, step_number: int) -> Dict[str, Any]:
        """Adapt a single implementation step to context."""
        adapted_step = step.copy()
        
        # Add step number
        adapted_step['step_number'] = step_number
        
        # Add context-specific details
        if issue_context.tech_stack and issue_context.tech_stack.primary_language:
            language = issue_context.tech_stack.primary_language
            if 'description' in adapted_step:
                adapted_step['description'] += f" (using {language})"
        
        # Add risk considerations if high complexity
        if issue_context.complexity in ['high', 'very-high']:
            adapted_step['risk_considerations'] = [
                "Review step carefully for edge cases",
                "Consider breaking into smaller sub-steps"
            ]
        
        return adapted_step
    
    def _generate_context_specific_steps(self, pattern: Pattern, 
                                       issue_context: IssueContext) -> List[Dict[str, Any]]:
        """Generate additional context-specific steps."""
        steps = []
        
        # Add integration step for complex issues
        if issue_context.complexity in ['high', 'very-high']:
            steps.append({
                "step_number": len(pattern.implementation_steps or []) + 1,
                "title": "Integration and System Testing",
                "description": "Integrate with existing systems and perform end-to-end testing",
                "estimated_time": "2-6 hours"
            })
        
        # Add deployment step if needed
        if 'deployment' in issue_context.labels or 'production' in issue_context.labels:
            steps.append({
                "step_number": len(steps) + len(pattern.implementation_steps or []) + 1,
                "title": "Deployment Preparation",
                "description": "Prepare for production deployment",
                "estimated_time": "1-3 hours"
            })
        
        return steps
    
    # Additional helper methods would continue here...
    def _identify_tech_adaptations(self, pattern_tech: TechStack, 
                                 context_tech: TechStack) -> List[str]:
        """Identify technology-specific adaptations needed."""
        adaptations = []
        
        if pattern_tech.primary_language != context_tech.primary_language:
            adaptations.append(f"Language adaptation: {pattern_tech.primary_language} → {context_tech.primary_language}")
        
        pattern_frameworks = set(pattern_tech.frameworks)
        context_frameworks = set(context_tech.frameworks)
        if not pattern_frameworks.intersection(context_frameworks):
            adaptations.append("Framework adaptation required")
        
        return adaptations
    
    def _assess_adaptation_complexity(self, pattern: Pattern, 
                                    issue_context: IssueContext) -> str:
        """Assess complexity of pattern adaptation."""
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        complexity_fit = self._assess_complexity_fit(pattern, issue_context)
        
        if tech_fit == "excellent" and complexity_fit == "perfect_match":
            return "minimal"
        elif tech_fit in ["good", "fair"] and complexity_fit in ["perfect_match", "good_match"]:
            return "moderate"
        elif tech_fit == "requires_adaptation" or complexity_fit == "significant_mismatch":
            return "extensive"
        else:
            return "significant"
    
    def _has_semantic_relevance(self, pattern: Pattern, 
                              issue_context: IssueContext) -> bool:
        """Check if pattern has semantic relevance to issue."""
        pattern_text = f"{pattern.name} {pattern.description}".lower()
        issue_text = f"{issue_context.title} {issue_context.description}".lower()
        
        # Simple keyword overlap check
        pattern_words = set(pattern_text.split())
        issue_words = set(issue_text.split())
        
        overlap = len(pattern_words.intersection(issue_words))
        return overlap > 2  # Arbitrary threshold
    
    def _explain_technology_compatibility(self, pattern: Pattern, 
                                        issue_context: IssueContext) -> str:
        """Explain technology compatibility."""
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        
        explanations = {
            "excellent": "Perfect technology stack match",
            "good": "Compatible technology stack with minor differences",
            "fair": "Partially compatible technology stack",
            "requires_adaptation": "Technology stack requires adaptation",
            "neutral": "Technology compatibility unclear"
        }
        
        return explanations.get(tech_fit, "")
    
    def _estimate_success_rate(self, pattern: Pattern, 
                             issue_context: IssueContext) -> float:
        """Estimate success rate for this specific application."""
        base_rate = pattern.success_rate
        
        # Adjust based on context fit
        tech_fit = self._assess_technology_fit(pattern, issue_context)
        tech_adjustment = {
            "excellent": 0.1,
            "good": 0.05,
            "fair": 0.0,
            "requires_adaptation": -0.1,
            "neutral": 0.0
        }.get(tech_fit, 0.0)
        
        complexity_fit = self._assess_complexity_fit(pattern, issue_context)
        complexity_adjustment = {
            "perfect_match": 0.1,
            "good_match": 0.05,
            "moderate_mismatch": -0.05,
            "significant_mismatch": -0.15,
            "unknown": 0.0
        }.get(complexity_fit, 0.0)
        
        estimated_rate = base_rate + tech_adjustment + complexity_adjustment
        return max(0.0, min(1.0, estimated_rate))
    
    def _adapt_single_code_example(self, example: Dict[str, str], 
                                 pattern: Pattern, 
                                 issue_context: IssueContext) -> Dict[str, str]:
        """Adapt a single code example to context."""
        adapted = example.copy()
        
        # Add context-specific comments or modifications
        if 'code' in adapted and issue_context.tech_stack:
            language = issue_context.tech_stack.primary_language
            adapted['language'] = language
            adapted['context_note'] = f"Adapted for {language} implementation"
        
        return adapted
    
    def _generate_context_code_examples(self, pattern: Pattern, 
                                      issue_context: IssueContext) -> List[Dict[str, str]]:
        """Generate context-specific code examples."""
        examples = []
        
        if issue_context.tech_stack and issue_context.tech_stack.primary_language:
            language = issue_context.tech_stack.primary_language
            
            # Generate basic example structure
            example = {
                "title": f"Basic {pattern.name} implementation",
                "language": language,
                "description": f"Basic implementation of {pattern.name} pattern in {language}",
                "code": f"// {pattern.name} implementation placeholder\n// TODO: Implement pattern logic",
                "context_note": "Template for implementation - customize as needed"
            }
            
            examples.append(example)
        
        return examples
    
    def _identify_complexity_adaptations(self, pattern_complexity: str, 
                                       issue_complexity: str) -> List[str]:
        """Identify complexity-related adaptations."""
        adaptations = []
        
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            pattern_idx = complexity_levels.index(pattern_complexity)
            issue_idx = complexity_levels.index(issue_complexity)
            
            if issue_idx > pattern_idx:
                adaptations.append(f"Scale up from {pattern_complexity} to {issue_complexity} complexity")
            elif issue_idx < pattern_idx:
                adaptations.append(f"Simplify from {pattern_complexity} to {issue_complexity} complexity")
                
        except ValueError:
            adaptations.append("Assess complexity alignment during implementation")
        
        return adaptations
    
    def _identify_domain_adaptations(self, pattern_domain: str, 
                                   issue_domain: str) -> List[str]:
        """Identify domain-related adaptations."""
        adaptations = []
        
        if pattern_domain != issue_domain and pattern_domain != 'general':
            adaptations.append(f"Adapt pattern from {pattern_domain} domain to {issue_domain} domain")
        
        return adaptations
    
    def _identify_constraint_adaptations(self, pattern: Pattern, 
                                       issue_context: IssueContext) -> List[str]:
        """Identify constraint-related adaptations."""
        adaptations = []
        
        # This would analyze issue constraints and identify adaptations needed
        # For now, return general constraint considerations
        if issue_context.constraints:
            if issue_context.constraints.timeline:
                adaptations.append("Consider timeline constraints in implementation planning")
            
            if issue_context.constraints.quality_gates:
                adaptations.append("Ensure implementation meets specified quality gates")
        
        return adaptations


# Extension of PatternRecommendation class to include to_dict method
def _add_to_dict_method():
    """Add to_dict method to PatternRecommendation class."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'recommendation_strength': self.recommendation_strength,
            'confidence_score': self.confidence_score,
            'estimated_success_rate': self.estimated_success_rate,
            'implementation_steps': self.implementation_steps,
            'adaptation_requirements': self.adaptation_requirements,
            'code_examples': self.code_examples,
            'applicability_explanation': self.applicability_explanation,
            'technology_fit': self.technology_fit,
            'complexity_assessment': self.complexity_assessment,
            'risk_factors': self.risk_factors,
            'mitigation_strategies': self.mitigation_strategies,
            'estimated_effort': self.estimated_effort,
            'required_expertise': self.required_expertise,
            'success_criteria': self.success_criteria,
            'validation_steps': self.validation_steps,
            'quality_gates': self.quality_gates,
            'similar_applications': self.similar_applications,
            'lessons_learned': self.lessons_learned
        }
    
    PatternRecommendation.to_dict = to_dict

# Add the method to the class
_add_to_dict_method()