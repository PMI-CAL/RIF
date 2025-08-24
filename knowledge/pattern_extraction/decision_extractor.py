"""
Decision Pattern Extractor - Text mining and decision analysis.

This module extracts decision patterns from architectural decisions, issue comments,
design documents, and trade-off analyses using natural language processing
and structured decision analysis.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter, defaultdict
import statistics

from .discovery_engine import ExtractedPattern, PatternSignature


@dataclass
class DecisionContext:
    """Represents the context of a decision."""
    problem: str
    constraints: List[str]
    stakeholders: List[str]
    timeline: Optional[str]
    impact_scope: str
    risk_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionAlternative:
    """Represents an alternative considered in a decision."""
    name: str
    description: str
    pros: List[str]
    cons: List[str]
    cost_estimate: Optional[str]
    risk_assessment: Optional[str]
    feasibility: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionOutcome:
    """Represents the outcome and consequences of a decision."""
    chosen_alternative: str
    rationale: str
    expected_benefits: List[str]
    expected_risks: List[str]
    success_criteria: List[str]
    monitoring_approach: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionPattern:
    """Represents a complete decision pattern."""
    decision_id: str
    title: str
    decision_type: str
    context: DecisionContext
    alternatives: List[DecisionAlternative]
    outcome: DecisionOutcome
    success_rate: Optional[float] = None
    lessons_learned: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['context'] = self.context.to_dict()
        data['alternatives'] = [alt.to_dict() for alt in self.alternatives]
        data['outcome'] = self.outcome.to_dict()
        return data


class DecisionPatternExtractor:
    """
    Extracts decision patterns from text using NLP and structured analysis.
    
    This extractor analyzes decision records, issue discussions, architectural
    choices, and design documents to identify reusable decision patterns and
    trade-off frameworks.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Decision type patterns
        self.decision_types = {
            'architectural': self._detect_architectural_decision,
            'technical': self._detect_technical_decision,
            'process': self._detect_process_decision,
            'strategic': self._detect_strategic_decision,
            'tactical': self._detect_tactical_decision,
            'design': self._detect_design_decision,
            'implementation': self._detect_implementation_decision,
            'quality': self._detect_quality_decision
        }
        
        # Decision framework patterns
        self.frameworks = {
            'trade_off_analysis': self._detect_trade_off_analysis,
            'cost_benefit': self._detect_cost_benefit,
            'risk_assessment': self._detect_risk_assessment,
            'stakeholder_analysis': self._detect_stakeholder_analysis,
            'constraint_analysis': self._detect_constraint_analysis,
            'impact_analysis': self._detect_impact_analysis,
            'alternative_evaluation': self._detect_alternative_evaluation
        }
        
        # Text patterns for decision extraction
        self.text_patterns = {
            'problem_indicators': [
                r'problem[:\s]*(.*?)(?:\n|$)',
                r'issue[:\s]*(.*?)(?:\n|$)',
                r'challenge[:\s]*(.*?)(?:\n|$)',
                r'need to[:\s]*(.*?)(?:\n|$)'
            ],
            'decision_indicators': [
                r'decided? to[:\s]*(.*?)(?:\n|$)',
                r'chose[:\s]*(.*?)(?:\n|$)',
                r'selected[:\s]*(.*?)(?:\n|$)',
                r'approach[:\s]*(.*?)(?:\n|$)'
            ],
            'alternative_indicators': [
                r'alternative[:\s]*(.*?)(?:\n|$)',
                r'option[:\s]*(.*?)(?:\n|$)',
                r'consider(?:ed)?[:\s]*(.*?)(?:\n|$)',
                r'could[:\s]*(.*?)(?:\n|$)'
            ],
            'rationale_indicators': [
                r'because[:\s]*(.*?)(?:\n|$)',
                r'reason[:\s]*(.*?)(?:\n|$)',
                r'rationale[:\s]*(.*?)(?:\n|$)',
                r'since[:\s]*(.*?)(?:\n|$)'
            ],
            'consequence_indicators': [
                r'consequence[:\s]*(.*?)(?:\n|$)',
                r'result[:\s]*(.*?)(?:\n|$)',
                r'impact[:\s]*(.*?)(?:\n|$)',
                r'effect[:\s]*(.*?)(?:\n|$)'
            ]
        }
        
        # Keywords for different decision domains
        self.domain_keywords = {
            'architecture': ['architecture', 'design', 'structure', 'pattern', 'component', 'module'],
            'technology': ['technology', 'framework', 'library', 'tool', 'platform', 'stack'],
            'performance': ['performance', 'speed', 'latency', 'throughput', 'scalability', 'optimization'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'vulnerability'],
            'database': ['database', 'sql', 'nosql', 'storage', 'persistence', 'query'],
            'integration': ['integration', 'api', 'interface', 'protocol', 'communication', 'service'],
            'testing': ['testing', 'validation', 'verification', 'quality', 'coverage', 'automation'],
            'deployment': ['deployment', 'infrastructure', 'devops', 'pipeline', 'environment', 'release']
        }
    
    def extract_patterns(self, completed_issue: Dict[str, Any]) -> List[ExtractedPattern]:
        """
        Extract decision patterns from completed issue data.
        
        Args:
            completed_issue: Issue data containing decisions and discussions
            
        Returns:
            List of extracted decision patterns
        """
        patterns = []
        issue_id = completed_issue.get('issue_number', 'unknown')
        
        self.logger.info(f"Extracting decision patterns from issue #{issue_id}")
        
        try:
            # Extract from structured decision records
            if 'decisions' in completed_issue:
                patterns.extend(self._extract_from_decision_records(completed_issue))
            
            # Extract from issue discussions
            if 'comments' in completed_issue:
                patterns.extend(self._extract_from_discussions(completed_issue))
            
            # Extract from design documents
            if 'design_documents' in completed_issue:
                patterns.extend(self._extract_from_design_docs(completed_issue))
            
            # Extract from issue description and title
            patterns.extend(self._extract_from_issue_text(completed_issue))
            
            # Extract implicit decisions from code changes
            if 'code_changes' in completed_issue:
                patterns.extend(self._extract_implicit_decisions(completed_issue))
            
            self.logger.info(f"Extracted {len(patterns)} decision patterns from issue #{issue_id}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error extracting decision patterns from issue #{issue_id}: {e}")
            return []
    
    def _extract_from_decision_records(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from structured decision records."""
        patterns = []
        decisions = issue_data.get('decisions', [])
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            for i, decision in enumerate(decisions):
                # Parse structured decision record
                decision_pattern = self._parse_decision_record(decision, f"{issue_id}-{i}")
                
                if decision_pattern:
                    # Determine decision type
                    decision_type = self._classify_decision_type(decision)
                    
                    # Identify framework used
                    framework = self._identify_decision_framework(decision)
                    
                    # Create extracted pattern
                    pattern = ExtractedPattern(
                        title=f"{decision_type.title()} Decision Pattern",
                        description=decision_pattern.title or f"Decision pattern from issue #{issue_id}",
                        pattern_type='decision',
                        source=f"issue-{issue_id}-decision-{i}",
                        content={
                            'decision_pattern': decision_pattern.to_dict(),
                            'decision_type': decision_type,
                            'framework_used': framework,
                            'complexity_indicators': self._assess_decision_complexity(decision_pattern),
                            'success_factors': self._identify_success_factors(decision_pattern)
                        },
                        context={
                            'domain': self._identify_decision_domain(decision),
                            'stakeholder_count': len(decision_pattern.context.stakeholders),
                            'alternative_count': len(decision_pattern.alternatives),
                            'risk_level': decision_pattern.context.risk_level
                        },
                        signature=PatternSignature.from_pattern({
                            'title': f"{decision_type} Decision Pattern",
                            'description': decision_pattern.title or "Decision pattern",
                            'complexity': decision_pattern.context.risk_level,
                            'domain': self._identify_decision_domain(decision)
                        }),
                        extraction_method='structured_decision_analysis',
                        confidence=0.9,
                        created_at=datetime.now()
                    )
                    
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting from decision records: {e}")
        
        return patterns
    
    def _extract_from_discussions(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract decision patterns from issue discussions and comments."""
        patterns = []
        comments = issue_data.get('comments', [])
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            # Combine all comments for analysis
            discussion_text = ' '.join([
                comment.get('body', '') for comment in comments
                if comment.get('body')
            ])
            
            if not discussion_text:
                return patterns
            
            # Extract decisions from discussion
            implicit_decisions = self._extract_implicit_decisions_from_text(discussion_text)
            
            for i, decision_data in enumerate(implicit_decisions):
                decision_type = self._classify_decision_from_text(decision_data['decision_text'])
                
                pattern = ExtractedPattern(
                    title=f"Discussion-Based {decision_type.title()} Decision",
                    description=f"Decision pattern extracted from issue discussion",
                    pattern_type='decision',
                    source=f"issue-{issue_id}-discussion-{i}",
                    content={
                        'decision_text': decision_data['decision_text'],
                        'context_text': decision_data.get('context', ''),
                        'decision_type': decision_type,
                        'confidence_indicators': decision_data.get('confidence_indicators', []),
                        'stakeholder_involvement': self._analyze_stakeholder_involvement(comments)
                    },
                    context={
                        'discussion_length': len(discussion_text),
                        'participant_count': len(set(c.get('author', 'unknown') for c in comments)),
                        'decision_confidence': decision_data.get('confidence', 0.5),
                        'domain': self._identify_domain_from_text(decision_data['decision_text'])
                    },
                    signature=PatternSignature.from_pattern({
                        'title': f"Discussion {decision_type} Decision",
                        'description': decision_data['decision_text'][:100],
                        'complexity': 'medium',
                        'domain': self._identify_domain_from_text(decision_data['decision_text'])
                    }),
                    extraction_method='discussion_analysis',
                    confidence=decision_data.get('confidence', 0.5),
                    created_at=datetime.now()
                )
                
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting from discussions: {e}")
        
        return patterns
    
    def _extract_from_design_docs(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract decision patterns from design documents."""
        patterns = []
        design_docs = issue_data.get('design_documents', [])
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            for i, doc in enumerate(design_docs):
                if isinstance(doc, dict) and 'content' in doc:
                    doc_content = doc['content']
                elif isinstance(doc, str):
                    doc_content = doc
                else:
                    continue
                
                # Extract design decisions from document
                design_decisions = self._extract_design_decisions_from_text(doc_content)
                
                for j, decision in enumerate(design_decisions):
                    pattern = ExtractedPattern(
                        title=f"Design Decision Pattern",
                        description=f"Design decision from documentation",
                        pattern_type='decision',
                        source=f"issue-{issue_id}-design-{i}-{j}",
                        content={
                            'design_decision': decision,
                            'document_context': doc.get('title', 'Design Document') if isinstance(doc, dict) else f"Document {i}",
                            'decision_rationale': decision.get('rationale', ''),
                            'technical_constraints': decision.get('constraints', []),
                            'architectural_impact': decision.get('impact', 'medium')
                        },
                        context={
                            'document_type': 'design',
                            'technical_depth': self._assess_technical_depth(doc_content),
                            'design_complexity': decision.get('complexity', 'medium'),
                            'domain': self._identify_domain_from_text(doc_content)
                        },
                        signature=PatternSignature.from_pattern({
                            'title': 'Design Decision Pattern',
                            'description': decision.get('summary', 'Design decision'),
                            'complexity': decision.get('complexity', 'medium'),
                            'domain': 'design'
                        }),
                        extraction_method='design_document_analysis',
                        confidence=0.7,
                        created_at=datetime.now()
                    )
                    
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting from design docs: {e}")
        
        return patterns
    
    def _extract_from_issue_text(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract decision patterns from issue title and description."""
        patterns = []
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            title = issue_data.get('title', '')
            body = issue_data.get('body', '')
            combined_text = f"{title}\n{body}"
            
            if not combined_text.strip():
                return patterns
            
            # Look for explicit decisions in issue text
            decisions = self._extract_decisions_from_text(combined_text)
            
            for i, decision in enumerate(decisions):
                decision_type = self._classify_decision_from_text(decision['text'])
                
                pattern = ExtractedPattern(
                    title=f"Issue-Level {decision_type.title()} Decision",
                    description=f"Decision pattern from issue #{issue_id} description",
                    pattern_type='decision',
                    source=f"issue-{issue_id}-text-{i}",
                    content={
                        'decision_statement': decision['text'],
                        'context_clues': decision.get('context_clues', []),
                        'decision_type': decision_type,
                        'scope': self._determine_decision_scope(combined_text),
                        'urgency': self._assess_decision_urgency(combined_text)
                    },
                    context={
                        'issue_complexity': issue_data.get('complexity', 'medium'),
                        'issue_priority': self._extract_priority(combined_text),
                        'scope': decision.get('scope', 'local'),
                        'domain': self._identify_domain_from_text(combined_text)
                    },
                    signature=PatternSignature.from_pattern({
                        'title': f"{decision_type} Decision from Issue",
                        'description': decision['text'][:100],
                        'complexity': issue_data.get('complexity', 'medium'),
                        'domain': self._identify_domain_from_text(combined_text)
                    }),
                    extraction_method='issue_text_analysis',
                    confidence=0.6,
                    created_at=datetime.now()
                )
                
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting from issue text: {e}")
        
        return patterns
    
    def _extract_implicit_decisions(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract implicit decisions from code changes and implementation choices."""
        patterns = []
        code_changes = issue_data.get('code_changes', {})
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            for file_path, changes in code_changes.items():
                # Analyze implementation choices
                impl_decisions = self._analyze_implementation_decisions(changes, file_path)
                
                for i, decision in enumerate(impl_decisions):
                    pattern = ExtractedPattern(
                        title=f"Implementation Decision Pattern",
                        description=f"Implicit decision from code implementation",
                        pattern_type='decision',
                        source=f"issue-{issue_id}-impl-{i}",
                        content={
                            'implementation_choice': decision['choice'],
                            'file_context': file_path,
                            'technical_rationale': decision.get('rationale', 'Inferred from implementation'),
                            'alternatives_considered': decision.get('alternatives', []),
                            'code_evidence': decision.get('evidence', [])
                        },
                        context={
                            'file_type': self._get_file_type(file_path),
                            'implementation_scope': decision.get('scope', 'local'),
                            'technical_debt': decision.get('technical_debt', 'low'),
                            'domain': self._identify_domain_from_path(file_path)
                        },
                        signature=PatternSignature.from_pattern({
                            'title': 'Implementation Decision',
                            'description': decision['choice'],
                            'complexity': 'medium',
                            'domain': 'implementation'
                        }),
                        extraction_method='code_analysis',
                        confidence=0.4,
                        created_at=datetime.now()
                    )
                    
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting implicit decisions: {e}")
        
        return patterns
    
    def _parse_decision_record(self, decision_data: Dict[str, Any], decision_id: str) -> Optional[DecisionPattern]:
        """Parse a structured decision record into a DecisionPattern."""
        try:
            title = decision_data.get('title', f'Decision {decision_id}')
            
            # Parse context
            context = DecisionContext(
                problem=decision_data.get('context', decision_data.get('problem', '')),
                constraints=decision_data.get('constraints', []),
                stakeholders=decision_data.get('stakeholders', []),
                timeline=decision_data.get('timeline'),
                impact_scope=decision_data.get('impact', 'medium'),
                risk_level=decision_data.get('risk_level', 'medium')
            )
            
            # Parse alternatives
            alternatives = []
            for alt_data in decision_data.get('alternatives', []):
                if isinstance(alt_data, dict):
                    alternative = DecisionAlternative(
                        name=alt_data.get('name', 'Alternative'),
                        description=alt_data.get('description', ''),
                        pros=alt_data.get('pros', []),
                        cons=alt_data.get('cons', []),
                        cost_estimate=alt_data.get('cost'),
                        risk_assessment=alt_data.get('risk'),
                        feasibility=alt_data.get('feasibility', 'medium')
                    )
                    alternatives.append(alternative)
            
            # Parse outcome
            outcome = DecisionOutcome(
                chosen_alternative=decision_data.get('decision', ''),
                rationale=decision_data.get('rationale', ''),
                expected_benefits=decision_data.get('benefits', []),
                expected_risks=decision_data.get('risks', []),
                success_criteria=decision_data.get('success_criteria', []),
                monitoring_approach=decision_data.get('monitoring')
            )
            
            return DecisionPattern(
                decision_id=decision_id,
                title=title,
                decision_type=self._classify_decision_type(decision_data),
                context=context,
                alternatives=alternatives,
                outcome=outcome,
                lessons_learned=decision_data.get('lessons_learned', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing decision record {decision_id}: {e}")
            return None
    
    def _classify_decision_type(self, decision_data: Dict[str, Any]) -> str:
        """Classify the type of decision based on content."""
        text = f"{decision_data.get('title', '')} {decision_data.get('context', '')} {decision_data.get('decision', '')}"
        
        for decision_type, detector in self.decision_types.items():
            if detector(text, decision_data):
                return decision_type
        
        return 'general'
    
    def _identify_decision_framework(self, decision_data: Dict[str, Any]) -> str:
        """Identify the decision framework used."""
        for framework, detector in self.frameworks.items():
            if detector(decision_data):
                return framework
        
        return 'informal'
    
    def _assess_decision_complexity(self, decision_pattern: DecisionPattern) -> Dict[str, Any]:
        """Assess the complexity indicators of a decision."""
        return {
            'stakeholder_complexity': len(decision_pattern.context.stakeholders),
            'alternative_complexity': len(decision_pattern.alternatives),
            'constraint_complexity': len(decision_pattern.context.constraints),
            'risk_complexity': decision_pattern.context.risk_level,
            'impact_complexity': decision_pattern.context.impact_scope
        }
    
    def _identify_success_factors(self, decision_pattern: DecisionPattern) -> List[str]:
        """Identify factors that contribute to decision success."""
        factors = []
        
        if len(decision_pattern.alternatives) > 1:
            factors.append('multiple_alternatives_considered')
        
        if decision_pattern.outcome.rationale:
            factors.append('clear_rationale')
        
        if decision_pattern.context.constraints:
            factors.append('constraints_identified')
        
        if decision_pattern.outcome.success_criteria:
            factors.append('success_criteria_defined')
        
        if decision_pattern.context.stakeholders:
            factors.append('stakeholder_involvement')
        
        return factors
    
    def _identify_decision_domain(self, decision_data: Dict[str, Any]) -> str:
        """Identify the domain of a decision."""
        text = f"{decision_data.get('title', '')} {decision_data.get('context', '')} {decision_data.get('decision', '')}"
        return self._identify_domain_from_text(text)
    
    def _identify_domain_from_text(self, text: str) -> str:
        """Identify domain from text content."""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _identify_domain_from_path(self, file_path: str) -> str:
        """Identify domain from file path."""
        path_lower = file_path.lower()
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in path_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _extract_implicit_decisions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract implicit decisions from discussion text."""
        decisions = []
        
        # Look for decision indicators in text
        for pattern in self.text_patterns['decision_indicators']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                decision_text = match.group(1).strip()
                if len(decision_text) > 10:  # Ignore very short matches
                    
                    # Look for context around the decision
                    context = self._extract_context_around_match(text, match)
                    
                    # Assess confidence based on context
                    confidence = self._assess_decision_confidence(decision_text, context)
                    
                    decisions.append({
                        'decision_text': decision_text,
                        'context': context,
                        'confidence': confidence,
                        'confidence_indicators': self._get_confidence_indicators(decision_text, context)
                    })
        
        return decisions
    
    def _extract_context_around_match(self, text: str, match) -> str:
        """Extract context around a regex match."""
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        return text[start:end]
    
    def _assess_decision_confidence(self, decision_text: str, context: str) -> float:
        """Assess confidence in a detected decision."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for explicit decision language
        explicit_indicators = ['decided', 'chose', 'selected', 'determined', 'concluded']
        if any(indicator in decision_text.lower() for indicator in explicit_indicators):
            confidence += 0.2
        
        # Boost confidence for rationale present
        rationale_indicators = ['because', 'since', 'due to', 'reason', 'rationale']
        if any(indicator in context.lower() for indicator in rationale_indicators):
            confidence += 0.2
        
        # Boost confidence for alternatives mentioned
        if 'alternative' in context.lower() or 'option' in context.lower():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_confidence_indicators(self, decision_text: str, context: str) -> List[str]:
        """Get indicators that support decision confidence."""
        indicators = []
        
        if any(word in decision_text.lower() for word in ['decided', 'chose', 'selected']):
            indicators.append('explicit_decision_language')
        
        if any(word in context.lower() for word in ['because', 'since', 'rationale']):
            indicators.append('rationale_provided')
        
        if any(word in context.lower() for word in ['alternative', 'option', 'consider']):
            indicators.append('alternatives_mentioned')
        
        return indicators
    
    def _classify_decision_from_text(self, text: str) -> str:
        """Classify decision type from text content."""
        text_lower = text.lower()
        
        # Check for architectural decisions
        if any(word in text_lower for word in ['architecture', 'design', 'structure', 'pattern']):
            return 'architectural'
        
        # Check for technical decisions
        if any(word in text_lower for word in ['technology', 'framework', 'library', 'implementation']):
            return 'technical'
        
        # Check for process decisions
        if any(word in text_lower for word in ['process', 'workflow', 'procedure', 'methodology']):
            return 'process'
        
        # Check for quality decisions
        if any(word in text_lower for word in ['testing', 'quality', 'validation', 'verification']):
            return 'quality'
        
        return 'general'
    
    def _analyze_stakeholder_involvement(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stakeholder involvement in discussions."""
        authors = [comment.get('author', 'unknown') for comment in comments if comment.get('author')]
        unique_authors = set(authors)
        
        return {
            'participant_count': len(unique_authors),
            'total_comments': len(comments),
            'average_participation': len(comments) / max(len(unique_authors), 1),
            'most_active_participant': Counter(authors).most_common(1)[0][0] if authors else 'unknown'
        }
    
    def _extract_design_decisions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract design decisions from document text."""
        decisions = []
        
        # Look for design decision patterns
        design_patterns = [
            r'design decision[:\s]*(.*?)(?:\n\n|$)',
            r'architectural choice[:\s]*(.*?)(?:\n\n|$)',
            r'approach[:\s]*(.*?)(?:\n\n|$)',
            r'solution[:\s]*(.*?)(?:\n\n|$)'
        ]
        
        for pattern in design_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                decision_content = match.group(1).strip()
                if len(decision_content) > 20:  # Filter out too short matches
                    
                    # Extract rationale if present
                    rationale = self._extract_rationale_from_text(decision_content)
                    
                    # Assess complexity
                    complexity = self._assess_text_complexity(decision_content)
                    
                    decisions.append({
                        'summary': decision_content[:100] + '...' if len(decision_content) > 100 else decision_content,
                        'full_content': decision_content,
                        'rationale': rationale,
                        'complexity': complexity,
                        'constraints': self._extract_constraints_from_text(decision_content),
                        'impact': self._assess_impact_from_text(decision_content)
                    })
        
        return decisions
    
    def _extract_rationale_from_text(self, text: str) -> str:
        """Extract rationale from decision text."""
        rationale_patterns = [
            r'because[:\s]*(.*?)(?:\n|$)',
            r'rationale[:\s]*(.*?)(?:\n|$)',
            r'reason[:\s]*(.*?)(?:\n|$)',
            r'since[:\s]*(.*?)(?:\n|$)'
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ''
    
    def _assess_text_complexity(self, text: str) -> str:
        """Assess complexity of decision text."""
        word_count = len(text.split())
        
        # Simple heuristic based on length and technical terms
        technical_terms = ['architecture', 'implementation', 'algorithm', 'optimization', 'scalability']
        technical_count = sum(1 for term in technical_terms if term in text.lower())
        
        complexity_score = word_count / 50 + technical_count
        
        if complexity_score < 2:
            return 'low'
        elif complexity_score < 5:
            return 'medium'
        else:
            return 'high'
    
    def _extract_constraints_from_text(self, text: str) -> List[str]:
        """Extract constraints mentioned in text."""
        constraints = []
        
        constraint_patterns = [
            r'constraint[:\s]*(.*?)(?:\n|$)',
            r'limitation[:\s]*(.*?)(?:\n|$)',
            r'must[:\s]*(.*?)(?:\n|$)',
            r'cannot[:\s]*(.*?)(?:\n|$)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.group(1).strip()
                if len(constraint) > 10:
                    constraints.append(constraint)
        
        return constraints
    
    def _assess_impact_from_text(self, text: str) -> str:
        """Assess impact level from text."""
        high_impact_indicators = ['critical', 'major', 'significant', 'breaking', 'fundamental']
        medium_impact_indicators = ['important', 'notable', 'affects', 'changes', 'improves']
        
        text_lower = text.lower()
        
        if any(indicator in text_lower for indicator in high_impact_indicators):
            return 'high'
        elif any(indicator in text_lower for indicator in medium_impact_indicators):
            return 'medium'
        else:
            return 'low'
    
    def _extract_decisions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract explicit decisions from text."""
        decisions = []
        
        for pattern in self.text_patterns['decision_indicators']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                decision_text = match.group(1).strip()
                if len(decision_text) > 10:
                    
                    # Extract context clues
                    context_clues = self._extract_context_clues(text, match)
                    
                    # Determine scope
                    scope = self._determine_text_scope(decision_text)
                    
                    decisions.append({
                        'text': decision_text,
                        'context_clues': context_clues,
                        'scope': scope
                    })
        
        return decisions
    
    def _extract_context_clues(self, text: str, match) -> List[str]:
        """Extract context clues around a decision."""
        clues = []
        
        # Get surrounding text
        context = self._extract_context_around_match(text, match)
        
        # Look for problem statements
        for pattern in self.text_patterns['problem_indicators']:
            if re.search(pattern, context, re.IGNORECASE):
                clues.append('problem_statement_present')
                break
        
        # Look for alternatives
        for pattern in self.text_patterns['alternative_indicators']:
            if re.search(pattern, context, re.IGNORECASE):
                clues.append('alternatives_mentioned')
                break
        
        # Look for rationale
        for pattern in self.text_patterns['rationale_indicators']:
            if re.search(pattern, context, re.IGNORECASE):
                clues.append('rationale_provided')
                break
        
        return clues
    
    def _determine_decision_scope(self, text: str) -> str:
        """Determine the scope of a decision from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['system', 'global', 'overall', 'entire']):
            return 'global'
        elif any(word in text_lower for word in ['module', 'component', 'specific', 'local']):
            return 'local'
        else:
            return 'moderate'
    
    def _determine_text_scope(self, text: str) -> str:
        """Determine scope from decision text."""
        return self._determine_decision_scope(text)
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority level from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['critical', 'urgent', 'high priority', 'asap']):
            return 'high'
        elif any(word in text_lower for word in ['important', 'medium priority', 'soon']):
            return 'medium'
        else:
            return 'low'
    
    def _assess_decision_urgency(self, text: str) -> str:
        """Assess urgency of a decision from text."""
        text_lower = text.lower()
        
        urgent_indicators = ['urgent', 'asap', 'immediate', 'critical', 'deadline']
        if any(indicator in text_lower for indicator in urgent_indicators):
            return 'high'
        
        moderate_indicators = ['soon', 'important', 'needed', 'should']
        if any(indicator in text_lower for indicator in moderate_indicators):
            return 'medium'
        
        return 'low'
    
    def _assess_technical_depth(self, text: str) -> str:
        """Assess technical depth of content."""
        technical_indicators = [
            'algorithm', 'architecture', 'implementation', 'optimization', 'scalability',
            'performance', 'database', 'api', 'framework', 'library', 'protocol'
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        if technical_count >= 5:
            return 'high'
        elif technical_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_implementation_decisions(self, changes: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
        """Analyze implementation decisions from code changes."""
        decisions = []
        
        try:
            # Analyze added/modified lines for implementation choices
            added_lines = changes.get('added_lines', [])
            
            if isinstance(added_lines, str):
                added_content = added_lines
            elif isinstance(added_lines, list):
                added_content = '\n'.join(added_lines)
            else:
                added_content = str(added_lines)
            
            # Look for implementation patterns that indicate decisions
            implementation_patterns = [
                (r'import\s+(\w+)', 'library_choice'),
                (r'class\s+(\w+)', 'design_pattern_choice'),
                (r'def\s+(\w+)', 'interface_design_choice'),
                (r'@\w+', 'decorator_choice'),
                (r'if\s+.*:', 'conditional_logic_choice'),
                (r'try\s*:', 'error_handling_choice')
            ]
            
            for pattern, choice_type in implementation_patterns:
                matches = re.finditer(pattern, added_content)
                for match in matches:
                    decision = {
                        'choice': f"{choice_type}: {match.group(0)}",
                        'rationale': f'Implementation decision inferred from code pattern',
                        'scope': 'local',
                        'evidence': [match.group(0)],
                        'alternatives': []  # Could be enhanced with more analysis
                    }
                    decisions.append(decision)
        
        except Exception as e:
            self.logger.error(f"Error analyzing implementation decisions in {file_path}: {e}")
        
        return decisions
    
    def _get_file_type(self, file_path: str) -> str:
        """Get file type from path."""
        extension = file_path.split('.')[-1].lower() if '.' in file_path else 'unknown'
        
        type_mapping = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'rs': 'rust',
            'go': 'go',
            'rb': 'ruby',
            'php': 'php',
            'yaml': 'configuration',
            'yml': 'configuration',
            'json': 'configuration',
            'md': 'documentation'
        }
        
        return type_mapping.get(extension, extension)
    
    # Decision type detectors
    def _detect_architectural_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect architectural decision."""
        keywords = ['architecture', 'design', 'structure', 'pattern', 'component', 'system']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_technical_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect technical decision."""
        keywords = ['technology', 'framework', 'library', 'tool', 'implementation', 'algorithm', 
                    'database', 'postgresql', 'mysql', 'json', 'migration', 'performance']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_process_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect process decision."""
        keywords = ['process', 'workflow', 'procedure', 'methodology', 'approach']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_strategic_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect strategic decision."""
        keywords = ['strategy', 'direction', 'vision', 'roadmap', 'goal', 'objective']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_tactical_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect tactical decision."""
        keywords = ['tactics', 'immediate', 'short-term', 'quick', 'temporary']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_design_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect design decision."""
        keywords = ['design', 'interface', 'user experience', 'layout', 'visual']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_implementation_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect implementation decision."""
        keywords = ['implementation', 'coding', 'development', 'build', 'create']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _detect_quality_decision(self, text: str, data: Dict[str, Any]) -> bool:
        """Detect quality decision."""
        keywords = ['quality', 'testing', 'validation', 'verification', 'standards']
        return any(keyword in text.lower() for keyword in keywords)
    
    # Framework detectors
    def _detect_trade_off_analysis(self, data: Dict[str, Any]) -> bool:
        """Detect trade-off analysis framework."""
        has_alternatives = 'alternatives' in data and len(data.get('alternatives', [])) > 1
        has_pros_cons = any(
            isinstance(alt, dict) and ('pros' in alt or 'cons' in alt)
            for alt in data.get('alternatives', [])
        )
        return has_alternatives and has_pros_cons
    
    def _detect_cost_benefit(self, data: Dict[str, Any]) -> bool:
        """Detect cost-benefit analysis framework."""
        text = str(data).lower()
        return 'cost' in text and 'benefit' in text
    
    def _detect_risk_assessment(self, data: Dict[str, Any]) -> bool:
        """Detect risk assessment framework."""
        return 'risk' in data or 'risk_assessment' in data or 'risks' in data
    
    def _detect_stakeholder_analysis(self, data: Dict[str, Any]) -> bool:
        """Detect stakeholder analysis framework."""
        return 'stakeholders' in data and data.get('stakeholders')
    
    def _detect_constraint_analysis(self, data: Dict[str, Any]) -> bool:
        """Detect constraint analysis framework."""
        return 'constraints' in data and data.get('constraints')
    
    def _detect_impact_analysis(self, data: Dict[str, Any]) -> bool:
        """Detect impact analysis framework."""
        return 'impact' in data or 'consequences' in data
    
    def _detect_alternative_evaluation(self, data: Dict[str, Any]) -> bool:
        """Detect alternative evaluation framework."""
        return 'alternatives' in data and len(data.get('alternatives', [])) > 0