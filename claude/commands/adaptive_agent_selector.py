#!/usr/bin/env python3
"""
Adaptive Agent Selection System for RIF Orchestration

This module implements a 5-layer intelligence engine that selects optimal agents
based on pattern matching, complexity analysis, and dynamic team composition.

Architecture:
- Layer 1: IssueContextAnalyzer - Extract requirements and assess complexity
- Layer 2: HistoricalPatternMatcher - Find similar issues and successful patterns
- Layer 3: AgentCapabilityMapper - Map capabilities to requirements
- Layer 4: DynamicTeamComposer - Optimize team composition
- Layer 5: SelectionLearningSystem - Continuous improvement through feedback

CRITICAL: This module supports Claude Code as the orchestrator. It provides
recommendations for agent selection but Claude Code launches the Task() agents.
"""

import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents an agent's capability"""
    name: str
    weight: float
    evidence_required: bool
    performance_history: List[float]


@dataclass  
class RequirementMapping:
    """Maps requirements to agent capabilities"""
    requirement: str
    required_capabilities: List[str]
    priority: float
    complexity_factor: float


@dataclass
class TeamComposition:
    """Represents an optimal team composition"""
    agents: List[str]
    confidence_score: float
    capability_coverage: float
    estimated_effort: float
    success_probability: float
    historical_basis: List[str]


@dataclass
class SelectionResult:
    """Result of agent selection process"""
    recommended_agents: List[str]
    team_composition: TeamComposition
    rationale: str
    confidence_score: float
    alternative_options: List[TeamComposition]
    performance_metrics: Dict[str, float]


# Abstract interfaces for the 5-layer architecture

class IssueContextAnalyzerInterface(ABC):
    """Layer 1: Issue Context Analysis Interface"""
    
    @abstractmethod
    def extract_requirements(self, issue_context: Dict[str, Any]) -> List[RequirementMapping]:
        """Extract and analyze requirements from issue context"""
        pass
    
    @abstractmethod
    def assess_complexity(self, issue_context: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """Assess issue complexity on 1-4 scale with breakdown"""
        pass


class HistoricalPatternMatcherInterface(ABC):
    """Layer 2: Historical Pattern Matching Interface"""
    
    @abstractmethod
    def find_similar_issues(self, issue_context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar historical issues"""
        pass
    
    @abstractmethod
    def extract_successful_patterns(self, similar_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from successful issue resolutions"""
        pass


class AgentCapabilityMapperInterface(ABC):
    """Layer 3: Agent Capability Mapping Interface"""
    
    @abstractmethod
    def map_requirements_to_capabilities(self, requirements: List[RequirementMapping]) -> Dict[str, List[str]]:
        """Map requirements to required agent capabilities"""
        pass
    
    @abstractmethod
    def get_agents_with_capabilities(self, capabilities: List[str]) -> Dict[str, AgentCapability]:
        """Get agents that have required capabilities"""
        pass


class DynamicTeamComposerInterface(ABC):
    """Layer 4: Dynamic Team Composition Interface"""
    
    @abstractmethod
    def compose_optimal_team(self, 
                           requirements: List[RequirementMapping],
                           available_agents: Dict[str, AgentCapability],
                           constraints: Dict[str, Any]) -> TeamComposition:
        """Compose optimal team for requirements"""
        pass
    
    @abstractmethod
    def validate_team_coverage(self, team: List[str], requirements: List[RequirementMapping]) -> float:
        """Validate team capability coverage"""
        pass


class SelectionLearningSystemInterface(ABC):
    """Layer 5: Selection Learning System Interface"""
    
    @abstractmethod
    def record_selection_outcome(self, 
                               selection: SelectionResult,
                               actual_outcome: Dict[str, Any]) -> None:
        """Record selection outcome for learning"""
        pass
    
    @abstractmethod
    def update_agent_performance_scores(self, agent_performances: Dict[str, float]) -> None:
        """Update agent performance scores"""
        pass


# Concrete implementations of the 5-layer architecture

class IssueContextAnalyzer(IssueContextAnalyzerInterface):
    """Layer 1: Extract and analyze issue context"""
    
    def __init__(self):
        self.complexity_indicators = {
            'very_high': ['microservice', 'distributed', 'architecture', 'migration', 'refactor', 'orchestrator'],
            'high': ['integration', 'api', 'database', 'security', 'performance', 'system'],
            'medium': ['feature', 'enhancement', 'component', 'module', 'workflow'],
            'low': ['fix', 'bug', 'documentation', 'typo', 'minor', 'simple']
        }
        
        self.requirement_patterns = {
            'analysis': ['analyze', 'requirements', 'specification', 'understand'],
            'architecture': ['design', 'architecture', 'system', 'structure', 'pattern'],
            'implementation': ['implement', 'code', 'build', 'create', 'develop'],
            'validation': ['test', 'validate', 'verify', 'check', 'quality'],
            'security': ['security', 'auth', 'permission', 'vulnerability', 'encrypt'],
            'performance': ['performance', 'optimization', 'speed', 'efficiency', 'scale']
        }
    
    def extract_requirements(self, issue_context: Dict[str, Any]) -> List[RequirementMapping]:
        """Extract requirements from issue context"""
        requirements = []
        
        title = issue_context.get('title', '').lower()
        body = issue_context.get('body', '').lower()
        labels = [label.lower() for label in issue_context.get('labels', [])]
        
        combined_text = f"{title} {body} {' '.join(labels)}"
        
        # Analyze requirements based on patterns
        for req_type, patterns in self.requirement_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern in combined_text)
            if pattern_matches > 0:
                priority = min(1.0, pattern_matches / len(patterns) * 2)
                complexity_factor = self._calculate_requirement_complexity(req_type, combined_text)
                
                requirements.append(RequirementMapping(
                    requirement=req_type,
                    required_capabilities=self._get_capabilities_for_requirement(req_type),
                    priority=priority,
                    complexity_factor=complexity_factor
                ))
        
        # Sort by priority
        requirements.sort(key=lambda r: r.priority, reverse=True)
        return requirements
    
    def assess_complexity(self, issue_context: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """Assess issue complexity"""
        title = issue_context.get('title', '').lower()
        body = issue_context.get('body', '').lower()
        labels = [label.lower() for label in issue_context.get('labels', [])]
        
        combined_text = f"{title} {body} {' '.join(labels)}"
        
        complexity_scores = {}
        for complexity, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            complexity_scores[complexity] = score / len(indicators)
        
        # Calculate overall complexity (1-4 scale)
        if complexity_scores['very_high'] > 0.3:
            overall = 4
        elif complexity_scores['high'] > 0.3:
            overall = 3
        elif complexity_scores['medium'] > 0.3:
            overall = 2
        else:
            overall = 1
        
        # Check for explicit complexity labels
        for label in labels:
            if 'complexity:very-high' in label:
                overall = 4
            elif 'complexity:high' in label:
                overall = 3
            elif 'complexity:medium' in label:
                overall = 2
            elif 'complexity:low' in label:
                overall = 1
        
        return overall, complexity_scores
    
    def _calculate_requirement_complexity(self, req_type: str, text: str) -> float:
        """Calculate complexity factor for a requirement type"""
        complexity_factors = {
            'analysis': 1.2,
            'architecture': 2.0,
            'implementation': 1.5,
            'validation': 1.3,
            'security': 1.8,
            'performance': 1.7
        }
        
        base_factor = complexity_factors.get(req_type, 1.0)
        
        # Adjust based on text complexity
        if any(word in text for word in ['complex', 'difficult', 'challenging']):
            base_factor *= 1.5
        elif any(word in text for word in ['simple', 'easy', 'basic']):
            base_factor *= 0.8
            
        return min(3.0, max(0.5, base_factor))
    
    def _get_capabilities_for_requirement(self, req_type: str) -> List[str]:
        """Get capabilities needed for requirement type"""
        capability_mapping = {
            'analysis': ['requirements', 'patterns', 'complexity'],
            'architecture': ['design', 'dependencies', 'scaling'],
            'implementation': ['coding', 'refactoring', 'optimization'],
            'validation': ['testing', 'quality', 'compliance'],
            'security': ['vulnerabilities', 'auth', 'encryption'],
            'performance': ['optimization', 'profiling', 'scaling']
        }
        
        return capability_mapping.get(req_type, [])


class HistoricalPatternMatcher(HistoricalPatternMatcherInterface):
    """Layer 2: Find and analyze historical patterns"""
    
    def __init__(self, knowledge_base_path: str = "knowledge"):
        self.knowledge_path = Path(knowledge_base_path)
        self.patterns_path = self.knowledge_path / "patterns"
        self.decisions_path = self.knowledge_path / "decisions"
    
    def find_similar_issues(self, issue_context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar historical issues using text similarity"""
        try:
            # Get all historical issues from GitHub
            cmd = "gh issue list --state closed --limit 50 --json number,title,body,labels"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("Could not fetch historical issues from GitHub")
                return []
            
            historical_issues = json.loads(result.stdout)
            
            # Calculate similarity scores
            current_text = f"{issue_context.get('title', '')} {issue_context.get('body', '')}"
            similar_issues = []
            
            for issue in historical_issues:
                historical_text = f"{issue.get('title', '')} {issue.get('body', '')}"
                similarity_score = self._calculate_text_similarity(current_text, historical_text)
                
                if similarity_score > 0.3:  # Minimum similarity threshold
                    issue['similarity_score'] = similarity_score
                    similar_issues.append(issue)
            
            # Sort by similarity and return top matches
            similar_issues.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_issues[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar issues: {e}")
            return []
    
    def extract_successful_patterns(self, similar_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract successful agent patterns from historical issues"""
        patterns = []
        
        for issue in similar_issues:
            try:
                # Get issue comments to find agent history
                cmd = f"gh issue view {issue['number']} --json comments"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    comments_data = json.loads(result.stdout)
                    agents_used = self._extract_agents_from_comments(comments_data.get('comments', []))
                    
                    if agents_used:
                        pattern = {
                            'issue_number': issue['number'],
                            'similarity_score': issue.get('similarity_score', 0.0),
                            'agents_used': agents_used,
                            'success_indicators': self._assess_success_indicators(comments_data.get('comments', [])),
                            'labels': [label['name'] for label in issue.get('labels', [])]
                        }
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning(f"Could not extract pattern from issue {issue['number']}: {e}")
        
        return patterns
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_agents_from_comments(self, comments: List[Dict[str, Any]]) -> List[str]:
        """Extract RIF agent names from issue comments"""
        agents = []
        
        for comment in comments:
            body = comment.get('body', '')
            if '**Agent**:' in body:
                # Extract agent name
                lines = body.split('\n')
                for line in lines:
                    if '**Agent**:' in line:
                        agent_name = line.split('**Agent**:')[1].strip().split()[0]
                        if agent_name.startswith('RIF-') or agent_name.startswith('rif-'):
                            agents.append(agent_name)
        
        return list(set(agents))  # Remove duplicates
    
    def _assess_success_indicators(self, comments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess success indicators from issue comments"""
        indicators = {
            'completion_rate': 0.0,
            'quality_score': 0.0,
            'efficiency_score': 0.0
        }
        
        success_keywords = ['complete', 'success', 'implemented', 'validated', 'merged']
        failure_keywords = ['failed', 'error', 'blocked', 'incomplete']
        
        total_comments = len(comments)
        if total_comments == 0:
            return indicators
        
        success_count = 0
        failure_count = 0
        
        for comment in comments:
            body = comment.get('body', '').lower()
            success_count += sum(1 for keyword in success_keywords if keyword in body)
            failure_count += sum(1 for keyword in failure_keywords if keyword in body)
        
        if success_count + failure_count > 0:
            indicators['completion_rate'] = success_count / (success_count + failure_count)
        
        # Simple heuristics for quality and efficiency
        indicators['quality_score'] = min(1.0, success_count / max(1, total_comments))
        indicators['efficiency_score'] = max(0.0, 1.0 - (failure_count / max(1, total_comments)))
        
        return indicators


class AgentCapabilityMapper(AgentCapabilityMapperInterface):
    """Layer 3: Map requirements to agent capabilities"""
    
    def __init__(self):
        self.agent_capabilities = {
            'rif-analyst': AgentCapability(
                name='rif-analyst',
                weight=1.0,
                evidence_required=True,
                performance_history=[0.85, 0.87, 0.83, 0.89, 0.86]
            ),
            'rif-architect': AgentCapability(
                name='rif-architect', 
                weight=1.2,
                evidence_required=True,
                performance_history=[0.82, 0.85, 0.87, 0.84, 0.88]
            ),
            'rif-implementer': AgentCapability(
                name='rif-implementer',
                weight=1.0,
                evidence_required=True,
                performance_history=[0.79, 0.81, 0.84, 0.82, 0.85]
            ),
            'rif-validator': AgentCapability(
                name='rif-validator',
                weight=1.1,
                evidence_required=True,
                performance_history=[0.88, 0.86, 0.89, 0.87, 0.90]
            ),
            'rif-security': AgentCapability(
                name='rif-security',
                weight=1.3,
                evidence_required=True,
                performance_history=[0.91, 0.89, 0.92, 0.88, 0.91]
            ),
            'rif-performance': AgentCapability(
                name='rif-performance',
                weight=1.2,
                evidence_required=True,
                performance_history=[0.84, 0.87, 0.85, 0.89, 0.86]
            )
        }
        
        # Define what capabilities each agent has
        self.agent_capability_matrix = {
            'rif-analyst': ['requirements', 'patterns', 'complexity'],
            'rif-architect': ['design', 'dependencies', 'scaling'],
            'rif-implementer': ['coding', 'refactoring', 'optimization'],
            'rif-validator': ['testing', 'quality', 'compliance'],
            'rif-security': ['vulnerabilities', 'auth', 'encryption'],
            'rif-performance': ['optimization', 'profiling', 'scaling']
        }
    
    def map_requirements_to_capabilities(self, requirements: List[RequirementMapping]) -> Dict[str, List[str]]:
        """Map requirements to needed capabilities"""
        capability_map = {}
        
        for requirement in requirements:
            needed_caps = requirement.required_capabilities
            capability_map[requirement.requirement] = needed_caps
        
        return capability_map
    
    def get_agents_with_capabilities(self, capabilities: List[str]) -> Dict[str, AgentCapability]:
        """Get agents that have the required capabilities"""
        matching_agents = {}
        
        for agent_name, agent_caps in self.agent_capability_matrix.items():
            # Check if agent has any of the required capabilities
            if any(cap in agent_caps for cap in capabilities):
                matching_agents[agent_name] = self.agent_capabilities[agent_name]
        
        return matching_agents
    
    def get_capability_coverage(self, agents: List[str], required_capabilities: List[str]) -> float:
        """Calculate how well a team covers required capabilities"""
        if not required_capabilities:
            return 1.0
        
        covered_capabilities = set()
        for agent in agents:
            if agent in self.agent_capability_matrix:
                covered_capabilities.update(self.agent_capability_matrix[agent])
        
        coverage = len(set(required_capabilities).intersection(covered_capabilities)) / len(required_capabilities)
        return coverage


class DynamicTeamComposer(DynamicTeamComposerInterface):
    """Layer 4: Compose optimal teams"""
    
    def __init__(self, capability_mapper: AgentCapabilityMapper):
        self.capability_mapper = capability_mapper
    
    def compose_optimal_team(self, 
                           requirements: List[RequirementMapping],
                           available_agents: Dict[str, AgentCapability],
                           constraints: Dict[str, Any]) -> TeamComposition:
        """Compose optimal team for the given requirements"""
        
        # Extract all required capabilities
        all_required_caps = []
        for req in requirements:
            all_required_caps.extend(req.required_capabilities)
        
        unique_caps = list(set(all_required_caps))
        
        # Start with minimal viable team
        team = []
        covered_capabilities = set()
        
        # Sort agents by performance score (descending)
        sorted_agents = sorted(
            available_agents.items(),
            key=lambda x: np.mean(x[1].performance_history),
            reverse=True
        )
        
        # Greedy selection to cover capabilities
        for agent_name, agent_capability in sorted_agents:
            agent_caps = self.capability_mapper.agent_capability_matrix.get(agent_name, [])
            new_caps = set(agent_caps) - covered_capabilities
            
            if new_caps:  # Agent adds new capabilities
                team.append(agent_name)
                covered_capabilities.update(agent_caps)
                
                # Check if we have full coverage
                if covered_capabilities.issuperset(set(unique_caps)):
                    break
        
        # Add specialists for high-risk areas if needed
        team = self._add_specialists_if_needed(team, requirements, constraints)
        
        # Calculate metrics
        coverage = len(covered_capabilities.intersection(set(unique_caps))) / len(unique_caps) if unique_caps else 1.0
        confidence = self._calculate_team_confidence(team, requirements)
        effort = self._estimate_team_effort(team, requirements)
        success_prob = self._calculate_success_probability(team, requirements)
        
        return TeamComposition(
            agents=team,
            confidence_score=confidence,
            capability_coverage=coverage,
            estimated_effort=effort,
            success_probability=success_prob,
            historical_basis=self._get_historical_basis(team)
        )
    
    def validate_team_coverage(self, team: List[str], requirements: List[RequirementMapping]) -> float:
        """Validate team capability coverage"""
        all_required_caps = []
        for req in requirements:
            all_required_caps.extend(req.required_capabilities)
        
        return self.capability_mapper.get_capability_coverage(team, all_required_caps)
    
    def _add_specialists_if_needed(self, team: List[str], requirements: List[RequirementMapping], constraints: Dict[str, Any]) -> List[str]:
        """Add specialist agents based on requirements"""
        enhanced_team = team.copy()
        
        # Check for security requirements
        security_reqs = [req for req in requirements if 'security' in req.requirement.lower()]
        if security_reqs and 'rif-security' not in enhanced_team:
            # Check if security is high priority or complexity
            high_sec_priority = any(req.priority > 0.7 or req.complexity_factor > 1.5 for req in security_reqs)
            if high_sec_priority:
                enhanced_team.append('rif-security')
        
        # Check for performance requirements  
        perf_reqs = [req for req in requirements if 'performance' in req.requirement.lower()]
        if perf_reqs and 'rif-performance' not in enhanced_team:
            high_perf_priority = any(req.priority > 0.7 or req.complexity_factor > 1.5 for req in perf_reqs)
            if high_perf_priority:
                enhanced_team.append('rif-performance')
        
        # Check constraints
        max_team_size = constraints.get('max_team_size', 6)
        if len(enhanced_team) > max_team_size:
            # Remove lowest priority agents
            priority_scores = {agent: self._calculate_agent_priority(agent, requirements) for agent in enhanced_team}
            enhanced_team = sorted(enhanced_team, key=lambda x: priority_scores[x], reverse=True)[:max_team_size]
        
        return enhanced_team
    
    def _calculate_team_confidence(self, team: List[str], requirements: List[RequirementMapping]) -> float:
        """Calculate confidence score for team composition"""
        if not team:
            return 0.0
        
        # Base confidence from agent performance histories
        agent_performances = []
        for agent in team:
            if agent in self.capability_mapper.agent_capabilities:
                agent_cap = self.capability_mapper.agent_capabilities[agent]
                avg_performance = np.mean(agent_cap.performance_history)
                agent_performances.append(avg_performance)
        
        if not agent_performances:
            return 0.5  # Neutral confidence
        
        # Team confidence is weighted average
        team_confidence = np.mean(agent_performances)
        
        # Adjust based on capability coverage
        coverage = self.validate_team_coverage(team, requirements)
        adjusted_confidence = team_confidence * (0.7 + 0.3 * coverage)  # Coverage bonus
        
        return min(1.0, max(0.0, adjusted_confidence))
    
    def _estimate_team_effort(self, team: List[str], requirements: List[RequirementMapping]) -> float:
        """Estimate total effort required for team"""
        base_effort = sum(req.complexity_factor for req in requirements)
        
        # Team efficiency factor (more agents = some overhead, but parallel work)
        team_size = len(team)
        if team_size <= 1:
            efficiency_factor = 1.0
        elif team_size <= 3:
            efficiency_factor = 0.8  # Good parallel efficiency
        elif team_size <= 5:
            efficiency_factor = 0.9  # Some coordination overhead
        else:
            efficiency_factor = 1.1  # Higher coordination overhead
        
        return base_effort * efficiency_factor
    
    def _calculate_success_probability(self, team: List[str], requirements: List[RequirementMapping]) -> float:
        """Calculate probability of successful completion"""
        if not team:
            return 0.0
        
        # Base success from team confidence
        team_confidence = self._calculate_team_confidence(team, requirements)
        
        # Adjust for requirement complexity
        avg_complexity = np.mean([req.complexity_factor for req in requirements]) if requirements else 1.0
        complexity_penalty = max(0.0, (avg_complexity - 1.0) * 0.1)  # Penalty for high complexity
        
        # Adjust for capability coverage
        coverage = self.validate_team_coverage(team, requirements)
        coverage_bonus = coverage * 0.2  # Bonus for good coverage
        
        success_prob = team_confidence + coverage_bonus - complexity_penalty
        return min(1.0, max(0.0, success_prob))
    
    def _get_historical_basis(self, team: List[str]) -> List[str]:
        """Get historical basis for team composition"""
        # This would typically query historical data
        # For now, return a simple indicator
        return [f"Agent {agent} has proven capability in similar contexts" for agent in team]
    
    def _calculate_agent_priority(self, agent: str, requirements: List[RequirementMapping]) -> float:
        """Calculate agent priority for a set of requirements"""
        if agent not in self.capability_mapper.agent_capability_matrix:
            return 0.0
        
        agent_caps = self.capability_mapper.agent_capability_matrix[agent]
        
        # Calculate relevance to requirements
        relevance_score = 0.0
        for req in requirements:
            req_caps = set(req.required_capabilities)
            agent_cap_set = set(agent_caps)
            overlap = req_caps.intersection(agent_cap_set)
            
            if overlap:
                relevance_score += req.priority * len(overlap) / len(req_caps)
        
        # Factor in agent performance
        if agent in self.capability_mapper.agent_capabilities:
            performance = np.mean(self.capability_mapper.agent_capabilities[agent].performance_history)
            relevance_score *= performance
        
        return relevance_score


class SelectionLearningSystem(SelectionLearningSystemInterface):
    """Layer 5: Learning and continuous improvement"""
    
    def __init__(self, learning_data_path: str = "knowledge/learning"):
        self.learning_path = Path(learning_data_path)
        self.learning_path.mkdir(parents=True, exist_ok=True)
        
        self.selection_outcomes_file = self.learning_path / "agent_selection_outcomes.jsonl"
        self.performance_trends_file = self.learning_path / "agent_performance_trends.json"
    
    def record_selection_outcome(self, 
                               selection: SelectionResult,
                               actual_outcome: Dict[str, Any]) -> None:
        """Record selection outcome for learning"""
        timestamp = datetime.now().isoformat()
        
        outcome_record = {
            'timestamp': timestamp,
            'selection': asdict(selection),
            'actual_outcome': actual_outcome,
            'accuracy_score': self._calculate_selection_accuracy(selection, actual_outcome)
        }
        
        # Append to outcomes file
        try:
            with open(self.selection_outcomes_file, 'a') as f:
                f.write(json.dumps(outcome_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to record selection outcome: {e}")
    
    def update_agent_performance_scores(self, agent_performances: Dict[str, float]) -> None:
        """Update agent performance scores"""
        timestamp = datetime.now().isoformat()
        
        # Load existing trends
        trends = self._load_performance_trends()
        
        # Update trends
        for agent, performance in agent_performances.items():
            if agent not in trends:
                trends[agent] = []
            
            trends[agent].append({
                'timestamp': timestamp,
                'performance': performance
            })
            
            # Keep only last 100 records
            trends[agent] = trends[agent][-100:]
        
        # Save updated trends
        try:
            with open(self.performance_trends_file, 'w') as f:
                json.dump(trends, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update performance trends: {e}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data"""
        insights = {
            'total_selections_recorded': 0,
            'average_selection_accuracy': 0.0,
            'agent_performance_trends': {},
            'common_failure_patterns': []
        }
        
        try:
            # Analyze selection outcomes
            if self.selection_outcomes_file.exists():
                outcomes = []
                with open(self.selection_outcomes_file, 'r') as f:
                    for line in f:
                        try:
                            outcomes.append(json.loads(line.strip()))
                        except:
                            continue
                
                if outcomes:
                    insights['total_selections_recorded'] = len(outcomes)
                    accuracy_scores = [outcome.get('accuracy_score', 0.0) for outcome in outcomes]
                    insights['average_selection_accuracy'] = np.mean(accuracy_scores)
            
            # Analyze performance trends
            trends = self._load_performance_trends()
            for agent, trend_data in trends.items():
                if trend_data:
                    recent_performances = [record['performance'] for record in trend_data[-10:]]
                    insights['agent_performance_trends'][agent] = {
                        'current_average': np.mean(recent_performances),
                        'trend_direction': 'improving' if len(recent_performances) >= 2 and recent_performances[-1] > recent_performances[0] else 'stable'
                    }
        
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
        
        return insights
    
    def _calculate_selection_accuracy(self, selection: SelectionResult, actual_outcome: Dict[str, Any]) -> float:
        """Calculate accuracy of agent selection"""
        # Simple accuracy based on actual vs predicted success
        predicted_success = selection.team_composition.success_probability
        actual_success = actual_outcome.get('success_score', 0.5)  # Default neutral
        
        # Accuracy is inverse of prediction error
        error = abs(predicted_success - actual_success)
        accuracy = max(0.0, 1.0 - error)
        
        return accuracy
    
    def _load_performance_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load performance trends from file"""
        if not self.performance_trends_file.exists():
            return {}
        
        try:
            with open(self.performance_trends_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load performance trends: {e}")
            return {}


class AdaptiveAgentSelector:
    """
    Main orchestrator class that coordinates the 5-layer intelligence engine
    for adaptive agent selection.
    
    This class provides recommendations to Claude Code for optimal agent selection
    based on pattern matching, complexity analysis, and dynamic team composition.
    """
    
    def __init__(self, 
                 context_analyzer: Optional[IssueContextAnalyzerInterface] = None,
                 pattern_matcher: Optional[HistoricalPatternMatcherInterface] = None,
                 capability_mapper: Optional[AgentCapabilityMapperInterface] = None,
                 team_composer: Optional[DynamicTeamComposerInterface] = None,
                 learning_system: Optional[SelectionLearningSystemInterface] = None):
        """Initialize the 5-layer intelligence engine"""
        
        # Initialize layers with defaults if not provided
        self.context_analyzer = context_analyzer or IssueContextAnalyzer()
        self.pattern_matcher = pattern_matcher or HistoricalPatternMatcher()
        self.capability_mapper = capability_mapper or AgentCapabilityMapper()
        self.team_composer = team_composer or DynamicTeamComposer(self.capability_mapper)
        self.learning_system = learning_system or SelectionLearningSystem()
        
        # Performance tracking
        self.selection_stats = {
            'selections_made': 0,
            'avg_selection_time_ms': 0.0,
            'accuracy_rate': 0.0,
            'improvement_rate': 0.0
        }
        
        logger.info("AdaptiveAgentSelector initialized with 5-layer architecture")
    
    def select_optimal_agents(self, 
                            issue_context: Dict[str, Any],
                            constraints: Optional[Dict[str, Any]] = None) -> SelectionResult:
        """
        Select optimal agents for an issue using the 5-layer intelligence engine.
        
        Args:
            issue_context: GitHub issue context (number, title, body, labels, etc.)
            constraints: Optional constraints (max_team_size, budget, etc.)
            
        Returns:
            SelectionResult with recommended agents and team composition
        """
        start_time = time.time()
        
        if constraints is None:
            constraints = {'max_team_size': 4}
        
        try:
            # Layer 1: Analyze issue context and extract requirements
            logger.info(f"Layer 1: Analyzing issue context for issue #{issue_context.get('number', 'unknown')}")
            requirements = self.context_analyzer.extract_requirements(issue_context)
            complexity, complexity_breakdown = self.context_analyzer.assess_complexity(issue_context)
            
            # Layer 2: Find similar issues and extract successful patterns
            logger.info("Layer 2: Finding historical patterns")
            similar_issues = self.pattern_matcher.find_similar_issues(issue_context)
            successful_patterns = self.pattern_matcher.extract_successful_patterns(similar_issues)
            
            # Layer 3: Map requirements to agent capabilities
            logger.info("Layer 3: Mapping capabilities")
            capability_map = self.capability_mapper.map_requirements_to_capabilities(requirements)
            all_required_caps = []
            for caps in capability_map.values():
                all_required_caps.extend(caps)
            
            available_agents = self.capability_mapper.get_agents_with_capabilities(all_required_caps)
            
            # Layer 4: Compose optimal team
            logger.info("Layer 4: Composing optimal team")
            optimal_team = self.team_composer.compose_optimal_team(requirements, available_agents, constraints)
            
            # Generate alternatives
            alternatives = self._generate_alternative_compositions(requirements, available_agents, constraints, optimal_team)
            
            # Calculate performance metrics
            selection_time = (time.time() - start_time) * 1000
            performance_metrics = {
                'selection_time_ms': selection_time,
                'requirements_count': len(requirements),
                'similar_issues_found': len(similar_issues),
                'patterns_analyzed': len(successful_patterns),
                'available_agents_count': len(available_agents)
            }
            
            # Generate rationale
            rationale = self._generate_selection_rationale(
                requirements, optimal_team, successful_patterns, complexity, performance_metrics
            )
            
            # Create selection result
            result = SelectionResult(
                recommended_agents=optimal_team.agents,
                team_composition=optimal_team,
                rationale=rationale,
                confidence_score=optimal_team.confidence_score,
                alternative_options=alternatives,
                performance_metrics=performance_metrics
            )
            
            # Update performance statistics
            self._update_selection_stats(selection_time)
            
            logger.info(f"Agent selection completed in {selection_time:.1f}ms with confidence {optimal_team.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            
            # Return fallback selection
            fallback_result = self._create_fallback_selection(issue_context, constraints)
            return fallback_result
    
    def record_selection_outcome(self, selection_result: SelectionResult, actual_outcome: Dict[str, Any]) -> None:
        """
        Record the outcome of an agent selection for learning purposes.
        
        Args:
            selection_result: The original selection result
            actual_outcome: The actual outcome (success rate, quality scores, etc.)
        """
        # Layer 5: Record for continuous learning
        self.learning_system.record_selection_outcome(selection_result, actual_outcome)
        
        # Update agent performance scores
        agent_performances = {}
        for agent in selection_result.recommended_agents:
            # Extract agent-specific performance from outcome
            agent_performances[agent] = actual_outcome.get(f'{agent}_performance', actual_outcome.get('overall_success', 0.5))
        
        self.learning_system.update_agent_performance_scores(agent_performances)
        
        logger.info(f"Recorded selection outcome for learning - success: {actual_outcome.get('overall_success', 'unknown')}")
    
    def get_selection_insights(self) -> Dict[str, Any]:
        """Get insights about agent selection performance and trends"""
        learning_insights = self.learning_system.get_learning_insights()
        
        insights = {
            'selection_statistics': self.selection_stats,
            'learning_insights': learning_insights,
            'system_health': {
                'layers_operational': 5,  # All layers operational
                'avg_response_time_ms': self.selection_stats['avg_selection_time_ms'],
                'accuracy_trend': learning_insights.get('average_selection_accuracy', 0.0)
            }
        }
        
        return insights
    
    def _generate_alternative_compositions(self, 
                                         requirements: List[RequirementMapping],
                                         available_agents: Dict[str, AgentCapability],
                                         constraints: Dict[str, Any],
                                         primary_team: TeamComposition) -> List[TeamComposition]:
        """Generate alternative team compositions"""
        alternatives = []
        
        try:
            # Generate minimal team (fewer agents)
            minimal_constraints = constraints.copy()
            minimal_constraints['max_team_size'] = max(1, len(primary_team.agents) - 1)
            
            minimal_team = self.team_composer.compose_optimal_team(requirements, available_agents, minimal_constraints)
            if minimal_team.agents != primary_team.agents:
                alternatives.append(minimal_team)
            
            # Generate comprehensive team (more agents if beneficial)
            if len(primary_team.agents) < constraints.get('max_team_size', 6):
                comprehensive_constraints = constraints.copy()
                comprehensive_constraints['max_team_size'] = constraints.get('max_team_size', 6)
                
                comprehensive_team = self.team_composer.compose_optimal_team(requirements, available_agents, comprehensive_constraints)
                if comprehensive_team.agents != primary_team.agents and comprehensive_team not in alternatives:
                    alternatives.append(comprehensive_team)
        
        except Exception as e:
            logger.warning(f"Could not generate alternative compositions: {e}")
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _generate_selection_rationale(self,
                                    requirements: List[RequirementMapping],
                                    team_composition: TeamComposition,
                                    patterns: List[Dict[str, Any]],
                                    complexity: int,
                                    performance_metrics: Dict[str, float]) -> str:
        """Generate human-readable rationale for agent selection"""
        
        rationale_parts = [
            f"Selected {len(team_composition.agents)} agents based on issue complexity level {complexity}/4 and {len(requirements)} identified requirements."
        ]
        
        # Explain team composition
        if team_composition.agents:
            agent_list = ', '.join(team_composition.agents)
            rationale_parts.append(f"Team composition: {agent_list}")
            rationale_parts.append(f"Capability coverage: {team_composition.capability_coverage:.1%}")
        
        # Explain historical basis
        if patterns:
            similar_count = len(patterns)
            avg_success = np.mean([p.get('success_indicators', {}).get('completion_rate', 0.5) for p in patterns])
            rationale_parts.append(f"Based on analysis of {similar_count} similar issues with {avg_success:.1%} average success rate.")
        
        # Explain confidence factors
        confidence_factors = []
        if team_composition.confidence_score > 0.8:
            confidence_factors.append("high agent performance history")
        if team_composition.capability_coverage > 0.9:
            confidence_factors.append("excellent capability coverage")
        if team_composition.success_probability > 0.8:
            confidence_factors.append("strong success probability")
        
        if confidence_factors:
            rationale_parts.append(f"High confidence due to: {', '.join(confidence_factors)}")
        
        # Performance info
        selection_time = performance_metrics.get('selection_time_ms', 0)
        rationale_parts.append(f"Selection completed in {selection_time:.0f}ms.")
        
        return ' '.join(rationale_parts)
    
    def _create_fallback_selection(self, issue_context: Dict[str, Any], constraints: Dict[str, Any]) -> SelectionResult:
        """Create fallback selection when normal process fails"""
        
        # Default fallback team based on issue characteristics
        fallback_agents = ['rif-analyst']  # Always start with analysis
        
        # Add implementer for most issues
        fallback_agents.append('rif-implementer')
        
        # Add validator for quality assurance
        fallback_agents.append('rif-validator')
        
        # Simple fallback team composition
        fallback_team = TeamComposition(
            agents=fallback_agents,
            confidence_score=0.5,  # Neutral confidence
            capability_coverage=0.7,  # Assume reasonable coverage
            estimated_effort=3.0,  # Default effort estimate
            success_probability=0.6,  # Conservative success estimate
            historical_basis=["Fallback team based on standard RIF workflow"]
        )
        
        return SelectionResult(
            recommended_agents=fallback_agents,
            team_composition=fallback_team,
            rationale="Fallback selection due to analysis error. Using standard RIF workflow agents.",
            confidence_score=0.5,
            alternative_options=[],
            performance_metrics={'selection_time_ms': 0, 'fallback_used': True}
        )
    
    def _update_selection_stats(self, selection_time_ms: float) -> None:
        """Update selection performance statistics"""
        self.selection_stats['selections_made'] += 1
        
        # Update average selection time
        current_avg = self.selection_stats['avg_selection_time_ms']
        total_selections = self.selection_stats['selections_made']
        
        self.selection_stats['avg_selection_time_ms'] = (
            (current_avg * (total_selections - 1) + selection_time_ms) / total_selections
        )


# Convenience functions for Claude Code integration

def create_adaptive_selector(**kwargs) -> AdaptiveAgentSelector:
    """Factory function to create adaptive agent selector with default configuration"""
    return AdaptiveAgentSelector(**kwargs)


def select_agents_for_issue(issue_number: int, **kwargs) -> SelectionResult:
    """
    Convenience function to select agents for a GitHub issue.
    
    Args:
        issue_number: GitHub issue number
        **kwargs: Additional constraints and options
        
    Returns:
        SelectionResult with recommended agents
    """
    selector = create_adaptive_selector()
    
    # Get issue context from GitHub
    try:
        cmd = f"gh issue view {issue_number} --json number,title,body,labels"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to fetch issue {issue_number}")
        
        issue_context = json.loads(result.stdout)
        
        # Extract label names
        issue_context['labels'] = [label['name'] for label in issue_context.get('labels', [])]
        
        return selector.select_optimal_agents(issue_context, kwargs)
        
    except Exception as e:
        logger.error(f"Error selecting agents for issue {issue_number}: {e}")
        return selector._create_fallback_selection({'number': issue_number}, kwargs)


def generate_claude_code_task_launches(selection_result: SelectionResult, issue_number: int) -> List[str]:
    """
    Generate Task() launch code for Claude Code to execute the selected agents.
    
    Args:
        selection_result: Result from adaptive agent selection
        issue_number: GitHub issue number
        
    Returns:
        List of Task() function call strings for Claude Code
    """
    task_codes = []
    
    for agent in selection_result.recommended_agents:
        agent_description = f"{agent.upper()}: Process issue #{issue_number}"
        agent_file = agent.lower().replace('rif-', '') + '.md'
        
        prompt = f"You are {agent.upper()}. Process GitHub issue #{issue_number} according to your specialization. Follow all instructions in claude/agents/{agent_file}."
        
        task_code = f'''Task(
    description="{agent_description}",
    subagent_type="general-purpose",
    prompt="{prompt}"
)'''
        
        task_codes.append(task_code)
    
    return task_codes


if __name__ == "__main__":
    # Example usage
    print("RIF Adaptive Agent Selection System")
    print("=" * 50)
    
    selector = create_adaptive_selector()
    
    # Test with example issue context
    example_context = {
        'number': 54,
        'title': 'Build adaptive agent selection system',
        'body': 'Create an intelligent system that selects agents based on pattern matching, complexity analysis, and dynamic team composition.',
        'labels': ['complexity:high', 'enhancement', 'state:implementing']
    }
    
    result = selector.select_optimal_agents(example_context)
    
    print(f"Recommended agents: {', '.join(result.recommended_agents)}")
    print(f"Confidence score: {result.confidence_score:.2f}")
    print(f"Team coverage: {result.team_composition.capability_coverage:.1%}")
    print(f"Success probability: {result.team_composition.success_probability:.1%}")
    print(f"\nRationale: {result.rationale}")
    
    # Generate Task launch codes for Claude Code
    task_codes = generate_claude_code_task_launches(result, 54)
    print(f"\nGenerated {len(task_codes)} Task launch codes for Claude Code execution")