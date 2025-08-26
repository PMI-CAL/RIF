#!/usr/bin/env python3
"""
ContentAnalysisEngine - Replace Label Dependency with Intelligent Content Analysis

This module implements content-based orchestration state determination to replace
the label-dependent approach identified in Issue #273. The engine analyzes GitHub
issue text to derive state, complexity, dependencies, and orchestration decisions.

CRITICAL FIX: Eliminates reliance on GitHub labels (current_state_label) throughout 
the orchestration system, enabling true content-driven intelligent decision making.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for content analysis predictions"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

class IssueState(Enum):
    """Content-derived issue states"""
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
    """Content-derived complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very-high"

@dataclass
class ContentAnalysisResult:
    """Results of content analysis replacing label dependencies"""
    derived_state: IssueState
    confidence: float
    complexity: ComplexityLevel
    dependencies: List[str]
    blocking_indicators: List[str]
    implementation_hints: List[str]
    validation_requirements: List[str]
    semantic_tags: List[str]
    analysis_timestamp: str
    
class ContentAnalysisEngine:
    """
    Replace label dependency with intelligent content analysis.
    
    This engine addresses Issue #273 by analyzing GitHub issue content to derive
    orchestration state, complexity, and dependencies - eliminating reliance on
    current_state_label and other GitHub labels.
    """
    
    def __init__(self, knowledge_base_path: Optional[Path] = None):
        """Initialize the content analysis engine"""
        self.knowledge_base_path = knowledge_base_path or Path("knowledge")
        self.patterns_path = self.knowledge_base_path / "patterns"
        self.decisions_path = self.knowledge_base_path / "decisions"
        
        # Load content analysis patterns
        self._load_analysis_patterns()
        
        # Initialize state detection patterns
        self._initialize_state_patterns()
        self._initialize_complexity_patterns()
        self._initialize_dependency_patterns()
        
    def _load_analysis_patterns(self):
        """Load learned patterns for content analysis"""
        try:
            pattern_file = self.patterns_path / "content_analysis_patterns.json"
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    self.learned_patterns = json.load(f)
            else:
                self.learned_patterns = {}
        except Exception as e:
            logger.warning(f"Could not load analysis patterns: {e}")
            self.learned_patterns = {}
            
    def _initialize_state_patterns(self):
        """Initialize patterns for state detection from content"""
        self.state_patterns = {
            IssueState.NEW: [
                r'\bnew\b.*\bissue\b',
                r'\bfresh\b.*\btask\b', 
                r'\binitial\b.*\brequirement\b',
                r'\bstarting\b.*\bwork\b',
                r'\bneed\s+to\s+(implement|create|add|build)',
                r'^(?!.*\b(done|complete|finished|implemented|tested)\b)',  # Not completed
            ],
            IssueState.ANALYZING: [
                r'\banalyzing\b|\banalysis\b|\bresearch\b',
                r'\binvestigating\b|\bexploring\b',
                r'\brequirements?\b.*\bgathering\b',
                r'\bunderstanding\b.*\bproblem\b',
                r'\bneed\s+to\s+(understand|analyze|research)',
            ],
            IssueState.PLANNING: [
                r'\bplanning\b|\bdesigning\b|\barchitecture\b',
                r'\bapproach\b|\bstrategy\b|\bmethodology\b',
                r'\bhigh.level\b.*\bdesign\b',
                r'\bworkflow\b|\bprocess\b',
                r'\bneed\s+to\s+(plan|design|architect)',
            ],
            IssueState.IMPLEMENTING: [
                r'\bimplementing?\b|\bcodification\b|\bdeveloping\b',
                r'\bwriting\b.*\bcode\b|\bcoding\b',
                r'\bbuilding\b.*\b(feature|component|system)\b',
                r'\bin\s+progress\b|\bworking\s+on\b',
                r'\bcode\b.*\b(changes|updates|implementation)\b',
            ],
            IssueState.VALIDATING: [
                r'\btesting\b|\bvalidating\b|\bverifying\b',
                r'\bquality\s+assurance\b|\bqa\b',
                r'\btest\s+(suite|cases|scenarios)\b',
                r'\bready\s+for\s+(test|validation|review)\b',
                r'\bvalidation\b.*\brequired\b',
            ],
            IssueState.BLOCKED: [
                r'\bblocked?\b|\bblocking\b',
                r'\bwait(ing)?\s+(for|on)\b',
                r'\bdependency\b.*\bmissing\b',
                r'\bcannot\s+proceed\b|\bstuck\b',
                r'\bthis\s+issue\s+blocks\s+(all\s+)?others?\b',
                r'\bmust\s+complete\s+before\b',
            ],
            IssueState.COMPLETE: [
                r'\bcompleted?\b|\bfinished\b|\bdone\b',
                r'\bimplemented\b|\bresolved\b|\bfixed\b',
                r'\bready\s+to\s+close\b|\bclosing\b',
                r'\bsuccessfully\b.*\b(completed|implemented|tested)\b',
            ],
        }
        
    def _initialize_complexity_patterns(self):
        """Initialize patterns for complexity detection from content"""
        self.complexity_patterns = {
            ComplexityLevel.LOW: [
                r'\bsimple\b|\bminor\b|\bquick\b|\bsmall\b',
                r'\bone.line\b|\bsingle\b.*\bchange\b',
                r'\bfix\b.*\btypo\b|\bsmall\s+bug\b',
                r'\b(less\s+than|<)\s*\d+\s*(line|loc)\b',
                r'\btrivial\b|\beasy\b.*\bfix\b',
                r'\btesting\b.*\b(single|simple)\b',
            ],
            ComplexityLevel.MEDIUM: [
                r'\bmoderate\b|\bmedium\b|\bstandard\b',
                r'\b(few|several)\s+(file|component)s?\b',
                r'\brefactoring?\b|\benhancement\b',
                r'\b\d{2,3}\s*(line|loc)\b',  # 10-999 lines
                r'\bfeature\b.*\baddition\b',
                r'\bauthentication\b|\buser\b.*\bmanagement\b',
                r'\bapi\b.*\bendpoint',
                r'\bimplement\b.*\bfunctionality\b',
            ],
            ComplexityLevel.HIGH: [
                r'\bcomplex\b|\badvanced\b|\bmajor\b',
                r'\b(many|multiple|numerous)\s+(file|component|system)s?\b',
                r'\barchitecture\b.*\bchange\b',
                r'\b\d{3,4}\s*(line|loc)\b',  # 100-9999 lines
                r'\bsignificant\b.*\b(refactor|rewrite)\b',
                r'\bintegration\b.*\b(complex|multiple)\b',
                r'\bperformance\b.*\b(issue|bottleneck)\b',
                r'\bdatabase\b.*\b(migration|schema)\b',
            ],
            ComplexityLevel.VERY_HIGH: [
                r'\bvery\s+(complex|high|difficult)\b|\bcritical\b',
                r'\bmassive\b|\bentire\s+system\b',
                r'\bfull\s+(rewrite|redesign)\b',
                r'\b\d{4,}\s*(line|loc)\b',  # 1000+ lines
                r'\bmulti.system\b|\bcross.cutting\b',
                r'\bbreaking\s+change\b|\bmigration\b',
                r'\bmicroservice\b.*\barchitecture\b',
                r'\bmultiple\b.*\bsystem\b',
                r'\bblocked\b.*\bcritical\b',
            ],
        }
        
    def _initialize_dependency_patterns(self):
        """Initialize patterns for dependency detection"""
        self.dependency_patterns = [
            r'\bdepends?\s+(on|upon)\s+(?:issue\s+)?#?(\d+)',
            r'\brequires?\s+(?:issue\s+)?#?(\d+)',
            r'\bblocked\s+by\s+(?:issue\s+)?#?(\d+)',
            r'\bafter\s+(?:issue\s+)?#?(\d+)',
            r'\bprerequisi[t]e.*?(?:issue\s+)?#?(\d+)',
            r'\bmust\s+(?:wait\s+for|complete)\s+(?:issue\s+)?#?(\d+)',
            r'(?:issue\s+)?#(\d+)\s+\([^)]+\)',  # issue #42 (description)
            r'\band\s+(?:requires\s+)?#?(\d+)',  # "and requires #15"
        ]
        
    def analyze_issue_content(self, issue_title: str, issue_body: str, 
                            issue_number: Optional[int] = None,
                            comments: Optional[List[str]] = None) -> ContentAnalysisResult:
        """
        Main analysis method - replaces current_state_label dependency.
        
        This method addresses the critical fix in Issue #273 line 668:
        Instead of: current_state = context_model.issue_context.current_state_label
        Now use: analysis_result = engine.analyze_issue_content(...)
                current_state = analysis_result.derived_state
        """
        
        # Combine all text content for analysis
        full_text = f"{issue_title}\n\n{issue_body}"
        if comments:
            full_text += "\n\n" + "\n".join(comments)
            
        # Derive state from content (replaces label dependency)
        derived_state, confidence = self._derive_state_from_content(full_text)
        
        # Analyze complexity from content
        complexity = self._determine_complexity_from_content(full_text)
        
        # Extract dependencies from content
        dependencies = self._extract_dependencies_from_content(full_text)
        
        # Detect blocking indicators
        blocking_indicators = self._detect_blocking_indicators(full_text)
        
        # Extract implementation hints
        implementation_hints = self._extract_implementation_hints(full_text)
        
        # Extract validation requirements
        validation_requirements = self._extract_validation_requirements(full_text)
        
        # Generate semantic tags
        semantic_tags = self._generate_semantic_tags(full_text)
        
        return ContentAnalysisResult(
            derived_state=derived_state,
            confidence=confidence,
            complexity=complexity,
            dependencies=dependencies,
            blocking_indicators=blocking_indicators,
            implementation_hints=implementation_hints,
            validation_requirements=validation_requirements,
            semantic_tags=semantic_tags,
            analysis_timestamp=datetime.now().isoformat()
        )
        
    def _derive_state_from_content(self, text: str) -> Tuple[IssueState, float]:
        """Derive issue state from content analysis - replaces label dependency"""
        text_lower = text.lower()
        state_scores = {}
        
        for state, patterns in self.state_patterns.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                if pattern.startswith('^(?!.*'):  # Negative lookahead
                    # Special handling for negative patterns
                    if not re.search(pattern[7:-1], text_lower):  # Remove ^(?!.* and )
                        score += 0.3
                        matches += 1
                else:
                    matches_found = len(re.findall(pattern, text_lower))
                    if matches_found > 0:
                        score += matches_found * 0.8
                        matches += matches_found
                        
            # Normalize score by pattern count
            if matches > 0:
                state_scores[state] = min(score / len(patterns), 1.0)
            else:
                state_scores[state] = 0.0
                
        # Find the state with highest confidence
        if not state_scores:
            return IssueState.NEW, ConfidenceLevel.LOW.value
            
        best_state = max(state_scores, key=state_scores.get)
        confidence = state_scores[best_state]
        
        # Apply heuristics to improve accuracy
        confidence = self._apply_state_heuristics(text_lower, best_state, confidence)
        
        return best_state, confidence
        
    def _determine_complexity_from_content(self, text: str) -> ComplexityLevel:
        """Determine complexity from issue content"""
        text_lower = text.lower()
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.5
                
            complexity_scores[complexity] = score
            
        # Additional heuristics based on content length and structure
        word_count = len(text.split())
        if word_count > 1000:
            complexity_scores[ComplexityLevel.VERY_HIGH] += 0.5
        elif word_count > 500:
            complexity_scores[ComplexityLevel.HIGH] += 0.3
        elif word_count < 100:
            complexity_scores[ComplexityLevel.LOW] += 0.2
            
        # Count code blocks, technical terms, etc.
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        technical_terms = len(re.findall(r'\b(API|database|architecture|algorithm|framework|system|integration|infrastructure)\b', text_lower))
        
        if code_blocks > 3 or technical_terms > 10:
            complexity_scores[ComplexityLevel.HIGH] += 0.4
        elif code_blocks > 1 or technical_terms > 5:
            complexity_scores[ComplexityLevel.MEDIUM] += 0.3
            
        # Return complexity with highest score
        if not complexity_scores:
            return ComplexityLevel.MEDIUM
            
        return max(complexity_scores, key=complexity_scores.get)
        
    def _extract_dependencies_from_content(self, text: str) -> List[str]:
        """Extract issue dependencies from content"""
        dependencies = []
        
        for pattern in self.dependency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Extract numeric parts from tuple
                    for part in match:
                        if str(part).isdigit():
                            dependencies.append(str(part))
                elif str(match).isdigit():
                    dependencies.append(str(match))
        
        # Additional manual patterns for complex dependency expressions
        # Handle "depends on issue #42 (core API) and requires #15"
        complex_pattern = r'(?:depends?\s+(?:on|upon)\s+)?(?:issue\s+)?#(\d+)(?:\s+\([^)]+\))?\s+and\s+(?:requires?\s+)?(?:issue\s+)?#(\d+)'
        complex_matches = re.findall(complex_pattern, text, re.IGNORECASE)
        for match in complex_matches:
            dependencies.extend([str(m) for m in match if str(m).isdigit()])
        
        # Handle comma-separated dependencies: "blocked by #23, #45, and #67"
        list_pattern = r'(?:blocked\s+by|depends?\s+(?:on|upon)|requires?)\s+(?:issues?\s+)?#(\d+)(?:\s*,\s*#(\d+))*(?:\s*,?\s*and\s+#(\d+))?'
        list_matches = re.findall(list_pattern, text, re.IGNORECASE)
        for match in list_matches:
            dependencies.extend([str(m) for m in match if str(m).isdigit()])
                    
        return sorted(list(set(dependencies)))  # Remove duplicates and sort
        
    def _detect_blocking_indicators(self, text: str) -> List[str]:
        """Detect if this issue blocks other work"""
        blocking_indicators = []
        text_lower = text.lower()
        
        blocking_patterns = [
            r'this\s+issue\s+blocks\s+(all\s+)?others?',
            r'must\s+complete\s+before\s+(all\s+)?(other\s+)?work',
            r'critical\s+infrastructure',
            r'blocks\s+all\s+(other\s+)?(work|issues)',
            r'stop\s+all\s+work',
            r'halt.*orchestration'
        ]
        
        for pattern in blocking_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                blocking_indicators.append(pattern)
                
        return blocking_indicators
        
    def _extract_implementation_hints(self, text: str) -> List[str]:
        """Extract implementation hints from issue content"""
        hints = []
        
        # Look for code snippets, file names, technical approaches
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        hints.extend([f"Code example: {block[:100]}..." for block in code_blocks[:3]])
        
        # Look for file references
        file_refs = re.findall(r'[\w/.-]+\.(py|js|ts|java|cpp|c|h|md|json|yaml|yml)', text)
        hints.extend([f"File reference: {ref}" for ref in set(file_refs[:5])])
        
        # Look for class/method references
        class_refs = re.findall(r'\b[A-Z][a-zA-Z]*(?:Engine|Manager|Service|Handler|Controller)\b', text)
        hints.extend([f"Class reference: {ref}" for ref in set(class_refs[:3])])
        
        return hints
        
    def _extract_validation_requirements(self, text: str) -> List[str]:
        """Extract validation and testing requirements"""
        requirements = []
        text_lower = text.lower()
        
        test_patterns = [
            r'test\s+(coverage|suite|cases)',
            r'unit\s+test',
            r'integration\s+test',
            r'performance\s+(test|benchmark)',
            r'security\s+(scan|test)',
            r'quality\s+gate'
        ]
        
        for pattern in test_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                clean_pattern = pattern.replace(r'\s+', ' ')
                requirements.append(f"Requires: {clean_pattern}")
                
        return requirements
        
    def _generate_semantic_tags(self, text: str) -> List[str]:
        """Generate semantic tags for better orchestration decisions"""
        tags = []
        text_lower = text.lower()
        
        # Technology tags
        tech_patterns = {
            'python': r'\bpython\b',
            'javascript': r'\b(javascript|js|node)\b', 
            'database': r'\b(database|db|sql|nosql|mongodb|postgresql)\b',
            'api': r'\bapi\b',
            'frontend': r'\b(frontend|ui|ux|react|vue|angular)\b',
            'backend': r'\b(backend|server|service)\b',
            'devops': r'\b(docker|kubernetes|ci/cd|pipeline|deployment)\b',
            'security': r'\b(security|auth|encryption|vulnerability)\b',
            'performance': r'\b(performance|optimization|speed|latency)\b'
        }
        
        for tag, pattern in tech_patterns.items():
            if re.search(pattern, text_lower):
                tags.append(tag)
                
        # Complexity indicators
        if len(text.split()) > 500:
            tags.append('detailed-requirements')
        if len(re.findall(r'```[\s\S]*?```', text)) > 0:
            tags.append('code-examples')
            
        return tags
        
    def _apply_state_heuristics(self, text: str, state: IssueState, confidence: float) -> float:
        """Apply heuristics to improve state detection accuracy"""
        
        # Recent activity indicators
        if any(word in text for word in ['today', 'now', 'currently', 'just', 'recently']):
            if state in [IssueState.IMPLEMENTING, IssueState.VALIDATING]:
                confidence += 0.1
                
        # Completion indicators
        completion_words = ['done', 'finished', 'completed', 'resolved', 'fixed']
        if any(word in text for word in completion_words):
            if state == IssueState.COMPLETE:
                confidence += 0.2
            else:
                confidence -= 0.1
                
        # Urgency indicators
        urgency_words = ['urgent', 'critical', 'asap', 'immediately', 'priority']
        if any(word in text for word in urgency_words):
            confidence += 0.05
            
        return min(confidence, 1.0)  # Cap at 1.0
        
    def get_replacement_state(self, issue_title: str, issue_body: str, 
                            current_state_label: Optional[str] = None) -> str:
        """
        Direct replacement method for current_state_label dependency.
        
        Use this method to replace:
        current_state = context_model.issue_context.current_state_label
        
        With:
        current_state = content_engine.get_replacement_state(title, body)
        """
        analysis = self.analyze_issue_content(issue_title, issue_body)
        return analysis.derived_state.value
        
    def validate_analysis_accuracy(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Validate the accuracy of content analysis against known cases"""
        results = {
            'state_accuracy': 0.0,
            'complexity_accuracy': 0.0,
            'total_cases': len(test_cases),
            'correct_states': 0,
            'correct_complexity': 0
        }
        
        for case in test_cases:
            analysis = self.analyze_issue_content(
                case['title'], 
                case['body']
            )
            
            if analysis.derived_state.value == case.get('expected_state'):
                results['correct_states'] += 1
                
            if analysis.complexity.value == case.get('expected_complexity'):
                results['correct_complexity'] += 1
                
        if results['total_cases'] > 0:
            results['state_accuracy'] = results['correct_states'] / results['total_cases']
            results['complexity_accuracy'] = results['correct_complexity'] / results['total_cases']
            
        return results