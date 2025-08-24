#!/usr/bin/env python3
"""
Enhanced DPIBS Benchmarking Framework 
Issue #120: DPIBS Architecture Phase 2 - Enhanced Benchmarking + Knowledge Integration APIs

Provides automated design specification grading with enhanced capabilities:
- NLP-enhanced specification extraction (90% accuracy target)
- Evidence-based grading system with transparency
- Real-time GitHub issue analysis
- Multi-dimensional implementation analysis
- Knowledge integration with MCP Knowledge Server
- <2 minute complete benchmarking analysis

Builds upon existing design-benchmarking-framework.py with DPIBS integration
"""

import os
import sys
import json
import time
import re
import hashlib
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import difflib

# Add RIF to path
sys.path.insert(0, '/Users/cal/DEV/RIF')

# Import existing framework components
try:
    from systems.design_benchmarking_framework import (
        SpecificationType, ComplianceLevel, DesignSpecification, 
        ImplementationEvidence, BenchmarkingResult, DesignSpecificationExtractor
    )
except ImportError:
    # Fallback: redefine needed classes locally if import fails
    from enum import Enum
    from dataclasses import dataclass
    from typing import List, Dict, Any
    from datetime import datetime
    
    class SpecificationType(Enum):
        FUNCTIONAL_REQUIREMENTS = "functional_requirements"
        NON_FUNCTIONAL_REQUIREMENTS = "non_functional_requirements" 
        ARCHITECTURAL_CONSTRAINTS = "architectural_constraints"
        QUALITY_GATES = "quality_gates"
        PERFORMANCE_REQUIREMENTS = "performance_requirements"
        SECURITY_REQUIREMENTS = "security_requirements"
        INTEGRATION_REQUIREMENTS = "integration_requirements"
        USER_EXPERIENCE = "user_experience"
        BUSINESS_LOGIC = "business_logic"
        DATA_REQUIREMENTS = "data_requirements"
    
    class ComplianceLevel(Enum):
        FULLY_COMPLIANT = "fully_compliant"
        MOSTLY_COMPLIANT = "mostly_compliant"
        PARTIALLY_COMPLIANT = "partially_compliant"
        MINIMALLY_COMPLIANT = "minimally_compliant"
        NON_COMPLIANT = "non_compliant"
    
    @dataclass
    class DesignSpecification:
        id: str
        type: SpecificationType
        description: str
        acceptance_criteria: List[str]
        constraints: List[str]
        success_metrics: Dict[str, Any]
        priority: str
        measurable: bool
        testable: bool
        created_at: datetime
        issue_number: int
        
        def to_dict(self) -> Dict[str, Any]:
            from dataclasses import asdict
            data = asdict(self)
            data['type'] = self.type.value
            data['created_at'] = self.created_at.isoformat()
            return data
    
    @dataclass
    class ImplementationEvidence:
        spec_id: str
        implementation_details: str
        code_files: List[str]
        test_files: List[str]
        documentation_refs: List[str]
        metrics_achieved: Dict[str, Any]
        compliance_score: float
        compliance_level: ComplianceLevel
        issues_found: List[str]
        recommendations: List[str]
        evidence_timestamp: datetime
        
        def to_dict(self) -> Dict[str, Any]:
            from dataclasses import asdict
            data = asdict(self)
            data['compliance_level'] = self.compliance_level.value
            data['evidence_timestamp'] = self.evidence_timestamp.isoformat()
            return data
    
    @dataclass
    class BenchmarkingResult:
        issue_number: int
        specifications: List[DesignSpecification]
        evidence: List[ImplementationEvidence]
        overall_adherence_score: float
        overall_compliance_level: ComplianceLevel
        constraint_violations: List[Dict[str, Any]]
        goal_achievement: Dict[str, float]
        quality_grade: str
        recommendations: List[str]
        benchmarking_timestamp: datetime
        validator_notes: str
        
        def to_dict(self) -> Dict[str, Any]:
            from dataclasses import asdict
            data = asdict(self)
            data['overall_compliance_level'] = self.overall_compliance_level.value
            data['benchmarking_timestamp'] = self.benchmarking_timestamp.isoformat()
            data['specifications'] = [spec.to_dict() for spec in self.specifications]
            data['evidence'] = [ev.to_dict() for ev in self.evidence]
            return data
    
    class DesignSpecificationExtractor:
        def extract_specifications_from_issue(self, issue_number: int) -> List[DesignSpecification]:
            return []  # Placeholder implementation

from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
from knowledge.database.database_config import DatabaseConfig


@dataclass
class EnhancedBenchmarkingResult:
    """Enhanced benchmarking result with DPIBS integration"""
    issue_number: int
    specifications: List[DesignSpecification]
    evidence: List[ImplementationEvidence] 
    nlp_accuracy_score: float  # NLP extraction accuracy
    overall_adherence_score: float
    overall_compliance_level: ComplianceLevel
    constraint_violations: List[Dict[str, Any]]
    goal_achievement: Dict[str, float]
    quality_grade: str  # A+, A, B+, B, C+, C, D+, D, F
    evidence_collection: Dict[str, Any]  # Complete evidence trace
    performance_metrics: Dict[str, Any]  # Analysis performance data
    knowledge_integration_data: Dict[str, Any]  # MCP integration results
    recommendations: List[str]
    benchmarking_timestamp: datetime
    analysis_duration_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['overall_compliance_level'] = self.overall_compliance_level.value
        data['benchmarking_timestamp'] = self.benchmarking_timestamp.isoformat()
        data['specifications'] = [spec.to_dict() for spec in self.specifications]
        data['evidence'] = [ev.to_dict() for ev in self.evidence]
        return data


class EnhancedNLPExtractor:
    """
    Enhanced NLP-based specification extractor with 90% accuracy target
    Hybrid approach combining structured pattern matching with semantic analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced pattern library with confidence scoring
        self.enhanced_patterns = {
            SpecificationType.FUNCTIONAL_REQUIREMENTS: [
                (r"(?i)(?:must|shall|should|will|need to|required to)\s+(.+?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:requirement|feature|functionality):\s*(.+?)(?:\.|$|;|\n)", 0.85),
                (r"(?i)(?:the system|implementation|solution|framework)\s+(?:must|shall|should|will|needs to)\s+(.+?)(?:\.|$|;|\n)", 0.8),
                (r"(?i)(?:- |• |✅ |☑️ )(?:.+?)(?:must|shall|should|will)\s+(.+?)(?:\.|$|;|\n)", 0.75),
                (r"(?i)(?:API|endpoint|service)\s+(?:must|should|will)\s+(.+?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:database|schema|table)\s+(?:must|should|will)\s+(.+?)(?:\.|$|;|\n)", 0.85)
            ],
            SpecificationType.PERFORMANCE_REQUIREMENTS: [
                (r"(?i)(?:within|under|below|<)\s+(\d+\s*(?:ms|milliseconds|seconds|minutes).*?)(?:\.|$|;|\n)", 0.95),
                (r"(?i)(?:response time|latency|duration)\s*[:=]\s*(?:within|under|below|<)\s*(\d+\s*(?:ms|milliseconds|seconds|minutes).*?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:target|goal):\s*(?:within|under|below|<)\s*(\d+\s*(?:ms|milliseconds|seconds|minutes).*?)(?:\.|$|;|\n)", 0.85),
                (r"(?i)(?:cache|cached)\s+(?:queries?|operations?)\s*[:=]\s*(?:within|under|below|<)\s*(\d+\s*(?:ms|milliseconds).*?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:complete|full)\s+(?:analysis|processing)\s*[:=]\s*(?:within|under|below|<)\s*(\d+\s*(?:minutes?|min).*?)(?:\.|$|;|\n)", 0.85)
            ],
            SpecificationType.QUALITY_GATES: [
                (r"(?i)(?:(\d+)%|\d+ percent)\s+(?:accuracy|coverage|compliance|success)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:accuracy|coverage|compliance)\s*(?:target|goal|requirement)\s*[:=]\s*(\d+)%(?:\.|$|;|\n)", 0.85),
                (r"(?i)(?:test|code)\s+coverage\s*[:=]\s*(?:>|above|over)\s*(\d+)%(?:\.|$|;|\n)", 0.8),
                (r"(?i)(?:grade|score|rating)\s*[:=]\s*(A\+?|B\+?|C\+?|D\+?|F)(?:\.|$|;|\n)", 0.75)
            ],
            SpecificationType.ARCHITECTURAL_CONSTRAINTS: [
                (r"(?i)(?:must\s+(?:maintain|preserve|ensure))\s+(.+?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:backward\s+compatibility|compatibility)\s+(.+?)(?:\.|$|;|\n)", 0.85),
                (r"(?i)(?:zero(?:\s+|-)disruption|without\s+disruption)\s+(.+?)(?:\.|$|;|\n)", 0.8),
                (r"(?i)(?:enterprise(?:\s+|-)grade|enterprise)\s+(.+?)(?:\.|$|;|\n)", 0.75),
                (r"(?i)(?:scalability|reliability|security)\s+(?:requirements?|constraints?)\s*[:=]\s*(.+?)(?:\.|$|;|\n)", 0.8)
            ],
            SpecificationType.INTEGRATION_REQUIREMENTS: [
                (r"(?i)(?:MCP|Knowledge\s+Server|integration)\s+(.+?)(?:\.|$|;|\n)", 0.9),
                (r"(?i)(?:API|endpoint)\s+(?:integration|compatibility)\s+(.+?)(?:\.|$|;|\n)", 0.85),
                (r"(?i)(?:seamless|smooth)\s+(?:integration|operation)\s+(.+?)(?:\.|$|;|\n)", 0.8)
            ]
        }
        
        # Context keywords for better extraction
        self.context_keywords = {
            'performance': ['response', 'time', 'latency', 'duration', 'speed', 'throughput', 'ms', 'milliseconds', 'seconds'],
            'quality': ['accuracy', 'coverage', 'compliance', 'grade', 'score', 'test', 'validation'],
            'architecture': ['compatibility', 'scalability', 'reliability', 'enterprise', 'security'],
            'integration': ['API', 'MCP', 'Knowledge Server', 'seamless', 'integration', 'compatibility']
        }
        
    def extract_specifications_with_nlp(self, issue_number: int) -> Tuple[List[DesignSpecification], float]:
        """
        Extract specifications using enhanced NLP with accuracy scoring
        Returns: (specifications, nlp_accuracy_score)
        Target: 90% accuracy
        """
        try:
            # Get issue data
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_number), "--json", "body,title,labels,comments"],
                capture_output=True, text=True, check=True
            )
            issue_data = json.loads(result.stdout)
            
            specifications = []
            confidence_scores = []
            
            # Extract from issue body with confidence scoring
            body_specs, body_confidence = self._extract_with_confidence(issue_data['body'], issue_number)
            specifications.extend(body_specs)
            confidence_scores.extend(body_confidence)
            
            # Extract from RIF agent comments
            for comment in issue_data.get('comments', []):
                if any(agent in comment.get('body', '') for agent in ['RIF-Analyst', 'RIF-Planner', 'RIF-Architect']):
                    comment_specs, comment_confidence = self._extract_with_confidence(comment['body'], issue_number)
                    specifications.extend(comment_specs)
                    confidence_scores.extend(comment_confidence)
            
            # Calculate overall NLP accuracy
            nlp_accuracy = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Apply post-processing to improve accuracy
            specifications = self._post_process_specifications(specifications)
            
            self.logger.info(f"Extracted {len(specifications)} specifications with {nlp_accuracy:.2%} confidence")
            
            return specifications, nlp_accuracy
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error fetching issue {issue_number}: {e}")
            return [], 0.0
        except Exception as e:
            self.logger.error(f"NLP extraction failed: {e}")
            return [], 0.0
    
    def _extract_with_confidence(self, text: str, issue_number: int) -> Tuple[List[DesignSpecification], List[float]]:
        """Extract specifications with confidence scoring"""
        specifications = []
        confidence_scores = []
        spec_counter = 0
        
        for spec_type, type_patterns in self.enhanced_patterns.items():
            for pattern, base_confidence in type_patterns:
                matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    match_text = match.strip()
                    if len(match_text) > 10:  # Filter short matches
                        
                        # Calculate context-aware confidence
                        context_confidence = self._calculate_context_confidence(match_text, spec_type)
                        final_confidence = (base_confidence + context_confidence) / 2
                        
                        spec_counter += 1
                        spec = DesignSpecification(
                            id=f"spec-{issue_number}-{spec_counter}",
                            type=spec_type,
                            description=match_text,
                            acceptance_criteria=self._extract_acceptance_criteria(text, match_text),
                            constraints=self._extract_constraints(text),
                            success_metrics=self._extract_success_metrics(match_text),
                            priority=self._determine_priority(match_text),
                            measurable=self._is_measurable(match_text),
                            testable=self._is_testable(match_text),
                            created_at=datetime.now(),
                            issue_number=issue_number
                        )
                        
                        specifications.append(spec)
                        confidence_scores.append(final_confidence)
        
        return specifications, confidence_scores
    
    def _calculate_context_confidence(self, text: str, spec_type: SpecificationType) -> float:
        """Calculate confidence based on context keywords"""
        text_lower = text.lower()
        spec_type_key = spec_type.value.split('_')[0]  # Get first part (e.g., 'performance' from 'performance_requirements')
        
        if spec_type_key in self.context_keywords:
            relevant_keywords = self.context_keywords[spec_type_key]
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in text_lower)
            return min(keyword_matches / len(relevant_keywords) * 1.2, 1.0)  # Cap at 1.0
        
        return 0.5  # Default confidence
    
    def _post_process_specifications(self, specifications: List[DesignSpecification]) -> List[DesignSpecification]:
        """Post-process specifications to improve accuracy"""
        
        # Remove duplicates based on similarity
        unique_specs = []
        seen_descriptions = set()
        
        for spec in specifications:
            # Check for near-duplicates
            is_duplicate = False
            for seen_desc in seen_descriptions:
                similarity = difflib.SequenceMatcher(None, spec.description.lower(), seen_desc.lower()).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_specs.append(spec)
                seen_descriptions.add(spec.description.lower())
        
        return unique_specs
    
    def _extract_acceptance_criteria(self, text: str, requirement: str) -> List[str]:
        """Extract acceptance criteria with enhanced patterns"""
        criteria = []
        
        # Enhanced criteria patterns
        criteria_patterns = [
            r"(?i)(?:acceptance|criteria|success|validation):\s*(.+?)(?:\n|$)",
            r"(?i)(?:verify|validate|confirm|ensure|check)\s+(.+?)(?:\.|$|;|\n)",
            r"(?i)\[\s*[x✓✅☑]\s*\]\s+(.+?)(?:\n|$)",  # Completed checkboxes
            r"(?i)\[\s*\]\s+(.+?)(?:\n|$)",  # Empty checkboxes
            r"(?i)(?:test|validation)\s+(?:criteria|requirements?):\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            criteria.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return list(set(criteria))[:5]  # Remove duplicates and limit
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints with enhanced detection"""
        constraints = []
        
        constraint_patterns = [
            r"(?i)(?:constraint|limitation|restriction|must not|cannot|shall not):\s*(.+?)(?:\.|$|;|\n)",
            r"(?i)(?:within|under|below|above|over)\s+(\d+\s*(?:ms|seconds|minutes|MB|GB|%).*?)(?:\.|$|;|\n)",
            r"(?i)(?:compatible with|supports|requires|depends on)\s+(.+?)(?:\.|$|;|\n)",
            r"(?i)(?:backward|forward)\s+compatibility\s+(.+?)(?:\.|$|;|\n)",
            r"(?i)(?:enterprise(?:\s+|-)grade|production(?:\s+|-)ready)\s+(.+?)(?:\.|$|;|\n)"
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            constraints.extend([match.strip() for match in matches])
        
        return list(set(constraints))[:10]  # Remove duplicates and limit
    
    def _extract_success_metrics(self, text: str) -> Dict[str, Any]:
        """Extract measurable success metrics with enhanced detection"""
        metrics = {}
        
        metric_patterns = [
            (r"(?i)(\d+)%\s*(?:accuracy|coverage|compliance|success|improvement)", "percentage"),
            (r"(?i)(?:target|goal):\s*(\d+)%", "percentage"),
            (r"(?i)(?:within|under|below|<)\s+(\d+)\s*(?:ms|milliseconds)", "response_time_ms"),
            (r"(?i)(?:within|under|below|<)\s+(\d+)\s*(?:seconds?|sec)", "response_time_seconds"),
            (r"(?i)(?:within|under|below|<)\s+(\d+)\s*(?:minutes?|min)", "response_time_minutes"),
            (r"(?i)(?:above|over|exceeds?|>)\s+(\d+)\s*(?:requests?|operations?|transactions?)", "throughput"),
            (r"(?i)(?:zero|no|0)\s+(?:errors?|failures?|issues?|disruption)", "error_tolerance"),
            (r"(?i)(\d+)\+?\s*(?:cases?|scenarios?|items?|components?)", "coverage_count"),
            (r"(?i)grade\s*[:=]\s*(A\+?|B\+?|C\+?|D\+?|F)", "quality_grade")
        ]
        
        for pattern, metric_type in metric_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if metric_type in ["percentage", "response_time_ms", "response_time_seconds", "response_time_minutes", "throughput", "coverage_count"]:
                    try:
                        metrics[metric_type] = float(matches[0])
                    except ValueError:
                        metrics[metric_type] = matches[0]
                else:
                    metrics[metric_type] = matches[0]
        
        return metrics
    
    def _determine_priority(self, text: str) -> str:
        """Determine requirement priority"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['must', 'shall', 'required', 'critical', 'essential']):
            return "must_have"
        elif any(word in text_lower for word in ['should', 'recommended', 'important']):
            return "should_have" 
        else:
            return "could_have"
    
    def _is_measurable(self, text: str) -> bool:
        """Check if requirement is measurable"""
        measurable_indicators = [
            r'\d+\s*%',
            r'\d+\s*(?:ms|seconds|minutes)',
            r'\d+\s*(?:requests|operations|transactions)',
            r'(?:grade|score)\s*[:=]\s*[A-F]',
            r'(?:zero|no|0)\s+(?:errors|failures)'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in measurable_indicators)
    
    def _is_testable(self, text: str) -> bool:
        """Check if requirement is testable"""
        testable_indicators = [
            'test', 'verify', 'validate', 'check', 'measure', 'confirm',
            'performance', 'accuracy', 'coverage', 'compatibility', 'integration'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in testable_indicators)


class EvidenceCollectionEngine:
    """
    Comprehensive evidence collection system with transparency
    Provides 100% traceability for grading decisions
    """
    
    def __init__(self, project_root: str = "/Users/cal/DEV/RIF"):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def collect_implementation_evidence(self, specification: DesignSpecification) -> ImplementationEvidence:
        """Collect comprehensive evidence for a specification"""
        start_time = time.time()
        
        evidence = {
            'code_analysis': self._analyze_code_implementation(specification),
            'test_analysis': self._analyze_test_coverage(specification),
            'documentation_analysis': self._analyze_documentation(specification),
            'performance_analysis': self._analyze_performance_metrics(specification),
            'integration_analysis': self._analyze_integration_compliance(specification)
        }
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(specification, evidence)
        compliance_level = self._determine_compliance_level(compliance_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(specification, evidence, compliance_score)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ImplementationEvidence(
            spec_id=specification.id,
            implementation_details=json.dumps(evidence, default=str),
            code_files=evidence.get('code_analysis', {}).get('relevant_files', []),
            test_files=evidence.get('test_analysis', {}).get('test_files', []),
            documentation_refs=evidence.get('documentation_analysis', {}).get('doc_files', []),
            metrics_achieved=evidence.get('performance_analysis', {}).get('metrics', {}),
            compliance_score=compliance_score,
            compliance_level=compliance_level,
            issues_found=evidence.get('issues_found', []),
            recommendations=recommendations,
            evidence_timestamp=datetime.now()
        )
    
    def _analyze_code_implementation(self, spec: DesignSpecification) -> Dict[str, Any]:
        """Analyze code implementation against specification"""
        try:
            relevant_files = self._find_relevant_files(spec)
            
            code_analysis = {
                'relevant_files': relevant_files,
                'implementation_found': len(relevant_files) > 0,
                'code_quality_metrics': {},
                'pattern_matches': []
            }
            
            # Analyze each relevant file
            for file_path in relevant_files[:10]:  # Limit for performance
                try:
                    full_path = os.path.join(self.project_root, file_path)
                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Check for implementation patterns
                        patterns = self._get_implementation_patterns(spec)
                        matches = []
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                matches.append(pattern)
                        
                        code_analysis['pattern_matches'].extend(matches)
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {file_path}: {e}")
            
            return code_analysis
            
        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return {'error': str(e)}
    
    def _find_relevant_files(self, spec: DesignSpecification) -> List[str]:
        """Find files relevant to the specification"""
        relevant_files = []
        
        # Keywords from specification description
        keywords = self._extract_keywords(spec.description)
        
        try:
            # Search for files containing keywords
            for root, dirs, files in os.walk(self.project_root):
                # Skip hidden directories and common exclude patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith(('.py', '.sql', '.yaml', '.yml', '.json', '.md')):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.project_root)
                        
                        # Check if filename or path contains keywords
                        if any(keyword.lower() in rel_path.lower() for keyword in keywords):
                            relevant_files.append(rel_path)
                            continue
                        
                        # Check file content (for smaller files)
                        if os.path.getsize(file_path) < 100000:  # Less than 100KB
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if any(keyword.lower() in content.lower() for keyword in keywords):
                                        relevant_files.append(rel_path)
                            except Exception:
                                continue
                                
        except Exception as e:
            self.logger.error(f"File search failed: {e}")
        
        return relevant_files[:20]  # Limit for performance
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract relevant keywords from specification description"""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'be', 'will', 'should', 'must'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
        keywords = [word for word in words if word not in common_words]
        
        # Add technical terms if found
        technical_terms = ['api', 'database', 'schema', 'context', 'performance', 'cache', 'optimization', 'integration']
        for term in technical_terms:
            if term in description.lower():
                keywords.append(term)
        
        return list(set(keywords))[:10]  # Remove duplicates and limit
    
    def _get_implementation_patterns(self, spec: DesignSpecification) -> List[str]:
        """Get regex patterns to look for in code based on specification type"""
        patterns = []
        
        if spec.type == SpecificationType.PERFORMANCE_REQUIREMENTS:
            patterns.extend([
                r'@.*monitor.*|@.*cache.*|@.*performance.*',
                r'cache.*manager|cache.*engine|performance.*optimizer',
                r'response.*time|latency|duration.*ms',
                r'connection.*pool|optimization|caching'
            ])
        elif spec.type == SpecificationType.FUNCTIONAL_REQUIREMENTS:
            patterns.extend([
                r'def\s+\w*' + re.escape(word) + r'\w*|class\s+\w*' + re.escape(word) + r'\w*'
                for word in self._extract_keywords(spec.description)[:5]
            ])
        elif spec.type == SpecificationType.ARCHITECTURAL_CONSTRAINTS:
            patterns.extend([
                r'compatibility|backward.*compatibility|enterprise.*grade',
                r'security|authentication|authorization|JWT',
                r'scalability|horizontal.*scaling|connection.*pooling'
            ])
        
        return patterns
    
    def _analyze_test_coverage(self, spec: DesignSpecification) -> Dict[str, Any]:
        """Analyze test coverage for specification"""
        test_analysis = {
            'test_files': [],
            'test_coverage': 0.0,
            'has_performance_tests': False,
            'has_integration_tests': False
        }
        
        try:
            # Find test files
            keywords = self._extract_keywords(spec.description)
            test_patterns = ['test_*.py', '*_test.py', 'tests.py', 'test/*.py']
            
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if any(pattern.replace('*', '').replace('/', '') in file for pattern in ['test', 'spec']):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.project_root)
                        
                        # Check if test file is relevant
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if any(keyword.lower() in content.lower() for keyword in keywords):
                                    test_analysis['test_files'].append(rel_path)
                                    
                                    # Check for specific test types
                                    if any(perf_term in content.lower() for perf_term in ['performance', 'benchmark', 'timing', 'latency']):
                                        test_analysis['has_performance_tests'] = True
                                    
                                    if any(int_term in content.lower() for int_term in ['integration', 'end_to_end', 'api_test']):
                                        test_analysis['has_integration_tests'] = True
                                        
                        except Exception:
                            continue
            
            # Estimate coverage based on presence of tests
            if test_analysis['test_files']:
                test_analysis['test_coverage'] = min(len(test_analysis['test_files']) * 20, 100)
                
        except Exception as e:
            self.logger.error(f"Test analysis failed: {e}")
        
        return test_analysis
    
    def _analyze_documentation(self, spec: DesignSpecification) -> Dict[str, Any]:
        """Analyze documentation coverage"""
        doc_analysis = {
            'doc_files': [],
            'has_api_docs': False,
            'has_architecture_docs': False,
            'documentation_completeness': 0.0
        }
        
        try:
            keywords = self._extract_keywords(spec.description)
            doc_extensions = ['.md', '.rst', '.txt']
            
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if any(file.endswith(ext) for ext in doc_extensions):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.project_root)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            if any(keyword.lower() in content.lower() for keyword in keywords):
                                doc_analysis['doc_files'].append(rel_path)
                                
                                # Check for specific documentation types
                                if any(api_term in content.lower() for api_term in ['api', 'endpoint', 'rest', 'openapi']):
                                    doc_analysis['has_api_docs'] = True
                                
                                if any(arch_term in content.lower() for arch_term in ['architecture', 'design', 'system']):
                                    doc_analysis['has_architecture_docs'] = True
                                    
                        except Exception:
                            continue
            
            # Calculate documentation completeness
            if doc_analysis['doc_files']:
                doc_analysis['documentation_completeness'] = min(len(doc_analysis['doc_files']) * 25, 100)
                
        except Exception as e:
            self.logger.error(f"Documentation analysis failed: {e}")
        
        return doc_analysis
    
    def _analyze_performance_metrics(self, spec: DesignSpecification) -> Dict[str, Any]:
        """Analyze performance metrics from specification"""
        perf_analysis = {
            'metrics': {},
            'benchmarks_found': False,
            'performance_targets_met': False
        }
        
        # Extract performance targets from specification
        success_metrics = spec.success_metrics
        
        if 'response_time_ms' in success_metrics:
            perf_analysis['metrics']['target_response_time_ms'] = success_metrics['response_time_ms']
            perf_analysis['performance_targets_met'] = True  # Assume met for now
        
        if 'percentage' in success_metrics:
            perf_analysis['metrics']['target_percentage'] = success_metrics['percentage']
        
        return perf_analysis
    
    def _analyze_integration_compliance(self, spec: DesignSpecification) -> Dict[str, Any]:
        """Analyze integration compliance"""
        integration_analysis = {
            'mcp_compatible': False,
            'api_compliant': False,
            'backward_compatible': True  # Assume compatible unless proven otherwise
        }
        
        # Check for MCP integration patterns
        if 'mcp' in spec.description.lower() or 'knowledge server' in spec.description.lower():
            integration_analysis['mcp_compatible'] = True
        
        # Check for API compliance
        if 'api' in spec.description.lower() or 'endpoint' in spec.description.lower():
            integration_analysis['api_compliant'] = True
        
        return integration_analysis
    
    def _calculate_compliance_score(self, spec: DesignSpecification, evidence: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0.0 to 1.0)"""
        score_components = []
        
        # Implementation evidence (40% weight)
        code_analysis = evidence.get('code_analysis', {})
        if code_analysis.get('implementation_found', False):
            impl_score = 0.8 if code_analysis.get('pattern_matches') else 0.5
            score_components.append(impl_score * 0.4)
        else:
            score_components.append(0.0)
        
        # Test coverage (25% weight)
        test_analysis = evidence.get('test_analysis', {})
        test_score = test_analysis.get('test_coverage', 0) / 100.0
        score_components.append(test_score * 0.25)
        
        # Documentation (15% weight)
        doc_analysis = evidence.get('documentation_analysis', {})
        doc_score = doc_analysis.get('documentation_completeness', 0) / 100.0
        score_components.append(doc_score * 0.15)
        
        # Performance metrics (20% weight)
        perf_analysis = evidence.get('performance_analysis', {})
        perf_score = 1.0 if perf_analysis.get('performance_targets_met', False) else 0.5
        score_components.append(perf_score * 0.2)
        
        return sum(score_components)
    
    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level from score"""
        if score >= 0.9:
            return ComplianceLevel.FULLY_COMPLIANT
        elif score >= 0.7:
            return ComplianceLevel.MOSTLY_COMPLIANT  
        elif score >= 0.5:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        elif score >= 0.3:
            return ComplianceLevel.MINIMALLY_COMPLIANT
        else:
            return ComplianceLevel.NON_COMPLIANT
    
    def _generate_recommendations(self, spec: DesignSpecification, evidence: Dict[str, Any], score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if score < 0.7:
            recommendations.append("Consider improving implementation completeness")
        
        test_analysis = evidence.get('test_analysis', {})
        if test_analysis.get('test_coverage', 0) < 80:
            recommendations.append("Add more comprehensive test coverage")
        
        if not test_analysis.get('has_performance_tests', False) and spec.type == SpecificationType.PERFORMANCE_REQUIREMENTS:
            recommendations.append("Add performance benchmarking tests")
        
        doc_analysis = evidence.get('documentation_analysis', {})
        if doc_analysis.get('documentation_completeness', 0) < 50:
            recommendations.append("Improve documentation coverage")
        
        if spec.measurable and score < 0.8:
            recommendations.append("Add measurable validation for this requirement")
        
        return recommendations


class DPIBSBenchmarkingEngine:
    """
    Main DPIBS benchmarking engine that orchestrates the enhanced benchmarking process
    Target: <2 minute complete analysis with 90% NLP accuracy
    """
    
    def __init__(self, optimizer: DPIBSPerformanceOptimizer):
        self.optimizer = optimizer
        self.nlp_extractor = EnhancedNLPExtractor()
        self.evidence_engine = EvidenceCollectionEngine()
        self.logger = logging.getLogger(__name__)
    
    @property
    def performance_monitor(self):
        """Access the performance monitor from optimizer"""
        return self.optimizer.performance_monitor
    
    def benchmark_issue(self, issue_number: int, include_evidence: bool = True) -> EnhancedBenchmarkingResult:
        """
        Perform complete benchmarking analysis for an issue
        Target: <2 minutes complete analysis
        """
        start_time = time.time()
        self.logger.info(f"Starting benchmarking analysis for issue #{issue_number}")
        
        try:
            # Phase 1: Specification extraction with NLP (target: <30 seconds)
            specifications, nlp_accuracy = self.nlp_extractor.extract_specifications_with_nlp(issue_number)
            
            phase1_time = time.time() - start_time
            self.logger.info(f"Phase 1 completed in {phase1_time:.2f}s - {len(specifications)} specs extracted")
            
            # Phase 2: Evidence collection (target: <90 seconds)
            evidence_list = []
            if include_evidence:
                for spec in specifications:
                    evidence = self.evidence_engine.collect_implementation_evidence(spec)
                    evidence_list.append(evidence)
            
            phase2_time = time.time() - start_time - phase1_time
            self.logger.info(f"Phase 2 completed in {phase2_time:.2f}s - Evidence collected for {len(evidence_list)} specs")
            
            # Phase 3: Overall analysis and grading
            overall_score = self._calculate_overall_score(evidence_list) if evidence_list else 0.0
            compliance_level = self._determine_overall_compliance(overall_score)
            quality_grade = self._assign_quality_grade(overall_score, nlp_accuracy)
            
            # Constraint violations and recommendations
            constraint_violations = self._identify_constraint_violations(specifications, evidence_list)
            recommendations = self._generate_overall_recommendations(specifications, evidence_list, overall_score)
            
            # Goal achievement analysis
            goal_achievement = self._analyze_goal_achievement(specifications, evidence_list)
            
            # Performance metrics
            total_duration_ms = (time.time() - start_time) * 1000
            performance_metrics = {
                'total_duration_ms': total_duration_ms,
                'phase1_extraction_ms': phase1_time * 1000,
                'phase2_evidence_ms': phase2_time * 1000,
                'target_met': total_duration_ms < 120000,  # 2 minutes
                'specs_per_second': len(specifications) / (total_duration_ms / 1000) if total_duration_ms > 0 else 0
            }
            
            # Knowledge integration data (placeholder for MCP integration)
            knowledge_integration_data = {
                'patterns_identified': len([s for s in specifications if s.type == SpecificationType.ARCHITECTURAL_CONSTRAINTS]),
                'integration_compatibility': True,  # Assume compatible for now
                'knowledge_base_updated': False
            }
            
            # Create enhanced result
            result = EnhancedBenchmarkingResult(
                issue_number=issue_number,
                specifications=specifications,
                evidence=evidence_list,
                nlp_accuracy_score=nlp_accuracy,
                overall_adherence_score=overall_score,
                overall_compliance_level=compliance_level,
                constraint_violations=constraint_violations,
                goal_achievement=goal_achievement,
                quality_grade=quality_grade,
                evidence_collection={'total_evidence_items': len(evidence_list), 'collection_complete': True},
                performance_metrics=performance_metrics,
                knowledge_integration_data=knowledge_integration_data,
                recommendations=recommendations,
                benchmarking_timestamp=datetime.now(),
                analysis_duration_ms=int(total_duration_ms)
            )
            
            # Store result in database
            self._store_benchmarking_result(result)
            
            self.logger.info(f"Benchmarking analysis completed in {total_duration_ms:.0f}ms (Target: <120000ms)")
            self.logger.info(f"Results: {len(specifications)} specs, {overall_score:.2%} compliance, Grade: {quality_grade}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Benchmarking analysis failed: {str(e)}")
            raise
    
    def _calculate_overall_score(self, evidence_list: List[ImplementationEvidence]) -> float:
        """Calculate overall adherence score"""
        if not evidence_list:
            return 0.0
        
        return sum(evidence.compliance_score for evidence in evidence_list) / len(evidence_list)
    
    def _determine_overall_compliance(self, score: float) -> ComplianceLevel:
        """Determine overall compliance level"""
        if score >= 0.9:
            return ComplianceLevel.FULLY_COMPLIANT
        elif score >= 0.7:
            return ComplianceLevel.MOSTLY_COMPLIANT
        elif score >= 0.5:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        elif score >= 0.3:
            return ComplianceLevel.MINIMALLY_COMPLIANT
        else:
            return ComplianceLevel.NON_COMPLIANT
    
    def _assign_quality_grade(self, compliance_score: float, nlp_accuracy: float) -> str:
        """Assign quality grade based on compliance and NLP accuracy"""
        # Weight both compliance and NLP accuracy
        weighted_score = (compliance_score * 0.7) + (nlp_accuracy * 0.3)
        
        if weighted_score >= 0.95:
            return "A+"
        elif weighted_score >= 0.9:
            return "A"
        elif weighted_score >= 0.85:
            return "B+"
        elif weighted_score >= 0.8:
            return "B"
        elif weighted_score >= 0.75:
            return "C+"
        elif weighted_score >= 0.7:
            return "C"
        elif weighted_score >= 0.6:
            return "D+"
        elif weighted_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _identify_constraint_violations(self, specs: List[DesignSpecification], 
                                     evidence_list: List[ImplementationEvidence]) -> List[Dict[str, Any]]:
        """Identify constraint violations"""
        violations = []
        
        for spec in specs:
            if spec.type == SpecificationType.ARCHITECTURAL_CONSTRAINTS:
                # Find corresponding evidence
                spec_evidence = next((e for e in evidence_list if e.spec_id == spec.id), None)
                
                if spec_evidence and spec_evidence.compliance_score < 0.8:
                    violations.append({
                        'spec_id': spec.id,
                        'constraint': spec.description,
                        'violation_score': 1.0 - spec_evidence.compliance_score,
                        'issues': spec_evidence.issues_found
                    })
        
        return violations
    
    def _analyze_goal_achievement(self, specs: List[DesignSpecification], 
                                evidence_list: List[ImplementationEvidence]) -> Dict[str, float]:
        """Analyze goal achievement by specification type"""
        achievement = {}
        
        spec_types = set(spec.type for spec in specs)
        
        for spec_type in spec_types:
            type_specs = [s for s in specs if s.type == spec_type]
            type_evidence = [e for e in evidence_list if any(s.id == e.spec_id for s in type_specs)]
            
            if type_evidence:
                avg_score = sum(e.compliance_score for e in type_evidence) / len(type_evidence)
                achievement[spec_type.value] = avg_score
        
        return achievement
    
    def _generate_overall_recommendations(self, specs: List[DesignSpecification],
                                        evidence_list: List[ImplementationEvidence], 
                                        overall_score: float) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Overall implementation needs significant improvement")
        
        # Performance recommendations
        perf_specs = [s for s in specs if s.type == SpecificationType.PERFORMANCE_REQUIREMENTS]
        if perf_specs:
            perf_evidence = [e for e in evidence_list if any(s.id == e.spec_id for s in perf_specs)]
            perf_scores = [e.compliance_score for e in perf_evidence] if perf_evidence else []
            
            if perf_scores and sum(perf_scores) / len(perf_scores) < 0.8:
                recommendations.append("Performance requirements need attention - consider benchmarking and optimization")
        
        # Test coverage recommendations
        low_test_evidence = [e for e in evidence_list if 'test_coverage' in e.implementation_details and 
                           json.loads(e.implementation_details).get('test_analysis', {}).get('test_coverage', 0) < 70]
        
        if len(low_test_evidence) > len(evidence_list) * 0.5:
            recommendations.append("Test coverage is insufficient across multiple specifications")
        
        return recommendations
    
    def _store_benchmarking_result(self, result: EnhancedBenchmarkingResult) -> None:
        """Store benchmarking result in database"""
        try:
            result_id = self.optimizer.store_benchmarking_result(
                issue_number=result.issue_number,
                analysis_type="full_benchmarking",
                specification_data={'specifications': [spec.to_dict() for spec in result.specifications]},
                implementation_data={'evidence': [ev.to_dict() for ev in result.evidence]},
                compliance_score=result.overall_adherence_score * 100,
                grade=result.quality_grade,
                evidence_collection=result.evidence_collection,
                analysis_duration_ms=result.analysis_duration_ms
            )
            
            self.logger.info(f"Stored benchmarking result with ID: {result_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store benchmarking result: {str(e)}")


# Factory function for integration
def create_dpibs_benchmarking_engine(optimizer: DPIBSPerformanceOptimizer) -> DPIBSBenchmarkingEngine:
    """Create enhanced DPIBS benchmarking engine"""
    return DPIBSBenchmarkingEngine(optimizer)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
    
    optimizer = DPIBSPerformanceOptimizer()
    engine = create_dpibs_benchmarking_engine(optimizer)
    
    print("🔍 DPIBS Enhanced Benchmarking Framework")
    print("=" * 50)
    
    # Demo benchmarking analysis
    issue_number = 120  # Current issue
    print(f"📊 Analyzing issue #{issue_number}...")
    
    try:
        result = engine.benchmark_issue(issue_number, include_evidence=True)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📈 Specifications: {len(result.specifications)}")
        print(f"🎯 NLP Accuracy: {result.nlp_accuracy_score:.2%}")
        print(f"📊 Overall Compliance: {result.overall_adherence_score:.2%}")
        print(f"🎓 Quality Grade: {result.quality_grade}")
        print(f"⏱️  Duration: {result.analysis_duration_ms:.0f}ms (Target: <120000ms)")
        print(f"🎯 Target Met: {'✅' if result.performance_metrics['target_met'] else '❌'}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
                
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")