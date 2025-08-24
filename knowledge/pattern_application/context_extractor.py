"""
Context Extraction Engine

This module extracts structured context information from GitHub issues,
including technology stack detection, complexity assessment, and constraint
extraction for pattern matching and adaptation.
"""

from typing import Dict, List, Any, Optional, Set
import re
import json
import logging
from datetime import datetime

from .core import (
    IssueContext, TechStack, IssueConstraints,
    ContextExtractionInterface, ContextExtractionError
)

logger = logging.getLogger(__name__)


class ContextExtractor(ContextExtractionInterface):
    """
    Context extraction implementation for GitHub issues.
    
    This class parses GitHub issue data and extracts structured context
    information including technology stack, complexity, constraints,
    and domain classification.
    """
    
    def __init__(self):
        """Initialize context extractor with language and framework patterns."""
        self._load_tech_patterns()
        self._load_complexity_indicators()
        self._load_constraint_patterns()
    
    def extract_issue_context(self, issue_data: Dict[str, Any]) -> IssueContext:
        """
        Extract structured context from GitHub issue data.
        
        Args:
            issue_data: GitHub issue data (from gh CLI)
            
        Returns:
            IssueContext with extracted information
        """
        try:
            # Extract basic information
            issue_id = str(issue_data.get('number', 'unknown'))
            title = issue_data.get('title', '')
            description = issue_data.get('body', '')
            labels = [label.get('name', '') for label in issue_data.get('labels', [])]
            
            # Combine text for analysis
            full_text = f"{title}\n\n{description}"
            
            # Extract technology stack
            tech_stack = self.detect_tech_stack(full_text)
            
            # Extract constraints
            constraints = self.extract_constraints(full_text)
            
            # Create preliminary context for complexity assessment
            preliminary_context = IssueContext(
                issue_id=issue_id,
                title=title,
                description=description,
                complexity='medium',  # Will be updated
                tech_stack=tech_stack,
                constraints=constraints,
                domain=self._detect_domain(full_text),
                labels=labels
            )
            
            # Assess complexity
            complexity = self.assess_complexity(preliminary_context)
            preliminary_context.complexity = complexity
            
            # Extract similar issues and patterns from labels/content
            preliminary_context.similar_issues = self._extract_similar_issues(full_text)
            preliminary_context.existing_patterns = self._extract_pattern_references(full_text, labels)
            
            logger.info(f"Extracted context for issue {issue_id}: {tech_stack.primary_language}, {complexity} complexity")
            
            return preliminary_context
            
        except Exception as e:
            logger.error(f"Failed to extract issue context: {str(e)}")
            raise ContextExtractionError(f"Context extraction failed: {str(e)}") from e
    
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
        content_lower = issue_content.lower()
        
        # Detect primary language
        primary_language = self._detect_primary_language(content_lower)
        
        # Detect frameworks
        frameworks = self._detect_frameworks(content_lower, primary_language)
        
        # Detect databases
        databases = self._detect_databases(content_lower)
        
        # Detect tools
        tools = self._detect_tools(content_lower)
        
        # Detect architecture patterns
        architecture_pattern = self._detect_architecture_pattern(content_lower)
        
        # Detect deployment targets
        deployment_target = self._detect_deployment_target(content_lower)
        
        # Use repository data if available to enhance detection
        if repository_data:
            repo_language = repository_data.get('language', '').lower()
            if repo_language and not primary_language:
                primary_language = repo_language
        
        return TechStack(
            primary_language=primary_language,
            frameworks=frameworks,
            databases=databases,
            tools=tools,
            architecture_pattern=architecture_pattern,
            deployment_target=deployment_target
        )
    
    def assess_complexity(self, issue_context: IssueContext) -> str:
        """
        Assess complexity level for an issue.
        
        Args:
            issue_context: Context to assess
            
        Returns:
            Complexity level: 'low', 'medium', 'high', 'very-high'
        """
        complexity_score = 0
        
        # Content-based indicators
        content = f"{issue_context.title} {issue_context.description}".lower()
        
        # Check complexity indicators
        for indicator, score in self.complexity_indicators.items():
            if indicator in content:
                complexity_score += score
        
        # Technology stack complexity
        if issue_context.tech_stack:
            # Multiple languages increase complexity
            if len(issue_context.tech_stack.frameworks) > 2:
                complexity_score += 10
            
            # Database operations add complexity
            if issue_context.tech_stack.databases:
                complexity_score += 5
            
            # Architecture patterns add complexity
            if issue_context.tech_stack.architecture_pattern:
                complexity_score += 10
        
        # Constraint-based complexity
        if issue_context.constraints:
            if issue_context.constraints.quality_gates:
                complexity_score += len(issue_context.constraints.quality_gates) * 3
            
            if issue_context.constraints.performance_requirements:
                complexity_score += 15
            
            if issue_context.constraints.compliance_requirements:
                complexity_score += len(issue_context.constraints.compliance_requirements) * 5
        
        # Label-based complexity
        high_complexity_labels = ['epic', 'major', 'breaking-change', 'architecture', 'refactor']
        for label in issue_context.labels:
            if any(complex_label in label.lower() for complex_label in high_complexity_labels):
                complexity_score += 15
        
        # Map score to complexity level
        if complexity_score >= 50:
            return 'very-high'
        elif complexity_score >= 30:
            return 'high'
        elif complexity_score >= 15:
            return 'medium'
        else:
            return 'low'
    
    def extract_constraints(self, issue_content: str) -> IssueConstraints:
        """
        Extract constraints and requirements from issue content.
        
        Args:
            issue_content: Issue description and comments
            
        Returns:
            IssueConstraints with extracted requirements
        """
        content_lower = issue_content.lower()
        
        # Extract timeline constraints
        timeline = self._extract_timeline(content_lower)
        
        # Extract resource limits
        resource_limits = self._extract_resource_limits(content_lower)
        
        # Extract quality gates
        quality_gates = self._extract_quality_gates(content_lower)
        
        # Extract compliance requirements
        compliance_requirements = self._extract_compliance_requirements(content_lower)
        
        # Extract performance requirements
        performance_requirements = self._extract_performance_requirements(content_lower)
        
        # Extract compatibility requirements
        compatibility_requirements = self._extract_compatibility_requirements(content_lower)
        
        return IssueConstraints(
            timeline=timeline,
            resource_limits=resource_limits,
            quality_gates=quality_gates,
            compliance_requirements=compliance_requirements,
            performance_requirements=performance_requirements,
            compatibility_requirements=compatibility_requirements
        )
    
    # Private helper methods
    
    def _load_tech_patterns(self):
        """Load technology detection patterns."""
        self.language_patterns = {
            'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'typescript', 'ts'],
            'python': ['python', 'py', 'django', 'flask', 'fastapi', 'pytest'],
            'java': ['java', 'spring', 'maven', 'gradle', 'junit'],
            'go': ['golang', 'go', 'gin', 'echo', 'goroutine'],
            'rust': ['rust', 'cargo', 'tokio', 'actix'],
            'csharp': ['c#', 'csharp', '.net', 'asp.net', 'entity framework'],
            'php': ['php', 'laravel', 'symfony', 'composer'],
            'ruby': ['ruby', 'rails', 'gem', 'bundler'],
        }
        
        self.framework_patterns = {
            'javascript': ['react', 'vue', 'angular', 'express', 'next.js', 'gatsby'],
            'python': ['django', 'flask', 'fastapi', 'tornado', 'pyramid'],
            'java': ['spring boot', 'spring', 'hibernate', 'struts'],
            'go': ['gin', 'echo', 'fiber', 'beego'],
            'rust': ['actix', 'warp', 'rocket', 'axum'],
            'csharp': ['asp.net', 'blazor', 'xamarin', 'maui'],
            'php': ['laravel', 'symfony', 'codeigniter', 'cakephp'],
            'ruby': ['rails', 'sinatra', 'hanami'],
        }
        
        self.database_patterns = [
            'postgresql', 'postgres', 'mysql', 'sqlite', 'mongodb', 'redis',
            'elasticsearch', 'dynamodb', 'cassandra', 'neo4j', 'influxdb',
            'oracle', 'sql server', 'mariadb', 'couchdb', 'duckdb'
        ]
        
        self.tool_patterns = [
            'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
            'github actions', 'gitlab ci', 'aws', 'azure', 'gcp',
            'nginx', 'apache', 'traefik', 'consul', 'vault'
        ]
    
    def _load_complexity_indicators(self):
        """Load complexity indicator patterns and their scores."""
        self.complexity_indicators = {
            # High complexity indicators
            'microservices': 20,
            'distributed': 15,
            'scalable': 10,
            'performance': 10,
            'optimization': 10,
            'architecture': 15,
            'refactor': 15,
            'migration': 15,
            'integration': 10,
            'real-time': 15,
            'concurrent': 10,
            'parallel': 10,
            
            # Medium complexity indicators
            'api': 8,
            'database': 8,
            'authentication': 8,
            'authorization': 8,
            'security': 10,
            'testing': 5,
            'deployment': 8,
            'monitoring': 8,
            'logging': 5,
            
            # Lower complexity indicators
            'bug fix': -5,
            'update': 3,
            'configuration': 3,
            'documentation': 2,
            'ui': 5,
            'frontend': 5,
        }
    
    def _load_constraint_patterns(self):
        """Load constraint extraction patterns."""
        self.timeline_patterns = [
            r'deadline[:\s]+([^.\n]+)',
            r'due[:\s]+([^.\n]+)',
            r'by[:\s]+(\w+\s+\d+)',
            r'urgent',
            r'asap',
            r'high priority'
        ]
        
        self.quality_gate_patterns = [
            r'test coverage[:\s]+(\d+)%',
            'code review required',
            'security scan',
            'performance test',
            'integration test',
            'unit test',
            'regression test'
        ]
    
    def _detect_primary_language(self, content: str) -> str:
        """Detect primary programming language from content."""
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                score += content.count(pattern)
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return 'unknown'
    
    def _detect_frameworks(self, content: str, primary_language: str) -> List[str]:
        """Detect frameworks from content."""
        frameworks = []
        
        # Check language-specific frameworks
        if primary_language in self.framework_patterns:
            for framework in self.framework_patterns[primary_language]:
                if framework in content:
                    frameworks.append(framework)
        
        # Check all frameworks if no language-specific ones found
        if not frameworks:
            for lang_frameworks in self.framework_patterns.values():
                for framework in lang_frameworks:
                    if framework in content:
                        frameworks.append(framework)
        
        return list(set(frameworks))
    
    def _detect_databases(self, content: str) -> List[str]:
        """Detect databases from content."""
        databases = []
        for db in self.database_patterns:
            if db in content:
                databases.append(db)
        return databases
    
    def _detect_tools(self, content: str) -> List[str]:
        """Detect tools from content."""
        tools = []
        for tool in self.tool_patterns:
            if tool in content:
                tools.append(tool)
        return tools
    
    def _detect_architecture_pattern(self, content: str) -> Optional[str]:
        """Detect architecture pattern from content."""
        patterns = {
            'microservices': ['microservice', 'microservices', 'service mesh'],
            'monolith': ['monolith', 'monolithic'],
            'serverless': ['serverless', 'lambda', 'function as a service'],
            'event-driven': ['event-driven', 'event sourcing', 'cqrs'],
            'mvc': ['mvc', 'model view controller'],
            'rest': ['rest api', 'restful', 'rest'],
            'graphql': ['graphql', 'graph ql'],
        }
        
        for pattern_name, keywords in patterns.items():
            if any(keyword in content for keyword in keywords):
                return pattern_name
        
        return None
    
    def _detect_deployment_target(self, content: str) -> Optional[str]:
        """Detect deployment target from content."""
        targets = {
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'cloud'],
            'kubernetes': ['kubernetes', 'k8s', 'helm'],
            'docker': ['docker', 'container'],
            'serverless': ['lambda', 'azure functions', 'cloud functions'],
            'traditional': ['vm', 'server', 'on-premise'],
        }
        
        for target_name, keywords in targets.items():
            if any(keyword in content for keyword in keywords):
                return target_name
        
        return None
    
    def _detect_domain(self, content: str) -> str:
        """Detect domain/category of the issue."""
        domains = {
            'backend': ['backend', 'server', 'api', 'database', 'service'],
            'frontend': ['frontend', 'ui', 'react', 'vue', 'angular', 'css'],
            'devops': ['deployment', 'docker', 'kubernetes', 'ci/cd', 'infrastructure'],
            'security': ['security', 'authentication', 'authorization', 'encryption'],
            'performance': ['performance', 'optimization', 'speed', 'memory'],
            'testing': ['test', 'testing', 'unit test', 'integration test'],
            'documentation': ['documentation', 'readme', 'docs', 'guide'],
        }
        
        content_lower = content.lower()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _extract_timeline(self, content: str) -> Optional[str]:
        """Extract timeline constraints from content."""
        for pattern in self.timeline_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        return None
    
    def _extract_resource_limits(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract resource limits from content."""
        limits = {}
        
        # Memory limits
        memory_match = re.search(r'memory[:\s]+(\d+)\s*(gb|mb)', content)
        if memory_match:
            limits['memory'] = f"{memory_match.group(1)}{memory_match.group(2)}"
        
        # CPU limits
        cpu_match = re.search(r'cpu[:\s]+(\d+)', content)
        if cpu_match:
            limits['cpu'] = f"{cpu_match.group(1)} cores"
        
        # Budget limits
        budget_match = re.search(r'budget[:\s]+\$?(\d+)', content)
        if budget_match:
            limits['budget'] = f"${budget_match.group(1)}"
        
        return limits if limits else None
    
    def _extract_quality_gates(self, content: str) -> List[str]:
        """Extract quality gates from content."""
        gates = []
        
        for pattern in self.quality_gate_patterns:
            if isinstance(pattern, str):
                if pattern in content:
                    gates.append(pattern)
            else:
                match = re.search(pattern, content)
                if match:
                    gates.append(match.group(0))
        
        return gates
    
    def _extract_compliance_requirements(self, content: str) -> List[str]:
        """Extract compliance requirements from content."""
        compliance_patterns = [
            'gdpr', 'hipaa', 'pci', 'sox', 'iso 27001', 'fips',
            'accessibility', 'wcag', 'ada compliance'
        ]
        
        requirements = []
        for pattern in compliance_patterns:
            if pattern in content:
                requirements.append(pattern.upper())
        
        return requirements
    
    def _extract_performance_requirements(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract performance requirements from content."""
        requirements = {}
        
        # Response time requirements
        response_match = re.search(r'response time[:\s]+(\d+)\s*(ms|seconds?)', content)
        if response_match:
            requirements['response_time'] = f"{response_match.group(1)} {response_match.group(2)}"
        
        # Throughput requirements
        throughput_match = re.search(r'throughput[:\s]+(\d+)\s*(rps|requests)', content)
        if throughput_match:
            requirements['throughput'] = f"{throughput_match.group(1)} {throughput_match.group(2)}"
        
        # Availability requirements
        availability_match = re.search(r'availability[:\s]+(\d+(?:\.\d+)?)\s*%', content)
        if availability_match:
            requirements['availability'] = f"{availability_match.group(1)}%"
        
        return requirements if requirements else None
    
    def _extract_compatibility_requirements(self, content: str) -> List[str]:
        """Extract compatibility requirements from content."""
        compatibility = []
        
        # Browser compatibility
        browsers = ['chrome', 'firefox', 'safari', 'edge', 'ie']
        for browser in browsers:
            if browser in content:
                compatibility.append(f"{browser} compatibility")
        
        # OS compatibility
        os_systems = ['windows', 'macos', 'linux', 'android', 'ios']
        for os in os_systems:
            if os in content:
                compatibility.append(f"{os} compatibility")
        
        # Version compatibility
        version_match = re.search(r'backward[s]?\s+compatible', content)
        if version_match:
            compatibility.append('backward compatibility')
        
        return compatibility
    
    def _extract_similar_issues(self, content: str) -> List[str]:
        """Extract references to similar issues."""
        issue_references = []
        
        # Look for issue references like #123 or issue #123
        issue_pattern = r'#(\d+)'
        matches = re.findall(issue_pattern, content)
        issue_references.extend(matches)
        
        # Look for "similar to" or "like" references
        similar_pattern = r'similar to[:\s]+#?(\d+)'
        matches = re.findall(similar_pattern, content, re.IGNORECASE)
        issue_references.extend(matches)
        
        return list(set(issue_references))
    
    def _extract_pattern_references(self, content: str, labels: List[str]) -> List[str]:
        """Extract references to existing patterns."""
        patterns = []
        
        # Check for pattern-related labels
        pattern_labels = [label for label in labels if 'pattern' in label.lower()]
        patterns.extend(pattern_labels)
        
        # Look for explicit pattern mentions
        pattern_keywords = ['pattern', 'template', 'approach', 'strategy']
        for keyword in pattern_keywords:
            if keyword in content.lower():
                patterns.append(f"{keyword}_reference")
        
        return patterns