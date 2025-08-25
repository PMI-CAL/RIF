#!/usr/bin/env python3
"""
Issue Decomposition Algorithm for RIF Framework

Decomposes requirements into optimally-sized GitHub issues following BMAD-METHOD principles.
Each issue is designed to be completable in a single Claude Code session with appropriate
context and complexity constraints.

Author: RIF-Implementer
Issue: #242
Dependencies: 
- Issue #240 (requirement_extractor.py) - provides requirements data
- Issue #241 (context_embedder.py) - provides context windows (optional integration)
"""

import re
import json
import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IssueSize(Enum):
    """Issue size categories for single Claude Code session optimization"""
    TINY = "tiny"          # < 30 min, < 50 LOC
    SMALL = "small"        # 30-60 min, 50-100 LOC  
    MEDIUM = "medium"      # 1-2 hours, 100-300 LOC
    LARGE = "large"        # 2-4 hours, 300-500 LOC
    TOO_LARGE = "too_large"  # Needs decomposition


class DecompositionStrategy(Enum):
    """Different strategies for breaking down requirements"""
    LAYER_BASED = "layer_based"      # UI -> Service -> Data layers
    FEATURE_BASED = "feature_based"  # CRUD operations, features
    WORKFLOW_BASED = "workflow_based" # Sequential process steps
    COMPONENT_BASED = "component_based" # By system components
    MIXED = "mixed"                  # Combination of strategies


class ImplementationType(Enum):
    """Types of implementation work for issue categorization"""
    SETUP = "setup"                  # Project setup, scaffolding
    DATABASE = "database"            # Schema, migrations, data layer
    BACKEND = "backend"              # APIs, business logic, services
    FRONTEND = "frontend"            # UI components, user interfaces
    INTEGRATION = "integration"      # Third-party APIs, system integration
    TESTING = "testing"              # Test implementation, validation
    DOCUMENTATION = "documentation"  # Code documentation, guides
    CONFIGURATION = "configuration"  # Environment setup, configs


@dataclass
class DecomposedIssue:
    """A decomposed GitHub issue with complete metadata"""
    id: str
    title: str
    description: str
    requirement_id: str
    parent_epic: Optional[str]
    size: IssueSize
    estimated_loc: int
    estimated_time_minutes: int
    dependencies: List[str]
    sequence_order: int
    context_tokens: int
    implementation_type: ImplementationType
    labels: List[str]
    milestone: Optional[str]
    acceptance_criteria: List[str]
    technical_notes: List[str]
    risk_factors: List[str]
    github_ready: bool = True


@dataclass
class DecompositionResult:
    """Complete decomposition result with metadata"""
    issues: List[DecomposedIssue]
    statistics: Dict
    dependency_graph: Dict
    sequence_plan: List[str]
    strategy_used: DecompositionStrategy
    decomposition_metadata: Dict


class ComplexityCalculator:
    """Calculates complexity scores for accurate size estimation"""
    
    COMPLEXITY_FACTORS = {
        'lines_of_description': 0.5,
        'acceptance_criteria': 1.0,
        'dependencies': 1.5,
        'integration_points': 2.0,
        'data_operations': 1.0,
        'ui_components': 0.8,
        'business_rules': 1.5,
        'external_apis': 2.5,
        'security_requirements': 2.0,
        'performance_requirements': 1.5
    }
    
    @staticmethod
    def calculate_complexity(requirement: Dict, additional_context: str = "") -> float:
        """Calculate complexity score for accurate decomposition"""
        content = requirement.get('description', '') + additional_context
        title = requirement.get('title', '')
        full_text = (title + " " + content).lower()
        
        factors = {
            'lines_of_description': len(content.split('\n')),
            'acceptance_criteria': len(requirement.get('acceptance_criteria', [])) * 2,
            'dependencies': len(requirement.get('dependencies', [])) * 3,
            'integration_points': ComplexityCalculator._count_integrations(full_text),
            'data_operations': ComplexityCalculator._count_data_operations(full_text),
            'ui_components': ComplexityCalculator._count_ui_components(full_text),
            'business_rules': ComplexityCalculator._count_business_rules(full_text),
            'external_apis': ComplexityCalculator._count_external_apis(full_text),
            'security_requirements': ComplexityCalculator._count_security_requirements(full_text),
            'performance_requirements': ComplexityCalculator._count_performance_requirements(full_text)
        }
        
        complexity_score = sum(
            factors[key] * ComplexityCalculator.COMPLEXITY_FACTORS[key] 
            for key in factors
        )
        
        return complexity_score
    
    @staticmethod
    def _count_integrations(content: str) -> int:
        """Count integration complexity indicators"""
        integration_keywords = ['api', 'webhook', 'third-party', 'external', 'integration', 
                               'oauth', 'sso', 'microservice', 'service']
        return sum(1 for keyword in integration_keywords if keyword in content)
    
    @staticmethod
    def _count_data_operations(content: str) -> int:
        """Count data operation complexity indicators"""
        data_keywords = ['database', 'crud', 'query', 'migration', 'schema', 'table', 
                        'index', 'transaction', 'sql', 'nosql']
        return sum(1 for keyword in data_keywords if keyword in content)
    
    @staticmethod
    def _count_ui_components(content: str) -> int:
        """Count UI complexity indicators"""
        ui_keywords = ['form', 'button', 'modal', 'table', 'chart', 'dashboard', 
                      'responsive', 'mobile', 'ui', 'ux', 'interface']
        return sum(1 for keyword in ui_keywords if keyword in content)
    
    @staticmethod
    def _count_business_rules(content: str) -> int:
        """Count business rule complexity indicators"""
        business_keywords = ['validation', 'business rule', 'policy', 'workflow', 
                           'approval', 'permission', 'role', 'calculation']
        return sum(1 for keyword in business_keywords if keyword in content)
    
    @staticmethod
    def _count_external_apis(content: str) -> int:
        """Count external API complexity indicators"""
        api_keywords = ['rest api', 'graphql', 'soap', 'json', 'xml', 'http', 
                       'endpoint', 'service call', 'remote']
        return sum(1 for keyword in api_keywords if keyword in content)
    
    @staticmethod
    def _count_security_requirements(content: str) -> int:
        """Count security complexity indicators"""
        security_keywords = ['authentication', 'authorization', 'encryption', 'security',
                           'token', 'certificate', 'compliance', 'audit']
        return sum(1 for keyword in security_keywords if keyword in content)
    
    @staticmethod
    def _count_performance_requirements(content: str) -> int:
        """Count performance complexity indicators"""
        perf_keywords = ['performance', 'optimization', 'caching', 'scalability',
                        'load', 'speed', 'latency', 'throughput']
        return sum(1 for keyword in perf_keywords if keyword in content)


class DecompositionStrategies:
    """Different strategies for breaking down requirements"""
    
    @staticmethod
    def layer_based_decomposition(requirement: Dict, complexity: float) -> List[Dict]:
        """Decompose by architectural layers (UI -> Service -> Data)"""
        layers = []
        content = requirement.get('description', '').lower()
        
        # Determine which layers are needed
        needs_ui = any(keyword in content for keyword in 
                      ['interface', 'ui', 'form', 'screen', 'display', 'user'])
        needs_api = any(keyword in content for keyword in 
                       ['api', 'service', 'endpoint', 'business', 'logic'])
        needs_data = any(keyword in content for keyword in 
                        ['database', 'storage', 'persist', 'data', 'table'])
        
        base_complexity = complexity / 3  # Distribute complexity across layers
        
        if needs_data or not (needs_ui or needs_api):
            layers.append({
                'name': 'Data Layer Implementation',
                'type': ImplementationType.DATABASE.value,
                'estimated_loc': min(int(base_complexity * 25), 200),
                'estimated_time': min(int(base_complexity * 15), 120),
                'dependencies': [],
                'description': 'Implement database schema, models, and data access layer'
            })
        
        if needs_api or not (needs_ui or needs_data):
            api_deps = ['Data Layer Implementation'] if needs_data else []
            layers.append({
                'name': 'Business Logic & API',
                'type': ImplementationType.BACKEND.value,
                'estimated_loc': min(int(base_complexity * 30), 250),
                'estimated_time': min(int(base_complexity * 20), 150),
                'dependencies': api_deps,
                'description': 'Implement business logic, API endpoints, and service layer'
            })
        
        if needs_ui:
            ui_deps = []
            if needs_api:
                ui_deps.append('Business Logic & API')
            elif needs_data:
                ui_deps.append('Data Layer Implementation')
                
            layers.append({
                'name': 'User Interface Components',
                'type': ImplementationType.FRONTEND.value,
                'estimated_loc': min(int(base_complexity * 35), 300),
                'estimated_time': min(int(base_complexity * 25), 180),
                'dependencies': ui_deps,
                'description': 'Implement user interface components and interactions'
            })
        
        return layers if layers else [{'name': 'Core Implementation', 'type': 'backend'}]
    
    @staticmethod
    def feature_based_decomposition(requirement: Dict, complexity: float) -> List[Dict]:
        """Decompose by feature boundaries (CRUD, functionality)"""
        features = []
        content = requirement.get('description', '').lower()
        
        # Extract feature operations
        feature_operations = {
            'create': ['create', 'add', 'insert', 'new'],
            'read': ['read', 'view', 'display', 'show', 'list', 'get'],
            'update': ['update', 'edit', 'modify', 'change'],
            'delete': ['delete', 'remove', 'destroy'],
            'search': ['search', 'filter', 'find', 'query'],
            'validate': ['validate', 'verify', 'check', 'confirm'],
            'export': ['export', 'download', 'extract'],
            'import': ['import', 'upload', 'load'],
            'authenticate': ['login', 'authenticate', 'signin', 'auth'],
            'authorize': ['authorize', 'permission', 'access', 'role']
        }
        
        found_operations = []
        for operation, keywords in feature_operations.items():
            if any(keyword in content for keyword in keywords):
                found_operations.append(operation)
        
        # If no specific operations found, create generic features
        if not found_operations:
            found_operations = ['core_functionality']
        
        base_complexity = complexity / len(found_operations)
        
        for operation in found_operations:
            if operation == 'core_functionality':
                features.append({
                    'name': 'Core Functionality',
                    'type': ImplementationType.BACKEND.value,
                    'estimated_loc': min(int(base_complexity * 30), 250),
                    'estimated_time': min(int(base_complexity * 20), 120),
                    'dependencies': [],
                    'description': 'Implement main functionality and core features'
                })
            else:
                features.append({
                    'name': f'{operation.capitalize()} Feature',
                    'type': ImplementationType.BACKEND.value,
                    'estimated_loc': min(int(base_complexity * 25), 200),
                    'estimated_time': min(int(base_complexity * 15), 90),
                    'dependencies': [],
                    'description': f'Implement {operation} functionality'
                })
        
        return features
    
    @staticmethod
    def workflow_based_decomposition(requirement: Dict, complexity: float) -> List[Dict]:
        """Decompose by workflow steps"""
        steps = []
        
        # Check if requirement has explicit steps
        description = requirement.get('description', '')
        acceptance_criteria = requirement.get('acceptance_criteria', [])
        
        # Look for numbered steps, bullet points, or process indicators
        step_patterns = [
            r'(?:step|phase)\s*(\d+)[:\s]*([^\n]+)',
            r'(\d+)\.\s*([^\n]+)',
            r'[-*]\s*([^\n]+)',
            r'(?:first|then|next|finally)[,\s]*([^\n]+)'
        ]
        
        found_steps = []
        for pattern in step_patterns:
            matches = re.finditer(pattern, description + '\n'.join(acceptance_criteria), re.IGNORECASE)
            for match in matches:
                step_text = match.group(-1).strip()  # Last group is always the description
                if len(step_text) > 10 and step_text not in [s['description'] for s in found_steps]:
                    found_steps.append({
                        'description': step_text,
                        'order': len(found_steps) + 1
                    })
        
        # If no explicit steps found, create logical workflow steps
        if not found_steps:
            workflow_keywords = {
                'initialization': ['setup', 'initialize', 'configure', 'prepare'],
                'processing': ['process', 'calculate', 'transform', 'execute'],
                'validation': ['validate', 'verify', 'check', 'confirm'],
                'completion': ['complete', 'finish', 'finalize', 'save']
            }
            
            content_lower = description.lower()
            for phase, keywords in workflow_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    found_steps.append({
                        'description': f'{phase.capitalize()} phase',
                        'order': len(found_steps) + 1
                    })
        
        # Ensure at least one step
        if not found_steps:
            found_steps = [{'description': 'Implementation', 'order': 1}]
        
        base_complexity = complexity / len(found_steps)
        
        for i, step in enumerate(found_steps):
            dependencies = [f"Step {i}: {found_steps[i-1]['description'][:30]}..."] if i > 0 else []
            
            steps.append({
                'name': f"Step {step['order']}: {step['description'][:50]}",
                'type': ImplementationType.BACKEND.value,
                'estimated_loc': min(int(base_complexity * 25), 200),
                'estimated_time': min(int(base_complexity * 15), 90),
                'dependencies': dependencies,
                'description': step['description']
            })
        
        return steps
    
    @staticmethod
    def component_based_decomposition(requirement: Dict, complexity: float) -> List[Dict]:
        """Decompose by system components"""
        components = []
        content = requirement.get('description', '').lower()
        
        # Identify system components mentioned
        component_keywords = {
            'authentication': ['auth', 'login', 'user', 'session', 'token'],
            'authorization': ['permission', 'role', 'access', 'security'],
            'notification': ['notify', 'email', 'alert', 'message'],
            'reporting': ['report', 'analytics', 'dashboard', 'metrics'],
            'integration': ['api', 'webhook', 'third-party', 'external'],
            'storage': ['database', 'file', 'storage', 'persist'],
            'processing': ['process', 'calculate', 'transform', 'logic'],
            'interface': ['ui', 'form', 'screen', 'display']
        }
        
        found_components = []
        for component, keywords in component_keywords.items():
            if any(keyword in content for keyword in keywords):
                found_components.append(component)
        
        # Default to core component if none identified
        if not found_components:
            found_components = ['core']
        
        base_complexity = complexity / len(found_components)
        
        for component in found_components:
            component_type = ImplementationType.BACKEND.value
            if component == 'interface':
                component_type = ImplementationType.FRONTEND.value
            elif component == 'storage':
                component_type = ImplementationType.DATABASE.value
            elif component == 'integration':
                component_type = ImplementationType.INTEGRATION.value
            
            components.append({
                'name': f'{component.capitalize()} Component',
                'type': component_type,
                'estimated_loc': min(int(base_complexity * 30), 250),
                'estimated_time': min(int(base_complexity * 20), 120),
                'dependencies': [],
                'description': f'Implement {component} component functionality'
            })
        
        return components


class IssueDecomposer:
    """
    Decomposes requirements into optimally-sized GitHub issues.
    Follows BMAD-METHOD principle of self-contained story files.
    """
    
    # Optimal targets for Claude Code sessions
    TARGETS = {
        'max_loc_per_issue': 300,      # Maximum lines of code
        'max_time_minutes': 120,        # Maximum time (2 hours)
        'max_complexity_score': 20,     # Complexity threshold
        'optimal_token_count': 2000,    # Optimal context size
        'max_token_count': 2500         # Maximum context size
    }
    
    def __init__(self, requirements: Dict, context_windows: Optional[Dict] = None):
        """
        Initialize the issue decomposer
        
        Args:
            requirements: Requirements data from RequirementExtractor
            context_windows: Context window data from ContextEmbedder (optional)
        """
        self.requirements = requirements
        self.context_windows = context_windows or {}
        self.issues: List[DecomposedIssue] = []
        self.dependency_graph = nx.DiGraph()
        self.issue_counter = 0
        self.complexity_calculator = ComplexityCalculator()
        
        logger.info("IssueDecomposer initialized")
    
    def decompose_all_requirements(self, strategy: DecompositionStrategy = DecompositionStrategy.MIXED) -> DecompositionResult:
        """
        Main decomposition method - breaks down all requirements into issues
        
        Args:
            strategy: Decomposition strategy to use
            
        Returns:
            DecompositionResult with all decomposed issues and metadata
        """
        logger.info(f"Starting decomposition with strategy: {strategy.value}")
        start_time = datetime.now()
        
        # Clear previous results
        self.issues = []
        self.dependency_graph = nx.DiGraph()
        self.issue_counter = 0
        
        # Process epics first (they contain multiple stories/requirements)
        epics_processed = 0
        for epic in self.requirements.get('epics', []):
            self._decompose_epic(epic, strategy)
            epics_processed += 1
        
        # Process standalone stories
        stories_processed = 0
        for story in self.requirements.get('stories', []):
            if not self._is_part_of_epic(story):
                self._decompose_story(story, strategy)
                stories_processed += 1
        
        # Process remaining standalone requirements
        requirements_processed = 0
        for req in self.requirements.get('requirements', []):
            if not self._is_already_decomposed(req):
                self._decompose_requirement(req, strategy)
                requirements_processed += 1
        
        # Build dependency relationships
        self._build_dependency_graph()
        
        # Optimize issue sequence for efficient implementation
        sequence_plan = self._optimize_sequence()
        
        # Validate all issues meet single-session criteria
        self._validate_decomposition()
        
        # Generate statistics and metadata
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        statistics = self._generate_statistics()
        dependency_graph_data = self._export_dependency_graph()
        
        result = DecompositionResult(
            issues=self.issues,
            statistics=statistics,
            dependency_graph=dependency_graph_data,
            sequence_plan=sequence_plan,
            strategy_used=strategy,
            decomposition_metadata={
                'decomposed_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'epics_processed': epics_processed,
                'stories_processed': stories_processed,
                'requirements_processed': requirements_processed,
                'total_issues_created': len(self.issues),
                'strategy': strategy.value
            }
        )
        
        logger.info(f"Decomposition completed in {duration:.2f} seconds")
        logger.info(f"Created {len(self.issues)} issues from {epics_processed} epics, {stories_processed} stories, {requirements_processed} requirements")
        
        return result
    
    def _decompose_epic(self, epic: Dict, strategy: DecompositionStrategy) -> List[DecomposedIssue]:
        """Decompose an epic into multiple issues"""
        logger.debug(f"Decomposing epic: {epic.get('title', 'Unknown')}")
        
        epic_issues = []
        epic_id = epic.get('id', '')
        
        # Calculate epic complexity
        complexity = self.complexity_calculator.calculate_complexity(epic)
        
        # Decide decomposition approach based on complexity
        if complexity <= self.TARGETS['max_complexity_score']:
            # Simple epic - create foundation + implementation issues
            epic_issues.extend(self._create_simple_epic_issues(epic, strategy))
        else:
            # Complex epic - needs detailed breakdown
            epic_issues.extend(self._create_complex_epic_issues(epic, strategy, complexity))
        
        # Add epic issues to main collection
        self.issues.extend(epic_issues)
        
        # Build dependencies within epic
        self._add_epic_dependencies(epic_issues)
        
        return epic_issues
    
    def _add_epic_dependencies(self, epic_issues: List[DecomposedIssue]):
        """Add dependency relationships within an epic"""
        # For epic issues, create sequential dependencies where appropriate
        setup_issues = [issue for issue in epic_issues if issue.implementation_type == ImplementationType.SETUP]
        impl_issues = [issue for issue in epic_issues if issue.implementation_type in [
            ImplementationType.BACKEND, ImplementationType.FRONTEND, ImplementationType.DATABASE
        ]]
        integration_issues = [issue for issue in epic_issues if issue.implementation_type == ImplementationType.INTEGRATION]
        test_issues = [issue for issue in epic_issues if issue.implementation_type == ImplementationType.TESTING]
        
        # Implementation issues depend on setup
        for issue in impl_issues:
            for setup_issue in setup_issues:
                if setup_issue.id not in issue.dependencies:
                    issue.dependencies.append(setup_issue.id)
        
        # Integration issues depend on implementation
        for issue in integration_issues:
            for impl_issue in impl_issues:
                if impl_issue.id not in issue.dependencies:
                    issue.dependencies.append(impl_issue.id)
        
        # Test issues depend on integration (if exists) or implementation
        test_deps = integration_issues if integration_issues else impl_issues
        for issue in test_issues:
            for dep_issue in test_deps:
                if dep_issue.id not in issue.dependencies:
                    issue.dependencies.append(dep_issue.id)
    
    def _create_simple_epic_issues(self, epic: Dict, strategy: DecompositionStrategy) -> List[DecomposedIssue]:
        """Create issues for a simple epic"""
        issues = []
        epic_id = epic.get('id', '')
        epic_title = epic.get('title', 'Unknown Epic')
        
        # 1. Setup issue
        setup_issue = self._create_issue(
            title=f"Setup: Initialize {epic_title} Foundation",
            description=self._generate_setup_description(epic),
            requirement_id=epic_id,
            parent_epic=epic_id,
            implementation_type=ImplementationType.SETUP,
            estimated_loc=50,
            estimated_time=30,
            dependencies=[],
            acceptance_criteria=self._generate_setup_criteria(epic)
        )
        issues.append(setup_issue)
        
        # 2. Core implementation issue
        impl_issue = self._create_issue(
            title=f"Implement: {epic_title} Core Features",
            description=self._generate_implementation_description(epic),
            requirement_id=epic_id,
            parent_epic=epic_id,
            implementation_type=ImplementationType.BACKEND,
            estimated_loc=200,
            estimated_time=120,
            dependencies=[setup_issue.id],
            acceptance_criteria=epic.get('acceptance_criteria', [])
        )
        issues.append(impl_issue)
        
        # 3. Testing issue (if complex enough)
        complexity = self.complexity_calculator.calculate_complexity(epic)
        if complexity > 10:
            test_issue = self._create_issue(
                title=f"Test: {epic_title} Validation",
                description=self._generate_testing_description(epic),
                requirement_id=epic_id,
                parent_epic=epic_id,
                implementation_type=ImplementationType.TESTING,
                estimated_loc=80,
                estimated_time=60,
                dependencies=[impl_issue.id],
                acceptance_criteria=self._generate_test_criteria(epic)
            )
            issues.append(test_issue)
        
        return issues
    
    def _create_complex_epic_issues(self, epic: Dict, strategy: DecompositionStrategy, complexity: float) -> List[DecomposedIssue]:
        """Create issues for a complex epic requiring detailed breakdown"""
        issues = []
        epic_id = epic.get('id', '')
        epic_title = epic.get('title', 'Unknown Epic')
        
        # 1. Always start with setup
        setup_issue = self._create_issue(
            title=f"Setup: {epic_title} Project Structure",
            description=self._generate_setup_description(epic),
            requirement_id=epic_id,
            parent_epic=epic_id,
            implementation_type=ImplementationType.SETUP,
            estimated_loc=80,
            estimated_time=45,
            dependencies=[],
            acceptance_criteria=self._generate_setup_criteria(epic)
        )
        issues.append(setup_issue)
        
        # 2. Break down into components using selected strategy
        components = self._apply_decomposition_strategy(epic, strategy, complexity)
        
        previous_issue_ids = [setup_issue.id]
        
        for component in components:
            component_issue = self._create_issue(
                title=f"{epic_title}: {component['name']}",
                description=component.get('description', f"Implement {component['name']}"),
                requirement_id=epic_id,
                parent_epic=epic_id,
                implementation_type=ImplementationType(component.get('type', 'backend')),
                estimated_loc=component.get('estimated_loc', 150),
                estimated_time=component.get('estimated_time', 90),
                dependencies=component.get('dependencies', previous_issue_ids),
                acceptance_criteria=self._generate_component_criteria(component)
            )
            issues.append(component_issue)
            
            # For sequential dependencies, each component depends on the previous
            if strategy in [DecompositionStrategy.WORKFLOW_BASED, DecompositionStrategy.LAYER_BASED]:
                previous_issue_ids = [component_issue.id]
            else:
                # For parallel work, all depend on setup
                previous_issue_ids = [setup_issue.id]
        
        # 3. Integration issue for complex epics
        if len(components) > 2:
            integration_issue = self._create_issue(
                title=f"Integration: {epic_title} System Integration",
                description=self._generate_integration_description(epic, components),
                requirement_id=epic_id,
                parent_epic=epic_id,
                implementation_type=ImplementationType.INTEGRATION,
                estimated_loc=100,
                estimated_time=75,
                dependencies=[issue.id for issue in issues[-len(components):]],  # Depend on all components
                acceptance_criteria=self._generate_integration_criteria(epic, components)
            )
            issues.append(integration_issue)
        
        # 4. Final testing issue
        test_issue = self._create_issue(
            title=f"Test: {epic_title} End-to-End Validation",
            description=self._generate_testing_description(epic),
            requirement_id=epic_id,
            parent_epic=epic_id,
            implementation_type=ImplementationType.TESTING,
            estimated_loc=120,
            estimated_time=90,
            dependencies=[issues[-1].id],  # Depend on last implementation issue
            acceptance_criteria=self._generate_test_criteria(epic)
        )
        issues.append(test_issue)
        
        return issues
    
    def _decompose_story(self, story: Dict, strategy: DecompositionStrategy) -> List[DecomposedIssue]:
        """Decompose a user story into implementation issues"""
        logger.debug(f"Decomposing story: {story.get('title', 'Unknown')}")
        
        story_complexity = self.complexity_calculator.calculate_complexity(story)
        
        # Simple story - single issue
        if story_complexity <= self.TARGETS['max_complexity_score'] and \
           story.get('estimated_points', 0) <= 5:
            issue = self._create_story_issue(story)
            self.issues.append(issue)
            return [issue]
        
        # Complex story - break down
        return self._decompose_complex_story(story, strategy, story_complexity)
    
    def _decompose_complex_story(self, story: Dict, strategy: DecompositionStrategy, complexity: float) -> List[DecomposedIssue]:
        """Decompose a complex story into multiple issues"""
        issues = []
        story_id = story.get('id', '')
        story_title = story.get('title', 'Unknown Story')
        
        # Apply decomposition strategy to story
        components = self._apply_decomposition_strategy(story, strategy, complexity)
        
        for i, component in enumerate(components):
            # Build dependencies: later components depend on earlier ones (for workflow/layer strategies)
            dependencies = []
            if strategy in [DecompositionStrategy.WORKFLOW_BASED, DecompositionStrategy.LAYER_BASED] and i > 0:
                dependencies = [issues[-1].id]
            
            component_issue = self._create_issue(
                title=f"{story_title}: {component['name']}",
                description=component.get('description', f"Implement {component['name']} for user story"),
                requirement_id=story_id,
                parent_epic=story.get('epic_id'),
                implementation_type=ImplementationType(component.get('type', 'backend')),
                estimated_loc=component.get('estimated_loc', 120),
                estimated_time=component.get('estimated_time', 75),
                dependencies=dependencies,
                acceptance_criteria=self._extract_component_criteria(story, component)
            )
            issues.append(component_issue)
            self.issues.append(component_issue)
        
        return issues
    
    def _decompose_requirement(self, requirement: Dict, strategy: DecompositionStrategy) -> List[DecomposedIssue]:
        """Decompose a standalone requirement into issues"""
        logger.debug(f"Decomposing requirement: {requirement.get('title', 'Unknown')}")
        
        req_complexity = self.complexity_calculator.calculate_complexity(requirement)
        
        # Simple requirement - single issue
        if req_complexity <= self.TARGETS['max_complexity_score']:
            issue = self._create_requirement_issue(requirement)
            self.issues.append(issue)
            return [issue]
        
        # Complex requirement - break down
        return self._decompose_complex_requirement(requirement, strategy, req_complexity)
    
    def _decompose_complex_requirement(self, requirement: Dict, strategy: DecompositionStrategy, complexity: float) -> List[DecomposedIssue]:
        """Decompose a complex requirement into multiple issues"""
        issues = []
        req_id = requirement.get('id', '')
        req_title = requirement.get('title', 'Unknown Requirement')
        
        # Apply decomposition strategy
        components = self._apply_decomposition_strategy(requirement, strategy, complexity)
        
        for i, component in enumerate(components):
            dependencies = []
            if strategy in [DecompositionStrategy.WORKFLOW_BASED, DecompositionStrategy.LAYER_BASED] and i > 0:
                dependencies = [issues[-1].id]
            
            component_issue = self._create_issue(
                title=f"{req_title}: {component['name']}",
                description=component.get('description', f"Implement {component['name']}"),
                requirement_id=req_id,
                parent_epic=None,
                implementation_type=ImplementationType(component.get('type', 'backend')),
                estimated_loc=component.get('estimated_loc', 100),
                estimated_time=component.get('estimated_time', 60),
                dependencies=dependencies,
                acceptance_criteria=self._extract_component_criteria(requirement, component)
            )
            issues.append(component_issue)
            self.issues.append(component_issue)
        
        return issues
    
    def _apply_decomposition_strategy(self, item: Dict, strategy: DecompositionStrategy, complexity: float) -> List[Dict]:
        """Apply the selected decomposition strategy to an item"""
        if strategy == DecompositionStrategy.LAYER_BASED:
            return DecompositionStrategies.layer_based_decomposition(item, complexity)
        elif strategy == DecompositionStrategy.FEATURE_BASED:
            return DecompositionStrategies.feature_based_decomposition(item, complexity)
        elif strategy == DecompositionStrategy.WORKFLOW_BASED:
            return DecompositionStrategies.workflow_based_decomposition(item, complexity)
        elif strategy == DecompositionStrategy.COMPONENT_BASED:
            return DecompositionStrategies.component_based_decomposition(item, complexity)
        else:  # MIXED strategy
            # Choose best strategy based on content analysis
            content = item.get('description', '').lower()
            
            # Count indicators for each strategy
            layer_indicators = sum(1 for word in ['ui', 'frontend', 'backend', 'database', 'api'] if word in content)
            feature_indicators = sum(1 for word in ['create', 'read', 'update', 'delete', 'search'] if word in content)
            workflow_indicators = sum(1 for word in ['step', 'process', 'workflow', 'sequence', 'then'] if word in content)
            component_indicators = sum(1 for word in ['component', 'module', 'service', 'system'] if word in content)
            
            # Select strategy with highest indicator count
            strategy_scores = {
                DecompositionStrategy.LAYER_BASED: layer_indicators,
                DecompositionStrategy.FEATURE_BASED: feature_indicators,
                DecompositionStrategy.WORKFLOW_BASED: workflow_indicators,
                DecompositionStrategy.COMPONENT_BASED: component_indicators
            }
            
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            return self._apply_decomposition_strategy(item, best_strategy, complexity)
    
    def _create_issue(self, title: str, description: str, requirement_id: str, 
                     parent_epic: Optional[str], implementation_type: ImplementationType,
                     estimated_loc: int, estimated_time: int, dependencies: List[str],
                     acceptance_criteria: List[str]) -> DecomposedIssue:
        """Create a decomposed issue with complete metadata"""
        self.issue_counter += 1
        issue_id = f"ISSUE-{self.issue_counter:03d}"
        
        # Calculate context tokens needed
        context_tokens = self._calculate_context_tokens(description, acceptance_criteria, dependencies)
        
        # Determine issue size
        issue_size = self._determine_issue_size(estimated_loc, estimated_time, context_tokens)
        
        # Generate labels
        labels = self._generate_issue_labels(implementation_type, issue_size, parent_epic)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(description, estimated_loc, estimated_time, len(dependencies))
        
        # Generate technical notes
        technical_notes = self._generate_technical_notes(description, implementation_type)
        
        issue = DecomposedIssue(
            id=issue_id,
            title=title,
            description=self._enrich_issue_description(description, requirement_id, implementation_type),
            requirement_id=requirement_id,
            parent_epic=parent_epic,
            size=issue_size,
            estimated_loc=estimated_loc,
            estimated_time_minutes=estimated_time,
            dependencies=dependencies,
            sequence_order=0,  # Will be set by optimizer
            context_tokens=context_tokens,
            implementation_type=implementation_type,
            labels=labels,
            milestone=self._determine_milestone(parent_epic, implementation_type),
            acceptance_criteria=acceptance_criteria,
            technical_notes=technical_notes,
            risk_factors=risk_factors,
            github_ready=True
        )
        
        return issue
    
    def _create_story_issue(self, story: Dict) -> DecomposedIssue:
        """Create a single issue for a simple story"""
        story_id = story.get('id', '')
        story_title = story.get('title', 'Unknown Story')
        
        # Estimate implementation details
        estimated_points = story.get('estimated_points', 3)
        estimated_loc = min(estimated_points * 30, self.TARGETS['max_loc_per_issue'])
        estimated_time = min(estimated_points * 20, self.TARGETS['max_time_minutes'])
        
        # Determine implementation type based on story content
        impl_type = self._determine_implementation_type(story.get('description', ''))
        
        return self._create_issue(
            title=f"Story: {story_title}",
            description=self._format_story_description(story),
            requirement_id=story_id,
            parent_epic=story.get('epic_id'),
            implementation_type=impl_type,
            estimated_loc=estimated_loc,
            estimated_time=estimated_time,
            dependencies=[],
            acceptance_criteria=story.get('acceptance_criteria', [])
        )
    
    def _create_requirement_issue(self, requirement: Dict) -> DecomposedIssue:
        """Create a single issue for a simple requirement"""
        req_id = requirement.get('id', '')
        req_title = requirement.get('title', 'Unknown Requirement')
        
        # Estimate based on requirement complexity
        complexity_level = requirement.get('complexity', 'medium')
        complexity_multiplier = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}.get(complexity_level, 2)
        
        estimated_loc = min(complexity_multiplier * 50, self.TARGETS['max_loc_per_issue'])
        estimated_time = min(complexity_multiplier * 30, self.TARGETS['max_time_minutes'])
        
        impl_type = self._determine_implementation_type(requirement.get('description', ''))
        
        return self._create_issue(
            title=f"Requirement: {req_title}",
            description=self._format_requirement_description(requirement),
            requirement_id=req_id,
            parent_epic=None,
            implementation_type=impl_type,
            estimated_loc=estimated_loc,
            estimated_time=estimated_time,
            dependencies=[],
            acceptance_criteria=requirement.get('acceptance_criteria', [])
        )
    
    def _determine_implementation_type(self, content: str) -> ImplementationType:
        """Determine implementation type based on content analysis"""
        content_lower = content.lower()
        
        type_indicators = {
            ImplementationType.FRONTEND: ['ui', 'interface', 'form', 'button', 'screen', 'user interface'],
            ImplementationType.BACKEND: ['api', 'service', 'business logic', 'endpoint', 'server'],
            ImplementationType.DATABASE: ['database', 'table', 'schema', 'migration', 'data model'],
            ImplementationType.INTEGRATION: ['integration', 'third-party', 'external api', 'webhook'],
            ImplementationType.TESTING: ['test', 'testing', 'validation', 'verify'],
            ImplementationType.CONFIGURATION: ['config', 'environment', 'setup', 'deployment']
        }
        
        for impl_type, indicators in type_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return impl_type
        
        return ImplementationType.BACKEND  # Default
    
    def _calculate_context_tokens(self, description: str, acceptance_criteria: List[str], dependencies: List[str]) -> int:
        """Calculate estimated context tokens needed for this issue"""
        base_tokens = len(description) // 4  # Rough estimation: 1 token ≈ 4 characters
        criteria_tokens = sum(len(criterion) // 4 for criterion in acceptance_criteria)
        dependency_tokens = len(dependencies) * 100  # Context for each dependency
        framework_tokens = 800  # RIF framework context
        
        return int(base_tokens + criteria_tokens + dependency_tokens + framework_tokens)
    
    def _determine_issue_size(self, estimated_loc: int, estimated_time: int, context_tokens: int) -> IssueSize:
        """Determine issue size category based on estimates"""
        if estimated_loc > self.TARGETS['max_loc_per_issue'] or \
           estimated_time > self.TARGETS['max_time_minutes'] or \
           context_tokens > self.TARGETS['max_token_count']:
            return IssueSize.TOO_LARGE
        elif estimated_loc >= 200 or estimated_time >= 90:
            return IssueSize.LARGE
        elif estimated_loc >= 100 or estimated_time >= 60:
            return IssueSize.MEDIUM
        elif estimated_loc >= 50 or estimated_time >= 30:
            return IssueSize.SMALL
        else:
            return IssueSize.TINY
    
    def _generate_issue_labels(self, impl_type: ImplementationType, size: IssueSize, parent_epic: Optional[str]) -> List[str]:
        """Generate appropriate labels for the issue"""
        labels = [
            f"type:{impl_type.value}",
            f"size:{size.value}",
            "state:new"
        ]
        
        if parent_epic:
            labels.append(f"epic:{parent_epic.lower()}")
        
        # Add priority labels based on implementation type
        if impl_type in [ImplementationType.SETUP, ImplementationType.DATABASE]:
            labels.append("priority:high")
        elif impl_type == ImplementationType.TESTING:
            labels.append("priority:medium")
        
        return labels
    
    def _identify_risk_factors(self, description: str, estimated_loc: int, estimated_time: int, dependency_count: int) -> List[str]:
        """Identify potential risk factors for the issue"""
        risks = []
        content_lower = description.lower()
        
        # Size-based risks
        if estimated_loc > 200:
            risks.append("High complexity - consider breaking down further")
        if estimated_time > 90:
            risks.append("Long implementation time - may exceed single session")
        
        # Content-based risks
        risk_indicators = {
            "External API integration": ["external api", "third-party", "webhook"],
            "Database migration required": ["migration", "schema change", "alter table"],
            "Security considerations": ["authentication", "authorization", "security", "encryption"],
            "Performance optimization needed": ["performance", "optimization", "caching", "scalability"],
            "UI/UX design required": ["design", "mockup", "wireframe", "prototype"]
        }
        
        for risk_desc, indicators in risk_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                risks.append(risk_desc)
        
        # Dependency-based risks
        if dependency_count > 3:
            risks.append("High dependency count - coordination required")
        
        return risks
    
    def _generate_technical_notes(self, description: str, impl_type: ImplementationType) -> List[str]:
        """Generate technical implementation notes"""
        notes = []
        content_lower = description.lower()
        
        # Type-specific notes
        if impl_type == ImplementationType.FRONTEND:
            notes.append("Consider responsive design requirements")
            notes.append("Ensure accessibility compliance")
        elif impl_type == ImplementationType.BACKEND:
            notes.append("Implement proper error handling")
            notes.append("Add logging and monitoring")
        elif impl_type == ImplementationType.DATABASE:
            notes.append("Create database migration scripts")
            notes.append("Consider data validation constraints")
        elif impl_type == ImplementationType.INTEGRATION:
            notes.append("Implement retry logic for external calls")
            notes.append("Add proper timeout handling")
        
        # Content-specific notes
        if "api" in content_lower:
            notes.append("Document API endpoints and responses")
        if "security" in content_lower:
            notes.append("Security review required before deployment")
        if "performance" in content_lower:
            notes.append("Benchmark performance before and after")
        
        return notes
    
    def _determine_milestone(self, parent_epic: Optional[str], impl_type: ImplementationType) -> Optional[str]:
        """Determine appropriate milestone for the issue"""
        if parent_epic:
            return f"Epic: {parent_epic}"
        
        # Default milestones by implementation type
        milestone_mapping = {
            ImplementationType.SETUP: "Foundation",
            ImplementationType.DATABASE: "Data Layer",
            ImplementationType.BACKEND: "Core Features",
            ImplementationType.FRONTEND: "User Interface",
            ImplementationType.INTEGRATION: "System Integration",
            ImplementationType.TESTING: "Quality Assurance"
        }
        
        return milestone_mapping.get(impl_type, "Development")
    
    def _build_dependency_graph(self):
        """Build the dependency graph for all issues"""
        logger.info("Building dependency graph...")
        
        # Add all issues as nodes
        for issue in self.issues:
            self.dependency_graph.add_node(issue.id, issue=issue)
        
        # Add dependency edges
        for issue in self.issues:
            for dep_id in issue.dependencies:
                if dep_id and self.dependency_graph.has_node(dep_id):
                    self.dependency_graph.add_edge(dep_id, issue.id)
    
    def _optimize_sequence(self) -> List[str]:
        """Optimize issue sequence for efficient implementation"""
        logger.info("Optimizing issue sequence...")
        
        try:
            # Topological sort for dependency order
            sorted_order = list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXError:
            # Cycle detected - break cycles and retry
            logger.warning("Dependency cycle detected - breaking cycles")
            self._break_dependency_cycles()
            sorted_order = list(nx.topological_sort(self.dependency_graph))
        
        # Assign sequence orders
        for i, issue_id in enumerate(sorted_order):
            issue = next(iss for iss in self.issues if iss.id == issue_id)
            issue.sequence_order = i + 1
        
        # Group similar issues together within dependency constraints
        self._group_similar_issues()
        
        return sorted_order
    
    def _break_dependency_cycles(self):
        """Break dependency cycles in the graph"""
        cycles = list(nx.simple_cycles(self.dependency_graph))
        
        for cycle in cycles:
            # Remove the edge that creates the cycle (last -> first)
            if len(cycle) >= 2:
                self.dependency_graph.remove_edge(cycle[-1], cycle[0])
                logger.warning(f"Removed cyclic dependency: {cycle[-1]} -> {cycle[0]}")
                
                # Update the corresponding issue's dependencies
                for issue in self.issues:
                    if issue.id == cycle[0] and cycle[-1] in issue.dependencies:
                        issue.dependencies.remove(cycle[-1])
    
    def _group_similar_issues(self):
        """Group similar issues together to maintain context"""
        # Group by implementation type for context efficiency
        type_groups = {}
        for issue in self.issues:
            impl_type = issue.implementation_type.value
            if impl_type not in type_groups:
                type_groups[impl_type] = []
            type_groups[impl_type].append(issue)
        
        # Reorder within dependency constraints to group similar work
        # This is a complex optimization that would need sophisticated scheduling
        # For now, we rely on the topological sort order
        logger.info(f"Identified {len(type_groups)} implementation type groups")
    
    def _validate_decomposition(self):
        """Validate all issues meet single-session criteria"""
        logger.info("Validating decomposition against single-session criteria...")
        
        issues_needing_further_decomposition = []
        
        for issue in self.issues:
            violations = []
            
            # Check size constraints
            if issue.estimated_loc > self.TARGETS['max_loc_per_issue']:
                violations.append(f"LOC ({issue.estimated_loc}) exceeds limit ({self.TARGETS['max_loc_per_issue']})")
            
            if issue.estimated_time_minutes > self.TARGETS['max_time_minutes']:
                violations.append(f"Time ({issue.estimated_time_minutes}min) exceeds limit ({self.TARGETS['max_time_minutes']}min)")
            
            if issue.context_tokens > self.TARGETS['max_token_count']:
                violations.append(f"Context ({issue.context_tokens}) exceeds limit ({self.TARGETS['max_token_count']})")
            
            if violations:
                issue.size = IssueSize.TOO_LARGE
                issue.github_ready = False
                issues_needing_further_decomposition.append({
                    'issue_id': issue.id,
                    'title': issue.title,
                    'violations': violations
                })
                logger.warning(f"Issue {issue.id} needs further decomposition: {', '.join(violations)}")
        
        if issues_needing_further_decomposition:
            logger.warning(f"{len(issues_needing_further_decomposition)} issues need further decomposition")
        else:
            logger.info("All issues validated successfully for single-session completion")
    
    def _generate_statistics(self) -> Dict:
        """Generate comprehensive statistics about the decomposition"""
        if not self.issues:
            return {}
        
        # Basic counts
        total_issues = len(self.issues)
        total_estimated_time = sum(issue.estimated_time_minutes for issue in self.issues)
        total_estimated_loc = sum(issue.estimated_loc for issue in self.issues)
        
        # Size distribution
        size_distribution = {}
        for size in IssueSize:
            count = sum(1 for issue in self.issues if issue.size == size)
            size_distribution[size.value] = count
        
        # Implementation type distribution
        type_distribution = {}
        for impl_type in ImplementationType:
            count = sum(1 for issue in self.issues if issue.implementation_type == impl_type)
            if count > 0:
                type_distribution[impl_type.value] = count
        
        # Dependency analysis
        issues_with_deps = sum(1 for issue in self.issues if issue.dependencies)
        max_dependencies = max(len(issue.dependencies) for issue in self.issues) if self.issues else 0
        avg_dependencies = sum(len(issue.dependencies) for issue in self.issues) / total_issues
        
        # Context analysis
        avg_context_tokens = sum(issue.context_tokens for issue in self.issues) / total_issues
        max_context_tokens = max(issue.context_tokens for issue in self.issues)
        
        # Risk analysis
        issues_with_risks = sum(1 for issue in self.issues if issue.risk_factors)
        total_risks = sum(len(issue.risk_factors) for issue in self.issues)
        
        return {
            'total_issues': total_issues,
            'total_estimated_hours': round(total_estimated_time / 60, 2),
            'total_estimated_loc': total_estimated_loc,
            'average_issue_time_minutes': round(total_estimated_time / total_issues, 1),
            'average_issue_loc': round(total_estimated_loc / total_issues, 1),
            'size_distribution': size_distribution,
            'implementation_type_distribution': type_distribution,
            'dependency_stats': {
                'issues_with_dependencies': issues_with_deps,
                'max_dependencies_per_issue': max_dependencies,
                'average_dependencies_per_issue': round(avg_dependencies, 1),
                'dependency_depth': self._calculate_dependency_depth()
            },
            'context_stats': {
                'average_context_tokens': round(avg_context_tokens, 1),
                'max_context_tokens': max_context_tokens,
                'issues_over_token_limit': sum(1 for issue in self.issues 
                                             if issue.context_tokens > self.TARGETS['max_token_count'])
            },
            'risk_stats': {
                'issues_with_risks': issues_with_risks,
                'total_risk_factors': total_risks,
                'average_risks_per_issue': round(total_risks / total_issues, 1)
            },
            'quality_metrics': {
                'issues_ready_for_github': sum(1 for issue in self.issues if issue.github_ready),
                'issues_needing_decomposition': sum(1 for issue in self.issues if issue.size == IssueSize.TOO_LARGE),
                'completeness_score': round(sum(1 for issue in self.issues if issue.github_ready) / total_issues * 100, 1)
            }
        }
    
    def _calculate_dependency_depth(self) -> int:
        """Calculate the maximum dependency depth in the graph"""
        if not self.dependency_graph.nodes():
            return 0
        
        try:
            # Find the longest path in the DAG
            longest_path = nx.dag_longest_path_length(self.dependency_graph)
            return longest_path + 1  # Add 1 because path length is edges, depth is nodes
        except:
            return 0
    
    def _export_dependency_graph(self) -> Dict:
        """Export dependency graph data for visualization"""
        nodes = []
        edges = []
        
        for node_id in self.dependency_graph.nodes():
            issue = next(issue for issue in self.issues if issue.id == node_id)
            nodes.append({
                'id': node_id,
                'title': issue.title,
                'type': issue.implementation_type.value,
                'size': issue.size.value,
                'estimated_time': issue.estimated_time_minutes,
                'sequence_order': issue.sequence_order
            })
        
        for source, target in self.dependency_graph.edges():
            edges.append({
                'source': source,
                'target': target
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'is_dag': nx.is_directed_acyclic_graph(self.dependency_graph),
                'connected_components': nx.number_weakly_connected_components(self.dependency_graph)
            }
        }
    
    def _is_part_of_epic(self, story: Dict) -> bool:
        """Check if a story is already part of an epic"""
        story_id = story.get('id', '')
        for epic in self.requirements.get('epics', []):
            if story_id in epic.get('stories', []):
                return True
        return False
    
    def _is_already_decomposed(self, requirement: Dict) -> bool:
        """Check if a requirement has already been decomposed as part of epic/story"""
        req_id = requirement.get('id', '')
        
        # Check if it's part of any existing issues
        return any(issue.requirement_id == req_id for issue in self.issues)
    
    # Helper methods for generating descriptions and criteria
    def _generate_setup_description(self, epic: Dict) -> str:
        """Generate setup issue description"""
        return f"""## Setup: {epic.get('title', 'Project')} Foundation

This issue establishes the basic project structure and dependencies needed for implementing {epic.get('title', 'this epic')}.

### Scope
- Project scaffolding and directory structure  
- Dependency installation and configuration
- Basic configuration files
- Initial documentation setup

### Context
Epic: {epic.get('title', 'Unknown')}
Source: {epic.get('source_document', 'Unknown document')}

### Implementation Notes
{epic.get('description', '')[:500]}...
"""
    
    def _generate_implementation_description(self, epic: Dict) -> str:
        """Generate implementation issue description"""
        return f"""## Implementation: {epic.get('title', 'Core Features')}

Implement the core functionality for {epic.get('title', 'this epic')}.

### Requirements
{epic.get('description', '')}

### Scope
This issue covers the main implementation work as defined in the epic requirements.

### Context
Epic: {epic.get('title', 'Unknown')}
Source: {epic.get('source_document', 'Unknown document')}
"""
    
    def _generate_testing_description(self, epic: Dict) -> str:
        """Generate testing issue description"""
        return f"""## Testing: {epic.get('title', 'Validation')}

Comprehensive testing and validation for {epic.get('title', 'this epic')}.

### Scope
- Unit tests for core functionality
- Integration tests for system interactions  
- End-to-end validation scenarios
- Performance and security testing as needed

### Context
Epic: {epic.get('title', 'Unknown')}
Source: {epic.get('source_document', 'Unknown document')}
"""
    
    def _generate_integration_description(self, epic: Dict, components: List[Dict]) -> str:
        """Generate integration issue description"""
        component_names = [comp.get('name', 'Unknown') for comp in components]
        return f"""## Integration: {epic.get('title', 'System Integration')}

Integrate all components of {epic.get('title', 'this epic')} into a cohesive system.

### Components to Integrate
{chr(10).join(f"- {name}" for name in component_names)}

### Scope
- Component integration and coordination
- End-to-end workflow validation
- System-level configuration
- Integration testing

### Context
Epic: {epic.get('title', 'Unknown')}
Source: {epic.get('source_document', 'Unknown document')}
"""
    
    def _generate_setup_criteria(self, epic: Dict) -> List[str]:
        """Generate acceptance criteria for setup issues"""
        return [
            "Project structure is created with appropriate directories",
            "All dependencies are installed and configured",
            "Basic configuration files are in place",
            "Initial documentation is created",
            "Development environment is ready for implementation"
        ]
    
    def _generate_component_criteria(self, component: Dict) -> List[str]:
        """Generate acceptance criteria for component issues"""
        return [
            f"{component.get('name', 'Component')} is fully implemented",
            "All functionality works as specified",
            "Code follows project standards and conventions",
            "Basic tests are included",
            "Documentation is updated"
        ]
    
    def _generate_test_criteria(self, epic: Dict) -> List[str]:
        """Generate acceptance criteria for testing issues"""
        return [
            "All unit tests pass",
            "Integration tests validate system behavior",
            "Test coverage meets minimum requirements",
            "Performance benchmarks are established",
            "Security validation is complete"
        ]
    
    def _generate_integration_criteria(self, epic: Dict, components: List[Dict]) -> List[str]:
        """Generate acceptance criteria for integration issues"""
        return [
            "All components integrate successfully",
            "End-to-end workflows function correctly",
            "System configuration is complete",
            "Integration tests pass",
            "System is ready for deployment"
        ]
    
    def _extract_component_criteria(self, parent: Dict, component: Dict) -> List[str]:
        """Extract relevant acceptance criteria for a component"""
        parent_criteria = parent.get('acceptance_criteria', [])
        
        # Filter criteria relevant to this component
        component_name = component.get('name', '').lower()
        relevant_criteria = []
        
        for criterion in parent_criteria:
            if any(word in criterion.lower() for word in component_name.split()):
                relevant_criteria.append(criterion)
        
        # Add generic criteria if none found
        if not relevant_criteria:
            relevant_criteria = [
                f"Component {component.get('name', 'functionality')} works correctly",
                "Implementation meets specified requirements"
            ]
        
        return relevant_criteria
    
    def _format_story_description(self, story: Dict) -> str:
        """Format story description for GitHub issue"""
        actor = story.get('actor', 'user')
        action = story.get('action', '')
        benefit = story.get('benefit', '')
        
        description = f"""## User Story: {story.get('title', 'Unknown')}

**As a** {actor}  
**I want** {action}  
**So that** {benefit if benefit else 'I can achieve my goals'}

### Description
{story.get('description', '')}

### Context
Story ID: {story.get('id', 'Unknown')}
Source: {story.get('source_document', 'Unknown document')}
Points: {story.get('estimated_points', 'Not estimated')}
"""
        return description
    
    def _format_requirement_description(self, requirement: Dict) -> str:
        """Format requirement description for GitHub issue"""
        return f"""## Requirement: {requirement.get('title', 'Unknown')}

### Description
{requirement.get('description', '')}

### Type
{requirement.get('type', 'Unknown')}

### Priority
{requirement.get('priority', 'Medium')}

### Context
Requirement ID: {requirement.get('id', 'Unknown')}
Source: {requirement.get('source_document', 'Unknown document')}
Section: {requirement.get('source_section', 'Unknown section')}
"""
    
    def _enrich_issue_description(self, description: str, requirement_id: str, impl_type: ImplementationType) -> str:
        """Enrich issue description with RIF context and metadata"""
        enriched = f"""{description}

---

## RIF Implementation Context

**Issue Type:** {impl_type.value.title()}  
**Requirement ID:** {requirement_id}  
**Framework:** RIF (Reactive Intelligence Framework)  

### Implementation Guidelines
- Follow BMAD-METHOD principles for self-contained implementation
- Ensure all code changes are within single session limits
- Include appropriate error handling and logging
- Update documentation as needed
- Consider security and performance implications

### Success Criteria
This issue is complete when:
- All acceptance criteria are met
- Code passes quality gates
- Documentation is updated
- Tests are implemented and passing
"""
        return enriched
    
    def export_to_json(self, output_path: str) -> Path:
        """Export decomposition result to JSON file"""
        result = self.decompose_all_requirements()
        
        # Convert to serializable format
        serializable_result = {
            'issues': [asdict(issue) for issue in result.issues],
            'statistics': result.statistics,
            'dependency_graph': result.dependency_graph,
            'sequence_plan': result.sequence_plan,
            'strategy_used': result.strategy_used.value,
            'decomposition_metadata': result.decomposition_metadata
        }
        
        # Convert enums to strings in issues
        for issue_data in serializable_result['issues']:
            issue_data['size'] = issue_data['size'].value if hasattr(issue_data['size'], 'value') else issue_data['size']
            issue_data['implementation_type'] = issue_data['implementation_type'].value if hasattr(issue_data['implementation_type'], 'value') else issue_data['implementation_type']
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Decomposition result exported to: {output_path}")
        return output_path


def main():
    """Main function for CLI usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Decompose requirements into GitHub issues")
    parser.add_argument("requirements_json", help="Path to requirements JSON file from RequirementExtractor")
    parser.add_argument("-o", "--output", help="Output JSON file path", default="decomposed_issues.json")
    parser.add_argument("-s", "--strategy", choices=[s.value for s in DecompositionStrategy], 
                       default="mixed", help="Decomposition strategy to use")
    parser.add_argument("-c", "--context", help="Path to context windows JSON file (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--stats-only", action="store_true", help="Show only statistics")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load requirements data
        with open(args.requirements_json, 'r', encoding='utf-8') as f:
            requirements_data = json.load(f)
        
        # Load context data if provided
        context_data = None
        if args.context and Path(args.context).exists():
            with open(args.context, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
        
        # Initialize decomposer
        strategy = DecompositionStrategy(args.strategy)
        decomposer = IssueDecomposer(requirements_data, context_data)
        
        # Perform decomposition
        result = decomposer.decompose_all_requirements(strategy)
        
        if args.stats_only:
            print(json.dumps(result.statistics, indent=2))
        else:
            # Export full results
            output_path = decomposer.export_to_json(args.output)
            print(f"Decomposed issues exported to: {output_path}")
            print(f"\nSummary:")
            print(f"  Total Issues Created: {len(result.issues)}")
            print(f"  Strategy Used: {result.strategy_used.value}")
            print(f"  Total Estimated Hours: {result.statistics.get('total_estimated_hours', 0)}")
            print(f"  Issues Ready for GitHub: {result.statistics.get('quality_metrics', {}).get('issues_ready_for_github', 0)}")
            print(f"  Completeness Score: {result.statistics.get('quality_metrics', {}).get('completeness_score', 0)}%")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()