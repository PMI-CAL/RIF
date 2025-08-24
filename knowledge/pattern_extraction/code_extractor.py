"""
Code Pattern Extractor - AST-based code pattern analysis.

This module extracts code patterns from git commits, file changes, and
architectural decisions using Abstract Syntax Tree analysis and
structural pattern recognition.
"""

import ast
import json
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib

from .discovery_engine import ExtractedPattern, PatternSignature


@dataclass
class CodeStructure:
    """Represents the structure of a code component."""
    classes: List[str]
    functions: List[str]
    imports: List[str]
    decorators: List[str]
    patterns: List[str]
    complexity_metrics: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ArchitecturalElement:
    """Represents an architectural element extracted from code."""
    element_type: str  # 'class', 'function', 'module', 'interface'
    name: str
    relationships: List[str]
    responsibilities: List[str]
    patterns_used: List[str]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CodePatternExtractor:
    """
    Extracts code patterns using AST analysis and structural recognition.
    
    This extractor analyzes code changes, architectural decisions, and
    implementation patterns to identify reusable code structures and
    design patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known design patterns to detect
        self.design_patterns = {
            'singleton': self._detect_singleton,
            'factory': self._detect_factory,
            'observer': self._detect_observer,
            'strategy': self._detect_strategy,
            'adapter': self._detect_adapter,
            'decorator': self._detect_decorator,
            'command': self._detect_command,
            'state': self._detect_state,
            'builder': self._detect_builder,
            'repository': self._detect_repository
        }
        
        # Architectural patterns
        self.architectural_patterns = {
            'layered_architecture': self._detect_layered_architecture,
            'microservices': self._detect_microservices,
            'mvc': self._detect_mvc,
            'mvp': self._detect_mvp,
            'dependency_injection': self._detect_dependency_injection,
            'event_driven': self._detect_event_driven,
            'pipeline': self._detect_pipeline,
            'hexagonal': self._detect_hexagonal_architecture
        }
        
        # Code quality patterns
        self.quality_patterns = {
            'error_handling': self._detect_error_handling,
            'logging': self._detect_logging_pattern,
            'testing': self._detect_testing_pattern,
            'documentation': self._detect_documentation_pattern,
            'configuration': self._detect_configuration_pattern,
            'validation': self._detect_validation_pattern
        }
    
    def extract_patterns(self, completed_issue: Dict[str, Any]) -> List[ExtractedPattern]:
        """
        Extract code patterns from completed issue data.
        
        Args:
            completed_issue: Issue data containing code changes and decisions
            
        Returns:
            List of extracted code patterns
        """
        patterns = []
        issue_id = completed_issue.get('issue_number', 'unknown')
        
        self.logger.info(f"Extracting code patterns from issue #{issue_id}")
        
        try:
            # Extract from code changes
            if 'code_changes' in completed_issue:
                patterns.extend(self._extract_from_code_changes(completed_issue))
            
            # Extract from architectural decisions
            if 'decisions' in completed_issue:
                patterns.extend(self._extract_from_decisions(completed_issue))
            
            # Extract from implementation files
            if 'files_created' in completed_issue or 'files_modified' in completed_issue:
                patterns.extend(self._extract_from_implementation(completed_issue))
            
            self.logger.info(f"Extracted {len(patterns)} code patterns from issue #{issue_id}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error extracting code patterns from issue #{issue_id}: {e}")
            return []
    
    def _extract_from_code_changes(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from git diff and code changes."""
        patterns = []
        code_changes = issue_data.get('code_changes', {})
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            # Analyze each file change
            for file_path, change_data in code_changes.items():
                if not self._is_code_file(file_path):
                    continue
                
                # Extract patterns from added/modified code
                if 'added_lines' in change_data:
                    file_patterns = self._analyze_code_content(
                        change_data['added_lines'],
                        file_path,
                        f"issue-{issue_id}"
                    )
                    patterns.extend(file_patterns)
        
        except Exception as e:
            self.logger.error(f"Error extracting from code changes: {e}")
        
        return patterns
    
    def _extract_from_decisions(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from architectural decisions."""
        patterns = []
        decisions = issue_data.get('decisions', [])
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            for decision in decisions:
                # Look for architectural patterns mentioned in decisions
                decision_text = f"{decision.get('title', '')} {decision.get('decision', '')} {decision.get('context', '')}"
                
                architectural_patterns = self._detect_architectural_patterns_in_text(decision_text)
                
                for pattern_name, confidence in architectural_patterns:
                    pattern = ExtractedPattern(
                        title=f"{pattern_name.replace('_', ' ').title()} Pattern",
                        description=f"Decision pattern identified: {decision.get('title', 'Untitled')}",
                        pattern_type='decision',
                        source=f"issue-{issue_id}-decision",
                        content={
                            'pattern_name': pattern_name,
                            'decision_context': decision.get('context', ''),
                            'implementation_details': decision.get('decision', ''),
                            'consequences': decision.get('consequences', ''),
                            'architectural_impact': decision.get('impact', 'medium')
                        },
                        context={
                            'decision_title': decision.get('title', ''),
                            'issue_complexity': issue_data.get('complexity', 'medium'),
                            'domain': self._extract_domain(decision_text)
                        },
                        signature=PatternSignature.from_pattern({
                            'title': f"{pattern_name} Pattern",
                            'description': decision.get('decision', ''),
                            'architecture': {'pattern': pattern_name},
                            'complexity': issue_data.get('complexity', 'medium'),
                            'domain': self._extract_domain(decision_text)
                        }),
                        extraction_method='decision_analysis',
                        confidence=confidence,
                        created_at=datetime.now()
                    )
                    
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting from decisions: {e}")
        
        return patterns
    
    def _extract_from_implementation(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from implementation files."""
        patterns = []
        issue_id = issue_data.get('issue_number', 'unknown')
        
        try:
            # Collect all files to analyze
            files_to_analyze = []
            files_to_analyze.extend(issue_data.get('files_created', []))
            files_to_analyze.extend(issue_data.get('files_modified', []))
            
            for file_path in files_to_analyze:
                if not self._is_code_file(file_path):
                    continue
                
                try:
                    # Read file content if accessible
                    if Path(file_path).exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        file_patterns = self._analyze_code_content(
                            content,
                            file_path,
                            f"issue-{issue_id}"
                        )
                        patterns.extend(file_patterns)
                        
                except Exception as e:
                    self.logger.warning(f"Could not analyze file {file_path}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error extracting from implementation: {e}")
        
        return patterns
    
    def _analyze_code_content(self, code_content: str, file_path: str, source: str) -> List[ExtractedPattern]:
        """Analyze code content for patterns."""
        patterns = []
        
        try:
            # Try to parse as Python AST
            if file_path.endswith('.py'):
                patterns.extend(self._analyze_python_ast(code_content, file_path, source))
            
            # General pattern analysis (works for any language)
            patterns.extend(self._analyze_general_patterns(code_content, file_path, source))
            
        except Exception as e:
            self.logger.error(f"Error analyzing code content from {file_path}: {e}")
        
        return patterns
    
    def _analyze_python_ast(self, code_content: str, file_path: str, source: str) -> List[ExtractedPattern]:
        """Analyze Python code using AST."""
        patterns = []
        
        try:
            tree = ast.parse(code_content)
            
            # Extract code structure
            structure = self._extract_code_structure(tree)
            
            # Detect design patterns
            for pattern_name, detector in self.design_patterns.items():
                if detector(tree, structure):
                    pattern = self._create_design_pattern(
                        pattern_name, tree, structure, file_path, source
                    )
                    if pattern:
                        patterns.append(pattern)
            
            # Detect quality patterns
            for pattern_name, detector in self.quality_patterns.items():
                if detector(tree, structure):
                    pattern = self._create_quality_pattern(
                        pattern_name, tree, structure, file_path, source
                    )
                    if pattern:
                        patterns.append(pattern)
        
        except SyntaxError as e:
            self.logger.warning(f"Syntax error parsing {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error in Python AST analysis for {file_path}: {e}")
        
        return patterns
    
    def _analyze_general_patterns(self, code_content: str, file_path: str, source: str) -> List[ExtractedPattern]:
        """Analyze general code patterns that work across languages."""
        patterns = []
        
        try:
            # Interface/contract patterns
            if self._has_interface_pattern(code_content):
                patterns.append(self._create_interface_pattern(code_content, file_path, source))
            
            # Configuration patterns
            if self._has_configuration_pattern(code_content):
                patterns.append(self._create_configuration_pattern(code_content, file_path, source))
            
            # Error handling patterns
            if self._has_error_handling_pattern(code_content):
                patterns.append(self._create_error_handling_pattern(code_content, file_path, source))
            
            # Testing patterns
            if self._has_testing_pattern(code_content):
                patterns.append(self._create_testing_pattern(code_content, file_path, source))
        
        except Exception as e:
            self.logger.error(f"Error in general pattern analysis for {file_path}: {e}")
        
        return patterns
    
    def _extract_code_structure(self, tree: ast.AST) -> CodeStructure:
        """Extract structural information from AST."""
        classes = []
        functions = []
        imports = []
        decorators = []
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                decorators.extend([d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list])
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                decorators.extend([d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list])
            elif isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        complexity_metrics = {
            'class_count': len(classes),
            'function_count': len(functions),
            'import_count': len(imports),
            'decorator_count': len(decorators)
        }
        
        return CodeStructure(
            classes=classes,
            functions=functions,
            imports=imports,
            decorators=decorators,
            patterns=patterns,
            complexity_metrics=complexity_metrics
        )
    
    # Design pattern detectors
    def _detect_singleton(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect singleton pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if (isinstance(method, ast.FunctionDef) and 
                        method.name == '__new__' and
                        any('instance' in str(child) for child in ast.walk(method))):
                        return True
        return False
    
    def _detect_factory(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect factory pattern."""
        factory_indicators = ['create', 'build', 'make', 'factory']
        return any(
            any(indicator in func.lower() for indicator in factory_indicators)
            for func in structure.functions
        )
    
    def _detect_observer(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect observer pattern."""
        observer_indicators = ['notify', 'subscribe', 'unsubscribe', 'observer', 'listener']
        return any(
            any(indicator in func.lower() for indicator in observer_indicators)
            for func in structure.functions
        )
    
    def _detect_strategy(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect strategy pattern."""
        # Look for strategy-like structure
        has_strategy_interface = any('strategy' in cls.lower() for cls in structure.classes)
        has_context = any('context' in cls.lower() for cls in structure.classes)
        return has_strategy_interface or has_context
    
    def _detect_adapter(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect adapter pattern."""
        adapter_indicators = ['adapter', 'wrapper', 'bridge']
        return any(
            any(indicator in cls.lower() for indicator in adapter_indicators)
            for cls in structure.classes
        )
    
    def _detect_decorator(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect decorator pattern (beyond Python decorators)."""
        return len(structure.decorators) > 0 or any('decorator' in cls.lower() for cls in structure.classes)
    
    def _detect_command(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect command pattern."""
        command_indicators = ['execute', 'command', 'action', 'invoke']
        return any(
            any(indicator in func.lower() for indicator in command_indicators)
            for func in structure.functions
        )
    
    def _detect_state(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect state pattern."""
        state_indicators = ['state', 'transition', 'handle']
        return any(
            any(indicator in name.lower() for indicator in state_indicators)
            for name in structure.classes + structure.functions
        )
    
    def _detect_builder(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect builder pattern."""
        builder_indicators = ['builder', 'build', 'with_']
        return any(
            any(indicator in name.lower() for indicator in builder_indicators)
            for name in structure.classes + structure.functions
        )
    
    def _detect_repository(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect repository pattern."""
        repo_indicators = ['repository', 'repo', 'dao', 'store', 'persistence']
        
        # Check class names
        class_name_match = any(
            any(indicator in cls.lower() for indicator in repo_indicators)
            for cls in structure.classes
        )
        
        # Check for repository usage in code (dependency injection, attributes, etc.)
        code_usage_match = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check for self.repository or similar patterns
                if (isinstance(node.attr, str) and 
                    any(indicator in node.attr.lower() for indicator in repo_indicators)):
                    code_usage_match = True
                    break
            elif isinstance(node, ast.Name):
                # Check for repository variables
                if (isinstance(node.id, str) and 
                    any(indicator in node.id.lower() for indicator in repo_indicators)):
                    code_usage_match = True
                    break
        
        return class_name_match or code_usage_match
    
    # Architectural pattern detectors
    def _detect_layered_architecture(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect layered architecture pattern."""
        layer_indicators = ['controller', 'service', 'repository', 'model', 'view']
        return sum(
            1 for cls in structure.classes
            if any(indicator in cls.lower() for indicator in layer_indicators)
        ) >= 2
    
    def _detect_microservices(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect microservices patterns."""
        microservice_indicators = ['api', 'service', 'client', 'gateway', 'proxy']
        return any(
            any(indicator in name.lower() for indicator in microservice_indicators)
            for name in structure.classes + structure.functions
        )
    
    def _detect_mvc(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect MVC pattern."""
        mvc_components = ['model', 'view', 'controller']
        return sum(
            1 for component in mvc_components
            if any(component in cls.lower() for cls in structure.classes)
        ) >= 2
    
    def _detect_mvp(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect MVP pattern."""
        mvp_components = ['model', 'view', 'presenter']
        return sum(
            1 for component in mvp_components
            if any(component in cls.lower() for cls in structure.classes)
        ) >= 2
    
    def _detect_dependency_injection(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect dependency injection pattern."""
        di_indicators = ['inject', 'dependency', 'container', 'provider']
        return any(
            any(indicator in name.lower() for indicator in di_indicators)
            for name in structure.classes + structure.functions + structure.decorators
        )
    
    def _detect_event_driven(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect event-driven pattern."""
        event_indicators = ['event', 'emit', 'listen', 'trigger', 'handler']
        return any(
            any(indicator in name.lower() for indicator in event_indicators)
            for name in structure.classes + structure.functions
        )
    
    def _detect_pipeline(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect pipeline pattern."""
        pipeline_indicators = ['pipeline', 'process', 'stage', 'step', 'transform']
        return any(
            any(indicator in name.lower() for indicator in pipeline_indicators)
            for name in structure.classes + structure.functions
        )
    
    def _detect_hexagonal_architecture(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect hexagonal architecture pattern."""
        hex_indicators = ['port', 'adapter', 'domain', 'infrastructure', 'application']
        return sum(
            1 for indicator in hex_indicators
            if any(indicator in cls.lower() for cls in structure.classes)
        ) >= 3
    
    # Quality pattern detectors
    def _detect_error_handling(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect error handling patterns."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Try, ast.ExceptHandler, ast.Raise)):
                return True
        return False
    
    def _detect_logging_pattern(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect logging patterns."""
        logging_indicators = ['log', 'logger', 'debug', 'info', 'warn', 'error']
        return any(
            any(indicator in imp for indicator in logging_indicators)
            for imp in structure.imports
        ) or any(
            any(indicator in func.lower() for indicator in logging_indicators)
            for func in structure.functions
        )
    
    def _detect_testing_pattern(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect testing patterns."""
        test_indicators = ['test_', 'unittest', 'pytest', 'mock', 'assert']
        return any(
            any(indicator in name.lower() for indicator in test_indicators)
            for name in structure.functions + structure.imports
        )
    
    def _detect_documentation_pattern(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect documentation patterns."""
        doc_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    doc_count += 1
        return doc_count > 0
    
    def _detect_configuration_pattern(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect configuration patterns."""
        config_indicators = ['config', 'settings', 'env', 'yaml', 'json', 'ini']
        return any(
            any(indicator in name.lower() for indicator in config_indicators)
            for name in structure.classes + structure.functions + structure.imports
        )
    
    def _detect_validation_pattern(self, tree: ast.AST, structure: CodeStructure) -> bool:
        """Detect validation patterns."""
        validation_indicators = ['validate', 'check', 'verify', 'assert', 'ensure']
        return any(
            any(indicator in func.lower() for indicator in validation_indicators)
            for func in structure.functions
        )
    
    # General pattern detectors (language agnostic)
    def _has_interface_pattern(self, code: str) -> bool:
        """Detect interface/contract patterns."""
        interface_keywords = ['interface', 'abstract', 'protocol', 'trait', 'contract']
        return any(keyword in code.lower() for keyword in interface_keywords)
    
    def _has_configuration_pattern(self, code: str) -> bool:
        """Detect configuration patterns."""
        config_patterns = [
            r'config\s*=',
            r'settings\s*=',
            r'\.env',
            r'\.yaml',
            r'\.json',
            r'\.ini'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in config_patterns)
    
    def _has_error_handling_pattern(self, code: str) -> bool:
        """Detect error handling patterns."""
        error_patterns = [
            r'try\s*{',
            r'catch\s*\(',
            r'except\s*:',
            r'throw\s+',
            r'raise\s+',
            r'error\s*\('
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in error_patterns)
    
    def _has_testing_pattern(self, code: str) -> bool:
        """Detect testing patterns."""
        test_patterns = [
            r'test_\w+',
            r'@test',
            r'assert\s+',
            r'expect\s*\(',
            r'mock\s*\(',
            r'TestCase'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in test_patterns)
    
    # Pattern creation methods
    def _create_design_pattern(self, pattern_name: str, tree: ast.AST, 
                             structure: CodeStructure, file_path: str, source: str) -> Optional[ExtractedPattern]:
        """Create design pattern from detected pattern."""
        try:
            return ExtractedPattern(
                title=f"{pattern_name.replace('_', ' ').title()} Pattern",
                description=f"Design pattern detected in {Path(file_path).name}",
                pattern_type='design',
                source=source,
                content={
                    'pattern_name': pattern_name,
                    'file_path': file_path,
                    'classes': structure.classes,
                    'functions': structure.functions,
                    'complexity_metrics': structure.complexity_metrics,
                    'implementation_details': self._get_pattern_details(pattern_name, tree)
                },
                context={
                    'file_type': Path(file_path).suffix,
                    'code_structure': structure.to_dict(),
                    'architectural_layer': self._determine_architectural_layer(file_path)
                },
                signature=PatternSignature.from_pattern({
                    'title': f"{pattern_name} Pattern",
                    'description': f"Design pattern in {file_path}",
                    'architecture': {'pattern': pattern_name},
                    'complexity': self._calculate_complexity(structure),
                    'domain': self._extract_domain_from_path(file_path)
                }),
                extraction_method='ast_analysis',
                confidence=0.8,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error creating design pattern {pattern_name}: {e}")
            return None
    
    def _create_quality_pattern(self, pattern_name: str, tree: ast.AST,
                              structure: CodeStructure, file_path: str, source: str) -> Optional[ExtractedPattern]:
        """Create quality pattern from detected pattern."""
        try:
            return ExtractedPattern(
                title=f"{pattern_name.replace('_', ' ').title()} Pattern",
                description=f"Quality pattern detected in {Path(file_path).name}",
                pattern_type='quality',
                source=source,
                content={
                    'pattern_name': pattern_name,
                    'file_path': file_path,
                    'quality_metrics': self._calculate_quality_metrics(tree, structure),
                    'implementation_examples': self._get_pattern_examples(pattern_name, tree)
                },
                context={
                    'file_type': Path(file_path).suffix,
                    'code_structure': structure.to_dict(),
                    'quality_score': self._calculate_quality_score(tree, structure)
                },
                signature=PatternSignature.from_pattern({
                    'title': f"{pattern_name} Quality Pattern",
                    'description': f"Quality pattern in {file_path}",
                    'complexity': self._calculate_complexity(structure),
                    'domain': 'quality_assurance'
                }),
                extraction_method='quality_analysis',
                confidence=0.7,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error creating quality pattern {pattern_name}: {e}")
            return None
    
    def _create_interface_pattern(self, code: str, file_path: str, source: str) -> ExtractedPattern:
        """Create interface pattern."""
        return ExtractedPattern(
            title="Interface/Contract Pattern",
            description=f"Interface pattern detected in {Path(file_path).name}",
            pattern_type='architectural',
            source=source,
            content={
                'pattern_name': 'interface',
                'file_path': file_path,
                'interface_definition': self._extract_interface_definition(code),
                'contract_methods': self._extract_contract_methods(code)
            },
            context={
                'file_type': Path(file_path).suffix,
                'architectural_significance': 'high'
            },
            signature=PatternSignature.from_pattern({
                'title': 'Interface Pattern',
                'description': f"Interface pattern in {file_path}",
                'architecture': {'pattern': 'interface'},
                'complexity': 'medium',
                'domain': 'contracts'
            }),
            extraction_method='text_analysis',
            confidence=0.6,
            created_at=datetime.now()
        )
    
    def _create_configuration_pattern(self, code: str, file_path: str, source: str) -> ExtractedPattern:
        """Create configuration pattern."""
        return ExtractedPattern(
            title="Configuration Pattern",
            description=f"Configuration pattern detected in {Path(file_path).name}",
            pattern_type='infrastructure',
            source=source,
            content={
                'pattern_name': 'configuration',
                'file_path': file_path,
                'config_types': self._identify_config_types(code),
                'configuration_structure': self._extract_config_structure(code)
            },
            context={
                'file_type': Path(file_path).suffix,
                'configuration_scope': self._determine_config_scope(file_path)
            },
            signature=PatternSignature.from_pattern({
                'title': 'Configuration Pattern',
                'description': f"Configuration pattern in {file_path}",
                'complexity': 'low',
                'domain': 'configuration'
            }),
            extraction_method='text_analysis',
            confidence=0.7,
            created_at=datetime.now()
        )
    
    def _create_error_handling_pattern(self, code: str, file_path: str, source: str) -> ExtractedPattern:
        """Create error handling pattern."""
        return ExtractedPattern(
            title="Error Handling Pattern",
            description=f"Error handling pattern detected in {Path(file_path).name}",
            pattern_type='quality',
            source=source,
            content={
                'pattern_name': 'error_handling',
                'file_path': file_path,
                'error_handling_strategies': self._identify_error_strategies(code),
                'exception_types': self._extract_exception_types(code)
            },
            context={
                'file_type': Path(file_path).suffix,
                'error_handling_completeness': self._assess_error_handling(code)
            },
            signature=PatternSignature.from_pattern({
                'title': 'Error Handling Pattern',
                'description': f"Error handling in {file_path}",
                'complexity': 'medium',
                'domain': 'error_handling'
            }),
            extraction_method='text_analysis',
            confidence=0.6,
            created_at=datetime.now()
        )
    
    def _create_testing_pattern(self, code: str, file_path: str, source: str) -> ExtractedPattern:
        """Create testing pattern."""
        return ExtractedPattern(
            title="Testing Pattern",
            description=f"Testing pattern detected in {Path(file_path).name}",
            pattern_type='quality',
            source=source,
            content={
                'pattern_name': 'testing',
                'file_path': file_path,
                'testing_frameworks': self._identify_testing_frameworks(code),
                'test_types': self._identify_test_types(code),
                'assertion_patterns': self._extract_assertion_patterns(code)
            },
            context={
                'file_type': Path(file_path).suffix,
                'test_coverage_indicators': self._assess_test_coverage(code)
            },
            signature=PatternSignature.from_pattern({
                'title': 'Testing Pattern',
                'description': f"Testing pattern in {file_path}",
                'complexity': 'medium',
                'domain': 'testing'
            }),
            extraction_method='text_analysis',
            confidence=0.7,
            created_at=datetime.now()
        )
    
    # Helper methods
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt'}
        return Path(file_path).suffix.lower() in code_extensions
    
    def _extract_domain(self, text: str) -> str:
        """Extract domain from text."""
        domains = ['web', 'data', 'api', 'ui', 'backend', 'frontend', 'database', 'ml', 'ai', 'security']
        for domain in domains:
            if domain in text.lower():
                return domain
        return 'general'
    
    def _extract_domain_from_path(self, file_path: str) -> str:
        """Extract domain from file path."""
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part in ['api', 'web', 'data', 'ui', 'backend', 'frontend', 'db', 'security', 'ml', 'ai']:
                return part
        return 'general'
    
    def _calculate_complexity(self, structure: CodeStructure) -> str:
        """Calculate complexity from code structure."""
        total_complexity = (
            structure.complexity_metrics.get('class_count', 0) * 3 +
            structure.complexity_metrics.get('function_count', 0) * 2 +
            structure.complexity_metrics.get('import_count', 0)
        )
        
        if total_complexity < 10:
            return 'low'
        elif total_complexity < 25:
            return 'medium'
        elif total_complexity < 50:
            return 'high'
        else:
            return 'very-high'
    
    def _determine_architectural_layer(self, file_path: str) -> str:
        """Determine architectural layer from file path."""
        path_lower = file_path.lower()
        if 'controller' in path_lower or 'api' in path_lower:
            return 'presentation'
        elif 'service' in path_lower or 'business' in path_lower:
            return 'business'
        elif 'repository' in path_lower or 'dao' in path_lower or 'data' in path_lower:
            return 'data'
        elif 'model' in path_lower or 'entity' in path_lower:
            return 'domain'
        else:
            return 'infrastructure'
    
    def _get_pattern_details(self, pattern_name: str, tree: ast.AST) -> Dict[str, Any]:
        """Get detailed information about a specific pattern."""
        details = {}
        
        # Extract key methods/classes related to the pattern
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if pattern_name.lower() in node.name.lower():
                    details['key_class'] = node.name
                    details['methods'] = [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
            elif isinstance(node, ast.FunctionDef):
                if pattern_name.lower() in node.name.lower():
                    details['key_functions'] = details.get('key_functions', []) + [node.name]
        
        return details
    
    def _calculate_quality_metrics(self, tree: ast.AST, structure: CodeStructure) -> Dict[str, Any]:
        """Calculate quality metrics."""
        return {
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
            'documentation_coverage': self._calculate_documentation_coverage(tree),
            'error_handling_coverage': self._calculate_error_handling_coverage(tree),
            'test_indicator_count': len([f for f in structure.functions if f.startswith('test_')])
        }
    
    def _calculate_quality_score(self, tree: ast.AST, structure: CodeStructure) -> float:
        """Calculate overall quality score."""
        metrics = self._calculate_quality_metrics(tree, structure)
        
        # Simple scoring algorithm
        score = 0.5  # base score
        
        if metrics['documentation_coverage'] > 0.5:
            score += 0.2
        
        if metrics['error_handling_coverage'] > 0.3:
            score += 0.2
        
        if metrics['test_indicator_count'] > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_pattern_examples(self, pattern_name: str, tree: ast.AST) -> List[str]:
        """Get code examples of the pattern."""
        examples = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if pattern_name.lower() in node.name.lower():
                    # Get the first few lines as an example
                    try:
                        code_lines = ast.get_source_segment(ast.unparse(tree), node)
                        if code_lines:
                            examples.append(code_lines[:200] + "..." if len(code_lines) > 200 else code_lines)
                    except:
                        examples.append(f"{type(node).__name__}: {node.name}")
        
        return examples
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_documentation_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage ratio."""
        documented = 0
        total = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node):
                    documented += 1
        
        return documented / max(total, 1)
    
    def _calculate_error_handling_coverage(self, tree: ast.AST) -> float:
        """Calculate error handling coverage ratio."""
        functions_with_error_handling = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                # Check if function has try/except
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        functions_with_error_handling += 1
                        break
        
        return functions_with_error_handling / max(total_functions, 1)
    
    def _detect_architectural_patterns_in_text(self, text: str) -> List[Tuple[str, float]]:
        """Detect architectural and design patterns mentioned in text."""
        patterns = []
        
        # Check architectural patterns
        for pattern_name in self.architectural_patterns:
            pattern_words = pattern_name.replace('_', ' ').split()
            if any(word in text.lower() for word in pattern_words):
                # Calculate confidence based on context
                confidence = 0.6
                if pattern_name in text.lower():
                    confidence = 0.8
                patterns.append((pattern_name, confidence))
        
        # Also check design patterns (like repository pattern)
        for pattern_name in self.design_patterns:
            pattern_words = pattern_name.replace('_', ' ').split()
            if any(word in text.lower() for word in pattern_words):
                # Calculate confidence based on context
                confidence = 0.6
                if pattern_name in text.lower():
                    confidence = 0.8
                patterns.append((pattern_name, confidence))
        
        return patterns
    
    # Text analysis helper methods
    def _extract_interface_definition(self, code: str) -> Dict[str, Any]:
        """Extract interface definition from code."""
        return {'interface_keywords_found': ['interface', 'abstract', 'protocol']}
    
    def _extract_contract_methods(self, code: str) -> List[str]:
        """Extract contract methods from code."""
        return re.findall(r'def\s+(\w+)', code)
    
    def _identify_config_types(self, code: str) -> List[str]:
        """Identify configuration types in code."""
        config_types = []
        if '.yaml' in code or '.yml' in code:
            config_types.append('yaml')
        if '.json' in code:
            config_types.append('json')
        if '.env' in code:
            config_types.append('environment')
        if '.ini' in code:
            config_types.append('ini')
        return config_types
    
    def _extract_config_structure(self, code: str) -> Dict[str, Any]:
        """Extract configuration structure from code."""
        return {'config_patterns_detected': True}
    
    def _determine_config_scope(self, file_path: str) -> str:
        """Determine configuration scope."""
        if 'global' in file_path or 'system' in file_path:
            return 'global'
        elif 'app' in file_path or 'application' in file_path:
            return 'application'
        else:
            return 'module'
    
    def _identify_error_strategies(self, code: str) -> List[str]:
        """Identify error handling strategies."""
        strategies = []
        if 'try' in code and 'except' in code:
            strategies.append('exception_handling')
        if 'raise' in code:
            strategies.append('exception_raising')
        if 'assert' in code:
            strategies.append('assertion')
        return strategies
    
    def _extract_exception_types(self, code: str) -> List[str]:
        """Extract exception types from code."""
        return re.findall(r'except\s+(\w+)', code)
    
    def _assess_error_handling(self, code: str) -> str:
        """Assess error handling completeness."""
        if 'try' in code and 'except' in code and 'finally' in code:
            return 'comprehensive'
        elif 'try' in code and 'except' in code:
            return 'basic'
        else:
            return 'minimal'
    
    def _identify_testing_frameworks(self, code: str) -> List[str]:
        """Identify testing frameworks used."""
        frameworks = []
        if 'unittest' in code:
            frameworks.append('unittest')
        if 'pytest' in code:
            frameworks.append('pytest')
        if 'mock' in code:
            frameworks.append('mock')
        return frameworks
    
    def _identify_test_types(self, code: str) -> List[str]:
        """Identify types of tests."""
        test_types = []
        if 'test_unit' in code or 'TestCase' in code:
            test_types.append('unit')
        if 'test_integration' in code:
            test_types.append('integration')
        if 'test_end_to_end' in code or 'test_e2e' in code:
            test_types.append('e2e')
        return test_types
    
    def _extract_assertion_patterns(self, code: str) -> List[str]:
        """Extract assertion patterns from test code."""
        return re.findall(r'assert\w*\(', code)
    
    def _assess_test_coverage(self, code: str) -> Dict[str, Any]:
        """Assess test coverage indicators."""
        return {
            'has_assertions': 'assert' in code,
            'has_mocks': 'mock' in code,
            'has_fixtures': 'fixture' in code
        }