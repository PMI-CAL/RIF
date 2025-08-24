#!/usr/bin/env python3
"""
Architectural Risk Assessor - Issue #93 Phase 3
Advanced architectural risk assessment for the Multi-Dimensional Quality Scoring System.

This module provides:
- System architecture pattern analysis
- Dependency risk assessment  
- Design pattern compliance evaluation
- Architectural debt detection
- Integration risk analysis
"""

import os
import json
import yaml
import subprocess
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import re
import ast
from collections import defaultdict, Counter

@dataclass
class ArchitecturalMetrics:
    """Container for architectural measurement data."""
    coupling_score: float = 0.0
    cohesion_score: float = 1.0
    dependency_count: int = 0
    circular_dependencies: int = 0
    architecture_violations: int = 0
    design_pattern_adherence: float = 1.0

@dataclass
class DependencyRisk:
    """Risk assessment for dependencies."""
    dependency_name: str
    risk_level: str = "low"  # low, medium, high, critical
    risk_factors: List[str] = field(default_factory=list)
    version_risk: float = 0.0
    security_risk: float = 0.0
    maintenance_risk: float = 0.0

@dataclass
class ArchitecturalRiskProfile:
    """Comprehensive architectural risk assessment."""
    overall_risk_score: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    architecture_patterns: Dict[str, float] = field(default_factory=dict)
    dependency_risks: List[DependencyRisk] = field(default_factory=list)
    design_violations: List[str] = field(default_factory=list)
    architectural_debt: Dict[str, float] = field(default_factory=dict)
    integration_risks: List[str] = field(default_factory=list)
    mitigation_recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class FileArchitecturalAnalysis:
    """Architectural analysis for individual file."""
    file_path: str
    module_coupling: float = 0.0
    responsibility_violations: List[str] = field(default_factory=list)
    design_patterns: List[str] = field(default_factory=list)
    architectural_smells: List[str] = field(default_factory=list)
    dependency_analysis: Dict[str, Any] = field(default_factory=dict)
    analysis_time_ms: float = 0.0

class ArchitecturalRiskAssessor:
    """
    Advanced architectural risk assessor for multi-dimensional quality scoring.
    
    Analyzes code changes for architectural impact including:
    - Design pattern compliance
    - Dependency management risks
    - System coupling and cohesion
    - Architectural debt accumulation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize architectural risk assessor."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/architectural-risk-patterns.yaml"
        self.cache = {}
        self.cache_ttl = timedelta(hours=2)
        self.logger = logging.getLogger(__name__)
        
        # Load architectural patterns and rules
        self._load_architectural_config()
        
        # Architectural risk patterns (simplified for performance)
        self.architectural_smells = {
            'god_class': r'class\s+\w+.*:',  # Simplified - count methods separately
            'long_method': r'def\s+\w+\([^)]*\):',  # Simplified - measure by line count
            'feature_envy': r'self\.\w+\.\w+',
            'data_clumps': r'def\s+\w+\([^)]*(?:\w+,\s*){4,}[^)]*\)',
            'dead_code': r'#\s*TODO|#\s*FIXME|\bpass\b',
            'magic_numbers': r'\b\d{2,}\b',
            'long_parameter_list': r'def\s+\w+\([^)]{50,}\)',
            'inappropriate_intimacy': r'import.*private|from.*private',
        }
        
        # Design pattern indicators
        self.design_patterns = {
            'singleton': r'class.*Singleton|_instance\s*=\s*None',
            'factory': r'class.*Factory|def\s+create_\w+',
            'observer': r'class.*Observer|def\s+notify|def\s+update',
            'strategy': r'class.*Strategy|def\s+execute|def\s+algorithm',
            'decorator': r'@\w+|def\s+decorator|__call__.*wrapper',
            'adapter': r'class.*Adapter|def\s+adapt',
            'facade': r'class.*Facade|def\s+simplified_\w+',
            'template_method': r'def\s+template_\w+|abstract.*method',
            'command': r'class.*Command|def\s+execute|def\s+undo',
            'composite': r'class.*Composite|children.*list|add.*child',
        }
        
        # Dependency risk indicators
        self.dependency_patterns = {
            'circular_import': r'import.*circular|from.*circular',  # Simplified pattern
            'excessive_imports': r'^(import|from).*$',
            'wildcard_import': r'from\s+\w+\s+import\s+\*',
            'relative_import': r'from\s+\.+\w*\s+import',
            'deep_import_chain': r'from\s+\w+\.\w+\.\w+\.\w+',
            'external_dependency': r'import\s+(requests|numpy|pandas|django)',
        }
        
        # Architecture violation patterns
        self.architecture_violations = {
            'layering_violation': r'from\s+.*\.ui\s+import.*models|from.*models.*import.*ui',
            'separation_of_concerns': r'class.*Controller.*Model|class.*Model.*View',
            'tight_coupling': r'import\s+.*\.implementation',
            'interface_segregation': r'class.*Interface.*def.*unused',
            'dependency_inversion': r'import.*concrete.*implementation',
        }
    
    def _load_architectural_config(self):
        """Load architectural risk configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.architectural_weights = config.get('architectural_weights', {})
                    self.risk_thresholds = config.get('risk_thresholds', {})
                    self.pattern_weights = config.get('pattern_weights', {})
            else:
                self._set_default_architectural_config()
        except Exception as e:
            self.logger.warning(f"Failed to load architectural configuration: {e}")
            self._set_default_architectural_config()
    
    def _set_default_architectural_config(self):
        """Set default architectural configuration."""
        self.architectural_weights = {
            'design_quality': 0.30,
            'dependency_management': 0.25,
            'coupling_cohesion': 0.20,
            'architectural_debt': 0.15,
            'pattern_compliance': 0.10
        }
        self.risk_thresholds = {
            'low': 0.25,
            'medium': 0.50,
            'high': 0.75,
            'critical': 0.90
        }
        self.pattern_weights = {
            'god_class': 0.9,
            'long_method': 0.6,
            'duplicate_code': 0.8,
            'circular_import': 1.0,
            'layering_violation': 0.9
        }
    
    def assess_architectural_risk(self,
                                files: List[str],
                                project_context: Optional[Dict] = None,
                                architecture_rules: Optional[Dict] = None) -> ArchitecturalRiskProfile:
        """
        Assess comprehensive architectural risk.
        
        Args:
            files: List of files to analyze
            project_context: Optional project context information
            architecture_rules: Optional custom architecture rules
            
        Returns:
            ArchitecturalRiskProfile with detailed assessment
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"arch_risk:{hash(tuple(sorted(files)))}"
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_result
        
        try:
            # Analyze each file for architectural risks
            file_analyses = []
            for file_path in files:
                if os.path.exists(file_path) and file_path.endswith('.py'):
                    analysis = self._analyze_file_architectural_risk(file_path)
                    file_analyses.append(analysis)
            
            # Analyze project-level architecture
            dependency_graph = self._build_dependency_graph(files)
            
            # Calculate overall architectural risk profile
            profile = self._calculate_architectural_risk_profile(
                file_analyses, dependency_graph, project_context, architecture_rules
            )
            
            # Cache result
            self.cache[cache_key] = (profile, datetime.now())
            
            # Clean old cache entries
            self._cleanup_cache()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Architectural risk assessment failed: {e}")
            return ArchitecturalRiskProfile(
                overall_risk_score=0.4,
                risk_level="medium",
                confidence=0.6
            )
    
    def _analyze_file_architectural_risk(self, file_path: str) -> FileArchitecturalAnalysis:
        """Analyze architectural risk for individual file."""
        start_time = time.time()
        
        analysis = FileArchitecturalAnalysis(file_path=file_path)
        
        try:
            # Read and parse file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Try to parse as Python AST for deeper analysis
            try:
                tree = ast.parse(content)
                analysis.dependency_analysis = self._analyze_ast_dependencies(tree)
                analysis.module_coupling = self._calculate_module_coupling(tree)
            except SyntaxError:
                # Fallback to regex-based analysis
                pass
            
            # Analyze architectural smells
            for smell_name, pattern in self.architectural_smells.items():
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                if matches:
                    analysis.architectural_smells.append(
                        f"{smell_name}: {len(matches)} occurrences"
                    )
            
            # Detect design patterns
            for pattern_name, pattern in self.design_patterns.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    analysis.design_patterns.append(pattern_name)
            
            # Analyze dependency patterns
            dependency_risks = {}
            for dep_name, pattern in self.dependency_patterns.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    dependency_risks[dep_name] = len(matches)
            
            analysis.dependency_analysis['risk_patterns'] = dependency_risks
            
            # Check for architecture violations
            for violation_name, pattern in self.architecture_violations.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    analysis.responsibility_violations.append(
                        f"{violation_name}: {len(matches)} violations"
                    )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze architectural risk for {file_path}: {e}")
        
        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        return analysis
    
    def _analyze_ast_dependencies(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze dependencies using AST parsing."""
        dependencies = {
            'imports': [],
            'from_imports': [],
            'function_calls': [],
            'class_dependencies': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies['imports'].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                dependencies['from_imports'].append({
                    'module': module,
                    'names': names
                })
            elif isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                dependencies['function_calls'].append(node.func.id)
            elif isinstance(node, ast.ClassDef):
                bases = [base.id for base in node.bases if hasattr(base, 'id')]
                dependencies['class_dependencies'].append({
                    'class': node.name,
                    'bases': bases
                })
        
        return dependencies
    
    def _calculate_module_coupling(self, tree: ast.AST) -> float:
        """Calculate module coupling score from AST."""
        import_count = 0
        external_call_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
            elif isinstance(node, ast.Attribute):
                external_call_count += 1
        
        # Higher import count and external calls = higher coupling
        coupling_score = min((import_count * 0.1) + (external_call_count * 0.01), 1.0)
        return coupling_score
    
    def _build_dependency_graph(self, files: List[str]) -> Dict[str, List[str]]:
        """Build dependency graph for the project."""
        dependency_graph = defaultdict(list)
        
        for file_path in files:
            if not file_path.endswith('.py'):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract imports
                import_matches = re.findall(r'^(?:from\s+([\w.]+)\s+)?import\s+([\w., *]+)', 
                                          content, re.MULTILINE)
                
                for match in import_matches:
                    module, imports = match
                    if module:
                        dependency_graph[file_path].append(module)
                    else:
                        # Direct imports
                        for imp in imports.split(','):
                            imp = imp.strip()
                            if imp and imp != '*':
                                dependency_graph[file_path].append(imp)
                                
            except Exception as e:
                self.logger.warning(f"Failed to analyze dependencies for {file_path}: {e}")
        
        return dict(dependency_graph)
    
    def _calculate_architectural_risk_profile(self,
                                            file_analyses: List[FileArchitecturalAnalysis],
                                            dependency_graph: Dict[str, List[str]],
                                            project_context: Optional[Dict],
                                            architecture_rules: Optional[Dict]) -> ArchitecturalRiskProfile:
        """Calculate overall architectural risk profile."""
        profile = ArchitecturalRiskProfile()
        
        if not file_analyses:
            return profile
        
        # Aggregate architectural smells
        all_smells = []
        all_violations = []
        all_patterns = []
        
        for analysis in file_analyses:
            all_smells.extend(analysis.architectural_smells)
            all_violations.extend(analysis.responsibility_violations)
            all_patterns.extend(analysis.design_patterns)
        
        # Calculate risk components
        risk_components = {}
        
        # Design quality risk
        smell_count = len(all_smells)
        design_risk = min(smell_count * 0.1, 1.0)
        risk_components['design_quality'] = design_risk
        
        # Dependency management risk  
        dependency_risks = self._assess_dependency_risks(file_analyses, dependency_graph)
        dependency_risk = min(len(dependency_risks) * 0.2, 1.0)
        risk_components['dependency_management'] = dependency_risk
        profile.dependency_risks = dependency_risks
        
        # Coupling and cohesion risk
        avg_coupling = sum(analysis.module_coupling for analysis in file_analyses) / len(file_analyses)
        coupling_risk = min(avg_coupling * 1.2, 1.0)
        risk_components['coupling_cohesion'] = coupling_risk
        
        # Architectural debt risk
        violation_count = len(all_violations)
        debt_risk = min(violation_count * 0.15, 1.0)
        risk_components['architectural_debt'] = debt_risk
        
        # Pattern compliance risk
        unique_patterns = set(all_patterns)
        expected_patterns = max(len(file_analyses) // 5, 1)  # Expected patterns based on size
        pattern_risk = max(0, 1.0 - (len(unique_patterns) / expected_patterns))
        risk_components['pattern_compliance'] = min(pattern_risk, 1.0)
        
        # Calculate overall risk score
        profile.overall_risk_score = min(
            sum(risk_components[comp] * self.architectural_weights.get(comp, 0.2)
                for comp in risk_components),
            1.0
        )
        
        profile.architecture_patterns = risk_components
        profile.design_violations = list(set(all_violations))
        
        # Calculate architectural debt details
        profile.architectural_debt = {
            'code_smells': len(all_smells),
            'architecture_violations': len(all_violations),
            'dependency_issues': len(dependency_risks),
            'coupling_score': avg_coupling
        }
        
        # Determine risk level
        if profile.overall_risk_score >= self.risk_thresholds['critical']:
            profile.risk_level = "critical"
        elif profile.overall_risk_score >= self.risk_thresholds['high']:
            profile.risk_level = "high"
        elif profile.overall_risk_score >= self.risk_thresholds['medium']:
            profile.risk_level = "medium"
        else:
            profile.risk_level = "low"
        
        # Assess integration risks
        profile.integration_risks = self._assess_integration_risks(
            dependency_graph, project_context
        )
        
        # Generate mitigation recommendations
        profile.mitigation_recommendations = self._generate_architectural_mitigations(
            risk_components, all_smells, all_violations, dependency_risks
        )
        
        # Set confidence
        profile.confidence = min(len(file_analyses) / 10.0, 1.0)
        
        return profile
    
    def _assess_dependency_risks(self,
                               file_analyses: List[FileArchitecturalAnalysis],
                               dependency_graph: Dict[str, List[str]]) -> List[DependencyRisk]:
        """Assess risks in project dependencies."""
        dependency_risks = []
        dependency_counts = Counter()
        
        # Count dependency usage
        for deps in dependency_graph.values():
            for dep in deps:
                dependency_counts[dep] += 1
        
        # Analyze high-usage dependencies
        for dep, count in dependency_counts.most_common(10):
            risk = DependencyRisk(dependency_name=dep)
            
            # Assess risk factors
            if count > 5:
                risk.risk_factors.append("High coupling - used in many modules")
                risk.risk_level = "medium"
            
            if any(keyword in dep.lower() for keyword in ['test', 'debug', 'temp']):
                risk.risk_factors.append("Development dependency in production code")
                risk.risk_level = "high"
            
            if '.' in dep and dep.count('.') > 3:
                risk.risk_factors.append("Deep dependency chain")
                risk.risk_level = "medium"
            
            # Check for circular dependencies (simplified)
            if dep in dependency_graph:
                for sub_dep in dependency_graph[dep]:
                    if sub_dep in dependency_counts and dependency_counts[sub_dep] > 2:
                        risk.risk_factors.append("Potential circular dependency")
                        risk.risk_level = "high"
                        break
            
            if risk.risk_factors:
                dependency_risks.append(risk)
        
        return dependency_risks
    
    def _assess_integration_risks(self,
                                dependency_graph: Dict[str, List[str]],
                                project_context: Optional[Dict]) -> List[str]:
        """Assess integration risks in the architecture."""
        risks = []
        
        # Check for integration patterns
        external_deps = set()
        for deps in dependency_graph.values():
            for dep in deps:
                if any(keyword in dep.lower() for keyword in 
                      ['requests', 'http', 'api', 'client', 'service']):
                    external_deps.add(dep)
        
        if len(external_deps) > 3:
            risks.append("High number of external service integrations")
        
        # Check for database dependencies
        db_deps = [dep for deps in dependency_graph.values() for dep in deps
                  if any(keyword in dep.lower() for keyword in 
                        ['sql', 'database', 'db', 'orm', 'model'])]
        
        if len(set(db_deps)) > 2:
            risks.append("Multiple database integration patterns")
        
        # Project context risks
        if project_context and project_context.get('microservices', False):
            if len(external_deps) < 2:
                risks.append("Microservice with insufficient external integrations")
        
        return risks
    
    def _generate_architectural_mitigations(self,
                                          risk_components: Dict[str, float],
                                          smells: List[str],
                                          violations: List[str],
                                          dependency_risks: List[DependencyRisk]) -> List[str]:
        """Generate architectural mitigation recommendations."""
        recommendations = []
        
        # Design quality mitigations
        if risk_components.get('design_quality', 0) > 0.5:
            recommendations.append("Refactor code smells to improve design quality")
            
            # Specific smell mitigations
            god_classes = [s for s in smells if 'god_class' in s]
            if god_classes:
                recommendations.append("Break down large classes using Single Responsibility Principle")
            
            long_methods = [s for s in smells if 'long_method' in s]
            if long_methods:
                recommendations.append("Extract smaller methods from large functions")
        
        # Dependency management mitigations
        if risk_components.get('dependency_management', 0) > 0.5:
            recommendations.append("Review and optimize dependency management")
            
            if len(dependency_risks) > 3:
                recommendations.append("Implement dependency injection to reduce coupling")
        
        # Coupling and cohesion mitigations
        if risk_components.get('coupling_cohesion', 0) > 0.6:
            recommendations.append("Apply loose coupling and high cohesion principles")
            recommendations.append("Use interfaces and abstract classes to reduce direct dependencies")
        
        # Architectural debt mitigations
        if risk_components.get('architectural_debt', 0) > 0.4:
            recommendations.append("Address architectural violations systematically")
            
            layering_violations = [v for v in violations if 'layering_violation' in v]
            if layering_violations:
                recommendations.append("Enforce proper layering architecture")
        
        # Pattern compliance mitigations
        if risk_components.get('pattern_compliance', 0) > 0.6:
            recommendations.append("Introduce appropriate design patterns for better structure")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _cleanup_cache(self):
        """Clean expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, cache_time) in self.cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

def main():
    """CLI interface for architectural risk assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Architectural Risk Assessor')
    parser.add_argument('--files', nargs='+', required=True, help='Files to analyze')
    parser.add_argument('--project-context', help='JSON file with project context')
    parser.add_argument('--architecture-rules', help='JSON file with custom architecture rules')
    parser.add_argument('--output', choices=['score', 'level', 'debt', 'full'], 
                       default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize assessor
    assessor = ArchitecturalRiskAssessor()
    
    # Load project context if provided
    project_context = None
    if args.project_context and os.path.exists(args.project_context):
        with open(args.project_context, 'r') as f:
            project_context = json.load(f)
    
    # Load architecture rules if provided
    architecture_rules = None
    if args.architecture_rules and os.path.exists(args.architecture_rules):
        with open(args.architecture_rules, 'r') as f:
            architecture_rules = json.load(f)
    
    # Perform analysis
    profile = assessor.assess_architectural_risk(
        files=args.files,
        project_context=project_context,
        architecture_rules=architecture_rules
    )
    
    # Output results
    if args.output == 'score':
        print(f"{profile.overall_risk_score:.3f}")
    elif args.output == 'level':
        print(profile.risk_level)
    elif args.output == 'debt':
        print(json.dumps(profile.architectural_debt, indent=2))
    else:
        result = {
            'overall_risk_score': profile.overall_risk_score,
            'risk_level': profile.risk_level,
            'architecture_patterns': profile.architecture_patterns,
            'architectural_debt': profile.architectural_debt,
            'integration_risks': profile.integration_risks,
            'mitigation_recommendations': profile.mitigation_recommendations,
            'confidence': profile.confidence
        }
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()