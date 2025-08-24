#!/usr/bin/env python3
"""
RIF Dynamic Dependency Tracking System

Implements real-time dependency analysis and "how things work" visualization
that updates automatically as the system evolves. Tracks component relationships,
dependency changes, and provides impact assessment for modifications.

Addresses Issue #112 requirement for live dependency tracking and system relationships.
"""

import json
import os
import subprocess
import re
import ast
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import networkx as nx
from pathlib import Path
import yaml
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mpatches = None
    FancyBboxPatch = None
    MATPLOTLIB_AVAILABLE = False

class DependencyType(Enum):
    """Types of dependencies"""
    CODE = "code"              # Code-level dependencies (imports, functions)
    DATA = "data"              # Data flow dependencies
    CONFIG = "configuration"   # Configuration dependencies
    SERVICE = "service"        # Service-level dependencies
    WORKFLOW = "workflow"      # Workflow dependencies
    KNOWLEDGE = "knowledge"    # Knowledge/information dependencies
    TOOL = "tool"             # Tool/utility dependencies
    INTEGRATION = "integration" # External integration dependencies

class ImpactLevel(Enum):
    """Impact levels for dependency changes"""
    CRITICAL = "critical"     # System-breaking changes
    HIGH = "high"            # Major functionality impact
    MEDIUM = "medium"        # Moderate impact
    LOW = "low"              # Minor impact
    NONE = "none"            # No impact

class ChangeType(Enum):
    """Types of changes that can occur"""
    ADDED = "added"
    REMOVED = "removed" 
    MODIFIED = "modified"
    MOVED = "moved"
    RENAMED = "renamed"

@dataclass
class DependencyNode:
    """A node in the dependency graph"""
    id: str
    name: str
    type: str
    description: str
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.last_modified:
            data['last_modified'] = self.last_modified.isoformat()
        return data

@dataclass
class DependencyEdge:
    """An edge in the dependency graph"""
    source: str
    target: str
    dependency_type: DependencyType
    strength: float  # 0.0 to 1.0
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_verified: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['dependency_type'] = self.dependency_type.value
        if self.last_verified:
            data['last_verified'] = self.last_verified.isoformat()
        return data

@dataclass
class ChangeImpact:
    """Impact analysis for a change"""
    change_id: str
    change_type: ChangeType
    affected_component: str
    impact_level: ImpactLevel
    affected_components: List[str]
    risk_assessment: str
    mitigation_recommendations: List[str]
    estimated_effort: str
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['change_type'] = self.change_type.value
        data['impact_level'] = self.impact_level.value
        data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data

@dataclass
class DependencyGraph:
    """Complete dependency graph representation"""
    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }

class CodeAnalyzer:
    """Analyzes code dependencies"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def analyze_python_dependencies(self, file_path: Path) -> List[Tuple[str, DependencyType, str]]:
        """Analyze Python file dependencies"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        dependencies.append((name.name, DependencyType.CODE, f"import {name.name}"))
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        dependencies.append((f"{module}.{name.name}", DependencyType.CODE, 
                                          f"from {module} import {name.name}"))
                
                elif isinstance(node, ast.Call):
                    # Look for function calls that might be dependencies
                    if isinstance(node.func, ast.Attribute):
                        if hasattr(node.func.value, 'id'):
                            obj_name = node.func.value.id
                            method_name = node.func.attr
                            dependencies.append((f"{obj_name}.{method_name}", DependencyType.CODE,
                                              f"method call: {obj_name}.{method_name}"))
        
        except Exception as e:
            print(f"Error analyzing Python file {file_path}: {e}")
        
        return dependencies
    
    def analyze_shell_dependencies(self, file_path: Path) -> List[Tuple[str, DependencyType, str]]:
        """Analyze shell script dependencies"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for command executions
            commands = re.findall(r'([a-zA-Z_][a-zA-Z0-9_-]*)\s+', content)
            for cmd in set(commands):
                if cmd in ['gh', 'git', 'python', 'npm', 'docker', 'curl', 'wget']:
                    dependencies.append((cmd, DependencyType.TOOL, f"shell command: {cmd}"))
            
            # Look for file references
            file_refs = re.findall(r'["\']([^"\']*\.[a-zA-Z0-9]+)["\']', content)
            for file_ref in set(file_refs):
                if not file_ref.startswith('http'):
                    dependencies.append((file_ref, DependencyType.DATA, f"file reference: {file_ref}"))
        
        except Exception as e:
            print(f"Error analyzing shell file {file_path}: {e}")
        
        return dependencies
    
    def analyze_yaml_dependencies(self, file_path: Path) -> List[Tuple[str, DependencyType, str]]:
        """Analyze YAML configuration dependencies"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Extract references from YAML structure
            def extract_refs(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        if isinstance(value, str):
                            # Look for agent references
                            if 'agent' in key.lower() and value.startswith('rif-'):
                                dependencies.append((value, DependencyType.WORKFLOW, 
                                                  f"agent reference: {value}"))
                            # Look for file references
                            elif '/' in value and '.' in value:
                                dependencies.append((value, DependencyType.CONFIG,
                                                  f"file reference: {value}"))
                        else:
                            extract_refs(value, new_path)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_refs(item, path)
            
            extract_refs(data)
        
        except Exception as e:
            print(f"Error analyzing YAML file {file_path}: {e}")
        
        return dependencies
    
    def analyze_markdown_dependencies(self, file_path: Path) -> List[Tuple[str, DependencyType, str]]:
        """Analyze Markdown file dependencies (agent instructions, etc.)"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for agent references
            agent_refs = re.findall(r'RIF-(\w+)', content)
            for agent in set(agent_refs):
                dependencies.append((f"rif-{agent.lower()}", DependencyType.WORKFLOW,
                                  f"agent reference: RIF-{agent}"))
            
            # Look for tool references
            tool_refs = re.findall(r'`([a-zA-Z_][a-zA-Z0-9_-]*)`', content)
            for tool in set(tool_refs):
                if tool in ['gh', 'git', 'npm', 'docker', 'pytest']:
                    dependencies.append((tool, DependencyType.TOOL, f"tool reference: {tool}"))
            
            # Look for file path references
            file_refs = re.findall(r'`([^`]*\.[a-zA-Z0-9]+)`', content)
            for file_ref in set(file_refs):
                if '/' in file_ref:
                    dependencies.append((file_ref, DependencyType.DATA, f"file reference: {file_ref}"))
            
            # Look for MCP tool references
            mcp_refs = re.findall(r'mcp__([a-zA-Z_][a-zA-Z0-9_-]*)', content)
            for mcp_tool in set(mcp_refs):
                dependencies.append((f"mcp__{mcp_tool}", DependencyType.INTEGRATION,
                                  f"MCP tool: mcp__{mcp_tool}"))
        
        except Exception as e:
            print(f"Error analyzing Markdown file {file_path}: {e}")
        
        return dependencies

class DependencyScanner:
    """Scans repository for dependencies"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.code_analyzer = CodeAnalyzer(repo_path)
        
        # File type analyzers
        self.analyzers = {
            '.py': self.code_analyzer.analyze_python_dependencies,
            '.sh': self.code_analyzer.analyze_shell_dependencies,
            '.yaml': self.code_analyzer.analyze_yaml_dependencies,
            '.yml': self.code_analyzer.analyze_yaml_dependencies,
            '.md': self.code_analyzer.analyze_markdown_dependencies,
        }
    
    def scan_repository(self) -> DependencyGraph:
        """Scan entire repository for dependencies"""
        print("Scanning repository for dependencies...")
        
        graph = DependencyGraph()
        
        # Scan all relevant files
        for file_path in self._get_scannable_files():
            node_id = str(file_path.relative_to(self.repo_path))
            
            # Create node for file
            file_node = DependencyNode(
                id=node_id,
                name=file_path.name,
                type=self._determine_file_type(file_path),
                description=f"File: {file_path.name}",
                file_path=str(file_path),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                checksum=self._calculate_checksum(file_path)
            )
            
            graph.nodes[node_id] = file_node
            
            # Analyze dependencies
            file_dependencies = self._analyze_file_dependencies(file_path)
            
            for dep_target, dep_type, dep_description in file_dependencies:
                # Create dependency node if it doesn't exist
                if dep_target not in graph.nodes:
                    dep_node = DependencyNode(
                        id=dep_target,
                        name=dep_target.split('.')[-1] if '.' in dep_target else dep_target,
                        type=self._infer_dependency_type(dep_target, dep_type),
                        description=f"Dependency: {dep_target}"
                    )
                    graph.nodes[dep_target] = dep_node
                
                # Create edge
                edge = DependencyEdge(
                    source=node_id,
                    target=dep_target,
                    dependency_type=dep_type,
                    strength=self._calculate_dependency_strength(dep_type, dep_description),
                    description=dep_description,
                    last_verified=datetime.now()
                )
                
                graph.edges.append(edge)
                
                # Update node relationships
                file_node.dependencies.append(dep_target)
                graph.nodes[dep_target].dependents.append(node_id)
        
        # Add system-level components
        self._add_system_components(graph)
        
        graph.last_updated = datetime.now()
        return graph
    
    def _get_scannable_files(self) -> List[Path]:
        """Get list of files to scan for dependencies"""
        scannable_files = []
        
        # Directories to scan
        scan_dirs = [
            "claude/agents",
            "config", 
            "systems",
            "knowledge",
            "."  # Root directory
        ]
        
        for scan_dir in scan_dirs:
            dir_path = self.repo_path / scan_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix in self.analyzers:
                        # Skip hidden files and common ignored files
                        if not any(part.startswith('.') for part in file_path.parts):
                            scannable_files.append(file_path)
        
        return scannable_files
    
    def _analyze_file_dependencies(self, file_path: Path) -> List[Tuple[str, DependencyType, str]]:
        """Analyze dependencies for a single file"""
        analyzer = self.analyzers.get(file_path.suffix)
        if analyzer:
            return analyzer(file_path)
        return []
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of file"""
        if 'agent' in str(file_path):
            return 'agent'
        elif 'config' in str(file_path):
            return 'configuration'
        elif 'system' in str(file_path):
            return 'system_tool'
        elif file_path.suffix == '.py':
            return 'python_module'
        elif file_path.suffix in ['.sh', '.bash']:
            return 'shell_script'
        elif file_path.suffix in ['.yaml', '.yml']:
            return 'configuration'
        elif file_path.suffix == '.md':
            return 'documentation'
        else:
            return 'unknown'
    
    def _infer_dependency_type(self, dep_target: str, dep_type: DependencyType) -> str:
        """Infer the type of dependency target"""
        if dep_target.startswith('rif-'):
            return 'agent'
        elif dep_target.startswith('mcp__'):
            return 'mcp_tool'
        elif dep_target in ['gh', 'git', 'python', 'npm']:
            return 'external_tool'
        elif '/' in dep_target and '.' in dep_target:
            return 'file'
        elif dep_type == DependencyType.CODE:
            return 'module'
        else:
            return 'unknown'
    
    def _calculate_dependency_strength(self, dep_type: DependencyType, description: str) -> float:
        """Calculate dependency strength based on type and usage"""
        base_strength = {
            DependencyType.CODE: 0.8,
            DependencyType.WORKFLOW: 0.9,
            DependencyType.TOOL: 0.6,
            DependencyType.CONFIG: 0.7,
            DependencyType.DATA: 0.5,
            DependencyType.INTEGRATION: 0.8,
            DependencyType.SERVICE: 0.9,
            DependencyType.KNOWLEDGE: 0.4
        }.get(dep_type, 0.5)
        
        # Adjust based on description
        if 'critical' in description.lower():
            base_strength += 0.2
        elif 'optional' in description.lower():
            base_strength -= 0.2
        
        return min(1.0, max(0.1, base_strength))
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _add_system_components(self, graph: DependencyGraph) -> None:
        """Add system-level components to the graph"""
        system_components = [
            ("github-integration", "GitHub Integration", "integration"),
            ("git-vcs", "Git Version Control", "tool"),
            ("claude-code", "Claude Code Platform", "integration"),
            ("workflow-engine", "RIF Workflow Engine", "framework"),
            ("knowledge-base", "RIF Knowledge Base", "knowledge"),
            ("quality-gates", "Quality Gate System", "framework")
        ]
        
        for comp_id, comp_name, comp_type in system_components:
            if comp_id not in graph.nodes:
                node = DependencyNode(
                    id=comp_id,
                    name=comp_name,
                    type=comp_type,
                    description=f"System component: {comp_name}"
                )
                graph.nodes[comp_id] = node

class DependencyAnalyzer:
    """Analyzes dependency graphs for insights"""
    
    def __init__(self):
        self.graph = None
    
    def analyze_graph(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Perform comprehensive graph analysis"""
        self.graph = graph
        
        analysis = {
            "summary": self._generate_summary(),
            "critical_paths": self._find_critical_paths(),
            "circular_dependencies": self._find_circular_dependencies(),
            "orphaned_components": self._find_orphaned_components(),
            "hub_components": self._find_hub_components(),
            "fragile_connections": self._find_fragile_connections(),
            "dependency_chains": self._analyze_dependency_chains(),
            "impact_zones": self._identify_impact_zones()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate graph summary statistics"""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "node_types": self._count_node_types(),
            "dependency_types": self._count_dependency_types(),
            "average_dependencies": self._calculate_average_dependencies(),
            "max_dependency_depth": self._calculate_max_depth()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for node in self.graph.nodes.values():
            node_type = node.type
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_dependency_types(self) -> Dict[str, int]:
        """Count edges by dependency type"""
        type_counts = {}
        for edge in self.graph.edges:
            dep_type = edge.dependency_type.value
            type_counts[dep_type] = type_counts.get(dep_type, 0) + 1
        return type_counts
    
    def _calculate_average_dependencies(self) -> float:
        """Calculate average number of dependencies per node"""
        if not self.graph.nodes:
            return 0.0
        
        total_deps = sum(len(node.dependencies) for node in self.graph.nodes.values())
        return total_deps / len(self.graph.nodes)
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum dependency depth"""
        # This would require topological sorting for accurate depth calculation
        # For now, return a simple approximation
        max_deps = max((len(node.dependencies) for node in self.graph.nodes.values()), default=0)
        return min(max_deps, 10)  # Cap at reasonable depth
    
    def _find_critical_paths(self) -> List[List[str]]:
        """Find critical dependency paths"""
        critical_paths = []
        
        # Find paths with high-strength dependencies
        for edge in self.graph.edges:
            if edge.strength > 0.8:
                # Simple path construction (could be enhanced with graph traversal)
                path = [edge.source, edge.target]
                critical_paths.append(path)
        
        return critical_paths[:10]  # Limit results
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies"""
        # Simple circular dependency detection
        circular_deps = []
        
        for node_id, node in self.graph.nodes.items():
            for dep_id in node.dependencies:
                dep_node = self.graph.nodes.get(dep_id)
                if dep_node and node_id in dep_node.dependencies:
                    circular_deps.append([node_id, dep_id])
        
        return circular_deps
    
    def _find_orphaned_components(self) -> List[str]:
        """Find components with no dependencies or dependents"""
        orphaned = []
        
        for node_id, node in self.graph.nodes.items():
            if not node.dependencies and not node.dependents:
                orphaned.append(node_id)
        
        return orphaned
    
    def _find_hub_components(self) -> List[Tuple[str, int]]:
        """Find components with many dependents (hubs)"""
        hubs = []
        
        for node_id, node in self.graph.nodes.items():
            dependent_count = len(node.dependents)
            if dependent_count > 3:  # Threshold for hub
                hubs.append((node_id, dependent_count))
        
        return sorted(hubs, key=lambda x: x[1], reverse=True)[:10]
    
    def _find_fragile_connections(self) -> List[Dict[str, Any]]:
        """Find connections that could be fragile"""
        fragile = []
        
        for edge in self.graph.edges:
            if edge.strength < 0.3:  # Low strength threshold
                fragile.append({
                    "source": edge.source,
                    "target": edge.target,
                    "strength": edge.strength,
                    "type": edge.dependency_type.value,
                    "description": edge.description
                })
        
        return fragile[:10]
    
    def _analyze_dependency_chains(self) -> List[Dict[str, Any]]:
        """Analyze long dependency chains"""
        chains = []
        
        # Find nodes with many dependencies
        for node_id, node in self.graph.nodes.items():
            if len(node.dependencies) > 5:
                chains.append({
                    "component": node_id,
                    "dependency_count": len(node.dependencies),
                    "dependencies": node.dependencies[:10]  # Limit for readability
                })
        
        return sorted(chains, key=lambda x: x['dependency_count'], reverse=True)[:5]
    
    def _identify_impact_zones(self) -> List[Dict[str, Any]]:
        """Identify zones that would be impacted by changes"""
        impact_zones = []
        
        # Group related components
        agent_zone = [node_id for node_id, node in self.graph.nodes.items() 
                     if node.type == 'agent']
        config_zone = [node_id for node_id, node in self.graph.nodes.items()
                      if node.type == 'configuration']
        system_zone = [node_id for node_id, node in self.graph.nodes.items()
                      if node.type == 'system_tool']
        
        zones = [
            ("Agent Zone", agent_zone),
            ("Configuration Zone", config_zone),
            ("System Tools Zone", system_zone)
        ]
        
        for zone_name, zone_components in zones:
            if zone_components:
                # Calculate interconnectedness
                internal_edges = len([e for e in self.graph.edges 
                                    if e.source in zone_components and e.target in zone_components])
                external_edges = len([e for e in self.graph.edges 
                                    if (e.source in zone_components) != (e.target in zone_components)])
                
                impact_zones.append({
                    "zone": zone_name,
                    "components": len(zone_components),
                    "internal_connections": internal_edges,
                    "external_connections": external_edges,
                    "coupling_ratio": external_edges / max(len(zone_components), 1)
                })
        
        return impact_zones

class ChangeImpactAssessor:
    """Assesses impact of changes on dependency graph"""
    
    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.analyzer = DependencyAnalyzer()
    
    def assess_change_impact(self, component_id: str, change_type: ChangeType,
                           change_description: str = "") -> ChangeImpact:
        """Assess impact of a change to a component"""
        
        affected_components = self._find_affected_components(component_id)
        impact_level = self._assess_impact_level(component_id, change_type, affected_components)
        risk_assessment = self._assess_risk(component_id, change_type, affected_components)
        recommendations = self._generate_recommendations(component_id, change_type, 
                                                       affected_components, impact_level)
        effort_estimate = self._estimate_effort(impact_level, len(affected_components))
        
        return ChangeImpact(
            change_id=f"{component_id}-{change_type.value}-{int(time.time())}",
            change_type=change_type,
            affected_component=component_id,
            impact_level=impact_level,
            affected_components=affected_components,
            risk_assessment=risk_assessment,
            mitigation_recommendations=recommendations,
            estimated_effort=effort_estimate,
            analysis_timestamp=datetime.now()
        )
    
    def _find_affected_components(self, component_id: str) -> List[str]:
        """Find components affected by changes to given component"""
        affected = set()
        
        # Direct dependents
        node = self.graph.nodes.get(component_id)
        if node:
            affected.update(node.dependents)
            
            # Transitive dependents (one level)
            for dependent in node.dependents:
                dependent_node = self.graph.nodes.get(dependent)
                if dependent_node:
                    affected.update(dependent_node.dependents)
        
        return list(affected)
    
    def _assess_impact_level(self, component_id: str, change_type: ChangeType,
                           affected_components: List[str]) -> ImpactLevel:
        """Assess the impact level of a change"""
        
        node = self.graph.nodes.get(component_id)
        if not node:
            return ImpactLevel.NONE
        
        # Factors for impact assessment
        dependent_count = len(affected_components)
        is_system_component = node.type in ['framework', 'integration', 'workflow']
        is_removal = change_type == ChangeType.REMOVED
        
        # Calculate impact score
        impact_score = 0
        
        if dependent_count > 10:
            impact_score += 3
        elif dependent_count > 5:
            impact_score += 2
        elif dependent_count > 0:
            impact_score += 1
        
        if is_system_component:
            impact_score += 2
        
        if is_removal:
            impact_score += 2
        
        # Map score to level
        if impact_score >= 6:
            return ImpactLevel.CRITICAL
        elif impact_score >= 4:
            return ImpactLevel.HIGH
        elif impact_score >= 2:
            return ImpactLevel.MEDIUM
        elif impact_score >= 1:
            return ImpactLevel.LOW
        else:
            return ImpactLevel.NONE
    
    def _assess_risk(self, component_id: str, change_type: ChangeType,
                    affected_components: List[str]) -> str:
        """Assess risk of the change"""
        
        risks = []
        
        if change_type == ChangeType.REMOVED:
            risks.append("Breaking change - dependent components will fail")
        
        if len(affected_components) > 5:
            risks.append("Wide-reaching change affecting multiple systems")
        
        # Check for critical components in affected list
        critical_components = {'github-integration', 'workflow-engine', 'claude-code'}
        if any(comp in critical_components for comp in affected_components):
            risks.append("Affects critical system infrastructure")
        
        # Check for agent impacts
        agent_components = [comp for comp in affected_components 
                          if comp.startswith('agent-')]
        if len(agent_components) > 3:
            risks.append(f"Impacts {len(agent_components)} agents - workflow disruption possible")
        
        if not risks:
            risks.append("Low risk change with limited scope")
        
        return " | ".join(risks)
    
    def _generate_recommendations(self, component_id: str, change_type: ChangeType,
                                affected_components: List[str], 
                                impact_level: ImpactLevel) -> List[str]:
        """Generate mitigation recommendations"""
        
        recommendations = []
        
        if impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
            recommendations.append("Create rollback plan before implementing change")
            recommendations.append("Test change in isolated environment first")
        
        if change_type == ChangeType.REMOVED:
            recommendations.append("Update all dependent components before removal")
            recommendations.append("Deprecate component with warning period before removal")
        
        if len(affected_components) > 5:
            recommendations.append("Implement change in phases to reduce blast radius")
            recommendations.append("Monitor affected components closely after change")
        
        # Agent-specific recommendations
        agent_components = [comp for comp in affected_components 
                          if comp.startswith('agent-')]
        if agent_components:
            recommendations.append("Update agent instructions to reflect changes")
            recommendations.append("Test agent workflows after change implementation")
        
        if impact_level == ImpactLevel.CRITICAL:
            recommendations.append("Consider if change is absolutely necessary")
            recommendations.append("Get stakeholder approval before proceeding")
        
        return recommendations
    
    def _estimate_effort(self, impact_level: ImpactLevel, affected_count: int) -> str:
        """Estimate effort required for change"""
        
        base_effort = {
            ImpactLevel.CRITICAL: 8,    # 8+ hours
            ImpactLevel.HIGH: 4,        # 4-8 hours
            ImpactLevel.MEDIUM: 2,      # 2-4 hours
            ImpactLevel.LOW: 1,         # 1-2 hours
            ImpactLevel.NONE: 0.5       # <1 hour
        }.get(impact_level, 2)
        
        # Adjust based on affected components
        if affected_count > 10:
            base_effort *= 2
        elif affected_count > 5:
            base_effort *= 1.5
        
        if base_effort >= 8:
            return "8+ hours (Major effort)"
        elif base_effort >= 4:
            return "4-8 hours (Significant effort)"
        elif base_effort >= 2:
            return "2-4 hours (Moderate effort)"
        elif base_effort >= 1:
            return "1-2 hours (Minor effort)"
        else:
            return "<1 hour (Minimal effort)"

class DynamicDependencyTracker:
    """Main dynamic dependency tracking system"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = Path(repo_path)
        self.scanner = DependencyScanner(str(repo_path))
        self.analyzer = DependencyAnalyzer()
        self.current_graph = None
        self.previous_graph = None
        
        # Create storage directory
        self.storage_dir = self.repo_path / "systems" / "dependency-tracking"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance monitoring
        self.performance_metrics = {
            "scan_times": [],
            "analysis_times": [],
            "update_times": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        self._file_watchers = {}
    
    def initialize_tracking(self) -> DependencyGraph:
        """Initialize dependency tracking"""
        print("Initializing dynamic dependency tracking...")
        
        # Initial scan
        self.current_graph = self.scanner.scan_repository()
        
        # Save initial graph
        self._save_graph(self.current_graph, "initial")
        
        print(f"Dependency tracking initialized:")
        print(f"- {len(self.current_graph.nodes)} components tracked")
        print(f"- {len(self.current_graph.edges)} dependencies identified")
        
        return self.current_graph
    
    def update_dependencies(self, force_rescan: bool = False) -> Dict[str, Any]:
        """Update dependency tracking"""
        print("Updating dependency tracking...")
        
        # Store previous graph
        self.previous_graph = self.current_graph
        
        # Scan for current state
        new_graph = self.scanner.scan_repository()
        
        # Compare with previous if available
        changes_detected = []
        if self.previous_graph:
            changes_detected = self._detect_changes(self.previous_graph, new_graph)
        
        # Update current graph
        self.current_graph = new_graph
        
        # Save updated graph
        self._save_graph(self.current_graph, "updated")
        
        # Analyze current state
        analysis = self.analyzer.analyze_graph(self.current_graph)
        
        update_result = {
            "timestamp": datetime.now().isoformat(),
            "components_tracked": len(self.current_graph.nodes),
            "dependencies_tracked": len(self.current_graph.edges),
            "changes_detected": len(changes_detected),
            "changes": changes_detected,
            "analysis": analysis
        }
        
        print(f"Dependency tracking updated:")
        print(f"- {len(self.current_graph.nodes)} components")
        print(f"- {len(self.current_graph.edges)} dependencies") 
        print(f"- {len(changes_detected)} changes detected")
        
        return update_result
    
    def _detect_changes(self, old_graph: DependencyGraph, 
                       new_graph: DependencyGraph) -> List[Dict[str, Any]]:
        """Detect changes between graph versions"""
        changes = []
        
        # Detect node changes
        old_nodes = set(old_graph.nodes.keys())
        new_nodes = set(new_graph.nodes.keys())
        
        # Added nodes
        for node_id in new_nodes - old_nodes:
            changes.append({
                "type": "node_added",
                "component": node_id,
                "description": f"New component added: {node_id}"
            })
        
        # Removed nodes
        for node_id in old_nodes - new_nodes:
            changes.append({
                "type": "node_removed", 
                "component": node_id,
                "description": f"Component removed: {node_id}"
            })
        
        # Modified nodes (check checksums)
        for node_id in old_nodes & new_nodes:
            old_node = old_graph.nodes[node_id]
            new_node = new_graph.nodes[node_id]
            
            if old_node.checksum and new_node.checksum:
                if old_node.checksum != new_node.checksum:
                    changes.append({
                        "type": "node_modified",
                        "component": node_id,
                        "description": f"Component modified: {node_id}"
                    })
        
        # Detect edge changes (simplified)
        old_edge_pairs = {(e.source, e.target) for e in old_graph.edges}
        new_edge_pairs = {(e.source, e.target) for e in new_graph.edges}
        
        # Added dependencies
        for source, target in new_edge_pairs - old_edge_pairs:
            changes.append({
                "type": "dependency_added",
                "component": source,
                "description": f"New dependency: {source} -> {target}"
            })
        
        # Removed dependencies
        for source, target in old_edge_pairs - new_edge_pairs:
            changes.append({
                "type": "dependency_removed",
                "component": source, 
                "description": f"Dependency removed: {source} -> {target}"
            })
        
        return changes
    
    def assess_change_impact(self, component_id: str, 
                           change_type: str = "modified") -> ChangeImpact:
        """Assess impact of changes to a component"""
        if not self.current_graph:
            raise ValueError("Dependency tracking not initialized")
        
        assessor = ChangeImpactAssessor(self.current_graph)
        change_type_enum = ChangeType(change_type.lower())
        
        return assessor.assess_change_impact(component_id, change_type_enum)
    
    def get_component_info(self, component_id: str) -> Dict[str, Any]:
        """Get detailed information about a component"""
        if not self.current_graph or component_id not in self.current_graph.nodes:
            return {"error": f"Component {component_id} not found"}
        
        node = self.current_graph.nodes[component_id]
        
        # Find related edges
        incoming_edges = [e for e in self.current_graph.edges if e.target == component_id]
        outgoing_edges = [e for e in self.current_graph.edges if e.source == component_id]
        
        return {
            "component": node.to_dict(),
            "dependencies": {
                "count": len(node.dependencies),
                "details": [{"target": e.target, "type": e.dependency_type.value, 
                           "strength": e.strength, "description": e.description}
                          for e in outgoing_edges]
            },
            "dependents": {
                "count": len(node.dependents),
                "details": [{"source": e.source, "type": e.dependency_type.value,
                           "strength": e.strength, "description": e.description}
                          for e in incoming_edges]
            }
        }
    
    def generate_how_things_work_documentation(self) -> str:
        """Generate 'how things work' documentation"""
        if not self.current_graph:
            return "Dependency tracking not initialized"
        
        analysis = self.analyzer.analyze_graph(self.current_graph)
        
        doc = []
        doc.append("# How Things Work - RIF System Dependencies")
        doc.append(f"**Generated**: {datetime.now().isoformat()}")
        doc.append(f"**Components**: {len(self.current_graph.nodes)}")
        doc.append(f"**Dependencies**: {len(self.current_graph.edges)}")
        doc.append("")
        
        doc.append("## System Overview")
        doc.append(f"The RIF system consists of {analysis['summary']['total_nodes']} components "
                  f"connected by {analysis['summary']['total_edges']} dependencies.")
        doc.append("")
        
        doc.append("## Component Types")
        for comp_type, count in analysis['summary']['node_types'].items():
            doc.append(f"- **{comp_type.title()}**: {count} components")
        doc.append("")
        
        doc.append("## Key Architecture Patterns")
        if analysis['hub_components']:
            doc.append("### Hub Components (High Dependents)")
            for hub, dependent_count in analysis['hub_components'][:5]:
                doc.append(f"- **{hub}**: {dependent_count} dependents")
            doc.append("")
        
        doc.append("### Critical Paths")
        for path in analysis['critical_paths'][:5]:
            doc.append(f"- {' → '.join(path)}")
        doc.append("")
        
        if analysis['circular_dependencies']:
            doc.append("### Circular Dependencies (Potential Issues)")
            for cycle in analysis['circular_dependencies'][:5]:
                doc.append(f"- {' ↔ '.join(cycle)}")
            doc.append("")
        
        doc.append("## Impact Zones")
        for zone in analysis['impact_zones']:
            doc.append(f"### {zone['zone']}")
            doc.append(f"- Components: {zone['components']}")
            doc.append(f"- Internal connections: {zone['internal_connections']}")
            doc.append(f"- External connections: {zone['external_connections']}")
            doc.append(f"- Coupling ratio: {zone['coupling_ratio']:.2f}")
            doc.append("")
        
        if analysis['fragile_connections']:
            doc.append("## Fragile Connections (Monitor Closely)")
            for connection in analysis['fragile_connections'][:5]:
                doc.append(f"- **{connection['source']}** → **{connection['target']}** "
                          f"(strength: {connection['strength']:.2f}) - {connection['description']}")
            doc.append("")
        
        doc.append("## Maintenance Recommendations")
        
        if analysis['orphaned_components']:
            doc.append("- **Orphaned components**: Consider removing or integrating: " + 
                      ", ".join(analysis['orphaned_components'][:3]))
        
        if analysis['circular_dependencies']:
            doc.append("- **Circular dependencies**: Refactor to remove cycles")
        
        if analysis['fragile_connections']:
            doc.append("- **Fragile connections**: Strengthen low-strength dependencies")
        
        doc.append("\n---")
        doc.append("*This documentation is automatically generated from dependency analysis*")
        
        return "\n".join(doc)
    
    def start_real_time_monitoring(self, check_interval: int = 60) -> None:
        """Start real-time dependency monitoring"""
        if self._monitoring_active:
            print("Real-time monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        print(f"Started real-time dependency monitoring (check interval: {check_interval}s)")
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time dependency monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        print("Stopped real-time dependency monitoring")
    
    def _monitoring_loop(self, check_interval: int) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Check for changes and update if needed
                update_result = self.update_dependencies()
                if update_result['changes_detected'] > 0:
                    print(f"Real-time update: {update_result['changes_detected']} changes detected")
                    
                    # Log significant changes
                    for change in update_result['changes']:
                        if change['type'] in ['node_removed', 'dependency_removed']:
                            logging.warning(f"Breaking change detected: {change['description']}")
                        
                time.sleep(check_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def generate_interactive_visualization(self, output_path: Optional[str] = None) -> str:
        """Generate interactive dependency visualization"""
        if not self.current_graph:
            raise ValueError("No dependency graph available")
        
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available - generating text-based visualization")
            return self._generate_text_visualization(output_path)
        
        # Create the visualization
        plt.style.use('default')  # Use default style as seaborn may not be available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle('RIF Dynamic Dependency Analysis', fontsize=16, fontweight='bold')
        
        # Left plot: Dependency Graph
        self._plot_dependency_graph(ax1)
        
        # Right plot: Analysis Summary
        self._plot_analysis_summary(ax2)
        
        # Save the plot
        if not output_path:
            output_path = self.storage_dir / "dependency-visualization.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Interactive visualization saved to: {output_path}")
        
        return str(output_path)
    
    def _generate_text_visualization(self, output_path: Optional[str] = None) -> str:
        """Generate text-based visualization when matplotlib is not available"""
        if not output_path:
            output_path = self.storage_dir / "dependency-visualization.txt"
        
        analysis = self.analyzer.analyze_graph(self.current_graph)
        
        content = []
        content.append("RIF DYNAMIC DEPENDENCY ANALYSIS")
        content.append("=" * 40)
        content.append("")
        
        # Summary
        summary = analysis['summary']
        content.append("SYSTEM SUMMARY")
        content.append("-" * 20)
        content.append(f"Total Components: {summary['total_nodes']}")
        content.append(f"Total Dependencies: {summary['total_edges']}")
        content.append(f"Average Dependencies/Component: {summary['average_dependencies']:.1f}")
        content.append("")
        
        # Component Types
        content.append("COMPONENT TYPES")
        content.append("-" * 20)
        for comp_type, count in summary['node_types'].items():
            content.append(f"  {comp_type}: {count}")
        content.append("")
        
        # Dependency Types
        content.append("DEPENDENCY TYPES")
        content.append("-" * 20)
        for dep_type, count in summary['dependency_types'].items():
            content.append(f"  {dep_type}: {count}")
        content.append("")
        
        # Hub Components
        if analysis['hub_components']:
            content.append("HUB COMPONENTS")
            content.append("-" * 20)
            for hub, count in analysis['hub_components'][:5]:
                content.append(f"  {hub}: {count} dependents")
            content.append("")
        
        # Impact Zones
        content.append("IMPACT ZONES")
        content.append("-" * 20)
        for zone in analysis['impact_zones']:
            content.append(f"  {zone['zone']}: {zone['components']} components")
            content.append(f"    Internal connections: {zone['internal_connections']}")
            content.append(f"    External connections: {zone['external_connections']}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Text-based visualization saved to: {output_path}")
        return str(output_path)
    
    def _plot_dependency_graph(self, ax) -> None:
        """Plot the dependency graph structure"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        G = nx.DiGraph()
        
        # Add nodes and edges to NetworkX graph
        for node_id, node in self.current_graph.nodes.items():
            G.add_node(node_id, type=node.type, name=node.name)
        
        for edge in self.current_graph.edges:
            G.add_edge(edge.source, edge.target, strength=edge.strength,
                      dep_type=edge.dependency_type.value)
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Color nodes by type
        node_colors = {
            'agent': '#FF6B6B',
            'configuration': '#4ECDC4', 
            'system_tool': '#45B7D1',
            'python_module': '#96CEB4',
            'shell_script': '#FECA57',
            'documentation': '#FF9FF3'
        }
        
        colors = [node_colors.get(self.current_graph.nodes[node].type, '#DDD') 
                 for node in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, ax=ax, node_color=colors, node_size=300, 
               with_labels=False, arrows=True, edge_color='gray',
               arrowsize=10, alpha=0.8)
        
        # Add node labels for important nodes
        important_nodes = {node: data for node, data in G.nodes(data=True)
                          if data['type'] in ['agent', 'system_tool']}
        
        for node, (x, y) in pos.items():
            if node in important_nodes:
                ax.text(x, y+0.1, self.current_graph.nodes[node].name[:10], 
                       horizontalalignment='center', fontsize=8)
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=node_type.replace('_', ' ').title()) 
                          for node_type, color in node_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        ax.set_title('System Dependency Graph', fontweight='bold')
        ax.set_aspect('equal')
    
    def _plot_analysis_summary(self, ax) -> None:
        """Plot analysis summary statistics"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        analysis = self.analyzer.analyze_graph(self.current_graph)
        
        # Prepare data for visualization
        node_types = analysis['summary']['node_types']
        dep_types = analysis['summary']['dependency_types']
        
        # Create subplot layout
        ax.axis('off')
        
        # Node types pie chart
        ax1 = plt.subplot(2, 2, 2)
        if node_types:
            ax1.pie(node_types.values(), labels=node_types.keys(), autopct='%1.1f%%')
            ax1.set_title('Component Types')
        
        # Dependency types bar chart
        ax2 = plt.subplot(2, 2, 4)
        if dep_types:
            bars = ax2.bar(dep_types.keys(), dep_types.values())
            ax2.set_title('Dependency Types')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color bars
            colors = plt.cm.Set3(range(len(dep_types)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Summary statistics text
        ax3 = plt.subplot(2, 2, (1, 3))
        ax3.axis('off')
        
        summary_text = f"""
        SYSTEM ANALYSIS SUMMARY
        
        Components: {analysis['summary']['total_nodes']}
        Dependencies: {analysis['summary']['total_edges']}
        Avg Dependencies/Component: {analysis['summary']['average_dependencies']:.1f}
        Max Dependency Depth: {analysis['summary']['max_dependency_depth']}
        
        Hub Components: {len(analysis['hub_components'])}
        Circular Dependencies: {len(analysis['circular_dependencies'])}
        Orphaned Components: {len(analysis['orphaned_components'])}
        Fragile Connections: {len(analysis['fragile_connections'])}
        
        Impact Zones:
        """
        
        for zone in analysis['impact_zones'][:3]:
            summary_text += f"\n  • {zone['zone']}: {zone['components']} components"
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def benchmark_performance(self, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark system performance"""
        print(f"Running performance benchmark ({iterations} iterations)...")
        
        benchmark_results = {
            "scan_times": [],
            "analysis_times": [],
            "total_times": [],
            "components_analyzed": [],
            "dependencies_analyzed": []
        }
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")
            
            # Time the full scan process
            start_time = time.time()
            graph = self.scanner.scan_repository()
            scan_time = time.time() - start_time
            
            # Time the analysis process
            start_time = time.time()
            analysis = self.analyzer.analyze_graph(graph)
            analysis_time = time.time() - start_time
            
            total_time = scan_time + analysis_time
            
            benchmark_results["scan_times"].append(scan_time)
            benchmark_results["analysis_times"].append(analysis_time)
            benchmark_results["total_times"].append(total_time)
            benchmark_results["components_analyzed"].append(len(graph.nodes))
            benchmark_results["dependencies_analyzed"].append(len(graph.edges))
        
        # Calculate statistics
        stats = {}
        for key in ["scan_times", "analysis_times", "total_times"]:
            times = benchmark_results[key]
            stats[key] = {
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "median": sorted(times)[len(times)//2]
            }
        
        # Performance assessment
        avg_total_time = stats["total_times"]["mean"]
        performance_rating = "excellent" if avg_total_time < 5 else \
                           "good" if avg_total_time < 15 else \
                           "acceptable" if avg_total_time < 30 else "needs_optimization"
        
        benchmark_summary = {
            "iterations": iterations,
            "statistics": stats,
            "performance_rating": performance_rating,
            "meets_requirements": avg_total_time < 30,  # 30 second requirement
            "components_count": benchmark_results["components_analyzed"][-1],
            "dependencies_count": benchmark_results["dependencies_analyzed"][-1]
        }
        
        print(f"Benchmark complete:")
        print(f"  Average total time: {avg_total_time:.2f}s")
        print(f"  Performance rating: {performance_rating}")
        print(f"  Meets requirements: {benchmark_summary['meets_requirements']}")
        
        return benchmark_summary
    
    def generate_change_impact_report(self, component_id: str, change_type: str = "modified") -> Dict[str, Any]:
        """Generate comprehensive change impact report"""
        if not self.current_graph:
            raise ValueError("Dependency tracking not initialized")
        
        impact = self.assess_change_impact(component_id, change_type)
        
        # Enhanced impact analysis
        component_info = self.get_component_info(component_id)
        
        # Generate recommendations based on impact
        detailed_recommendations = self._generate_detailed_recommendations(impact, component_info)
        
        # Risk assessment matrix
        risk_matrix = self._generate_risk_matrix(impact)
        
        report = {
            "component_id": component_id,
            "change_type": change_type,
            "impact_assessment": impact.to_dict(),
            "component_details": component_info,
            "detailed_recommendations": detailed_recommendations,
            "risk_matrix": risk_matrix,
            "testing_checklist": self._generate_testing_checklist(impact),
            "rollback_plan": self._generate_rollback_plan(impact),
            "monitoring_plan": self._generate_monitoring_plan(impact)
        }
        
        return report
    
    def _generate_detailed_recommendations(self, impact: ChangeImpact, component_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed recommendations with priorities"""
        recommendations = []
        
        for rec in impact.mitigation_recommendations:
            priority = "high" if impact.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH] else "medium"
            
            recommendations.append({
                "action": rec,
                "priority": priority,
                "estimated_time": self._estimate_recommendation_time(rec),
                "responsible": self._assign_responsibility(rec)
            })
        
        return recommendations
    
    def _generate_risk_matrix(self, impact: ChangeImpact) -> Dict[str, Any]:
        """Generate risk assessment matrix"""
        return {
            "impact_level": impact.impact_level.value,
            "affected_component_count": len(impact.affected_components),
            "risk_factors": {
                "breaking_changes": impact.change_type == ChangeType.REMOVED,
                "wide_impact": len(impact.affected_components) > 5,
                "critical_systems": any("critical" in comp for comp in impact.affected_components),
                "agent_impacts": len([c for c in impact.affected_components if "agent" in c]) > 0
            },
            "mitigation_coverage": len(impact.mitigation_recommendations) / max(len(impact.affected_components), 1)
        }
    
    def _generate_testing_checklist(self, impact: ChangeImpact) -> List[str]:
        """Generate testing checklist for the change"""
        checklist = [
            "Unit tests for modified component",
            "Integration tests for direct dependencies"
        ]
        
        if len(impact.affected_components) > 3:
            checklist.append("System-wide regression testing")
        
        if impact.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
            checklist.extend([
                "Performance impact testing",
                "Load testing under realistic conditions",
                "Rollback procedure validation"
            ])
        
        return checklist
    
    def _generate_rollback_plan(self, impact: ChangeImpact) -> Dict[str, Any]:
        """Generate rollback plan for the change"""
        plan = {
            "required": impact.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH],
            "steps": [],
            "validation": [],
            "estimated_time": "5-15 minutes"
        }
        
        if plan["required"]:
            plan["steps"] = [
                "Create backup of current state",
                "Document change points",
                "Prepare revert commits",
                "Test rollback in staging environment"
            ]
            
            plan["validation"] = [
                "Verify system functionality post-rollback",
                "Check all affected components",
                "Monitor system performance"
            ]
        
        return plan
    
    def _generate_monitoring_plan(self, impact: ChangeImpact) -> Dict[str, Any]:
        """Generate monitoring plan post-change"""
        return {
            "duration": "24-48 hours" if impact.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH] else "4-8 hours",
            "metrics_to_monitor": [
                "System response times",
                "Error rates in affected components",
                "Dependency resolution success"
            ],
            "alert_thresholds": {
                "error_rate_increase": ">10% from baseline",
                "response_time_degradation": ">20% from baseline"
            }
        }
    
    def _estimate_recommendation_time(self, recommendation: str) -> str:
        """Estimate time for recommendation implementation"""
        if "rollback" in recommendation.lower():
            return "15-30 minutes"
        elif "test" in recommendation.lower():
            return "1-2 hours"
        elif "monitor" in recommendation.lower():
            return "Ongoing"
        else:
            return "30-60 minutes"
    
    def _assign_responsibility(self, recommendation: str) -> str:
        """Assign responsibility for recommendation"""
        if "test" in recommendation.lower():
            return "RIF-Validator"
        elif "rollback" in recommendation.lower():
            return "RIF-Implementer"
        elif "monitor" in recommendation.lower():
            return "RIF-Shadow-Auditor"
        else:
            return "RIF-Implementer"
    
    def _save_graph(self, graph: DependencyGraph, suffix: str = "") -> None:
        """Save dependency graph to file"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f"dependency-graph-{suffix}-{timestamp}.json" if suffix else f"dependency-graph-{timestamp}.json"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(graph.to_dict(), f, indent=2, default=str)
            
            # Also save as 'latest' for easy access
            latest_path = self.storage_dir / "dependency-graph-latest.json"
            with open(latest_path, 'w') as f:
                json.dump(graph.to_dict(), f, indent=2, default=str)
                
            print(f"Dependency graph saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving graph: {e}")
    
    def load_graph(self, filepath: Optional[str] = None) -> Optional[DependencyGraph]:
        """Load dependency graph from file"""
        if not filepath:
            filepath = self.storage_dir / "dependency-graph-latest.json"
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct graph from data (simplified)
            graph = DependencyGraph()
            graph.last_updated = datetime.fromisoformat(data['last_updated'])
            graph.version = data.get('version', '1.0')
            graph.metadata = data.get('metadata', {})
            
            # Load nodes
            for node_id, node_data in data['nodes'].items():
                node = DependencyNode(
                    id=node_data['id'],
                    name=node_data['name'],
                    type=node_data['type'],
                    description=node_data['description'],
                    file_path=node_data.get('file_path'),
                    dependencies=node_data.get('dependencies', []),
                    dependents=node_data.get('dependents', []),
                    metadata=node_data.get('metadata', {}),
                    checksum=node_data.get('checksum')
                )
                
                if node_data.get('last_modified'):
                    node.last_modified = datetime.fromisoformat(node_data['last_modified'])
                
                graph.nodes[node_id] = node
            
            # Load edges
            for edge_data in data['edges']:
                edge = DependencyEdge(
                    source=edge_data['source'],
                    target=edge_data['target'],
                    dependency_type=DependencyType(edge_data['dependency_type']),
                    strength=edge_data['strength'],
                    description=edge_data['description'],
                    metadata=edge_data.get('metadata', {})
                )
                
                if edge_data.get('last_verified'):
                    edge.last_verified = datetime.fromisoformat(edge_data['last_verified'])
                
                graph.edges.append(edge)
            
            self.current_graph = graph
            print(f"Dependency graph loaded from: {filepath}")
            return graph
            
        except Exception as e:
            print(f"Error loading graph: {e}")
            return None

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Dynamic Dependency Tracker")
    parser.add_argument("--init", action="store_true", help="Initialize dependency tracking")
    parser.add_argument("--update", action="store_true", help="Update dependency tracking")
    parser.add_argument("--analyze", action="store_true", help="Analyze current dependencies")
    parser.add_argument("--component", type=str, help="Get info about specific component")
    parser.add_argument("--impact", type=str, help="Assess impact of changing component")
    parser.add_argument("--docs", action="store_true", help="Generate how-things-work documentation")
    parser.add_argument("--repo", type=str, default="/Users/cal/DEV/RIF", help="Repository path")
    
    args = parser.parse_args()
    
    tracker = DynamicDependencyTracker(args.repo)
    
    if args.init:
        graph = tracker.initialize_tracking()
        print(f"Tracking initialized with {len(graph.nodes)} components")
    
    elif args.update:
        result = tracker.update_dependencies()
        print(f"Update complete - {result['changes_detected']} changes detected")
    
    elif args.analyze:
        if not tracker.current_graph:
            tracker.load_graph()
        
        if tracker.current_graph:
            analysis = tracker.analyzer.analyze_graph(tracker.current_graph)
            print("\n=== Dependency Analysis ===")
            print(f"Components: {analysis['summary']['total_nodes']}")
            print(f"Dependencies: {analysis['summary']['total_edges']}")
            print(f"Hub components: {len(analysis['hub_components'])}")
            print(f"Circular dependencies: {len(analysis['circular_dependencies'])}")
            print(f"Orphaned components: {len(analysis['orphaned_components'])}")
        else:
            print("No dependency graph available - run --init first")
    
    elif args.component:
        if not tracker.current_graph:
            tracker.load_graph()
        
        if tracker.current_graph:
            info = tracker.get_component_info(args.component)
            print(f"\n=== Component Info: {args.component} ===")
            print(json.dumps(info, indent=2, default=str))
        else:
            print("No dependency graph available - run --init first")
    
    elif args.impact:
        if not tracker.current_graph:
            tracker.load_graph()
        
        if tracker.current_graph:
            impact = tracker.assess_change_impact(args.impact, "modified")
            print(f"\n=== Change Impact Assessment: {args.impact} ===")
            print(f"Impact Level: {impact.impact_level.value}")
            print(f"Affected Components: {len(impact.affected_components)}")
            print(f"Risk Assessment: {impact.risk_assessment}")
            print(f"Estimated Effort: {impact.estimated_effort}")
            print("\nRecommendations:")
            for rec in impact.mitigation_recommendations:
                print(f"- {rec}")
        else:
            print("No dependency graph available - run --init first")
    
    elif args.docs:
        if not tracker.current_graph:
            tracker.load_graph()
        
        if tracker.current_graph:
            docs = tracker.generate_how_things_work_documentation()
            print(docs)
        else:
            print("No dependency graph available - run --init first")
    
    else:
        parser.print_help()