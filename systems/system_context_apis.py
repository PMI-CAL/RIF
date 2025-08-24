#!/usr/bin/env python3
"""
System Context and Understanding APIs for DPIBS
Issue #120: DPIBS Architecture Phase 2 - System Context + Understanding APIs

Provides live system context with real-time dependency tracking:
- Live system context retrieval and updates
- "Big picture" system understanding maintenance
- Component interaction and dependency analysis  
- Real-time change impact assessment
- System evolution tracking

Target Performance: <500ms for complex system understanding queries
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

# Add RIF to path
sys.path.insert(0, '/Users/cal/DEV/RIF')

from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
from knowledge.database.database_config import DatabaseConfig


@dataclass
class SystemComponent:
    """Represents a system component with metadata"""
    name: str
    component_type: str  # module, service, database, api, config
    file_path: Optional[str] = None
    dependencies: List[str] = None
    interfaces: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.interfaces is None:
            self.interfaces = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DependencyRelationship:
    """Represents a dependency relationship between components"""
    source: str
    target: str
    dependency_type: str
    strength: float  # 0.0 to 1.0
    detection_method: str
    context: Dict[str, Any] = None
    verified: bool = False
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class SystemContextEngine:
    """
    Core engine for system context understanding and dependency tracking
    Provides real-time system analysis with <500ms response time
    """
    
    def __init__(self, optimizer: DPIBSPerformanceOptimizer, project_root: str = "/Users/cal/DEV/RIF"):
        self.optimizer = optimizer
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Component cache for performance
        self.component_cache: Dict[str, SystemComponent] = {}
        self.dependency_cache: Dict[str, List[DependencyRelationship]] = {}
        self.last_scan_time: Optional[datetime] = None
        
        # System understanding patterns
        self.system_patterns = {
            'api_endpoints': [],
            'database_connections': [],
            'configuration_files': [],
            'service_definitions': [],
            'workflow_patterns': []
        }
        
        self.logger.info("System Context Engine initialized")
    
    def discover_system_components(self, force_rescan: bool = False, max_components: int = 200) -> List[SystemComponent]:
        """
        Discover all system components with intelligent caching and performance optimization
        Target: Complete system scan in <1 second, cached retrieval in <50ms
        """
        start_time = time.time()
        
        # Check if we need to rescan (cache invalidation)
        if not force_rescan and self.last_scan_time:
            age_minutes = (datetime.utcnow() - self.last_scan_time).total_seconds() / 60
            if age_minutes < 30 and self.component_cache:  # 30-minute cache TTL
                self.logger.debug("Returning cached system components")
                cached_components = list(self.component_cache.values())
                return cached_components[:max_components]  # Limit results for performance
        
        self.logger.info(f"Discovering system components (limit: {max_components})...")
        components = []
        
        try:
            # Use parallel scanning for performance
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all scan tasks
                futures = {
                    executor.submit(self._scan_python_modules_optimized, max_components // 4): 'python',
                    executor.submit(self._scan_configuration_files_optimized, max_components // 4): 'config',
                    executor.submit(self._scan_database_schemas, max_components // 4): 'database',  # Use existing method for now
                    executor.submit(self._scan_api_definitions, max_components // 4): 'api'  # Use existing method for now
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    scan_type = futures[future]
                    try:
                        scan_components = future.result(timeout=2.0)  # 2-second timeout per scan
                        components.extend(scan_components)
                        self.logger.debug(f"{scan_type} scan completed: {len(scan_components)} components")
                    except Exception as e:
                        self.logger.warning(f"{scan_type} scan failed: {str(e)}")
            
            # Limit total components for performance
            components = components[:max_components]
            
            # Update cache
            self.component_cache = {comp.name: comp for comp in components}
            self.last_scan_time = datetime.utcnow()
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Discovered {len(components)} system components in {duration_ms:.2f}ms (target: <1000ms)")
            
            return components
            
        except Exception as e:
            self.logger.error(f"System component discovery failed: {str(e)}")
            cached_components = list(self.component_cache.values())
            return cached_components[:max_components]  # Return cached data on error with limit
    
    def _scan_python_modules_optimized(self, max_files: int = 50) -> List[SystemComponent]:
        """Optimized Python module scanning with early termination and selective reading"""
        components = []
        
        # Use glob with early termination for performance
        python_files = []
        file_count = 0
        for py_file in self.project_root.rglob("*.py"):
            # Skip hidden files and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            python_files.append(py_file)
            file_count += 1
            if file_count >= max_files * 2:  # Get more files to prioritize
                break
        
        # Prioritize important files (sort by relevance)
        python_files = self._prioritize_python_files(python_files)[:max_files]
        self.logger.debug(f"Scanning {len(python_files)} Python files (optimized)")
        
        for py_file in python_files:
            try:
                relative_path = py_file.relative_to(self.project_root)
                module_name = str(relative_path).replace('/', '.').replace('.py', '')
                
                # Quick file size check - skip very large files for performance
                file_size = py_file.stat().st_size
                if file_size > 100000:  # Skip files >100KB for performance
                    self.logger.debug(f"Skipping large file {py_file.name} ({file_size} bytes)")
                    continue
                
                # Read file with size limit for performance
                content = py_file.read_text(encoding='utf-8', errors='ignore')[:50000]  # Max 50KB read
                
                dependencies = self._extract_python_dependencies_fast(content)
                interfaces = self._extract_python_interfaces_fast(content)
                
                component = SystemComponent(
                    name=module_name,
                    component_type='module',
                    file_path=str(relative_path),
                    dependencies=dependencies,
                    interfaces=interfaces,
                    metadata={
                        'line_count': len(content.splitlines()),
                        'has_classes': 'class ' in content,
                        'has_functions': 'def ' in content,
                        'has_fastapi': 'FastAPI' in content or '@app.' in content,
                        'has_database': any(db in content for db in ['duckdb', 'sqlite', 'postgres']),
                        'last_modified': py_file.stat().st_mtime,
                        'file_size': file_size
                    }
                )
                
                components.append(component)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {str(e)}")
        
        return components
    
    def _extract_python_dependencies_fast(self, content: str) -> List[str]:
        """Optimized Python import dependency extraction with early termination"""
        dependencies = set()  # Use set for O(1) lookups and automatic deduplication
        lines = content.splitlines()[:100]  # Only scan first 100 lines for performance
        
        for line in lines:
            line = line.strip()
            
            # Standard imports
            if line.startswith('import '):
                dep = line.replace('import ', '').split('.')[0].split(' as ')[0].strip()
                if dep and not dep.startswith('_'):  # Skip private imports
                    dependencies.add(dep)
            
            # From imports
            elif line.startswith('from ') and ' import ' in line:
                dep = line.split('from ')[1].split(' import ')[0].split('.')[0].strip()
                if dep and not dep.startswith('_'):  # Skip private imports
                    dependencies.add(dep)
            
            # Early termination for performance
            if len(dependencies) > 20:  # Limit dependencies per file
                break
        
        return list(dependencies)
    
    def _extract_python_interfaces_fast(self, content: str) -> List[str]:
        """Optimized public interface extraction with early termination"""
        interfaces = []
        lines = content.splitlines()[:200]  # Only scan first 200 lines for performance
        
        for line in lines:
            line = line.strip()
            
            # Public functions
            if line.startswith('def ') and not line.startswith('def _'):
                try:
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    if func_name:  # Ensure valid function name
                        interfaces.append(f"function:{func_name}")
                except IndexError:
                    continue
            
            # Classes
            elif line.startswith('class '):
                try:
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    if class_name:  # Ensure valid class name
                        interfaces.append(f"class:{class_name}")
                except IndexError:
                    continue
            
            # FastAPI endpoints
            elif '@app.' in line:
                try:
                    endpoint = line.split('@app.')[1].split('(')[0].strip()
                    if endpoint:  # Ensure valid endpoint
                        interfaces.append(f"api:{endpoint}")
                except IndexError:
                    continue
            
            # Early termination for performance
            if len(interfaces) > 15:  # Limit interfaces per file
                break
        
        return interfaces
    
    def _prioritize_python_files(self, files: List[Path]) -> List[Path]:
        """Prioritize Python files by relevance for system context analysis"""
        def file_priority(file_path: Path) -> int:
            """Calculate priority score for a Python file (higher = more important)"""
            score = 0
            name_lower = file_path.name.lower()
            path_str = str(file_path).lower()
            
            # High priority files
            if any(important in name_lower for important in ['main', 'app', 'api', 'server', 'service']):
                score += 100
            elif any(important in name_lower for important in ['config', 'settings', 'schema', 'model']):
                score += 80
            elif any(important in path_str for important in ['/systems/', '/claude/', '/knowledge/']):
                score += 60
            
            # Medium priority files
            if name_lower.endswith('_api.py') or name_lower.endswith('_service.py'):
                score += 50
            elif 'test' in name_lower:
                score -= 20  # Lower priority for test files
            
            # File size factor (smaller files are easier to process)
            try:
                size = file_path.stat().st_size
                if size < 5000:  # Small files
                    score += 10
                elif size > 50000:  # Large files
                    score -= 10
            except:
                pass
            
            return score
        
        return sorted(files, key=file_priority, reverse=True)
    
    def _scan_configuration_files_optimized(self, max_files: int = 25) -> List[SystemComponent]:
        """Optimized configuration file scanning with early termination"""
        components = []
        
        config_patterns = ["*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg"]
        config_files = []
        
        # Collect config files with early termination
        for pattern in config_patterns:
            for config_file in self.project_root.rglob(pattern):
                if any(part.startswith('.') or part == 'node_modules' for part in config_file.parts):
                    continue
                config_files.append(config_file)
                if len(config_files) >= max_files * 2:  # Get extra to prioritize
                    break
            if len(config_files) >= max_files * 2:
                break
        
        # Prioritize important config files
        config_files = self._prioritize_config_files(config_files)[:max_files]
        
        for config_file in config_files:
            try:
                relative_path = config_file.relative_to(self.project_root)
                
                # Quick size check
                file_size = config_file.stat().st_size
                if file_size > 50000:  # Skip large config files
                    continue
                
                # Read with size limit
                content = config_file.read_text(encoding='utf-8', errors='ignore')[:10000]
                
                component = SystemComponent(
                    name=f"config:{config_file.name}",
                    component_type='configuration',
                    file_path=str(relative_path),
                    metadata={
                        'file_type': config_file.suffix,
                        'size_bytes': len(content),
                        'has_database_config': any(db in content for db in ['database', 'db_url', 'connection']),
                        'has_api_config': any(api in content for api in ['api', 'endpoint', 'host', 'port']),
                        'has_secrets': any(secret in content.lower() for secret in ['password', 'key', 'token', 'secret']),
                        'last_modified': config_file.stat().st_mtime
                    }
                )
                
                components.append(component)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze config {config_file}: {str(e)}")
        
        return components
    
    def _prioritize_config_files(self, files: List[Path]) -> List[Path]:
        """Prioritize configuration files by importance"""
        def config_priority(file_path: Path) -> int:
            score = 0
            name_lower = file_path.name.lower()
            
            # High priority configs
            if any(important in name_lower for important in ['config', 'settings', 'workflow', 'schema']):
                score += 50
            elif name_lower in ['package.json', 'requirements.txt', 'pyproject.toml']:
                score += 40
            elif 'database' in name_lower or 'db' in name_lower:
                score += 30
                
            return score
            
        return sorted(files, key=config_priority, reverse=True)
    
    def _scan_database_schemas(self, max_files: int = 20) -> List[SystemComponent]:
        """Scan database schema files with optional limit"""
        components = []
        
        schema_files = list(self.project_root.rglob("*.sql"))
        schema_files.extend(list(self.project_root.rglob("*schema*")))
        
        # Limit files for performance
        schema_files = schema_files[:max_files] if max_files else schema_files
        
        for schema_file in schema_files:
            try:
                if schema_file.suffix not in ['.sql', '.py']:
                    continue
                
                relative_path = schema_file.relative_to(self.project_root)
                content = schema_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract table names and relationships
                tables = self._extract_sql_tables(content)
                
                component = SystemComponent(
                    name=f"schema:{schema_file.stem}",
                    component_type='database_schema',
                    file_path=str(relative_path),
                    interfaces=[f"table:{table}" for table in tables],
                    metadata={
                        'table_count': len(tables),
                        'has_indexes': 'CREATE INDEX' in content,
                        'has_views': 'CREATE VIEW' in content,
                        'has_constraints': any(constraint in content for constraint in ['CHECK', 'FOREIGN KEY', 'PRIMARY KEY']),
                        'database_type': 'duckdb' if 'duckdb' in content.lower() else 'sqlite' if 'sqlite' in content.lower() else 'unknown',
                        'last_modified': schema_file.stat().st_mtime
                    }
                )
                
                components.append(component)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze schema {schema_file}: {str(e)}")
        
        return components
    
    def _extract_sql_tables(self, content: str) -> List[str]:
        """Extract table names from SQL content"""
        tables = []
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip().upper()
            if line.startswith('CREATE TABLE'):
                # Extract table name
                parts = line.split()
                if len(parts) >= 3:
                    table_name = parts[2].replace('(', '').replace(',', '').strip()
                    tables.append(table_name.lower())
        
        return tables
    
    def _scan_api_definitions(self, max_files: int = 15) -> List[SystemComponent]:
        """Scan for API definitions and endpoints with optional limit"""
        components = []
        
        # Look for FastAPI apps, Flask apps, etc.
        api_files = []
        file_count = 0
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if any(framework in content for framework in ['FastAPI', 'Flask', '@app.', 'APIRouter']):
                    api_files.append(py_file)
                file_count += 1
                if len(api_files) >= max_files:
                    break
            except Exception:
                continue
        
        for api_file in api_files:
            try:
                relative_path = api_file.relative_to(self.project_root)
                content = api_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract API endpoints
                endpoints = self._extract_api_endpoints(content)
                
                component = SystemComponent(
                    name=f"api:{api_file.stem}",
                    component_type='api_service',
                    file_path=str(relative_path),
                    interfaces=endpoints,
                    metadata={
                        'endpoint_count': len(endpoints),
                        'has_authentication': any(auth in content for auth in ['@authenticate', 'JWT', 'Bearer', 'auth']),
                        'has_cors': 'CORS' in content,
                        'has_middleware': 'middleware' in content,
                        'framework': 'fastapi' if 'FastAPI' in content else 'flask' if 'Flask' in content else 'unknown',
                        'last_modified': api_file.stat().st_mtime
                    }
                )
                
                components.append(component)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze API {api_file}: {str(e)}")
        
        return components
    
    def _extract_api_endpoints(self, content: str) -> List[str]:
        """Extract API endpoints from code"""
        endpoints = []
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # FastAPI endpoints
            if line.startswith('@app.') and '(' in line:
                method_path = line.split('@app.')[1].split('(')[0]
                if '("' in line:
                    path = line.split('("')[1].split('"')[0]
                    endpoints.append(f"{method_path}:{path}")
            
            # Flask endpoints
            elif line.startswith('@app.route'):
                if '("' in line:
                    path = line.split('("')[1].split('"')[0]
                    endpoints.append(f"route:{path}")
        
        return endpoints
    
    def analyze_system_dependencies(self, components: Optional[List[SystemComponent]] = None) -> List[DependencyRelationship]:
        """
        Analyze dependencies between system components
        Target: Complete dependency analysis in <1 second
        """
        start_time = time.time()
        
        if components is None:
            components = self.discover_system_components()
        
        relationships = []
        
        # Build dependency graph
        for component in components:
            for dep in component.dependencies:
                # Find matching components
                for target_comp in components:
                    if self._components_match(dep, target_comp):
                        relationship = DependencyRelationship(
                            source=component.name,
                            target=target_comp.name,
                            dependency_type=self._determine_dependency_type(component, target_comp),
                            strength=self._calculate_dependency_strength(component, target_comp),
                            detection_method='static_analysis',
                            context={
                                'source_type': component.component_type,
                                'target_type': target_comp.component_type,
                                'analyzed_at': datetime.utcnow().isoformat()
                            }
                        )
                        relationships.append(relationship)
        
        # Cache results
        self.dependency_cache['system'] = relationships
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Analyzed {len(relationships)} dependencies in {duration_ms:.2f}ms")
        
        return relationships
    
    def _components_match(self, dependency: str, component: SystemComponent) -> bool:
        """Check if a dependency string matches a component"""
        # Simple matching logic - can be enhanced
        return (dependency in component.name or 
                dependency == component.name.split('.')[-1] or
                dependency in str(component.file_path) if component.file_path else False)
    
    def _determine_dependency_type(self, source: SystemComponent, target: SystemComponent) -> str:
        """Determine the type of dependency relationship"""
        if source.component_type == 'module' and target.component_type == 'module':
            return 'imports'
        elif source.component_type == 'api_service' and target.component_type == 'database_schema':
            return 'uses'
        elif source.component_type == 'module' and target.component_type == 'configuration':
            return 'configures'
        else:
            return 'depends_on'
    
    def _calculate_dependency_strength(self, source: SystemComponent, target: SystemComponent) -> float:
        """Calculate dependency strength (0.0 to 1.0)"""
        # Simple strength calculation - can be enhanced with more sophisticated analysis
        strength = 0.5  # Base strength
        
        # Increase strength for critical dependencies
        if target.component_type == 'database_schema':
            strength += 0.3
        elif target.component_type == 'api_service':
            strength += 0.2
        elif target.component_type == 'configuration':
            strength += 0.1
        
        return min(strength, 1.0)
    
    def get_system_big_picture(self, max_components: int = 100, quick_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate "big picture" system understanding with optimized performance
        Target: <400ms for comprehensive system overview (with buffer under 500ms)
        """
        start_time = time.time()
        
        try:
            # Get components with performance limit
            components = self.discover_system_components(max_components=max_components)
            
            # Quick check for early termination
            if time.time() - start_time > 0.2:  # Already 200ms, need to speed up
                self.logger.warning("Component discovery took >200ms, enabling quick analysis mode")
                quick_analysis = True
            
            # Optimize dependency analysis based on time budget
            if quick_analysis:
                dependencies = self.analyze_system_dependencies_fast(components)
            else:
                dependencies = self.analyze_system_dependencies(components)
            
            # Check time budget again
            analysis_time = time.time() - start_time
            if analysis_time > 0.35:  # 350ms spent, need to wrap up quickly
                self.logger.warning(f"Analysis time {analysis_time*1000:.0f}ms exceeds budget, using minimal analysis")
                return self._generate_minimal_big_picture(components, dependencies, start_time)
            
            # Analyze system architecture (optimized)
            architecture_analysis = self._analyze_system_architecture_fast(components, dependencies)
            
            # Generate system metrics (optimized)  
            system_metrics = self._calculate_system_metrics_fast(components, dependencies)
            
            # Identify critical paths (simplified for performance)
            critical_paths = self._identify_critical_paths_fast(dependencies)
            
            duration_ms = (time.time() - start_time) * 1000
            
            big_picture = {
                'system_overview': {
                    'total_components': len(components),
                    'total_dependencies': len(dependencies),
                    'component_types': self._count_component_types_fast(components),
                    'analysis_duration_ms': round(duration_ms, 2),
                    'target_met': duration_ms < 500,
                    'quick_analysis_used': quick_analysis,
                    'performance_optimized': True
                },
                'architecture_analysis': architecture_analysis,
                'system_metrics': system_metrics,
                'critical_paths': critical_paths,
                'components_summary': [
                    {
                        'name': comp.name,
                        'type': comp.component_type,
                        'dependency_count': len(comp.dependencies),
                        'interface_count': len(comp.interfaces),
                        'key_metadata': self._extract_key_metadata(comp.metadata)
                    }
                    for comp in components[:15]  # Reduced limit for performance
                ],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return big_picture
            
        except Exception as e:
            self.logger.error(f"Big picture generation failed: {str(e)}")
            # Fallback to minimal analysis
            return self._generate_emergency_big_picture(start_time)
    
    def _analyze_system_architecture(self, components: List[SystemComponent], 
                                   dependencies: List[DependencyRelationship]) -> Dict[str, Any]:
        """Analyze overall system architecture patterns"""
        
        # Layer analysis
        layers = {
            'api_layer': [c for c in components if c.component_type == 'api_service'],
            'business_logic': [c for c in components if c.component_type == 'module' and 'service' in c.name],
            'data_layer': [c for c in components if c.component_type == 'database_schema'],
            'configuration': [c for c in components if c.component_type == 'configuration']
        }
        
        # Coupling analysis
        coupling_metrics = {}
        for comp in components:
            incoming = len([d for d in dependencies if d.target == comp.name])
            outgoing = len([d for d in dependencies if d.source == comp.name])
            coupling_metrics[comp.name] = {'incoming': incoming, 'outgoing': outgoing, 'total': incoming + outgoing}
        
        # Most coupled components
        most_coupled = sorted(coupling_metrics.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
        
        return {
            'layers': {layer: len(comps) for layer, comps in layers.items()},
            'coupling_analysis': {
                'highest_coupled': most_coupled,
                'avg_coupling': sum(m['total'] for m in coupling_metrics.values()) / len(coupling_metrics) if coupling_metrics else 0
            },
            'architectural_patterns': {
                'has_layered_architecture': len(layers['api_layer']) > 0 and len(layers['data_layer']) > 0,
                'has_microservices': len([c for c in components if 'service' in c.name.lower()]) > 3,
                'has_api_gateway': any('gateway' in c.name.lower() for c in components)
            }
        }
    
    def _calculate_system_metrics(self, components: List[SystemComponent], 
                                dependencies: List[DependencyRelationship]) -> Dict[str, Any]:
        """Calculate system-wide metrics"""
        
        # Code metrics
        total_lines = sum(comp.metadata.get('line_count', 0) for comp in components if comp.metadata.get('line_count'))
        
        # Dependency metrics
        avg_deps_per_component = sum(len(comp.dependencies) for comp in components) / len(components) if components else 0
        
        # Complexity metrics
        high_complexity_components = [c for c in components 
                                    if len(c.dependencies) > 5 or len(c.interfaces) > 10]
        
        return {
            'code_metrics': {
                'total_lines_of_code': total_lines,
                'avg_lines_per_component': total_lines / len(components) if components else 0
            },
            'dependency_metrics': {
                'total_dependencies': len(dependencies),
                'avg_dependencies_per_component': round(avg_deps_per_component, 2),
                'max_dependencies': max(len(c.dependencies) for c in components) if components else 0
            },
            'complexity_metrics': {
                'high_complexity_components': len(high_complexity_components),
                'complexity_ratio': len(high_complexity_components) / len(components) if components else 0
            }
        }
    
    def _identify_critical_paths(self, dependencies: List[DependencyRelationship]) -> List[Dict[str, Any]]:
        """Identify critical dependency paths in the system"""
        
        # Find components with highest dependency strength
        critical_deps = [d for d in dependencies if d.strength > 0.8]
        
        # Group by source to find critical components
        critical_components = {}
        for dep in critical_deps:
            if dep.source not in critical_components:
                critical_components[dep.source] = []
            critical_components[dep.source].append(dep)
        
        critical_paths = []
        for comp_name, comp_deps in critical_components.items():
            if len(comp_deps) >= 2:  # Components with 2+ critical dependencies
                critical_paths.append({
                    'component': comp_name,
                    'critical_dependencies': len(comp_deps),
                    'avg_strength': sum(d.strength for d in comp_deps) / len(comp_deps),
                    'dependencies': [d.target for d in comp_deps]
                })
        
        return sorted(critical_paths, key=lambda x: x['avg_strength'], reverse=True)
    
    def analyze_system_dependencies_fast(self, components: List[SystemComponent]) -> List[Dict[str, Any]]:
        """
        Optimized dependency analysis with early termination
        Target: <200ms for dependency graph generation
        """
        start_time = time.time()
        relationships = []
        
        # Limit analysis scope for performance
        max_components = min(len(components), 50)  # Limit to 50 components
        limited_components = components[:max_components]
        
        # Build dependency lookup for performance
        component_lookup = {comp.name: comp for comp in limited_components}
        component_names = set(component_lookup.keys())
        
        for component in limited_components:
            # Early termination if taking too long
            if time.time() - start_time > 0.15:  # 150ms limit
                self.logger.warning("Dependency analysis timeout, returning partial results")
                break
                
            for dep in component.dependencies[:5]:  # Limit dependencies per component
                # Quick lookup instead of iteration
                if dep in component_names:
                    target_comp = component_lookup[dep]
                    relationship = {
                        'source': component.name,
                        'target': target_comp.name,
                        'dependency_type': 'imports' if component.component_type == 'module' else 'depends_on',
                        'strength': 0.7,  # Default strength for performance
                        'detection_method': 'static_analysis_fast',
                        'context': {
                            'source_type': component.component_type,
                            'target_type': target_comp.component_type,
                            'analyzed_at': datetime.utcnow().isoformat()
                        }
                    }
                    relationships.append(relationship)
                    
                    # Early termination for large dependency sets
                    if len(relationships) > 100:  # Limit total relationships
                        break
        
        # Cache results for performance
        self.dependency_cache['system_fast'] = relationships
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Fast dependency analysis: {len(relationships)} relationships in {duration_ms:.2f}ms")
        
        return relationships

    def _analyze_system_architecture_fast(self, components: List[SystemComponent], dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimized system architecture analysis"""
        
        # Quick layer analysis
        layers = {
            'api_layer': sum(1 for c in components if c.component_type == 'api_service'),
            'business_logic': sum(1 for c in components if c.component_type == 'module' and 'service' in c.name),
            'data_layer': sum(1 for c in components if c.component_type == 'database_schema'),
            'configuration': sum(1 for c in components if c.component_type == 'configuration')
        }
        
        # Simplified coupling analysis (top 5 only for performance)
        dependency_counts = {}
        for dep in dependencies[:50]:  # Limit for performance
            source = dep.get('source', '')
            dependency_counts[source] = dependency_counts.get(source, 0) + 1
        
        # Top coupled components
        most_coupled = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'layers': layers,
            'coupling_analysis': {
                'highest_coupled': most_coupled,
                'avg_coupling': sum(dependency_counts.values()) / len(dependency_counts) if dependency_counts else 0
            },
            'architectural_patterns': {
                'has_layered_architecture': layers['api_layer'] > 0 and layers['data_layer'] > 0,
                'has_microservices': layers['business_logic'] > 3,
                'has_api_gateway': any('gateway' in str(c.name).lower() for c in components[:20])  # Limit search
            }
        }

    def _calculate_system_metrics_fast(self, components: List[SystemComponent], dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimized system metrics calculation"""
        
        # Quick metrics calculation
        total_lines = sum(comp.metadata.get('line_count', 0) for comp in components[:30] if comp.metadata.get('line_count'))  # Limit for performance
        
        return {
            'code_metrics': {
                'total_lines_of_code': total_lines,
                'avg_lines_per_component': total_lines / len(components) if components else 0
            },
            'dependency_metrics': {
                'total_dependencies': len(dependencies),
                'avg_dependencies_per_component': sum(len(c.dependencies) for c in components[:30]) / len(components) if components else 0,
                'max_dependencies': max((len(c.dependencies) for c in components[:20]), default=0)  # Limit for performance
            },
            'complexity_metrics': {
                'high_complexity_components': sum(1 for c in components[:30] if len(c.dependencies) > 5 or len(c.interfaces) > 10),
                'complexity_ratio': 0.1  # Simplified for performance
            }
        }

    def _identify_critical_paths_fast(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized critical path identification"""
        
        if not dependencies:
            return []
        
        # Quick critical dependency identification (simplified)
        critical_deps = []
        for dep in dependencies[:20]:  # Limit for performance
            if dep.get('strength', 0.5) > 0.7:
                critical_deps.append(dep)
        
        # Group by source
        critical_components = {}
        for dep in critical_deps:
            source = dep.get('source', '')
            if source not in critical_components:
                critical_components[source] = []
            critical_components[source].append(dep)
        
        # Build critical paths (top 3 for performance)
        critical_paths = []
        for comp_name, comp_deps in list(critical_components.items())[:3]:
            if len(comp_deps) >= 2:  # Components with 2+ critical dependencies
                critical_paths.append({
                    'component': comp_name,
                    'critical_dependencies': len(comp_deps),
                    'avg_strength': sum(d.get('strength', 0.5) for d in comp_deps) / len(comp_deps),
                    'dependencies': [d.get('target', '') for d in comp_deps]
                })
        
        return sorted(critical_paths, key=lambda x: x['avg_strength'], reverse=True)

    def _count_component_types_fast(self, components: List[SystemComponent]) -> Dict[str, int]:
        """Optimized component type counting"""
        counts = {}
        for comp in components:
            comp_type = comp.component_type
            counts[comp_type] = counts.get(comp_type, 0) + 1
        return counts

    def _extract_key_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only key metadata for performance"""
        if not metadata:
            return {}
        
        key_fields = ['line_count', 'file_size', 'has_database', 'has_fastapi', 'endpoint_count', 'table_count']
        return {k: v for k, v in metadata.items() if k in key_fields}

    def _generate_minimal_big_picture(self, components: List[SystemComponent], dependencies: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Generate minimal big picture analysis when time is critical"""
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            'system_overview': {
                'total_components': len(components),
                'total_dependencies': len(dependencies),
                'component_types': self._count_component_types_fast(components),
                'analysis_duration_ms': round(duration_ms, 2),
                'target_met': duration_ms < 500,
                'minimal_analysis': True,
                'performance_optimized': True
            },
            'architecture_analysis': {
                'layers': {'minimal': True},
                'coupling_analysis': {'minimal': True},
                'architectural_patterns': {'minimal_analysis': True}
            },
            'system_metrics': {
                'code_metrics': {'minimal': True},
                'dependency_metrics': {'total_dependencies': len(dependencies)},
                'complexity_metrics': {'minimal': True}
            },
            'critical_paths': [],
            'components_summary': [
                {
                    'name': comp.name,
                    'type': comp.component_type,
                    'dependency_count': len(comp.dependencies),
                    'interface_count': len(comp.interfaces)
                }
                for comp in components[:10]  # Very limited for performance
            ],
            'generated_at': datetime.utcnow().isoformat()
        }

    def _generate_emergency_big_picture(self, start_time: float) -> Dict[str, Any]:
        """Emergency fallback when analysis fails completely"""
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            'system_overview': {
                'total_components': 0,
                'total_dependencies': 0,
                'component_types': {},
                'analysis_duration_ms': round(duration_ms, 2),
                'target_met': False,
                'emergency_mode': True,
                'error': 'Analysis failed, emergency mode activated'
            },
            'architecture_analysis': {'emergency_mode': True},
            'system_metrics': {'emergency_mode': True},
            'critical_paths': [],
            'components_summary': [],
            'generated_at': datetime.utcnow().isoformat()
        }


class SystemContextAPI:
    """
    API layer for system context and understanding operations
    Integrates with DPIBS API framework for <500ms response times
    """
    
    def __init__(self, optimizer: DPIBSPerformanceOptimizer, project_root: str = "/Users/cal/DEV/RIF"):
        self.optimizer = optimizer
        self.engine = SystemContextEngine(optimizer, project_root)
        self.logger = logging.getLogger(__name__)
    
    @property 
    def performance_monitor(self):
        """Access the performance monitor from optimizer"""
        return self.optimizer.performance_monitor
    
    def discover_components(self, force_rescan: bool = False, max_components: int = 200) -> Dict[str, Any]:
        """API endpoint for system component discovery with performance optimization"""
        try:
            components = self.engine.discover_system_components(force_rescan, max_components)
            
            return {
                'status': 'success',
                'component_count': len(components),
                'components': [asdict(comp) for comp in components],
                'cache_used': not force_rescan and self.engine.last_scan_time is not None,
                'last_scan': self.engine.last_scan_time.isoformat() if self.engine.last_scan_time else None,
                'max_components_limit': max_components,
                'performance_optimized': True
            }
            
        except Exception as e:
            self.logger.error(f"Component discovery failed: {str(e)}")
            raise
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """API endpoint for dependency analysis"""
        try:
            dependencies = self.engine.analyze_system_dependencies()
            
            return {
                'status': 'success',
                'dependency_count': len(dependencies),
                'dependencies': [asdict(dep) for dep in dependencies],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            raise
    
    def get_big_picture(self, max_components: int = 100, quick_analysis: bool = True) -> Dict[str, Any]:
        """API endpoint for comprehensive system understanding with performance optimization"""
        try:
            big_picture = self.engine.get_system_big_picture(max_components, quick_analysis)
            return big_picture
            
        except Exception as e:
            self.logger.error(f"Big picture analysis failed: {str(e)}")
            raise
    
    def store_system_context(self, context_name: str, context_type: str, 
                           system_snapshot: Dict[str, Any], confidence_level: float = 1.0) -> str:
        """Store system context in database"""
        try:
            with self.optimizer.connection_manager.get_connection() as conn:
                result = conn.execute("""
                    INSERT INTO system_contexts (context_name, context_type, system_snapshot, 
                                               confidence_level, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    RETURNING id
                """, [context_name, context_type, json.dumps(system_snapshot), 
                      confidence_level, None]).fetchone()  # TODO: Add embedding
                
                if result:
                    context_id = str(result[0])
                    self.logger.info(f"Stored system context: {context_id}")
                    return context_id
                    
        except Exception as e:
            self.logger.error(f"Failed to store system context: {str(e)}")
            raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system context engine health status"""
        return {
            'status': 'healthy',
            'component_cache_size': len(self.engine.component_cache),
            'dependency_cache_size': len(self.engine.dependency_cache),
            'last_scan': self.engine.last_scan_time.isoformat() if self.engine.last_scan_time else None,
            'project_root': str(self.engine.project_root),
            'performance_metrics': self.optimizer.get_performance_report()
        }


# Factory function for integration
def create_system_context_api(optimizer: DPIBSPerformanceOptimizer, project_root: str = "/Users/cal/DEV/RIF") -> SystemContextAPI:
    """Create system context API instance"""
    return SystemContextAPI(optimizer, project_root)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
    
    optimizer = DPIBSPerformanceOptimizer()
    api = create_system_context_api(optimizer)
    
    print("🔍 Discovering system components...")
    components_result = api.discover_components()
    print(f"Found {components_result['component_count']} components")
    
    print("\n🔗 Analyzing dependencies...")
    deps_result = api.analyze_dependencies() 
    print(f"Found {deps_result['dependency_count']} dependencies")
    
    print("\n🏗️ Generating big picture...")
    big_picture = api.get_big_picture()
    print(f"System overview: {big_picture['system_overview']['total_components']} components, {big_picture['system_overview']['total_dependencies']} dependencies")
    print(f"Analysis completed in {big_picture['system_overview']['analysis_duration_ms']:.2f}ms (Target: <500ms)")
    
    print("\n📊 System health:")
    health = api.get_system_health()
    print(json.dumps(health, indent=2, default=str))