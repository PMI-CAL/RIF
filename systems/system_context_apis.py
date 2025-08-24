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
    
    def discover_system_components(self, force_rescan: bool = False) -> List[SystemComponent]:
        """
        Discover all system components with intelligent caching
        Target: Complete system scan in <2 seconds, cached retrieval in <100ms
        """
        start_time = time.time()
        
        # Check if we need to rescan (cache invalidation)
        if not force_rescan and self.last_scan_time:
            age_minutes = (datetime.utcnow() - self.last_scan_time).total_seconds() / 60
            if age_minutes < 30 and self.component_cache:  # 30-minute cache TTL
                self.logger.debug("Returning cached system components")
                return list(self.component_cache.values())
        
        self.logger.info("Discovering system components...")
        components = []
        
        try:
            # Scan Python modules
            python_components = self._scan_python_modules()
            components.extend(python_components)
            
            # Scan configuration files
            config_components = self._scan_configuration_files()
            components.extend(config_components)
            
            # Scan database schemas
            db_components = self._scan_database_schemas()
            components.extend(db_components)
            
            # Scan API definitions
            api_components = self._scan_api_definitions()
            components.extend(api_components)
            
            # Update cache
            self.component_cache = {comp.name: comp for comp in components}
            self.last_scan_time = datetime.utcnow()
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Discovered {len(components)} system components in {duration_ms:.2f}ms")
            
            return components
            
        except Exception as e:
            self.logger.error(f"System component discovery failed: {str(e)}")
            return list(self.component_cache.values())  # Return cached data on error
    
    def _scan_python_modules(self) -> List[SystemComponent]:
        """Scan Python modules and extract component information"""
        components = []
        
        python_files = list(self.project_root.rglob("*.py"))
        self.logger.debug(f"Scanning {len(python_files)} Python files")
        
        for py_file in python_files:
            try:
                # Skip hidden files and __pycache__
                if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                    continue
                
                relative_path = py_file.relative_to(self.project_root)
                module_name = str(relative_path).replace('/', '.').replace('.py', '')
                
                # Read file to analyze imports and exports
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                dependencies = self._extract_python_dependencies(content)
                interfaces = self._extract_python_interfaces(content)
                
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
                        'last_modified': py_file.stat().st_mtime
                    }
                )
                
                components.append(component)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {str(e)}")
        
        return components
    
    def _extract_python_dependencies(self, content: str) -> List[str]:
        """Extract Python import dependencies"""
        dependencies = []
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # Standard imports
            if line.startswith('import '):
                dep = line.replace('import ', '').split('.')[0].split(' as ')[0]
                dependencies.append(dep)
            
            # From imports
            elif line.startswith('from ') and ' import ' in line:
                dep = line.split('from ')[1].split(' import ')[0].split('.')[0]
                dependencies.append(dep)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_python_interfaces(self, content: str) -> List[str]:
        """Extract public interfaces (functions, classes, APIs)"""
        interfaces = []
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # Public functions
            if line.startswith('def ') and not line.startswith('def _'):
                func_name = line.split('def ')[1].split('(')[0]
                interfaces.append(f"function:{func_name}")
            
            # Classes
            elif line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0]
                interfaces.append(f"class:{class_name}")
            
            # FastAPI endpoints
            elif '@app.' in line:
                endpoint = line.split('@app.')[1].split('(')[0]
                interfaces.append(f"api:{endpoint}")
        
        return interfaces
    
    def _scan_configuration_files(self) -> List[SystemComponent]:
        """Scan configuration files (YAML, JSON, etc.)"""
        components = []
        
        config_patterns = ["*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg"]
        
        for pattern in config_patterns:
            config_files = list(self.project_root.rglob(pattern))
            
            for config_file in config_files:
                try:
                    # Skip hidden files and node_modules
                    if any(part.startswith('.') or part == 'node_modules' for part in config_file.parts):
                        continue
                    
                    relative_path = config_file.relative_to(self.project_root)
                    
                    # Read and analyze configuration
                    content = config_file.read_text(encoding='utf-8', errors='ignore')
                    
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
    
    def _scan_database_schemas(self) -> List[SystemComponent]:
        """Scan database schema files"""
        components = []
        
        schema_files = list(self.project_root.rglob("*.sql"))
        schema_files.extend(list(self.project_root.rglob("*schema*")))
        
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
    
    def _scan_api_definitions(self) -> List[SystemComponent]:
        """Scan for API definitions and endpoints"""
        components = []
        
        # Look for FastAPI apps, Flask apps, etc.
        api_files = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if any(framework in content for framework in ['FastAPI', 'Flask', '@app.', 'APIRouter']):
                    api_files.append(py_file)
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
    
    def get_system_big_picture(self) -> Dict[str, Any]:
        """
        Generate "big picture" system understanding
        Target: <500ms for comprehensive system overview
        """
        start_time = time.time()
        
        # Get components and dependencies
        components = self.discover_system_components()
        dependencies = self.analyze_system_dependencies(components)
        
        # Analyze system architecture
        architecture_analysis = self._analyze_system_architecture(components, dependencies)
        
        # Generate system metrics
        system_metrics = self._calculate_system_metrics(components, dependencies)
        
        # Identify critical paths and bottlenecks
        critical_paths = self._identify_critical_paths(dependencies)
        
        duration_ms = (time.time() - start_time) * 1000
        
        big_picture = {
            'system_overview': {
                'total_components': len(components),
                'total_dependencies': len(dependencies),
                'component_types': {comp_type: len([c for c in components if c.component_type == comp_type])
                                  for comp_type in set(c.component_type for c in components)},
                'analysis_duration_ms': round(duration_ms, 2),
                'target_met': duration_ms < 500
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
                    'metadata': comp.metadata
                }
                for comp in components[:20]  # Limit for performance
            ],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return big_picture
    
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
    
    @performance_monitor("system_component_discovery", cache_ttl=30)
    def discover_components(self, force_rescan: bool = False) -> Dict[str, Any]:
        """API endpoint for system component discovery"""
        try:
            components = self.engine.discover_system_components(force_rescan)
            
            return {
                'status': 'success',
                'component_count': len(components),
                'components': [asdict(comp) for comp in components],
                'cache_used': not force_rescan and self.engine.last_scan_time is not None,
                'last_scan': self.engine.last_scan_time.isoformat() if self.engine.last_scan_time else None
            }
            
        except Exception as e:
            self.logger.error(f"Component discovery failed: {str(e)}")
            raise
    
    @performance_monitor("system_dependency_analysis", cache_ttl=20)
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
    
    @performance_monitor("system_big_picture", cache_ttl=15)
    def get_big_picture(self) -> Dict[str, Any]:
        """API endpoint for comprehensive system understanding"""
        try:
            big_picture = self.engine.get_system_big_picture()
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