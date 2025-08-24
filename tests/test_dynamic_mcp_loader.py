"""
Tests for Dynamic MCP Loader

Comprehensive test suite for the dynamic MCP loader implementation.

Issue: #82 - Implement dynamic MCP loader  
Component: Test suite
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Add the mcp module to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.loader.dynamic_loader import DynamicMCPLoader, ProjectContext, ServerLoadResult
from mcp.loader.requirement_detector import RequirementDetector
from mcp.loader.server_mapper import ServerMapper
from mcp.registry.server_registry import MCPServerRegistry


class TestDynamicMCPLoader:
    """Test suite for DynamicMCPLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing"""
        return DynamicMCPLoader(max_concurrent_loads=2, resource_budget_mb=256)
    
    @pytest.fixture
    def sample_project_context(self):
        """Create a sample project context"""
        return ProjectContext(
            project_path="/tmp/test_project",
            technology_stack={'python', 'git', 'github'},
            integrations={'github', 'database'},
            needs_reasoning=True,
            needs_memory=False,
            needs_database=True,
            needs_cloud=False,
            complexity='medium',
            agent_type='rif-implementer'
        )
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory with sample files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            
            # Create sample files
            (project_path / "requirements.txt").write_text("flask==2.0.1\npytest==6.2.4")
            (project_path / "main.py").write_text("print('Hello World')")
            (project_path / ".git").mkdir()
            (project_path / ".github").mkdir()
            (project_path / ".github/workflows").mkdir()
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_detect_requirements_basic(self, loader, sample_project_context):
        """Test basic requirement detection"""
        requirements = await loader.detect_requirements(sample_project_context)
        
        assert 'github_mcp' in requirements
        assert 'git_mcp' in requirements
        assert 'sequential_thinking' in requirements  # needs_reasoning=True
        assert 'database_mcp' in requirements  # needs_database=True
        assert 'code_generation' in requirements  # agent_type=rif-implementer
    
    @pytest.mark.asyncio
    async def test_map_requirements_to_servers(self, loader, sample_project_context):
        """Test requirement to server mapping"""
        requirements = {'github_mcp', 'git_mcp', 'python_tools'}
        servers = await loader.map_requirements_to_servers(requirements, sample_project_context)
        
        assert len(servers) > 0
        server_ids = [s['server_id'] for s in servers]
        assert any('github' in sid for sid in server_ids)
        assert any('git' in sid for sid in server_ids)
    
    @pytest.mark.asyncio
    async def test_validate_resources_success(self, loader):
        """Test resource validation success case"""
        server_config = {
            'server_id': 'test-server',
            'version': '1.0.0',
            'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5},
            'dependencies': []
        }
        
        is_valid = await loader.validate_resources(server_config)
        assert is_valid is True
    
    @pytest.mark.asyncio 
    async def test_validate_resources_insufficient_memory(self, loader):
        """Test resource validation with insufficient memory"""
        server_config = {
            'server_id': 'memory-heavy-server',
            'version': '1.0.0',
            'resource_requirements': {'memory_mb': 600, 'cpu_percent': 5},  # Exceeds 512MB budget
            'dependencies': []
        }
        
        is_valid = await loader.validate_resources(server_config)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_load_server_success(self, loader):
        """Test successful server loading"""
        server_config = {
            'server_id': 'test-server-v1.0.0',
            'name': 'Test Server',
            'version': '1.0.0',
            'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5},
            'dependencies': [],
            'security_level': 'low',
            'configuration': {'required': [], 'optional': []}
        }
        
        result = await loader.load_server(server_config)
        
        assert isinstance(result, ServerLoadResult)
        assert result.status == 'success'
        assert result.server_id == 'test-server-v1.0.0'
        assert result.load_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_load_servers_for_project(self, loader, temp_project_dir):
        """Test loading servers for a complete project"""
        project_context = ProjectContext(
            project_path=temp_project_dir,
            technology_stack={'python'},
            integrations={'github'},
            needs_reasoning=False,
            needs_memory=False,
            needs_database=False,
            needs_cloud=False,
            complexity='low'
        )
        
        results = await loader.load_servers_for_project(project_context)
        
        assert len(results) > 0
        assert all(isinstance(r, ServerLoadResult) for r in results)
        
        # At least one should be successful
        successful_results = [r for r in results if r.status == 'success']
        assert len(successful_results) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_server_loading(self, loader):
        """Test parallel loading of multiple servers"""
        server_configs = [
            {
                'server_id': f'test-server-{i}',
                'name': f'Test Server {i}',
                'version': f'1.{i}.0',
                'resource_requirements': {'memory_mb': 32, 'cpu_percent': 2},
                'dependencies': [],
                'security_level': 'low',
                'configuration': {'required': [], 'optional': []}
            }
            for i in range(4)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await loader._load_servers_parallel(server_configs)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 4
        
        # Parallel loading should be faster than sequential
        # With 2 concurrent loads, 4 servers should take roughly 2 load cycles
        assert end_time - start_time < 1.0  # Should complete quickly with mocks
    
    @pytest.mark.asyncio
    async def test_unload_server(self, loader):
        """Test server unloading"""
        # First load a server
        server_config = {
            'server_id': 'test-server-unload',
            'name': 'Test Server Unload',
            'version': '1.0.0',
            'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5},
            'dependencies': [],
            'security_level': 'low',
            'configuration': {'required': [], 'optional': []}
        }
        
        result = await loader.load_server(server_config)
        assert result.status == 'success'
        assert 'test-server-unload' in loader.active_servers
        
        # Now unload it
        unload_success = await loader.unload_server('test-server-unload')
        assert unload_success is True
        assert 'test-server-unload' not in loader.active_servers
    
    @pytest.mark.asyncio
    async def test_get_active_servers(self, loader):
        """Test getting active server information"""
        # Load a test server
        server_config = {
            'server_id': 'test-active-server',
            'name': 'Test Active Server',
            'version': '1.0.0', 
            'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5},
            'dependencies': [],
            'security_level': 'low',
            'configuration': {'required': [], 'optional': []}
        }
        
        await loader.load_server(server_config)
        
        active_servers = await loader.get_active_servers()
        
        assert 'test-active-server' in active_servers
        assert active_servers['test-active-server']['name'] == 'Test Active Server'
        assert active_servers['test-active-server']['health'] in ['healthy', 'degraded']  # Mock can return either
    
    @pytest.mark.asyncio
    async def test_get_loading_metrics(self, loader, sample_project_context):
        """Test loading metrics collection"""
        # Initially no metrics
        metrics = await loader.get_loading_metrics()
        assert 'message' in metrics or metrics.get('total_load_attempts', 0) == 0
        
        # Load servers for a project to generate metrics
        results = await loader.load_servers_for_project(sample_project_context)
        
        metrics = await loader.get_loading_metrics()
        
        assert metrics['total_load_attempts'] >= 1
        assert metrics['successful_loads'] >= 0
        assert 'success_rate' in metrics
        assert 'average_load_time_ms' in metrics
        assert 'active_servers' in metrics
        assert 'total_resource_usage_mb' in metrics


class TestRequirementDetector:
    """Test suite for RequirementDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return RequirementDetector()
    
    @pytest.fixture
    def nodejs_project_dir(self):
        """Create a temporary Node.js project directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            
            # Create Node.js project files
            package_json = {
                "name": "test-project",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^17.0.0",
                    "express": "^4.17.0"
                },
                "devDependencies": {
                    "typescript": "^4.5.0"
                }
            }
            (project_path / "package.json").write_text(json.dumps(package_json))
            (project_path / "src").mkdir()
            (project_path / "src/index.js").write_text("console.log('Hello');")
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_detect_technology_stack_nodejs(self, detector, nodejs_project_dir):
        """Test technology stack detection for Node.js project"""
        technologies = await detector.detect_technology_stack(nodejs_project_dir)
        
        assert 'nodejs' in technologies
        assert 'javascript' in technologies
        assert 'react' in technologies
        assert 'express' in technologies
        assert 'typescript' in technologies
    
    @pytest.mark.asyncio
    async def test_detect_integration_needs(self, detector, nodejs_project_dir):
        """Test integration needs detection"""
        # Add environment file with database config
        env_content = "DATABASE_URL=postgres://localhost/test\nAWS_ACCESS_KEY_ID=test"
        (Path(nodejs_project_dir) / '.env').write_text(env_content)
        
        integrations = await detector.detect_integration_needs(nodejs_project_dir)
        
        assert 'database' in integrations
        assert 'cloud' in integrations
    
    @pytest.mark.asyncio
    async def test_assess_complexity_high(self, detector, nodejs_project_dir):
        """Test complexity assessment for high complexity project"""
        project_path = Path(nodejs_project_dir)
        
        # Add complexity indicators
        (project_path / 'microservices').mkdir()
        (project_path / 'kubernetes').mkdir()
        (project_path / 'README.md').write_text("This is a distributed microservices architecture")
        
        # Add many files to increase complexity
        for i in range(50):
            (project_path / f'file_{i}.js').write_text(f"// File {i}")
        
        technologies = {'nodejs', 'javascript', 'react', 'express', 'docker', 'kubernetes'}
        complexity = await detector.assess_complexity(nodejs_project_dir, technologies)
        
        assert complexity in ['high', 'very-high']
    
    @pytest.mark.asyncio
    async def test_assess_complexity_low(self, detector):
        """Test complexity assessment for low complexity project"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir)
            (project_path / 'main.py').write_text("print('Hello')")
            
            technologies = {'python'}
            complexity = await detector.assess_complexity(tmp_dir, technologies)
            
            assert complexity in ['low', 'medium']


class TestServerMapper:
    """Test suite for ServerMapper"""
    
    @pytest.fixture
    def server_registry(self):
        """Create a server registry for testing"""
        return MCPServerRegistry()
    
    @pytest.fixture
    def mapper(self, server_registry):
        """Create a server mapper instance for testing"""
        return ServerMapper(server_registry)
    
    @pytest.fixture
    def simple_project_context(self):
        """Create a simple project context"""
        return ProjectContext(
            project_path="/tmp/simple_project",
            technology_stack={'python'},
            integrations=set(),
            needs_reasoning=False,
            needs_memory=False,
            needs_database=False,
            needs_cloud=False,
            complexity='low'
        )
    
    @pytest.mark.asyncio
    async def test_map_requirements_to_servers_basic(self, mapper, simple_project_context):
        """Test basic requirement mapping"""
        requirements = {'github_mcp', 'git_mcp'}
        servers = await mapper.map_requirements_to_servers(requirements, simple_project_context)
        
        assert len(servers) == 2
        server_ids = [s['server_id'] for s in servers]
        assert any('github' in sid for sid in server_ids)
        assert any('git' in sid for sid in server_ids)
    
    @pytest.mark.asyncio
    async def test_optimization_by_priority(self, mapper):
        """Test server selection optimization by priority"""
        project_context = ProjectContext(
            project_path="/tmp/test",
            technology_stack=set(),
            integrations=set(),
            needs_reasoning=False,
            needs_memory=False,
            needs_database=False,
            needs_cloud=False,
            complexity='low'
        )
        
        # Mix of different priority servers
        requirements = {'github_mcp', 'pattern_recognition', 'performance_monitoring'}
        servers = await mapper.map_requirements_to_servers(requirements, project_context)
        
        # Should prioritize essential servers (priority 1) over optional (priority 3)
        essential_servers = [s for s in servers if s['priority'] == 1]
        optional_servers = [s for s in servers if s['priority'] == 3]
        
        # For low complexity, should include essential but may limit optional
        assert len(essential_servers) > 0
        # Optional servers may be limited based on complexity
    
    @pytest.mark.asyncio
    async def test_resource_budget_constraints(self, mapper):
        """Test resource budget constraints in server selection"""
        project_context = ProjectContext(
            project_path="/tmp/test",
            technology_stack=set(),
            integrations=set(),
            needs_reasoning=False,
            needs_memory=False,
            needs_database=False,
            needs_cloud=False,
            complexity='medium',
            performance_requirements={'max_memory_mb': 128}  # Limited budget
        )
        
        # Request many servers that would exceed budget
        requirements = {
            'github_mcp', 'git_mcp', 'sequential_thinking', 'memory_mcp',
            'aws_mcp', 'pattern_recognition', 'performance_monitoring'
        }
        
        servers = await mapper.map_requirements_to_servers(requirements, project_context)
        
        # Calculate total memory usage
        total_memory = sum(s['resource_requirements']['memory_mb'] for s in servers)
        
        # Should respect the 128MB budget
        assert total_memory <= 128
    
    @pytest.mark.asyncio
    async def test_get_server_capabilities(self, mapper):
        """Test getting capabilities for a specific server"""
        capabilities = await mapper.get_server_capabilities('github-mcp-v1.2.0')
        
        assert 'github_api' in capabilities
        assert 'issue_management' in capabilities
        assert 'pr_operations' in capabilities
    
    @pytest.mark.asyncio
    async def test_get_servers_by_category(self, mapper):
        """Test getting servers by category"""
        essential_servers = await mapper.get_servers_by_category('essential')
        
        assert len(essential_servers) > 0
        # Should include core servers like github, git, filesystem
        server_names = [s['server_id'] for s in essential_servers]
        assert any('github' in name for name in server_names)
    
    @pytest.mark.asyncio
    async def test_estimate_resource_usage(self, mapper):
        """Test resource usage estimation"""
        server_configs = [
            {'resource_requirements': {'memory_mb': 128, 'cpu_percent': 8}},
            {'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5}},
            {'resource_requirements': {'memory_mb': 96, 'cpu_percent': 6}}
        ]
        
        usage = await mapper.estimate_resource_usage(server_configs)
        
        assert usage['memory_mb'] == 288  # 128 + 64 + 96
        assert usage['cpu_percent'] == 19  # 8 + 5 + 6
        assert usage['server_count'] == 3


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])