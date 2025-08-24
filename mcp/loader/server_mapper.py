"""
Server Mapper

Maps detected requirements to specific MCP server configurations
based on capability matching and optimization priorities.

Issue: #82 - Implement dynamic MCP loader  
Component: Server mapping and optimization
"""

import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class ServerCapability:
    """Represents a capability that an MCP server provides"""
    name: str
    category: str  # core, analysis, integration, optimization
    priority: int  # 1=essential, 2=important, 3=optional
    resource_cost: int  # 1=low, 2=medium, 3=high


class ServerMapper:
    """
    Maps requirements to appropriate MCP server configurations
    with intelligent prioritization and resource optimization
    """
    
    def __init__(self, server_registry):
        """
        Initialize server mapper
        
        Args:
            server_registry: MCP server registry for capability lookup
        """
        self.server_registry = server_registry
        
        # Define requirement to server mappings
        self.requirement_mappings = {
            # Core development servers
            'github_mcp': {
                'server_id': 'github-mcp-v1.2.0',
                'name': 'GitHub MCP Server',
                'priority': 1,
                'capabilities': ['github_api', 'issue_management', 'pr_operations'],
                'resource_requirements': {'memory_mb': 128, 'cpu_percent': 8}
            },
            'git_mcp': {
                'server_id': 'git-mcp-v1.0.0', 
                'name': 'Git MCP Server',
                'priority': 1,
                'capabilities': ['git_operations', 'repository_analysis', 'branch_management'],
                'resource_requirements': {'memory_mb': 64, 'cpu_percent': 5}
            },
            'filesystem_mcp': {
                'server_id': 'filesystem-mcp-v1.1.0',
                'name': 'Filesystem MCP Server', 
                'priority': 1,
                'capabilities': ['file_operations', 'directory_scanning', 'file_watching'],
                'resource_requirements': {'memory_mb': 96, 'cpu_percent': 6}
            },
            
            # Language-specific servers
            'nodejs_tools': {
                'server_id': 'nodejs-tools-mcp-v1.0.0',
                'name': 'Node.js Tools MCP',
                'priority': 2,
                'capabilities': ['npm_operations', 'package_analysis', 'node_debugging'],
                'resource_requirements': {'memory_mb': 112, 'cpu_percent': 7}
            },
            'python_tools': {
                'server_id': 'python-tools-mcp-v1.0.0',
                'name': 'Python Tools MCP',
                'priority': 2, 
                'capabilities': ['pip_operations', 'virtual_env', 'python_debugging'],
                'resource_requirements': {'memory_mb': 104, 'cpu_percent': 6}
            },
            
            # Reasoning and memory servers
            'sequential_thinking': {
                'server_id': 'sequential-thinking-mcp-v1.0.0',
                'name': 'Sequential Thinking MCP',
                'priority': 2,
                'capabilities': ['structured_reasoning', 'problem_decomposition', 'logic_chains'],
                'resource_requirements': {'memory_mb': 256, 'cpu_percent': 12}
            },
            'memory_mcp': {
                'server_id': 'memory-mcp-v1.0.0',
                'name': 'Memory MCP Server',
                'priority': 2,
                'capabilities': ['context_storage', 'knowledge_retrieval', 'pattern_memory'],
                'resource_requirements': {'memory_mb': 192, 'cpu_percent': 10}
            },
            
            # Integration servers
            'database_mcp': {
                'server_id': 'database-mcp-v1.0.0', 
                'name': 'Database MCP Server',
                'priority': 2,
                'capabilities': ['sql_operations', 'schema_analysis', 'query_optimization'],
                'resource_requirements': {'memory_mb': 128, 'cpu_percent': 8}
            },
            'docker_mcp': {
                'server_id': 'docker-mcp-v1.0.0',
                'name': 'Docker MCP Server',
                'priority': 2,
                'capabilities': ['container_management', 'image_operations', 'docker_compose'],
                'resource_requirements': {'memory_mb': 144, 'cpu_percent': 9}
            },
            
            # Cloud servers
            'aws_mcp': {
                'server_id': 'aws-mcp-v1.0.0',
                'name': 'AWS MCP Server', 
                'priority': 3,
                'capabilities': ['ec2_operations', 's3_operations', 'lambda_functions'],
                'resource_requirements': {'memory_mb': 200, 'cpu_percent': 12}
            },
            'azure_mcp': {
                'server_id': 'azure-mcp-v1.0.0',
                'name': 'Azure MCP Server',
                'priority': 3,
                'capabilities': ['vm_operations', 'storage_operations', 'functions'],
                'resource_requirements': {'memory_mb': 200, 'cpu_percent': 12}
            },
            
            # Agent-specific servers
            'analysis_tools': {
                'server_id': 'analysis-tools-mcp-v1.0.0',
                'name': 'Analysis Tools MCP',
                'priority': 2,
                'capabilities': ['code_analysis', 'pattern_detection', 'complexity_metrics'],
                'resource_requirements': {'memory_mb': 160, 'cpu_percent': 10}
            },
            'code_generation': {
                'server_id': 'code-generation-mcp-v1.0.0',
                'name': 'Code Generation MCP',
                'priority': 2,
                'capabilities': ['template_generation', 'boilerplate_creation', 'refactoring'],
                'resource_requirements': {'memory_mb': 144, 'cpu_percent': 9}
            },
            'testing_tools': {
                'server_id': 'testing-tools-mcp-v1.0.0',
                'name': 'Testing Tools MCP',
                'priority': 2,
                'capabilities': ['test_generation', 'coverage_analysis', 'mock_creation'],
                'resource_requirements': {'memory_mb': 128, 'cpu_percent': 8}
            },
            'quality_gates': {
                'server_id': 'quality-gates-mcp-v1.0.0',
                'name': 'Quality Gates MCP',
                'priority': 2,
                'capabilities': ['code_quality', 'security_scanning', 'standards_checking'],
                'resource_requirements': {'memory_mb': 176, 'cpu_percent': 11}
            },
            'security_scanner': {
                'server_id': 'security-scanner-mcp-v1.0.0',
                'name': 'Security Scanner MCP',
                'priority': 2,
                'capabilities': ['vulnerability_scanning', 'dependency_audit', 'security_reports'],
                'resource_requirements': {'memory_mb': 192, 'cpu_percent': 10}
            },
            'pattern_recognition': {
                'server_id': 'pattern-recognition-mcp-v1.0.0',
                'name': 'Pattern Recognition MCP',
                'priority': 3,
                'capabilities': ['pattern_matching', 'similarity_analysis', 'trend_detection'],
                'resource_requirements': {'memory_mb': 224, 'cpu_percent': 14}
            },
            
            # Performance servers
            'performance_monitoring': {
                'server_id': 'performance-monitoring-mcp-v1.0.0',
                'name': 'Performance Monitoring MCP',
                'priority': 3,
                'capabilities': ['performance_metrics', 'bottleneck_detection', 'optimization_suggestions'],
                'resource_requirements': {'memory_mb': 160, 'cpu_percent': 10}
            },
            'resource_optimization': {
                'server_id': 'resource-optimization-mcp-v1.0.0',
                'name': 'Resource Optimization MCP',
                'priority': 3,
                'capabilities': ['resource_analysis', 'optimization_recommendations', 'efficiency_metrics'],
                'resource_requirements': {'memory_mb': 144, 'cpu_percent': 9}
            }
        }
        
        # Define server categories for optimization
        self.server_categories = {
            'essential': ['github_mcp', 'git_mcp', 'filesystem_mcp'],
            'development': ['nodejs_tools', 'python_tools', 'docker_mcp'],
            'intelligence': ['sequential_thinking', 'memory_mcp', 'analysis_tools'],
            'integration': ['database_mcp', 'aws_mcp', 'azure_mcp'],
            'quality': ['testing_tools', 'quality_gates', 'security_scanner'],
            'optimization': ['performance_monitoring', 'resource_optimization', 'pattern_recognition']
        }
    
    async def map_requirements_to_servers(self, requirements: Set[str], 
                                        project_context: Any) -> List[Dict[str, Any]]:
        """
        Map requirements to specific server configurations with optimization
        
        Args:
            requirements: Set of requirement identifiers
            project_context: Project context for optimization decisions
            
        Returns:
            List of optimized server configurations
        """
        logger.info(f"Mapping {len(requirements)} requirements to servers")
        
        # Step 1: Direct mapping
        candidate_servers = []
        for requirement in requirements:
            if requirement in self.requirement_mappings:
                server_config = self.requirement_mappings[requirement].copy()
                server_config['requirement'] = requirement
                candidate_servers.append(server_config)
                logger.debug(f"Mapped {requirement} to {server_config['name']}")
            else:
                logger.warning(f"No server mapping found for requirement: {requirement}")
        
        # Step 2: Apply optimization strategies
        optimized_servers = await self._optimize_server_selection(
            candidate_servers, project_context
        )
        
        # Step 3: Add server metadata and dependencies
        final_servers = []
        for server in optimized_servers:
            enhanced_server = await self._enhance_server_config(server, project_context)
            final_servers.append(enhanced_server)
        
        logger.info(f"Selected {len(final_servers)} optimized servers for loading")
        return final_servers
    
    async def _optimize_server_selection(self, candidate_servers: List[Dict[str, Any]],
                                       project_context: Any) -> List[Dict[str, Any]]:
        """
        Optimize server selection based on priority, resources, and context
        
        Args:
            candidate_servers: List of candidate server configurations
            project_context: Project context for optimization
            
        Returns:
            Optimized list of servers
        """
        # Sort by priority (1=highest, 3=lowest)
        candidate_servers.sort(key=lambda s: s['priority'])
        
        # Apply resource-based filtering
        resource_budget_mb = getattr(project_context, 'performance_requirements', {}).get(
            'max_memory_mb', 512
        ) if hasattr(project_context, 'performance_requirements') and project_context.performance_requirements else 512
        
        optimized = []
        total_memory = 0
        
        # Essential servers (priority 1) - always include if possible
        essential_servers = [s for s in candidate_servers if s['priority'] == 1]
        for server in essential_servers:
            memory_req = server['resource_requirements']['memory_mb']
            if total_memory + memory_req <= resource_budget_mb:
                optimized.append(server)
                total_memory += memory_req
                logger.debug(f"Added essential server: {server['name']} ({memory_req}MB)")
        
        # Important servers (priority 2) - add based on complexity and available resources
        important_servers = [s for s in candidate_servers if s['priority'] == 2]
        complexity = getattr(project_context, 'complexity', 'medium')
        
        max_important = {'low': 2, 'medium': 4, 'high': 6, 'very-high': 8}.get(complexity, 4)
        important_added = 0
        
        for server in important_servers:
            if important_added >= max_important:
                break
                
            memory_req = server['resource_requirements']['memory_mb'] 
            if total_memory + memory_req <= resource_budget_mb:
                optimized.append(server)
                total_memory += memory_req
                important_added += 1
                logger.debug(f"Added important server: {server['name']} ({memory_req}MB)")
        
        # Optional servers (priority 3) - add only for high complexity with available resources
        if complexity in ['high', 'very-high']:
            optional_servers = [s for s in candidate_servers if s['priority'] == 3]
            max_optional = {'high': 2, 'very-high': 4}.get(complexity, 0)
            optional_added = 0
            
            for server in optional_servers:
                if optional_added >= max_optional:
                    break
                    
                memory_req = server['resource_requirements']['memory_mb']
                if total_memory + memory_req <= resource_budget_mb * 0.9:  # Leave 10% buffer
                    optimized.append(server)
                    total_memory += memory_req
                    optional_added += 1
                    logger.debug(f"Added optional server: {server['name']} ({memory_req}MB)")
        
        logger.info(f"Optimized selection: {len(optimized)} servers, {total_memory}MB total")
        return optimized
    
    async def _enhance_server_config(self, server_config: Dict[str, Any], 
                                   project_context: Any) -> Dict[str, Any]:
        """
        Enhance server configuration with metadata and dependencies
        
        Args:
            server_config: Base server configuration
            project_context: Project context
            
        Returns:
            Enhanced server configuration
        """
        enhanced = server_config.copy()
        
        # Add metadata
        enhanced.update({
            'type': 'mcp_server',
            'version': enhanced.get('server_id', '').split('-v')[-1] if '-v' in enhanced.get('server_id', '') else '1.0.0',
            'security_level': self._determine_security_level(server_config),
            'health_check': {
                'endpoint': '/health',
                'interval_seconds': 30,
                'timeout_seconds': 10
            },
            'integration_patterns': ['agent_context', 'tool_execution'],
            'configuration': await self._generate_server_configuration(server_config, project_context)
        })
        
        # Add dependencies
        enhanced['dependencies'] = self._determine_dependencies(server_config)
        
        # Add environment-specific settings
        if hasattr(project_context, 'project_path'):
            enhanced['project_context'] = {
                'project_path': project_context.project_path,
                'complexity': getattr(project_context, 'complexity', 'medium')
            }
        
        return enhanced
    
    def _determine_security_level(self, server_config: Dict[str, Any]) -> str:
        """
        Determine security level required for server
        
        Args:
            server_config: Server configuration
            
        Returns:
            Security level: low, medium, high, very-high
        """
        server_id = server_config.get('server_id', '')
        capabilities = server_config.get('capabilities', [])
        
        # High security for cloud and database servers
        if any(cloud in server_id for cloud in ['aws', 'azure', 'gcp']):
            return 'very-high'
        
        if 'database' in server_id or any('database' in cap for cap in capabilities):
            return 'high'
        
        # Medium security for integration servers
        if any(integration in server_id for integration in ['github', 'git', 'docker']):
            return 'medium'
        
        # Low security for local tools
        return 'low'
    
    async def _generate_server_configuration(self, server_config: Dict[str, Any],
                                          project_context: Any) -> Dict[str, Any]:
        """
        Generate server-specific configuration
        
        Args:
            server_config: Server configuration
            project_context: Project context
            
        Returns:
            Server configuration parameters
        """
        config = {}
        server_id = server_config.get('server_id', '')
        
        # GitHub server configuration
        if 'github' in server_id:
            config.update({
                'required': ['github_token'],
                'optional': ['rate_limit_override', 'webhook_secret'],
                'settings': {
                    'api_version': 'v3',
                    'timeout_seconds': 30,
                    'retry_attempts': 3
                }
            })
        
        # Git server configuration
        elif 'git' in server_id:
            config.update({
                'required': [],
                'optional': ['ssh_key_path', 'git_user_name', 'git_user_email'],
                'settings': {
                    'default_branch': 'main',
                    'auto_fetch': True,
                    'timeout_seconds': 60
                }
            })
        
        # Database server configuration
        elif 'database' in server_id:
            config.update({
                'required': ['database_url'],
                'optional': ['connection_pool_size', 'query_timeout'],
                'settings': {
                    'pool_size': 5,
                    'timeout_seconds': 30,
                    'auto_reconnect': True
                }
            })
        
        # Cloud server configuration
        elif any(cloud in server_id for cloud in ['aws', 'azure', 'gcp']):
            config.update({
                'required': ['cloud_credentials'],
                'optional': ['region', 'resource_tags'],
                'settings': {
                    'timeout_seconds': 60,
                    'retry_policy': 'exponential_backoff',
                    'security_mode': 'strict'
                }
            })
        
        # Default configuration
        else:
            config.update({
                'required': [],
                'optional': [],
                'settings': {
                    'timeout_seconds': 30,
                    'retry_attempts': 2
                }
            })
        
        return config
    
    def _determine_dependencies(self, server_config: Dict[str, Any]) -> List[str]:
        """
        Determine dependencies for server
        
        Args:
            server_config: Server configuration
            
        Returns:
            List of dependency identifiers
        """
        dependencies = []
        server_id = server_config.get('server_id', '')
        
        # Common dependencies
        if 'git' in server_id:
            dependencies.append('git')
        
        if 'github' in server_id:
            dependencies.extend(['git', 'gh_cli'])
        
        if 'docker' in server_id:
            dependencies.append('docker')
        
        if 'nodejs' in server_id:
            dependencies.append('node')
        
        if 'python' in server_id:
            dependencies.append('python')
        
        return dependencies
    
    async def get_server_capabilities(self, server_id: str) -> List[str]:
        """
        Get capabilities for a specific server
        
        Args:
            server_id: Server identifier
            
        Returns:
            List of server capabilities
        """
        for req, config in self.requirement_mappings.items():
            if config['server_id'] == server_id:
                return config.get('capabilities', [])
        
        return []
    
    async def get_servers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get servers in a specific category
        
        Args:
            category: Server category
            
        Returns:
            List of servers in category
        """
        if category not in self.server_categories:
            return []
        
        servers = []
        for requirement in self.server_categories[category]:
            if requirement in self.requirement_mappings:
                servers.append(self.requirement_mappings[requirement])
        
        return servers
    
    async def estimate_resource_usage(self, server_configs: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Estimate total resource usage for server configurations
        
        Args:
            server_configs: List of server configurations
            
        Returns:
            Dictionary with estimated resource usage
        """
        total_memory = sum(
            config.get('resource_requirements', {}).get('memory_mb', 64) 
            for config in server_configs
        )
        
        total_cpu = sum(
            config.get('resource_requirements', {}).get('cpu_percent', 5)
            for config in server_configs
        )
        
        return {
            'memory_mb': total_memory,
            'cpu_percent': min(total_cpu, 100),  # Cap at 100%
            'server_count': len(server_configs)
        }