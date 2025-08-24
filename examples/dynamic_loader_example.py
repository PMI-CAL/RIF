#!/usr/bin/env python3
"""
Dynamic MCP Loader Example

Demonstrates the usage of the Dynamic MCP Loader for issue #82.

Issue: #82 - Implement dynamic MCP loader
Component: Example and demonstration
"""

import asyncio
import sys
import os

# Add mcp module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp.loader.dynamic_loader import DynamicMCPLoader, ProjectContext


async def main():
    """Demonstrate dynamic MCP loader functionality"""
    print("🚀 Dynamic MCP Loader Example")
    print("=" * 50)
    
    # Create a dynamic loader instance
    loader = DynamicMCPLoader(
        max_concurrent_loads=4,
        resource_budget_mb=512
    )
    
    print(f"✅ Created loader with {loader.max_concurrent_loads} concurrent loads")
    print(f"   Resource budget: {loader.resource_budget_mb}MB")
    print()
    
    # Example 1: Simple Python project with GitHub integration
    print("📁 Example 1: Python project with GitHub integration")
    simple_context = ProjectContext(
        project_path="/tmp/python_project",
        technology_stack={'python', 'git'},
        integrations={'github'},
        needs_reasoning=False,
        needs_memory=False,
        needs_database=False,
        needs_cloud=False,
        complexity='low',
        agent_type='rif-implementer'
    )
    
    # Detect requirements
    requirements = await loader.detect_requirements(simple_context)
    print(f"📋 Detected requirements: {sorted(requirements)}")
    
    # Map to servers
    servers = await loader.map_requirements_to_servers(requirements, simple_context)
    print(f"🔗 Mapped to {len(servers)} servers:")
    for server in servers:
        memory_mb = server['resource_requirements']['memory_mb']
        print(f"   • {server['name']} ({memory_mb}MB)")
    
    # Load servers
    print("⚡ Loading servers...")
    results = await loader.load_servers_for_project(simple_context)
    
    successful = [r for r in results if r.status == 'success']
    failed = [r for r in results if r.status == 'failed']
    
    print(f"✅ Successfully loaded: {len(successful)}")
    print(f"❌ Failed to load: {len(failed)}")
    
    for result in successful:
        print(f"   • {result.server_name} ({result.load_time_ms}ms)")
    
    print()
    
    # Example 2: Complex project with multiple requirements
    print("📁 Example 2: Complex project with reasoning and database")
    complex_context = ProjectContext(
        project_path="/tmp/complex_project",
        technology_stack={'python', 'nodejs', 'docker', 'git'},
        integrations={'github', 'database', 'cloud'},
        needs_reasoning=True,
        needs_memory=True,
        needs_database=True,
        needs_cloud=True,
        complexity='high',
        agent_type='rif-analyst'
    )
    
    # Detect requirements for complex project
    complex_requirements = await loader.detect_requirements(complex_context)
    print(f"📋 Complex project requirements: {sorted(complex_requirements)}")
    
    # Map to servers with resource constraints
    complex_servers = await loader.map_requirements_to_servers(complex_requirements, complex_context)
    print(f"🔗 Mapped to {len(complex_servers)} optimized servers:")
    
    total_memory = 0
    for server in complex_servers:
        memory_mb = server['resource_requirements']['memory_mb']
        total_memory += memory_mb
        priority = server['priority']
        print(f"   • {server['name']} ({memory_mb}MB, priority {priority})")
    
    print(f"📊 Total estimated memory usage: {total_memory}MB")
    print()
    
    # Show active servers
    active_servers = await loader.get_active_servers()
    print("🔍 Currently active servers:")
    for server_id, info in active_servers.items():
        print(f"   • {info['name']}: {info['health']}")
    
    # Show loading metrics
    metrics = await loader.get_loading_metrics()
    print("📈 Loading metrics:")
    if 'message' not in metrics:
        print(f"   • Total load attempts: {metrics['total_load_attempts']}")
        print(f"   • Success rate: {metrics['success_rate']:.1f}%")
        print(f"   • Average load time: {metrics['average_load_time_ms']:.0f}ms")
        print(f"   • Resource utilization: {metrics['resource_utilization_percent']:.1f}%")
    else:
        print(f"   • {metrics['message']}")
    
    print()
    
    # Cleanup - unload some servers
    print("🧹 Cleanup: Unloading servers...")
    for server_id in list(loader.active_servers.keys())[:2]:  # Unload first 2 servers
        success = await loader.unload_server(server_id)
        print(f"   • Unloaded {server_id}: {'✅' if success else '❌'}")
    
    print()
    print("🎉 Dynamic MCP Loader demonstration complete!")
    print(f"   Final active servers: {len(loader.active_servers)}")
    print(f"   Total resource usage: {loader.total_resource_usage_mb}MB")


if __name__ == '__main__':
    asyncio.run(main())