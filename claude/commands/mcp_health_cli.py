#!/usr/bin/env python3
"""
MCP Health Monitor CLI
Simple command-line interface for MCP server health monitoring

Usage:
  python mcp_health_cli.py --help
  python mcp_health_cli.py monitor --dashboard
  python mcp_health_cli.py servers
  python mcp_health_cli.py metrics

Issue: #84 - Create MCP health monitor
Component: Simplified CLI for health monitoring
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.monitor.health_monitor import MCPHealthMonitor
from mcp.monitor.protocols import HealthStatus
from mcp.registry.server_registry import MCPServerRegistry
from mcp.mock.mock_server import MockMCPServer


class HealthMonitorCLI:
    """Simple CLI for MCP health monitoring"""
    
    def __init__(self, storage_path: str = "knowledge/monitoring", registry_file: str = "knowledge/mcp_registry.json"):
        """Initialize the CLI with paths"""
        self.storage_path = storage_path
        self.registry_file = registry_file
        self.monitor: Optional[MCPHealthMonitor] = None
        self.registry: Optional[MCPServerRegistry] = None
        self.mock_servers: Dict[str, Any] = {}
        self.is_running = False
        
    async def initialize(self):
        """Initialize monitor and registry"""
        try:
            # Initialize server registry
            self.registry = MCPServerRegistry(registry_file=self.registry_file, auto_save=True)
            
            # Initialize health monitor
            self.monitor = MCPHealthMonitor(
                check_interval_seconds=30,
                storage_path=self.storage_path
            )
            
            # Add alert callback
            self.monitor.add_alert_callback(self._handle_alert)
            
            print("🏥 MCP Health Monitor initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def _handle_alert(self, alert_data: Dict[str, Any]):
        """Handle alerts from monitoring system"""
        severity = alert_data.get('severity', 'info').upper()
        server_name = alert_data.get('server_name', 'Unknown')
        message = alert_data.get('message', 'Health alert')
        
        if severity == 'CRITICAL':
            print(f"🚨 CRITICAL: {server_name} - {message}")
        elif severity == 'WARNING':
            print(f"⚠️  WARNING: {server_name} - {message}")
        else:
            print(f"ℹ️  INFO: {server_name} - {message}")
    
    async def register_demo_servers(self):
        """Register some demo mock servers for testing"""
        demo_servers = [
            {
                "server_id": "demo-github-api",
                "name": "Demo GitHub API Server",
                "capabilities": ["github_api", "repositories", "issues"]
            },
            {
                "server_id": "demo-database-connector", 
                "name": "Demo Database Connector",
                "capabilities": ["database", "sql", "migrations"]
            },
            {
                "server_id": "demo-file-processor",
                "name": "Demo File Processor", 
                "capabilities": ["file_processing", "validation", "parsing"]
            }
        ]
        
        for server_config in demo_servers:
            # Create mock server with proper config format
            full_config = {**server_config, "mock": True}
            mock_server = MockMCPServer(server_config=full_config)
            await mock_server.initialize()
            
            # Register with registry
            await self.registry.register_server(full_config)
            
            # Register with health monitor
            await self.monitor.register_server(mock_server, full_config)
            
            # Store for cleanup
            self.mock_servers[server_config["server_id"]] = mock_server
            
            print(f"✅ Registered: {server_config['name']}")
    
    async def start_monitoring(self):
        """Start the health monitoring system"""
        if not self.monitor:
            print("❌ Health monitor not initialized")
            return False
        
        # Register demo servers
        print("📝 Registering demo servers...")
        await self.register_demo_servers()
        
        # Start monitoring
        await self.monitor.start_monitoring()
        self.is_running = True
        
        print("🔄 Health monitoring started")
        print("   Monitoring servers every 30 seconds...")
        return True
    
    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        print("⏹️  Stopping health monitoring...")
        
        if self.monitor:
            await self.monitor.stop_monitoring()
        
        # Stop mock servers
        for mock_server in self.mock_servers.values():
            await mock_server.cleanup()
        
        self.is_running = False
        print("✅ Health monitoring stopped")
    
    async def show_status(self):
        """Show current health status"""
        if not self.monitor:
            print("❌ Health monitor not initialized")
            return
        
        print("\\n" + "=" * 60)
        print("🏥 MCP Health Monitor Status")
        print("=" * 60)
        
        # System metrics
        system_metrics = await self.monitor.get_system_metrics()
        print(f"📊 System Metrics:")
        print(f"   Active: {'✅ Yes' if system_metrics['monitoring_active'] else '❌ No'}")
        print(f"   Servers: {system_metrics['monitored_servers']}")
        print(f"   Health Checks: {system_metrics['total_health_checks']}")
        print(f"   Recoveries: {system_metrics['total_recoveries']}")
        print(f"   Alerts: {system_metrics['total_alerts']}")
        print(f"   Uptime: {system_metrics['uptime_hours']:.1f} hours")
        
        # Server health
        all_health = await self.monitor.get_all_server_health()
        print(f"\\n🖥️  Server Health ({len(all_health)} servers):")
        
        for server_id, health in all_health.items():
            status = health['status']
            name = health['server_name']
            
            if status == 'healthy':
                status_display = f"✅ {status.upper()}"
            elif status == 'degraded':
                status_display = f"⚠️  {status.upper()}"
            elif status == 'unhealthy':
                status_display = f"❌ {status.upper()}"
            else:
                status_display = f"❓ {status.upper()}"
            
            print(f"   {name} ({server_id})")
            print(f"     Status: {status_display}")
            print(f"     Uptime: {health['uptime_percent']:.1f}%")
            print(f"     Response: {health['average_response_time_ms']:.1f}ms")
            print(f"     Checks: {health['total_checks']} (Failures: {health['total_failures']})")
        
        print("\\n" + "=" * 60)
    
    async def show_servers(self):
        """Show all registered servers"""
        if not self.registry:
            print("❌ Registry not initialized")
            return
        
        servers = await self.registry.list_servers()
        
        print("\\n📋 Registered MCP Servers")
        print("=" * 40)
        
        for server in servers:
            print(f"🖥️  {server['name']}")
            print(f"   ID: {server['server_id']}")
            print(f"   Capabilities: {', '.join(server.get('capabilities', []))}")
            print(f"   Health: {server.get('health_status', 'unknown')}")
            print()
    
    async def show_detailed_metrics(self, server_id: Optional[str] = None):
        """Show detailed health metrics"""
        if not self.monitor:
            print("❌ Health monitor not initialized")
            return
        
        if server_id:
            health = await self.monitor.get_server_health(server_id)
            if not health:
                print(f"❌ Server {server_id} not found")
                return
            
            print(f"\\n📈 Detailed Metrics: {health['server_name']}")
            print("=" * 50)
            self._print_server_metrics(health)
        else:
            all_health = await self.monitor.get_all_server_health()
            print("\\n📈 All Server Metrics")
            print("=" * 50)
            
            for server_id, health in all_health.items():
                print(f"\\n🖥️  {health['server_name']} ({server_id})")
                print("-" * 30)
                self._print_server_metrics(health)
    
    def _print_server_metrics(self, health: Dict[str, Any]):
        """Print formatted server metrics"""
        status = health['status']
        print(f"   Status: {status.upper()}")
        print(f"   Uptime: {health['uptime_percent']:.2f}%")
        print(f"   Avg Response Time: {health['average_response_time_ms']:.2f}ms")
        print(f"   Total Checks: {health['total_checks']}")
        print(f"   Failed Checks: {health['total_failures']}")
        print(f"   Recovery Attempts: {health['recovery_attempts']}")
        print(f"   Successful Recoveries: {health['successful_recoveries']}")
        print(f"   Alerts Sent: {health['alerts_sent']}")
        print(f"   Last Check: {health['last_check']}")
    
    async def run_dashboard(self, update_interval: int = 5):
        """Run interactive dashboard"""
        try:
            while self.is_running:
                # Clear screen (works on Unix-like systems)
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🏥 MCP Health Monitor Dashboard")
                print("=" * 60)
                print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("Press Ctrl+C to exit...")
                
                await self.show_status()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\\n⏹️  Dashboard stopped")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='MCP Health Monitor CLI')
    parser.add_argument('--storage-path', default='knowledge/monitoring', 
                       help='Storage path for monitoring data')
    parser.add_argument('--registry-file', default='knowledge/mcp_registry.json',
                       help='Server registry file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start health monitoring')
    monitor_parser.add_argument('--dashboard', action='store_true', 
                               help='Show live dashboard')
    monitor_parser.add_argument('--duration', type=int, default=0,
                               help='Monitor duration in seconds (0 = forever)')
    
    # Status command
    subparsers.add_parser('status', help='Show current health status')
    
    # Servers command
    subparsers.add_parser('servers', help='List all registered servers')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show detailed metrics')
    metrics_parser.add_argument('--server-id', help='Show metrics for specific server')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = HealthMonitorCLI(args.storage_path, args.registry_file)
    
    if not await cli.initialize():
        sys.exit(1)
    
    try:
        if args.command == 'monitor':
            await cli.start_monitoring()
            
            if args.dashboard:
                # Run dashboard
                await cli.run_dashboard()
            elif args.duration > 0:
                # Run for specified duration
                print(f"⏰ Monitoring for {args.duration} seconds...")
                await asyncio.sleep(args.duration)
            else:
                # Run forever
                print("🔄 Monitoring (Press Ctrl+C to stop)...")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
                    
        elif args.command == 'status':
            await cli.initialize()
            await cli.start_monitoring()
            await asyncio.sleep(2)  # Give it time to do some checks
            await cli.show_status()
            
        elif args.command == 'servers':
            await cli.show_servers()
            
        elif args.command == 'metrics':
            await cli.start_monitoring()
            await asyncio.sleep(2)  # Give it time to collect metrics
            await cli.show_detailed_metrics(args.server_id)
            
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\\n⏹️  Interrupted by user")
    
    finally:
        await cli.stop_monitoring()


if __name__ == '__main__':
    asyncio.run(main())