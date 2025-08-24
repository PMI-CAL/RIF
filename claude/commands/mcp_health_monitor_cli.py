"""
MCP Health Monitor CLI Command
Interactive command-line interface for MCP server health monitoring system

Features:
- Start/stop health monitoring for registered MCP servers
- Real-time health status dashboard
- Server registration and configuration
- Alert system management
- Performance metrics and trend analysis
- Recovery strategy monitoring

Issue: #84 - Create MCP health monitor
Component: Command-line interface for health monitoring
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import click

# Add the mcp module to Python path
sys.path.append(str(Path(__file__).parent.parent))

from mcp.monitor.health_monitor import MCPHealthMonitor, ServerHealthRecord
from mcp.monitor.protocols import HealthStatus, HealthCheckType
from mcp.registry.server_registry import MCPServerRegistry
from mcp.mock.mock_server import MockMCPServer


class HealthMonitorCLI:
    """Command-line interface for MCP health monitoring system"""
    
    def __init__(self):
        """Initialize the health monitor CLI"""
        self.monitor: Optional[MCPHealthMonitor] = None
        self.registry: Optional[MCPServerRegistry] = None
        self.mock_servers: Dict[str, Any] = {}
        self.is_running = False
        
    async def initialize(self, storage_path: str = "knowledge/monitoring", registry_file: str = "knowledge/mcp_registry.json"):
        """
        Initialize health monitor and server registry
        
        Args:
            storage_path: Path for monitoring data storage
            registry_file: Path for server registry file
        """
        # Initialize server registry
        self.registry = MCPServerRegistry(registry_file=registry_file, auto_save=True)
        
        # Initialize health monitor
        self.monitor = MCPHealthMonitor(
            check_interval_seconds=30,
            storage_path=storage_path
        )
        
        # Set up alert callback
        self.monitor.add_alert_callback(self._handle_alert)
        
        click.echo("🏥 MCP Health Monitor initialized successfully")
        
    def _handle_alert(self, alert_data: Dict[str, Any]):
        """Handle health alerts from the monitoring system"""
        severity = alert_data.get('severity', 'info').upper()
        server_name = alert_data.get('server_name', 'Unknown')
        message = alert_data.get('message', 'Health alert')
        
        if severity == 'CRITICAL':
            click.secho(f"🚨 CRITICAL ALERT: {server_name} - {message}", fg='red', bold=True)
        elif severity == 'WARNING':
            click.secho(f"⚠️  WARNING: {server_name} - {message}", fg='yellow')
        else:
            click.secho(f"ℹ️  INFO: {server_name} - {message}", fg='blue')
    
    async def register_mock_server(self, server_id: str, name: str, capabilities: List[str]):
        """Register a mock MCP server for testing"""
        mock_server = MockMCPServer(server_id=server_id, name=name, capabilities=capabilities)
        await mock_server.start()
        
        server_config = {
            "server_id": server_id,
            "name": name,
            "capabilities": capabilities,
            "mock": True
        }
        
        # Register with registry
        await self.registry.register_server(server_config)
        
        # Register with health monitor
        await self.monitor.register_server(mock_server, server_config)
        
        # Store for cleanup
        self.mock_servers[server_id] = mock_server
        
        click.echo(f"✅ Registered mock server: {name} ({server_id})")
    
    async def start_monitoring(self):
        """Start the health monitoring system"""
        if not self.monitor:
            click.echo("❌ Health monitor not initialized")
            return
        
        # Load servers from registry and register them with monitor
        servers = await self.registry.list_servers()
        
        for server_data in servers:
            server_id = server_data["server_id"]
            
            # For now, only register mock servers with the monitor
            # Real servers would need actual server instances
            if server_data.get("mock", False) and server_id not in self.mock_servers:
                await self.register_mock_server(
                    server_id,
                    server_data["name"],
                    server_data["capabilities"]
                )
        
        await self.monitor.start_monitoring()
        self.is_running = True
        
        click.secho("🔄 Health monitoring started", fg='green', bold=True)
        click.echo("Monitoring servers every 30 seconds...")
        click.echo("Press Ctrl+C to stop monitoring")
    
    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        if self.monitor:
            await self.monitor.stop_monitoring()
        
        # Stop mock servers
        for mock_server in self.mock_servers.values():
            await mock_server.stop()
        
        self.is_running = False
        click.secho("⏹️  Health monitoring stopped", fg='red')
    
    async def show_dashboard(self):
        """Display real-time health monitoring dashboard"""
        if not self.monitor:
            click.echo("❌ Health monitor not initialized")
            return
        
        try:
            while self.is_running:
                # Clear screen
                click.clear()
                
                # Header
                click.secho("=" * 80, fg='blue')
                click.secho("🏥 MCP Health Monitor Dashboard", fg='blue', bold=True)
                click.secho("=" * 80, fg='blue')
                click.echo(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo()
                
                # System metrics
                system_metrics = await self.monitor.get_system_metrics()
                click.secho("📊 System Metrics", fg='cyan', bold=True)
                click.echo(f"  Active Monitoring: {'✅ Yes' if system_metrics['monitoring_active'] else '❌ No'}")
                click.echo(f"  Monitored Servers: {system_metrics['monitored_servers']}")
                click.echo(f"  Total Health Checks: {system_metrics['total_health_checks']}")
                click.echo(f"  Total Recoveries: {system_metrics['total_recoveries']}")
                click.echo(f"  Total Alerts: {system_metrics['total_alerts']}")
                click.echo(f"  Uptime: {system_metrics['uptime_hours']:.1f} hours")
                click.echo(f"  Avg Overhead: {system_metrics['average_monitoring_overhead_ms']:.2f}ms")
                click.echo()
                
                # Server health
                all_health = await self.monitor.get_all_server_health()
                click.secho("🖥️  Server Health Status", fg='cyan', bold=True)
                
                if not all_health:
                    click.echo("  No servers registered for monitoring")
                else:
                    for server_id, health in all_health.items():
                        status = health['status']
                        name = health['server_name']
                        
                        # Status color coding
                        if status == 'healthy':
                            status_color = 'green'
                            status_icon = '✅'
                        elif status == 'degraded':
                            status_color = 'yellow'
                            status_icon = '⚠️ '
                        elif status == 'unhealthy':
                            status_color = 'red'
                            status_icon = '❌'
                        else:
                            status_color = 'white'
                            status_icon = '❓'
                        
                        click.echo(f"  {status_icon} {name} ({server_id})")
                        click.secho(f"     Status: {status.upper()}", fg=status_color)
                        click.echo(f"     Uptime: {health['uptime_percent']:.1f}%")
                        click.echo(f"     Avg Response: {health['average_response_time_ms']:.2f}ms")
                        click.echo(f"     Checks: {health['total_checks']} (Failures: {health['total_failures']})")
                        click.echo(f"     Recoveries: {health['successful_recoveries']}/{health['recovery_attempts']}")
                        click.echo(f"     Alerts: {health['alerts_sent']}")
                        click.echo(f"     Last Check: {health['last_check']}")
                        click.echo()
                
                click.secho("Press Ctrl+C to exit dashboard", fg='white', dim=True)
                
                # Wait before next update
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            pass
    
    async def list_servers(self):
        """List all registered servers"""
        if not self.registry:
            click.echo("❌ Registry not initialized")
            return
        
        servers = await self.registry.list_servers()
        
        click.secho("📋 Registered MCP Servers", fg='cyan', bold=True)
        click.echo("=" * 50)
        
        for server in servers:
            click.echo(f"🖥️  {server['name']} ({server['server_id']})")
            click.echo(f"   Version: {server.get('version', 'unknown')}")
            click.echo(f"   Capabilities: {', '.join(server.get('capabilities', []))}")
            click.echo(f"   Health: {server.get('health_status', 'unknown')}")
            click.echo(f"   Mock Server: {'Yes' if server.get('mock', False) else 'No'}")
            click.echo()
    
    async def show_metrics(self, server_id: Optional[str] = None):
        """Show detailed metrics for a server or all servers"""
        if not self.monitor:
            click.echo("❌ Health monitor not initialized")
            return
        
        if server_id:
            health = await self.monitor.get_server_health(server_id)
            if not health:
                click.echo(f"❌ Server {server_id} not found")
                return
            
            click.secho(f"📈 Detailed Metrics for {health['server_name']}", fg='cyan', bold=True)
            click.echo("=" * 50)
            self._print_server_metrics(health)
        else:
            all_health = await self.monitor.get_all_server_health()
            click.secho("📈 All Server Metrics", fg='cyan', bold=True)
            click.echo("=" * 50)
            
            for server_id, health in all_health.items():
                click.secho(f"\n🖥️  {health['server_name']} ({server_id})", fg='white', bold=True)
                self._print_server_metrics(health)
    
    def _print_server_metrics(self, health: Dict[str, Any]):
        """Print formatted server metrics"""
        status = health['status']
        if status == 'healthy':
            status_display = click.style(status.upper(), fg='green')
        elif status == 'degraded':
            status_display = click.style(status.upper(), fg='yellow')
        else:
            status_display = click.style(status.upper(), fg='red')
        
        click.echo(f"  Status: {status_display}")
        click.echo(f"  Uptime: {health['uptime_percent']:.2f}%")
        click.echo(f"  Response Time: {health['average_response_time_ms']:.2f}ms")
        click.echo(f"  Performance Trend: {health.get('performance_trend', 'unknown')}")
        click.echo(f"  Total Checks: {health['total_checks']}")
        click.echo(f"  Failed Checks: {health['total_failures']}")
        click.echo(f"  Recovery Attempts: {health['recovery_attempts']}")
        click.echo(f"  Successful Recoveries: {health['successful_recoveries']}")
        click.echo(f"  Alerts Sent: {health['alerts_sent']}")
        click.echo(f"  Last Check: {health['last_check']}")


# CLI Command Groups
@click.group()
@click.option('--storage-path', default="knowledge/monitoring", help='Storage path for monitoring data')
@click.option('--registry-file', default="knowledge/mcp_registry.json", help='Server registry file path')
@click.pass_context
async def cli(ctx, storage_path: str, registry_file: str):
    """MCP Health Monitor - Enterprise-grade server health monitoring"""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = HealthMonitorCLI()
    await ctx.obj['cli'].initialize(storage_path, registry_file)


@cli.command()
@click.pass_context
async def start(ctx):
    """Start the health monitoring system"""
    cli_obj = ctx.obj['cli']
    await cli_obj.start_monitoring()


@cli.command()
@click.pass_context
async def stop(ctx):
    """Stop the health monitoring system"""
    cli_obj = ctx.obj['cli']
    await cli_obj.stop_monitoring()


@cli.command()
@click.pass_context
async def dashboard(ctx):
    """Show real-time health monitoring dashboard"""
    cli_obj = ctx.obj['cli']
    await cli_obj.start_monitoring()
    await cli_obj.show_dashboard()


@cli.command()
@click.pass_context
async def servers(ctx):
    """List all registered MCP servers"""
    cli_obj = ctx.obj['cli']
    await cli_obj.list_servers()


@cli.command()
@click.option('--server-id', help='Show metrics for specific server')
@click.pass_context
async def metrics(ctx, server_id: Optional[str]):
    """Show detailed health metrics"""
    cli_obj = ctx.obj['cli']
    await cli_obj.show_metrics(server_id)


@cli.command()
@click.option('--server-id', required=True, help='Server identifier')
@click.option('--name', required=True, help='Server name')
@click.option('--capabilities', multiple=True, help='Server capabilities (can specify multiple)')
@click.pass_context
async def register_mock(ctx, server_id: str, name: str, capabilities: tuple):
    """Register a mock MCP server for testing"""
    cli_obj = ctx.obj['cli']
    await cli_obj.register_mock_server(server_id, name, list(capabilities))


@cli.command()
@click.pass_context
async def monitor(ctx):
    """Start monitoring with live dashboard"""
    cli_obj = ctx.obj['cli']
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        click.echo("\n⏹️  Shutting down health monitor...")
        asyncio.create_task(cli_obj.stop_monitoring())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring and dashboard
    await cli_obj.start_monitoring()
    
    # Run dashboard in background task
    dashboard_task = asyncio.create_task(cli_obj.show_dashboard())
    
    try:
        await dashboard_task
    except KeyboardInterrupt:
        pass
    finally:
        await cli_obj.stop_monitoring()


if __name__ == '__main__':
    # For async CLI support, we need to wrap the click commands
    import asyncio
    
    def async_command(f):
        """Decorator to make click commands async"""
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to all commands
    for command in cli.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            command.callback = async_command(command.callback)
    
    # Apply to main CLI
    if asyncio.iscoroutinefunction(cli.callback):
        cli.callback = async_command(cli.callback)
    
    cli()