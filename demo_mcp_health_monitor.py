#!/usr/bin/env python3
"""
MCP Health Monitor Demo
Comprehensive demonstration of the MCP health monitoring system

This demo shows:
- Health monitor initialization and configuration
- Server registration and health checking
- Real-time monitoring and alerts
- Recovery strategy execution
- Performance metrics and trending
- Integration with server registry

Issue: #84 - Create MCP health monitor
Component: Complete demonstration of health monitoring system
"""

import asyncio
import json
import random
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.monitor.health_monitor import MCPHealthMonitor
from mcp.monitor.protocols import HealthStatus, HealthCheckType
from mcp.registry.server_registry import MCPServerRegistry
from mcp.mock.mock_server import MockMCPServer


class MCPHealthMonitorDemo:
    """Comprehensive demo of MCP health monitoring system"""
    
    def __init__(self):
        """Initialize demo components"""
        self.monitor = None
        self.registry = None
        self.mock_servers = {}
        self.demo_running = False
        
    async def setup_demo(self):
        """Set up the demo environment"""
        print("🔧 Setting up MCP Health Monitor Demo...")
        print("=" * 60)
        
        # Initialize registry
        self.registry = MCPServerRegistry(
            registry_file="knowledge/demo_health_registry.json",
            auto_save=True
        )
        
        # Initialize health monitor
        self.monitor = MCPHealthMonitor(
            check_interval_seconds=10,  # Faster for demo
            storage_path="knowledge/demo_monitoring"
        )
        
        # Set up alert callback
        self.monitor.add_alert_callback(self.handle_alert)
        
        print("✅ Registry and Health Monitor initialized")
        
        # Create demo servers with different behaviors
        demo_configs = [
            {
                "server_id": "stable-api-server",
                "name": "Stable API Server",
                "capabilities": ["api", "json", "rest"],
                "resource_requirements": {"memory_mb": 128, "cpu_percent": 10}
            },
            {
                "server_id": "unreliable-db-server", 
                "name": "Unreliable Database Server",
                "capabilities": ["database", "sql", "transactions"],
                "resource_requirements": {"memory_mb": 256, "cpu_percent": 20}
            },
            {
                "server_id": "performance-server",
                "name": "Performance Test Server", 
                "capabilities": ["performance", "benchmarking", "stress"],
                "resource_requirements": {"memory_mb": 64, "cpu_percent": 5}
            }
        ]
        
        print("\n📋 Creating demo servers...")
        
        for config in demo_configs:
            # Create mock server
            full_config = {**config, "mock": True}
            mock_server = self.create_custom_mock_server(full_config)
            await mock_server.initialize()
            
            # Register with registry
            await self.registry.register_server(full_config)
            
            # Register with health monitor  
            await self.monitor.register_server(mock_server, full_config)
            
            # Store for later use
            self.mock_servers[config["server_id"]] = mock_server
            
            print(f"   ✅ {config['name']} registered")
        
        print(f"\n🔄 Starting health monitoring...")
        await self.monitor.start_monitoring()
        self.demo_running = True
        
        print("🏥 Demo environment ready!")
        return True
    
    def create_custom_mock_server(self, config):
        """Create a mock server with custom behavior for demo purposes"""
        
        class CustomMockServer(MockMCPServer):
            def __init__(self, server_config):
                super().__init__(server_config)
                self.failure_mode = False
                self.response_delay = 0.1
                
            async def health_check(self) -> str:
                """Custom health check with demo behaviors"""
                if not self.is_running:
                    return "unhealthy"
                
                # Simulate response time variation
                await asyncio.sleep(self.response_delay)
                
                # Different behaviors for different servers
                if "unreliable" in self.server_id:
                    # Unreliable server - periodic failures
                    if random.random() < 0.3:  # 30% failure rate
                        return "unhealthy"
                    elif random.random() < 0.2:  # 20% degraded
                        return "degraded"
                    return "healthy"
                    
                elif "performance" in self.server_id:
                    # Performance server - variable response times
                    self.response_delay = random.uniform(0.05, 0.5)
                    if self.response_delay > 0.3:
                        return "degraded"
                    return "healthy"
                    
                else:
                    # Stable server - mostly healthy
                    if random.random() < 0.05:  # 5% degraded
                        return "degraded" 
                    return "healthy"
                    
            async def restart(self):
                """Mock restart operation"""
                print(f"   🔄 Restarting {self.name}...")
                await asyncio.sleep(1.0)  # Simulate restart time
                self.failure_mode = False
                return True
                
            async def reload(self):
                """Mock reload operation"""
                print(f"   ⚡ Reloading {self.name}...")
                await asyncio.sleep(0.5)  # Simulate reload time
                return True
        
        return CustomMockServer(config)
    
    async def handle_alert(self, alert_data):
        """Handle health alerts with demo formatting"""
        severity = alert_data.get('severity', 'info').upper()
        server_name = alert_data.get('server_name', 'Unknown')
        message = alert_data.get('message', 'Health alert')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if severity == 'CRITICAL':
            print(f"🚨 {timestamp} CRITICAL: {server_name} - {message}")
        elif severity == 'WARNING':
            print(f"⚠️  {timestamp} WARNING: {server_name} - {message}")
        else:
            print(f"ℹ️  {timestamp} INFO: {server_name} - {message}")
    
    async def run_monitoring_demo(self, duration_minutes=5):
        """Run the monitoring demo for specified duration"""
        print(f"\n🎬 Running Health Monitor Demo for {duration_minutes} minutes")
        print("=" * 60)
        print("Watching for health changes, alerts, and recovery actions...")
        print("Press Ctrl+C to stop the demo early")
        print()
        
        demo_start = datetime.now()
        status_interval = 30  # Show status every 30 seconds
        last_status = demo_start
        
        try:
            while self.demo_running:
                current_time = datetime.now()
                
                # Check if demo duration has elapsed
                if (current_time - demo_start).seconds > duration_minutes * 60:
                    print(f"\n⏰ Demo duration ({duration_minutes} minutes) completed")
                    break
                
                # Show status periodically
                if (current_time - last_status).seconds >= status_interval:
                    await self.show_status_update()
                    last_status = current_time
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        
        print("\n📊 Final Demo Results:")
        print("=" * 40)
        await self.show_final_results()
    
    async def show_status_update(self):
        """Show periodic status updates during demo"""
        print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # System metrics
        system_metrics = await self.monitor.get_system_metrics()
        print(f"   Health Checks: {system_metrics['total_health_checks']}")
        print(f"   Recoveries: {system_metrics['total_recoveries']}")  
        print(f"   Alerts: {system_metrics['total_alerts']}")
        
        # Server statuses
        all_health = await self.monitor.get_all_server_health()
        healthy_count = sum(1 for h in all_health.values() if h['status'] == 'healthy')
        degraded_count = sum(1 for h in all_health.values() if h['status'] == 'degraded')
        unhealthy_count = sum(1 for h in all_health.values() if h['status'] == 'unhealthy')
        
        print(f"   Servers: ✅ {healthy_count} healthy, ⚠️  {degraded_count} degraded, ❌ {unhealthy_count} unhealthy")
        print()
    
    async def show_final_results(self):
        """Show comprehensive final results"""
        # System metrics
        system_metrics = await self.monitor.get_system_metrics()
        print("🔍 System Performance:")
        print(f"   Total Health Checks: {system_metrics['total_health_checks']}")
        print(f"   Total Recovery Attempts: {system_metrics['total_recoveries']}")
        print(f"   Total Alerts Generated: {system_metrics['total_alerts']}")
        print(f"   Monitoring Uptime: {system_metrics['uptime_hours']:.2f} hours")
        print(f"   Avg Monitoring Overhead: {system_metrics['average_monitoring_overhead_ms']:.2f}ms")
        
        print("\n🖥️  Server Health Summary:")
        all_health = await self.monitor.get_all_server_health()
        
        for server_id, health in all_health.items():
            status = health['status']
            name = health['server_name']
            
            if status == 'healthy':
                status_icon = '✅'
            elif status == 'degraded':
                status_icon = '⚠️ '
            else:
                status_icon = '❌'
            
            print(f"   {status_icon} {name}")
            print(f"      Status: {status.upper()}")
            print(f"      Uptime: {health['uptime_percent']:.1f}%")
            print(f"      Avg Response: {health['average_response_time_ms']:.2f}ms")
            print(f"      Checks: {health['total_checks']} (Failures: {health['total_failures']})")
            print(f"      Recoveries: {health['successful_recoveries']}/{health['recovery_attempts']}")
            print(f"      Alerts: {health['alerts_sent']}")
        
        # Recovery strategy effectiveness
        print("\n🛠️  Recovery Strategy Performance:")
        for strategy in self.monitor.recovery_strategies:
            if strategy.usage_count > 0:
                success_rate = (strategy.success_count / strategy.usage_count) * 100
                print(f"   {strategy.name}: {strategy.usage_count} attempts, {success_rate:.1f}% success rate")
        
        print("\n📈 Performance Trends:")
        for server_id, health in all_health.items():
            health_record = self.monitor.health_records[server_id]
            trend = self.monitor._calculate_performance_trend(health_record)
            print(f"   {health['server_name']}: {trend}")
    
    async def demonstrate_features(self):
        """Demonstrate specific health monitor features"""
        print("\n🎯 Feature Demonstrations")
        print("=" * 40)
        
        # 1. Manual health check
        print("\n1️⃣  Manual Health Check:")
        server_id = "stable-api-server"
        await self.monitor._check_server_health(server_id)
        health = await self.monitor.get_server_health(server_id)
        print(f"   Status: {health['status']}")
        print(f"   Response Time: {health['average_response_time_ms']:.2f}ms")
        
        # 2. Alert system test
        print("\n2️⃣  Alert System Test:")
        print("   Triggering alert by simulating unhealthy server...")
        health_record = self.monitor.health_records[server_id]
        original_status = health_record.current_status
        health_record.current_status = HealthStatus.UNHEALTHY
        health_record.consecutive_failures = 5
        await self.monitor._process_alerts()
        health_record.current_status = original_status  # Restore
        health_record.consecutive_failures = 0
        
        # 3. Performance metrics
        print("\n3️⃣  Performance Metrics:")
        health_record = self.monitor.health_records[server_id]
        # Add some sample response times
        sample_times = [100, 120, 110, 130, 125]
        for rt in sample_times:
            health_record.response_times.append(rt)
        self.monitor._update_response_time_average(server_id)
        trend = self.monitor._calculate_performance_trend(health_record)
        print(f"   Performance Trend: {trend}")
        print(f"   Average Response Time: {health_record.average_response_time:.2f}ms")
        
        print("\n✨ Feature demonstration complete!")
    
    async def cleanup_demo(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo resources...")
        
        if self.monitor:
            await self.monitor.stop_monitoring()
        
        for mock_server in self.mock_servers.values():
            await mock_server.cleanup()
        
        self.demo_running = False
        print("✅ Demo cleanup complete")


async def main():
    """Main demo entry point"""
    demo = MCPHealthMonitorDemo()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n⏹️  Shutting down demo...")
        demo.demo_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Setup
        if not await demo.setup_demo():
            print("❌ Demo setup failed")
            return
        
        # Run feature demonstrations
        await demo.demonstrate_features()
        
        # Run monitoring demo
        await demo.run_monitoring_demo(duration_minutes=2)  # 2 minute demo
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await demo.cleanup_demo()
        print("\n🎬 MCP Health Monitor Demo Complete!")
        print("Thank you for trying the health monitoring system!")


if __name__ == '__main__':
    print("🏥 MCP Health Monitor - Comprehensive Demo")
    print("Issue: #84 - Create MCP health monitor")
    print("=" * 60)
    asyncio.run(main())