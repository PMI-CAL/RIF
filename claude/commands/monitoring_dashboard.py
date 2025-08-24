#!/usr/bin/env python3
"""
RIF Monitoring Dashboard
Simple web interface for viewing system monitoring data
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading
import webbrowser

class MonitoringDashboard:
    """Simple HTTP-based dashboard for monitoring data"""
    
    def __init__(self, monitoring_config_path: str, port: int = 8080):
        self.base_path = Path(monitoring_config_path).parent
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """Start the dashboard web server"""
        handler = self._create_request_handler()
        self.server = HTTPServer(('localhost', self.port), handler)
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        print(f"🖥️  RIF Monitoring Dashboard started at http://localhost:{self.port}")
        
    def stop_server(self):
        """Stop the dashboard web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
        if self.server_thread:
            self.server_thread.join(timeout=5)
            
    def _create_request_handler(self):
        """Create the HTTP request handler class"""
        dashboard = self
        
        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == '/' or self.path == '/dashboard':
                        self._serve_dashboard()
                    elif self.path == '/api/metrics':
                        self._serve_metrics_api()
                    elif self.path == '/api/alerts':
                        self._serve_alerts_api()
                    elif self.path == '/api/health':
                        self._serve_health_api()
                    elif self.path == '/api/consensus':
                        self._serve_consensus_api()
                    elif self.path == '/api/mcp/health':
                        self._serve_mcp_health_api()
                    elif self.path == '/api/mcp/metrics':
                        self._serve_mcp_metrics_api()
                    elif self.path == '/api/mcp/alerts':
                        self._serve_mcp_alerts_api()
                    elif self.path.startswith('/static/'):
                        self._serve_static_file()
                    else:
                        self._send_404()
                        
                except Exception as e:
                    self._send_error(str(e))
                    
            def _serve_dashboard(self):
                """Serve the main dashboard HTML"""
                html = dashboard._generate_dashboard_html()
                self._send_html_response(html)
                
            def _serve_metrics_api(self):
                """Serve metrics data as JSON API"""
                metrics = dashboard._load_recent_metrics()
                self._send_json_response(metrics)
                
            def _serve_alerts_api(self):
                """Serve alerts data as JSON API"""
                alerts = dashboard._load_recent_alerts()
                self._send_json_response(alerts)
                
            def _serve_health_api(self):
                """Serve system health status"""
                health = dashboard._get_health_status()
                self._send_json_response(health)
                
            def _serve_consensus_api(self):
                """Serve consensus data as JSON API"""
                consensus = dashboard._load_consensus_data()
                self._send_json_response(consensus)
                
            def _serve_mcp_health_api(self):
                """Serve MCP server health data as JSON API"""
                mcp_health = dashboard._load_mcp_health_data()
                self._send_json_response(mcp_health)
                
            def _serve_mcp_metrics_api(self):
                """Serve MCP performance metrics data as JSON API"""
                mcp_metrics = dashboard._load_mcp_metrics_data()
                self._send_json_response(mcp_metrics)
                
            def _serve_mcp_alerts_api(self):
                """Serve MCP alert data as JSON API"""
                mcp_alerts = dashboard._load_mcp_alerts_data()
                self._send_json_response(mcp_alerts)
                
            def _serve_static_file(self):
                """Serve static files (CSS, JS)"""
                # For now, return 404 - could extend to serve actual static files
                self._send_404()
                
            def _send_html_response(self, html: str):
                """Send HTML response"""
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
                
            def _send_json_response(self, data: Dict[str, Any]):
                """Send JSON response"""
                json_data = json.dumps(data, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json_data.encode())
                
            def _send_404(self):
                """Send 404 response"""
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<h1>404 - Not Found</h1>')
                
            def _send_error(self, error_msg: str):
                """Send error response"""
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'<h1>Error: {error_msg}</h1>'.encode())
                
            def log_message(self, format, *args):
                """Override to reduce request logging"""
                pass
                
        return DashboardRequestHandler
        
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        # Get current data
        metrics = self._load_recent_metrics()
        alerts = self._load_recent_alerts()
        consensus = self._load_consensus_data()
        health = self._get_health_status()
        mcp_health = self._load_mcp_health_data()
        mcp_alerts = self._load_mcp_alerts_data()
        
        # Calculate summary statistics
        total_metrics = len(metrics.get('metrics', []))
        active_alerts = len([a for a in alerts.get('alerts', []) if not a.get('resolved', True)])
        consensus_summary = consensus.get('summary', {})
        total_consensus_sessions = consensus_summary.get('total_sessions', 0)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RIF Monitoring Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
            color: #1d1d1f;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            color: #007aff;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .status-good {{ color: #34c759; }}
        .status-warning {{ color: #ff9500; }}
        .status-critical {{ color: #ff3b30; }}
        
        .section {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
        }}
        
        .alert {{
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 4px solid;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-color: #ff9500;
            color: #856404;
        }}
        .alert-critical {{
            background-color: #f8d7da;
            border-color: #ff3b30;
            color: #721c24;
        }}
        
        .metric-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric-item:last-child {{
            border-bottom: none;
        }}
        .metric-name {{
            font-weight: 500;
        }}
        .metric-value {{
            font-family: monospace;
            color: #666;
        }}
        
        .refresh-btn {{
            background: #007aff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
        }}
        .refresh-btn:hover {{
            background: #005ecf;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        
        .auto-refresh {{
            margin-left: 10px;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ RIF System Monitoring</h1>
        <p>Real-time monitoring of the Reactive Intelligence Framework</p>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        <span class="auto-refresh">Auto-refresh every 30s</span>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">System Health</div>
            <div class="stat-value status-{health['status'].lower()}">{health['status']}</div>
            <div class="timestamp">Last updated: {health.get('timestamp', 'Unknown')}</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Active Alerts</div>
            <div class="stat-value {'status-critical' if active_alerts > 0 else 'status-good'}">{active_alerts}</div>
            <div class="timestamp">Out of {len(alerts.get('alerts', []))} total alerts</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Metrics Collected</div>
            <div class="stat-value status-good">{total_metrics}</div>
            <div class="timestamp">In the last hour</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Knowledge System</div>
            <div class="stat-value {'status-good' if health.get('knowledge_accessible', False) else 'status-warning'}">
                {'Online' if health.get('knowledge_accessible', False) else 'Unknown'}
            </div>
            <div class="timestamp">LightRAG status</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Consensus Sessions</div>
            <div class="stat-value status-good">{total_consensus_sessions}</div>
            <div class="timestamp">Avg Agreement: {consensus_summary.get('average_agreement', 0.0):.1%}</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">MCP Servers</div>
            <div class="stat-value {'status-good' if mcp_health['summary']['healthy_servers'] == mcp_health['summary']['total_servers'] else 'status-warning' if mcp_health['summary']['degraded_servers'] > 0 else 'status-critical'}">{mcp_health['summary']['total_servers']}</div>
            <div class="timestamp">Health: {mcp_health['summary'].get('health_percentage', 100):.1f}% ({mcp_health['summary']['healthy_servers']} healthy)</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">MCP Alerts</div>
            <div class="stat-value {'status-critical' if mcp_alerts['critical_count'] > 0 else 'status-warning' if mcp_alerts['warning_count'] > 0 else 'status-good'}">{mcp_alerts['active_count']}</div>
            <div class="timestamp">{mcp_alerts['critical_count']} critical, {mcp_alerts['warning_count']} warnings</div>
        </div>
    </div>
    
    {self._generate_alerts_section(alerts)}
    {self._generate_mcp_health_section(mcp_health, mcp_alerts)}
    {self._generate_consensus_section(consensus)}
    {self._generate_metrics_section(metrics)}
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {{
            location.reload();
        }}, 30000);
        
        // Add timestamp to show when page was loaded
        document.addEventListener('DOMContentLoaded', function() {{
            const now = new Date().toLocaleTimeString();
            console.log('Dashboard loaded at:', now);
        }});
    </script>
</body>
</html>"""
        
        return html
        
    def _generate_alerts_section(self, alerts: Dict[str, Any]) -> str:
        """Generate the alerts section HTML"""
        alerts_list = alerts.get('alerts', [])
        active_alerts = [a for a in alerts_list if not a.get('resolved', True)]
        
        if not active_alerts:
            return """
            <div class="section">
                <h2>🟢 Alerts</h2>
                <p>No active alerts. System is running normally.</p>
            </div>
            """
            
        alerts_html = "<div class=\"section\"><h2>🚨 Active Alerts</h2>"
        
        for alert in active_alerts[:10]:  # Show max 10 alerts
            severity = alert.get('severity', 'warning')
            css_class = f"alert alert-{severity}"
            
            alerts_html += f"""
            <div class="{css_class}">
                <strong>{alert.get('name', 'Unknown Alert')}</strong><br>
                {alert.get('message', 'No message')}
                <div class="timestamp">
                    {alert.get('timestamp', 'Unknown time')} - 
                    Metric: {alert.get('metric_name', 'Unknown')} = {alert.get('metric_value', 'N/A')}
                </div>
            </div>
            """
            
        alerts_html += "</div>"
        return alerts_html
        
    def _generate_mcp_health_section(self, mcp_health: Dict[str, Any], mcp_alerts: Dict[str, Any]) -> str:
        """Generate the MCP server health section HTML"""
        servers = mcp_health.get('servers', [])
        summary = mcp_health.get('summary', {})
        active_alerts = mcp_alerts.get('active_alerts', [])
        
        if not servers and not active_alerts:
            return """
            <div class="section">
                <h2>🔧 MCP Server Health</h2>
                <p>No MCP servers are currently being monitored. Servers will appear here once the health monitoring system detects active MCP servers.</p>
            </div>
            """
        
        # Health status summary
        total_servers = summary.get('total_servers', 0)
        healthy_servers = summary.get('healthy_servers', 0)
        degraded_servers = summary.get('degraded_servers', 0)
        unhealthy_servers = summary.get('unhealthy_servers', 0)
        health_percentage = summary.get('health_percentage', 100)
        
        # Determine overall health status
        if health_percentage >= 90:
            health_status = "excellent"
            health_icon = "✅"
            health_class = "status-good"
        elif health_percentage >= 70:
            health_status = "good"
            health_icon = "✅"
            health_class = "status-good"
        elif health_percentage >= 50:
            health_status = "degraded"
            health_icon = "⚠️"
            health_class = "status-warning"
        else:
            health_status = "critical"
            health_icon = "🚨"
            health_class = "status-critical"
        
        section_html = f"""
        <div class="section">
            <h2>🔧 MCP Server Health</h2>
            
            <div class="stats-grid" style="margin-bottom: 20px;">
                <div class="stat-card">
                    <div class="stat-label">Overall Health</div>
                    <div class="stat-value {health_class}">{health_icon} {health_status.title()}</div>
                    <div class="timestamp">{health_percentage:.1f}% healthy ({healthy_servers}/{total_servers} servers)</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Server Status</div>
                    <div class="stat-value status-good">{healthy_servers}</div>
                    <div class="timestamp">Healthy • {degraded_servers} degraded • {unhealthy_servers} unhealthy</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Active Alerts</div>
                    <div class="stat-value {'status-critical' if mcp_alerts['critical_count'] > 0 else 'status-warning' if mcp_alerts['warning_count'] > 0 else 'status-good'}">{mcp_alerts['active_count']}</div>
                    <div class="timestamp">{mcp_alerts['critical_count']} critical • {mcp_alerts['warning_count']} warnings</div>
                </div>
            </div>
        """
        
        # Show recent server status
        if servers:
            section_html += "<h3>Server Status</h3>"
            
            # Sort servers by status (unhealthy first, then degraded, then healthy)
            status_priority = {"unhealthy": 0, "degraded": 1, "healthy": 2, "unknown": 3}
            sorted_servers = sorted(servers, key=lambda s: status_priority.get(s.get('status', 'unknown'), 4))
            
            for server in sorted_servers[:10]:  # Show up to 10 servers
                server_id = server.get('server_id', 'unknown')
                status = server.get('status', 'unknown')
                response_time = server.get('response_time_ms', 0)
                timestamp = server.get('timestamp', 'Unknown')
                error = server.get('error', '')
                
                # Status styling
                if status == "healthy":
                    status_class = "status-good"
                    status_icon = "✅"
                elif status == "degraded":
                    status_class = "status-warning"
                    status_icon = "⚠️"
                elif status == "unhealthy":
                    status_class = "status-critical"
                    status_icon = "❌"
                else:
                    status_class = "status-warning"
                    status_icon = "❓"
                
                error_text = f" • Error: {error}" if error else ""
                response_text = f" • {response_time:.0f}ms" if response_time > 0 else ""
                
                section_html += f"""
                <div class="metric-item">
                    <div>
                        <span class="{status_class}">{status_icon} {server_id}</span><br>
                        <small>Status: {status.title()}{response_text}{error_text}</small><br>
                        <small class="timestamp">{timestamp}</small>
                    </div>
                    <div class="metric-value {status_class}">
                        {status.upper()}
                    </div>
                </div>
                """
        
        # Show active alerts
        if active_alerts:
            section_html += "<h3>Active MCP Alerts</h3>"
            
            for alert in active_alerts[:5]:  # Show up to 5 alerts
                severity = alert.get('severity', 'info')
                message = alert.get('message', 'No message')
                server_name = alert.get('server_name', 'Unknown server')
                timestamp = alert.get('timestamp', 'Unknown time')
                
                # Alert styling
                if severity == 'critical':
                    alert_class = "alert alert-critical"
                    alert_icon = "🚨"
                elif severity == 'warning':
                    alert_class = "alert alert-warning"
                    alert_icon = "⚠️"
                else:
                    alert_class = "alert"
                    alert_icon = "ℹ️"
                
                section_html += f"""
                <div class="{alert_class}">
                    <strong>{alert_icon} {server_name}</strong><br>
                    {message}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """
        
        section_html += "</div>"
        return section_html
        
    def _generate_consensus_section(self, consensus: Dict[str, Any]) -> str:
        """Generate the consensus monitoring section HTML"""
        sessions = consensus.get('sessions', [])
        summary = consensus.get('summary', {})
        
        if not sessions and not summary.get('total_sessions', 0):
            return """
            <div class="section">
                <h2>🤝 Consensus Monitoring</h2>
                <p>No consensus sessions recorded yet. This section will show voting patterns and agreement trends once agents start making collective decisions.</p>
            </div>
            """
            
        # Generate summary statistics
        avg_agreement = summary.get('average_agreement', 0.0)
        avg_confidence = summary.get('average_confidence', 0.0)
        total_sessions = summary.get('total_sessions', 0)
        dissenter_events = summary.get('total_dissenter_events', 0)
        avg_decision_time = summary.get('average_decision_time', 0.0)
        
        # Determine health status
        consensus_health = "good"
        if avg_agreement < 0.6:
            consensus_health = "warning"
        if avg_agreement < 0.4:
            consensus_health = "critical"
            
        section_html = f"""
        <div class="section">
            <h2>🤝 Consensus Monitoring</h2>
            
            <div class="stats-grid" style="margin-bottom: 20px;">
                <div class="stat-card">
                    <div class="stat-label">Average Agreement</div>
                    <div class="stat-value status-{consensus_health}">{avg_agreement:.1%}</div>
                    <div class="timestamp">Across {total_sessions} sessions</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Average Confidence</div>
                    <div class="stat-value status-{'good' if avg_confidence > 0.7 else 'warning' if avg_confidence > 0.5 else 'critical'}">{avg_confidence:.1%}</div>
                    <div class="timestamp">Agent confidence in decisions</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Dissenter Events</div>
                    <div class="stat-value status-{'good' if dissenter_events < total_sessions * 0.2 else 'warning'}">{dissenter_events}</div>
                    <div class="timestamp">Total disagreements recorded</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Avg Decision Time</div>
                    <div class="stat-value status-{'good' if avg_decision_time < 600000 else 'warning' if avg_decision_time < 1800000 else 'critical'}">{self._format_time(avg_decision_time)}</div>
                    <div class="timestamp">Time to reach consensus</div>
                </div>
            </div>
        """
        
        # Show recent sessions if available
        if sessions:
            section_html += "<h3>Recent Consensus Sessions</h3>"
            
            # Sort sessions by timestamp (most recent first)
            recent_sessions = sorted(sessions, 
                                   key=lambda x: x.get('timestamp', ''), 
                                   reverse=True)[:10]
                                   
            for session in recent_sessions:
                session_id = session.get('session_id', 'Unknown')
                timestamp = session.get('timestamp', 'Unknown')
                
                # Handle both session and report formats
                if "metrics" in session:
                    metrics = session["metrics"]
                    agreement = metrics.get("agreement_level", 0.0)
                    decision_time = metrics.get("decision_time", 0)
                    dissenters = metrics.get("dissenting_agents", [])
                    total_participants = metrics.get("total_participants", 0)
                else:
                    # Direct session format
                    agreement = session.get("agreement_level", 0.0)
                    decision_time = session.get("decision_time_ms", 0)
                    dissenters = [d.get("agent", "unknown") for d in session.get("dissenters", [])]
                    total_participants = session.get("total_votes", 0)
                
                # Status styling based on agreement level
                if agreement >= 0.8:
                    status_class = "status-good"
                    status_icon = "✅"
                elif agreement >= 0.6:
                    status_class = "status-warning"
                    status_icon = "⚠️"
                else:
                    status_class = "status-critical"  
                    status_icon = "❌"
                    
                dissenter_text = f"{len(dissenters)} dissenters" if dissenters else "No dissenters"
                if dissenters:
                    dissenter_text += f" ({', '.join(dissenters[:3])}{'...' if len(dissenters) > 3 else ''})"
                    
                section_html += f"""
                <div class="metric-item">
                    <div>
                        <span class="{status_class}">{status_icon} Session {session_id}</span><br>
                        <small>{agreement:.1%} agreement • {total_participants} participants • {dissenter_text}</small><br>
                        <small class="timestamp">{timestamp} • {self._format_time(decision_time)}</small>
                    </div>
                    <div class="metric-value {status_class}">
                        {agreement:.1%}
                    </div>
                </div>
                """
        
        section_html += "</div>"
        return section_html
        
    def _format_time(self, time_ms: float) -> str:
        """Format time in milliseconds to human readable format"""
        if time_ms < 1000:
            return f"{time_ms:.0f}ms"
        elif time_ms < 60000:
            return f"{time_ms/1000:.1f}s"
        elif time_ms < 3600000:
            return f"{time_ms/60000:.1f}m"
        else:
            return f"{time_ms/3600000:.1f}h"
        
    def _generate_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Generate the metrics section HTML"""
        metrics_list = metrics.get('metrics', [])
        
        if not metrics_list:
            return """
            <div class="section">
                <h2>📊 Metrics</h2>
                <p>No recent metrics data available.</p>
            </div>
            """
            
        # Group metrics by category
        categories = {}
        for metric in metrics_list[-50:]:  # Show last 50 metrics
            category = metric.get('name', '').split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(metric)
            
        section_html = "<div class=\"section\"><h2>📊 Recent Metrics</h2>"
        
        for category, cat_metrics in categories.items():
            section_html += f"<h3>{category.title()} Metrics</h3>"
            
            for metric in cat_metrics[-10:]:  # Last 10 per category
                name = metric.get('name', 'Unknown')
                value = metric.get('value', 0)
                unit = metric.get('unit', '')
                timestamp = metric.get('timestamp', 'Unknown')
                
                # Format value based on unit
                if unit == 'bytes':
                    if value > 1024*1024*1024:
                        display_value = f"{value/(1024*1024*1024):.1f} GB"
                    elif value > 1024*1024:
                        display_value = f"{value/(1024*1024):.1f} MB"
                    elif value > 1024:
                        display_value = f"{value/1024:.1f} KB"
                    else:
                        display_value = f"{value:.0f} bytes"
                elif unit == 'ms':
                    display_value = f"{value:.1f} ms"
                elif unit == '%':
                    display_value = f"{value:.1f}%"
                else:
                    display_value = f"{value:.2f} {unit}".strip()
                    
                section_html += f"""
                <div class="metric-item">
                    <span class="metric-name">{name}</span>
                    <span class="metric-value">{display_value}</span>
                </div>
                """
                
        section_html += "</div>"
        return section_html
        
    def _load_recent_metrics(self) -> Dict[str, Any]:
        """Load recent metrics from storage"""
        try:
            metrics_dir = self.base_path / "knowledge" / "monitoring" / "metrics"
            
            if not metrics_dir.exists():
                return {"metrics": []}
                
            # Load today's metrics
            today = datetime.now().strftime("%Y%m%d")
            metrics_file = metrics_dir / f"metrics_{today}.jsonl"
            
            metrics = []
            if metrics_file.exists():
                with open(metrics_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                metrics.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                                
            # Also try to load recent files if today's is empty
            if not metrics:
                for days_back in range(1, 8):  # Try up to 7 days back
                    date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                    file_path = metrics_dir / f"metrics_{date}.jsonl"
                    
                    if file_path.exists():
                        with open(file_path) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        metrics.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                        break
                        
            return {"metrics": metrics[-200:]}  # Last 200 metrics
            
        except Exception as e:
            return {"metrics": [], "error": str(e)}
            
    def _load_recent_alerts(self) -> Dict[str, Any]:
        """Load recent alerts from storage"""
        try:
            alerts_dir = self.base_path / "knowledge" / "monitoring" / "alerts"
            
            if not alerts_dir.exists():
                return {"alerts": []}
                
            alerts = []
            alert_files = sorted(alerts_dir.glob("alert_*.json"), reverse=True)
            
            for alert_file in alert_files[:50]:  # Load last 50 alerts
                try:
                    with open(alert_file) as f:
                        alert_data = json.load(f)
                        alerts.append(alert_data)
                except (json.JSONDecodeError, IOError):
                    continue
                    
            return {"alerts": alerts}
            
        except Exception as e:
            return {"alerts": [], "error": str(e)}
            
    def _load_consensus_data(self) -> Dict[str, Any]:
        """Load recent consensus session data"""
        try:
            consensus_dir = self.base_path / "knowledge" / "monitoring" / "consensus"
            
            if not consensus_dir.exists():
                return {"sessions": [], "summary": {}}
                
            sessions = []
            
            # Load today's consensus data
            today = datetime.now().strftime("%Y%m%d")
            consensus_files = [
                consensus_dir / f"consensus_sessions_{today}.jsonl",
                consensus_dir / f"consensus_reports_{today}.jsonl"
            ]
            
            for consensus_file in consensus_files:
                if consensus_file.exists():
                    with open(consensus_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    session_data = json.loads(line)
                                    sessions.append(session_data)
                                except json.JSONDecodeError:
                                    continue
                                    
            # Also try recent files if today's is empty
            if not sessions:
                for days_back in range(1, 8):  # Try up to 7 days back
                    date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                    for file_pattern in ["consensus_sessions_", "consensus_reports_"]:
                        file_path = consensus_dir / f"{file_pattern}{date}.jsonl"
                        
                        if file_path.exists():
                            with open(file_path) as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            sessions.append(json.loads(line))
                                        except json.JSONDecodeError:
                                            continue
                            break
                    if sessions:
                        break
                        
            # Calculate summary statistics
            summary = self._calculate_consensus_summary(sessions)
            
            return {"sessions": sessions[-50:], "summary": summary}  # Last 50 sessions
            
        except Exception as e:
            return {"sessions": [], "summary": {}, "error": str(e)}
    
    def _load_mcp_health_data(self) -> Dict[str, Any]:
        """Load MCP server health monitoring data"""
        try:
            mcp_health_dir = self.base_path / "knowledge" / "monitoring" / "health_history"
            
            if not mcp_health_dir.exists():
                return {"servers": [], "summary": {"total_servers": 0, "healthy_servers": 0, "degraded_servers": 0, "unhealthy_servers": 0}}
            
            # Load today's health data
            today = datetime.now().strftime("%Y%m%d")
            health_file = mcp_health_dir / f"health_{today}.jsonl"
            
            health_data = []
            if health_file.exists():
                with open(health_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                health_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            
            # Aggregate by server
            server_health = {}
            for entry in health_data[-500:]:  # Last 500 entries
                server_id = entry.get("server_id", "unknown")
                if server_id not in server_health or entry.get("timestamp", "") > server_health[server_id].get("timestamp", ""):
                    server_health[server_id] = entry
            
            # Calculate summary
            healthy = sum(1 for h in server_health.values() if h.get("status") == "healthy")
            degraded = sum(1 for h in server_health.values() if h.get("status") == "degraded")
            unhealthy = sum(1 for h in server_health.values() if h.get("status") == "unhealthy")
            
            return {
                "servers": list(server_health.values()),
                "summary": {
                    "total_servers": len(server_health),
                    "healthy_servers": healthy,
                    "degraded_servers": degraded,
                    "unhealthy_servers": unhealthy,
                    "health_percentage": round((healthy / len(server_health)) * 100, 1) if server_health else 100
                }
            }
            
        except Exception as e:
            return {"servers": [], "summary": {"total_servers": 0}, "error": str(e)}
    
    def _load_mcp_metrics_data(self) -> Dict[str, Any]:
        """Load MCP performance metrics data"""
        try:
            mcp_metrics_dir = self.base_path / "knowledge" / "monitoring" / "metrics"
            
            if not mcp_metrics_dir.exists():
                return {"metrics": [], "summary": {}}
            
            # Load today's metrics data
            today = datetime.now().strftime("%Y%m%d")
            metrics_file = mcp_metrics_dir / f"performance_metrics_{today}.jsonl"
            
            metrics_data = []
            if metrics_file.exists():
                with open(metrics_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                metrics_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            
            # Recent metrics only (last hour)
            cutoff_time = time.time() - 3600
            recent_metrics = [m for m in metrics_data if m.get("timestamp", 0) >= cutoff_time]
            
            # Calculate summary by server
            server_metrics = {}
            for metric in recent_metrics:
                server_id = metric.get("server_id", "unknown")
                if server_id not in server_metrics:
                    server_metrics[server_id] = {"response_times": [], "error_rates": [], "throughput": []}
                
                metric_name = metric.get("metric_name", "")
                value = metric.get("value", 0)
                
                if "response_time" in metric_name:
                    server_metrics[server_id]["response_times"].append(value)
                elif "error" in metric_name:
                    server_metrics[server_id]["error_rates"].append(value)
                elif "throughput" in metric_name or "requests" in metric_name:
                    server_metrics[server_id]["throughput"].append(value)
            
            # Calculate averages
            summary = {}
            for server_id, metrics in server_metrics.items():
                summary[server_id] = {
                    "avg_response_time": round(sum(metrics["response_times"]) / len(metrics["response_times"]), 2) if metrics["response_times"] else 0,
                    "avg_error_rate": round(sum(metrics["error_rates"]) / len(metrics["error_rates"]), 2) if metrics["error_rates"] else 0,
                    "avg_throughput": round(sum(metrics["throughput"]) / len(metrics["throughput"]), 2) if metrics["throughput"] else 0
                }
            
            return {
                "metrics": recent_metrics[-200:],  # Last 200 metric points
                "summary": summary,
                "total_metrics": len(recent_metrics)
            }
            
        except Exception as e:
            return {"metrics": [], "summary": {}, "error": str(e)}
    
    def _load_mcp_alerts_data(self) -> Dict[str, Any]:
        """Load MCP alert data"""
        try:
            mcp_alerts_dir = self.base_path / "knowledge" / "monitoring" / "alerts"
            
            # Check for dashboard alerts file first (real-time alerts)
            dashboard_alerts_file = mcp_alerts_dir / "dashboard_alerts.json"
            if dashboard_alerts_file.exists():
                with open(dashboard_alerts_file, 'r') as f:
                    dashboard_alerts = json.load(f)
                
                active_alerts = [a for a in dashboard_alerts if not a.get("resolved", False)]
                
                return {
                    "active_alerts": active_alerts,
                    "total_alerts": len(dashboard_alerts),
                    "active_count": len(active_alerts),
                    "critical_count": len([a for a in active_alerts if a.get("severity") == "critical"]),
                    "warning_count": len([a for a in active_alerts if a.get("severity") == "warning"])
                }
            
            # Fallback to individual alert files
            if not mcp_alerts_dir.exists():
                return {"active_alerts": [], "total_alerts": 0, "active_count": 0}
            
            alerts = []
            alert_files = sorted(mcp_alerts_dir.glob("alert_*.json"), reverse=True)
            
            for alert_file in alert_files[:50]:  # Load last 50 alerts
                try:
                    with open(alert_file) as f:
                        alert_data = json.load(f)
                        alerts.append(alert_data)
                except (json.JSONDecodeError, IOError):
                    continue
            
            active_alerts = [a for a in alerts if not a.get("resolved", False)]
            
            return {
                "active_alerts": active_alerts,
                "total_alerts": len(alerts),
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.get("severity") == "critical"]),
                "warning_count": len([a for a in active_alerts if a.get("severity") == "warning"])
            }
            
        except Exception as e:
            return {"active_alerts": [], "total_alerts": 0, "error": str(e)}
            
    def _calculate_consensus_summary(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for consensus sessions"""
        if not sessions:
            return {
                "total_sessions": 0,
                "average_agreement": 0.0,
                "average_confidence": 0.0,
                "total_dissenter_events": 0,
                "average_decision_time": 0.0
            }
            
        agreement_levels = []
        confidence_averages = []
        decision_times = []
        dissenter_counts = []
        
        for session in sessions:
            # Handle both session format (direct metrics) and report format (nested metrics)
            if "metrics" in session:
                metrics = session["metrics"]
                agreement_levels.append(metrics.get("agreement_level", 0.0))
                conf_dist = metrics.get("confidence_distribution", {})
                confidence_averages.append(conf_dist.get("average", 0.0))
                decision_times.append(metrics.get("decision_time", 0))
                dissenter_counts.append(metrics.get("dissenter_count", 0))
            else:
                # Direct session format
                agreement_levels.append(session.get("agreement_level", 0.0))
                conf_stats = session.get("confidence_stats", {})
                confidence_averages.append(conf_stats.get("average", 0.0))
                decision_times.append(session.get("decision_time_ms", 0))
                dissenter_counts.append(session.get("dissenter_count", 0))
                
        return {
            "total_sessions": len(sessions),
            "average_agreement": statistics.mean(agreement_levels) if agreement_levels else 0.0,
            "average_confidence": statistics.mean([c for c in confidence_averages if c > 0]) if confidence_averages else 0.0,
            "total_dissenter_events": sum(dissenter_counts),
            "average_decision_time": statistics.mean([t for t in decision_times if t > 0]) if decision_times else 0.0,
            "agreement_trend": "stable",  # Would need historical comparison for real trend
            "recent_issues": []  # Could add analysis of recent problematic sessions
        }
            
    def _get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Check if monitoring directories exist
            monitoring_dir = self.base_path / "knowledge" / "monitoring"
            dirs_exist = monitoring_dir.exists()
            
            # Check if LightRAG is accessible
            lightrag_dir = self.base_path / "lightrag"
            knowledge_accessible = lightrag_dir.exists()
            
            # Check recent metrics
            metrics = self._load_recent_metrics()
            has_recent_metrics = len(metrics.get('metrics', [])) > 0
            
            # Check for critical alerts
            alerts = self._load_recent_alerts()
            critical_alerts = [a for a in alerts.get('alerts', []) 
                             if not a.get('resolved', True) and a.get('severity') == 'critical']
            
            # Determine overall status
            if critical_alerts:
                status = "CRITICAL"
            elif not (dirs_exist and knowledge_accessible and has_recent_metrics):
                status = "WARNING"  
            else:
                status = "GOOD"
                
            return {
                "status": status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "directories_exist": dirs_exist,
                "knowledge_accessible": knowledge_accessible,
                "has_recent_metrics": has_recent_metrics,
                "critical_alerts_count": len(critical_alerts)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }

def create_monitoring_dashboard(config_path: str = "/Users/cal/DEV/RIF/config/monitoring.yaml", 
                              port: int = 8080,
                              auto_open: bool = False) -> MonitoringDashboard:
    """Create and start a monitoring dashboard"""
    dashboard = MonitoringDashboard(config_path, port)
    dashboard.start_server()
    
    if auto_open:
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass  # Don't fail if browser can't be opened
            
    return dashboard

if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/cal/DEV/RIF/config/monitoring.yaml"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    dashboard = create_monitoring_dashboard(config_path, port, auto_open=True)
    
    try:
        print(f"Dashboard running at http://localhost:{port}")
        print("Press Ctrl+C to stop...")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nStopping dashboard...")
        dashboard.stop_server()