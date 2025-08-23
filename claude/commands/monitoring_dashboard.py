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
        health = self._get_health_status()
        
        # Calculate summary statistics
        total_metrics = len(metrics.get('metrics', []))
        active_alerts = len([a for a in alerts.get('alerts', []) if not a.get('resolved', True)])
        
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
    </div>
    
    {self._generate_alerts_section(alerts)}
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