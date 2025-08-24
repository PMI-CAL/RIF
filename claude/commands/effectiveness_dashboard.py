#!/usr/bin/env python3
"""
Effectiveness Visualization Dashboard

A comprehensive dashboard for visualizing quality gate effectiveness metrics,
providing real-time insights into quality gate performance and trends.

Part of RIF Issue #94: Quality Gate Effectiveness Monitoring - Phase 2
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Key metrics for dashboard display"""
    overall_effectiveness: float
    false_positive_rate: float
    false_negative_rate: float
    intervention_accuracy: float
    total_decisions: int
    avg_response_time: float
    trend_direction: str  # 'improving', 'stable', 'declining'
    last_updated: str

@dataclass
class TrendData:
    """Time series data for trending"""
    timestamps: List[str]
    effectiveness_scores: List[float]
    false_positive_rates: List[float] 
    false_negative_rates: List[float]
    decision_volumes: List[int]

@dataclass
class ComponentBreakdown:
    """Performance breakdown by component type"""
    component_name: str
    effectiveness_score: float
    decision_count: int
    avg_processing_time: float
    success_rate: float
    trend: str

@dataclass
class DashboardData:
    """Complete dashboard data structure"""
    summary_metrics: DashboardMetrics
    trend_data: TrendData
    component_breakdown: List[ComponentBreakdown]
    recent_insights: List[str]
    alerts: List[str]

class EffectivenessDashboard:
    """
    Analytics dashboard for quality gate effectiveness visualization.
    Provides comprehensive metrics and visualizations for monitoring quality gates.
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize dashboard with knowledge base path"""
        self.knowledge_path = Path(knowledge_base_path)
        self.metrics_path = self.knowledge_path / "quality_metrics"
        self.dashboard_path = self.knowledge_path / "dashboards"
        
        # Create dashboard directory
        self.dashboard_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        
    def generate_dashboard_data(self, days: int = 30) -> DashboardData:
        """
        Generate complete dashboard data for specified time period
        
        Args:
            days: Number of days to analyze (default: 30)
            
        Returns:
            DashboardData: Complete dashboard dataset
        """
        self.start_time = datetime.now()
        
        try:
            logger.info(f"Generating dashboard data for last {days} days...")
            
            # Load recent metrics data
            metrics_data = self._load_recent_metrics(days)
            
            # Calculate summary metrics
            summary = self._calculate_summary_metrics(metrics_data)
            
            # Generate trend data
            trends = self._generate_trend_data(metrics_data, days)
            
            # Component performance breakdown
            component_breakdown = self._analyze_component_performance(metrics_data)
            
            # Recent insights
            insights = self._generate_insights(metrics_data)
            
            # Current alerts
            alerts = self._check_alerts(summary, trends)
            
            dashboard_data = DashboardData(
                summary_metrics=summary,
                trend_data=trends,
                component_breakdown=component_breakdown,
                recent_insights=insights,
                alerts=alerts
            )
            
            # Save dashboard data
            self._save_dashboard_data(dashboard_data)
            
            processing_time = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.info(f"Dashboard generation completed in {processing_time:.2f}ms")
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
            return self._create_empty_dashboard()
    
    def _load_recent_metrics(self, days: int) -> List[Dict[str, Any]]:
        """Load metrics data from recent time period"""
        metrics_data = []
        
        # Load from recent metrics directory
        recent_dir = self.metrics_path / "recent"
        if recent_dir.exists():
            for metrics_file in recent_dir.glob("*.json"):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        
                    # Check if within date range
                    file_date = datetime.fromisoformat(data.get('timestamp', '2025-01-01T00:00:00'))
                    cutoff_date = datetime.now() - timedelta(days=days)
                    
                    if file_date >= cutoff_date:
                        metrics_data.append(data)
                        
                except Exception as e:
                    logger.warning(f"Error loading metrics file {metrics_file}: {str(e)}")
        
        # Load from realtime directory for current session data
        realtime_dir = self.metrics_path / "realtime"
        if realtime_dir.exists():
            for session_file in realtime_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Add session decisions to metrics
                    for decision in session_data.get('decisions', []):
                        if decision.get('timestamp'):
                            decision_date = datetime.fromisoformat(decision['timestamp'])
                            cutoff_date = datetime.now() - timedelta(days=days)
                            
                            if decision_date >= cutoff_date:
                                metrics_data.append(decision)
                                
                except Exception as e:
                    logger.warning(f"Error loading session file {session_file}: {str(e)}")
        
        logger.info(f"Loaded {len(metrics_data)} metrics records")
        return sorted(metrics_data, key=lambda x: x.get('timestamp', ''))
    
    def _calculate_summary_metrics(self, metrics_data: List[Dict[str, Any]]) -> DashboardMetrics:
        """Calculate overall summary metrics"""
        if not metrics_data:
            return self._create_empty_metrics()
        
        total_decisions = len(metrics_data)
        
        # Calculate effectiveness metrics
        true_positives = sum(1 for m in metrics_data if m.get('outcome') == 'true_positive')
        false_positives = sum(1 for m in metrics_data if m.get('outcome') == 'false_positive')
        false_negatives = sum(1 for m in metrics_data if m.get('outcome') == 'false_negative')
        true_negatives = sum(1 for m in metrics_data if m.get('outcome') == 'true_negative')
        
        # Calculate rates
        if (true_positives + false_positives) > 0:
            overall_effectiveness = true_positives / (true_positives + false_positives)
        else:
            overall_effectiveness = 1.0
            
        if total_decisions > 0:
            false_positive_rate = false_positives / total_decisions
            false_negative_rate = false_negatives / total_decisions
        else:
            false_positive_rate = 0.0
            false_negative_rate = 0.0
        
        # Manual intervention accuracy
        interventions = [m for m in metrics_data if m.get('manual_intervention', False)]
        appropriate_interventions = [m for m in interventions if m.get('intervention_appropriate', True)]
        
        if len(interventions) > 0:
            intervention_accuracy = len(appropriate_interventions) / len(interventions)
        else:
            intervention_accuracy = 1.0
        
        # Average response time
        response_times = [m.get('processing_time_ms', 0) for m in metrics_data if m.get('processing_time_ms')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Trend direction (simplified - compare recent vs older)
        trend_direction = self._calculate_trend_direction(metrics_data)
        
        return DashboardMetrics(
            overall_effectiveness=overall_effectiveness,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            intervention_accuracy=intervention_accuracy,
            total_decisions=total_decisions,
            avg_response_time=avg_response_time,
            trend_direction=trend_direction,
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_trend_data(self, metrics_data: List[Dict[str, Any]], days: int) -> TrendData:
        """Generate time series trend data"""
        if not metrics_data:
            return TrendData([], [], [], [], [])
        
        # Group data by day
        daily_metrics = {}
        
        for metric in metrics_data:
            timestamp = metric.get('timestamp', '')
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp).date()
                    date_str = date.isoformat()
                    
                    if date_str not in daily_metrics:
                        daily_metrics[date_str] = []
                    daily_metrics[date_str].append(metric)
                except:
                    continue
        
        # Calculate daily aggregates
        timestamps = []
        effectiveness_scores = []
        false_positive_rates = []
        false_negative_rates = []
        decision_volumes = []
        
        # Fill in missing days with empty data
        start_date = datetime.now().date() - timedelta(days=days)
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.isoformat()
            
            timestamps.append(date_str)
            
            if date_str in daily_metrics:
                day_data = daily_metrics[date_str]
                
                # Calculate daily effectiveness
                tp = sum(1 for m in day_data if m.get('outcome') == 'true_positive')
                fp = sum(1 for m in day_data if m.get('outcome') == 'false_positive')
                fn = sum(1 for m in day_data if m.get('outcome') == 'false_negative')
                
                if (tp + fp) > 0:
                    effectiveness = tp / (tp + fp)
                else:
                    effectiveness = 1.0 if len(day_data) > 0 else 0.0
                
                fp_rate = fp / len(day_data) if day_data else 0.0
                fn_rate = fn / len(day_data) if day_data else 0.0
                
                effectiveness_scores.append(effectiveness)
                false_positive_rates.append(fp_rate)
                false_negative_rates.append(fn_rate)
                decision_volumes.append(len(day_data))
            else:
                effectiveness_scores.append(0.0)
                false_positive_rates.append(0.0)
                false_negative_rates.append(0.0)
                decision_volumes.append(0)
        
        return TrendData(
            timestamps=timestamps,
            effectiveness_scores=effectiveness_scores,
            false_positive_rates=false_positive_rates,
            false_negative_rates=false_negative_rates,
            decision_volumes=decision_volumes
        )
    
    def _analyze_component_performance(self, metrics_data: List[Dict[str, Any]]) -> List[ComponentBreakdown]:
        """Analyze performance by component type"""
        component_metrics = {}
        
        for metric in metrics_data:
            component = metric.get('component_type', 'unknown')
            
            if component not in component_metrics:
                component_metrics[component] = []
            component_metrics[component].append(metric)
        
        breakdown = []
        
        for component, data in component_metrics.items():
            if not data:
                continue
                
            # Calculate component effectiveness
            tp = sum(1 for m in data if m.get('outcome') == 'true_positive')
            fp = sum(1 for m in data if m.get('outcome') == 'false_positive')
            
            if (tp + fp) > 0:
                effectiveness = tp / (tp + fp)
            else:
                effectiveness = 1.0
            
            # Average processing time
            processing_times = [m.get('processing_time_ms', 0) for m in data]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Success rate (non-error decisions)
            successful = sum(1 for m in data if not m.get('error', False))
            success_rate = successful / len(data) if data else 0.0
            
            # Simple trend calculation
            if len(data) >= 10:
                recent_data = data[-5:]
                older_data = data[-10:-5] if len(data) >= 10 else data[:-5]
                
                recent_effectiveness = sum(1 for m in recent_data if m.get('outcome') == 'true_positive') / max(1, len(recent_data))
                older_effectiveness = sum(1 for m in older_data if m.get('outcome') == 'true_positive') / max(1, len(older_data))
                
                if recent_effectiveness > older_effectiveness * 1.05:
                    trend = 'improving'
                elif recent_effectiveness < older_effectiveness * 0.95:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
            
            breakdown.append(ComponentBreakdown(
                component_name=component,
                effectiveness_score=effectiveness,
                decision_count=len(data),
                avg_processing_time=avg_processing_time,
                success_rate=success_rate,
                trend=trend
            ))
        
        return sorted(breakdown, key=lambda x: x.decision_count, reverse=True)
    
    def _generate_insights(self, metrics_data: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable insights from metrics data"""
        insights = []
        
        if not metrics_data:
            return ["No recent quality gate data available for analysis"]
        
        # Analyze false positive patterns
        false_positives = [m for m in metrics_data if m.get('outcome') == 'false_positive']
        if len(false_positives) > len(metrics_data) * 0.15:  # >15% false positive rate
            insights.append(f"High false positive rate detected: {len(false_positives)}/{len(metrics_data)} decisions. Consider threshold adjustment.")
        
        # Analyze response times
        slow_decisions = [m for m in metrics_data if m.get('processing_time_ms', 0) > 100]  # >100ms
        if len(slow_decisions) > len(metrics_data) * 0.1:  # >10% slow
            insights.append(f"Performance concern: {len(slow_decisions)} decisions exceeded 100ms processing time.")
        
        # Component-specific insights
        component_metrics = {}
        for metric in metrics_data:
            component = metric.get('component_type', 'unknown')
            if component not in component_metrics:
                component_metrics[component] = []
            component_metrics[component].append(metric)
        
        for component, data in component_metrics.items():
            if len(data) >= 10:  # Enough data for insights
                errors = sum(1 for m in data if m.get('error', False))
                if errors > len(data) * 0.05:  # >5% error rate
                    insights.append(f"Component '{component}' showing elevated error rate: {errors}/{len(data)} decisions failed.")
        
        # Trend insights
        if len(metrics_data) >= 20:  # Need enough data for trend analysis
            recent_data = metrics_data[-10:]
            older_data = metrics_data[-20:-10]
            
            recent_fp_rate = sum(1 for m in recent_data if m.get('outcome') == 'false_positive') / len(recent_data)
            older_fp_rate = sum(1 for m in older_data if m.get('outcome') == 'false_positive') / len(older_data)
            
            if recent_fp_rate > older_fp_rate * 1.3:  # 30% increase
                insights.append("False positive rate trending upward - quality gate tuning may be needed.")
            elif recent_fp_rate < older_fp_rate * 0.7:  # 30% decrease
                insights.append("False positive rate improving - recent quality gate optimizations showing positive impact.")
        
        if not insights:
            insights.append("Quality gates performing within expected parameters.")
        
        return insights[:5]  # Return top 5 insights
    
    def _check_alerts(self, summary: DashboardMetrics, trends: TrendData) -> List[str]:
        """Check for alertable conditions"""
        alerts = []
        
        # High false positive rate alert
        if summary.false_positive_rate > 0.2:  # >20%
            alerts.append(f"ALERT: High false positive rate ({summary.false_positive_rate:.1%}) - immediate attention required")
        
        # High false negative rate alert
        if summary.false_negative_rate > 0.05:  # >5%
            alerts.append(f"ALERT: Elevated false negative rate ({summary.false_negative_rate:.1%}) - quality gates may be too permissive")
        
        # Low intervention accuracy
        if summary.intervention_accuracy < 0.8:  # <80%
            alerts.append(f"ALERT: Low manual intervention accuracy ({summary.intervention_accuracy:.1%}) - specialist training needed")
        
        # Performance degradation
        if summary.avg_response_time > 200:  # >200ms
            alerts.append(f"ALERT: Quality gate response time degraded ({summary.avg_response_time:.0f}ms average)")
        
        # Trend-based alerts
        if trends.effectiveness_scores:
            recent_effectiveness = sum(trends.effectiveness_scores[-7:]) / 7 if len(trends.effectiveness_scores) >= 7 else 0
            if recent_effectiveness < 0.8:  # <80% in last week
                alerts.append("ALERT: Quality gate effectiveness declining over past week")
        
        return alerts
    
    def _calculate_trend_direction(self, metrics_data: List[Dict[str, Any]]) -> str:
        """Calculate overall trend direction"""
        if len(metrics_data) < 10:
            return 'insufficient_data'
        
        # Split data in half
        mid_point = len(metrics_data) // 2
        older_data = metrics_data[:mid_point]
        recent_data = metrics_data[mid_point:]
        
        # Calculate effectiveness for each half
        def calc_effectiveness(data):
            tp = sum(1 for m in data if m.get('outcome') == 'true_positive')
            fp = sum(1 for m in data if m.get('outcome') == 'false_positive')
            return tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        older_eff = calc_effectiveness(older_data)
        recent_eff = calc_effectiveness(recent_data)
        
        if recent_eff > older_eff * 1.05:  # >5% improvement
            return 'improving'
        elif recent_eff < older_eff * 0.95:  # >5% decline
            return 'declining'
        else:
            return 'stable'
    
    def _save_dashboard_data(self, dashboard_data: DashboardData):
        """Save dashboard data to file system"""
        try:
            # Save current dashboard data
            dashboard_file = self.dashboard_path / "current_dashboard.json"
            with open(dashboard_file, 'w') as f:
                json.dump(asdict(dashboard_data), f, indent=2)
            
            # Save timestamped version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            historical_file = self.dashboard_path / f"dashboard_{timestamp}.json"
            with open(historical_file, 'w') as f:
                json.dump(asdict(dashboard_data), f, indent=2)
            
            logger.info(f"Dashboard data saved to {dashboard_file}")
            
        except Exception as e:
            logger.error(f"Error saving dashboard data: {str(e)}")
    
    def _create_empty_dashboard(self) -> DashboardData:
        """Create empty dashboard data structure"""
        return DashboardData(
            summary_metrics=self._create_empty_metrics(),
            trend_data=TrendData([], [], [], [], []),
            component_breakdown=[],
            recent_insights=["No data available for analysis"],
            alerts=["No alerts - insufficient data"]
        )
    
    def _create_empty_metrics(self) -> DashboardMetrics:
        """Create empty metrics structure"""
        return DashboardMetrics(
            overall_effectiveness=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            intervention_accuracy=0.0,
            total_decisions=0,
            avg_response_time=0.0,
            trend_direction='no_data',
            last_updated=datetime.now().isoformat()
        )
    
    def generate_html_report(self, dashboard_data: DashboardData) -> str:
        """Generate HTML dashboard report"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Gate Effectiveness Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .dashboard {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; color: #2c3e50; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #27ae60; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; }}
        .metric-trend {{ font-size: 0.8em; margin-top: 5px; }}
        .trend-improving {{ color: #27ae60; }}
        .trend-declining {{ color: #e74c3c; }}
        .trend-stable {{ color: #f39c12; }}
        .components-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .component-item {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #ecf0f1; }}
        .component-name {{ font-weight: bold; }}
        .component-stats {{ display: flex; gap: 20px; font-size: 0.9em; color: #7f8c8d; }}
        .insights-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .insight-item {{ padding: 8px 0; }}
        .alerts-section {{ background: #fff5f5; padding: 20px; border-radius: 8px; border-left: 4px solid #e74c3c; margin-bottom: 20px; }}
        .alert-item {{ color: #c0392b; font-weight: bold; padding: 5px 0; }}
        .footer {{ text-align: center; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Quality Gate Effectiveness Dashboard</h1>
            <p>Last Updated: {dashboard_data.summary_metrics.last_updated}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Effectiveness</div>
                <div class="metric-value">{dashboard_data.summary_metrics.overall_effectiveness:.1%}</div>
                <div class="metric-trend trend-{dashboard_data.summary_metrics.trend_direction.replace('_', '-')}">
                    Trend: {dashboard_data.summary_metrics.trend_direction.title()}
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">False Positive Rate</div>
                <div class="metric-value" style="color: #e74c3c;">{dashboard_data.summary_metrics.false_positive_rate:.1%}</div>
                <div class="metric-trend">Target: &lt;10%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">False Negative Rate</div>
                <div class="metric-value" style="color: #e67e22;">{dashboard_data.summary_metrics.false_negative_rate:.1%}</div>
                <div class="metric-trend">Target: &lt;2%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Intervention Accuracy</div>
                <div class="metric-value">{dashboard_data.summary_metrics.intervention_accuracy:.1%}</div>
                <div class="metric-trend">Target: &gt;95%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Decisions</div>
                <div class="metric-value" style="color: #3498db;">{dashboard_data.summary_metrics.total_decisions}</div>
                <div class="metric-trend">Analysis Period</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Avg Response Time</div>
                <div class="metric-value" style="color: #9b59b6;">{dashboard_data.summary_metrics.avg_response_time:.0f}ms</div>
                <div class="metric-trend">Target: &lt;50ms</div>
            </div>
        </div>
        
        <div class="components-section">
            <h2>Component Performance Breakdown</h2>
            {self._generate_component_html(dashboard_data.component_breakdown)}
        </div>
        
        <div class="insights-section">
            <h2>Recent Insights</h2>
            {''.join([f'<div class="insight-item">• {insight}</div>' for insight in dashboard_data.recent_insights])}
        </div>
        
        {self._generate_alerts_html(dashboard_data.alerts)}
        
        <div class="footer">
            <p>Generated by RIF Quality Gate Effectiveness Monitoring System</p>
            <p>Issue #94 - Analytics Dashboard Implementation</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_component_html(self, components: List[ComponentBreakdown]) -> str:
        """Generate HTML for component breakdown"""
        if not components:
            return "<p>No component data available</p>"
        
        html = ""
        for component in components:
            trend_class = f"trend-{component.trend.replace('_', '-')}"
            html += f"""
            <div class="component-item">
                <div class="component-name">{component.component_name}</div>
                <div class="component-stats">
                    <span>Effectiveness: {component.effectiveness_score:.1%}</span>
                    <span>Decisions: {component.decision_count}</span>
                    <span>Avg Time: {component.avg_processing_time:.0f}ms</span>
                    <span>Success: {component.success_rate:.1%}</span>
                    <span class="{trend_class}">Trend: {component.trend.title()}</span>
                </div>
            </div>
            """
        return html
    
    def _generate_alerts_html(self, alerts: List[str]) -> str:
        """Generate HTML for alerts section"""
        if not alerts or all("No alerts" in alert for alert in alerts):
            return ""
        
        alert_items = ''.join([f'<div class="alert-item">{alert}</div>' for alert in alerts if "ALERT:" in alert])
        
        if not alert_items:
            return ""
        
        return f"""
        <div class="alerts-section">
            <h2>Active Alerts</h2>
            {alert_items}
        </div>
        """

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Generate Quality Gate Effectiveness Dashboard')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--output', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--knowledge-path', default='/Users/cal/DEV/RIF/knowledge', help='Path to knowledge base')
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = EffectivenessDashboard(args.knowledge_path)
    
    # Generate dashboard data
    print(f"Generating effectiveness dashboard for last {args.days} days...")
    dashboard_data = dashboard.generate_dashboard_data(args.days)
    
    # Output results
    if args.output in ['json', 'both']:
        print("\n=== DASHBOARD JSON DATA ===")
        print(json.dumps(asdict(dashboard_data), indent=2))
    
    if args.output in ['html', 'both']:
        html_report = dashboard.generate_html_report(dashboard_data)
        html_file = Path(args.knowledge_path) / "dashboards" / "effectiveness_dashboard.html"
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"\n=== HTML DASHBOARD GENERATED ===")
        print(f"Dashboard saved to: {html_file}")
        print(f"Open in browser to view: file://{html_file}")
    
    # Display summary
    print(f"\n=== DASHBOARD SUMMARY ===")
    print(f"Overall Effectiveness: {dashboard_data.summary_metrics.overall_effectiveness:.1%}")
    print(f"False Positive Rate: {dashboard_data.summary_metrics.false_positive_rate:.1%}")
    print(f"False Negative Rate: {dashboard_data.summary_metrics.false_negative_rate:.1%}")
    print(f"Intervention Accuracy: {dashboard_data.summary_metrics.intervention_accuracy:.1%}")
    print(f"Total Decisions Analyzed: {dashboard_data.summary_metrics.total_decisions}")
    print(f"Average Response Time: {dashboard_data.summary_metrics.avg_response_time:.0f}ms")
    print(f"Trend Direction: {dashboard_data.summary_metrics.trend_direction}")
    
    if dashboard_data.alerts:
        print(f"\nActive Alerts: {len([a for a in dashboard_data.alerts if 'ALERT:' in a])}")
        for alert in dashboard_data.alerts:
            if "ALERT:" in alert:
                print(f"  {alert}")
    
    print(f"\nTop Insights:")
    for i, insight in enumerate(dashboard_data.recent_insights[:3], 1):
        print(f"  {i}. {insight}")

if __name__ == "__main__":
    main()