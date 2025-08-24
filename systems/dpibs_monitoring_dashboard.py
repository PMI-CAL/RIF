#!/usr/bin/env python3
"""
DPIBS Monitoring Dashboard and Real-time Analytics
Issue #122: DPIBS Architecture Phase 4 - Performance Optimization and Caching Architecture

Real-time monitoring dashboard providing:
- Live performance metrics visualization
- Multi-level cache performance tracking
- Alert management and notification system
- System resource utilization monitoring
- Performance trend analysis and reporting
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import sys
import os

# Add RIF to path for imports
sys.path.insert(0, '/Users/cal/DEV/RIF')

from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer


class DPIBSMonitoringDashboard:
    """
    Real-time monitoring dashboard for DPIBS performance analytics
    Provides web-based interface for monitoring system health and performance
    """
    
    def __init__(self, optimizer: DPIBSPerformanceOptimizer):
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        
        # Dashboard state
        self.dashboard_active = False
        self.monitoring_thread = None
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {
            'performance': [],
            'cache': [],
            'alerts': [],
            'system_resources': []
        }
        
        # Real-time analytics
        self.analytics_engine = DPIBSAnalyticsEngine()
        
        # Dashboard configuration
        self.config = {
            'refresh_interval_seconds': 5,
            'metrics_retention_hours': 24,
            'alert_retention_hours': 72,
            'max_metrics_in_memory': 2880  # 24 hours at 30-second intervals
        }
        
        self.logger.info("DPIBS Monitoring Dashboard initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring and dashboard services"""
        if self.dashboard_active:
            self.logger.warning("Dashboard already active")
            return
        
        self.dashboard_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("DPIBS monitoring dashboard started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring dashboard"""
        self.dashboard_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("DPIBS monitoring dashboard stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for collecting real-time metrics"""
        self.logger.info("Starting monitoring loop")
        
        while self.dashboard_active:
            try:
                timestamp = datetime.utcnow()
                
                # Collect performance metrics
                self._collect_performance_metrics(timestamp)
                
                # Collect cache metrics
                self._collect_cache_metrics(timestamp)
                
                # Collect system resource metrics
                self._collect_system_metrics(timestamp)
                
                # Check for alerts
                self._check_and_collect_alerts(timestamp)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Run analytics
                self.analytics_engine.update_analytics(self.metrics_history)
                
                # Wait for next collection cycle
                time.sleep(self.config['refresh_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config['refresh_interval_seconds'])
    
    def _collect_performance_metrics(self, timestamp: datetime) -> None:
        """Collect current performance metrics"""
        try:
            perf_report = self.optimizer.get_enhanced_performance_report()
            
            performance_snapshot = {
                'timestamp': timestamp.isoformat(),
                'avg_response_time_ms': perf_report.get('performance_summary', {}).get('avg_response_time_ms', 0),
                'sub_200ms_operations': perf_report.get('performance_summary', {}).get('sub_200ms_operations', 0),
                'total_operations': perf_report.get('performance_summary', {}).get('total_operations', 0),
                'operations_breakdown': perf_report.get('operations_breakdown', {}),
                'phase4_compliance': perf_report.get('phase4_compliance', {})
            }
            
            self.metrics_history['performance'].append(performance_snapshot)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics: {e}")
    
    def _collect_cache_metrics(self, timestamp: datetime) -> None:
        """Collect multi-level cache performance metrics"""
        try:
            cache_stats = self.optimizer.cache_manager.get_cache_stats()
            
            cache_snapshot = {
                'timestamp': timestamp.isoformat(),
                'overall_hit_rate': cache_stats['overall']['hit_rate_percent'],
                'total_requests': cache_stats['overall']['total_requests'],
                'l1_performance': cache_stats['levels']['l1'],
                'l2_performance': cache_stats['levels']['l2'],
                'l3_performance': cache_stats['levels']['l3'],
                'storage_stats': cache_stats['storage']
            }
            
            self.metrics_history['cache'].append(cache_snapshot)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect cache metrics: {e}")
    
    def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system resource utilization metrics"""
        try:
            resource_info = self.optimizer.resource_monitor.get_system_resources()
            
            if 'cpu' in resource_info:
                system_snapshot = {
                    'timestamp': timestamp.isoformat(),
                    'cpu_usage': resource_info['cpu']['usage_percent'],
                    'memory_usage': resource_info['memory']['usage_percent'],
                    'memory_available_gb': resource_info['memory']['available_gb'],
                    'disk_usage': resource_info['disk']['usage_percent'],
                    'disk_available_gb': resource_info['disk']['available_gb']
                }
                
                self.metrics_history['system_resources'].append(system_snapshot)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _check_and_collect_alerts(self, timestamp: datetime) -> None:
        """Check for performance alerts and collect them"""
        try:
            alerts = self.optimizer.performance_monitor.check_performance_thresholds()
            
            if alerts:
                processed_alerts = self.optimizer.alert_manager.process_alerts(alerts)
                
                for alert in processed_alerts:
                    alert_record = {
                        'timestamp': timestamp.isoformat(),
                        'alert_id': alert['alert_id'],
                        'metric': alert['metric'],
                        'current_value': alert['current_value'],
                        'threshold': alert['threshold'],
                        'severity': alert['severity'],
                        'recommendations': alert.get('recommendations', [])
                    }
                    
                    self.metrics_history['alerts'].append(alert_record)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect alerts: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config['metrics_retention_hours'])
            alert_cutoff_time = datetime.utcnow() - timedelta(hours=self.config['alert_retention_hours'])
            
            for metric_type in ['performance', 'cache', 'system_resources']:
                original_count = len(self.metrics_history[metric_type])
                self.metrics_history[metric_type] = [
                    m for m in self.metrics_history[metric_type]
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]
                
                # Also limit by max count
                if len(self.metrics_history[metric_type]) > self.config['max_metrics_in_memory']:
                    self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.config['max_metrics_in_memory']:]
            
            # Clean up alerts with different retention
            original_alert_count = len(self.metrics_history['alerts'])
            self.metrics_history['alerts'] = [
                a for a in self.metrics_history['alerts']
                if datetime.fromisoformat(a['timestamp']) > alert_cutoff_time
            ]
            
        except Exception as e:
            self.logger.warning(f\"Failed to cleanup old metrics: {e}\")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        \"\"\"Get current dashboard data for display\"\"\"\n        current_time = datetime.utcnow()
        
        # Get latest metrics
        latest_performance = self.metrics_history['performance'][-1] if self.metrics_history['performance'] else {}
        latest_cache = self.metrics_history['cache'][-1] if self.metrics_history['cache'] else {}
        latest_system = self.metrics_history['system_resources'][-1] if self.metrics_history['system_resources'] else {}
        
        # Get recent alerts (last hour)
        recent_alerts = [
            alert for alert in self.metrics_history['alerts']
            if datetime.fromisoformat(alert['timestamp']) > current_time - timedelta(hours=1)
        ]
        
        # Calculate trends
        trends = self._calculate_trends()
        
        # Get analytics insights
        analytics = self.analytics_engine.get_current_insights()
        
        return {
            'timestamp': current_time.isoformat(),
            'status': 'active' if self.dashboard_active else 'inactive',
            'current_metrics': {
                'performance': latest_performance,
                'cache': latest_cache,
                'system': latest_system
            },
            'recent_alerts': recent_alerts,
            'trends': trends,
            'analytics': analytics,
            'historical_data': {
                'performance_history': self.metrics_history['performance'][-60:],  # Last hour at 1-minute intervals
                'cache_history': self.metrics_history['cache'][-60:],
                'system_history': self.metrics_history['system_resources'][-60:]
            }
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        \"\"\"Calculate performance trends from historical data\"\"\"\n        trends = {}
        
        try:
            # Performance trends
            if len(self.metrics_history['performance']) >= 10:
                recent_perf = self.metrics_history['performance'][-10:]
                older_perf = self.metrics_history['performance'][-20:-10] if len(self.metrics_history['performance']) >= 20 else self.metrics_history['performance'][:10]
                
                recent_avg_response = sum(p.get('avg_response_time_ms', 0) for p in recent_perf) / len(recent_perf)
                older_avg_response = sum(p.get('avg_response_time_ms', 0) for p in older_perf) / len(older_perf) if older_perf else recent_avg_response
                
                response_time_trend = 'improving' if recent_avg_response < older_avg_response * 0.95 else 'degrading' if recent_avg_response > older_avg_response * 1.05 else 'stable'\n                trends['response_time'] = response_time_trend
            
            # Cache trends
            if len(self.metrics_history['cache']) >= 10:
                recent_cache = self.metrics_history['cache'][-10:]
                older_cache = self.metrics_history['cache'][-20:-10] if len(self.metrics_history['cache']) >= 20 else self.metrics_history['cache'][:10]
                
                recent_hit_rate = sum(c.get('overall_hit_rate', 0) for c in recent_cache) / len(recent_cache)
                older_hit_rate = sum(c.get('overall_hit_rate', 0) for c in older_cache) / len(older_cache) if older_cache else recent_hit_rate
                
                cache_trend = 'improving' if recent_hit_rate > older_hit_rate * 1.05 else 'degrading' if recent_hit_rate < older_hit_rate * 0.95 else 'stable'\n                trends['cache_efficiency'] = cache_trend
            
            # System resource trends
            if len(self.metrics_history['system_resources']) >= 10:
                recent_system = self.metrics_history['system_resources'][-10:]
                
                avg_cpu = sum(s.get('cpu_usage', 0) for s in recent_system) / len(recent_system)
                avg_memory = sum(s.get('memory_usage', 0) for s in recent_system) / len(recent_system)
                
                trends['resource_utilization'] = {
                    'cpu_avg': round(avg_cpu, 2),
                    'memory_avg': round(avg_memory, 2),
                    'status': 'high' if avg_cpu > 80 or avg_memory > 80 else 'normal'\n                }
            
        except Exception as e:
            self.logger.warning(f\"Failed to calculate trends: {e}\")
            trends['error'] = str(e)
        
        return trends
    
    def export_metrics_report(self, hours: int = 24) -> Dict[str, Any]:
        \"\"\"Export comprehensive metrics report\"\"\"\n        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter metrics by time period
        period_performance = [
            m for m in self.metrics_history['performance']
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        period_cache = [
            m for m in self.metrics_history['cache']
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        period_alerts = [
            m for m in self.metrics_history['alerts']
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        # Calculate summary statistics
        report = {
            'report_period': f'{hours} hours',
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'data_points_collected': len(period_performance),
                'alerts_generated': len(period_alerts),
                'monitoring_uptime_percent': 100 if self.dashboard_active else 0
            }
        }
        
        if period_performance:
            response_times = [p.get('avg_response_time_ms', 0) for p in period_performance]
            report['performance_summary'] = {
                'avg_response_time_ms': round(sum(response_times) / len(response_times), 2),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'sub_200ms_compliance_percent': round((sum(1 for rt in response_times if rt < 200) / len(response_times)) * 100, 2)
            }
        
        if period_cache:
            hit_rates = [c.get('overall_hit_rate', 0) for c in period_cache]
            report['cache_summary'] = {
                'avg_hit_rate_percent': round(sum(hit_rates) / len(hit_rates), 2),
                'min_hit_rate_percent': min(hit_rates),
                'max_hit_rate_percent': max(hit_rates)
            }
        
        if period_alerts:
            alert_types = {}
            for alert in period_alerts:
                metric = alert['metric']
                alert_types[metric] = alert_types.get(metric, 0) + 1
            
            report['alert_summary'] = {
                'total_alerts': len(period_alerts),
                'alert_breakdown': alert_types,
                'most_common_alert': max(alert_types.items(), key=lambda x: x[1])[0] if alert_types else None
            }
        
        return report
    
    def get_real_time_status(self) -> Dict[str, Any]:
        \"\"\"Get real-time system status for quick health checks\"\"\"\n        try:
            health = self.optimizer.health_check()
            dashboard_data = self.get_dashboard_data()
            
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health': health.get('status', 'unknown'),
                'dashboard_active': self.dashboard_active,
                'monitoring_since': self.monitoring_thread.ident if self.monitoring_thread else None,
                'metrics_collected': {
                    'performance_points': len(self.metrics_history['performance']),
                    'cache_points': len(self.metrics_history['cache']),
                    'system_points': len(self.metrics_history['system_resources']),
                    'active_alerts': len([a for a in self.metrics_history['alerts'] 
                                        if datetime.fromisoformat(a['timestamp']) > datetime.utcnow() - timedelta(minutes=15)])
                }
            }
            
            # Add current performance indicators
            if dashboard_data['current_metrics']['performance']:
                perf = dashboard_data['current_metrics']['performance']
                status['current_performance'] = {
                    'response_time_ms': perf.get('avg_response_time_ms', 0),
                    'operations_count': perf.get('total_operations', 0),
                    'phase4_compliant': perf.get('phase4_compliance', {}).get('overall_score', 0) >= 75
                }
            
            if dashboard_data['current_metrics']['cache']:
                cache = dashboard_data['current_metrics']['cache']
                status['current_cache'] = {
                    'hit_rate_percent': cache.get('overall_hit_rate', 0),
                    'total_requests': cache.get('total_requests', 0),
                    'multi_level_active': bool(cache.get('l3_performance', {}).get('requests', 0) > 0)
                }
            
            return status
            
        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'overall_health': 'error',
                'dashboard_active': False
            }\n\n
class DPIBSAnalyticsEngine:
    \"\"\"Advanced analytics engine for DPIBS performance insights\"\"\"\n    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.insights_cache: Dict[str, Any] = {}
        self.anomaly_detection = DPIBSAnomalyDetector()
    
    def update_analytics(self, metrics_history: Dict[str, List[Dict[str, Any]]]) -> None:
        \"\"\"Update analytics based on current metrics history\"\"\"\n        try:
            # Performance pattern analysis
            self._analyze_performance_patterns(metrics_history.get('performance', []))
            
            # Cache optimization analysis
            self._analyze_cache_optimization(metrics_history.get('cache', []))
            
            # Anomaly detection
            self._detect_anomalies(metrics_history)
            
            # Predictive insights
            self._generate_predictive_insights(metrics_history)
            
        except Exception as e:
            self.logger.warning(f\"Analytics update failed: {e}\")
    
    def _analyze_performance_patterns(self, performance_data: List[Dict[str, Any]]) -> None:
        \"\"\"Analyze performance patterns and trends\"\"\"\n        if len(performance_data) < 10:
            return
        
        recent_data = performance_data[-60:]  # Last hour
        response_times = [p.get('avg_response_time_ms', 0) for p in recent_data]
        
        if response_times:
            pattern_analysis = {
                'avg_performance': sum(response_times) / len(response_times),
                'performance_stability': max(response_times) - min(response_times),
                'trend_direction': self._calculate_trend_direction(response_times),
                'peak_periods': self._identify_peak_periods(recent_data)
            }
            
            self.insights_cache['performance_patterns'] = pattern_analysis
    
    def _analyze_cache_optimization(self, cache_data: List[Dict[str, Any]]) -> None:
        \"\"\"Analyze cache performance and optimization opportunities\"\"\"\n        if len(cache_data) < 10:
            return
        
        recent_cache = cache_data[-60:]
        hit_rates = [c.get('overall_hit_rate', 0) for c in recent_cache]
        
        if hit_rates:
            cache_analysis = {
                'avg_hit_rate': sum(hit_rates) / len(hit_rates),
                'hit_rate_stability': max(hit_rates) - min(hit_rates),
                'cache_efficiency_trend': self._calculate_trend_direction(hit_rates),
                'optimization_opportunities': self._identify_cache_optimization_opportunities(recent_cache)
            }
            
            self.insights_cache['cache_optimization'] = cache_analysis
    
    def _detect_anomalies(self, metrics_history: Dict[str, List[Dict[str, Any]]]) -> None:
        \"\"\"Detect performance anomalies\"\"\"\n        anomalies = self.anomaly_detection.detect_anomalies(metrics_history)
        self.insights_cache['anomalies'] = anomalies
    
    def _generate_predictive_insights(self, metrics_history: Dict[str, List[Dict[str, Any]]]) -> None:
        \"\"\"Generate predictive insights for future performance\"\"\"\n        try:
            predictions = {
                'performance_forecast': self._predict_performance_trend(metrics_history.get('performance', [])),
                'capacity_warnings': self._predict_capacity_issues(metrics_history.get('system_resources', [])),
                'optimization_recommendations': self._generate_optimization_recommendations()
            }
            
            self.insights_cache['predictions'] = predictions
            
        except Exception as e:
            self.logger.warning(f\"Predictive analytics failed: {e}\")
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        \"\"\"Calculate trend direction from series of values\"\"\"\n        if len(values) < 5:
            return 'insufficient_data'\n        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if change_percent > 5:
            return 'increasing'\n        elif change_percent < -5:
            return 'decreasing'\n        else:
            return 'stable'\n    
    def _identify_peak_periods(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Identify peak performance periods\"\"\"\n        peaks = []
        
        for i, data_point in enumerate(performance_data):
            if i < 2 or i > len(performance_data) - 3:
                continue
            
            current_response = data_point.get('avg_response_time_ms', 0)
            prev_response = performance_data[i-1].get('avg_response_time_ms', 0)
            next_response = performance_data[i+1].get('avg_response_time_ms', 0)
            
            # Peak if significantly higher than neighbors
            if current_response > prev_response * 1.5 and current_response > next_response * 1.5:
                peaks.append({
                    'timestamp': data_point['timestamp'],
                    'response_time_ms': current_response,
                    'severity': 'high' if current_response > 500 else 'medium'\n                })
        
        return peaks
    
    def _identify_cache_optimization_opportunities(self, cache_data: List[Dict[str, Any]]) -> List[str]:
        \"\"\"Identify cache optimization opportunities\"\"\"\n        opportunities = []
        
        if not cache_data:
            return opportunities
        
        latest_cache = cache_data[-1]
        
        # Check L1 cache efficiency
        l1_hit_rate = latest_cache.get('l1_performance', {}).get('hit_rate_percent', 0)
        if l1_hit_rate < 80:
            opportunities.append(f\"L1 cache hit rate at {l1_hit_rate:.1f}% - consider increasing cache size or TTL\")
        
        # Check L3 cache utilization
        l3_stats = latest_cache.get('storage_stats', {}).get('l3_stats', {})
        if isinstance(l3_stats, dict) and l3_stats.get('entry_count', 0) > 0:
            l3_hit_rate = latest_cache.get('l3_performance', {}).get('hit_rate_percent', 0)
            if l3_hit_rate < 50:
                opportunities.append(f\"L3 persistent cache underperforming - review caching strategy\")
        
        # Check overall cache performance trend
        if len(cache_data) >= 10:
            recent_hit_rates = [c.get('overall_hit_rate', 0) for c in cache_data[-10:]]
            if all(rate < 70 for rate in recent_hit_rates):
                opportunities.append(\"Consistently low cache hit rates - review caching patterns\")
        
        return opportunities
    
    def _predict_performance_trend(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"Simple performance trend prediction\"\"\"\n        if len(performance_data) < 20:
            return {'status': 'insufficient_data'}
        
        recent_response_times = [p.get('avg_response_time_ms', 0) for p in performance_data[-20:]]
        trend = self._calculate_trend_direction(recent_response_times)
        
        prediction = {
            'trend': trend,
            'current_avg': sum(recent_response_times) / len(recent_response_times),
            'confidence': 'medium' if len(recent_response_times) >= 15 else 'low'\n        }
        
        if trend == 'increasing':
            prediction['warning'] = 'Performance degradation trend detected'\n        elif trend == 'decreasing':
            prediction['note'] = 'Performance improvement trend detected'\n        
        return prediction
    
    def _predict_capacity_issues(self, system_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Predict potential capacity issues\"\"\"\n        warnings = []
        
        if len(system_data) < 10:
            return warnings
        
        recent_system = system_data[-10:]
        
        # Check CPU trend
        cpu_values = [s.get('cpu_usage', 0) for s in recent_system]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        
        if avg_cpu > 70:
            warnings.append({
                'type': 'cpu_capacity',
                'current_usage': avg_cpu,
                'severity': 'high' if avg_cpu > 85 else 'medium',
                'recommendation': 'Consider scaling up or optimizing CPU-intensive operations'\n            })
        
        # Check memory trend
        memory_values = [s.get('memory_usage', 0) for s in recent_system]
        avg_memory = sum(memory_values) / len(memory_values)
        
        if avg_memory > 75:
            warnings.append({
                'type': 'memory_capacity',
                'current_usage': avg_memory,
                'severity': 'high' if avg_memory > 90 else 'medium',
                'recommendation': 'Consider increasing memory or reducing cache sizes'\n            })
        
        return warnings
    
    def _generate_optimization_recommendations(self) -> List[str]:
        \"\"\"Generate general optimization recommendations based on insights\"\"\"\n        recommendations = []
        
        # Performance-based recommendations
        perf_patterns = self.insights_cache.get('performance_patterns', {})
        if perf_patterns.get('avg_performance', 0) > 200:
            recommendations.append(\"Average response time exceeds 200ms target - implement caching optimizations\")
        
        # Cache-based recommendations
        cache_opt = self.insights_cache.get('cache_optimization', {})
        if cache_opt.get('avg_hit_rate', 100) < 70:
            recommendations.append(\"Cache hit rate below optimal - review cache sizing and TTL settings\")
        
        # Anomaly-based recommendations
        anomalies = self.insights_cache.get('anomalies', {})
        if anomalies.get('performance_anomalies', []):
            recommendations.append(\"Performance anomalies detected - investigate unusual patterns\")
        
        return recommendations
    
    def get_current_insights(self) -> Dict[str, Any]:
        \"\"\"Get current analytics insights\"\"\"\n        return {
            'timestamp': datetime.utcnow().isoformat(),
            'insights': self.insights_cache,
            'recommendations': self.insights_cache.get('predictions', {}).get('optimization_recommendations', [])
        }\n\n
class DPIBSAnomalyDetector:
    \"\"\"Simple anomaly detection for DPIBS performance metrics\"\"\"\n    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, metrics_history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        \"\"\"Detect anomalies in metrics data\"\"\"\n        anomalies = {
            'performance_anomalies': [],
            'cache_anomalies': [],
            'system_anomalies': []
        }
        
        try:
            # Performance anomalies
            performance_data = metrics_history.get('performance', [])
            if len(performance_data) >= 20:
                perf_anomalies = self._detect_performance_anomalies(performance_data)
                anomalies['performance_anomalies'] = perf_anomalies
            
            # Cache anomalies
            cache_data = metrics_history.get('cache', [])
            if len(cache_data) >= 20:
                cache_anomalies = self._detect_cache_anomalies(cache_data)
                anomalies['cache_anomalies'] = cache_anomalies
            
            # System resource anomalies
            system_data = metrics_history.get('system_resources', [])
            if len(system_data) >= 20:
                system_anomalies = self._detect_system_anomalies(system_data)
                anomalies['system_anomalies'] = system_anomalies
                
        except Exception as e:
            self.logger.warning(f\"Anomaly detection failed: {e}\")
            anomalies['error'] = str(e)
        
        return anomalies
    
    def _detect_performance_anomalies(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Detect performance anomalies using simple statistical methods\"\"\"\n        anomalies = []
        response_times = [p.get('avg_response_time_ms', 0) for p in performance_data[-20:]]
        
        # Calculate baseline statistics
        avg_response = sum(response_times) / len(response_times)
        variance = sum((x - avg_response) ** 2 for x in response_times) / len(response_times)
        std_dev = variance ** 0.5
        
        # Detect outliers (values beyond 2 standard deviations)
        threshold = avg_response + (2 * std_dev)
        
        for i, data_point in enumerate(performance_data[-10:]):
            response_time = data_point.get('avg_response_time_ms', 0)
            if response_time > threshold and response_time > avg_response * 1.5:
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': 'response_time',
                    'value': response_time,
                    'expected_range': f'<{threshold:.1f}ms',
                    'severity': 'high' if response_time > threshold * 1.5 else 'medium'\n                })
        
        return anomalies
    
    def _detect_cache_anomalies(self, cache_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Detect cache performance anomalies\"\"\"\n        anomalies = []
        hit_rates = [c.get('overall_hit_rate', 0) for c in cache_data[-20:]]
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        for i, data_point in enumerate(cache_data[-10:]):
            hit_rate = data_point.get('overall_hit_rate', 0)
            
            # Sudden drop in cache performance
            if hit_rate < avg_hit_rate * 0.7 and avg_hit_rate > 50:
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': 'cache_hit_rate',
                    'value': hit_rate,
                    'expected_range': f'>{avg_hit_rate * 0.8:.1f}%',
                    'severity': 'medium'\n                })
        
        return anomalies
    
    def _detect_system_anomalies(self, system_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Detect system resource anomalies\"\"\"\n        anomalies = []
        
        # Check for sudden spikes in resource usage
        for i, data_point in enumerate(system_data[-10:]):
            cpu_usage = data_point.get('cpu_usage', 0)
            memory_usage = data_point.get('memory_usage', 0)
            
            if cpu_usage > 90:
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': 'cpu_usage',
                    'value': cpu_usage,
                    'severity': 'high',
                    'note': 'Critical CPU utilization detected'\n                })
            
            if memory_usage > 95:
                anomalies.append({
                    'timestamp': data_point['timestamp'],
                    'metric': 'memory_usage',
                    'value': memory_usage,
                    'severity': 'critical',
                    'note': 'Critical memory utilization detected'\n                })
        
        return anomalies\n\n
# ============================================================================
# MAIN EXECUTION AND FACTORY FUNCTIONS
# ============================================================================\n
def create_monitoring_dashboard(optimizer: DPIBSPerformanceOptimizer) -> DPIBSMonitoringDashboard:
    \"\"\"Factory function to create DPIBS monitoring dashboard\"\"\"\n    return DPIBSMonitoringDashboard(optimizer)\n\n
if __name__ == \"__main__\":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n    )
    
    # Example usage
    from knowledge.database.database_config import DatabaseConfig
    
    # Create optimizer and dashboard
    optimizer = DPIBSPerformanceOptimizer(DatabaseConfig())
    dashboard = create_monitoring_dashboard(optimizer)
    
    try:
        print(\"🚀 Starting DPIBS Monitoring Dashboard...\")
        dashboard.start_monitoring()
        
        # Run for demonstration
        time.sleep(30)
        
        # Get current status
        status = dashboard.get_real_time_status()
        print(f\"📊 Dashboard Status: {status['overall_health']}\")
        print(f\"📈 Metrics Collected: {status['metrics_collected']}\")
        
        # Generate report
        report = dashboard.export_metrics_report(hours=1)
        print(f\"📋 Performance Report: {report.get('performance_summary', 'No data')}\")
        
    except KeyboardInterrupt:
        print(\"\\n🛑 Stopping monitoring dashboard...\")
    finally:
        dashboard.stop_monitoring()
        print(\"✅ Dashboard stopped\")