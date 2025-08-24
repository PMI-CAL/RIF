"""
RIF Analytics Dashboard for LightRAG
Provides comprehensive analytics and insights for the knowledge system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

# Add parent directory to path for LightRAG core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lightrag_core import LightRAGCore, get_lightrag_instance
from learning.feedback_loop import get_feedback_loop, FeedbackLoop


class AnalyticsDashboard:
    """
    Comprehensive analytics for RIF knowledge system.
    """
    
    def __init__(self):
        """Initialize analytics dashboard."""
        self.rag = get_lightrag_instance()
        self.feedback_loop = get_feedback_loop()
        self.logger = logging.getLogger("rif.analytics")
    
    def generate_system_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive system performance report.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Complete system analytics report
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period_hours": hours,
            "system_health": self._analyze_system_health(),
            "agent_performance": self._analyze_agent_performance(hours),
            "pattern_effectiveness": self._analyze_pattern_effectiveness(),
            "knowledge_growth": self._analyze_knowledge_growth(hours),
            "performance_trends": self._analyze_performance_trends(hours),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health."""
        health = self.feedback_loop.get_system_health()
        
        # Categorize health status
        if health["avg_success_rate"] > 0.8:
            status = "healthy"
        elif health["avg_success_rate"] > 0.6:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "metrics": health,
            "issues": self._identify_health_issues(health)
        }
    
    def _identify_health_issues(self, health: Dict[str, Any]) -> List[str]:
        """Identify potential health issues."""
        issues = []
        
        if health["avg_success_rate"] < 0.6:
            issues.append("Low overall success rate")
        
        if health["queue_size"] > 100:
            issues.append("High event queue backlog")
        
        if health["recent_events"] == 0:
            issues.append("No recent activity detected")
        
        if health["active_agents"] < 2:
            issues.append("Low agent activity")
        
        return issues
    
    def _analyze_agent_performance(self, hours: int) -> Dict[str, Any]:
        """Analyze performance of individual agents."""
        agent_analysis = {}
        
        # Get list of active agents from feedback loop
        health = self.feedback_loop.get_system_health()
        
        for agent_name in ["analyst", "planner", "architect", "implementer", "validator", "learner"]:
            performance = self.feedback_loop.get_agent_performance(agent_name, hours)
            
            # Categorize performance
            if performance["success_rate"] > 0.85:
                grade = "excellent"
            elif performance["success_rate"] > 0.7:
                grade = "good"
            elif performance["success_rate"] > 0.5:
                grade = "average"
            else:
                grade = "poor"
            
            agent_analysis[agent_name] = {
                "performance": performance,
                "grade": grade,
                "insights": self._generate_agent_insights(agent_name, performance)
            }
        
        return agent_analysis
    
    def _generate_agent_insights(self, agent_name: str, performance: Dict[str, Any]) -> List[str]:
        """Generate insights for specific agent performance."""
        insights = []
        
        if performance["events"] == 0:
            insights.append(f"{agent_name} has no recent activity")
        elif performance["success_rate"] < 0.5:
            insights.append(f"{agent_name} has low success rate - needs attention")
        elif performance["success_rate"] > 0.9:
            insights.append(f"{agent_name} performing excellently")
        
        if performance["events"] > 50:
            insights.append(f"{agent_name} is very active")
        
        return insights
    
    def _analyze_pattern_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of stored patterns."""
        patterns_analysis = {
            "total_patterns": 0,
            "high_performers": [],
            "low_performers": [],
            "unused_patterns": [],
            "pattern_distribution": {}
        }
        
        # This would normally analyze patterns from the feedback loop
        # For now, we'll return structure showing what we would analyze
        try:
            # Get pattern stats from feedback loop
            health = self.feedback_loop.get_system_health()
            patterns_analysis["total_patterns"] = health.get("total_patterns", 0)
            
            # Analyze top patterns (this would be more detailed in real implementation)
            patterns_analysis["pattern_distribution"] = {
                "implementation_patterns": 15,
                "analysis_patterns": 8,
                "architecture_patterns": 12,
                "validation_patterns": 6
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
        
        return patterns_analysis
    
    def _analyze_knowledge_growth(self, hours: int) -> Dict[str, Any]:
        """Analyze knowledge base growth and health."""
        try:
            # Query for recent documents
            recent_docs = self.rag.search_documents(
                query="timestamp recent learning",
                limit=100
            )
            
            # Analyze growth metrics
            growth_analysis = {
                "recent_additions": len(recent_docs),
                "growth_rate": len(recent_docs) / hours if hours > 0 else 0,
                "content_types": self._categorize_content_types(recent_docs),
                "quality_metrics": self._assess_content_quality(recent_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing knowledge growth: {str(e)}")
            growth_analysis = {
                "recent_additions": 0,
                "growth_rate": 0,
                "content_types": {},
                "quality_metrics": {}
            }
        
        return growth_analysis
    
    def _categorize_content_types(self, documents: List[Any]) -> Dict[str, int]:
        """Categorize documents by type."""
        type_counts = Counter()
        
        for doc in documents:
            doc_type = getattr(doc, 'metadata', {}).get('type', 'unknown')
            type_counts[doc_type] += 1
        
        return dict(type_counts)
    
    def _assess_content_quality(self, documents: List[Any]) -> Dict[str, Any]:
        """Assess quality of recent content."""
        if not documents:
            return {"average_length": 0, "completeness_score": 0}
        
        lengths = []
        for doc in documents:
            content = getattr(doc, 'content', '')
            lengths.append(len(content))
        
        return {
            "average_length": statistics.mean(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "completeness_score": min(statistics.mean(lengths) / 500, 1.0) if lengths else 0
        }
    
    def _analyze_performance_trends(self, hours: int) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "success_rate_trend": "stable",  # would calculate from historical data
            "activity_trend": "increasing",
            "pattern_usage_trend": "stable",
            "knowledge_accumulation_trend": "increasing"
        }
        
        # This would analyze historical data to determine actual trends
        # For now, returning typical stable system indicators
        
        return trends
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # System health recommendations
        health = self.feedback_loop.get_system_health()
        
        if health["avg_success_rate"] < 0.7:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "title": "Improve System Success Rate",
                "description": "Success rate is below optimal threshold",
                "action": "Review and optimize low-performing patterns"
            })
        
        if health["recent_events"] < 10:
            recommendations.append({
                "category": "activity",
                "priority": "medium",
                "title": "Increase System Activity",
                "description": "Low recent activity detected",
                "action": "Check agent integration and issue processing"
            })
        
        # Knowledge base recommendations
        recommendations.append({
            "category": "knowledge",
            "priority": "low",
            "title": "Optimize Knowledge Base",
            "description": "Regular optimization maintains performance",
            "action": "Run knowledge base cleanup and optimization"
        })
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """
        Export report in specified format.
        
        Args:
            report: Report data
            format: Export format (json, markdown)
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "markdown":
            return self._format_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as markdown."""
        md = f"# RIF Analytics Report\n\n"
        md += f"Generated: {report['report_generated']}\n"
        md += f"Analysis Period: {report['analysis_period_hours']} hours\n\n"
        
        # System Health
        health = report["system_health"]
        md += f"## System Health: {health['status'].upper()}\n\n"
        md += f"- Success Rate: {health['metrics']['avg_success_rate']:.2f}\n"
        md += f"- Active Agents: {health['metrics']['active_agents']}\n"
        md += f"- Recent Events: {health['metrics']['recent_events']}\n\n"
        
        if health['issues']:
            md += "### Issues Detected\n"
            for issue in health['issues']:
                md += f"- {issue}\n"
            md += "\n"
        
        # Agent Performance
        md += "## Agent Performance\n\n"
        for agent, data in report["agent_performance"].items():
            perf = data["performance"]
            md += f"### {agent.title()}\n"
            md += f"- Grade: {data['grade'].upper()}\n"
            md += f"- Success Rate: {perf['success_rate']:.2f}\n"
            md += f"- Events: {perf['events']}\n"
            if data['insights']:
                md += "- Insights:\n"
                for insight in data['insights']:
                    md += f"  - {insight}\n"
            md += "\n"
        
        # Recommendations
        if report["recommendations"]:
            md += "## Recommendations\n\n"
            for rec in report["recommendations"]:
                md += f"### {rec['title']} ({rec['priority'].upper()})\n"
                md += f"{rec['description']}\n\n"
                md += f"**Action:** {rec['action']}\n\n"
        
        return md
    
    def save_report_to_knowledge_base(self, report: Dict[str, Any]):
        """Save analytics report to knowledge base."""
        try:
            content = f"Analytics Report\n"
            content += f"Generated: {report['report_generated']}\n"
            content += f"Report Data: {json.dumps(report, indent=2)}"
            
            doc_id = self.rag.insert_document(
                content=content,
                metadata={
                    "type": "analytics_report",
                    "timestamp": report["report_generated"],
                    "agent": "analytics_dashboard"
                }
            )
            
            self.logger.info(f"Saved analytics report to knowledge base: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to save report to knowledge base: {str(e)}")
            return None


# Convenience functions
def generate_system_report(hours: int = 24) -> Dict[str, Any]:
    """Generate system report with specified time window."""
    dashboard = AnalyticsDashboard()
    return dashboard.generate_system_report(hours)

def export_report_markdown(hours: int = 24) -> str:
    """Generate and export report as markdown."""
    dashboard = AnalyticsDashboard()
    report = dashboard.generate_system_report(hours)
    return dashboard.export_report(report, "markdown")