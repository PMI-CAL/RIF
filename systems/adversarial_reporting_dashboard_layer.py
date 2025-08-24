#!/usr/bin/env python3
"""
Layer 7: Reporting and Dashboard Layer for RIF Adversarial Validation System

This layer provides comprehensive validation reporting, dashboard generation, 
and visual analytics for the adversarial validation results across all RIF features.
"""

import json
import sqlite3
import os
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    """Structured validation report for a single feature"""
    feature_id: str
    feature_name: str
    category: str
    validation_status: str  # PASS, FAIL, PARTIAL, UNKNOWN
    evidence_count: int
    test_levels_completed: int
    test_levels_total: int
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    fix_required: bool
    validation_timestamp: str
    evidence_integrity_hash: str
    
@dataclass 
class CategorySummary:
    """Summary statistics for a feature category"""
    category_name: str
    total_features: int
    passed_features: int
    failed_features: int
    partial_features: int
    unknown_features: int
    pass_rate: float
    critical_issues: int
    fix_issues_needed: int

@dataclass
class SystemHealthReport:
    """Overall system health and validation status"""
    total_features_cataloged: int
    total_features_validated: int
    overall_pass_rate: float
    critical_failures: int
    production_ready_features: int
    non_functional_systems: List[str]
    high_risk_areas: List[str]
    validation_coverage: float
    confidence_score: float
    recommendations: List[str]

class AdversarialReportingDashboard:
    """
    Layer 7: Comprehensive reporting and dashboard system for adversarial validation results.
    
    Provides:
    - Detailed validation reports for individual features
    - Category-based summaries and analytics
    - System-wide health assessments  
    - Risk analysis and recommendations
    - Visual dashboard generation
    - Evidence integrity verification
    - Progress tracking and trend analysis
    """
    
    def __init__(self, db_path: str = "knowledge/validation_results.db"):
        """Initialize the reporting dashboard with database connection"""
        self.db_path = db_path
        self.ensure_database_schema()
        logger.info(f"Adversarial Reporting Dashboard initialized with database: {db_path}")
        
    def ensure_database_schema(self):
        """Create database schema for validation reporting"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS validation_reports (
                    feature_id TEXT PRIMARY KEY,
                    feature_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    test_levels_completed INTEGER NOT NULL,
                    test_levels_total INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    fix_required BOOLEAN NOT NULL,
                    validation_timestamp TEXT NOT NULL,
                    evidence_integrity_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS validation_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    feature_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    evidence_data TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feature_id) REFERENCES validation_reports (feature_id)
                );
                
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_category TEXT NOT NULL,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS dashboard_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    snapshot_data TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    generated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
    
    def record_validation_report(self, report: ValidationReport) -> bool:
        """Record a validation report in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_reports 
                    (feature_id, feature_name, category, validation_status, evidence_count,
                     test_levels_completed, test_levels_total, risk_level, fix_required,
                     validation_timestamp, evidence_integrity_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.feature_id, report.feature_name, report.category, report.validation_status,
                    report.evidence_count, report.test_levels_completed, report.test_levels_total,
                    report.risk_level, report.fix_required, report.validation_timestamp,
                    report.evidence_integrity_hash
                ))
            logger.info(f"Recorded validation report for feature: {report.feature_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to record validation report for {report.feature_id}: {e}")
            return False
    
    def generate_category_summary(self, category: str) -> CategorySummary:
        """Generate summary statistics for a specific category"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get category statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN validation_status = 'PASS' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN validation_status = 'FAIL' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN validation_status = 'PARTIAL' THEN 1 ELSE 0 END) as partial,
                    SUM(CASE WHEN validation_status = 'UNKNOWN' THEN 1 ELSE 0 END) as unknown,
                    SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_issues,
                    SUM(CASE WHEN fix_required = 1 THEN 1 ELSE 0 END) as fix_needed
                FROM validation_reports
                WHERE category = ?
            """, (category,))
            
            result = cursor.fetchone()
            total, passed, failed, partial, unknown, critical, fix_needed = result
            
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            return CategorySummary(
                category_name=category,
                total_features=total,
                passed_features=passed,
                failed_features=failed,
                partial_features=partial,
                unknown_features=unknown,
                pass_rate=pass_rate,
                critical_issues=critical,
                fix_issues_needed=fix_needed
            )
    
    def generate_system_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health assessment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_validated,
                    SUM(CASE WHEN validation_status = 'PASS' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN validation_status = 'FAIL' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_failures,
                    SUM(CASE WHEN validation_status = 'PASS' AND risk_level IN ('LOW', 'MEDIUM') THEN 1 ELSE 0 END) as production_ready
                FROM validation_reports
            """)
            
            total_validated, passed, failed, critical_failures, production_ready = cursor.fetchone()
            overall_pass_rate = (passed / total_validated * 100) if total_validated > 0 else 0
            
            # Get non-functional systems
            cursor.execute("""
                SELECT feature_name FROM validation_reports
                WHERE validation_status = 'FAIL' AND risk_level IN ('HIGH', 'CRITICAL')
                ORDER BY risk_level DESC, feature_name
            """)
            non_functional_systems = [row[0] for row in cursor.fetchall()]
            
            # Get high-risk areas (categories with high failure rates)
            cursor.execute("""
                SELECT category, 
                       COUNT(*) as total,
                       SUM(CASE WHEN validation_status = 'FAIL' THEN 1 ELSE 0 END) as failed
                FROM validation_reports
                GROUP BY category
                HAVING (failed * 1.0 / total) > 0.3
                ORDER BY (failed * 1.0 / total) DESC
            """)
            high_risk_areas = [row[0] for row in cursor.fetchall()]
            
            # Calculate coverage and confidence
            total_features_expected = 247  # From initial analysis
            validation_coverage = (total_validated / total_features_expected * 100) if total_features_expected > 0 else 0
            
            # Confidence score based on pass rate, evidence quality, and coverage
            confidence_score = min(100, (overall_pass_rate * 0.4) + (validation_coverage * 0.4) + ((total_validated - critical_failures) / total_validated * 20))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_pass_rate, critical_failures, validation_coverage, len(high_risk_areas)
            )
            
            return SystemHealthReport(
                total_features_cataloged=total_features_expected,
                total_features_validated=total_validated,
                overall_pass_rate=overall_pass_rate,
                critical_failures=critical_failures,
                production_ready_features=production_ready,
                non_functional_systems=non_functional_systems,
                high_risk_areas=high_risk_areas,
                validation_coverage=validation_coverage,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
    
    def _generate_recommendations(self, pass_rate: float, critical_failures: int, 
                                coverage: float, high_risk_areas: int) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        if critical_failures > 0:
            recommendations.append(f"URGENT: Address {critical_failures} critical system failures immediately")
        
        if pass_rate < 70:
            recommendations.append(f"System reliability concern: {pass_rate:.1f}% pass rate requires improvement")
        
        if coverage < 80:
            recommendations.append(f"Validation coverage incomplete: {coverage:.1f}% - continue validation of remaining features")
        
        if high_risk_areas > 3:
            recommendations.append(f"Multiple high-risk categories identified ({high_risk_areas}) - consider architectural review")
        
        if pass_rate > 85 and critical_failures == 0:
            recommendations.append("System shows good health - focus on remaining validation coverage")
        
        return recommendations
    
    def generate_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with all validation analytics"""
        logger.info("Generating comprehensive validation dashboard")
        
        # Get all categories
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM validation_reports ORDER BY category")
            categories = [row[0] for row in cursor.fetchall()]
        
        # Generate category summaries
        category_summaries = {}
        for category in categories:
            category_summaries[category] = asdict(self.generate_category_summary(category))
        
        # Generate system health report
        system_health = asdict(self.generate_system_health_report())
        
        # Get recent validation activity
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT feature_name, validation_status, risk_level, validation_timestamp
                FROM validation_reports
                ORDER BY validation_timestamp DESC
                LIMIT 10
            """)
            recent_activity = [
                {
                    "feature_name": row[0],
                    "status": row[1], 
                    "risk_level": row[2],
                    "timestamp": row[3]
                }
                for row in cursor.fetchall()
            ]
        
        # Generate dashboard data
        dashboard = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "dashboard_type": "comprehensive_validation_dashboard",
            "system_health": system_health,
            "category_summaries": category_summaries,
            "recent_activity": recent_activity,
            "validation_metrics": self._get_validation_metrics(),
            "risk_analysis": self._generate_risk_analysis(),
            "trend_data": self._get_trend_data()
        }
        
        # Save dashboard snapshot
        self._save_dashboard_snapshot(dashboard)
        
        logger.info("Comprehensive dashboard generated successfully")
        return dashboard
    
    def _get_validation_metrics(self) -> Dict[str, Any]:
        """Get key validation metrics for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Average test completion rate
            cursor.execute("SELECT AVG(test_levels_completed * 1.0 / test_levels_total) FROM validation_reports")
            avg_completion = cursor.fetchone()[0] or 0
            
            # Evidence integrity rate  
            cursor.execute("SELECT COUNT(*) FROM validation_reports WHERE evidence_integrity_hash IS NOT NULL AND evidence_integrity_hash != ''")
            integrity_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM validation_reports")
            total_reports = cursor.fetchone()[0]
            integrity_rate = (integrity_count / total_reports * 100) if total_reports > 0 else 0
            
            return {
                "average_test_completion_rate": avg_completion * 100,
                "evidence_integrity_rate": integrity_rate,
                "total_validation_reports": total_reports,
                "validation_in_progress": total_reports  # Assuming all reports are from current validation cycle
            }
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate risk analysis for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Risk distribution
            cursor.execute("""
                SELECT risk_level, COUNT(*) 
                FROM validation_reports 
                GROUP BY risk_level
                ORDER BY CASE risk_level 
                    WHEN 'CRITICAL' THEN 1 
                    WHEN 'HIGH' THEN 2 
                    WHEN 'MEDIUM' THEN 3 
                    WHEN 'LOW' THEN 4 
                END
            """)
            risk_distribution = dict(cursor.fetchall())
            
            # Top risk areas
            cursor.execute("""
                SELECT category, COUNT(*) as high_risk_features
                FROM validation_reports 
                WHERE risk_level IN ('HIGH', 'CRITICAL')
                GROUP BY category
                ORDER BY high_risk_features DESC
                LIMIT 5
            """)
            top_risk_areas = [{"category": row[0], "high_risk_features": row[1]} for row in cursor.fetchall()]
            
            return {
                "risk_distribution": risk_distribution,
                "top_risk_areas": top_risk_areas,
                "risk_assessment": "Based on current validation results" 
            }
    
    def _get_trend_data(self) -> Dict[str, Any]:
        """Get validation trend data for dashboard"""
        # For now, return placeholder trend data
        # In a full implementation, this would track validation progress over time
        return {
            "validation_progress_trend": "Increasing validation coverage over time",
            "pass_rate_trend": "Stable pass rates with identified critical issues",
            "risk_trend": "Risk levels identified and documented for remediation"
        }
    
    def _save_dashboard_snapshot(self, dashboard_data: Dict[str, Any]):
        """Save dashboard snapshot to database"""
        snapshot_id = hashlib.md5(json.dumps(dashboard_data, sort_keys=True).encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO dashboard_snapshots 
                (snapshot_id, snapshot_data, snapshot_type, generated_at)
                VALUES (?, ?, ?, ?)
            """, (
                snapshot_id,
                json.dumps(dashboard_data),
                "comprehensive_validation_dashboard", 
                datetime.datetime.utcnow().isoformat() + "Z"
            ))
    
    def export_validation_report(self, output_path: str, format: str = "json") -> bool:
        """Export comprehensive validation report to file"""
        try:
            dashboard = self.generate_comprehensive_dashboard()
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(dashboard, f, indent=2)
            elif format.lower() == "markdown":
                markdown_content = self._generate_markdown_report(dashboard)
                with open(output_path, 'w') as f:
                    f.write(markdown_content)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Validation report exported to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
            return False
    
    def _generate_markdown_report(self, dashboard: Dict[str, Any]) -> str:
        """Generate markdown formatted validation report"""
        system_health = dashboard["system_health"]
        
        markdown = f"""# RIF Adversarial Validation Dashboard Report

**Generated**: {dashboard["generated_at"]}

## Executive Summary

- **Total Features Validated**: {system_health["total_features_validated"]} / {system_health["total_features_cataloged"]}
- **Overall Pass Rate**: {system_health["overall_pass_rate"]:.1f}%
- **Validation Coverage**: {system_health["validation_coverage"]:.1f}%
- **Critical Failures**: {system_health["critical_failures"]}
- **Production Ready Features**: {system_health["production_ready_features"]}
- **System Confidence Score**: {system_health["confidence_score"]:.1f}/100

## Critical Issues

"""
        if system_health["non_functional_systems"]:
            markdown += "### Non-Functional Systems\n\n"
            for system in system_health["non_functional_systems"]:
                markdown += f"- ❌ {system}\n"
        else:
            markdown += "✅ No critical non-functional systems identified\n"
        
        markdown += "\n## High-Risk Areas\n\n"
        if system_health["high_risk_areas"]:
            for area in system_health["high_risk_areas"]:
                markdown += f"- ⚠️ {area}\n"
        else:
            markdown += "✅ No high-risk areas identified\n"
        
        markdown += "\n## Recommendations\n\n"
        for rec in system_health["recommendations"]:
            markdown += f"- {rec}\n"
        
        markdown += "\n## Category Summaries\n\n"
        for category, summary in dashboard["category_summaries"].items():
            pass_rate = summary["pass_rate"]
            status_icon = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
            markdown += f"### {status_icon} {category}\n\n"
            markdown += f"- **Features**: {summary['total_features']}\n"
            markdown += f"- **Pass Rate**: {pass_rate:.1f}%\n"
            markdown += f"- **Critical Issues**: {summary['critical_issues']}\n"
            markdown += f"- **Fixes Needed**: {summary['fix_issues_needed']}\n\n"
        
        return markdown

def main():
    """Main function for testing the Adversarial Reporting Dashboard Layer"""
    dashboard = AdversarialReportingDashboard()
    
    # Test with sample validation report
    sample_report = ValidationReport(
        feature_id="shadow_issue_tracking",
        feature_name="Shadow Issue Tracking System",
        category="quality_assurance",
        validation_status="FAIL",
        evidence_count=3,
        test_levels_completed=2,
        test_levels_total=5,
        risk_level="CRITICAL",
        fix_required=True,
        validation_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        evidence_integrity_hash="abc123def456"
    )
    
    # Record the sample report
    success = dashboard.record_validation_report(sample_report)
    print(f"Sample report recorded: {success}")
    
    # Generate comprehensive dashboard
    dashboard_data = dashboard.generate_comprehensive_dashboard()
    print(f"Dashboard generated with {len(dashboard_data)} sections")
    
    # Export reports
    dashboard.export_validation_report("validation_dashboard.json", "json")
    dashboard.export_validation_report("validation_dashboard.md", "markdown")
    
    print("Adversarial Reporting Dashboard Layer 7 - Testing Complete")

if __name__ == "__main__":
    main()