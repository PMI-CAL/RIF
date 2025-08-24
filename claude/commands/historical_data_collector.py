#!/usr/bin/env python3
"""
Historical Data Collector for Adaptive Threshold Learning System
Issue #95: Adaptive Threshold Learning System

Collects and manages historical quality data for rule-based threshold optimization.
Uses file-based storage compatible with Claude Code session-based architecture.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import yaml

@dataclass
class QualityDecision:
    """Record of a quality gate decision."""
    timestamp: str
    issue_number: Optional[int]
    component_type: str
    threshold_used: float
    quality_score: float
    decision: str  # "pass", "fail", "manual_override"
    context: Dict[str, Any]
    outcome: Optional[str] = None  # "success", "defect_found", "false_positive"
    
@dataclass 
class ThresholdPerformance:
    """Track threshold effectiveness over time."""
    timestamp: str
    component_type: str
    threshold: float
    success_rate: float
    false_positive_rate: float
    false_negative_rate: float
    sample_size: int
    time_period_days: int

@dataclass
class TeamMetrics:
    """Team performance indicators."""
    timestamp: str
    team_id: str
    quality_trend: float  # Improvement rate over time
    average_threshold_success: float
    code_complexity_trend: float
    defect_resolution_time: float
    testing_coverage_trend: float

@dataclass
class ProjectCharacteristics:
    """Project context and complexity data."""
    timestamp: str
    project_name: str
    primary_language: str
    complexity_score: float
    team_size: int
    domain: str  # "web", "mobile", "backend", "data", etc.
    risk_level: str  # "low", "medium", "high", "critical"

class HistoricalDataCollector:
    """
    Collects and manages historical quality data for adaptive threshold learning.
    
    Features:
    - File-based storage compatible with Claude Code architecture
    - JSON Lines format for efficient append operations
    - Data validation and consistency checks
    - Automatic data aggregation and trend analysis
    - Integration with existing RIF knowledge base patterns
    """
    
    def __init__(self, quality_data_dir: str = "quality/historical"):
        """
        Initialize the historical data collector.
        
        Args:
            quality_data_dir: Directory for storing historical quality data
        """
        self.quality_data_dir = Path(quality_data_dir)
        self.quality_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data file paths
        self.quality_decisions_file = self.quality_data_dir / "quality_decisions.jsonl"
        self.threshold_performance_file = self.quality_data_dir / "threshold_performance.jsonl"
        self.team_metrics_file = self.quality_data_dir / "team_metrics.jsonl"
        self.project_characteristics_file = self.quality_data_dir / "project_characteristics.jsonl"
        
        self.setup_logging()
        self._initialize_data_files()
        
    def setup_logging(self):
        """Setup logging for historical data collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - HistoricalDataCollector - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_data_files(self):
        """Initialize data files if they don't exist."""
        for file_path in [self.quality_decisions_file, self.threshold_performance_file, 
                         self.team_metrics_file, self.project_characteristics_file]:
            if not file_path.exists():
                file_path.touch()
                self.logger.info(f"Initialized data file: {file_path}")
    
    def record_quality_decision(self, 
                              component_type: str,
                              threshold_used: float,
                              quality_score: float,
                              decision: str,
                              context: Dict[str, Any],
                              issue_number: Optional[int] = None,
                              outcome: Optional[str] = None) -> bool:
        """
        Record a quality gate decision for learning.
        
        Args:
            component_type: Type of component (e.g., "critical_algorithms")
            threshold_used: Quality threshold that was applied
            quality_score: Actual quality score achieved
            decision: Gate decision ("pass", "fail", "manual_override")
            context: Additional context (PR size, risk level, etc.)
            issue_number: Optional GitHub issue number
            outcome: Optional outcome tracking ("success", "defect_found", "false_positive")
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            decision_record = QualityDecision(
                timestamp=datetime.now(timezone.utc).isoformat(),
                issue_number=issue_number,
                component_type=component_type,
                threshold_used=threshold_used,
                quality_score=quality_score,
                decision=decision,
                context=context,
                outcome=outcome
            )
            
            with open(self.quality_decisions_file, 'a') as f:
                f.write(json.dumps(asdict(decision_record)) + "\n")
            
            self.logger.info(f"Recorded quality decision: {component_type} @ {threshold_used}% -> {decision}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record quality decision: {e}")
            return False
    
    def record_threshold_performance(self,
                                   component_type: str,
                                   threshold: float,
                                   success_rate: float,
                                   false_positive_rate: float,
                                   false_negative_rate: float,
                                   sample_size: int,
                                   time_period_days: int = 30) -> bool:
        """
        Record threshold performance metrics.
        
        Args:
            component_type: Type of component
            threshold: Threshold value being evaluated
            success_rate: Rate of correct decisions (0.0-1.0)
            false_positive_rate: Rate of false positives (0.0-1.0)
            false_negative_rate: Rate of false negatives (0.0-1.0)
            sample_size: Number of decisions in this evaluation
            time_period_days: Time period for this evaluation
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            performance_record = ThresholdPerformance(
                timestamp=datetime.now(timezone.utc).isoformat(),
                component_type=component_type,
                threshold=threshold,
                success_rate=success_rate,
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                sample_size=sample_size,
                time_period_days=time_period_days
            )
            
            with open(self.threshold_performance_file, 'a') as f:
                f.write(json.dumps(asdict(performance_record)) + "\n")
            
            self.logger.info(f"Recorded threshold performance: {component_type} @ {threshold}% success_rate={success_rate:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record threshold performance: {e}")
            return False
    
    def record_team_metrics(self,
                          team_id: str,
                          quality_trend: float,
                          average_threshold_success: float,
                          code_complexity_trend: float,
                          defect_resolution_time: float,
                          testing_coverage_trend: float) -> bool:
        """
        Record team performance metrics.
        
        Args:
            team_id: Team identifier
            quality_trend: Quality improvement rate (positive = improving)
            average_threshold_success: Average success rate across all thresholds
            code_complexity_trend: Complexity trend (positive = getting more complex)
            defect_resolution_time: Average time to resolve defects (hours)
            testing_coverage_trend: Test coverage trend (positive = improving)
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            team_record = TeamMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                team_id=team_id,
                quality_trend=quality_trend,
                average_threshold_success=average_threshold_success,
                code_complexity_trend=code_complexity_trend,
                defect_resolution_time=defect_resolution_time,
                testing_coverage_trend=testing_coverage_trend
            )
            
            with open(self.team_metrics_file, 'a') as f:
                f.write(json.dumps(asdict(team_record)) + "\n")
            
            self.logger.info(f"Recorded team metrics: {team_id} quality_trend={quality_trend:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record team metrics: {e}")
            return False
    
    def record_project_characteristics(self,
                                     project_name: str,
                                     primary_language: str,
                                     complexity_score: float,
                                     team_size: int,
                                     domain: str,
                                     risk_level: str) -> bool:
        """
        Record project characteristics for context-aware threshold optimization.
        
        Args:
            project_name: Name of the project
            primary_language: Primary programming language
            complexity_score: Overall project complexity (0.0-10.0)
            team_size: Number of team members
            domain: Project domain ("web", "mobile", "backend", "data", etc.)
            risk_level: Overall risk level ("low", "medium", "high", "critical")
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            project_record = ProjectCharacteristics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                project_name=project_name,
                primary_language=primary_language,
                complexity_score=complexity_score,
                team_size=team_size,
                domain=domain,
                risk_level=risk_level
            )
            
            with open(self.project_characteristics_file, 'a') as f:
                f.write(json.dumps(asdict(project_record)) + "\n")
            
            self.logger.info(f"Recorded project characteristics: {project_name} complexity={complexity_score}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record project characteristics: {e}")
            return False
    
    def get_quality_decisions(self, 
                            component_type: Optional[str] = None,
                            days_back: int = 90,
                            limit: Optional[int] = None) -> List[QualityDecision]:
        """
        Retrieve historical quality decisions.
        
        Args:
            component_type: Filter by component type (None for all)
            days_back: Number of days back to retrieve
            limit: Maximum number of records to return
            
        Returns:
            List of QualityDecision records
        """
        try:
            decisions = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            if not self.quality_decisions_file.exists():
                return decisions
            
            with open(self.quality_decisions_file, 'r') as f:
                for line in f:
                    try:
                        record_dict = json.loads(line.strip())
                        record = QualityDecision(**record_dict)
                        
                        # Apply time filter (use timezone-aware comparison)
                        record_time = datetime.fromisoformat(record.timestamp)
                        if record_time < cutoff_date:
                            continue
                        
                        # Apply component type filter
                        if component_type and record.component_type != component_type:
                            continue
                        
                        decisions.append(record)
                        
                        # Apply limit
                        if limit and len(decisions) >= limit:
                            break
                            
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        self.logger.warning(f"Skipping malformed record: {e}")
                        continue
            
            self.logger.info(f"Retrieved {len(decisions)} quality decisions")
            return decisions
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve quality decisions: {e}")
            return []
    
    def get_threshold_performance(self,
                                component_type: Optional[str] = None,
                                days_back: int = 90) -> List[ThresholdPerformance]:
        """
        Retrieve threshold performance data.
        
        Args:
            component_type: Filter by component type (None for all)
            days_back: Number of days back to retrieve
            
        Returns:
            List of ThresholdPerformance records
        """
        try:
            performance_data = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            if not self.threshold_performance_file.exists():
                return performance_data
            
            with open(self.threshold_performance_file, 'r') as f:
                for line in f:
                    try:
                        record_dict = json.loads(line.strip())
                        record = ThresholdPerformance(**record_dict)
                        
                        # Apply time filter (use timezone-aware comparison)
                        record_time = datetime.fromisoformat(record.timestamp)
                        if record_time < cutoff_date:
                            continue
                        
                        # Apply component type filter
                        if component_type and record.component_type != component_type:
                            continue
                        
                        performance_data.append(record)
                        
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        self.logger.warning(f"Skipping malformed performance record: {e}")
                        continue
            
            self.logger.info(f"Retrieved {len(performance_data)} threshold performance records")
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve threshold performance data: {e}")
            return []
    
    def analyze_threshold_effectiveness(self, 
                                      component_type: str,
                                      threshold_range: tuple = (70.0, 100.0),
                                      days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze threshold effectiveness for a component type.
        
        Args:
            component_type: Component type to analyze
            threshold_range: Range of thresholds to analyze (min, max)
            days_back: Number of days of data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            decisions = self.get_quality_decisions(component_type, days_back)
            
            if not decisions:
                return {
                    "component_type": component_type,
                    "analysis": "insufficient_data",
                    "total_decisions": 0,
                    "recommendation": f"No historical data available for {component_type}"
                }
            
            # Group decisions by threshold ranges
            threshold_buckets = {}
            for decision in decisions:
                threshold = decision.threshold_used
                bucket = int(threshold / 5) * 5  # Group into 5% buckets
                
                if bucket not in threshold_buckets:
                    threshold_buckets[bucket] = {
                        "pass": 0,
                        "fail": 0,
                        "manual_override": 0,
                        "total": 0,
                        "outcomes": {"success": 0, "defect_found": 0, "false_positive": 0, "unknown": 0}
                    }
                
                threshold_buckets[bucket][decision.decision] += 1
                threshold_buckets[bucket]["total"] += 1
                
                outcome = decision.outcome or "unknown"
                threshold_buckets[bucket]["outcomes"][outcome] += 1
            
            # Calculate effectiveness metrics
            analysis_results = []
            for threshold, data in threshold_buckets.items():
                if data["total"] < 5:  # Skip buckets with insufficient data
                    continue
                
                pass_rate = data["pass"] / data["total"]
                success_rate = (data["outcomes"]["success"] + data["outcomes"]["unknown"]) / data["total"]
                false_positive_rate = data["outcomes"]["false_positive"] / data["total"]
                
                effectiveness_score = success_rate * 0.6 + pass_rate * 0.3 - false_positive_rate * 0.1
                
                analysis_results.append({
                    "threshold": threshold,
                    "sample_size": data["total"],
                    "pass_rate": pass_rate,
                    "success_rate": success_rate,
                    "false_positive_rate": false_positive_rate,
                    "effectiveness_score": effectiveness_score
                })
            
            # Find optimal threshold
            if analysis_results:
                optimal = max(analysis_results, key=lambda x: x["effectiveness_score"])
                
                return {
                    "component_type": component_type,
                    "analysis": "complete",
                    "total_decisions": len(decisions),
                    "optimal_threshold": optimal["threshold"],
                    "optimal_effectiveness": optimal["effectiveness_score"],
                    "threshold_analysis": analysis_results,
                    "recommendation": f"Optimal threshold for {component_type} is {optimal['threshold']}% (effectiveness: {optimal['effectiveness_score']:.3f})"
                }
            else:
                return {
                    "component_type": component_type,
                    "analysis": "insufficient_samples",
                    "total_decisions": len(decisions),
                    "recommendation": f"Need more data samples for {component_type} (minimum 5 per threshold range)"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze threshold effectiveness: {e}")
            return {
                "component_type": component_type,
                "analysis": "error",
                "error": str(e),
                "recommendation": "Analysis failed due to error"
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected historical data.
        
        Returns:
            Dictionary with data summary statistics
        """
        try:
            summary = {
                "collection_period": {
                    "start_date": None,
                    "end_date": datetime.now(timezone.utc).isoformat(),
                    "total_days": 0
                },
                "quality_decisions": {
                    "total_count": 0,
                    "component_types": set(),
                    "decision_distribution": {"pass": 0, "fail": 0, "manual_override": 0}
                },
                "threshold_performance": {
                    "total_evaluations": 0,
                    "component_types": set()
                },
                "team_metrics": {
                    "teams_tracked": set(),
                    "latest_quality_trends": {}
                },
                "project_characteristics": {
                    "projects_tracked": set(),
                    "complexity_distribution": []
                },
                "data_quality": {
                    "completeness_score": 0.0,
                    "consistency_issues": []
                }
            }
            
            # Analyze quality decisions
            if self.quality_decisions_file.exists():
                with open(self.quality_decisions_file, 'r') as f:
                    earliest_date = None
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            summary["quality_decisions"]["total_count"] += 1
                            summary["quality_decisions"]["component_types"].add(record["component_type"])
                            summary["quality_decisions"]["decision_distribution"][record["decision"]] += 1
                            
                            record_date = datetime.fromisoformat(record["timestamp"])
                            if not earliest_date or record_date < earliest_date:
                                earliest_date = record_date
                        except:
                            continue
                    
                    if earliest_date:
                        summary["collection_period"]["start_date"] = earliest_date.isoformat()
                        summary["collection_period"]["total_days"] = (datetime.now(timezone.utc) - earliest_date).days
            
            # Convert sets to lists for JSON serialization
            summary["quality_decisions"]["component_types"] = list(summary["quality_decisions"]["component_types"])
            summary["threshold_performance"]["component_types"] = list(summary["threshold_performance"]["component_types"])
            summary["team_metrics"]["teams_tracked"] = list(summary["team_metrics"]["teams_tracked"])
            summary["project_characteristics"]["projects_tracked"] = list(summary["project_characteristics"]["projects_tracked"])
            
            # Calculate data quality score
            total_files = 4
            existing_files = sum(1 for f in [self.quality_decisions_file, self.threshold_performance_file, 
                                           self.team_metrics_file, self.project_characteristics_file] if f.exists())
            summary["data_quality"]["completeness_score"] = existing_files / total_files
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate data summary: {e}")
            return {"error": str(e)}

def main():
    """Command line interface for historical data collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical Data Collector for Adaptive Threshold Learning")
    parser.add_argument("--command", choices=["record", "analyze", "summary"], required=True,
                       help="Command to execute")
    parser.add_argument("--component-type", help="Component type for analysis")
    parser.add_argument("--threshold", type=float, help="Threshold value")
    parser.add_argument("--quality-score", type=float, help="Quality score")
    parser.add_argument("--decision", choices=["pass", "fail", "manual_override"], help="Quality gate decision")
    parser.add_argument("--days-back", type=int, default=90, help="Days of data to analyze")
    
    args = parser.parse_args()
    
    collector = HistoricalDataCollector()
    
    if args.command == "record" and args.component_type and args.threshold is not None and args.quality_score is not None and args.decision:
        result = collector.record_quality_decision(
            component_type=args.component_type,
            threshold_used=args.threshold,
            quality_score=args.quality_score,
            decision=args.decision,
            context={"source": "cli"}
        )
        print(f"Recording result: {'success' if result else 'failed'}")
    
    elif args.command == "analyze" and args.component_type:
        result = collector.analyze_threshold_effectiveness(args.component_type, days_back=args.days_back)
        print(json.dumps(result, indent=2))
    
    elif args.command == "summary":
        summary = collector.get_data_summary()
        print(json.dumps(summary, indent=2, default=str))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()