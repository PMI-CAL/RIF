#!/usr/bin/env python3
"""
Production Defect Correlation Tracker - Issue #94 Phase 1
Links quality gate decisions with actual production outcomes to measure effectiveness.

This component:
- Tracks production defects linked to GitHub issues
- Correlates quality scores with production outcomes  
- Calculates true/false positive and negative rates
- Provides data for threshold optimization
- Supports retrospective analysis of quality gate effectiveness
"""

import json
import subprocess
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class DefectSeverity(Enum):
    """Production defect severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    COSMETIC = "cosmetic"


class ImpactLevel(Enum):
    """Customer impact levels."""
    BLOCKING = "blocking"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    NONE = "none"


@dataclass
class ProductionDefect:
    """Represents a defect found in production."""
    defect_id: str
    issue_number: int
    change_id: str
    severity: str
    impact_level: str
    description: str
    detection_timestamp: str
    resolution_timestamp: Optional[str]
    detection_method: str
    affected_users: int
    rollback_required: bool
    hotfix_deployed: bool
    root_cause: Optional[str]
    prevention_measures: List[str]


@dataclass
class CorrelationResult:
    """Result of correlating quality gates with production outcomes."""
    issue_number: int
    change_id: str
    quality_score: float
    production_defects: List[ProductionDefect]
    correlation_accuracy: float
    gate_effectiveness: Dict[str, float]
    false_positive_indicators: List[str]
    false_negative_indicators: List[str]
    improvement_recommendations: List[str]


class ProductionCorrelationTracker:
    """
    Tracks and correlates production defects with quality gate decisions
    to measure and improve quality gate effectiveness.
    """
    
    def __init__(self, storage_path: str = "knowledge/quality_metrics"):
        """Initialize the production correlation tracker."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize correlation storage
        self.correlations_path = self.storage_path / "correlations"
        self.correlations_path.mkdir(exist_ok=True)
        
        # Initialize analysis storage
        self.analysis_path = self.storage_path / "analysis"
        self.analysis_path.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging for production correlation tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ProductionCorrelationTracker - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def track_production_defect(
        self,
        issue_number: int,
        change_id: str,
        severity: DefectSeverity,
        impact_level: ImpactLevel,
        description: str,
        detection_method: str = "automated",
        affected_users: int = 0,
        rollback_required: bool = False,
        hotfix_deployed: bool = False,
        root_cause: Optional[str] = None
    ) -> str:
        """
        Track a production defect linked to a specific issue/change.
        
        Args:
            issue_number: GitHub issue number that introduced the defect
            change_id: Unique identifier for the change (commit hash, PR number, etc.)
            severity: Severity level of the defect
            impact_level: Customer impact level
            description: Description of the defect
            detection_method: How the defect was detected
            affected_users: Number of users affected
            rollback_required: Whether a rollback was necessary
            hotfix_deployed: Whether a hotfix was deployed
            root_cause: Root cause analysis (if available)
            
        Returns:
            Defect ID for tracking purposes
        """
        try:
            defect = ProductionDefect(
                defect_id=self._generate_defect_id(issue_number, change_id),
                issue_number=issue_number,
                change_id=change_id,
                severity=severity.value,
                impact_level=impact_level.value,
                description=description,
                detection_timestamp=datetime.now().isoformat(),
                resolution_timestamp=None,
                detection_method=detection_method,
                affected_users=affected_users,
                rollback_required=rollback_required,
                hotfix_deployed=hotfix_deployed,
                root_cause=root_cause,
                prevention_measures=[]
            )
            
            # Store the defect
            self._store_defect(defect)
            
            # Update GitHub issue with defect information
            self._update_issue_with_defect(issue_number, defect)
            
            self.logger.info(
                f"Tracked production defect: {severity.value} severity, "
                f"issue #{issue_number}, change {change_id}"
            )
            
            return defect.defect_id
            
        except Exception as e:
            self.logger.error(f"Error tracking production defect: {e}")
            return ""
    
    def resolve_production_defect(
        self,
        defect_id: str,
        prevention_measures: List[str] = None
    ) -> bool:
        """
        Mark a production defect as resolved and record prevention measures.
        
        Args:
            defect_id: ID of the defect to resolve
            prevention_measures: Measures taken to prevent similar defects
            
        Returns:
            True if successfully resolved, False otherwise
        """
        try:
            # Load the defect
            defect = self._load_defect(defect_id)
            if not defect:
                self.logger.error(f"Defect not found: {defect_id}")
                return False
            
            # Update resolution information
            defect['resolution_timestamp'] = datetime.now().isoformat()
            defect['prevention_measures'] = prevention_measures or []
            
            # Store updated defect
            self._update_defect(defect)
            
            self.logger.info(f"Resolved production defect: {defect_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resolving production defect: {e}")
            return False
    
    def correlate_quality_with_outcomes(
        self,
        issue_number: int,
        days_back: int = 90
    ) -> Optional[CorrelationResult]:
        """
        Correlate quality gate decisions with production outcomes for an issue.
        
        Args:
            issue_number: GitHub issue number to analyze
            days_back: Number of days to look back for production defects
            
        Returns:
            Correlation analysis results or None if insufficient data
        """
        try:
            # Load quality gate decisions for the issue
            quality_decisions = self._load_quality_decisions(issue_number)
            if not quality_decisions:
                self.logger.warning(f"No quality decisions found for issue #{issue_number}")
                return None
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(quality_decisions)
            
            # Load production defects for the issue
            defects = self._load_production_defects(issue_number, days_back)
            
            # Determine change ID (use most recent decision context)
            change_id = self._extract_change_id(quality_decisions)
            
            # Calculate correlation metrics
            correlation_accuracy = self._calculate_correlation_accuracy(quality_score, defects)
            gate_effectiveness = self._analyze_gate_effectiveness(quality_decisions, defects)
            
            # Identify false positive/negative indicators
            fp_indicators = self._identify_false_positive_indicators(quality_decisions, defects)
            fn_indicators = self._identify_false_negative_indicators(quality_decisions, defects)
            
            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(
                quality_score, defects, gate_effectiveness
            )
            
            result = CorrelationResult(
                issue_number=issue_number,
                change_id=change_id,
                quality_score=quality_score,
                production_defects=[self._dict_to_defect(d) for d in defects],
                correlation_accuracy=correlation_accuracy,
                gate_effectiveness=gate_effectiveness,
                false_positive_indicators=fp_indicators,
                false_negative_indicators=fn_indicators,
                improvement_recommendations=recommendations
            )
            
            # Store correlation analysis
            self._store_correlation_result(result)
            
            self.logger.info(
                f"Correlation analysis complete for issue #{issue_number}: "
                f"Quality score {quality_score:.1f}, {len(defects)} defects, "
                f"accuracy {correlation_accuracy:.1f}%"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error correlating quality with outcomes: {e}")
            return None
    
    def get_effectiveness_trends(
        self,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Get quality gate effectiveness trends over time.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Trend analysis with recommendations
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Load all correlation results in the time period
            correlations = self._load_correlations_in_period(start_time, end_time)
            
            if not correlations:
                return {
                    'status': 'insufficient_data',
                    'message': f'No correlation data available for last {days_back} days'
                }
            
            # Calculate trend metrics
            avg_quality_score = sum(c['quality_score'] for c in correlations) / len(correlations)
            avg_defects_per_issue = sum(len(c['production_defects']) for c in correlations) / len(correlations)
            avg_correlation_accuracy = sum(c['correlation_accuracy'] for c in correlations) / len(correlations)
            
            # Calculate effectiveness by gate type
            gate_trends = self._calculate_gate_trends(correlations)
            
            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(correlations)
            
            # Calculate quality score vs defect correlation
            score_defect_correlation = self._calculate_score_defect_correlation(correlations)
            
            return {
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days_back
                },
                'summary': {
                    'total_issues_analyzed': len(correlations),
                    'average_quality_score': avg_quality_score,
                    'average_defects_per_issue': avg_defects_per_issue,
                    'average_correlation_accuracy': avg_correlation_accuracy
                },
                'gate_effectiveness_trends': gate_trends,
                'score_defect_correlation': score_defect_correlation,
                'improvement_opportunities': improvement_opportunities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating effectiveness trends: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_defect_id(self, issue_number: int, change_id: str) -> str:
        """Generate unique defect ID."""
        timestamp = datetime.now().isoformat()
        data = f"defect-{issue_number}-{change_id}-{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _store_defect(self, defect: ProductionDefect) -> None:
        """Store production defect to persistent storage."""
        defects_file = self.correlations_path / f"defects_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(defects_file, 'a') as f:
            f.write(json.dumps(asdict(defect)) + '\n')
    
    def _update_issue_with_defect(self, issue_number: int, defect: ProductionDefect) -> None:
        """Update GitHub issue with production defect information."""
        try:
            comment = f"""
## 🚨 Production Defect Detected

**Defect ID**: {defect.defect_id}
**Severity**: {defect.severity.upper()}
**Impact**: {defect.impact_level}
**Affected Users**: {defect.affected_users}

**Description**: {defect.description}

**Detection Method**: {defect.detection_method}
**Detection Time**: {defect.detection_timestamp}

{'**Rollback Required**: Yes' if defect.rollback_required else ''}
{'**Hotfix Deployed**: Yes' if defect.hotfix_deployed else ''}

This defect will be correlated with quality gate decisions to improve effectiveness.
"""
            
            result = subprocess.run([
                'gh', 'issue', 'comment', str(issue_number),
                '--body', comment
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Updated issue #{issue_number} with defect information")
            else:
                self.logger.warning(f"Failed to update issue #{issue_number}: {result.stderr}")
                
        except Exception as e:
            self.logger.warning(f"Error updating issue with defect: {e}")
    
    def _load_defect(self, defect_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific defect by ID."""
        for defects_file in self.correlations_path.glob("defects_*.jsonl"):
            try:
                with open(defects_file, 'r') as f:
                    for line in f:
                        defect = json.loads(line.strip())
                        if defect['defect_id'] == defect_id:
                            return defect
            except Exception as e:
                self.logger.warning(f"Error reading defects file {defects_file}: {e}")
                continue
        return None
    
    def _update_defect(self, updated_defect: Dict[str, Any]) -> None:
        """Update an existing defect record."""
        defect_id = updated_defect['defect_id']
        
        # Read all defects, update the matching one
        for defects_file in self.correlations_path.glob("defects_*.jsonl"):
            try:
                defects = []
                with open(defects_file, 'r') as f:
                    for line in f:
                        defect = json.loads(line.strip())
                        if defect['defect_id'] == defect_id:
                            defects.append(updated_defect)
                        else:
                            defects.append(defect)
                
                # Rewrite the file with updated data
                with open(defects_file, 'w') as f:
                    for defect in defects:
                        f.write(json.dumps(defect) + '\n')
                        
                break
            except Exception as e:
                self.logger.warning(f"Error updating defects file {defects_file}: {e}")
                continue
    
    def _load_quality_decisions(self, issue_number: int) -> List[Dict[str, Any]]:
        """Load quality gate decisions for an issue."""
        decisions = []
        recent_dir = Path("knowledge/quality_metrics/recent")
        
        if not recent_dir.exists():
            return decisions
        
        for decision_file in recent_dir.glob("decisions_*.jsonl"):
            try:
                with open(decision_file, 'r') as f:
                    for line in f:
                        decision = json.loads(line.strip())
                        if decision['issue_number'] == issue_number:
                            decisions.append(decision)
            except Exception as e:
                self.logger.warning(f"Error reading decision file {decision_file}: {e}")
                continue
        
        return decisions
    
    def _load_production_defects(self, issue_number: int, days_back: int) -> List[Dict[str, Any]]:
        """Load production defects for an issue within a time period."""
        defects = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        for defects_file in self.correlations_path.glob("defects_*.jsonl"):
            try:
                with open(defects_file, 'r') as f:
                    for line in f:
                        defect = json.loads(line.strip())
                        if (defect['issue_number'] == issue_number and
                            datetime.fromisoformat(defect['detection_timestamp']) >= cutoff_time):
                            defects.append(defect)
            except Exception as e:
                self.logger.warning(f"Error reading defects file {defects_file}: {e}")
                continue
        
        return defects
    
    def _calculate_overall_quality_score(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score from gate decisions."""
        if not decisions:
            return 0.0
        
        # Weight different gate types
        gate_weights = {
            'code_coverage': 0.2,
            'security_scan': 0.3,
            'linting': 0.1,
            'performance': 0.15,
            'documentation': 0.05,
            'evidence_requirements': 0.1,
            'risk_assessment': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        # Get latest decision for each gate type
        gate_decisions = {}
        for decision in decisions:
            gate_type = decision['gate_type']
            if (gate_type not in gate_decisions or 
                decision['timestamp'] > gate_decisions[gate_type]['timestamp']):
                gate_decisions[gate_type] = decision
        
        # Calculate weighted score
        for gate_type, decision in gate_decisions.items():
            weight = gate_weights.get(gate_type, 0.1)
            
            if decision['decision'] == 'pass':
                score = 100.0
            elif decision['decision'] == 'warning':
                score = 75.0
            elif decision['decision'] == 'fail':
                score = 0.0
            else:  # skip or manual_override
                score = 50.0
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_change_id(self, decisions: List[Dict[str, Any]]) -> str:
        """Extract change ID from quality decisions context."""
        for decision in decisions:
            context = decision.get('context', {})
            if 'change_id' in context:
                return context['change_id']
            if 'commit_hash' in context:
                return context['commit_hash']
            if 'pr_number' in context:
                return f"pr-{context['pr_number']}"
        
        # Fallback to issue number
        return f"issue-{decisions[0]['issue_number']}"
    
    def _calculate_correlation_accuracy(self, quality_score: float, defects: List[Dict[str, Any]]) -> float:
        """Calculate how well quality score predicted production outcomes."""
        has_critical_defects = any(d['severity'] in ['critical', 'high'] for d in defects)
        
        # High quality score should mean no critical defects
        if quality_score >= 80 and not has_critical_defects:
            return 100.0  # True negative - correctly predicted good quality
        elif quality_score >= 80 and has_critical_defects:
            return 0.0   # False negative - high score but had critical defects
        elif quality_score < 80 and has_critical_defects:
            return 100.0  # True positive - correctly predicted problems
        else:  # quality_score < 80 and not has_critical_defects
            return 25.0   # False positive - low score but no critical defects
    
    def _analyze_gate_effectiveness(self, decisions: List[Dict[str, Any]], defects: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze effectiveness of individual quality gates."""
        gate_effectiveness = {}
        has_defects = len(defects) > 0
        
        # Group decisions by gate type
        gate_decisions = {}
        for decision in decisions:
            gate_type = decision['gate_type']
            if gate_type not in gate_decisions:
                gate_decisions[gate_type] = []
            gate_decisions[gate_type].append(decision)
        
        # Analyze each gate type
        for gate_type, gate_decisions_list in gate_decisions.items():
            # Get latest decision
            latest_decision = max(gate_decisions_list, key=lambda x: x['timestamp'])
            decision_passed = latest_decision['decision'] == 'pass'
            
            # Calculate effectiveness (inverse correlation with defects)
            if decision_passed and not has_defects:
                effectiveness = 100.0  # Correctly passed
            elif decision_passed and has_defects:
                effectiveness = 20.0   # False negative
            elif not decision_passed and has_defects:
                effectiveness = 100.0  # Correctly failed
            else:  # not decision_passed and not has_defects
                effectiveness = 60.0   # False positive (but caught potential issue)
            
            gate_effectiveness[gate_type] = effectiveness
        
        return gate_effectiveness
    
    def _identify_false_positive_indicators(self, decisions: List[Dict[str, Any]], defects: List[Dict[str, Any]]) -> List[str]:
        """Identify indicators of false positive quality gate decisions."""
        indicators = []
        
        # No production defects but quality gates failed
        if len(defects) == 0:
            failed_gates = [d for d in decisions if d['decision'] == 'fail']
            if failed_gates:
                indicators.append("Quality gates failed but no production defects occurred")
                gate_types = [d['gate_type'] for d in failed_gates]
                indicators.append(f"Failed gates: {', '.join(set(gate_types))}")
        
        return indicators
    
    def _identify_false_negative_indicators(self, decisions: List[Dict[str, Any]], defects: List[Dict[str, Any]]) -> List[str]:
        """Identify indicators of false negative quality gate decisions."""
        indicators = []
        
        # Production defects but quality gates passed
        critical_defects = [d for d in defects if d['severity'] in ['critical', 'high']]
        if critical_defects:
            passed_gates = [d for d in decisions if d['decision'] == 'pass']
            if passed_gates:
                indicators.append("Critical production defects occurred despite quality gates passing")
                gate_types = [d['gate_type'] for d in passed_gates]
                indicators.append(f"Passed gates: {', '.join(set(gate_types))}")
        
        return indicators
    
    def _generate_improvement_recommendations(
        self, 
        quality_score: float, 
        defects: List[Dict[str, Any]], 
        gate_effectiveness: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving quality gate effectiveness."""
        recommendations = []
        
        # Overall score recommendations
        if quality_score >= 80 and len(defects) > 0:
            recommendations.append("Consider tightening quality thresholds - high scores with production defects")
        elif quality_score < 50 and len(defects) == 0:
            recommendations.append("Consider relaxing some quality thresholds - low scores without defects")
        
        # Gate-specific recommendations
        for gate_type, effectiveness in gate_effectiveness.items():
            if effectiveness < 50:
                recommendations.append(f"Review {gate_type} gate configuration - low effectiveness")
        
        # Defect pattern recommendations
        severity_counts = {}
        for defect in defects:
            severity = defect['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append("Add stricter security and risk assessment gates for critical defect prevention")
        
        return recommendations
    
    def _dict_to_defect(self, defect_dict: Dict[str, Any]) -> ProductionDefect:
        """Convert dictionary to ProductionDefect object."""
        return ProductionDefect(**defect_dict)
    
    def _store_correlation_result(self, result: CorrelationResult) -> None:
        """Store correlation analysis result."""
        analysis_file = self.analysis_path / f"correlation_{result.issue_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def _load_correlations_in_period(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load correlation results within a time period."""
        correlations = []
        
        for analysis_file in self.analysis_path.glob("correlation_*.json"):
            try:
                # Extract timestamp from filename
                filename_parts = analysis_file.stem.split('_')
                if len(filename_parts) >= 3:
                    timestamp_str = f"{filename_parts[2]}_{filename_parts[3]}"
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if start_time <= file_time <= end_time:
                        with open(analysis_file, 'r') as f:
                            correlation = json.load(f)
                            correlations.append(correlation)
            except Exception as e:
                self.logger.warning(f"Error reading correlation file {analysis_file}: {e}")
                continue
        
        return correlations
    
    def _calculate_gate_trends(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate effectiveness trends by gate type."""
        gate_trends = {}
        
        for correlation in correlations:
            for gate_type, effectiveness in correlation['gate_effectiveness'].items():
                if gate_type not in gate_trends:
                    gate_trends[gate_type] = []
                gate_trends[gate_type].append(effectiveness)
        
        # Calculate averages and trends
        trend_summary = {}
        for gate_type, effectiveness_values in gate_trends.items():
            trend_summary[gate_type] = {
                'average_effectiveness': sum(effectiveness_values) / len(effectiveness_values),
                'sample_size': len(effectiveness_values),
                'min_effectiveness': min(effectiveness_values),
                'max_effectiveness': max(effectiveness_values)
            }
        
        return trend_summary
    
    def _identify_improvement_opportunities(self, correlations: List[Dict[str, Any]]) -> List[str]:
        """Identify system-wide improvement opportunities."""
        opportunities = []
        
        # Calculate overall false positive/negative rates
        fp_count = sum(1 for c in correlations if len(c['false_positive_indicators']) > 0)
        fn_count = sum(1 for c in correlations if len(c['false_negative_indicators']) > 0)
        total = len(correlations)
        
        if fp_count / total > 0.1:  # More than 10% false positives
            opportunities.append(f"High false positive rate ({fp_count/total:.1%}) - consider relaxing thresholds")
        
        if fn_count / total > 0.05:  # More than 5% false negatives
            opportunities.append(f"High false negative rate ({fn_count/total:.1%}) - consider tightening thresholds")
        
        # Find consistently ineffective gates
        gate_effectiveness = {}
        for correlation in correlations:
            for gate_type, effectiveness in correlation['gate_effectiveness'].items():
                if gate_type not in gate_effectiveness:
                    gate_effectiveness[gate_type] = []
                gate_effectiveness[gate_type].append(effectiveness)
        
        for gate_type, values in gate_effectiveness.items():
            avg_effectiveness = sum(values) / len(values)
            if avg_effectiveness < 60:
                opportunities.append(f"{gate_type} gate shows low effectiveness ({avg_effectiveness:.1f}%) - needs review")
        
        return opportunities
    
    def _calculate_score_defect_correlation(self, correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlation between quality scores and defect counts."""
        if not correlations:
            return {'correlation': 0.0, 'sample_size': 0}
        
        scores = [c['quality_score'] for c in correlations]
        defect_counts = [len(c['production_defects']) for c in correlations]
        
        # Calculate Pearson correlation coefficient
        n = len(scores)
        if n < 2:
            return {'correlation': 0.0, 'sample_size': n}
        
        mean_score = sum(scores) / n
        mean_defects = sum(defect_counts) / n
        
        numerator = sum((scores[i] - mean_score) * (defect_counts[i] - mean_defects) for i in range(n))
        
        score_variance = sum((s - mean_score) ** 2 for s in scores)
        defect_variance = sum((d - mean_defects) ** 2 for d in defect_counts)
        
        denominator = (score_variance * defect_variance) ** 0.5
        
        correlation = numerator / denominator if denominator != 0 else 0.0
        
        return {
            'correlation': correlation,
            'sample_size': n,
            'interpretation': 'Strong negative correlation expected (higher scores = fewer defects)'
        }


def main():
    """Command line interface for production correlation tracker."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python production_correlation_tracker.py <command> [args]")
        print("Commands:")
        print("  track-defect <issue_number> <change_id> <severity> <description>")
        print("  resolve-defect <defect_id>")
        print("  correlate <issue_number> [days_back]")
        print("  trends [days_back]")
        return 1
    
    tracker = ProductionCorrelationTracker()
    command = sys.argv[1]
    
    if command == "track-defect" and len(sys.argv) >= 5:
        issue_num = int(sys.argv[2])
        change_id = sys.argv[3]
        severity = DefectSeverity(sys.argv[4])
        description = sys.argv[5] if len(sys.argv) > 5 else "No description provided"
        
        defect_id = tracker.track_production_defect(
            issue_num, change_id, severity, ImpactLevel.MODERATE, description
        )
        print(f"Defect tracked: {defect_id}")
        
    elif command == "resolve-defect" and len(sys.argv) >= 3:
        defect_id = sys.argv[2]
        success = tracker.resolve_production_defect(defect_id)
        print(f"Defect resolution: {'success' if success else 'failed'}")
        
    elif command == "correlate" and len(sys.argv) >= 3:
        issue_num = int(sys.argv[2])
        days_back = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        
        result = tracker.correlate_quality_with_outcomes(issue_num, days_back)
        if result:
            print(json.dumps(asdict(result), indent=2, default=str))
        else:
            print("No correlation data available")
        
    elif command == "trends":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        trends = tracker.get_effectiveness_trends(days_back)
        print(json.dumps(trends, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())