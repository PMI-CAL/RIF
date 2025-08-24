#!/usr/bin/env python3
"""
Quality Pattern Analyzer for Adaptive Threshold Learning System
Issue #95: Adaptive Threshold Learning System

Analyzes quality patterns from RIF knowledge base and GitHub issues to identify
successful threshold configurations and quality strategies.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import yaml

@dataclass
class QualityPattern:
    """Represents a successful quality pattern from historical data."""
    pattern_id: str
    pattern_type: str  # "threshold_config", "quality_strategy", "team_practice"
    component_type: str
    success_rate: float
    sample_size: int
    configuration: Dict[str, Any]
    context: Dict[str, Any]
    learnings: List[str]
    last_updated: str

@dataclass
class ThresholdStrategy:
    """Represents a successful threshold strategy."""
    strategy_id: str
    name: str
    description: str
    component_types: List[str]
    threshold_ranges: Dict[str, Tuple[float, float]]
    conditions: Dict[str, Any]
    effectiveness_score: float
    usage_count: int

@dataclass
class QualityInsight:
    """Represents a quality insight extracted from historical data."""
    insight_id: str
    insight_type: str  # "threshold_optimization", "pattern_recognition", "trend_analysis"
    description: str
    evidence: Dict[str, Any]
    confidence: float
    actionable_recommendations: List[str]
    timestamp: str

class QualityPatternAnalyzer:
    """
    Analyzes quality patterns from RIF knowledge base and historical data.
    
    Features:
    - Extract successful patterns from RIF knowledge base
    - Analyze GitHub issues for quality outcomes
    - Identify optimal threshold configurations
    - Generate actionable quality insights
    - Pattern matching for similar project contexts
    """
    
    def __init__(self, 
                 knowledge_base_dir: str = "knowledge",
                 quality_data_dir: str = "quality/historical"):
        """
        Initialize the quality pattern analyzer.
        
        Args:
            knowledge_base_dir: Path to RIF knowledge base
            quality_data_dir: Path to historical quality data
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.quality_data_dir = Path(quality_data_dir)
        self.patterns_dir = Path("quality/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        self._load_existing_patterns()
        
    def setup_logging(self):
        """Setup logging for quality pattern analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QualityPatternAnalyzer - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_existing_patterns(self):
        """Load existing quality patterns from storage."""
        self.quality_patterns = []
        self.threshold_strategies = []
        
        patterns_file = self.patterns_dir / "quality_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.quality_patterns = [QualityPattern(**p) for p in data.get("patterns", [])]
                    self.threshold_strategies = [ThresholdStrategy(**s) for s in data.get("strategies", [])]
                self.logger.info(f"Loaded {len(self.quality_patterns)} patterns and {len(self.threshold_strategies)} strategies")
            except Exception as e:
                self.logger.warning(f"Failed to load existing patterns: {e}")
    
    def analyze_rif_knowledge_base(self) -> List[QualityPattern]:
        """
        Extract quality patterns from RIF knowledge base.
        
        Returns:
            List of quality patterns discovered
        """
        discovered_patterns = []
        
        try:
            # Analyze patterns from knowledge/patterns/
            patterns_dir = self.knowledge_base_dir / "patterns"
            if patterns_dir.exists():
                for pattern_file in patterns_dir.glob("*.json"):
                    patterns = self._extract_patterns_from_file(pattern_file)
                    discovered_patterns.extend(patterns)
            
            # Analyze decisions from knowledge/decisions/
            decisions_dir = self.knowledge_base_dir / "decisions"
            if decisions_dir.exists():
                for decision_file in decisions_dir.glob("*.json"):
                    patterns = self._extract_patterns_from_decisions(decision_file)
                    discovered_patterns.extend(patterns)
            
            # Analyze learning artifacts from knowledge/learning/
            learning_dir = self.knowledge_base_dir / "learning"
            if learning_dir.exists():
                for learning_file in learning_dir.glob("*.json"):
                    patterns = self._extract_patterns_from_learning(learning_file)
                    discovered_patterns.extend(patterns)
            
            self.logger.info(f"Discovered {len(discovered_patterns)} quality patterns from RIF knowledge base")
            return discovered_patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze RIF knowledge base: {e}")
            return []
    
    def _extract_patterns_from_file(self, pattern_file: Path) -> List[QualityPattern]:
        """Extract quality patterns from a pattern file."""
        patterns = []
        
        try:
            with open(pattern_file, 'r') as f:
                data = json.load(f)
            
            # Look for quality-related patterns
            if self._contains_quality_info(data):
                pattern = self._create_quality_pattern_from_data(data, "knowledge_pattern", pattern_file.stem)
                if pattern:
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patterns from {pattern_file}: {e}")
        
        return patterns
    
    def _extract_patterns_from_decisions(self, decision_file: Path) -> List[QualityPattern]:
        """Extract quality patterns from decision files."""
        patterns = []
        
        try:
            with open(decision_file, 'r') as f:
                data = json.load(f)
            
            # Look for quality gate decisions and their outcomes
            if "quality_gates" in data or "quality" in str(data).lower():
                pattern = self._create_quality_pattern_from_data(data, "decision_pattern", decision_file.stem)
                if pattern:
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patterns from decisions {decision_file}: {e}")
        
        return patterns
    
    def _extract_patterns_from_learning(self, learning_file: Path) -> List[QualityPattern]:
        """Extract quality patterns from learning files."""
        patterns = []
        
        try:
            with open(learning_file, 'r') as f:
                data = json.load(f)
            
            # Look for learnings related to quality improvements
            if "quality" in str(data).lower() or "threshold" in str(data).lower():
                pattern = self._create_quality_pattern_from_data(data, "learning_pattern", learning_file.stem)
                if pattern:
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patterns from learning {learning_file}: {e}")
        
        return patterns
    
    def _contains_quality_info(self, data: Dict[str, Any]) -> bool:
        """Check if data contains quality-related information."""
        quality_keywords = ["quality", "threshold", "coverage", "test", "defect", "bug", "reliability"]
        data_str = json.dumps(data).lower()
        return any(keyword in data_str for keyword in quality_keywords)
    
    def _create_quality_pattern_from_data(self, 
                                        data: Dict[str, Any], 
                                        pattern_type: str, 
                                        pattern_id: str) -> Optional[QualityPattern]:
        """Create a quality pattern from data if it contains useful quality information."""
        try:
            # Extract component type
            component_type = self._infer_component_type(data)
            
            # Extract configuration information
            configuration = self._extract_configuration(data)
            
            # Extract context
            context = self._extract_context(data)
            
            # Extract learnings
            learnings = self._extract_learnings(data)
            
            # Calculate success indicators (heuristic)
            success_rate = self._calculate_success_rate(data)
            sample_size = self._estimate_sample_size(data)
            
            if success_rate > 0 and sample_size > 0:
                return QualityPattern(
                    pattern_id=f"{pattern_type}_{pattern_id}_{int(datetime.utcnow().timestamp())}",
                    pattern_type=pattern_type,
                    component_type=component_type,
                    success_rate=success_rate,
                    sample_size=sample_size,
                    configuration=configuration,
                    context=context,
                    learnings=learnings,
                    last_updated=datetime.utcnow().isoformat() + "Z"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to create pattern from {pattern_id}: {e}")
        
        return None
    
    def _infer_component_type(self, data: Dict[str, Any]) -> str:
        """Infer component type from data."""
        data_str = json.dumps(data).lower()
        
        # Component type keywords mapping
        type_keywords = {
            "critical_algorithms": ["algorithm", "crypto", "security", "encryption", "hash"],
            "public_apis": ["api", "endpoint", "route", "controller", "service"],
            "business_logic": ["business", "logic", "model", "domain", "service"],
            "ui_components": ["ui", "component", "frontend", "view", "template"],
            "integration_code": ["integration", "connector", "adapter", "client"],
            "test_utilities": ["test", "fixture", "mock", "spec"]
        }
        
        for component_type, keywords in type_keywords.items():
            if any(keyword in data_str for keyword in keywords):
                return component_type
        
        return "general"
    
    def _extract_configuration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration information from data."""
        config = {}
        
        # Look for threshold-related configuration
        if "threshold" in data:
            config["thresholds"] = data["threshold"]
        
        # Look for quality-related configuration
        if "quality_gates" in data:
            config["quality_gates"] = data["quality_gates"]
        
        # Look for test configuration
        if "test" in data or "testing" in data:
            config["testing"] = data.get("test") or data.get("testing")
        
        # Extract numeric values that might be thresholds
        for key, value in data.items():
            if isinstance(value, (int, float)) and 0 <= value <= 100:
                config[key] = value
        
        return config
    
    def _extract_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from data."""
        context = {}
        
        # Common context fields
        context_fields = [
            "complexity", "team_size", "project_type", "domain", "language",
            "risk_level", "priority", "issue_number", "timestamp"
        ]
        
        for field in context_fields:
            if field in data:
                context[field] = data[field]
        
        return context
    
    def _extract_learnings(self, data: Dict[str, Any]) -> List[str]:
        """Extract learning statements from data."""
        learnings = []
        
        # Look for explicit learnings
        if "learnings" in data:
            if isinstance(data["learnings"], list):
                learnings.extend(data["learnings"])
            elif isinstance(data["learnings"], str):
                learnings.append(data["learnings"])
        
        # Look for insights or lessons learned
        for key in ["insights", "lessons", "conclusions", "recommendations"]:
            if key in data:
                if isinstance(data[key], list):
                    learnings.extend(data[key])
                elif isinstance(data[key], str):
                    learnings.append(data[key])
        
        return learnings
    
    def _calculate_success_rate(self, data: Dict[str, Any]) -> float:
        """Calculate or estimate success rate from data."""
        # Look for explicit success metrics
        if "success_rate" in data:
            return float(data["success_rate"])
        
        # Look for pass/fail information
        if "passed" in data and "total" in data:
            return data["passed"] / data["total"]
        
        # Look for positive indicators
        positive_keywords = ["success", "pass", "complete", "resolved", "fixed"]
        negative_keywords = ["fail", "error", "bug", "defect", "issue"]
        
        data_str = json.dumps(data).lower()
        positive_count = sum(data_str.count(word) for word in positive_keywords)
        negative_count = sum(data_str.count(word) for word in negative_keywords)
        
        total_indicators = positive_count + negative_count
        if total_indicators > 0:
            return positive_count / total_indicators
        
        # Default moderate success rate for data that doesn't have clear indicators
        return 0.75
    
    def _estimate_sample_size(self, data: Dict[str, Any]) -> int:
        """Estimate sample size from data."""
        # Look for explicit sample size
        if "sample_size" in data:
            return int(data["sample_size"])
        
        # Look for test counts or similar metrics
        for key in ["tests", "count", "total", "issues"]:
            if key in data and isinstance(data[key], (int, float)):
                return int(data[key])
        
        # Default small sample size
        return 5
    
    def analyze_github_issues_quality_outcomes(self) -> List[QualityInsight]:
        """
        Analyze GitHub issues to extract quality outcome patterns.
        
        Returns:
            List of quality insights from GitHub issues
        """
        insights = []
        
        try:
            # Look for issue analysis files in knowledge base
            issues_dir = self.knowledge_base_dir / "issues"
            if issues_dir.exists():
                for issue_file in issues_dir.glob("*.json"):
                    issue_insights = self._analyze_issue_file(issue_file)
                    insights.extend(issue_insights)
            
            self.logger.info(f"Generated {len(insights)} quality insights from GitHub issues")
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to analyze GitHub issues: {e}")
            return []
    
    def _analyze_issue_file(self, issue_file: Path) -> List[QualityInsight]:
        """Analyze a single issue file for quality insights."""
        insights = []
        
        try:
            with open(issue_file, 'r') as f:
                data = json.load(f)
            
            # Extract quality-related insights
            if self._contains_quality_info(data):
                insight = self._create_quality_insight(data, issue_file.stem)
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze issue file {issue_file}: {e}")
        
        return insights
    
    def _create_quality_insight(self, data: Dict[str, Any], issue_id: str) -> Optional[QualityInsight]:
        """Create a quality insight from issue data."""
        try:
            # Determine insight type
            insight_type = self._classify_insight_type(data)
            
            # Generate description
            description = self._generate_insight_description(data, insight_type)
            
            # Extract evidence
            evidence = self._extract_evidence(data)
            
            # Calculate confidence
            confidence = self._calculate_insight_confidence(data, evidence)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(data, insight_type)
            
            if description and confidence > 0.5:
                return QualityInsight(
                    insight_id=f"insight_{issue_id}_{int(datetime.utcnow().timestamp())}",
                    insight_type=insight_type,
                    description=description,
                    evidence=evidence,
                    confidence=confidence,
                    actionable_recommendations=recommendations,
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to create insight from {issue_id}: {e}")
        
        return None
    
    def _classify_insight_type(self, data: Dict[str, Any]) -> str:
        """Classify the type of quality insight."""
        data_str = json.dumps(data).lower()
        
        if "threshold" in data_str:
            return "threshold_optimization"
        elif "pattern" in data_str:
            return "pattern_recognition"
        elif "trend" in data_str or "improvement" in data_str:
            return "trend_analysis"
        else:
            return "general_quality"
    
    def _generate_insight_description(self, data: Dict[str, Any], insight_type: str) -> str:
        """Generate a description for the quality insight."""
        # Extract key information for description
        component_type = self._infer_component_type(data)
        
        if insight_type == "threshold_optimization":
            return f"Threshold optimization opportunity identified for {component_type} components"
        elif insight_type == "pattern_recognition":
            return f"Quality pattern recognition for {component_type} showing consistent outcomes"
        elif insight_type == "trend_analysis":
            return f"Quality trend analysis indicates improvement opportunities for {component_type}"
        else:
            return f"General quality insight for {component_type} components"
    
    def _extract_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evidence supporting the quality insight."""
        evidence = {}
        
        # Look for metrics
        for key, value in data.items():
            if isinstance(value, (int, float)) and key in ["success_rate", "accuracy", "coverage", "score"]:
                evidence[key] = value
        
        # Look for test results
        if "test_results" in data:
            evidence["test_results"] = data["test_results"]
        
        return evidence
    
    def _calculate_insight_confidence(self, data: Dict[str, Any], evidence: Dict[str, Any]) -> float:
        """Calculate confidence level for the insight."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on evidence
        if evidence:
            confidence += 0.2
        
        # Increase confidence based on sample size
        sample_size = self._estimate_sample_size(data)
        if sample_size >= 10:
            confidence += 0.1
        if sample_size >= 50:
            confidence += 0.1
        
        # Increase confidence based on data completeness
        if len(data) > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_recommendations(self, data: Dict[str, Any], insight_type: str) -> List[str]:
        """Generate actionable recommendations based on the insight."""
        recommendations = []
        
        component_type = self._infer_component_type(data)
        
        if insight_type == "threshold_optimization":
            recommendations.extend([
                f"Consider adjusting quality thresholds for {component_type} based on historical performance",
                f"Implement A/B testing for new threshold values in {component_type}",
                f"Monitor success rates after threshold adjustments for {component_type}"
            ])
        elif insight_type == "pattern_recognition":
            recommendations.extend([
                f"Apply successful quality patterns to similar {component_type} components",
                f"Document and standardize quality practices for {component_type}",
                f"Share quality pattern learnings with team for {component_type}"
            ])
        elif insight_type == "trend_analysis":
            recommendations.extend([
                f"Investigate quality improvement trends for {component_type}",
                f"Identify factors contributing to quality changes in {component_type}",
                f"Implement continuous monitoring for {component_type} quality trends"
            ])
        
        return recommendations
    
    def find_optimal_thresholds(self, 
                               component_type: str,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find optimal thresholds for a component type based on patterns and historical data.
        
        Args:
            component_type: Component type to analyze
            context: Optional context for threshold optimization
            
        Returns:
            Dictionary with optimal threshold recommendations
        """
        try:
            # Find relevant patterns
            relevant_patterns = [p for p in self.quality_patterns if p.component_type == component_type]
            
            if not relevant_patterns:
                return {
                    "component_type": component_type,
                    "recommendation": "no_patterns_found",
                    "suggested_threshold": 80.0,
                    "confidence": 0.3,
                    "rationale": f"No historical patterns found for {component_type}, using conservative default"
                }
            
            # Calculate weighted optimal threshold
            total_weight = 0
            weighted_threshold = 0
            
            for pattern in relevant_patterns:
                # Weight by success rate and sample size
                weight = pattern.success_rate * min(pattern.sample_size, 100)  # Cap sample size weight
                
                # Extract threshold from configuration
                threshold = self._extract_threshold_from_config(pattern.configuration)
                
                weighted_threshold += threshold * weight
                total_weight += weight
            
            optimal_threshold = weighted_threshold / total_weight if total_weight > 0 else 80.0
            
            # Apply context adjustments if provided
            if context:
                optimal_threshold = self._apply_context_adjustments(optimal_threshold, context)
            
            # Calculate confidence based on pattern quality
            confidence = min(len(relevant_patterns) * 0.2, 1.0)
            avg_success_rate = sum(p.success_rate for p in relevant_patterns) / len(relevant_patterns)
            confidence *= avg_success_rate
            
            return {
                "component_type": component_type,
                "recommendation": "patterns_based",
                "suggested_threshold": round(optimal_threshold, 1),
                "confidence": round(confidence, 2),
                "patterns_used": len(relevant_patterns),
                "rationale": f"Based on {len(relevant_patterns)} patterns with average success rate {avg_success_rate:.2f}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to find optimal thresholds: {e}")
            return {
                "component_type": component_type,
                "recommendation": "error",
                "error": str(e),
                "suggested_threshold": 80.0,
                "confidence": 0.0
            }
    
    def _extract_threshold_from_config(self, configuration: Dict[str, Any]) -> float:
        """Extract threshold value from configuration."""
        # Look for explicit threshold
        if "threshold" in configuration:
            return float(configuration["threshold"])
        
        # Look for coverage threshold
        if "coverage_threshold" in configuration:
            return float(configuration["coverage_threshold"])
        
        # Look for quality threshold
        if "quality_threshold" in configuration:
            return float(configuration["quality_threshold"])
        
        # Look for any numeric value that could be a threshold (0-100 range)
        for value in configuration.values():
            if isinstance(value, (int, float)) and 0 <= value <= 100:
                return float(value)
        
        # Default threshold
        return 80.0
    
    def _apply_context_adjustments(self, threshold: float, context: Dict[str, Any]) -> float:
        """Apply context-based adjustments to threshold."""
        adjusted = threshold
        
        # Risk level adjustments
        if "risk_level" in context:
            risk_adjustments = {
                "critical": 1.2,
                "high": 1.1,
                "medium": 1.0,
                "low": 0.9
            }
            risk_modifier = risk_adjustments.get(context["risk_level"], 1.0)
            adjusted *= risk_modifier
        
        # Team size adjustments (larger teams might need higher thresholds)
        if "team_size" in context:
            team_size = context["team_size"]
            if team_size > 10:
                adjusted *= 1.05
            elif team_size < 3:
                adjusted *= 0.95
        
        # Project complexity adjustments
        if "complexity" in context:
            complexity = context["complexity"]
            if complexity > 7:
                adjusted *= 1.1
            elif complexity < 3:
                adjusted *= 0.9
        
        return max(60.0, min(100.0, adjusted))
    
    def save_patterns_and_insights(self) -> bool:
        """Save discovered patterns and insights to storage."""
        try:
            patterns_file = self.patterns_dir / "quality_patterns.json"
            
            # Combine existing patterns with any new ones
            all_patterns = self.quality_patterns
            
            # Save patterns and strategies
            data = {
                "patterns": [asdict(p) for p in all_patterns],
                "strategies": [asdict(s) for s in self.threshold_strategies],
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "total_patterns": len(all_patterns),
                "total_strategies": len(self.threshold_strategies)
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {len(all_patterns)} patterns and {len(self.threshold_strategies)} strategies")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save patterns and insights: {e}")
            return False

def main():
    """Command line interface for quality pattern analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Pattern Analyzer for Adaptive Threshold Learning")
    parser.add_argument("--command", choices=["analyze", "patterns", "optimal", "insights"], required=True,
                       help="Command to execute")
    parser.add_argument("--component-type", help="Component type for analysis")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    analyzer = QualityPatternAnalyzer()
    
    if args.command == "analyze":
        print("Analyzing RIF knowledge base for quality patterns...")
        patterns = analyzer.analyze_rif_knowledge_base()
        analyzer.quality_patterns.extend(patterns)
        analyzer.save_patterns_and_insights()
        print(f"Discovered and saved {len(patterns)} new quality patterns")
    
    elif args.command == "patterns":
        analyzer.analyze_rif_knowledge_base()
        print(f"Total patterns: {len(analyzer.quality_patterns)}")
        for pattern in analyzer.quality_patterns[:5]:  # Show first 5
            print(f"- {pattern.pattern_id}: {pattern.component_type} (success: {pattern.success_rate:.2f})")
    
    elif args.command == "optimal" and args.component_type:
        result = analyzer.find_optimal_thresholds(args.component_type)
        print(json.dumps(result, indent=2))
    
    elif args.command == "insights":
        print("Analyzing GitHub issues for quality insights...")
        insights = analyzer.analyze_github_issues_quality_outcomes()
        print(f"Generated {len(insights)} quality insights")
        for insight in insights[:3]:  # Show first 3
            print(f"- {insight.description} (confidence: {insight.confidence:.2f})")

if __name__ == "__main__":
    main()