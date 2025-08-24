#!/usr/bin/env python3
"""
Adaptive Threshold System for Rule-Based Quality Optimization
Issue #95: Adaptive Threshold Learning System

Main system that orchestrates adaptive threshold learning using rule-based optimization
instead of ML to be compatible with Claude Code's session-based architecture.
"""

import json
import logging
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from .historical_data_collector import HistoricalDataCollector, QualityDecision
from .quality_pattern_analyzer import QualityPatternAnalyzer, QualityPattern
from .threshold_optimizer import ThresholdOptimizer
from .configuration_manager import ConfigurationManager

@dataclass
class AdaptiveThresholdRecommendation:
    """Represents a recommendation for threshold adjustment."""
    component_type: str
    current_threshold: float
    recommended_threshold: float
    confidence: float
    rationale: str
    supporting_evidence: Dict[str, Any]
    risk_assessment: str
    implementation_priority: str  # "high", "medium", "low"
    estimated_impact: Dict[str, float]  # predicted changes in metrics
    
@dataclass
class SystemOptimizationResult:
    """Results of a full system optimization run."""
    optimization_id: str
    timestamp: str
    recommendations: List[AdaptiveThresholdRecommendation]
    overall_confidence: float
    system_health_score: float
    quality_improvement_prediction: float
    implementation_plan: List[Dict[str, Any]]
    rollback_checkpoints: List[str]

class AdaptiveThresholdSystem:
    """
    Main adaptive threshold system that coordinates rule-based threshold optimization.
    
    Features:
    - Analyzes historical quality data for optimization opportunities
    - Generates evidence-based threshold recommendations
    - Provides safe configuration update workflows
    - Tracks optimization effectiveness over time
    - Maintains rollback capabilities for failed optimizations
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 quality_data_dir: str = "quality",
                 knowledge_base_dir: str = "knowledge"):
        """
        Initialize the adaptive threshold system.
        
        Args:
            config_dir: Directory containing configuration files
            quality_data_dir: Directory for quality data storage
            knowledge_base_dir: RIF knowledge base directory
        """
        self.config_dir = Path(config_dir)
        self.quality_data_dir = Path(quality_data_dir)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        
        # Initialize subsystems
        self.data_collector = HistoricalDataCollector(str(self.quality_data_dir / "historical"))
        self.pattern_analyzer = QualityPatternAnalyzer(str(self.knowledge_base_dir), str(self.quality_data_dir / "historical"))
        self.threshold_optimizer = ThresholdOptimizer()
        self.config_manager = ConfigurationManager(str(self.config_dir))
        
        self.setup_logging()
        self._load_system_config()
        
    def setup_logging(self):
        """Setup logging for adaptive threshold system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - AdaptiveThresholdSystem - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_system_config(self):
        """Load system configuration."""
        try:
            config_file = self.config_dir / "adaptive-thresholds.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.system_config = yaml.safe_load(f)
            else:
                self.system_config = self._get_default_system_config()
                self._save_system_config()
            
            self.logger.info("Loaded adaptive threshold system configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {e}")
            self.system_config = self._get_default_system_config()
    
    def _get_default_system_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            "optimization": {
                "min_confidence_threshold": 0.7,
                "max_threshold_change_percent": 20.0,
                "min_historical_data_days": 30,
                "required_sample_size": 10,
                "optimization_frequency_days": 14
            },
            "safety": {
                "require_manual_approval": True,
                "max_simultaneous_changes": 3,
                "rollback_monitoring_hours": 24,
                "performance_degradation_threshold": 0.1
            },
            "component_priorities": {
                "critical_algorithms": "high",
                "public_apis": "high", 
                "business_logic": "medium",
                "integration_code": "medium",
                "ui_components": "low",
                "test_utilities": "low"
            },
            "learning": {
                "pattern_analysis_enabled": True,
                "github_issue_mining": True,
                "continuous_learning": True,
                "success_tracking_days": 90
            }
        }
    
    def _save_system_config(self):
        """Save system configuration."""
        try:
            config_file = self.config_dir / "adaptive-thresholds.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                yaml.dump(self.system_config, f, default_flow_style=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save system configuration: {e}")
    
    def analyze_current_system_performance(self) -> Dict[str, Any]:
        """
        Analyze current system performance to identify optimization opportunities.
        
        Returns:
            Dictionary with system performance analysis
        """
        try:
            self.logger.info("Analyzing current system performance...")
            
            # Get recent quality decisions
            min_days = self.system_config["optimization"]["min_historical_data_days"]
            quality_decisions = self.data_collector.get_quality_decisions(days_back=min_days)
            
            if len(quality_decisions) < self.system_config["optimization"]["required_sample_size"]:
                return {
                    "status": "insufficient_data",
                    "total_decisions": len(quality_decisions),
                    "required_minimum": self.system_config["optimization"]["required_sample_size"],
                    "recommendation": "Continue collecting data before optimization"
                }
            
            # Analyze performance by component type
            component_analysis = {}
            component_types = set(d.component_type for d in quality_decisions)
            
            for component_type in component_types:
                component_decisions = [d for d in quality_decisions if d.component_type == component_type]
                analysis = self._analyze_component_performance(component_type, component_decisions)
                component_analysis[component_type] = analysis
            
            # Calculate overall system health
            overall_health = self._calculate_system_health(component_analysis)
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(component_analysis)
            
            return {
                "status": "analysis_complete",
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
                "data_period_days": min_days,
                "total_decisions": len(quality_decisions),
                "component_types_analyzed": len(component_types),
                "system_health_score": overall_health,
                "component_analysis": component_analysis,
                "optimization_opportunities": optimization_opportunities,
                "recommendations": self._generate_system_recommendations(optimization_opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze system performance: {e}")
            return {
                "status": "error",
                "error": str(e),
                "recommendation": "Check system logs and retry analysis"
            }
    
    def _analyze_component_performance(self, 
                                     component_type: str, 
                                     decisions: List[QualityDecision]) -> Dict[str, Any]:
        """Analyze performance for a specific component type."""
        if not decisions:
            return {
                "sample_size": 0,
                "performance_score": 0.0,
                "issues_identified": ["No decisions found"]
            }
        
        # Calculate basic metrics
        total = len(decisions)
        passed = len([d for d in decisions if d.decision == "pass"])
        failed = len([d for d in decisions if d.decision == "fail"])
        manual_overrides = len([d for d in decisions if d.decision == "manual_override"])
        
        pass_rate = passed / total if total > 0 else 0
        override_rate = manual_overrides / total if total > 0 else 0
        
        # Analyze threshold usage
        thresholds_used = [d.threshold_used for d in decisions]
        avg_threshold = sum(thresholds_used) / len(thresholds_used) if thresholds_used else 0
        
        # Analyze quality scores vs thresholds
        threshold_effectiveness = self._analyze_threshold_effectiveness(decisions)
        
        # Calculate performance score
        performance_score = self._calculate_component_performance_score(
            pass_rate, override_rate, threshold_effectiveness
        )
        
        # Identify issues
        issues = self._identify_component_issues(decisions, pass_rate, override_rate, threshold_effectiveness)
        
        return {
            "sample_size": total,
            "pass_rate": round(pass_rate, 3),
            "fail_rate": round(failed / total, 3) if total > 0 else 0,
            "override_rate": round(override_rate, 3),
            "average_threshold_used": round(avg_threshold, 1),
            "threshold_effectiveness": threshold_effectiveness,
            "performance_score": round(performance_score, 3),
            "issues_identified": issues,
            "optimization_potential": self._calculate_optimization_potential(decisions)
        }
    
    def _analyze_threshold_effectiveness(self, decisions: List[QualityDecision]) -> Dict[str, Any]:
        """Analyze how effective different thresholds are for the decisions."""
        if not decisions:
            return {"status": "no_data"}
        
        # Group by threshold ranges
        threshold_groups = {}
        for decision in decisions:
            threshold_bucket = int(decision.threshold_used / 10) * 10  # Group by 10% buckets
            
            if threshold_bucket not in threshold_groups:
                threshold_groups[threshold_bucket] = {
                    "decisions": [],
                    "pass": 0,
                    "fail": 0,
                    "override": 0
                }
            
            threshold_groups[threshold_bucket]["decisions"].append(decision)
            threshold_groups[threshold_bucket][decision.decision] += 1
        
        # Calculate effectiveness for each group
        effectiveness_analysis = {}
        for threshold, group in threshold_groups.items():
            total = len(group["decisions"])
            if total >= 3:  # Only analyze groups with sufficient data
                pass_rate = group["pass"] / total
                override_rate = group["override"] / total
                
                # Effectiveness = high pass rate with low override rate
                effectiveness = pass_rate - (override_rate * 0.5)
                
                effectiveness_analysis[threshold] = {
                    "sample_size": total,
                    "pass_rate": pass_rate,
                    "override_rate": override_rate,
                    "effectiveness_score": effectiveness
                }
        
        # Find best performing threshold
        best_threshold = None
        best_score = -1
        if effectiveness_analysis:
            best_threshold = max(effectiveness_analysis.keys(), 
                               key=lambda k: effectiveness_analysis[k]["effectiveness_score"])
            best_score = effectiveness_analysis[best_threshold]["effectiveness_score"]
        
        return {
            "status": "analyzed",
            "threshold_groups": effectiveness_analysis,
            "best_performing_threshold": best_threshold,
            "best_effectiveness_score": round(best_score, 3) if best_score >= 0 else None,
            "total_threshold_groups": len(effectiveness_analysis)
        }
    
    def _calculate_component_performance_score(self, 
                                             pass_rate: float, 
                                             override_rate: float, 
                                             threshold_effectiveness: Dict[str, Any]) -> float:
        """Calculate overall performance score for a component."""
        # Base score from pass rate
        score = pass_rate * 0.6
        
        # Penalty for high override rates (indicates threshold too strict)
        score -= override_rate * 0.3
        
        # Bonus for effective threshold usage
        if threshold_effectiveness.get("best_effectiveness_score"):
            score += threshold_effectiveness["best_effectiveness_score"] * 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _identify_component_issues(self, 
                                 decisions: List[QualityDecision],
                                 pass_rate: float,
                                 override_rate: float,
                                 threshold_effectiveness: Dict[str, Any]) -> List[str]:
        """Identify issues with current component configuration."""
        issues = []
        
        # Low pass rate issues
        if pass_rate < 0.6:
            issues.append(f"Low pass rate ({pass_rate:.1%}) suggests thresholds may be too strict")
        
        # High override rate issues
        if override_rate > 0.2:
            issues.append(f"High override rate ({override_rate:.1%}) indicates threshold misalignment")
        
        # Threshold effectiveness issues
        if threshold_effectiveness.get("status") == "analyzed":
            if threshold_effectiveness.get("total_threshold_groups", 0) < 2:
                issues.append("Limited threshold diversity - consider more adaptive thresholds")
            
            best_score = threshold_effectiveness.get("best_effectiveness_score", 0)
            if best_score < 0.5:
                issues.append("No thresholds showing strong effectiveness - major optimization needed")
        
        # Sample size issues
        if len(decisions) < 20:
            issues.append("Limited sample size - results may not be statistically significant")
        
        return issues
    
    def _calculate_optimization_potential(self, decisions: List[QualityDecision]) -> Dict[str, Any]:
        """Calculate potential for optimization based on decision patterns."""
        if not decisions:
            return {"potential": "none", "score": 0.0}
        
        # Calculate variance in threshold usage
        thresholds = [d.threshold_used for d in decisions]
        if len(set(thresholds)) > 1:
            threshold_variance = sum((t - sum(thresholds)/len(thresholds))**2 for t in thresholds) / len(thresholds)
        else:
            threshold_variance = 0
        
        # High variance suggests opportunity for optimization
        variance_potential = min(threshold_variance / 100, 0.3)  # Max 0.3 from variance
        
        # Override rate suggests misconfigured thresholds
        override_count = len([d for d in decisions if d.decision == "manual_override"])
        override_potential = min(override_count / len(decisions), 0.4)  # Max 0.4 from overrides
        
        # Recent failures suggest need for adjustment
        recent_decisions = [d for d in decisions[-10:]]  # Last 10 decisions
        if recent_decisions:
            recent_fail_rate = len([d for d in recent_decisions if d.decision == "fail"]) / len(recent_decisions)
            failure_potential = min(recent_fail_rate, 0.3)  # Max 0.3 from recent failures
        else:
            failure_potential = 0
        
        total_potential = variance_potential + override_potential + failure_potential
        
        if total_potential > 0.6:
            potential_level = "high"
        elif total_potential > 0.3:
            potential_level = "medium"
        elif total_potential > 0.1:
            potential_level = "low"
        else:
            potential_level = "minimal"
        
        return {
            "potential": potential_level,
            "score": round(total_potential, 3),
            "factors": {
                "threshold_variance": round(variance_potential, 3),
                "override_rate": round(override_potential, 3),
                "recent_failures": round(failure_potential, 3)
            }
        }
    
    def _calculate_system_health(self, component_analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        if not component_analysis:
            return 0.0
        
        # Weight components by priority
        priority_weights = {
            "high": 0.4,
            "medium": 0.3,
            "low": 0.2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for component_type, analysis in component_analysis.items():
            priority = self.system_config["component_priorities"].get(component_type, "medium")
            weight = priority_weights.get(priority, 0.3)
            
            performance_score = analysis.get("performance_score", 0)
            weighted_score += performance_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_optimization_opportunities(self, component_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        for component_type, analysis in component_analysis.items():
            optimization_potential = analysis.get("optimization_potential", {})
            potential_score = optimization_potential.get("score", 0)
            
            if potential_score > 0.3:  # Significant optimization potential
                priority = self.system_config["component_priorities"].get(component_type, "medium")
                
                opportunity = {
                    "component_type": component_type,
                    "potential_score": potential_score,
                    "priority": priority,
                    "issues": analysis.get("issues_identified", []),
                    "current_performance": analysis.get("performance_score", 0),
                    "sample_size": analysis.get("sample_size", 0),
                    "recommended_actions": self._generate_component_recommendations(component_type, analysis)
                }
                
                opportunities.append(opportunity)
        
        # Sort by priority and potential
        priority_order = {"high": 3, "medium": 2, "low": 1}
        opportunities.sort(key=lambda x: (priority_order.get(x["priority"], 1), x["potential_score"]), reverse=True)
        
        return opportunities
    
    def _generate_component_recommendations(self, component_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for a component."""
        recommendations = []
        
        issues = analysis.get("issues_identified", [])
        optimization_factors = analysis.get("optimization_potential", {}).get("factors", {})
        
        # Threshold-related recommendations
        if optimization_factors.get("threshold_variance", 0) > 0.1:
            recommendations.append(f"Standardize threshold usage for {component_type} to reduce variance")
        
        if optimization_factors.get("override_rate", 0) > 0.2:
            recommendations.append(f"Lower thresholds for {component_type} to reduce manual overrides")
        
        if optimization_factors.get("recent_failures", 0) > 0.2:
            recommendations.append(f"Investigate recent quality failures and adjust {component_type} thresholds")
        
        # Performance-based recommendations
        performance_score = analysis.get("performance_score", 0)
        if performance_score < 0.6:
            recommendations.append(f"Comprehensive threshold review needed for {component_type}")
        elif performance_score < 0.8:
            recommendations.append(f"Minor threshold adjustments recommended for {component_type}")
        
        # Sample size recommendations
        if analysis.get("sample_size", 0) < 20:
            recommendations.append(f"Continue collecting data for {component_type} before major changes")
        
        return recommendations
    
    def _generate_system_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate overall system recommendations."""
        recommendations = []
        
        if not opportunities:
            recommendations.append("System performance is stable - no immediate optimizations needed")
            return recommendations
        
        # High priority recommendations
        high_priority = [op for op in opportunities if op["priority"] == "high"]
        if high_priority:
            recommendations.append(f"Prioritize optimization for {len(high_priority)} high-priority components")
            for op in high_priority[:3]:  # Top 3 high priority
                recommendations.append(f"- {op['component_type']}: {op['recommended_actions'][0] if op['recommended_actions'] else 'Needs attention'}")
        
        # Overall system recommendations
        total_opportunities = len(opportunities)
        if total_opportunities > 5:
            recommendations.append("Consider phased approach to optimization - many components need attention")
        elif total_opportunities > 2:
            recommendations.append("Moderate optimization effort recommended across multiple components")
        
        # Data quality recommendations
        low_sample_components = [op for op in opportunities if op["sample_size"] < 10]
        if len(low_sample_components) > len(opportunities) / 2:
            recommendations.append("Focus on data collection before optimization - many components lack sufficient data")
        
        return recommendations
    
    def generate_threshold_recommendations(self, 
                                         component_types: Optional[List[str]] = None,
                                         force_analysis: bool = False) -> SystemOptimizationResult:
        """
        Generate threshold recommendations for the system.
        
        Args:
            component_types: Specific components to analyze (None for all)
            force_analysis: Force analysis even with limited data
            
        Returns:
            SystemOptimizationResult with recommendations
        """
        try:
            self.logger.info("Generating threshold recommendations...")
            
            # Analyze current system performance
            performance_analysis = self.analyze_current_system_performance()
            
            if performance_analysis["status"] == "insufficient_data" and not force_analysis:
                return SystemOptimizationResult(
                    optimization_id=f"opt_{int(datetime.utcnow().timestamp())}",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    recommendations=[],
                    overall_confidence=0.0,
                    system_health_score=0.0,
                    quality_improvement_prediction=0.0,
                    implementation_plan=[],
                    rollback_checkpoints=[]
                )
            
            # Get pattern-based recommendations
            self.pattern_analyzer.analyze_rif_knowledge_base()
            
            recommendations = []
            target_components = component_types or list(performance_analysis.get("component_analysis", {}).keys())
            
            for component_type in target_components:
                recommendation = self._generate_component_threshold_recommendation(
                    component_type, performance_analysis
                )
                if recommendation:
                    recommendations.append(recommendation)
            
            # Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(recommendations)
            system_health = performance_analysis.get("system_health_score", 0.0)
            quality_prediction = self._predict_quality_improvement(recommendations)
            
            # Generate implementation plan
            implementation_plan = self._create_implementation_plan(recommendations)
            
            # Create rollback checkpoints
            rollback_checkpoints = self._create_rollback_checkpoints(recommendations)
            
            result = SystemOptimizationResult(
                optimization_id=f"opt_{int(datetime.utcnow().timestamp())}",
                timestamp=datetime.utcnow().isoformat() + "Z",
                recommendations=recommendations,
                overall_confidence=overall_confidence,
                system_health_score=system_health,
                quality_improvement_prediction=quality_prediction,
                implementation_plan=implementation_plan,
                rollback_checkpoints=rollback_checkpoints
            )
            
            # Save optimization results
            self._save_optimization_results(result)
            
            self.logger.info(f"Generated {len(recommendations)} threshold recommendations")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate threshold recommendations: {e}")
            return SystemOptimizationResult(
                optimization_id="error",
                timestamp=datetime.utcnow().isoformat() + "Z",
                recommendations=[],
                overall_confidence=0.0,
                system_health_score=0.0,
                quality_improvement_prediction=0.0,
                implementation_plan=[],
                rollback_checkpoints=[]
            )
    
    def _generate_component_threshold_recommendation(self, 
                                                   component_type: str,
                                                   performance_analysis: Dict[str, Any]) -> Optional[AdaptiveThresholdRecommendation]:
        """Generate threshold recommendation for a specific component."""
        try:
            # Get current configuration
            current_config = self.config_manager.get_current_thresholds()
            current_threshold = current_config.get(component_type, 80.0)
            
            # Get pattern-based optimal threshold
            pattern_result = self.pattern_analyzer.find_optimal_thresholds(component_type)
            
            if pattern_result["recommendation"] == "no_patterns_found":
                # Use performance analysis for recommendation
                component_analysis = performance_analysis.get("component_analysis", {}).get(component_type)
                if not component_analysis:
                    return None
                
                recommended_threshold = self._calculate_threshold_from_performance(
                    current_threshold, component_analysis
                )
                confidence = 0.4
                rationale = "Based on performance analysis (no historical patterns available)"
                
            else:
                recommended_threshold = pattern_result["suggested_threshold"]
                confidence = pattern_result["confidence"]
                rationale = pattern_result["rationale"]
            
            # Apply safety limits
            max_change = self.system_config["optimization"]["max_threshold_change_percent"]
            max_adjustment = current_threshold * (max_change / 100)
            
            if abs(recommended_threshold - current_threshold) > max_adjustment:
                # Limit the change
                if recommended_threshold > current_threshold:
                    recommended_threshold = current_threshold + max_adjustment
                else:
                    recommended_threshold = current_threshold - max_adjustment
                
                rationale += f" (Limited to {max_change}% change for safety)"
                confidence *= 0.8  # Reduce confidence for limited changes
            
            # Only recommend if change is significant and meets confidence threshold
            min_change = 2.0  # Minimum 2% change to be worthwhile
            min_confidence = self.system_config["optimization"]["min_confidence_threshold"]
            
            if abs(recommended_threshold - current_threshold) < min_change or confidence < min_confidence:
                return None
            
            # Create recommendation
            return AdaptiveThresholdRecommendation(
                component_type=component_type,
                current_threshold=current_threshold,
                recommended_threshold=round(recommended_threshold, 1),
                confidence=round(confidence, 2),
                rationale=rationale,
                supporting_evidence=self._gather_supporting_evidence(component_type, performance_analysis),
                risk_assessment=self._assess_threshold_change_risk(current_threshold, recommended_threshold),
                implementation_priority=self.system_config["component_priorities"].get(component_type, "medium"),
                estimated_impact=self._estimate_threshold_impact(component_type, current_threshold, recommended_threshold)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation for {component_type}: {e}")
            return None
    
    def _calculate_threshold_from_performance(self, current_threshold: float, analysis: Dict[str, Any]) -> float:
        """Calculate recommended threshold based on performance analysis."""
        performance_score = analysis.get("performance_score", 0.5)
        override_rate = analysis.get("override_rate", 0)
        pass_rate = analysis.get("pass_rate", 0.8)
        
        # If performance is poor, adjust threshold
        if performance_score < 0.6:
            if override_rate > 0.2:  # Too many overrides, lower threshold
                return current_threshold * 0.9
            elif pass_rate < 0.6:  # Too many failures, might need higher or lower threshold
                return current_threshold * 1.05  # Try slightly higher first
        elif performance_score > 0.8 and override_rate < 0.1:
            # Good performance, maybe can afford to be more strict
            return current_threshold * 1.03
        
        return current_threshold  # No change recommended
    
    def _gather_supporting_evidence(self, component_type: str, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gather supporting evidence for the recommendation."""
        evidence = {}
        
        component_analysis = performance_analysis.get("component_analysis", {}).get(component_type, {})
        evidence["performance_metrics"] = {
            "current_performance_score": component_analysis.get("performance_score"),
            "pass_rate": component_analysis.get("pass_rate"),
            "override_rate": component_analysis.get("override_rate"),
            "sample_size": component_analysis.get("sample_size")
        }
        
        # Add pattern evidence if available
        pattern_result = self.pattern_analyzer.find_optimal_thresholds(component_type)
        if pattern_result["recommendation"] != "no_patterns_found":
            evidence["pattern_analysis"] = {
                "patterns_used": pattern_result.get("patterns_used", 0),
                "pattern_confidence": pattern_result.get("confidence", 0)
            }
        
        return evidence
    
    def _assess_threshold_change_risk(self, current: float, recommended: float) -> str:
        """Assess risk level of threshold change."""
        change_percent = abs(recommended - current) / current * 100
        
        if change_percent > 15:
            return "high"
        elif change_percent > 8:
            return "medium"
        else:
            return "low"
    
    def _estimate_threshold_impact(self, 
                                 component_type: str,
                                 current: float,
                                 recommended: float) -> Dict[str, float]:
        """Estimate impact of threshold change."""
        change_ratio = recommended / current
        
        # Estimate impact on different metrics
        if recommended > current:  # Stricter threshold
            impact = {
                "pass_rate_change": -0.1 * (change_ratio - 1),  # Fewer passes
                "quality_improvement": 0.15 * (change_ratio - 1),  # Better quality
                "false_positive_change": 0.05 * (change_ratio - 1)  # More false positives
            }
        else:  # More lenient threshold
            impact = {
                "pass_rate_change": 0.1 * (1 - change_ratio),  # More passes
                "quality_improvement": -0.1 * (1 - change_ratio),  # Potentially lower quality
                "false_positive_change": -0.05 * (1 - change_ratio)  # Fewer false positives
            }
        
        return {k: round(v, 3) for k, v in impact.items()}
    
    def _calculate_overall_confidence(self, recommendations: List[AdaptiveThresholdRecommendation]) -> float:
        """Calculate overall confidence in recommendations."""
        if not recommendations:
            return 0.0
        
        # Weight by component priority
        priority_weights = {"high": 0.5, "medium": 0.3, "low": 0.2}
        
        total_weight = 0
        weighted_confidence = 0
        
        for rec in recommendations:
            weight = priority_weights.get(rec.implementation_priority, 0.3)
            weighted_confidence += rec.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _predict_quality_improvement(self, recommendations: List[AdaptiveThresholdRecommendation]) -> float:
        """Predict overall quality improvement from recommendations."""
        if not recommendations:
            return 0.0
        
        total_improvement = 0
        for rec in recommendations:
            quality_impact = rec.estimated_impact.get("quality_improvement", 0)
            confidence_weight = rec.confidence
            total_improvement += quality_impact * confidence_weight
        
        return min(total_improvement, 0.25)  # Cap at 25% improvement prediction
    
    def _create_implementation_plan(self, recommendations: List[AdaptiveThresholdRecommendation]) -> List[Dict[str, Any]]:
        """Create implementation plan for recommendations."""
        plan = []
        
        # Sort by priority and risk
        sorted_recs = sorted(recommendations, 
                           key=lambda x: ({"high": 3, "medium": 2, "low": 1}[x.implementation_priority], -x.confidence),
                           reverse=True)
        
        for i, rec in enumerate(sorted_recs):
            plan_item = {
                "step": i + 1,
                "component_type": rec.component_type,
                "action": "update_threshold",
                "current_value": rec.current_threshold,
                "new_value": rec.recommended_threshold,
                "requires_approval": self.system_config["safety"]["require_manual_approval"],
                "risk_level": rec.risk_assessment,
                "estimated_duration_hours": 1 if rec.risk_assessment == "low" else 2,
                "rollback_checkpoint": f"checkpoint_{rec.component_type}_{int(datetime.utcnow().timestamp())}",
                "monitoring_period_hours": self.system_config["safety"]["rollback_monitoring_hours"],
                "success_criteria": {
                    "pass_rate_maintained": rec.current_threshold * 0.9,  # Don't drop pass rate too much
                    "no_performance_degradation": True,
                    "manual_override_rate_threshold": 0.3
                }
            }
            plan.append(plan_item)
        
        return plan
    
    def _create_rollback_checkpoints(self, recommendations: List[AdaptiveThresholdRecommendation]) -> List[str]:
        """Create rollback checkpoints for recommendations."""
        checkpoints = []
        
        for rec in recommendations:
            checkpoint_id = f"checkpoint_{rec.component_type}_{int(datetime.utcnow().timestamp())}"
            checkpoints.append(checkpoint_id)
            
            # Save current configuration as checkpoint
            self.config_manager.create_checkpoint(
                checkpoint_id,
                {rec.component_type: rec.current_threshold},
                f"Pre-optimization checkpoint for {rec.component_type}"
            )
        
        return checkpoints
    
    def _save_optimization_results(self, result: SystemOptimizationResult):
        """Save optimization results for tracking."""
        try:
            results_file = self.quality_data_dir / "reports" / "optimization_results.jsonl"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'a') as f:
                f.write(json.dumps(asdict(result), default=str) + "\n")
            
            self.logger.info(f"Saved optimization results: {result.optimization_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

def main():
    """Command line interface for adaptive threshold system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Threshold System")
    parser.add_argument("--command", choices=["analyze", "optimize", "status"], required=True,
                       help="Command to execute")
    parser.add_argument("--component-type", help="Specific component type to analyze")
    parser.add_argument("--force", action="store_true", help="Force analysis with limited data")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    system = AdaptiveThresholdSystem()
    
    if args.command == "analyze":
        print("Analyzing system performance...")
        analysis = system.analyze_current_system_performance()
        print(json.dumps(analysis, indent=2, default=str))
    
    elif args.command == "optimize":
        print("Generating threshold recommendations...")
        component_types = [args.component_type] if args.component_type else None
        result = system.generate_threshold_recommendations(component_types, args.force)
        
        print(f"Generated {len(result.recommendations)} recommendations")
        print(f"Overall confidence: {result.overall_confidence:.2f}")
        print(f"System health: {result.system_health_score:.2f}")
        print(f"Predicted quality improvement: {result.quality_improvement_prediction:.1%}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            print(f"Saved results to {args.output}")
    
    elif args.command == "status":
        print("System Status:")
        print("- Adaptive threshold system initialized")
        print("- Configuration loaded")
        print("- Ready for optimization analysis")

if __name__ == "__main__":
    main()