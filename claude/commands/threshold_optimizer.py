#!/usr/bin/env python3
"""
Threshold Optimizer for Adaptive Threshold Learning System
Issue #95: Adaptive Threshold Learning System

Implements rule-based threshold optimization algorithms using historical data
and quality patterns instead of ML models for Claude Code compatibility.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class OptimizationRule:
    """Represents a rule for threshold optimization."""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int
    confidence_impact: float

@dataclass 
class ThresholdOptimizationResult:
    """Result of threshold optimization."""
    component_type: str
    original_threshold: float
    optimized_threshold: float
    confidence: float
    optimization_method: str
    rules_applied: List[str]
    supporting_data: Dict[str, Any]
    risk_assessment: str

class ThresholdOptimizer:
    """
    Rule-based threshold optimizer for adaptive quality gates.
    
    Features:
    - Rule-based optimization algorithms
    - Historical performance analysis
    - Statistical trend analysis
    - Context-aware adjustments
    - Safety constraints and validation
    """
    
    def __init__(self):
        """Initialize the threshold optimizer."""
        self.setup_logging()
        self.optimization_rules = self._load_optimization_rules()
        
    def setup_logging(self):
        """Setup logging for threshold optimizer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ThresholdOptimizer - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_optimization_rules(self) -> List[OptimizationRule]:
        """Load optimization rules for threshold adjustment."""
        rules = [
            # Performance-based rules
            OptimizationRule(
                rule_id="low_pass_rate",
                name="Low Pass Rate Adjustment",
                description="Lower threshold when pass rate is consistently low",
                condition={"pass_rate": "<0.6", "sample_size": ">10"},
                action={"adjustment": -0.1, "max_change": 0.15},
                priority=8,
                confidence_impact=0.8
            ),
            
            OptimizationRule(
                rule_id="high_override_rate",
                name="High Override Rate Adjustment", 
                description="Lower threshold when manual override rate is high",
                condition={"override_rate": ">0.25", "sample_size": ">5"},
                action={"adjustment": -0.12, "max_change": 0.2},
                priority=9,
                confidence_impact=0.85
            ),
            
            OptimizationRule(
                rule_id="excellent_performance",
                name="Excellent Performance Optimization",
                description="Slightly increase threshold when performance is excellent",
                condition={"pass_rate": ">0.9", "override_rate": "<0.1", "sample_size": ">20"},
                action={"adjustment": 0.05, "max_change": 0.08},
                priority=4,
                confidence_impact=0.7
            ),
            
            # Trend-based rules
            OptimizationRule(
                rule_id="improving_trend",
                name="Improving Quality Trend",
                description="Gradually increase threshold when quality is consistently improving",
                condition={"quality_trend": "improving", "trend_periods": ">3"},
                action={"adjustment": 0.03, "max_change": 0.06},
                priority=5,
                confidence_impact=0.6
            ),
            
            OptimizationRule(
                rule_id="degrading_trend",
                name="Degrading Quality Trend",
                description="Lower threshold when quality is degrading to maintain pass rates",
                condition={"quality_trend": "degrading", "trend_periods": ">2"},
                action={"adjustment": -0.08, "max_change": 0.12},
                priority=7,
                confidence_impact=0.75
            ),
            
            # Context-based rules
            OptimizationRule(
                rule_id="critical_component_safety",
                name="Critical Component Safety",
                description="Maintain higher thresholds for critical components",
                condition={"component_type": "critical_algorithms", "threshold": "<0.9"},
                action={"adjustment": 0.1, "min_threshold": 0.9},
                priority=10,
                confidence_impact=0.9
            ),
            
            OptimizationRule(
                rule_id="test_utility_leniency",
                name="Test Utility Leniency",
                description="Allow lower thresholds for test utilities",
                condition={"component_type": "test_utilities", "threshold": ">0.8"},
                action={"adjustment": -0.1, "min_threshold": 0.6},
                priority=3,
                confidence_impact=0.8
            ),
            
            # Statistical rules
            OptimizationRule(
                rule_id="threshold_variance_optimization",
                name="Threshold Variance Optimization",
                description="Optimize when threshold usage shows high variance",
                condition={"threshold_variance": ">15", "sample_size": ">15"},
                action={"method": "statistical_optimization"},
                priority=6,
                confidence_impact=0.65
            ),
            
            # Safety rules
            OptimizationRule(
                rule_id="insufficient_data_safety",
                name="Insufficient Data Safety",
                description="Conservative approach when sample size is small",
                condition={"sample_size": "<10"},
                action={"adjustment": 0, "require_more_data": True},
                priority=2,
                confidence_impact=0.3
            ),
            
            OptimizationRule(
                rule_id="recent_failure_safety",
                name="Recent Failure Safety",
                description="Cautious adjustment after recent failures",
                condition={"recent_failures": ">0.3", "time_period": "<7_days"},
                action={"adjustment": -0.05, "max_change": 0.1},
                priority=8,
                confidence_impact=0.7
            )
        ]
        
        self.logger.info(f"Loaded {len(rules)} optimization rules")
        return rules
    
    def optimize_threshold(self, 
                          component_type: str,
                          current_threshold: float,
                          historical_data: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> ThresholdOptimizationResult:
        """
        Optimize threshold for a component using rule-based analysis.
        
        Args:
            component_type: Type of component to optimize
            current_threshold: Current threshold value (0-100)
            historical_data: Historical performance data
            context: Optional context information
            
        Returns:
            ThresholdOptimizationResult with optimization recommendation
        """
        try:
            self.logger.info(f"Optimizing threshold for {component_type} (current: {current_threshold}%)")
            
            # Prepare analysis context
            analysis_context = self._prepare_analysis_context(
                component_type, current_threshold, historical_data, context
            )
            
            # Apply optimization rules
            applicable_rules = self._find_applicable_rules(analysis_context)
            
            if not applicable_rules:
                return ThresholdOptimizationResult(
                    component_type=component_type,
                    original_threshold=current_threshold,
                    optimized_threshold=current_threshold,
                    confidence=0.3,
                    optimization_method="no_rules_applicable",
                    rules_applied=[],
                    supporting_data=analysis_context,
                    risk_assessment="low"
                )
            
            # Calculate optimization
            optimization_result = self._apply_optimization_rules(
                current_threshold, applicable_rules, analysis_context
            )
            
            # Validate and constrain result
            final_threshold = self._apply_safety_constraints(
                current_threshold, optimization_result["optimized_threshold"], analysis_context
            )
            
            # Calculate confidence
            confidence = self._calculate_optimization_confidence(
                applicable_rules, analysis_context, final_threshold - current_threshold
            )
            
            # Assess risk
            risk_assessment = self._assess_optimization_risk(
                current_threshold, final_threshold, analysis_context
            )
            
            result = ThresholdOptimizationResult(
                component_type=component_type,
                original_threshold=current_threshold,
                optimized_threshold=final_threshold,
                confidence=confidence,
                optimization_method=optimization_result["method"],
                rules_applied=[rule.rule_id for rule in applicable_rules],
                supporting_data=analysis_context,
                risk_assessment=risk_assessment
            )
            
            self.logger.info(f"Optimization complete: {current_threshold}% -> {final_threshold}% (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Threshold optimization failed: {e}")
            return ThresholdOptimizationResult(
                component_type=component_type,
                original_threshold=current_threshold,
                optimized_threshold=current_threshold,
                confidence=0.0,
                optimization_method="error",
                rules_applied=[],
                supporting_data={"error": str(e)},
                risk_assessment="unknown"
            )
    
    def _prepare_analysis_context(self, 
                                component_type: str,
                                current_threshold: float, 
                                historical_data: Dict[str, Any],
                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare analysis context from available data."""
        analysis_context = {
            "component_type": component_type,
            "current_threshold": current_threshold,
            "sample_size": historical_data.get("sample_size", 0),
            "pass_rate": historical_data.get("pass_rate", 0.8),
            "override_rate": historical_data.get("override_rate", 0.0),
            "performance_score": historical_data.get("performance_score", 0.7),
        }
        
        # Add trend analysis
        analysis_context.update(self._analyze_quality_trends(historical_data))
        
        # Add statistical analysis
        analysis_context.update(self._calculate_statistical_metrics(historical_data))
        
        # Add context if provided
        if context:
            analysis_context.update(context)
        
        return analysis_context
    
    def _analyze_quality_trends(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality trends from historical data."""
        # Simulate trend analysis (in real implementation, this would analyze time series data)
        performance_score = historical_data.get("performance_score", 0.7)
        
        if performance_score > 0.8:
            quality_trend = "improving"
            trend_strength = min((performance_score - 0.8) * 5, 1.0)
        elif performance_score < 0.6:
            quality_trend = "degrading"  
            trend_strength = min((0.6 - performance_score) * 5, 1.0)
        else:
            quality_trend = "stable"
            trend_strength = 0.5
        
        return {
            "quality_trend": quality_trend,
            "trend_strength": trend_strength,
            "trend_periods": 3  # Simulated - would be calculated from actual time series
        }
    
    def _calculate_statistical_metrics(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical metrics for optimization."""
        # Simulate statistical calculations
        sample_size = historical_data.get("sample_size", 0)
        
        if sample_size > 10:
            # Simulate threshold variance calculation
            threshold_variance = 8.0 + (sample_size % 20)  # Simulated variance
            statistical_confidence = min(sample_size / 50, 1.0)
        else:
            threshold_variance = 0.0
            statistical_confidence = 0.2
        
        # Simulate recent failure analysis
        performance_score = historical_data.get("performance_score", 0.7)
        recent_failures = max(0, (0.8 - performance_score) * 2)  # Simulate failure rate
        
        return {
            "threshold_variance": threshold_variance,
            "statistical_confidence": statistical_confidence,
            "recent_failures": recent_failures
        }
    
    def _find_applicable_rules(self, analysis_context: Dict[str, Any]) -> List[OptimizationRule]:
        """Find optimization rules applicable to the current context."""
        applicable_rules = []
        
        for rule in self.optimization_rules:
            if self._rule_matches_context(rule, analysis_context):
                applicable_rules.append(rule)
        
        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        self.logger.debug(f"Found {len(applicable_rules)} applicable rules")
        return applicable_rules
    
    def _rule_matches_context(self, rule: OptimizationRule, context: Dict[str, Any]) -> bool:
        """Check if a rule matches the current context."""
        try:
            for condition_key, condition_value in rule.condition.items():
                if condition_key not in context:
                    return False
                
                context_value = context[condition_key]
                
                # Handle different condition types
                if isinstance(condition_value, str):
                    if condition_value.startswith('>'):
                        threshold = float(condition_value[1:])
                        if not (isinstance(context_value, (int, float)) and context_value > threshold):
                            return False
                    elif condition_value.startswith('<'):
                        threshold = float(condition_value[1:])
                        if not (isinstance(context_value, (int, float)) and context_value < threshold):
                            return False
                    elif condition_value != str(context_value):
                        return False
                elif condition_value != context_value:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Rule matching error for {rule.rule_id}: {e}")
            return False
    
    def _apply_optimization_rules(self, 
                                current_threshold: float,
                                applicable_rules: List[OptimizationRule],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization rules to calculate new threshold."""
        if not applicable_rules:
            return {
                "optimized_threshold": current_threshold,
                "method": "no_change",
                "adjustments": []
            }
        
        adjustments = []
        total_adjustment = 0.0
        optimization_methods = []
        
        for rule in applicable_rules:
            action = rule.action
            
            if "adjustment" in action:
                # Direct adjustment
                adjustment = float(action["adjustment"])
                
                # Apply confidence weighting
                weighted_adjustment = adjustment * rule.confidence_impact
                
                # Apply max change limits
                max_change = action.get("max_change", 0.2)
                if abs(weighted_adjustment) > max_change:
                    weighted_adjustment = math.copysign(max_change, weighted_adjustment)
                
                total_adjustment += weighted_adjustment
                adjustments.append({
                    "rule": rule.rule_id,
                    "raw_adjustment": adjustment,
                    "weighted_adjustment": weighted_adjustment,
                    "confidence_impact": rule.confidence_impact
                })
                optimization_methods.append("rule_based")
            
            elif "method" in action:
                # Special optimization method
                method = action["method"]
                if method == "statistical_optimization":
                    stat_adjustment = self._apply_statistical_optimization(current_threshold, context)
                    total_adjustment += stat_adjustment
                    adjustments.append({
                        "rule": rule.rule_id,
                        "method": method,
                        "adjustment": stat_adjustment
                    })
                    optimization_methods.append("statistical")
        
        # Calculate final threshold
        optimized_threshold = current_threshold + (total_adjustment * current_threshold)
        
        # Ensure reasonable bounds
        optimized_threshold = max(50.0, min(100.0, optimized_threshold))
        
        return {
            "optimized_threshold": optimized_threshold,
            "method": "_".join(set(optimization_methods)) if optimization_methods else "rule_based",
            "adjustments": adjustments,
            "total_adjustment": total_adjustment
        }
    
    def _apply_statistical_optimization(self, current_threshold: float, context: Dict[str, Any]) -> float:
        """Apply statistical optimization method."""
        # Simple statistical optimization based on variance
        variance = context.get("threshold_variance", 0)
        sample_size = context.get("sample_size", 0)
        
        if variance > 10 and sample_size > 15:
            # High variance suggests suboptimal threshold
            performance_score = context.get("performance_score", 0.7)
            
            if performance_score > 0.7:
                # Performance is good, try to reduce variance by slight increase
                return 0.03
            else:
                # Performance is poor, try to reduce variance by slight decrease
                return -0.05
        
        return 0.0
    
    def _apply_safety_constraints(self, 
                                current_threshold: float,
                                optimized_threshold: float,
                                context: Dict[str, Any]) -> float:
        """Apply safety constraints to optimization result."""
        # Maximum change per optimization
        max_change_percent = 20.0
        max_change = current_threshold * (max_change_percent / 100)
        
        if abs(optimized_threshold - current_threshold) > max_change:
            if optimized_threshold > current_threshold:
                constrained_threshold = current_threshold + max_change
            else:
                constrained_threshold = current_threshold - max_change
            
            self.logger.info(f"Applied safety constraint: {optimized_threshold:.1f} -> {constrained_threshold:.1f}")
            optimized_threshold = constrained_threshold
        
        # Component-specific minimums
        component_type = context.get("component_type", "")
        
        component_minimums = {
            "critical_algorithms": 90.0,
            "public_apis": 85.0,
            "business_logic": 80.0,
            "integration_code": 75.0,
            "ui_components": 65.0,
            "test_utilities": 60.0
        }
        
        min_threshold = component_minimums.get(component_type, 70.0)
        if optimized_threshold < min_threshold:
            self.logger.info(f"Applied component minimum: {optimized_threshold:.1f} -> {min_threshold:.1f}")
            optimized_threshold = min_threshold
        
        # Absolute bounds
        optimized_threshold = max(50.0, min(100.0, optimized_threshold))
        
        return round(optimized_threshold, 1)
    
    def _calculate_optimization_confidence(self, 
                                         applicable_rules: List[OptimizationRule],
                                         context: Dict[str, Any],
                                         threshold_change: float) -> float:
        """Calculate confidence in the optimization result."""
        if not applicable_rules:
            return 0.3
        
        # Base confidence from rules
        rule_confidences = [rule.confidence_impact for rule in applicable_rules]
        base_confidence = sum(rule_confidences) / len(rule_confidences)
        
        # Adjust for sample size
        sample_size = context.get("sample_size", 0)
        if sample_size >= 20:
            sample_confidence = 1.0
        elif sample_size >= 10:
            sample_confidence = 0.8
        elif sample_size >= 5:
            sample_confidence = 0.6
        else:
            sample_confidence = 0.3
        
        # Adjust for magnitude of change
        change_confidence = 1.0 - min(abs(threshold_change) / 20, 0.5)  # Penalize large changes
        
        # Statistical confidence
        statistical_confidence = context.get("statistical_confidence", 0.5)
        
        # Combined confidence
        combined_confidence = (
            base_confidence * 0.4 +
            sample_confidence * 0.3 + 
            change_confidence * 0.2 +
            statistical_confidence * 0.1
        )
        
        return round(min(combined_confidence, 1.0), 2)
    
    def _assess_optimization_risk(self, 
                                current_threshold: float,
                                optimized_threshold: float,
                                context: Dict[str, Any]) -> str:
        """Assess risk level of the optimization."""
        change_percent = abs(optimized_threshold - current_threshold) / current_threshold * 100
        
        # Risk factors
        sample_size = context.get("sample_size", 0)
        performance_score = context.get("performance_score", 0.7)
        component_type = context.get("component_type", "")
        
        risk_score = 0
        
        # Change magnitude risk
        if change_percent > 15:
            risk_score += 3
        elif change_percent > 8:
            risk_score += 2
        elif change_percent > 3:
            risk_score += 1
        
        # Sample size risk
        if sample_size < 10:
            risk_score += 2
        elif sample_size < 20:
            risk_score += 1
        
        # Performance risk
        if performance_score < 0.6:
            risk_score += 2
        elif performance_score < 0.7:
            risk_score += 1
        
        # Component type risk
        if component_type in ["critical_algorithms", "public_apis"]:
            risk_score += 1
        
        # Risk assessment
        if risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def batch_optimize_thresholds(self, 
                                 threshold_configs: Dict[str, float],
                                 historical_data: Dict[str, Dict[str, Any]],
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, ThresholdOptimizationResult]:
        """
        Optimize thresholds for multiple components.
        
        Args:
            threshold_configs: Current threshold configurations
            historical_data: Historical data per component type
            context: Optional shared context
            
        Returns:
            Dictionary of optimization results per component
        """
        results = {}
        
        for component_type, current_threshold in threshold_configs.items():
            component_data = historical_data.get(component_type, {})
            component_context = dict(context) if context else {}
            
            # Add component-specific context
            component_context["component_type"] = component_type
            
            result = self.optimize_threshold(
                component_type, current_threshold, component_data, component_context
            )
            results[component_type] = result
        
        self.logger.info(f"Batch optimization complete for {len(results)} components")
        return results
    
    def explain_optimization(self, result: ThresholdOptimizationResult) -> Dict[str, Any]:
        """
        Explain the reasoning behind an optimization result.
        
        Args:
            result: Optimization result to explain
            
        Returns:
            Dictionary with detailed explanation
        """
        explanation = {
            "component_type": result.component_type,
            "change_summary": {
                "original_threshold": result.original_threshold,
                "optimized_threshold": result.optimized_threshold,
                "change_amount": result.optimized_threshold - result.original_threshold,
                "change_percent": ((result.optimized_threshold - result.original_threshold) / result.original_threshold) * 100
            },
            "optimization_rationale": [],
            "rules_explanation": [],
            "risk_factors": [],
            "confidence_factors": []
        }
        
        # Explain rules applied
        for rule_id in result.rules_applied:
            rule = next((r for r in self.optimization_rules if r.rule_id == rule_id), None)
            if rule:
                explanation["rules_explanation"].append({
                    "rule_name": rule.name,
                    "description": rule.description,
                    "priority": rule.priority,
                    "confidence_impact": rule.confidence_impact
                })
        
        # Explain optimization rationale based on method
        if result.optimization_method == "rule_based":
            explanation["optimization_rationale"].append("Optimization based on rule-based analysis of historical performance")
        elif "statistical" in result.optimization_method:
            explanation["optimization_rationale"].append("Statistical optimization applied to reduce threshold variance")
        
        # Explain risk assessment
        risk_level = result.risk_assessment
        if risk_level == "high":
            explanation["risk_factors"].append("High risk due to significant threshold change or limited data")
        elif risk_level == "medium":
            explanation["risk_factors"].append("Medium risk requires monitoring after implementation")
        else:
            explanation["risk_factors"].append("Low risk optimization with good data support")
        
        # Explain confidence factors
        confidence = result.confidence
        if confidence > 0.8:
            explanation["confidence_factors"].append("High confidence based on strong rule matches and sufficient data")
        elif confidence > 0.6:
            explanation["confidence_factors"].append("Moderate confidence with good supporting evidence")
        else:
            explanation["confidence_factors"].append("Lower confidence due to limited data or weak rule matches")
        
        return explanation

def main():
    """Command line interface for threshold optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Threshold Optimizer")
    parser.add_argument("--component-type", required=True, help="Component type to optimize")
    parser.add_argument("--current-threshold", type=float, required=True, help="Current threshold")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size for simulation")
    parser.add_argument("--pass-rate", type=float, default=0.8, help="Pass rate for simulation")
    parser.add_argument("--override-rate", type=float, default=0.1, help="Override rate for simulation")
    
    args = parser.parse_args()
    
    optimizer = ThresholdOptimizer()
    
    # Create simulated historical data
    historical_data = {
        "sample_size": args.sample_size,
        "pass_rate": args.pass_rate,
        "override_rate": args.override_rate,
        "performance_score": args.pass_rate * 0.8 + (1 - args.override_rate) * 0.2
    }
    
    # Optimize threshold
    result = optimizer.optimize_threshold(
        args.component_type,
        args.current_threshold,
        historical_data
    )
    
    print(f"Optimization Result:")
    print(f"  Component: {result.component_type}")
    print(f"  Original: {result.original_threshold}%")
    print(f"  Optimized: {result.optimized_threshold}%")
    print(f"  Change: {result.optimized_threshold - result.original_threshold:+.1f}%")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Method: {result.optimization_method}")
    print(f"  Risk: {result.risk_assessment}")
    print(f"  Rules Applied: {', '.join(result.rules_applied)}")
    
    # Show explanation
    explanation = optimizer.explain_optimization(result)
    print(f"\nOptimization Explanation:")
    for rationale in explanation["optimization_rationale"]:
        print(f"  - {rationale}")
    
    if explanation["rules_explanation"]:
        print(f"\nRules Applied:")
        for rule_exp in explanation["rules_explanation"]:
            print(f"  - {rule_exp['rule_name']}: {rule_exp['description']}")

if __name__ == "__main__":
    main()