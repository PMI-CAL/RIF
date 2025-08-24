#!/usr/bin/env python3
"""
Integrated Risk Assessment - Issue #93 Phase 3
Unified risk assessment system integrating all risk dimensions for quality scoring.

This module provides:
- Integration of security, performance, and architectural risk assessments
- Comprehensive risk multiplier calculation for quality scoring
- Risk correlation analysis and weighting
- Unified mitigation strategy generation
"""

import os
import json
import yaml
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import statistics

# Import Phase 3 risk assessors
try:
    from .security_risk_detector import SecurityRiskDetector, SecurityRiskProfile
    from .performance_risk_calculator import PerformanceRiskCalculator, PerformanceRiskProfile
    from .architectural_risk_assessor import ArchitecturalRiskAssessor, ArchitecturalRiskProfile
    from .risk_adjustment_calculator import RiskAdjustmentCalculator, RiskAssessment
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from security_risk_detector import SecurityRiskDetector, SecurityRiskProfile
    from performance_risk_calculator import PerformanceRiskCalculator, PerformanceRiskProfile
    from architectural_risk_assessor import ArchitecturalRiskAssessor, ArchitecturalRiskProfile
    from risk_adjustment_calculator import RiskAdjustmentCalculator, RiskAssessment

@dataclass
class IntegratedRiskProfile:
    """Comprehensive integrated risk profile."""
    overall_risk_score: float = 0.0
    risk_multiplier: float = 0.0  # For quality score adjustment
    risk_level: str = "low"  # low, medium, high, critical
    
    # Individual risk component scores
    security_risk: float = 0.0
    performance_risk: float = 0.0
    architectural_risk: float = 0.0
    change_risk: float = 0.0
    
    # Risk correlation analysis
    risk_correlations: Dict[str, float] = field(default_factory=dict)
    compound_risks: List[str] = field(default_factory=list)
    risk_hotspots: List[str] = field(default_factory=list)
    
    # Comprehensive mitigation strategies
    priority_mitigations: List[str] = field(default_factory=list)
    risk_mitigation_map: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    assessment_time_ms: float = 0.0
    confidence: float = 1.0
    individual_profiles: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskConfiguration:
    """Configuration for integrated risk assessment."""
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        'security': 0.35,
        'performance': 0.25,
        'architectural': 0.25,
        'change': 0.15
    })
    correlation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high_correlation': 0.7,
        'medium_correlation': 0.4,
        'low_correlation': 0.2
    })
    risk_level_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.75,
        'critical': 0.9
    })
    max_risk_multiplier: float = 0.3

class IntegratedRiskAssessment:
    """
    Integrated risk assessment system for multi-dimensional quality scoring.
    
    Combines security, performance, architectural, and change risk assessments
    to provide a unified risk multiplier for quality score adjustment.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize integrated risk assessment system."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/integrated-risk-config.yaml"
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_configuration()
        
        # Initialize individual risk assessors
        self.security_detector = SecurityRiskDetector()
        self.performance_calculator = PerformanceRiskCalculator()
        self.architectural_assessor = ArchitecturalRiskAssessor()
        self.change_calculator = RiskAdjustmentCalculator()
        
        # Risk correlation patterns (learned from historical data)
        self.risk_correlations = {
            'security_performance': 0.3,    # Security changes often impact performance
            'security_architectural': 0.6,  # Security changes often require architecture changes
            'performance_architectural': 0.5, # Performance issues often indicate architecture problems
            'architectural_change': 0.4     # Large changes often involve architecture
        }
    
    def _load_configuration(self) -> RiskConfiguration:
        """Load integrated risk configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    return RiskConfiguration(**config_data)
            else:
                # Create default configuration file
                default_config = RiskConfiguration()
                self._save_default_configuration(default_config)
                return default_config
        except Exception as e:
            self.logger.warning(f"Failed to load risk configuration: {e}")
            return RiskConfiguration()
    
    def _save_default_configuration(self, config: RiskConfiguration):
        """Save default configuration to file."""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'risk_weights': config.risk_weights,
                'correlation_thresholds': config.correlation_thresholds,
                'risk_level_thresholds': config.risk_level_thresholds,
                'max_risk_multiplier': config.max_risk_multiplier
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save default configuration: {e}")
    
    def assess_integrated_risk(self,
                             files: List[str],
                             context: Optional[str] = None,
                             vulnerability_data: Optional[Dict] = None,
                             performance_data: Optional[Dict] = None,
                             performance_baseline: Optional[Dict] = None,
                             project_context: Optional[Dict] = None) -> IntegratedRiskProfile:
        """
        Perform comprehensive integrated risk assessment.
        
        Args:
            files: List of files to analyze
            context: Component context for risk weighting
            vulnerability_data: Security vulnerability scan results
            performance_data: Performance measurement data
            performance_baseline: Performance baseline for comparison
            project_context: Project context information
            
        Returns:
            IntegratedRiskProfile with comprehensive risk assessment
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(files, context, vulnerability_data, 
                                           performance_data, project_context)
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                self.logger.info("Returning cached integrated risk assessment")
                return cached_result
        
        try:
            # Perform individual risk assessments in parallel (conceptually)
            self.logger.info("Starting integrated risk assessment")
            
            # Security risk assessment
            security_profile = self.security_detector.analyze_security_risk(
                files=files,
                vulnerability_data=vulnerability_data,
                context=context
            )
            
            # Performance risk assessment
            performance_profile = self.performance_calculator.calculate_performance_risk(
                files=files,
                performance_data=performance_data,
                baseline=performance_baseline,
                context=context
            )
            
            # Architectural risk assessment
            architectural_profile = self.architectural_assessor.assess_architectural_risk(
                files=files,
                project_context=project_context
            )
            
            # Change-based risk assessment
            change_assessment = self.change_calculator.assess_change_risk(
                files=files,
                security_files=self._identify_security_files(files),
                test_coverage=performance_data.get('test_coverage', 80.0) if performance_data else 80.0
            )
            
            # Integrate all risk assessments
            integrated_profile = self._integrate_risk_profiles(
                security_profile=security_profile,
                performance_profile=performance_profile,
                architectural_profile=architectural_profile,
                change_assessment=change_assessment,
                context=context
            )
            
            # Store individual profiles for reference
            integrated_profile.individual_profiles = {
                'security': security_profile,
                'performance': performance_profile,
                'architectural': architectural_profile,
                'change': change_assessment
            }
            
            integrated_profile.assessment_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self.cache[cache_key] = (integrated_profile, datetime.now())
            self._cleanup_cache()
            
            self.logger.info(f"Integrated risk assessment completed in {integrated_profile.assessment_time_ms:.1f}ms")
            return integrated_profile
            
        except Exception as e:
            self.logger.error(f"Integrated risk assessment failed: {e}")
            # Return conservative default
            return IntegratedRiskProfile(
                overall_risk_score=0.5,
                risk_multiplier=0.15,
                risk_level="medium",
                confidence=0.3
            )
    
    def _integrate_risk_profiles(self,
                               security_profile: SecurityRiskProfile,
                               performance_profile: PerformanceRiskProfile,
                               architectural_profile: ArchitecturalRiskProfile,
                               change_assessment: RiskAssessment,
                               context: Optional[str]) -> IntegratedRiskProfile:
        """Integrate individual risk profiles into comprehensive assessment."""
        
        profile = IntegratedRiskProfile()
        
        # Extract individual risk scores
        profile.security_risk = security_profile.overall_risk_score
        profile.performance_risk = performance_profile.overall_risk_score
        profile.architectural_risk = architectural_profile.overall_risk_score
        profile.change_risk = change_assessment.risk_multiplier
        
        # Calculate weighted overall risk score
        weighted_scores = {
            'security': profile.security_risk * self.config.risk_weights['security'],
            'performance': profile.performance_risk * self.config.risk_weights['performance'],
            'architectural': profile.architectural_risk * self.config.risk_weights['architectural'],
            'change': profile.change_risk * self.config.risk_weights['change']
        }
        
        profile.overall_risk_score = sum(weighted_scores.values())
        
        # Apply context-specific adjustments
        context_multiplier = self._get_context_risk_multiplier(context)
        profile.overall_risk_score *= context_multiplier
        
        # Ensure risk score is within bounds
        profile.overall_risk_score = min(profile.overall_risk_score, 1.0)
        
        # Calculate risk multiplier for quality scoring (max 30% reduction)
        profile.risk_multiplier = min(
            profile.overall_risk_score * self.config.max_risk_multiplier,
            self.config.max_risk_multiplier
        )
        
        # Determine overall risk level
        profile.risk_level = self._determine_risk_level(profile.overall_risk_score)
        
        # Analyze risk correlations
        profile.risk_correlations = self._analyze_risk_correlations(
            security_profile, performance_profile, architectural_profile, change_assessment
        )
        
        # Identify compound risks and hotspots
        profile.compound_risks = self._identify_compound_risks(
            security_profile, performance_profile, architectural_profile
        )
        
        profile.risk_hotspots = self._identify_risk_hotspots(
            security_profile, performance_profile, architectural_profile
        )
        
        # Generate integrated mitigation strategies
        profile.priority_mitigations, profile.risk_mitigation_map = self._generate_integrated_mitigations(
            security_profile, performance_profile, architectural_profile, change_assessment
        )
        
        # Calculate confidence based on individual assessment confidence
        individual_confidences = [
            security_profile.confidence,
            performance_profile.confidence,
            architectural_profile.confidence,
            change_assessment.confidence
        ]
        profile.confidence = statistics.mean(individual_confidences)
        
        return profile
    
    def _get_context_risk_multiplier(self, context: Optional[str]) -> float:
        """Get context-specific risk multiplier."""
        if not context:
            return 1.0
        
        context_multipliers = {
            'critical_algorithms': 1.4,   # Critical algorithms need higher risk weighting
            'public_apis': 1.2,           # Public APIs have higher risk exposure
            'business_logic': 1.0,        # Standard business logic
            'integration_code': 1.3,      # Integration points are risky
            'ui_components': 0.8,         # UI components generally lower risk
            'test_code': 0.5              # Test code has lower risk impact
        }
        
        return context_multipliers.get(context, 1.0)
    
    def _determine_risk_level(self, overall_risk_score: float) -> str:
        """Determine overall risk level from score."""
        if overall_risk_score >= self.config.risk_level_thresholds['critical']:
            return "critical"
        elif overall_risk_score >= self.config.risk_level_thresholds['high']:
            return "high"
        elif overall_risk_score >= self.config.risk_level_thresholds['medium']:
            return "medium"
        else:
            return "low"
    
    def _analyze_risk_correlations(self,
                                 security_profile: SecurityRiskProfile,
                                 performance_profile: PerformanceRiskProfile,
                                 architectural_profile: ArchitecturalRiskProfile,
                                 change_assessment: RiskAssessment) -> Dict[str, float]:
        """Analyze correlations between different risk types."""
        correlations = {}
        
        # Security-Performance correlation
        if (security_profile.overall_risk_score > 0.5 and 
            performance_profile.overall_risk_score > 0.5):
            correlations['security_performance'] = min(
                security_profile.overall_risk_score * performance_profile.overall_risk_score * 2,
                1.0
            )
        
        # Security-Architectural correlation
        if (security_profile.overall_risk_score > 0.4 and 
            architectural_profile.overall_risk_score > 0.4):
            correlations['security_architectural'] = min(
                security_profile.overall_risk_score * architectural_profile.overall_risk_score * 1.8,
                1.0
            )
        
        # Performance-Architectural correlation
        if (performance_profile.overall_risk_score > 0.5 and 
            architectural_profile.overall_risk_score > 0.5):
            correlations['performance_architectural'] = min(
                performance_profile.overall_risk_score * architectural_profile.overall_risk_score * 1.5,
                1.0
            )
        
        # Change-All correlations
        if change_assessment.risk_multiplier > 0.15:
            correlations['change_security'] = change_assessment.risk_multiplier * security_profile.overall_risk_score
            correlations['change_performance'] = change_assessment.risk_multiplier * performance_profile.overall_risk_score
            correlations['change_architectural'] = change_assessment.risk_multiplier * architectural_profile.overall_risk_score
        
        return correlations
    
    def _identify_compound_risks(self,
                               security_profile: SecurityRiskProfile,
                               performance_profile: PerformanceRiskProfile,
                               architectural_profile: ArchitecturalRiskProfile) -> List[str]:
        """Identify compound risks where multiple risk types interact."""
        compound_risks = []
        
        # High security + high architectural risk
        if (security_profile.risk_level in ['high', 'critical'] and
            architectural_profile.risk_level in ['high', 'critical']):
            compound_risks.append("Security-Architecture compound risk: Security changes requiring architectural modifications")
        
        # High performance + high architectural risk  
        if (performance_profile.risk_level in ['high', 'critical'] and
            architectural_profile.risk_level in ['high', 'critical']):
            compound_risks.append("Performance-Architecture compound risk: Performance issues indicating architectural problems")
        
        # High security + high performance risk
        if (security_profile.risk_level in ['high', 'critical'] and
            performance_profile.risk_level in ['high', 'critical']):
            compound_risks.append("Security-Performance compound risk: Security hardening impacting performance")
        
        # Triple compound risk
        if (security_profile.risk_level in ['high', 'critical'] and
            performance_profile.risk_level in ['high', 'critical'] and
            architectural_profile.risk_level in ['high', 'critical']):
            compound_risks.append("Triple compound risk: High risk across security, performance, and architecture")
        
        return compound_risks
    
    def _identify_risk_hotspots(self,
                              security_profile: SecurityRiskProfile,
                              performance_profile: PerformanceRiskProfile,
                              architectural_profile: ArchitecturalRiskProfile) -> List[str]:
        """Identify specific risk hotspots requiring immediate attention."""
        hotspots = []
        
        # Security hotspots
        if security_profile.risk_level == 'critical':
            hotspots.extend([f"Critical security risk: {concern}" 
                           for concern in security_profile.mitigation_priorities[:2]])
        
        # Performance hotspots
        if performance_profile.risk_level == 'critical':
            hotspots.extend([f"Critical performance risk: {strategy}" 
                           for strategy in performance_profile.mitigation_strategies[:2]])
        
        # Architectural hotspots
        if architectural_profile.risk_level == 'critical':
            hotspots.extend([f"Critical architectural risk: {rec}" 
                           for rec in architectural_profile.mitigation_recommendations[:2]])
        
        return hotspots[:5]  # Limit to top 5 hotspots
    
    def _generate_integrated_mitigations(self,
                                       security_profile: SecurityRiskProfile,
                                       performance_profile: PerformanceRiskProfile,
                                       architectural_profile: ArchitecturalRiskProfile,
                                       change_assessment: RiskAssessment) -> Tuple[List[str], Dict[str, List[str]]]:
        """Generate integrated mitigation strategies prioritized by risk level."""
        
        priority_mitigations = []
        risk_mitigation_map = {
            'security': security_profile.mitigation_priorities,
            'performance': performance_profile.mitigation_strategies,
            'architectural': architectural_profile.mitigation_recommendations,
            'change': change_assessment.mitigation_suggestions
        }
        
        # Prioritize by risk level and impact
        risk_priorities = [
            ('security', security_profile.risk_level, security_profile.mitigation_priorities),
            ('performance', performance_profile.risk_level, performance_profile.mitigation_strategies),
            ('architectural', architectural_profile.risk_level, architectural_profile.mitigation_recommendations),
            ('change', change_assessment.risk_level, change_assessment.mitigation_suggestions)
        ]
        
        # Sort by risk level (critical > high > medium > low)
        risk_level_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        risk_priorities.sort(key=lambda x: risk_level_order.get(x[1], 0), reverse=True)
        
        # Add top mitigation from each high-priority risk area
        for risk_type, risk_level, mitigations in risk_priorities:
            if risk_level in ['critical', 'high'] and mitigations:
                priority_mitigations.append(f"{risk_type.title()}: {mitigations[0]}")
        
        # Add integrated mitigations for compound risks
        if (security_profile.risk_level in ['high', 'critical'] and
            architectural_profile.risk_level in ['high', 'critical']):
            priority_mitigations.append("Integrated: Conduct security architecture review")
        
        if (performance_profile.risk_level in ['high', 'critical'] and
            architectural_profile.risk_level in ['high', 'critical']):
            priority_mitigations.append("Integrated: Implement performance-aware architectural refactoring")
        
        return priority_mitigations[:6], risk_mitigation_map  # Limit to top 6 priorities
    
    def _identify_security_files(self, files: List[str]) -> List[str]:
        """Identify security-sensitive files from the file list."""
        security_patterns = [
            r'.*auth.*\.py$', r'.*security.*\.py$', r'.*crypto.*\.py$',
            r'.*login.*\.py$', r'.*token.*\.py$', r'.*session.*\.py$',
            r'.*password.*\.py$', r'.*permission.*\.py$'
        ]
        
        security_files = []
        for file_path in files:
            for pattern in security_patterns:
                if re.match(pattern, file_path, re.IGNORECASE):
                    security_files.append(file_path)
                    break
        
        return security_files
    
    def _generate_cache_key(self, files: List[str], context: Optional[str], 
                          vulnerability_data: Optional[Dict], 
                          performance_data: Optional[Dict],
                          project_context: Optional[Dict]) -> str:
        """Generate cache key for risk assessment."""
        key_components = [
            tuple(sorted(files)),
            context or "",
            json.dumps(vulnerability_data or {}, sort_keys=True),
            json.dumps(performance_data or {}, sort_keys=True),
            json.dumps(project_context or {}, sort_keys=True)
        ]
        return str(hash(str(key_components)))
    
    def _cleanup_cache(self):
        """Clean expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, cache_time) in self.cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]

def main():
    """CLI interface for integrated risk assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Risk Assessment System')
    parser.add_argument('--files', nargs='+', required=True, help='Files to analyze')
    parser.add_argument('--context', help='Component context')
    parser.add_argument('--vulnerability-data', help='JSON file with vulnerability data')
    parser.add_argument('--performance-data', help='JSON file with performance data')
    parser.add_argument('--performance-baseline', help='JSON file with performance baseline')
    parser.add_argument('--project-context', help='JSON file with project context')
    parser.add_argument('--output', choices=['score', 'multiplier', 'level', 'full'], 
                       default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize integrated risk assessment
    risk_assessor = IntegratedRiskAssessment()
    
    # Load optional data files
    vulnerability_data = None
    if args.vulnerability_data and os.path.exists(args.vulnerability_data):
        with open(args.vulnerability_data, 'r') as f:
            vulnerability_data = json.load(f)
    
    performance_data = None
    if args.performance_data and os.path.exists(args.performance_data):
        with open(args.performance_data, 'r') as f:
            performance_data = json.load(f)
    
    performance_baseline = None
    if args.performance_baseline and os.path.exists(args.performance_baseline):
        with open(args.performance_baseline, 'r') as f:
            performance_baseline = json.load(f)
    
    project_context = None
    if args.project_context and os.path.exists(args.project_context):
        with open(args.project_context, 'r') as f:
            project_context = json.load(f)
    
    # Perform integrated risk assessment
    risk_profile = risk_assessor.assess_integrated_risk(
        files=args.files,
        context=args.context,
        vulnerability_data=vulnerability_data,
        performance_data=performance_data,
        performance_baseline=performance_baseline,
        project_context=project_context
    )
    
    # Output results based on requested format
    if args.output == 'score':
        print(f"{risk_profile.overall_risk_score:.3f}")
    elif args.output == 'multiplier':
        print(f"{risk_profile.risk_multiplier:.3f}")
    elif args.output == 'level':
        print(risk_profile.risk_level)
    else:
        result = {
            'overall_risk_score': risk_profile.overall_risk_score,
            'risk_multiplier': risk_profile.risk_multiplier,
            'risk_level': risk_profile.risk_level,
            'individual_risks': {
                'security': risk_profile.security_risk,
                'performance': risk_profile.performance_risk,
                'architectural': risk_profile.architectural_risk,
                'change': risk_profile.change_risk
            },
            'risk_correlations': risk_profile.risk_correlations,
            'compound_risks': risk_profile.compound_risks,
            'risk_hotspots': risk_profile.risk_hotspots,
            'priority_mitigations': risk_profile.priority_mitigations,
            'assessment_time_ms': risk_profile.assessment_time_ms,
            'confidence': risk_profile.confidence
        }
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()