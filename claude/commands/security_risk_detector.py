#!/usr/bin/env python3
"""
Security Risk Detector - Issue #93 Phase 3
Advanced security risk assessment for the Multi-Dimensional Quality Scoring System.

This module provides:
- Deep security pattern analysis beyond basic vulnerability counts
- Contextual security risk based on architectural patterns
- Integration with existing risk adjustment calculator
- Security-specific risk mitigation recommendations
"""

import os
import json
import yaml
import subprocess
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import re

@dataclass
class SecurityRiskProfile:
    """Comprehensive security risk assessment profile."""
    overall_risk_score: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    security_patterns: Dict[str, float] = field(default_factory=dict)
    vulnerability_context: Dict[str, Any] = field(default_factory=dict)
    architectural_risks: List[str] = field(default_factory=list)
    mitigation_priorities: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class SecurityAnalysis:
    """Detailed security analysis result."""
    file_path: str
    risk_factors: Dict[str, float] = field(default_factory=dict)
    security_concerns: List[str] = field(default_factory=list)
    pattern_matches: List[str] = field(default_factory=list)
    context_risk: str = "low"
    analysis_time_ms: float = 0.0

class SecurityRiskDetector:
    """
    Advanced security risk detector for multi-dimensional quality scoring.
    
    Provides contextual security risk assessment beyond simple vulnerability counting,
    focusing on architectural patterns, data flow risks, and security design patterns.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize security risk detector with configuration."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/security-risk-patterns.yaml"
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.logger = logging.getLogger(__name__)
        
        # Load security patterns and risk configurations
        self._load_security_patterns()
        
        # Security-critical file patterns
        self.critical_security_patterns = [
            r'.*auth.*\.py$', r'.*security.*\.py$', r'.*crypto.*\.py$',
            r'.*login.*\.py$', r'.*token.*\.py$', r'.*session.*\.py$',
            r'.*password.*\.py$', r'.*cert.*\.py$', r'.*ssl.*\.py$',
            r'.*oauth.*\.py$', r'.*jwt.*\.py$', r'.*permission.*\.py$',
            r'config/.*\.yaml$', r'config/.*\.json$', r'.*\.env$',
            r'.*requirements.*\.txt$', r'.*package.*\.json$'
        ]
        
        # High-risk code patterns
        self.risky_code_patterns = {
            'sql_injection': r'(query|execute|cursor).*%.*[sf]',
            'command_injection': r'(subprocess|os\.system|eval|exec)',
            'path_traversal': r'\.\./', 
            'hardcoded_secrets': r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
            'unsafe_deserialization': r'(pickle|eval|exec|yaml\.load)',
            'weak_crypto': r'(md5|sha1|des|rc4)',
            'sql_concat': r'(SELECT|INSERT|UPDATE|DELETE).*\+.*["\']',
            'xss_vulnerable': r'innerHTML|outerHTML|document\.write',
            'csrf_missing': r'@csrf_exempt|csrf_token.*False'
        }
        
    def _load_security_patterns(self):
        """Load security risk patterns from configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.security_weights = config.get('security_weights', {})
                    self.risk_thresholds = config.get('risk_thresholds', {})
                    self.architectural_patterns = config.get('architectural_patterns', {})
            else:
                # Default configuration
                self.security_weights = {
                    'critical_files': 0.4,
                    'vulnerability_count': 0.3,
                    'risky_patterns': 0.2,
                    'architectural_risk': 0.1
                }
                self.risk_thresholds = {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.8,
                    'critical': 0.95
                }
                self.architectural_patterns = {
                    'authentication': ['auth', 'login', 'session', 'token'],
                    'authorization': ['permission', 'role', 'access', 'policy'],
                    'data_protection': ['encrypt', 'crypto', 'hash', 'secure'],
                    'communication': ['api', 'endpoint', 'request', 'response']
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to load security configuration: {e}")
            self._set_default_config()
    
    def _set_default_config(self):
        """Set default security configuration."""
        self.security_weights = {
            'critical_files': 0.4,
            'vulnerability_count': 0.3, 
            'risky_patterns': 0.2,
            'architectural_risk': 0.1
        }
        self.risk_thresholds = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 0.95}
        self.architectural_patterns = {
            'authentication': ['auth', 'login', 'session', 'token'],
            'authorization': ['permission', 'role', 'access', 'policy'],
            'data_protection': ['encrypt', 'crypto', 'hash', 'secure'],
            'communication': ['api', 'endpoint', 'request', 'response']
        }
    
    def analyze_security_risk(self, files: List[str], 
                            vulnerability_data: Optional[Dict] = None,
                            context: Optional[str] = None) -> SecurityRiskProfile:
        """
        Perform comprehensive security risk analysis.
        
        Args:
            files: List of files to analyze
            vulnerability_data: Optional vulnerability scan results
            context: Optional component context for risk weighting
            
        Returns:
            SecurityRiskProfile with detailed risk assessment
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"security_risk:{hash(tuple(sorted(files)))}"
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_result
        
        try:
            # Analyze each file individually
            file_analyses = []
            for file_path in files:
                if os.path.exists(file_path):
                    analysis = self._analyze_file_security_risk(file_path)
                    file_analyses.append(analysis)
            
            # Calculate overall risk profile
            profile = self._calculate_security_risk_profile(
                file_analyses, vulnerability_data, context
            )
            
            # Cache result
            self.cache[cache_key] = (profile, datetime.now())
            
            # Clean old cache entries
            self._cleanup_cache()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Security risk analysis failed: {e}")
            return SecurityRiskProfile(
                overall_risk_score=0.5,  # Default medium risk
                risk_level="medium",
                confidence=0.3
            )
    
    def _analyze_file_security_risk(self, file_path: str) -> SecurityAnalysis:
        """Analyze security risk for individual file."""
        start_time = time.time()
        
        analysis = SecurityAnalysis(file_path=file_path)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if file is security-critical
            is_critical = any(re.match(pattern, file_path) 
                            for pattern in self.critical_security_patterns)
            if is_critical:
                analysis.risk_factors['critical_file'] = 1.0
                analysis.security_concerns.append(f"Security-critical file: {file_path}")
            
            # Analyze risky code patterns
            pattern_risks = {}
            for pattern_name, pattern in self.risky_code_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    risk_score = min(len(matches) * 0.2, 1.0)
                    pattern_risks[pattern_name] = risk_score
                    analysis.pattern_matches.append(f"{pattern_name}: {len(matches)} matches")
                    analysis.security_concerns.append(f"Risky pattern '{pattern_name}' found")
            
            analysis.risk_factors.update(pattern_risks)
            
            # Determine contextual risk level
            total_risk = sum(analysis.risk_factors.values())
            if total_risk >= 2.0:
                analysis.context_risk = "critical"
            elif total_risk >= 1.0:
                analysis.context_risk = "high"
            elif total_risk >= 0.5:
                analysis.context_risk = "medium"
            else:
                analysis.context_risk = "low"
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze file {file_path}: {e}")
            analysis.context_risk = "medium"  # Conservative default
        
        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        return analysis
    
    def _calculate_security_risk_profile(self, 
                                       file_analyses: List[SecurityAnalysis],
                                       vulnerability_data: Optional[Dict],
                                       context: Optional[str]) -> SecurityRiskProfile:
        """Calculate overall security risk profile."""
        profile = SecurityRiskProfile()
        
        if not file_analyses:
            return profile
        
        # Aggregate file-level risks
        all_risk_factors = {}
        all_concerns = []
        critical_files = 0
        
        for analysis in file_analyses:
            # Aggregate risk factors
            for factor, score in analysis.risk_factors.items():
                all_risk_factors[factor] = all_risk_factors.get(factor, 0) + score
            
            # Collect concerns
            all_concerns.extend(analysis.security_concerns)
            
            # Count critical files
            if 'critical_file' in analysis.risk_factors:
                critical_files += 1
        
        # Calculate weighted risk score
        risk_components = {
            'critical_files': min(critical_files * 0.3, 1.0),
            'risky_patterns': min(sum(all_risk_factors.values()) * 0.1, 1.0),
            'architectural_risk': self._calculate_architectural_risk(file_analyses, context)
        }
        
        # Add vulnerability data if provided
        if vulnerability_data:
            vuln_score = self._calculate_vulnerability_risk(vulnerability_data)
            risk_components['vulnerability_count'] = vuln_score
        
        # Calculate overall risk score
        total_risk = sum(
            risk_components[component] * self.security_weights.get(component, 0.25)
            for component in risk_components
        )
        
        profile.overall_risk_score = min(total_risk, 1.0)
        profile.security_patterns = risk_components
        
        # Determine risk level
        if profile.overall_risk_score >= self.risk_thresholds['critical']:
            profile.risk_level = "critical"
        elif profile.overall_risk_score >= self.risk_thresholds['high']:
            profile.risk_level = "high"
        elif profile.overall_risk_score >= self.risk_thresholds['medium']:
            profile.risk_level = "medium"
        else:
            profile.risk_level = "low"
        
        # Generate mitigation priorities
        profile.mitigation_priorities = self._generate_mitigation_priorities(
            all_risk_factors, all_concerns, critical_files
        )
        
        # Set confidence based on analysis completeness
        profile.confidence = min(len(file_analyses) / 10.0, 1.0)  # More files = higher confidence
        
        return profile
    
    def _calculate_architectural_risk(self, 
                                    file_analyses: List[SecurityAnalysis], 
                                    context: Optional[str]) -> float:
        """Calculate architectural security risk."""
        if not context:
            return 0.1  # Default low architectural risk
        
        # Context-specific architectural risks
        context_risks = {
            'critical_algorithms': 0.9,  # Crypto/security algorithms are high risk
            'public_apis': 0.7,          # Public interfaces have attack surface
            'business_logic': 0.4,       # Business logic has moderate risk
            'integration_code': 0.6,     # Integration points are risky
            'ui_components': 0.3,        # UI has lower architectural risk
            'test_code': 0.1            # Test code has minimal risk
        }
        
        return context_risks.get(context, 0.3)
    
    def _calculate_vulnerability_risk(self, vulnerability_data: Dict) -> float:
        """Calculate risk score from vulnerability scan results."""
        if not vulnerability_data:
            return 0.0
        
        # Weight vulnerabilities by severity
        severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }
        
        total_risk = 0.0
        for severity, count in vulnerability_data.items():
            if severity.lower() in severity_weights:
                weight = severity_weights[severity.lower()]
                # Each vulnerability adds risk, with diminishing returns
                total_risk += min(count * weight * 0.2, weight)
        
        return min(total_risk, 1.0)
    
    def _generate_mitigation_priorities(self, 
                                      risk_factors: Dict[str, float],
                                      concerns: List[str], 
                                      critical_files: int) -> List[str]:
        """Generate prioritized mitigation recommendations."""
        priorities = []
        
        # Prioritize by risk factor scores
        sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        
        for risk_type, score in sorted_risks[:3]:  # Top 3 risks
            if risk_type == 'sql_injection' and score > 0.3:
                priorities.append("Implement parameterized queries for SQL injection prevention")
            elif risk_type == 'command_injection' and score > 0.3:
                priorities.append("Review and sanitize system command usage")
            elif risk_type == 'hardcoded_secrets' and score > 0.3:
                priorities.append("Move hardcoded secrets to secure configuration")
            elif risk_type == 'weak_crypto' and score > 0.3:
                priorities.append("Upgrade to strong cryptographic algorithms")
            elif risk_type == 'critical_file' and score > 0.5:
                priorities.append("Add comprehensive security testing for critical files")
        
        # Add general recommendations
        if critical_files > 0:
            priorities.append("Implement security code review for critical files")
        
        if len(concerns) > 5:
            priorities.append("Conduct comprehensive security audit")
        
        return priorities[:5]  # Limit to top 5 priorities
    
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
    """CLI interface for security risk detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Risk Detector')
    parser.add_argument('--files', nargs='+', required=True, help='Files to analyze')
    parser.add_argument('--context', help='Component context')
    parser.add_argument('--vulnerability-data', help='JSON file with vulnerability scan results')
    parser.add_argument('--output', choices=['score', 'level', 'full'], default='full',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SecurityRiskDetector()
    
    # Load vulnerability data if provided
    vulnerability_data = None
    if args.vulnerability_data and os.path.exists(args.vulnerability_data):
        with open(args.vulnerability_data, 'r') as f:
            vulnerability_data = json.load(f)
    
    # Perform analysis
    profile = detector.analyze_security_risk(
        files=args.files,
        vulnerability_data=vulnerability_data,
        context=args.context
    )
    
    # Output results
    if args.output == 'score':
        print(f"{profile.overall_risk_score:.3f}")
    elif args.output == 'level':
        print(profile.risk_level)
    else:
        result = {
            'overall_risk_score': profile.overall_risk_score,
            'risk_level': profile.risk_level,
            'security_patterns': profile.security_patterns,
            'mitigation_priorities': profile.mitigation_priorities,
            'confidence': profile.confidence
        }
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()