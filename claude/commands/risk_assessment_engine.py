#!/usr/bin/env python3
"""
Risk Assessment Engine - Issue #92 Phase 1
Core risk assessment system for Risk-Based Manual Intervention Framework.

This engine implements:
1. Multi-factor risk scoring algorithm
2. Security pattern detection
3. Code complexity analysis
4. Impact assessment based on files and LOC
5. Historical failure pattern analysis
6. Risk-based escalation decision making

Risk Score Formula:
risk_score = (
    security_risk_weight * security_changes +
    complexity_risk_weight * code_complexity +
    impact_risk_weight * files_affected +
    historical_risk_weight * past_failures
)
"""

import json
import subprocess
import yaml
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
from dataclasses import dataclass

@dataclass
class RiskFactors:
    """Container for risk factor components."""
    security_score: float = 0.0
    complexity_score: float = 0.0
    impact_score: float = 0.0
    historical_score: float = 0.0
    time_pressure_score: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class ChangeContext:
    """Container for code change context information."""
    files_changed: List[str]
    lines_added: int
    lines_removed: int
    issue_number: int
    issue_details: Dict[str, Any]
    pr_number: Optional[int] = None
    branch_name: Optional[str] = None
    author: Optional[str] = None
    commit_hash: Optional[str] = None

@dataclass
class RiskScore:
    """Container for calculated risk score and details."""
    total_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    factors: RiskFactors
    escalation_required: bool
    specialist_type: Optional[str]
    reasoning: List[str]
    confidence: float

class RiskAssessmentEngine:
    """
    Core risk assessment engine for automated escalation decisions.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the risk assessment engine."""
        self.config_path = config_path
        self.config = self._load_config()
        self.risk_weights = self._get_risk_weights()
        self.security_patterns = self._get_security_patterns()
        self.complexity_thresholds = self._get_complexity_thresholds()
        self.escalation_thresholds = self._get_escalation_thresholds()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for risk assessment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RiskAssessmentEngine - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load risk assessment configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk assessment configuration."""
        return {
            'risk_weights': {
                'security': 0.40,    # 40% weight - highest priority
                'complexity': 0.20,  # 20% weight
                'impact': 0.20,      # 20% weight
                'historical': 0.10,  # 10% weight
                'time_pressure': 0.10 # 10% weight
            },
            'security_patterns': {
                'high_risk_paths': [
                    'auth/**', '**/security/**', '*/payment/**', 
                    '**/authentication/**', '**/authorization/**',
                    '*/crypto/**', '**/secrets/**', '*/password/**'
                ],
                'medium_risk_paths': [
                    '**/api/**', '**/database/**', '**/config/**',
                    '**/admin/**', '**/user/**', '**/session/**'
                ],
                'high_risk_keywords': [
                    'password', 'secret', 'token', 'auth', 'login',
                    'crypto', 'encrypt', 'decrypt', 'certificate',
                    'private_key', 'api_key', 'jwt', 'oauth'
                ]
            },
            'complexity_thresholds': {
                'lines_changed': {'low': 50, 'medium': 200, 'high': 500, 'critical': 1000},
                'files_affected': {'low': 3, 'medium': 10, 'high': 25, 'critical': 50},
                'cyclomatic_complexity': {'low': 5, 'medium': 10, 'high': 20, 'critical': 30}
            },
            'escalation_thresholds': {
                'low': 0.3,      # 0-30% risk score
                'medium': 0.6,   # 30-60% risk score  
                'high': 0.8,     # 60-80% risk score
                'critical': 1.0  # 80-100% risk score
            },
            'specialist_routing': {
                'security': {
                    'triggers': ['security_changes', 'auth_modifications'],
                    'sla_hours': 4,
                    'blocking': True
                },
                'architecture': {
                    'triggers': ['large_changes', 'database_changes', 'api_changes'],
                    'sla_hours': 12,
                    'blocking': False
                },
                'compliance': {
                    'triggers': ['audit_changes', 'privacy_changes'],
                    'sla_hours': 6,
                    'blocking': True
                }
            }
        }
    
    def _get_risk_weights(self) -> Dict[str, float]:
        """Get risk factor weights from configuration."""
        return self.config.get('risk_weights', {})
    
    def _get_security_patterns(self) -> Dict[str, List[str]]:
        """Get security risk patterns from configuration."""
        return self.config.get('security_patterns', {})
    
    def _get_complexity_thresholds(self) -> Dict[str, Dict[str, int]]:
        """Get complexity assessment thresholds."""
        return self.config.get('complexity_thresholds', {})
    
    def _get_escalation_thresholds(self) -> Dict[str, float]:
        """Get escalation decision thresholds."""
        return self.config.get('escalation_thresholds', {})
    
    def assess_change_risk(self, change_context: ChangeContext) -> RiskScore:
        """
        Assess the risk level of a code change and determine escalation needs.
        
        Args:
            change_context: Context information about the code change
            
        Returns:
            RiskScore object with detailed risk assessment
        """
        self.logger.info(f"🔍 Assessing risk for issue #{change_context.issue_number}")
        
        try:
            # Calculate individual risk factors
            security_risk = self._assess_security_risk(change_context)
            complexity_risk = self._assess_complexity_risk(change_context)
            impact_risk = self._assess_impact_risk(change_context)
            historical_risk = self._assess_historical_risk(change_context)
            time_pressure_risk = self._assess_time_pressure_risk(change_context)
            
            # Create risk factors container
            factors = RiskFactors(
                security_score=security_risk['score'],
                complexity_score=complexity_risk['score'],
                impact_score=impact_risk['score'],
                historical_score=historical_risk['score'],
                time_pressure_score=time_pressure_risk['score'],
                metadata={
                    'security_details': security_risk,
                    'complexity_details': complexity_risk,
                    'impact_details': impact_risk,
                    'historical_details': historical_risk,
                    'time_pressure_details': time_pressure_risk
                }
            )
            
            # Calculate weighted total score
            total_score = (
                self.risk_weights.get('security', 0.4) * factors.security_score +
                self.risk_weights.get('complexity', 0.2) * factors.complexity_score +
                self.risk_weights.get('impact', 0.2) * factors.impact_score +
                self.risk_weights.get('historical', 0.1) * factors.historical_score +
                self.risk_weights.get('time_pressure', 0.1) * factors.time_pressure_score
            )
            
            # Determine risk level and escalation needs
            risk_level = self._determine_risk_level(total_score)
            escalation_required = self._requires_escalation(risk_level, factors)
            specialist_type = self._determine_specialist_type(factors, change_context)
            
            # Generate reasoning
            reasoning = self._generate_risk_reasoning(factors, change_context)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(factors, change_context)
            
            risk_score = RiskScore(
                total_score=total_score,
                risk_level=risk_level,
                factors=factors,
                escalation_required=escalation_required,
                specialist_type=specialist_type,
                reasoning=reasoning,
                confidence=confidence
            )
            
            self.logger.info(f"📊 Risk assessment complete: {risk_level} ({total_score:.2f}) - Escalation: {escalation_required}")
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Error assessing change risk: {e}")
            # Return safe default - high risk requiring manual review
            return RiskScore(
                total_score=0.9,
                risk_level='high',
                factors=RiskFactors(),
                escalation_required=True,
                specialist_type='architecture',
                reasoning=[f"Risk assessment error: {e}"],
                confidence=0.0
            )
    
    def _assess_security_risk(self, change_context: ChangeContext) -> Dict[str, Any]:
        """Assess security-related risk factors."""
        security_score = 0.0
        security_indicators = []
        
        try:
            # Check file paths against security patterns
            high_risk_paths = self.security_patterns.get('high_risk_paths', [])
            medium_risk_paths = self.security_patterns.get('medium_risk_paths', [])
            
            high_risk_files = []
            medium_risk_files = []
            
            for file_path in change_context.files_changed:
                if self._matches_patterns(file_path, high_risk_paths):
                    high_risk_files.append(file_path)
                    security_score += 0.3  # High risk file adds 30% to security score
                elif self._matches_patterns(file_path, medium_risk_paths):
                    medium_risk_files.append(file_path)
                    security_score += 0.15  # Medium risk file adds 15% to security score
            
            if high_risk_files:
                security_indicators.append(f"High-risk files modified: {len(high_risk_files)}")
            if medium_risk_files:
                security_indicators.append(f"Medium-risk files modified: {len(medium_risk_files)}")
            
            # Check for security-related keywords in issue
            issue_content = f"{change_context.issue_details.get('title', '')} {change_context.issue_details.get('body', '')}"
            high_risk_keywords = self.security_patterns.get('high_risk_keywords', [])
            
            security_keywords_found = []
            for keyword in high_risk_keywords:
                if keyword.lower() in issue_content.lower():
                    security_keywords_found.append(keyword)
                    security_score += 0.1  # Each keyword adds 10% to security score
            
            if security_keywords_found:
                security_indicators.append(f"Security keywords found: {', '.join(security_keywords_found)}")
            
            # Check for authentication/authorization changes
            auth_patterns = ['login', 'auth', 'permission', 'role', 'user', 'session']
            auth_changes = []
            for pattern in auth_patterns:
                for file_path in change_context.files_changed:
                    if pattern in file_path.lower():
                        auth_changes.append(file_path)
                        security_score += 0.2  # Auth-related file adds 20% to security score
                        break
            
            if auth_changes:
                security_indicators.append(f"Authentication/authorization files: {len(set(auth_changes))}")
            
            # Clamp score between 0 and 1
            security_score = min(1.0, security_score)
            
            return {
                'score': security_score,
                'level': self._score_to_level(security_score),
                'indicators': security_indicators,
                'high_risk_files': high_risk_files,
                'medium_risk_files': medium_risk_files,
                'security_keywords': security_keywords_found,
                'auth_changes': list(set(auth_changes))
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing security risk: {e}")
            return {
                'score': 0.5,  # Default to medium risk on error
                'level': 'medium',
                'indicators': [f"Security assessment error: {e}"],
                'error': str(e)
            }
    
    def _assess_complexity_risk(self, change_context: ChangeContext) -> Dict[str, Any]:
        """Assess code complexity risk factors."""
        complexity_score = 0.0
        complexity_indicators = []
        
        try:
            # Lines changed complexity
            total_lines_changed = change_context.lines_added + change_context.lines_removed
            lines_thresholds = self.complexity_thresholds.get('lines_changed', {})
            
            if total_lines_changed > lines_thresholds.get('critical', 1000):
                complexity_score += 0.4
                complexity_indicators.append(f"Critical: {total_lines_changed} lines changed")
            elif total_lines_changed > lines_thresholds.get('high', 500):
                complexity_score += 0.3
                complexity_indicators.append(f"High: {total_lines_changed} lines changed")
            elif total_lines_changed > lines_thresholds.get('medium', 200):
                complexity_score += 0.2
                complexity_indicators.append(f"Medium: {total_lines_changed} lines changed")
            elif total_lines_changed > lines_thresholds.get('low', 50):
                complexity_score += 0.1
                complexity_indicators.append(f"Low: {total_lines_changed} lines changed")
            
            # Files affected complexity
            files_count = len(change_context.files_changed)
            files_thresholds = self.complexity_thresholds.get('files_affected', {})
            
            if files_count > files_thresholds.get('critical', 50):
                complexity_score += 0.3
                complexity_indicators.append(f"Critical: {files_count} files affected")
            elif files_count > files_thresholds.get('high', 25):
                complexity_score += 0.25
                complexity_indicators.append(f"High: {files_count} files affected")
            elif files_count > files_thresholds.get('medium', 10):
                complexity_score += 0.15
                complexity_indicators.append(f"Medium: {files_count} files affected")
            elif files_count > files_thresholds.get('low', 3):
                complexity_score += 0.1
                complexity_indicators.append(f"Low: {files_count} files affected")
            
            # Check for cross-cutting concerns
            cross_cutting_patterns = ['config', 'util', 'common', 'shared', 'base', 'core']
            cross_cutting_files = []
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in cross_cutting_patterns):
                    cross_cutting_files.append(file_path)
            
            if cross_cutting_files:
                complexity_score += 0.2
                complexity_indicators.append(f"Cross-cutting concerns: {len(cross_cutting_files)} files")
            
            # Database schema changes (high complexity)
            db_patterns = ['migration', 'schema', 'database', '.sql', 'alembic']
            db_files = []
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in db_patterns):
                    db_files.append(file_path)
            
            if db_files:
                complexity_score += 0.25
                complexity_indicators.append(f"Database changes: {len(db_files)} files")
            
            # Clamp score between 0 and 1
            complexity_score = min(1.0, complexity_score)
            
            return {
                'score': complexity_score,
                'level': self._score_to_level(complexity_score),
                'indicators': complexity_indicators,
                'lines_changed': total_lines_changed,
                'files_count': files_count,
                'cross_cutting_files': cross_cutting_files,
                'database_files': db_files
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing complexity risk: {e}")
            return {
                'score': 0.3,  # Default to low-medium risk on error
                'level': 'medium',
                'indicators': [f"Complexity assessment error: {e}"],
                'error': str(e)
            }
    
    def _assess_impact_risk(self, change_context: ChangeContext) -> Dict[str, Any]:
        """Assess the impact risk of changes."""
        impact_score = 0.0
        impact_indicators = []
        
        try:
            # API changes (high impact)
            api_patterns = ['api', 'endpoint', 'route', 'controller', 'service']
            api_files = []
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in api_patterns):
                    api_files.append(file_path)
            
            if api_files:
                impact_score += 0.3
                impact_indicators.append(f"API changes: {len(api_files)} files")
            
            # Configuration changes (medium-high impact)
            config_patterns = ['config', 'settings', 'env', '.yaml', '.json', '.toml', '.ini']
            config_files = []
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in config_patterns):
                    config_files.append(file_path)
            
            if config_files:
                impact_score += 0.2
                impact_indicators.append(f"Configuration changes: {len(config_files)} files")
            
            # Infrastructure/deployment changes
            infra_patterns = ['docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'deploy']
            infra_files = []
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in infra_patterns):
                    infra_files.append(file_path)
            
            if infra_files:
                impact_score += 0.25
                impact_indicators.append(f"Infrastructure changes: {len(infra_files)} files")
            
            # Test file changes (assess test coverage impact)
            test_patterns = ['test', 'spec', '__tests__', '.test.', '.spec.']
            test_files = []
            production_files = []
            
            for file_path in change_context.files_changed:
                if any(pattern in file_path.lower() for pattern in test_patterns):
                    test_files.append(file_path)
                else:
                    production_files.append(file_path)
            
            # High impact if production changes without corresponding test changes
            if production_files and not test_files:
                impact_score += 0.3
                impact_indicators.append(f"Production changes without tests: {len(production_files)} files")
            elif test_files:
                impact_indicators.append(f"Test files included: {len(test_files)} files")
            
            # Issue priority/severity impact
            issue_labels = [label.get('name', '').lower() for label in change_context.issue_details.get('labels', [])]
            if 'critical' in issue_labels or 'urgent' in issue_labels:
                impact_score += 0.2
                impact_indicators.append("Critical/urgent issue priority")
            elif 'high' in issue_labels:
                impact_score += 0.15
                impact_indicators.append("High priority issue")
            
            # Clamp score between 0 and 1
            impact_score = min(1.0, impact_score)
            
            return {
                'score': impact_score,
                'level': self._score_to_level(impact_score),
                'indicators': impact_indicators,
                'api_files': api_files,
                'config_files': config_files,
                'infra_files': infra_files,
                'test_files': test_files,
                'production_files': production_files
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing impact risk: {e}")
            return {
                'score': 0.3,  # Default to low-medium risk on error
                'level': 'medium',
                'indicators': [f"Impact assessment error: {e}"],
                'error': str(e)
            }
    
    def _assess_historical_risk(self, change_context: ChangeContext) -> Dict[str, Any]:
        """Assess risk based on historical failure patterns."""
        historical_score = 0.0
        historical_indicators = []
        
        try:
            # Check for recent failures in similar areas
            # This would typically query a database of past failures
            # For now, implement basic heuristics
            
            # Check if files have been frequently modified (high churn = higher risk)
            high_churn_threshold = 5  # More than 5 recent changes
            medium_churn_threshold = 3  # More than 3 recent changes
            
            # Simulate churn analysis (in practice, this would query git history)
            high_churn_files = []
            for file_path in change_context.files_changed[:min(5, len(change_context.files_changed))]:
                # Simulate: assume config and core files have higher churn
                if any(pattern in file_path.lower() for pattern in ['config', 'core', 'main', 'index']):
                    high_churn_files.append(file_path)
            
            if high_churn_files:
                historical_score += 0.3
                historical_indicators.append(f"High-churn files: {len(high_churn_files)}")
            
            # Check for previous failure patterns in issue description
            failure_keywords = ['bug', 'error', 'fail', 'broke', 'regression', 'hotfix', 'urgent']
            issue_content = f"{change_context.issue_details.get('title', '')} {change_context.issue_details.get('body', '')}"
            
            failure_indicators_found = []
            for keyword in failure_keywords:
                if keyword.lower() in issue_content.lower():
                    failure_indicators_found.append(keyword)
            
            if failure_indicators_found:
                historical_score += 0.2
                historical_indicators.append(f"Failure keywords: {', '.join(failure_indicators_found)}")
            
            # Check for revert-related changes
            if 'revert' in issue_content.lower() or any('revert' in f.lower() for f in change_context.files_changed):
                historical_score += 0.4
                historical_indicators.append("Revert-related changes detected")
            
            # Check for emergency/hotfix patterns
            emergency_keywords = ['hotfix', 'emergency', 'critical', 'urgent', 'asap']
            emergency_found = any(keyword in issue_content.lower() for keyword in emergency_keywords)
            if emergency_found:
                historical_score += 0.3
                historical_indicators.append("Emergency/hotfix indicators")
            
            # Clamp score between 0 and 1
            historical_score = min(1.0, historical_score)
            
            return {
                'score': historical_score,
                'level': self._score_to_level(historical_score),
                'indicators': historical_indicators,
                'high_churn_files': high_churn_files,
                'failure_keywords': failure_indicators_found,
                'emergency_indicators': emergency_found
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing historical risk: {e}")
            return {
                'score': 0.1,  # Default to low risk on error
                'level': 'low',
                'indicators': [f"Historical assessment error: {e}"],
                'error': str(e)
            }
    
    def _assess_time_pressure_risk(self, change_context: ChangeContext) -> Dict[str, Any]:
        """Assess risk related to time pressure and urgency."""
        time_pressure_score = 0.0
        time_indicators = []
        
        try:
            # Check issue labels for urgency indicators
            issue_labels = [label.get('name', '').lower() for label in change_context.issue_details.get('labels', [])]
            
            if 'urgent' in issue_labels or 'asap' in issue_labels:
                time_pressure_score += 0.4
                time_indicators.append("Urgent priority label")
            elif 'high' in issue_labels:
                time_pressure_score += 0.2
                time_indicators.append("High priority label")
            
            # Check issue content for time pressure indicators
            issue_content = f"{change_context.issue_details.get('title', '')} {change_context.issue_details.get('body', '')}"
            time_keywords = ['urgent', 'asap', 'immediately', 'critical', 'deadline', 'emergency']
            
            time_keywords_found = []
            for keyword in time_keywords:
                if keyword.lower() in issue_content.lower():
                    time_keywords_found.append(keyword)
                    time_pressure_score += 0.1
            
            if time_keywords_found:
                time_indicators.append(f"Time pressure keywords: {', '.join(time_keywords_found)}")
            
            # Check for weekend/after-hours indicators (higher risk due to reduced oversight)
            issue_created = change_context.issue_details.get('createdAt', '')
            if issue_created:
                try:
                    created_time = datetime.fromisoformat(issue_created.replace('Z', '+00:00'))
                    # Weekend or after hours (before 8 AM or after 6 PM UTC)
                    if created_time.weekday() >= 5 or created_time.hour < 8 or created_time.hour > 18:
                        time_pressure_score += 0.15
                        time_indicators.append("Created during off-hours")
                except Exception:
                    pass  # Ignore time parsing errors
            
            # Clamp score between 0 and 1
            time_pressure_score = min(1.0, time_pressure_score)
            
            return {
                'score': time_pressure_score,
                'level': self._score_to_level(time_pressure_score),
                'indicators': time_indicators,
                'time_keywords': time_keywords_found,
                'urgency_labels': [label for label in issue_labels if label in ['urgent', 'asap', 'high', 'critical']]
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing time pressure risk: {e}")
            return {
                'score': 0.0,  # Default to no time pressure on error
                'level': 'low',
                'indicators': [f"Time pressure assessment error: {e}"],
                'error': str(e)
            }
    
    def _matches_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any of the given glob patterns."""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path.lower(), pattern.lower()):
                return True
        return False
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _determine_risk_level(self, total_score: float) -> str:
        """Determine overall risk level from total score."""
        thresholds = self.escalation_thresholds
        
        if total_score >= thresholds.get('critical', 0.8):
            return 'critical'
        elif total_score >= thresholds.get('high', 0.6):
            return 'high'
        elif total_score >= thresholds.get('medium', 0.3):
            return 'medium'
        else:
            return 'low'
    
    def _requires_escalation(self, risk_level: str, factors: RiskFactors) -> bool:
        """Determine if escalation is required based on risk level and factors."""
        # Always escalate critical and high risk
        if risk_level in ['critical', 'high']:
            return True
        
        # Escalate medium risk if security factors are present
        if risk_level == 'medium' and factors.security_score >= 0.5:
            return True
        
        # Escalate if multiple factors are elevated
        elevated_factors = sum([
            1 for score in [factors.security_score, factors.complexity_score, 
                          factors.impact_score, factors.historical_score]
            if score >= 0.6
        ])
        
        if elevated_factors >= 2:
            return True
        
        return False
    
    def _determine_specialist_type(self, factors: RiskFactors, change_context: ChangeContext) -> Optional[str]:
        """Determine what type of specialist should review the change."""
        specialist_routing = self.config.get('specialist_routing', {})
        
        # Security specialist for high security risk
        if factors.security_score >= 0.6:
            return 'security'
        
        # Architecture specialist for high complexity or impact
        if factors.complexity_score >= 0.7 or factors.impact_score >= 0.7:
            return 'architecture'
        
        # Compliance specialist for audit-related changes
        audit_keywords = ['audit', 'compliance', 'privacy', 'gdpr', 'regulation']
        issue_content = f"{change_context.issue_details.get('title', '')} {change_context.issue_details.get('body', '')}"
        if any(keyword in issue_content.lower() for keyword in audit_keywords):
            return 'compliance'
        
        # Default to architecture specialist for any escalation
        return 'architecture'
    
    def _generate_risk_reasoning(self, factors: RiskFactors, change_context: ChangeContext) -> List[str]:
        """Generate human-readable reasoning for the risk assessment."""
        reasoning = []
        
        # Security reasoning
        if factors.security_score >= 0.5:
            security_details = factors.metadata.get('security_details', {})
            reasoning.append(f"🔒 High security risk ({factors.security_score:.2f}): {', '.join(security_details.get('indicators', []))}")
        
        # Complexity reasoning
        if factors.complexity_score >= 0.5:
            complexity_details = factors.metadata.get('complexity_details', {})
            reasoning.append(f"🔧 High complexity ({factors.complexity_score:.2f}): {', '.join(complexity_details.get('indicators', []))}")
        
        # Impact reasoning
        if factors.impact_score >= 0.5:
            impact_details = factors.metadata.get('impact_details', {})
            reasoning.append(f"📊 High impact ({factors.impact_score:.2f}): {', '.join(impact_details.get('indicators', []))}")
        
        # Historical reasoning
        if factors.historical_score >= 0.3:
            historical_details = factors.metadata.get('historical_details', {})
            reasoning.append(f"⚠️ Historical risk factors ({factors.historical_score:.2f}): {', '.join(historical_details.get('indicators', []))}")
        
        # Time pressure reasoning
        if factors.time_pressure_score >= 0.3:
            time_details = factors.metadata.get('time_pressure_details', {})
            reasoning.append(f"⏰ Time pressure factors ({factors.time_pressure_score:.2f}): {', '.join(time_details.get('indicators', []))}")
        
        if not reasoning:
            reasoning.append("✅ Low risk change with standard review requirements")
        
        return reasoning
    
    def _calculate_confidence(self, factors: RiskFactors, change_context: ChangeContext) -> float:
        """Calculate confidence in the risk assessment."""
        confidence = 1.0
        
        # Reduce confidence if we have errors in any factor assessment
        for factor_name, factor_details in factors.metadata.items():
            if factor_details.get('error'):
                confidence -= 0.2
        
        # Reduce confidence if we have incomplete data
        if not change_context.files_changed:
            confidence -= 0.3
        
        if not change_context.issue_details:
            confidence -= 0.2
        
        # Increase confidence if we have rich data
        if len(change_context.files_changed) > 0 and change_context.lines_added + change_context.lines_removed > 0:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))

def create_change_context_from_issue(issue_number: int) -> Optional[ChangeContext]:
    """
    Create a ChangeContext from a GitHub issue number.
    
    Args:
        issue_number: GitHub issue number
        
    Returns:
        ChangeContext object with populated data from GitHub
    """
    try:
        # Get issue details
        result = subprocess.run([
            'gh', 'issue', 'view', str(issue_number),
            '--json', 'title,body,labels,createdAt,state'
        ], capture_output=True, text=True, check=True)
        
        issue_details = json.loads(result.stdout)
        
        # Try to get associated PR information
        try:
            pr_result = subprocess.run([
                'gh', 'pr', 'list', '--search', f'fixes #{issue_number}',
                '--json', 'number,headRefName,author'
            ], capture_output=True, text=True, check=True)
            
            prs = json.loads(pr_result.stdout)
            pr_info = prs[0] if prs else None
            
            if pr_info:
                # Get PR file changes
                pr_files_result = subprocess.run([
                    'gh', 'pr', 'view', str(pr_info['number']),
                    '--json', 'files'
                ], capture_output=True, text=True, check=True)
                
                pr_data = json.loads(pr_files_result.stdout)
                files_changed = [f['path'] for f in pr_data.get('files', [])]
                lines_added = sum(f.get('additions', 0) for f in pr_data.get('files', []))
                lines_removed = sum(f.get('deletions', 0) for f in pr_data.get('files', []))
                
                return ChangeContext(
                    files_changed=files_changed,
                    lines_added=lines_added,
                    lines_removed=lines_removed,
                    issue_number=issue_number,
                    issue_details=issue_details,
                    pr_number=pr_info['number'],
                    branch_name=pr_info.get('headRefName'),
                    author=pr_info.get('author', {}).get('login')
                )
        except Exception:
            pass  # No PR found, continue with issue-only context
        
        # Create context from issue only (simulate some file changes for analysis)
        return ChangeContext(
            files_changed=[],  # Empty files list when no PR available
            lines_added=0,
            lines_removed=0,
            issue_number=issue_number,
            issue_details=issue_details
        )
        
    except Exception as e:
        logging.error(f"Error creating change context from issue #{issue_number}: {e}")
        return None

def main():
    """Command line interface for risk assessment engine."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python risk_assessment_engine.py <command> [args]")
        print("Commands:")
        print("  assess-issue <issue_number>     - Assess risk for GitHub issue")
        print("  test-config                     - Test configuration loading")
        return
    
    command = sys.argv[1]
    engine = RiskAssessmentEngine()
    
    if command == "assess-issue" and len(sys.argv) >= 3:
        issue_num = int(sys.argv[2])
        change_context = create_change_context_from_issue(issue_num)
        
        if not change_context:
            print(f"Error: Could not create change context for issue #{issue_num}")
            return 1
        
        risk_score = engine.assess_change_risk(change_context)
        
        # Output detailed risk assessment
        result = {
            'issue_number': issue_num,
            'timestamp': datetime.now().isoformat(),
            'risk_assessment': {
                'total_score': risk_score.total_score,
                'risk_level': risk_score.risk_level,
                'escalation_required': risk_score.escalation_required,
                'specialist_type': risk_score.specialist_type,
                'confidence': risk_score.confidence,
                'reasoning': risk_score.reasoning,
                'risk_factors': {
                    'security': risk_score.factors.security_score,
                    'complexity': risk_score.factors.complexity_score,
                    'impact': risk_score.factors.impact_score,
                    'historical': risk_score.factors.historical_score,
                    'time_pressure': risk_score.factors.time_pressure_score
                }
            }
        }
        
        print(json.dumps(result, indent=2))
        
        # Return exit code based on escalation requirement
        return 2 if risk_score.escalation_required else 0
        
    elif command == "test-config":
        print("Configuration loaded successfully:")
        print(json.dumps(engine.config, indent=2))
        return 0
        
    else:
        print(f"Unknown command: {command}")
        return 1

if __name__ == "__main__":
    exit(main())