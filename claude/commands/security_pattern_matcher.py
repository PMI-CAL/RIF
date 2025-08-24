#!/usr/bin/env python3
"""
Security Pattern Matcher - Issue #92 Phase 1
Advanced security pattern detection system for risk assessment.

This module implements:
1. File path pattern matching for security-sensitive areas
2. Content analysis for security-related keywords
3. Change pattern analysis for authentication/authorization modifications
4. Configuration file security assessment
5. Dependency security analysis
6. Historical security incident correlation
"""

import json
import re
import fnmatch
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
import yaml

@dataclass
class SecurityPattern:
    """Container for security pattern information."""
    name: str
    pattern_type: str  # 'path', 'content', 'dependency', 'configuration'
    pattern: str
    risk_level: str    # 'low', 'medium', 'high', 'critical'
    description: str
    remediation_guidance: str = ""

@dataclass
class SecurityMatch:
    """Container for security pattern match results."""
    pattern: SecurityPattern
    matched_files: List[str]
    matched_content: List[str]
    confidence: float
    context: Dict[str, Any]

@dataclass
class SecurityAssessment:
    """Container for comprehensive security assessment results."""
    overall_risk_score: float
    risk_level: str
    matches: List[SecurityMatch]
    recommendations: List[str]
    escalation_required: bool
    specialist_required: bool
    confidence: float

class SecurityPatternMatcher:
    """
    Advanced security pattern detection and risk assessment.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the security pattern matcher."""
        self.config_path = config_path
        self.setup_logging()
        self.config = self._load_config()
        self.security_patterns = self._load_security_patterns()
    
    def setup_logging(self):
        """Setup logging for security pattern matching."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - SecurityPatternMatcher - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security pattern configuration."""
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
        """Get default security pattern configuration."""
        return {
            'security_patterns': {
                'high_risk_paths': [
                    'auth/**', '**/security/**', '*/payment/**', 
                    '**/authentication/**', '**/authorization/**'
                ],
                'medium_risk_paths': [
                    '**/api/**', '**/database/**', '**/config/**'
                ],
                'high_risk_keywords': [
                    'password', 'secret', 'token', 'auth', 'login',
                    'crypto', 'encrypt', 'decrypt', 'certificate'
                ]
            }
        }
    
    def _load_security_patterns(self) -> List[SecurityPattern]:
        """Load and create security patterns from configuration."""
        patterns = []
        
        try:
            security_config = self.config.get('security_patterns', {})
            
            # File path patterns - High Risk
            high_risk_paths = security_config.get('high_risk_paths', [])
            for path_pattern in high_risk_paths:
                patterns.append(SecurityPattern(
                    name=f"high_risk_path_{len(patterns)}",
                    pattern_type='path',
                    pattern=path_pattern,
                    risk_level='high',
                    description=f"High-risk security-sensitive file path: {path_pattern}",
                    remediation_guidance="Requires security specialist review"
                ))
            
            # File path patterns - Medium Risk
            medium_risk_paths = security_config.get('medium_risk_paths', [])
            for path_pattern in medium_risk_paths:
                patterns.append(SecurityPattern(
                    name=f"medium_risk_path_{len(patterns)}",
                    pattern_type='path',
                    pattern=path_pattern,
                    risk_level='medium',
                    description=f"Medium-risk security-relevant file path: {path_pattern}",
                    remediation_guidance="Consider security review"
                ))
            
            # Content-based patterns
            high_risk_keywords = security_config.get('high_risk_keywords', [])
            for keyword in high_risk_keywords:
                patterns.append(SecurityPattern(
                    name=f"keyword_{keyword}",
                    pattern_type='content',
                    pattern=keyword,
                    risk_level='medium',
                    description=f"Security-sensitive keyword: {keyword}",
                    remediation_guidance="Verify secure handling of sensitive data"
                ))
            
            # Add specialized security patterns
            patterns.extend(self._create_specialized_patterns())
            
            self.logger.info(f"Loaded {len(patterns)} security patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error loading security patterns: {e}")
            return []
    
    def _create_specialized_patterns(self) -> List[SecurityPattern]:
        """Create specialized security patterns for advanced detection."""
        patterns = []
        
        # Authentication bypass patterns
        patterns.append(SecurityPattern(
            name="auth_bypass_risk",
            pattern_type='content',
            pattern=r'(?i)(bypass|skip|ignore).*(auth|login|permission|check)',
            risk_level='critical',
            description="Potential authentication bypass pattern",
            remediation_guidance="CRITICAL: Manual security review required immediately"
        ))
        
        # SQL injection vulnerability patterns
        patterns.append(SecurityPattern(
            name="sql_injection_risk",
            pattern_type='content', 
            pattern=r'(?i)(execute|query|select).*(input|user|param|request)',
            risk_level='high',
            description="Potential SQL injection vulnerability pattern",
            remediation_guidance="Review for parameterized queries and input validation"
        ))
        
        # Hardcoded credentials patterns
        patterns.append(SecurityPattern(
            name="hardcoded_credentials",
            pattern_type='content',
            pattern=r'(?i)(password|secret|key|token)\s*=\s*["\'][\w\-_!@#$%^&*()+=]{8,}["\']',
            risk_level='critical',
            description="Potential hardcoded credentials",
            remediation_guidance="CRITICAL: Remove hardcoded credentials, use secure configuration"
        ))
        
        # Insecure random number generation
        patterns.append(SecurityPattern(
            name="weak_random",
            pattern_type='content',
            pattern=r'(?i)(math\.random|random\.random|rand\(\))',
            risk_level='medium',
            description="Insecure random number generation for security contexts",
            remediation_guidance="Use cryptographically secure random number generation"
        ))
        
        # Unsafe deserialization patterns
        patterns.append(SecurityPattern(
            name="unsafe_deserialization",
            pattern_type='content',
            pattern=r'(?i)(pickle\.loads|eval\(|exec\(|yaml\.load)',
            risk_level='high',
            description="Unsafe deserialization that could lead to RCE",
            remediation_guidance="Use safe deserialization methods and input validation"
        ))
        
        # CORS misconfiguration patterns
        patterns.append(SecurityPattern(
            name="cors_misconfiguration",
            pattern_type='content',
            pattern=r'(?i)(access-control-allow-origin.*\*|cors.*origin.*\*)',
            risk_level='medium',
            description="Potential CORS misconfiguration allowing any origin",
            remediation_guidance="Restrict CORS to specific trusted origins"
        ))
        
        # Debug/development code in production
        patterns.append(SecurityPattern(
            name="debug_code_production",
            pattern_type='content',
            pattern=r'(?i)(console\.log|print\(|debug\s*=\s*true|development.*mode)',
            risk_level='low',
            description="Debug/development code that shouldn't be in production",
            remediation_guidance="Remove debug code before production deployment"
        ))
        
        # Certificate and TLS patterns
        patterns.append(SecurityPattern(
            name="tls_security_config",
            pattern_type='path',
            pattern='**/ssl/**',
            risk_level='high',
            description="TLS/SSL configuration changes",
            remediation_guidance="Verify secure TLS configuration and certificate handling"
        ))
        
        # API key and token patterns in config
        patterns.append(SecurityPattern(
            name="config_secrets",
            pattern_type='path',
            pattern='**/*.{env,config,properties,yaml,json}',
            risk_level='medium',
            description="Configuration files that may contain secrets",
            remediation_guidance="Ensure no hardcoded secrets in configuration files"
        ))
        
        return patterns
    
    def assess_security_risk(self, files_changed: List[str], issue_content: str = "") -> SecurityAssessment:
        """
        Perform comprehensive security risk assessment.
        
        Args:
            files_changed: List of file paths that were changed
            issue_content: Combined issue title and body content
            
        Returns:
            SecurityAssessment with detailed results
        """
        self.logger.info(f"🔒 Assessing security risk for {len(files_changed)} files")
        
        matches = []
        overall_risk_score = 0.0
        recommendations = []
        
        try:
            # Analyze file path patterns
            path_matches = self._analyze_file_paths(files_changed)
            matches.extend(path_matches)
            
            # Analyze content patterns in issue description
            if issue_content:
                content_matches = self._analyze_content(issue_content)
                matches.extend(content_matches)
            
            # Analyze configuration file changes
            config_matches = self._analyze_configuration_files(files_changed)
            matches.extend(config_matches)
            
            # Analyze dependency changes (if any package files changed)
            dependency_matches = self._analyze_dependency_changes(files_changed)
            matches.extend(dependency_matches)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_security_risk_score(matches)
            
            # Determine risk level
            risk_level = self._determine_security_risk_level(overall_risk_score)
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(matches, risk_level)
            
            # Determine escalation needs
            escalation_required = self._requires_security_escalation(risk_level, matches)
            specialist_required = escalation_required or overall_risk_score >= 0.6
            
            # Calculate confidence in assessment
            confidence = self._calculate_assessment_confidence(matches, files_changed, issue_content)
            
            assessment = SecurityAssessment(
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                matches=matches,
                recommendations=recommendations,
                escalation_required=escalation_required,
                specialist_required=specialist_required,
                confidence=confidence
            )
            
            self.logger.info(f"🔒 Security assessment complete: {risk_level} risk ({overall_risk_score:.2f}), {len(matches)} matches")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in security risk assessment: {e}")
            # Return safe default - high risk requiring review
            return SecurityAssessment(
                overall_risk_score=0.8,
                risk_level='high',
                matches=[],
                recommendations=[f"Security assessment error: {e}"],
                escalation_required=True,
                specialist_required=True,
                confidence=0.0
            )
    
    def _analyze_file_paths(self, files_changed: List[str]) -> List[SecurityMatch]:
        """Analyze file paths for security patterns."""
        matches = []
        
        for pattern in self.security_patterns:
            if pattern.pattern_type != 'path':
                continue
            
            matched_files = []
            for file_path in files_changed:
                if self._matches_path_pattern(file_path, pattern.pattern):
                    matched_files.append(file_path)
            
            if matched_files:
                confidence = min(1.0, len(matched_files) * 0.3)  # Higher confidence with more matches
                matches.append(SecurityMatch(
                    pattern=pattern,
                    matched_files=matched_files,
                    matched_content=[],
                    confidence=confidence,
                    context={'match_type': 'file_path', 'file_count': len(matched_files)}
                ))
        
        return matches
    
    def _analyze_content(self, content: str) -> List[SecurityMatch]:
        """Analyze content for security patterns."""
        matches = []
        
        for pattern in self.security_patterns:
            if pattern.pattern_type != 'content':
                continue
            
            matched_content = []
            
            # Handle regex patterns
            if pattern.pattern.startswith('(?i)') or '\\' in pattern.pattern:
                try:
                    regex_matches = re.findall(pattern.pattern, content, re.IGNORECASE)
                    matched_content = [str(match) for match in regex_matches if match]
                except re.error:
                    # If regex fails, fall back to simple string search
                    if pattern.pattern.lower() in content.lower():
                        matched_content = [pattern.pattern]
            else:
                # Simple keyword search
                if pattern.pattern.lower() in content.lower():
                    matched_content = [pattern.pattern]
            
            if matched_content:
                confidence = min(1.0, len(matched_content) * 0.2)
                matches.append(SecurityMatch(
                    pattern=pattern,
                    matched_files=[],
                    matched_content=matched_content,
                    confidence=confidence,
                    context={'match_type': 'content', 'match_count': len(matched_content)}
                ))
        
        return matches
    
    def _analyze_configuration_files(self, files_changed: List[str]) -> List[SecurityMatch]:
        """Analyze configuration files for security risks."""
        matches = []
        
        config_extensions = ['.env', '.config', '.properties', '.yaml', '.yml', '.json', '.ini', '.conf']
        config_files = []
        
        for file_path in files_changed:
            file_lower = file_path.lower()
            if any(ext in file_lower for ext in config_extensions) or 'config' in file_lower:
                config_files.append(file_path)
        
        if config_files:
            # Create a match for configuration file changes
            pattern = SecurityPattern(
                name="configuration_file_changes",
                pattern_type='configuration',
                pattern="configuration",
                risk_level='medium',
                description="Configuration files changed - may contain sensitive settings",
                remediation_guidance="Review configuration changes for hardcoded secrets or security settings"
            )
            
            matches.append(SecurityMatch(
                pattern=pattern,
                matched_files=config_files,
                matched_content=[],
                confidence=0.8,
                context={'match_type': 'configuration', 'config_file_count': len(config_files)}
            ))
        
        return matches
    
    def _analyze_dependency_changes(self, files_changed: List[str]) -> List[SecurityMatch]:
        """Analyze dependency file changes for security risks."""
        matches = []
        
        dependency_files = [
            'package.json', 'package-lock.json', 'yarn.lock',
            'requirements.txt', 'Pipfile', 'Pipfile.lock', 'poetry.lock',
            'pom.xml', 'build.gradle', 'Cargo.toml', 'go.mod'
        ]
        
        changed_dependency_files = []
        for file_path in files_changed:
            if any(dep_file in file_path for dep_file in dependency_files):
                changed_dependency_files.append(file_path)
        
        if changed_dependency_files:
            pattern = SecurityPattern(
                name="dependency_changes",
                pattern_type='dependency',
                pattern="dependencies",
                risk_level='medium',
                description="Dependency changes - may introduce security vulnerabilities",
                remediation_guidance="Run security audit on dependencies and check for known vulnerabilities"
            )
            
            matches.append(SecurityMatch(
                pattern=pattern,
                matched_files=changed_dependency_files,
                matched_content=[],
                confidence=0.7,
                context={'match_type': 'dependency', 'dependency_file_count': len(changed_dependency_files)}
            ))
        
        return matches
    
    def _matches_path_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches the security pattern."""
        try:
            # Use fnmatch for glob-style pattern matching
            return fnmatch.fnmatch(file_path.lower(), pattern.lower())
        except Exception:
            # Fall back to simple substring matching
            return pattern.lower() in file_path.lower()
    
    def _calculate_security_risk_score(self, matches: List[SecurityMatch]) -> float:
        """Calculate overall security risk score from matches."""
        if not matches:
            return 0.0
        
        risk_weights = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for match in matches:
            risk_weight = risk_weights.get(match.pattern.risk_level, 0.25)
            weighted_score = risk_weight * match.confidence
            
            # Apply multiplier based on number of matches
            if match.matched_files:
                multiplier = min(2.0, 1.0 + len(match.matched_files) * 0.1)
            elif match.matched_content:
                multiplier = min(2.0, 1.0 + len(match.matched_content) * 0.05)
            else:
                multiplier = 1.0
            
            total_weighted_score += weighted_score * multiplier
            total_weight += risk_weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize and clamp between 0 and 1
        normalized_score = total_weighted_score / max(total_weight, 1.0)
        return min(1.0, normalized_score)
    
    def _determine_security_risk_level(self, risk_score: float) -> str:
        """Determine security risk level from score."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_security_recommendations(self, matches: List[SecurityMatch], risk_level: str) -> List[str]:
        """Generate security recommendations based on matches."""
        recommendations = []
        
        if not matches:
            recommendations.append("✅ No security patterns detected in the changes")
            return recommendations
        
        # Risk level recommendations
        if risk_level == 'critical':
            recommendations.append("🚨 CRITICAL: Immediate security specialist review required")
            recommendations.append("🚨 Do not merge until security concerns are addressed")
        elif risk_level == 'high':
            recommendations.append("⚠️ HIGH RISK: Security specialist review recommended")
            recommendations.append("⚠️ Consider additional security testing")
        elif risk_level == 'medium':
            recommendations.append("⚡ MEDIUM RISK: Security review recommended")
        
        # Pattern-specific recommendations
        critical_patterns = [m for m in matches if m.pattern.risk_level == 'critical']
        if critical_patterns:
            recommendations.append("🔴 Critical security patterns detected:")
            for match in critical_patterns:
                recommendations.append(f"   • {match.pattern.description}")
                if match.pattern.remediation_guidance:
                    recommendations.append(f"     Remediation: {match.pattern.remediation_guidance}")
        
        high_risk_patterns = [m for m in matches if m.pattern.risk_level == 'high']
        if high_risk_patterns:
            recommendations.append("🟡 High-risk security patterns detected:")
            for match in high_risk_patterns[:3]:  # Limit to top 3
                recommendations.append(f"   • {match.pattern.description}")
        
        # General security recommendations
        auth_matches = [m for m in matches if 'auth' in m.pattern.name.lower()]
        if auth_matches:
            recommendations.append("🔐 Authentication/authorization changes require careful review")
        
        config_matches = [m for m in matches if 'config' in m.pattern.pattern_type.lower()]
        if config_matches:
            recommendations.append("⚙️ Configuration changes - verify no hardcoded secrets")
        
        dependency_matches = [m for m in matches if 'dependency' in m.pattern.pattern_type.lower()]
        if dependency_matches:
            recommendations.append("📦 Dependency changes - run security audit")
        
        return recommendations
    
    def _requires_security_escalation(self, risk_level: str, matches: List[SecurityMatch]) -> bool:
        """Determine if security escalation is required."""
        # Always escalate critical and high risk
        if risk_level in ['critical', 'high']:
            return True
        
        # Escalate if we have critical pattern matches regardless of overall score
        critical_matches = [m for m in matches if m.pattern.risk_level == 'critical']
        if critical_matches:
            return True
        
        # Escalate if multiple high-confidence security matches
        high_confidence_matches = [m for m in matches if m.confidence >= 0.8 and m.pattern.risk_level in ['high', 'medium']]
        if len(high_confidence_matches) >= 3:
            return True
        
        return False
    
    def _calculate_assessment_confidence(self, matches: List[SecurityMatch], files_changed: List[str], issue_content: str) -> float:
        """Calculate confidence in the security assessment."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more data
        if files_changed:
            confidence += 0.2
        
        if issue_content and len(issue_content) > 50:
            confidence += 0.2
        
        # Increase confidence with high-confidence matches
        if matches:
            avg_match_confidence = sum(m.confidence for m in matches) / len(matches)
            confidence += avg_match_confidence * 0.3
        
        # Decrease confidence if we have error conditions
        error_matches = [m for m in matches if 'error' in m.context]
        if error_matches:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))

def main():
    """Command line interface for security pattern matcher."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python security_pattern_matcher.py <command> [args]")
        print("Commands:")
        print("  analyze-files <file1> <file2> ...  - Analyze file paths for security patterns")
        print("  analyze-content <content>          - Analyze content for security patterns")
        print("  list-patterns                      - List all loaded security patterns")
        print("  test-pattern <pattern> <text>      - Test a specific pattern against text")
        return
    
    command = sys.argv[1]
    matcher = SecurityPatternMatcher()
    
    if command == "analyze-files" and len(sys.argv) >= 3:
        files = sys.argv[2:]
        assessment = matcher.assess_security_risk(files, "")
        
        result = {
            'files_analyzed': files,
            'security_assessment': {
                'risk_score': assessment.overall_risk_score,
                'risk_level': assessment.risk_level,
                'escalation_required': assessment.escalation_required,
                'specialist_required': assessment.specialist_required,
                'confidence': assessment.confidence,
                'matches_found': len(assessment.matches),
                'recommendations': assessment.recommendations
            },
            'detailed_matches': [
                {
                    'pattern_name': match.pattern.name,
                    'risk_level': match.pattern.risk_level,
                    'description': match.pattern.description,
                    'matched_files': match.matched_files,
                    'confidence': match.confidence
                }
                for match in assessment.matches
            ]
        }
        
        print(json.dumps(result, indent=2))
        
    elif command == "analyze-content" and len(sys.argv) >= 3:
        content = sys.argv[2]
        assessment = matcher.assess_security_risk([], content)
        
        result = {
            'content_analyzed': len(content),
            'security_assessment': {
                'risk_score': assessment.overall_risk_score,
                'risk_level': assessment.risk_level,
                'matches_found': len(assessment.matches),
                'recommendations': assessment.recommendations
            }
        }
        
        print(json.dumps(result, indent=2))
        
    elif command == "list-patterns":
        patterns_info = []
        for pattern in matcher.security_patterns:
            patterns_info.append({
                'name': pattern.name,
                'type': pattern.pattern_type,
                'risk_level': pattern.risk_level,
                'description': pattern.description,
                'pattern': pattern.pattern
            })
        
        print(json.dumps({
            'total_patterns': len(patterns_info),
            'patterns': patterns_info
        }, indent=2))
        
    elif command == "test-pattern" and len(sys.argv) >= 4:
        pattern_text = sys.argv[2]
        test_text = sys.argv[3]
        
        # Test regex or simple match
        if pattern_text.startswith('(?i)') or '\\' in pattern_text:
            try:
                matches = re.findall(pattern_text, test_text, re.IGNORECASE)
                result = {'pattern': pattern_text, 'text': test_text, 'matches': matches}
            except re.error as e:
                result = {'pattern': pattern_text, 'text': test_text, 'error': str(e)}
        else:
            matches = pattern_text.lower() in test_text.lower()
            result = {'pattern': pattern_text, 'text': test_text, 'matches': matches}
        
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())