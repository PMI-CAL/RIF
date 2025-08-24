#!/usr/bin/env python3
"""
RIF Security Specialist Agent - Issue #74
Specialized agent for security vulnerability detection, compliance checking, and security best practices
"""

import re
import os
import json
import hashlib
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .domain_agent_base import DomainAgent, TaskResult, AgentStatus, AgentConfiguration

logger = logging.getLogger(__name__)

class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    SECRETS_EXPOSURE = "secrets_exposure"
    CRYPTO_FAILURES = "cryptographic_failures"

class SeverityLevel(Enum):
    """Severity levels for vulnerabilities"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    OWASP_TOP_10 = "owasp_top_10"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOC2 = "soc2"
    NIST = "nist"

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability found in code"""
    vuln_id: str
    vuln_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int = 0
    code_snippet: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    cwe_id: Optional[str] = None
    confidence: float = 0.8

@dataclass
class SecurityScanResult:
    """Complete result of security scan"""
    scan_id: str
    scan_type: str
    start_time: datetime
    end_time: datetime
    vulnerabilities: List[SecurityVulnerability]
    files_scanned: int
    lines_scanned: int
    summary: Dict[str, int]
    recommendations: List[str]

@dataclass
class ComplianceResult:
    """Result of compliance checking"""
    standard: ComplianceStandard
    overall_score: float
    passed_checks: int
    failed_checks: int
    total_checks: int
    findings: List[Dict[str, Any]]
    recommendations: List[str]

class SecuritySpecialistAgent(DomainAgent):
    """
    Security Specialist Agent for comprehensive security analysis
    
    Capabilities:
    - OWASP Top 10 vulnerability detection
    - Dependency vulnerability scanning
    - Secrets detection
    - Authentication/Authorization analysis
    - Compliance checking
    - Security best practices validation
    """
    
    def __init__(self, config: Optional[AgentConfiguration] = None):
        super().__init__(
            domain='security',
            capabilities=[
                'vulnerability_detection',
                'compliance_checking', 
                'threat_modeling',
                'security_scanning',
                'secrets_detection',
                'dependency_analysis',
                'auth_analysis',
                'crypto_analysis'
            ],
            name="SecuritySpecialist",
            expertise=[
                'owasp_top_10',
                'secure_coding',
                'penetration_testing',
                'compliance_auditing',
                'threat_analysis'
            ],
            tools=[
                'static_analysis',
                'dependency_checker',
                'secrets_scanner',
                'compliance_validator'
            ],
            config=config or AgentConfiguration()
        )
        
        # Security-specific patterns and rules
        self._init_security_patterns()
        
        # Track scan history
        self.scan_history: List[SecurityScanResult] = []
        self.compliance_history: List[ComplianceResult] = []

    def _init_security_patterns(self):
        """Initialize security vulnerability patterns"""
        # SQL Injection patterns
        self.sql_injection_patterns = [
            r'(?i)(select|insert|update|delete|union|drop)\s+.*\+.*["\']',
            r'(?i)execute\s*\(\s*["\'].*\+',
            r'(?i)executescript\s*\(\s*["\'].*\+',
            r'(?i)cursor\.execute\s*\(\s*["\'].*\+',
            r'(?i)query\s*\(\s*["\'].*\+.*["\']',
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r'document\.write\s*\(',
            r'innerHTML\s*=',
            r'outerHTML\s*=',
            r'eval\s*\(',
            r'setTimeout\s*\(\s*["\'].*\+',
            r'setInterval\s*\(\s*["\'].*\+'
        ]
        
        # Hardcoded secrets patterns
        self.secrets_patterns = [
            r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][^"\']{8,}["\']',
            r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\'][^"\']{8,}["\']',
            r'(?i)(password|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']',
            r'(?i)(token)\s*[=:]\s*["\'][^"\']{20,}["\']',
            r'(?i)(aws[_-]?access[_-]?key)\s*[=:]\s*["\'][A-Z0-9]{20}["\']',
            r'(?i)(aws[_-]?secret[_-]?key)\s*[=:]\s*["\'][A-Za-z0-9/+=]{40}["\']',
            r'(?i)(database[_-]?url)\s*[=:]\s*["\'][^"\']*://[^"\']*["\']',
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
        ]
        
        # Insecure crypto patterns
        self.crypto_patterns = [
            r'(?i)md5\s*\(',
            r'(?i)sha1\s*\(',
            r'(?i)des\s*\(',
            r'(?i)rc4\s*\(',
            r'(?i)\.encrypt\s*\(\s*.*\s*,\s*["\'].*["\']',  # Hardcoded keys
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./\.\./\.\.',
            r'\.\.\\\.\.\\\.\.\\',
            r'os\.path\.join\s*\(.*\.\.',
            r'open\s*\(\s*.*\.\.',
        ]

    def execute_primary_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """
        Execute security analysis task
        
        Args:
            task_data: Must contain 'scan_type' and 'target_path'
            
        Returns:
            TaskResult with security scan results
        """
        scan_type = task_data.get('scan_type', 'comprehensive')
        target_path = task_data.get('target_path', '.')
        
        result = TaskResult(
            task_id="",  # Will be set by parent class
            status=AgentStatus.ACTIVE,
            start_time=datetime.now()
        )
        
        try:
            if scan_type == 'comprehensive':
                scan_result = self.security_scan(target_path)
            elif scan_type == 'secrets':
                scan_result = self.detect_secrets(target_path)
            elif scan_type == 'dependencies':
                scan_result = self.check_dependencies(target_path)
            elif scan_type == 'owasp':
                scan_result = self.check_owasp_top10(target_path)
            elif scan_type == 'compliance':
                standards = task_data.get('standards', [ComplianceStandard.OWASP_TOP_10])
                scan_result = self.compliance_check(target_path, standards)
            else:
                raise ValueError(f"Unknown scan type: {scan_type}")
            
            result.result_data = scan_result
            result.confidence_score = self._calculate_confidence(scan_result)
            result.artifacts = self._generate_artifacts(scan_result)
            
            return result
            
        except Exception as e:
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
            return result

    def security_scan(self, code_base: str) -> Dict[str, Any]:
        """
        Comprehensive security scan of codebase
        
        Args:
            code_base: Path to code directory to scan
            
        Returns:
            Dict containing comprehensive security analysis
        """
        scan_start = datetime.now()
        scan_id = f"scan_{int(scan_start.timestamp())}"
        
        logger.info(f"Starting comprehensive security scan of {code_base}")
        
        vulnerabilities = []
        files_scanned = 0
        lines_scanned = 0
        
        # OWASP Top 10 checks
        owasp_vulns = self.check_owasp_top10(code_base)
        vulnerabilities.extend(owasp_vulns)
        
        # Dependency vulnerabilities
        dep_vulns = self.check_dependencies(code_base)
        vulnerabilities.extend(dep_vulns)
        
        # Secret detection
        secret_vulns = self.detect_secrets(code_base)
        vulnerabilities.extend(secret_vulns)
        
        # Authentication/Authorization checks
        auth_vulns = self.check_auth(code_base)
        vulnerabilities.extend(auth_vulns)
        
        # Count files and lines scanned
        for root, dirs, files in os.walk(code_base):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if self._is_source_file(file):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines_scanned += len(f.readlines())
                        files_scanned += 1
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
        
        # Prioritize vulnerabilities by severity
        prioritized_vulns = self.prioritize_vulnerabilities(vulnerabilities)
        
        scan_end = datetime.now()
        
        # Create summary
        severity_counts = {}
        for severity in SeverityLevel:
            severity_counts[severity.name.lower()] = sum(1 for v in prioritized_vulns if v.severity == severity)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(prioritized_vulns)
        
        scan_result = SecurityScanResult(
            scan_id=scan_id,
            scan_type="comprehensive",
            start_time=scan_start,
            end_time=scan_end,
            vulnerabilities=prioritized_vulns,
            files_scanned=files_scanned,
            lines_scanned=lines_scanned,
            summary=severity_counts,
            recommendations=recommendations
        )
        
        self.scan_history.append(scan_result)
        
        result = {
            "scan_id": scan_id,
            "vulnerabilities": [self._vuln_to_dict(v) for v in prioritized_vulns],
            "summary": severity_counts,
            "files_scanned": files_scanned,
            "lines_scanned": lines_scanned,
            "recommendations": recommendations,
            "scan_duration_seconds": (scan_end - scan_start).total_seconds()
        }
        
        logger.info(f"Security scan completed: {len(prioritized_vulns)} vulnerabilities found")
        return result

    def check_owasp_top10(self, code_base: str) -> List[SecurityVulnerability]:
        """
        Check for OWASP Top 10 vulnerabilities
        
        Args:
            code_base: Path to code directory
            
        Returns:
            List of OWASP vulnerabilities found
        """
        vulnerabilities = []
        
        logger.info("Checking for OWASP Top 10 vulnerabilities")
        
        for root, dirs, files in os.walk(code_base):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if self._is_source_file(file):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, code_base)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')
                        
                        # Check for various OWASP vulnerabilities
                        vulnerabilities.extend(self._check_injection_vulns(content, lines, rel_path))
                        vulnerabilities.extend(self._check_xss_vulns(content, lines, rel_path))
                        vulnerabilities.extend(self._check_broken_auth(content, lines, rel_path))
                        vulnerabilities.extend(self._check_sensitive_data(content, lines, rel_path))
                        vulnerabilities.extend(self._check_security_misconfig(content, lines, rel_path))
                        vulnerabilities.extend(self._check_crypto_failures(content, lines, rel_path))
                        
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")
        
        return vulnerabilities

    def _check_injection_vulns(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for injection vulnerabilities"""
        vulns = []
        
        for i, line in enumerate(lines):
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"inj_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.INJECTION,
                        severity=SeverityLevel.HIGH,
                        title="Potential SQL Injection",
                        description="Dynamic SQL query construction detected. This may be vulnerable to SQL injection attacks.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Use parameterized queries or prepared statements instead of string concatenation.",
                        references=["https://owasp.org/Top10/A03_2021-Injection/"],
                        cwe_id="CWE-89",
                        confidence=0.8
                    ))
        
        # Check for command injection
        cmd_injection_patterns = [
            r'os\.system\s*\(.*\+',
            r'subprocess\.(call|run|Popen)\s*\(.*\+',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        for i, line in enumerate(lines):
            for pattern in cmd_injection_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"cmd_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        title="Potential Command Injection",
                        description="Dynamic command execution detected. This may allow arbitrary command execution.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Validate and sanitize all input. Use subprocess with shell=False and a list of arguments.",
                        references=["https://owasp.org/Top10/A03_2021-Injection/"],
                        cwe_id="CWE-78",
                        confidence=0.9
                    ))
        
        return vulns

    def _check_xss_vulns(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for XSS vulnerabilities"""
        vulns = []
        
        for i, line in enumerate(lines):
            for pattern in self.xss_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"xss_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.XSS,
                        severity=SeverityLevel.MEDIUM,
                        title="Potential Cross-Site Scripting (XSS)",
                        description="Unsafe DOM manipulation detected. This may be vulnerable to XSS attacks.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Sanitize and validate all user input. Use textContent instead of innerHTML where possible.",
                        references=["https://owasp.org/Top10/A03_2021-Injection/"],
                        cwe_id="CWE-79",
                        confidence=0.7
                    ))
        
        return vulns

    def _check_broken_auth(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for broken authentication issues"""
        vulns = []
        
        # Check for weak session management
        weak_session_patterns = [
            r'(?i)session_id\s*=\s*["\'].*["\']',  # Hardcoded session ID
            r'(?i)session\.permanent\s*=\s*True',  # Permanent sessions
            r'(?i)remember_me\s*=\s*True',  # Dangerous remember me
        ]
        
        for i, line in enumerate(lines):
            for pattern in weak_session_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"auth_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.BROKEN_AUTH,
                        severity=SeverityLevel.MEDIUM,
                        title="Potential Broken Authentication",
                        description="Weak session management or authentication mechanism detected.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Implement secure session management with proper timeout, secure cookies, and session invalidation.",
                        references=["https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"],
                        cwe_id="CWE-287",
                        confidence=0.6
                    ))
        
        return vulns

    def _check_sensitive_data(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for sensitive data exposure"""
        vulns = []
        
        # Check for logging sensitive data
        sensitive_logging_patterns = [
            r'(?i)log.*password',
            r'(?i)log.*secret',
            r'(?i)log.*token',
            r'(?i)print.*password',
            r'(?i)console\.log.*password',
        ]
        
        for i, line in enumerate(lines):
            for pattern in sensitive_logging_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"data_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.SENSITIVE_DATA,
                        severity=SeverityLevel.MEDIUM,
                        title="Potential Sensitive Data Exposure",
                        description="Sensitive information may be logged or exposed.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Remove sensitive data from logs. Use proper data classification and handling.",
                        references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"],
                        cwe_id="CWE-200",
                        confidence=0.8
                    ))
        
        return vulns

    def _check_security_misconfig(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for security misconfigurations"""
        vulns = []
        
        # Check for debug mode in production
        debug_patterns = [
            r'(?i)debug\s*=\s*true',
            r'(?i)development\s*=\s*true',
            r'(?i)app\.run\s*\(\s*debug\s*=\s*true',
        ]
        
        for i, line in enumerate(lines):
            for pattern in debug_patterns:
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"misc_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.SECURITY_MISCONFIG,
                        severity=SeverityLevel.MEDIUM,
                        title="Security Misconfiguration",
                        description="Debug mode enabled. This may expose sensitive information in production.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Disable debug mode in production environments.",
                        references=["https://owasp.org/Top10/A05_2021-Security_Misconfiguration/"],
                        cwe_id="CWE-489",
                        confidence=0.9
                    ))
        
        return vulns

    def _check_crypto_failures(self, content: str, lines: List[str], file_path: str) -> List[SecurityVulnerability]:
        """Check for cryptographic failures"""
        vulns = []
        
        for i, line in enumerate(lines):
            for pattern in self.crypto_patterns:
                if re.search(pattern, line):
                    severity = SeverityLevel.HIGH if 'md5' in pattern.lower() or 'sha1' in pattern.lower() else SeverityLevel.MEDIUM
                    
                    vulns.append(SecurityVulnerability(
                        vuln_id=f"crypto_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        vuln_type=VulnerabilityType.CRYPTO_FAILURES,
                        severity=severity,
                        title="Cryptographic Failure",
                        description="Weak cryptographic algorithm or implementation detected.",
                        file_path=file_path,
                        line_number=i + 1,
                        code_snippet=line.strip(),
                        remediation="Use strong cryptographic algorithms (SHA-256+, AES, etc.) and proper key management.",
                        references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"],
                        cwe_id="CWE-327",
                        confidence=0.8
                    ))
        
        return vulns

    def detect_secrets(self, code_base: str) -> List[SecurityVulnerability]:
        """
        Detect hardcoded secrets in codebase
        
        Args:
            code_base: Path to code directory
            
        Returns:
            List of secret exposure vulnerabilities
        """
        vulnerabilities = []
        
        logger.info("Scanning for hardcoded secrets")
        
        for root, dirs, files in os.walk(code_base):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if self._is_source_file(file) or file.endswith(('.env', '.config', '.ini', '.yaml', '.yml', '.json')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, code_base)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')
                        
                        for i, line in enumerate(lines):
                            for pattern in self.secrets_patterns:
                                matches = re.finditer(pattern, line)
                                for match in matches:
                                    vulnerabilities.append(SecurityVulnerability(
                                        vuln_id=f"secret_{hashlib.md5(f'{file_path}:{i}:{match.group()}'.encode()).hexdigest()[:8]}",
                                        vuln_type=VulnerabilityType.SECRETS_EXPOSURE,
                                        severity=SeverityLevel.HIGH,
                                        title="Hardcoded Secret Detected",
                                        description="Potential hardcoded secret or API key found in source code.",
                                        file_path=rel_path,
                                        line_number=i + 1,
                                        code_snippet=line.strip(),
                                        remediation="Move secrets to environment variables or secure secret management system.",
                                        references=["https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"],
                                        cwe_id="CWE-798",
                                        confidence=0.9
                                    ))
                    
                    except Exception as e:
                        logger.warning(f"Error scanning {file_path} for secrets: {e}")
        
        return vulnerabilities

    def check_dependencies(self, code_base: str) -> List[SecurityVulnerability]:
        """
        Check for vulnerable dependencies
        
        Args:
            code_base: Path to code directory
            
        Returns:
            List of dependency vulnerabilities
        """
        vulnerabilities = []
        
        logger.info("Checking for vulnerable dependencies")
        
        # Check Python requirements
        req_files = ['requirements.txt', 'requirements-dev.txt', 'Pipfile', 'pyproject.toml']
        for req_file in req_files:
            req_path = os.path.join(code_base, req_file)
            if os.path.exists(req_path):
                vulnerabilities.extend(self._check_python_dependencies(req_path, req_file))
        
        # Check Node.js package.json
        package_json_path = os.path.join(code_base, 'package.json')
        if os.path.exists(package_json_path):
            vulnerabilities.extend(self._check_nodejs_dependencies(package_json_path))
        
        return vulnerabilities

    def _check_python_dependencies(self, req_file: str, filename: str) -> List[SecurityVulnerability]:
        """Check Python dependencies for known vulnerabilities"""
        vulns = []
        
        # Known vulnerable packages (simplified - in real implementation would use vulnerability database)
        known_vulns = {
            'flask': {'<1.1.0': 'CVE-2018-1000656'},
            'django': {'<2.2.10': 'CVE-2020-7471'},
            'requests': {'<2.20.0': 'CVE-2018-18074'},
            'urllib3': {'<1.24.2': 'CVE-2019-11324'},
        }
        
        try:
            with open(req_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse package name and version
                    if '==' in line:
                        package, version = line.split('==', 1)
                        package = package.strip()
                        version = version.strip()
                        
                        if package.lower() in known_vulns:
                            vuln_info = known_vulns[package.lower()]
                            for vuln_version, cve in vuln_info.items():
                                if self._version_is_vulnerable(version, vuln_version):
                                    vulns.append(SecurityVulnerability(
                                        vuln_id=f"dep_{hashlib.md5(f'{req_file}:{package}:{version}'.encode()).hexdigest()[:8]}",
                                        vuln_type=VulnerabilityType.VULNERABLE_COMPONENTS,
                                        severity=SeverityLevel.HIGH,
                                        title=f"Vulnerable Dependency: {package}",
                                        description=f"Package {package} version {version} has known security vulnerabilities ({cve}).",
                                        file_path=filename,
                                        line_number=i + 1,
                                        code_snippet=line,
                                        remediation=f"Update {package} to a secure version.",
                                        references=[f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve}"],
                                        confidence=0.9
                                    ))
        
        except Exception as e:
            logger.warning(f"Error checking Python dependencies in {req_file}: {e}")
        
        return vulns

    def _check_nodejs_dependencies(self, package_json_path: str) -> List[SecurityVulnerability]:
        """Check Node.js dependencies for known vulnerabilities"""
        vulns = []
        
        # Known vulnerable packages (simplified)
        known_vulns = {
            'express': {'<4.17.1': 'CVE-2019-5413'},
            'lodash': {'<4.17.19': 'CVE-2020-8203'},
            'axios': {'<0.21.1': 'CVE-2020-28168'},
        }
        
        try:
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            dependencies = {}
            dependencies.update(package_data.get('dependencies', {}))
            dependencies.update(package_data.get('devDependencies', {}))
            
            for package, version_spec in dependencies.items():
                # Simple version parsing (in real implementation would use semver)
                version = version_spec.lstrip('^~>=<')
                
                if package in known_vulns:
                    vuln_info = known_vulns[package]
                    for vuln_version, cve in vuln_info.items():
                        if self._version_is_vulnerable(version, vuln_version):
                            vulns.append(SecurityVulnerability(
                                vuln_id=f"dep_{hashlib.md5(f'{package_json_path}:{package}:{version}'.encode()).hexdigest()[:8]}",
                                vuln_type=VulnerabilityType.VULNERABLE_COMPONENTS,
                                severity=SeverityLevel.HIGH,
                                title=f"Vulnerable Dependency: {package}",
                                description=f"Package {package} version {version_spec} has known security vulnerabilities ({cve}).",
                                file_path="package.json",
                                line_number=0,
                                code_snippet=f'"{package}": "{version_spec}"',
                                remediation=f"Update {package} to a secure version.",
                                references=[f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve}"],
                                confidence=0.9
                            ))
        
        except Exception as e:
            logger.warning(f"Error checking Node.js dependencies: {e}")
        
        return vulns

    def check_auth(self, code_base: str) -> List[SecurityVulnerability]:
        """
        Check authentication and authorization mechanisms
        
        Args:
            code_base: Path to code directory
            
        Returns:
            List of auth-related vulnerabilities
        """
        vulnerabilities = []
        
        logger.info("Checking authentication and authorization")
        
        # Patterns for auth issues
        auth_patterns = {
            'weak_password': [
                r'(?i)password\s*=\s*["\'][^"\']{1,7}["\']',  # Short passwords
                r'(?i)password\s*=\s*["\']password["\']',     # Default passwords
                r'(?i)password\s*=\s*["\']123456["\']',       # Common weak passwords
            ],
            'missing_auth': [
                r'@app\.route\([^)]*\)\s*\n\s*def\s+\w+',     # Flask routes without auth
                r'def\s+\w+.*:\s*\n(?!.*@login_required)(?!.*@auth)',  # Functions without auth decorators
            ],
            'jwt_issues': [
                r'jwt\.decode\([^,]*,\s*verify=False',        # JWT without verification
                r'jwt\.decode\([^,]*,\s*options=.*verify.*False',  # JWT verification disabled
            ]
        }
        
        for root, dirs, files in os.walk(code_base):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if self._is_source_file(file):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, code_base)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')
                        
                        for category, patterns in auth_patterns.items():
                            for i, line in enumerate(lines):
                                for pattern in patterns:
                                    if re.search(pattern, line):
                                        vulnerabilities.append(self._create_auth_vulnerability(
                                            category, file_path, i + 1, line.strip(), rel_path
                                        ))
                    
                    except Exception as e:
                        logger.warning(f"Error checking auth in {file_path}: {e}")
        
        return vulnerabilities

    def _create_auth_vulnerability(self, category: str, file_path: str, line_num: int, code: str, rel_path: str) -> SecurityVulnerability:
        """Create auth vulnerability based on category"""
        vuln_map = {
            'weak_password': {
                'title': "Weak Password Policy",
                'severity': SeverityLevel.MEDIUM,
                'description': "Weak or default password detected.",
                'remediation': "Implement strong password policy and use secure defaults.",
                'cwe': "CWE-521"
            },
            'missing_auth': {
                'title': "Missing Authentication",
                'severity': SeverityLevel.HIGH,
                'description': "Endpoint or function missing authentication check.",
                'remediation': "Add proper authentication and authorization checks.",
                'cwe': "CWE-306"
            },
            'jwt_issues': {
                'title': "JWT Security Issue",
                'severity': SeverityLevel.HIGH,
                'description': "JWT verification disabled or improperly configured.",
                'remediation': "Enable JWT signature verification and use secure algorithms.",
                'cwe': "CWE-347"
            }
        }
        
        vuln_info = vuln_map.get(category, vuln_map['missing_auth'])
        
        return SecurityVulnerability(
            vuln_id=f"auth_{hashlib.md5(f'{file_path}:{line_num}:{code}'.encode()).hexdigest()[:8]}",
            vuln_type=VulnerabilityType.BROKEN_AUTH,
            severity=vuln_info['severity'],
            title=vuln_info['title'],
            description=vuln_info['description'],
            file_path=rel_path,
            line_number=line_num,
            code_snippet=code,
            remediation=vuln_info['remediation'],
            references=["https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"],
            cwe_id=vuln_info['cwe'],
            confidence=0.7
        )

    def prioritize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> List[SecurityVulnerability]:
        """
        Prioritize vulnerabilities by severity and confidence
        
        Args:
            vulnerabilities: List of vulnerabilities to prioritize
            
        Returns:
            Sorted list of vulnerabilities (highest priority first)
        """
        def priority_score(vuln):
            # Higher severity and confidence = higher priority
            severity_weight = vuln.severity.value * 20
            confidence_weight = vuln.confidence * 10
            return severity_weight + confidence_weight
        
        return sorted(vulnerabilities, key=priority_score, reverse=True)

    def compliance_check(self, code_base: str, standards: List[ComplianceStandard]) -> Dict[str, Any]:
        """
        Check compliance against security standards
        
        Args:
            code_base: Path to code directory  
            standards: List of compliance standards to check
            
        Returns:
            Dict containing compliance results
        """
        results = {}
        
        for standard in standards:
            logger.info(f"Checking compliance against {standard.value}")
            
            if standard == ComplianceStandard.OWASP_TOP_10:
                result = self._check_owasp_compliance(code_base)
            elif standard == ComplianceStandard.PCI_DSS:
                result = self._check_pci_compliance(code_base)
            else:
                # Placeholder for other standards
                result = ComplianceResult(
                    standard=standard,
                    overall_score=0.5,
                    passed_checks=0,
                    failed_checks=1,
                    total_checks=1,
                    findings=[{"message": "Standard not fully implemented"}],
                    recommendations=["Implement full compliance checking for this standard"]
                )
            
            results[standard.value] = {
                "overall_score": result.overall_score,
                "passed_checks": result.passed_checks,
                "failed_checks": result.failed_checks,
                "total_checks": result.total_checks,
                "findings": result.findings,
                "recommendations": result.recommendations
            }
        
        return results

    def _check_owasp_compliance(self, code_base: str) -> ComplianceResult:
        """Check OWASP Top 10 compliance"""
        # Run OWASP checks
        vulns = self.check_owasp_top10(code_base)
        
        # Count issues by OWASP category
        owasp_categories = {
            VulnerabilityType.INJECTION: "A03:2021 – Injection",
            VulnerabilityType.BROKEN_AUTH: "A07:2021 – Identification and Authentication Failures",
            VulnerabilityType.SENSITIVE_DATA: "A02:2021 – Cryptographic Failures",
            VulnerabilityType.XSS: "A03:2021 – Injection",
            VulnerabilityType.BROKEN_ACCESS: "A01:2021 – Broken Access Control",
            VulnerabilityType.SECURITY_MISCONFIG: "A05:2021 – Security Misconfiguration",
            VulnerabilityType.CRYPTO_FAILURES: "A02:2021 – Cryptographic Failures",
        }
        
        findings = []
        category_counts = {}
        
        for vuln in vulns:
            category = owasp_categories.get(vuln.vuln_type, "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            findings.append({
                "category": category,
                "severity": vuln.severity.name,
                "file": vuln.file_path,
                "line": vuln.line_number,
                "description": vuln.description
            })
        
        total_checks = len(owasp_categories)
        failed_checks = len([cat for cat in category_counts.values() if cat > 0])
        passed_checks = total_checks - failed_checks
        
        overall_score = passed_checks / total_checks if total_checks > 0 else 0
        
        recommendations = []
        if failed_checks > 0:
            recommendations.append("Address identified OWASP Top 10 vulnerabilities")
            recommendations.append("Implement secure coding training for development team")
            recommendations.append("Add security testing to CI/CD pipeline")
        
        return ComplianceResult(
            standard=ComplianceStandard.OWASP_TOP_10,
            overall_score=overall_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            total_checks=total_checks,
            findings=findings,
            recommendations=recommendations
        )

    def _check_pci_compliance(self, code_base: str) -> ComplianceResult:
        """Check PCI DSS compliance (simplified)"""
        findings = []
        
        # Check for PCI-specific issues
        pci_checks = [
            ("Strong cryptography", self._has_strong_crypto(code_base)),
            ("No default passwords", self._no_default_passwords(code_base)),
            ("Access controls", self._has_access_controls(code_base)),
            ("Secure transmission", self._secure_transmission(code_base)),
        ]
        
        passed_checks = sum(1 for _, passed in pci_checks if passed)
        failed_checks = len(pci_checks) - passed_checks
        
        for check_name, passed in pci_checks:
            if not passed:
                findings.append({
                    "requirement": check_name,
                    "status": "failed",
                    "description": f"PCI DSS requirement '{check_name}' not met"
                })
        
        overall_score = passed_checks / len(pci_checks)
        
        recommendations = []
        if failed_checks > 0:
            recommendations.append("Implement missing PCI DSS controls")
            recommendations.append("Conduct thorough PCI DSS assessment")
            recommendations.append("Engage qualified security assessor")
        
        return ComplianceResult(
            standard=ComplianceStandard.PCI_DSS,
            overall_score=overall_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            total_checks=len(pci_checks),
            findings=findings,
            recommendations=recommendations
        )

    def _has_strong_crypto(self, code_base: str) -> bool:
        """Check if strong cryptography is used"""
        # Simplified check - look for weak crypto usage
        weak_crypto = self.check_owasp_top10(code_base)
        crypto_vulns = [v for v in weak_crypto if v.vuln_type == VulnerabilityType.CRYPTO_FAILURES]
        return len(crypto_vulns) == 0

    def _no_default_passwords(self, code_base: str) -> bool:
        """Check for absence of default passwords"""
        secrets = self.detect_secrets(code_base)
        default_password_vulns = [v for v in secrets if 'default' in v.description.lower() or 'password' in v.code_snippet.lower()]
        return len(default_password_vulns) == 0

    def _has_access_controls(self, code_base: str) -> bool:
        """Check if access controls are implemented"""
        auth_vulns = self.check_auth(code_base)
        missing_auth_vulns = [v for v in auth_vulns if 'missing' in v.title.lower()]
        return len(missing_auth_vulns) == 0

    def _secure_transmission(self, code_base: str) -> bool:
        """Check if secure transmission is enforced"""
        # Look for HTTP usage instead of HTTPS
        insecure_transmission_patterns = [
            r'http://(?!localhost|127\.0\.0\.1)',
            r'ssl_verify\s*=\s*False',
            r'verify=False'
        ]
        
        for root, dirs, files in os.walk(code_base):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if self._is_source_file(file):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for pattern in insecure_transmission_patterns:
                                if re.search(pattern, content):
                                    return False
                    except Exception:
                        continue
        return True

    def _is_source_file(self, filename: str) -> bool:
        """Check if file is a source code file"""
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', 
                           '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala'}
        return any(filename.endswith(ext) for ext in source_extensions)

    def _version_is_vulnerable(self, version: str, vuln_version_spec: str) -> bool:
        """Simple version comparison (in real implementation would use proper semver)"""
        # Very simplified version comparison
        try:
            if vuln_version_spec.startswith('<'):
                threshold = vuln_version_spec[1:]
                # Simple string comparison (would need proper semver in production)
                return version < threshold
        except:
            pass
        return False

    def _calculate_confidence(self, scan_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for scan results"""
        if not scan_result.get('vulnerabilities'):
            return 0.9  # High confidence in clean scan
        
        vulns = scan_result['vulnerabilities']
        if isinstance(vulns[0], dict):
            # Already converted to dict
            confidences = [v.get('confidence', 0.5) for v in vulns]
        else:
            # Still vulnerability objects
            confidences = [v.confidence for v in vulns]
        
        return sum(confidences) / len(confidences) if confidences else 0.5

    def _generate_artifacts(self, scan_result: Dict[str, Any]) -> List[str]:
        """Generate artifact filenames for scan results"""
        artifacts = []
        
        if scan_result.get('scan_id'):
            scan_id = scan_result['scan_id']
            artifacts.extend([
                f"security_report_{scan_id}.json",
                f"vulnerability_summary_{scan_id}.md",
                f"remediation_plan_{scan_id}.md"
            ])
        
        return artifacts

    def _generate_security_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations based on found vulnerabilities"""
        recommendations = []
        
        # Count vulnerabilities by type
        vuln_counts = {}
        for vuln in vulnerabilities:
            vuln_counts[vuln.vuln_type] = vuln_counts.get(vuln.vuln_type, 0) + 1
        
        # Generate recommendations based on vulnerability types found
        if VulnerabilityType.INJECTION in vuln_counts:
            recommendations.append("Implement parameterized queries to prevent SQL injection")
            recommendations.append("Add input validation and sanitization")
        
        if VulnerabilityType.XSS in vuln_counts:
            recommendations.append("Implement Content Security Policy (CSP)")
            recommendations.append("Use secure DOM manipulation methods")
        
        if VulnerabilityType.SECRETS_EXPOSURE in vuln_counts:
            recommendations.append("Move secrets to environment variables or secure vault")
            recommendations.append("Implement secret scanning in CI/CD pipeline")
        
        if VulnerabilityType.CRYPTO_FAILURES in vuln_counts:
            recommendations.append("Upgrade to strong cryptographic algorithms (SHA-256+, AES)")
            recommendations.append("Implement proper key management")
        
        if VulnerabilityType.VULNERABLE_COMPONENTS in vuln_counts:
            recommendations.append("Update vulnerable dependencies to secure versions")
            recommendations.append("Implement dependency scanning automation")
        
        if VulnerabilityType.BROKEN_AUTH in vuln_counts:
            recommendations.append("Implement multi-factor authentication")
            recommendations.append("Review session management and timeout policies")
        
        # General recommendations
        if vulnerabilities:
            recommendations.extend([
                "Conduct regular security code reviews",
                "Implement security testing in CI/CD pipeline", 
                "Provide secure coding training for developers"
            ])
        
        return recommendations

    def _vuln_to_dict(self, vuln: SecurityVulnerability) -> Dict[str, Any]:
        """Convert vulnerability object to dictionary"""
        return {
            "vuln_id": vuln.vuln_id,
            "vuln_type": vuln.vuln_type.value,
            "severity": vuln.severity.name,
            "title": vuln.title,
            "description": vuln.description,
            "file_path": vuln.file_path,
            "line_number": vuln.line_number,
            "code_snippet": vuln.code_snippet,
            "remediation": vuln.remediation,
            "references": vuln.references,
            "cwe_id": vuln.cwe_id,
            "confidence": vuln.confidence
        }


# Example usage and testing functions
if __name__ == "__main__":
    import sys
    
    def demo_security_scan():
        """Demonstrate security agent capabilities"""
        agent = SecuritySpecialistAgent()
        
        # Example comprehensive scan
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': '.' if len(sys.argv) < 2 else sys.argv[1]
        }
        
        print(f"Starting security scan of {task_data['target_path']}...")
        result = agent.execute_task("demo_scan", task_data)
        
        print(f"\nScan Status: {result.status.value}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")
        
        if result.result_data:
            scan_data = result.result_data
            print(f"\nResults:")
            print(f"- Files scanned: {scan_data.get('files_scanned', 0)}")
            print(f"- Lines scanned: {scan_data.get('lines_scanned', 0)}")
            print(f"- Vulnerabilities found: {len(scan_data.get('vulnerabilities', []))}")
            
            # Show vulnerability summary
            summary = scan_data.get('summary', {})
            if summary:
                print(f"\nVulnerability Summary:")
                for severity, count in summary.items():
                    if count > 0:
                        print(f"- {severity.upper()}: {count}")
            
            # Show top 5 vulnerabilities
            vulnerabilities = scan_data.get('vulnerabilities', [])
            if vulnerabilities:
                print(f"\nTop Vulnerabilities:")
                for i, vuln in enumerate(vulnerabilities[:5], 1):
                    print(f"{i}. {vuln['title']} ({vuln['severity']})")
                    print(f"   File: {vuln['file_path']}:{vuln['line_number']}")
                    print(f"   {vuln['description'][:100]}...")
            
            # Show recommendations
            recommendations = scan_data.get('recommendations', [])
            if recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"{i}. {rec}")
        
        print(f"\nAgent Status: {agent.get_status()}")
    
    if __name__ == "__main__":
        demo_security_scan()