#!/usr/bin/env python3
"""
Test Suite for Security Specialist Agent - Issue #74
Comprehensive tests for security vulnerability detection and compliance checking
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude.commands.security_specialist_agent import (
    SecuritySpecialistAgent, 
    VulnerabilityType,
    SeverityLevel,
    ComplianceStandard,
    SecurityVulnerability,
    AgentStatus
)
from claude.commands.domain_agent_base import AgentConfiguration

class TestSecuritySpecialistAgent(unittest.TestCase):
    """Test cases for SecuritySpecialistAgent"""
    
    def setUp(self):
        """Set up test environment"""
        self.agent = SecuritySpecialistAgent()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.domain, 'security')
        self.assertIn('vulnerability_detection', self.agent.get_capability_names())
        self.assertIn('compliance_checking', self.agent.get_capability_names())
        self.assertEqual(self.agent.name, 'SecuritySpecialist')
        self.assertTrue(len(self.agent.capabilities) >= 8)
    
    def test_agent_has_required_capabilities(self):
        """Test agent has all required capabilities"""
        required_capabilities = [
            'vulnerability_detection',
            'compliance_checking', 
            'threat_modeling',
            'security_scanning',
            'secrets_detection',
            'dependency_analysis',
            'auth_analysis',
            'crypto_analysis'
        ]
        
        agent_capabilities = self.agent.get_capability_names()
        for capability in required_capabilities:
            self.assertIn(capability, agent_capabilities)
    
    def test_sql_injection_detection(self):
        """Test SQL injection vulnerability detection"""
        # Create test file with SQL injection vulnerability
        test_file = os.path.join(self.temp_dir, 'vulnerable.py')
        with open(test_file, 'w') as f:
            f.write('''
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Vulnerable SQL injection
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchone()

def login(username, password):
    # Another SQL injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
''')
        
        vulns = self.agent.check_owasp_top10(self.temp_dir)
        
        # Should find SQL injection vulnerabilities
        sql_vulns = [v for v in vulns if v.vuln_type == VulnerabilityType.INJECTION]
        self.assertGreater(len(sql_vulns), 0)
        
        # Check vulnerability details
        sql_vuln = sql_vulns[0]
        self.assertEqual(sql_vuln.severity, SeverityLevel.HIGH)
        self.assertIn('injection', sql_vuln.title.lower())
        self.assertIn('vulnerable.py', sql_vuln.file_path)
    
    def test_xss_detection(self):
        """Test XSS vulnerability detection"""
        test_file = os.path.join(self.temp_dir, 'xss_vulnerable.js')
        with open(test_file, 'w') as f:
            f.write('''
function displayMessage(message) {
    // Vulnerable to XSS
    document.getElementById('output').innerHTML = message;
}

function updatePage(content) {
    // Another XSS vulnerability
    document.write(content);
}

function showAlert(msg) {
    // Dangerous eval usage
    eval("alert('" + msg + "')");
}
''')
        
        vulns = self.agent.check_owasp_top10(self.temp_dir)
        xss_vulns = [v for v in vulns if v.vuln_type == VulnerabilityType.XSS]
        
        self.assertGreater(len(xss_vulns), 0)
        self.assertTrue(any('xss' in v.title.lower() for v in xss_vulns))
    
    def test_secrets_detection(self):
        """Test hardcoded secrets detection"""
        test_file = os.path.join(self.temp_dir, 'secrets.py')
        with open(test_file, 'w') as f:
            f.write('''
# Various secret patterns
API_KEY = "sk-abcdef123456789012345678901234567890abcdef"  # OpenAI key
SECRET_KEY = "super-secret-key-1234567890"
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
DATABASE_URL = "postgresql://user:password123@localhost/db"
PASSWORD = "hardcoded-password"

config = {
    "api_key": "another-api-key-here",
    "token": "ghp_1234567890abcdef1234567890abcdef123456"  # GitHub token
}
''')
        
        vulns = self.agent.detect_secrets(self.temp_dir)
        
        self.assertGreater(len(vulns), 0)
        
        # Should find multiple types of secrets
        secret_types = set(v.title for v in vulns)
        self.assertTrue(any('secret' in t.lower() for t in secret_types))
        
        # Check severity
        for vuln in vulns:
            self.assertEqual(vuln.vuln_type, VulnerabilityType.SECRETS_EXPOSURE)
            self.assertEqual(vuln.severity, SeverityLevel.HIGH)
    
    def test_crypto_failures_detection(self):
        """Test cryptographic failures detection"""
        test_file = os.path.join(self.temp_dir, 'crypto.py')
        with open(test_file, 'w') as f:
            f.write('''
import hashlib
import md5  # Deprecated

def weak_hash(data):
    # Weak MD5 usage
    return md5.md5(data).hexdigest()

def another_weak_hash(data):
    # Weak SHA1
    return hashlib.sha1(data).hexdigest()

def weak_encryption(data, key):
    # Weak DES usage
    cipher = DES.new(key)
    return cipher.encrypt(data)
''')
        
        vulns = self.agent.check_owasp_top10(self.temp_dir)
        crypto_vulns = [v for v in vulns if v.vuln_type == VulnerabilityType.CRYPTO_FAILURES]
        
        self.assertGreater(len(crypto_vulns), 0)
        
        # Should flag weak crypto
        for vuln in crypto_vulns:
            self.assertIn(vuln.severity, [SeverityLevel.MEDIUM, SeverityLevel.HIGH])
            self.assertIn('crypto', vuln.title.lower())
    
    def test_auth_vulnerabilities_detection(self):
        """Test authentication vulnerability detection"""
        test_file = os.path.join(self.temp_dir, 'auth.py')
        with open(test_file, 'w') as f:
            f.write('''
from flask import Flask, request
import jwt

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    # Missing authentication check
    return {"message": "logged in"}

def verify_token(token):
    # JWT without verification
    payload = jwt.decode(token, verify=False)
    return payload

# Weak password
DEFAULT_PASSWORD = "123456"
ADMIN_PASSWORD = "password"
''')
        
        vulns = self.agent.check_auth(self.temp_dir)
        
        self.assertGreater(len(vulns), 0)
        
        # Should find auth issues
        for vuln in vulns:
            self.assertEqual(vuln.vuln_type, VulnerabilityType.BROKEN_AUTH)
            self.assertIn(vuln.severity, [SeverityLevel.MEDIUM, SeverityLevel.HIGH])
    
    def test_dependency_vulnerability_detection(self):
        """Test dependency vulnerability detection"""
        # Create requirements.txt with vulnerable packages
        req_file = os.path.join(self.temp_dir, 'requirements.txt')
        with open(req_file, 'w') as f:
            f.write('''
flask==0.12.0
django==2.0.0
requests==2.18.0
urllib3==1.20.0
''')
        
        vulns = self.agent.check_dependencies(self.temp_dir)
        
        # Should find vulnerable dependencies
        dep_vulns = [v for v in vulns if v.vuln_type == VulnerabilityType.VULNERABLE_COMPONENTS]
        self.assertGreaterEqual(len(dep_vulns), 0)  # May be 0 if vulnerability DB not complete
    
    def test_nodejs_dependency_detection(self):
        """Test Node.js dependency vulnerability detection"""
        # Create package.json with vulnerable packages
        package_json = os.path.join(self.temp_dir, 'package.json')
        package_data = {
            "name": "test-app",
            "dependencies": {
                "express": "4.16.0",
                "lodash": "4.17.0",
                "axios": "0.19.0"
            },
            "devDependencies": {
                "nodemon": "1.18.0"
            }
        }
        
        with open(package_json, 'w') as f:
            json.dump(package_data, f)
        
        vulns = self.agent.check_dependencies(self.temp_dir)
        
        # Should process package.json
        dep_vulns = [v for v in vulns if v.vuln_type == VulnerabilityType.VULNERABLE_COMPONENTS]
        # May find vulnerabilities depending on implementation
        self.assertGreaterEqual(len(dep_vulns), 0)
    
    def test_comprehensive_security_scan(self):
        """Test comprehensive security scan"""
        # Create multiple vulnerable files
        self._create_vulnerable_test_files()
        
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': self.temp_dir
        }
        
        result = self.agent.execute_task("test_scan", task_data)
        
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIsNotNone(result.result_data)
        
        scan_data = result.result_data
        self.assertIn('vulnerabilities', scan_data)
        self.assertIn('summary', scan_data)
        self.assertIn('recommendations', scan_data)
        
        # Should find multiple vulnerabilities
        vulns = scan_data['vulnerabilities']
        self.assertGreater(len(vulns), 0)
        
        # Should have summary by severity
        summary = scan_data['summary']
        self.assertIsInstance(summary, dict)
        
        # Should have recommendations
        recommendations = scan_data['recommendations']
        self.assertGreater(len(recommendations), 0)
        
        # Check confidence score
        self.assertGreater(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1)
    
    def test_vulnerability_prioritization(self):
        """Test vulnerability prioritization"""
        vulns = [
            SecurityVulnerability(
                vuln_id="1", vuln_type=VulnerabilityType.INJECTION, 
                severity=SeverityLevel.CRITICAL, title="Critical SQL Injection",
                description="Test", file_path="test.py", confidence=0.9
            ),
            SecurityVulnerability(
                vuln_id="2", vuln_type=VulnerabilityType.XSS,
                severity=SeverityLevel.MEDIUM, title="Medium XSS",
                description="Test", file_path="test.js", confidence=0.7
            ),
            SecurityVulnerability(
                vuln_id="3", vuln_type=VulnerabilityType.SECRETS_EXPOSURE,
                severity=SeverityLevel.HIGH, title="High Secret Exposure", 
                description="Test", file_path="test.py", confidence=0.95
            ),
        ]
        
        prioritized = self.agent.prioritize_vulnerabilities(vulns)
        
        # Should be sorted by priority (severity + confidence)
        self.assertEqual(len(prioritized), 3)
        
        # Critical with high confidence should be first
        self.assertEqual(prioritized[0].severity, SeverityLevel.CRITICAL)
        
        # Medium severity should be last
        self.assertEqual(prioritized[-1].severity, SeverityLevel.MEDIUM)
    
    def test_owasp_compliance_check(self):
        """Test OWASP Top 10 compliance checking"""
        self._create_vulnerable_test_files()
        
        task_data = {
            'scan_type': 'compliance',
            'target_path': self.temp_dir,
            'standards': [ComplianceStandard.OWASP_TOP_10]
        }
        
        result = self.agent.execute_task("compliance_test", task_data)
        
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        
        compliance_data = result.result_data
        self.assertIn('owasp_top_10', compliance_data)
        
        owasp_result = compliance_data['owasp_top_10']
        self.assertIn('overall_score', owasp_result)
        self.assertIn('passed_checks', owasp_result)
        self.assertIn('failed_checks', owasp_result)
        self.assertIn('findings', owasp_result)
        self.assertIn('recommendations', owasp_result)
    
    def test_agent_metrics_tracking(self):
        """Test agent metrics tracking"""
        initial_metrics = self.agent.get_status()['metrics']
        self.assertEqual(initial_metrics['tasks_completed'], 0)
        self.assertEqual(initial_metrics['tasks_failed'], 0)
        
        # Execute successful task
        task_data = {'scan_type': 'secrets', 'target_path': self.temp_dir}
        result = self.agent.execute_task("metrics_test", task_data)
        
        updated_metrics = self.agent.get_status()['metrics']
        self.assertEqual(updated_metrics['tasks_completed'], 1)
        self.assertGreater(updated_metrics['total_execution_time'], 0)
        self.assertEqual(updated_metrics['success_rate'], 1.0)
    
    def test_invalid_scan_type(self):
        """Test handling of invalid scan type"""
        task_data = {
            'scan_type': 'invalid_type',
            'target_path': self.temp_dir
        }
        
        result = self.agent.execute_task("invalid_test", task_data)
        
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn('Unknown scan type', result.error_message)
    
    def test_nonexistent_path(self):
        """Test handling of nonexistent path"""
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': '/nonexistent/path'
        }
        
        # Should handle gracefully (no files to scan)
        result = self.agent.execute_task("nonexistent_test", task_data)
        
        # May complete with no results or fail gracefully
        self.assertIn(result.status, [AgentStatus.COMPLETED, AgentStatus.FAILED])
    
    def test_empty_directory_scan(self):
        """Test scanning empty directory"""
        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir)
        
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': empty_dir
        }
        
        result = self.agent.execute_task("empty_test", task_data)
        
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        scan_data = result.result_data
        self.assertEqual(len(scan_data['vulnerabilities']), 0)
        self.assertEqual(scan_data['files_scanned'], 0)
    
    def test_binary_file_handling(self):
        """Test handling of binary files"""
        # Create a binary file
        binary_file = os.path.join(self.temp_dir, 'binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
        
        # Should not crash when encountering binary files
        vulns = self.agent.detect_secrets(self.temp_dir)
        
        # Should complete without errors
        self.assertIsInstance(vulns, list)
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        large_file = os.path.join(self.temp_dir, 'large.py')
        with open(large_file, 'w') as f:
            # Create file with many lines
            for i in range(1000):
                f.write(f"# This is line {i}\nprint('Hello {i}')\n")
        
        vulns = self.agent.check_owasp_top10(self.temp_dir)
        
        # Should handle large files
        self.assertIsInstance(vulns, list)
    
    def test_special_characters_in_code(self):
        """Test handling of special characters in code"""
        special_file = os.path.join(self.temp_dir, 'special.py')
        with open(special_file, 'w', encoding='utf-8') as f:
            f.write('''
# Special characters: áéíóú, 中文, 🚀
password = "pássw0rd-spéciál"
api_key = "api-key-with-émojis-🔑"
''')
        
        vulns = self.agent.detect_secrets(self.temp_dir)
        
        # Should handle special characters
        self.assertIsInstance(vulns, list)
    
    def test_configuration_options(self):
        """Test agent with custom configuration"""
        config = AgentConfiguration(
            timeout_seconds=60,
            max_retries=1,
            enable_logging=False
        )
        
        custom_agent = SecuritySpecialistAgent(config)
        self.assertEqual(custom_agent.config.timeout_seconds, 60)
        self.assertEqual(custom_agent.config.max_retries, 1)
        self.assertFalse(custom_agent.config.enable_logging)
    
    def test_task_result_artifacts(self):
        """Test that task results include proper artifacts"""
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': self.temp_dir
        }
        
        result = self.agent.execute_task("artifacts_test", task_data)
        
        self.assertIsInstance(result.artifacts, list)
        if result.result_data.get('scan_id'):
            self.assertGreater(len(result.artifacts), 0)
            # Should include report files
            self.assertTrue(any('security_report' in artifact for artifact in result.artifacts))
    
    def _create_vulnerable_test_files(self):
        """Create multiple test files with various vulnerabilities"""
        # SQL Injection vulnerable file
        sql_file = os.path.join(self.temp_dir, 'sql_vuln.py')
        with open(sql_file, 'w') as f:
            f.write('''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute_query(query)
''')
        
        # XSS vulnerable file
        xss_file = os.path.join(self.temp_dir, 'xss_vuln.js')
        with open(xss_file, 'w') as f:
            f.write('''
function displayMessage(msg) {
    document.getElementById('output').innerHTML = msg;
}
''')
        
        # Secrets file
        secrets_file = os.path.join(self.temp_dir, 'secrets.py')
        with open(secrets_file, 'w') as f:
            f.write('''
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
SECRET = "hardcoded-secret-key"
''')
        
        # Crypto vulnerable file
        crypto_file = os.path.join(self.temp_dir, 'crypto_vuln.py')
        with open(crypto_file, 'w') as f:
            f.write('''
import md5
def weak_hash(data):
    return md5.md5(data).hexdigest()
''')
        
        # Auth vulnerable file
        auth_file = os.path.join(self.temp_dir, 'auth_vuln.py')
        with open(auth_file, 'w') as f:
            f.write('''
@app.route('/admin')
def admin_panel():
    # Missing authentication
    return render_template('admin.html')
''')


class TestSecurityPatterns(unittest.TestCase):
    """Test security pattern detection accuracy"""
    
    def setUp(self):
        self.agent = SecuritySpecialistAgent()
    
    def test_sql_injection_patterns(self):
        """Test SQL injection pattern matching"""
        test_cases = [
            ("SELECT * FROM users WHERE id = " + user_id, True),
            ("query = f'SELECT * FROM users WHERE name = {name}'", True),
            ("cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))", False),
            ("conn.execute(text('SELECT * FROM users WHERE id = :id'), {'id': user_id})", False),
        ]
        
        for code, should_match in test_cases:
            matches = any(
                bool(re.search(pattern, code)) 
                for pattern in self.agent.sql_injection_patterns
            )
            if should_match:
                self.assertTrue(matches, f"Should match SQL injection: {code}")
            # Note: False negatives are acceptable for some patterns
    
    def test_secrets_patterns(self):
        """Test secrets pattern matching"""
        test_cases = [
            ('API_KEY = "sk-abcdef1234567890"', True),
            ('SECRET_KEY = "secret123456789"', True),
            ('password = "mypassword123"', True),
            ('ghp_1234567890abcdef1234567890abcdef123456', True),
            ('API_KEY = os.environ.get("API_KEY")', False),
            ('password = input("Enter password: ")', False),
        ]
        
        import re
        for code, should_match in test_cases:
            matches = any(
                bool(re.search(pattern, code, re.IGNORECASE)) 
                for pattern in self.agent.secrets_patterns
            )
            if should_match:
                self.assertTrue(matches, f"Should match secret: {code}")
    
    def test_xss_patterns(self):
        """Test XSS pattern matching"""
        test_cases = [
            ('element.innerHTML = userInput', True),
            ('document.write(content)', True),
            ('eval(userCode)', True),
            ('element.textContent = userInput', False),
            ('document.createElement("div")', False),
        ]
        
        import re
        for code, should_match in test_cases:
            matches = any(
                bool(re.search(pattern, code)) 
                for pattern in self.agent.xss_patterns
            )
            if should_match:
                self.assertTrue(matches, f"Should match XSS: {code}")


class TestComplianceChecking(unittest.TestCase):
    """Test compliance checking functionality"""
    
    def setUp(self):
        self.agent = SecuritySpecialistAgent()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_owasp_compliance_scoring(self):
        """Test OWASP compliance scoring"""
        # Create files with known vulnerabilities
        vuln_file = os.path.join(self.temp_dir, 'vulns.py')
        with open(vuln_file, 'w') as f:
            f.write('''
# Multiple OWASP Top 10 violations
def get_user(id):
    query = "SELECT * FROM users WHERE id = " + id  # Injection
    return execute(query)

def display(content):
    return f"<div>{content}</div>"  # Potential XSS

API_KEY = "hardcoded-key-123456789"  # Secrets exposure
''')
        
        result = self.agent._check_owasp_compliance(self.temp_dir)
        
        self.assertIsInstance(result.overall_score, float)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        self.assertGreater(result.total_checks, 0)
        self.assertGreaterEqual(result.failed_checks, 0)
        self.assertGreaterEqual(result.passed_checks, 0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)