#!/usr/bin/env python3
"""
Demo script for Security Specialist Agent - Issue #74
Demonstrates the capabilities of the security specialist agent
"""

import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add claude commands to path
sys.path.append(str(Path(__file__).parent))

from claude.commands.security_specialist_agent import SecuritySpecialistAgent, ComplianceStandard

def create_demo_vulnerable_app():
    """Create a demo vulnerable application for testing"""
    demo_dir = tempfile.mkdtemp(prefix="security_demo_")
    print(f"Creating demo vulnerable app in: {demo_dir}")
    
    # Web application with multiple vulnerabilities
    webapp_py = os.path.join(demo_dir, "webapp.py")
    with open(webapp_py, 'w') as f:
        f.write('''
from flask import Flask, request, render_template_string
import sqlite3
import hashlib
import md5  # Deprecated
import subprocess

app = Flask(__name__)

# Hardcoded secrets (BAD!)
SECRET_KEY = "super-secret-key-123456789"
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
DATABASE_URL = "postgresql://admin:password123@localhost/myapp"

# Debug mode enabled (BAD!)
app.config['DEBUG'] = True

@app.route('/user/<user_id>')
def get_user(user_id):
    """SQL Injection vulnerability"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Vulnerable SQL query - allows injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        return f"User: {user[0]}"
    return "User not found"

@app.route('/search')
def search():
    """XSS vulnerability"""
    query = request.args.get('q', '')
    
    # Dangerous: directly embedding user input into HTML
    template = f"<h1>Search Results for: {query}</h1>"
    return render_template_string(template)

@app.route('/admin')
def admin_panel():
    """Missing authentication"""
    # No authentication check!
    return "Welcome to admin panel"

@app.route('/execute')  
def execute_command():
    """Command injection vulnerability"""
    cmd = request.args.get('cmd', '')
    
    # Dangerous: executing user input
    result = subprocess.run(f"echo {cmd}", shell=True, capture_output=True, text=True)
    return result.stdout

def weak_hash_password(password):
    """Weak cryptography"""
    # Using deprecated MD5 (BAD!)
    return md5.md5(password.encode()).hexdigest()

def another_weak_hash(data):
    """Another weak hash function"""
    # SHA1 is also considered weak
    return hashlib.sha1(data.encode()).hexdigest()

# Logging sensitive data (BAD!)
def log_user_credentials(username, password):
    print(f"User login: {username}, password: {password}")

if __name__ == '__main__':
    # Running with debug=True in production (BAD!)
    app.run(debug=True, host='0.0.0.0')
''')
    
    # Frontend JavaScript with XSS vulnerabilities
    frontend_js = os.path.join(demo_dir, "frontend.js")
    with open(frontend_js, 'w') as f:
        f.write('''
// XSS vulnerabilities in JavaScript
function displayMessage(userInput) {
    // Dangerous: directly setting innerHTML with user input
    document.getElementById('message').innerHTML = userInput;
}

function loadContent(url) {
    // Dangerous: using eval with user input
    eval("loadPage('" + url + "')");
}

function showNotification(message) {
    // Dangerous: document.write with user input
    document.write("<div class='notification'>" + message + "</div>");
}

// Hardcoded API credentials in client-side code (BAD!)
const config = {
    apiKey: "api-key-exposed-in-client-123456789",
    secretToken: "ghp_1234567890abcdef1234567890abcdef123456",
    databaseUrl: "mongodb://user:password@localhost/app"
};

function authenticate(username, password) {
    // Insecure authentication storage
    localStorage.setItem('password', password);  // Never store passwords in localStorage!
    
    // Weak session management
    document.cookie = "session=" + btoa(username + ":" + password);
}
''')
    
    # Configuration file with more secrets
    config_py = os.path.join(demo_dir, "config.py")
    with open(config_py, 'w') as f:
        f.write('''
# Configuration with exposed secrets

# Database configuration
DB_HOST = "localhost"
DB_USER = "admin" 
DB_PASSWORD = "admin123"  # Weak password
DB_NAME = "production_db"

# API keys
STRIPE_SECRET_KEY = "sk_live_1234567890abcdef1234567890abcdef12345678"
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Third-party services
SENDGRID_API_KEY = "SG.1234567890abcdef1234567890abcdef.1234567890abcdef1234567890abcdef1234567890abcdef"
TWILIO_AUTH_TOKEN = "abcdef1234567890abcdef1234567890abcdef12"

# OAuth secrets
GOOGLE_CLIENT_SECRET = "GOCSPX-1234567890abcdef1234567890abcdef12"
FACEBOOK_APP_SECRET = "1234567890abcdef1234567890abcdef12345678"

# Default admin credentials (BAD!)
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "password"

# Security settings
SECRET_KEY = "insecure-secret-key-do-not-use-in-production"
SECURITY_PASSWORD_SALT = "simple-salt"

# Debug settings (should be False in production)
DEBUG = True
TESTING = True
''')
    
    # Package dependencies with known vulnerabilities
    requirements_txt = os.path.join(demo_dir, "requirements.txt")
    with open(requirements_txt, 'w') as f:
        f.write('''
flask==0.12.0
django==2.0.0
requests==2.18.0
urllib3==1.20.0
pyyaml==3.12
jinja2==2.8
werkzeug==0.12
''')
    
    # Node.js dependencies with vulnerabilities
    package_json = os.path.join(demo_dir, "package.json")
    package_data = {
        "name": "vulnerable-demo-app",
        "version": "1.0.0",
        "dependencies": {
            "express": "4.16.0",
            "lodash": "4.17.0",
            "axios": "0.19.0",
            "moment": "2.19.0",
            "jquery": "3.3.1"
        },
        "devDependencies": {
            "webpack": "4.28.0",
            "nodemon": "1.18.0"
        }
    }
    
    import json
    with open(package_json, 'w') as f:
        json.dump(package_data, f, indent=2)
    
    # Add some safe files to show the agent can distinguish
    safe_py = os.path.join(demo_dir, "safe_code.py")
    with open(safe_py, 'w') as f:
        f.write('''
# This file demonstrates secure coding practices

import os
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# Secure configuration using environment variables
SECRET_KEY = os.environ.get('SECRET_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_user_secure(user_id):
    """Secure user lookup with parameterized query"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Safe parameterized query
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def hash_password_secure(password):
    """Secure password hashing"""
    return generate_password_hash(password, method='pbkdf2:sha256')

def verify_password(password, hash):
    """Secure password verification"""
    return check_password_hash(hash, password)

def secure_hash(data):
    """Using secure hash algorithm"""
    return hashlib.sha256(data.encode()).hexdigest()

# This is a comment, not a secret
# api_key = "get this from environment variables"
''')
    
    return demo_dir

def run_security_demo():
    """Run comprehensive security demonstration"""
    print("=" * 60)
    print("RIF Security Specialist Agent - Demo")
    print("=" * 60)
    
    # Create demo vulnerable application
    demo_dir = create_demo_vulnerable_app()
    
    try:
        # Initialize security agent
        print("\n🔒 Initializing Security Specialist Agent...")
        agent = SecuritySpecialistAgent()
        print(f"✅ Agent initialized with {len(agent.get_capability_names())} capabilities")
        
        # Run comprehensive security scan
        print("\n🔍 Running comprehensive security scan...")
        task_data = {
            'scan_type': 'comprehensive',
            'target_path': demo_dir
        }
        
        result = agent.execute_task('demo_comprehensive_scan', task_data)
        
        if result.status.value == 'completed' and result.result_data:
            scan_data = result.result_data
            
            print(f"✅ Scan completed successfully!")
            print(f"📊 Files scanned: {scan_data.get('files_scanned', 0)}")
            print(f"📊 Lines analyzed: {scan_data.get('lines_scanned', 0)}")
            print(f"⏱️  Scan duration: {scan_data.get('scan_duration_seconds', 0):.2f} seconds")
            
            # Show vulnerability summary
            summary = scan_data.get('summary', {})
            print(f"\n🚨 Vulnerability Summary:")
            total_vulns = sum(summary.values())
            print(f"   Total vulnerabilities found: {total_vulns}")
            
            for severity, count in summary.items():
                if count > 0:
                    emoji = {"critical": "🔥", "high": "🚨", "medium": "⚠️", "low": "ℹ️", "info": "💡"}.get(severity, "❓")
                    print(f"   {emoji} {severity.upper()}: {count}")
            
            # Show detailed vulnerabilities (top 10)
            vulnerabilities = scan_data.get('vulnerabilities', [])
            if vulnerabilities:
                print(f"\n🔍 Top Vulnerabilities Found:")
                for i, vuln in enumerate(vulnerabilities[:10], 1):
                    severity_emoji = {
                        "CRITICAL": "🔥", "HIGH": "🚨", "MEDIUM": "⚠️", 
                        "LOW": "ℹ️", "INFO": "💡"
                    }.get(vuln['severity'], "❓")
                    
                    print(f"\n   {i}. {severity_emoji} {vuln['title']} ({vuln['severity']})")
                    print(f"      📁 File: {vuln['file_path']}:{vuln['line_number']}")
                    print(f"      📝 {vuln['description'][:100]}...")
                    print(f"      🔧 {vuln['remediation'][:100]}...")
                    print(f"      🎯 Confidence: {vuln['confidence']:.0%}")
            
            # Show recommendations
            recommendations = scan_data.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Security Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {rec}")
        
        else:
            print(f"❌ Scan failed: {result.error_message}")
            return
        
        # Run specific scans
        print(f"\n" + "="*60)
        print("🔐 Running Targeted Security Scans")
        print("="*60)
        
        # Secrets scan
        print(f"\n🔑 Scanning for hardcoded secrets...")
        secrets_result = agent.execute_task('demo_secrets_scan', {
            'scan_type': 'secrets',
            'target_path': demo_dir
        })
        
        if secrets_result.status.value == 'completed':
            secrets_data = secrets_result.result_data
            secrets_found = len(secrets_data.get('vulnerabilities', []))
            print(f"✅ Secrets scan completed: {secrets_found} potential secrets found")
        
        # OWASP Top 10 scan
        print(f"\n🛡️  Running OWASP Top 10 scan...")
        owasp_result = agent.execute_task('demo_owasp_scan', {
            'scan_type': 'owasp',
            'target_path': demo_dir
        })
        
        if owasp_result.status.value == 'completed':
            owasp_data = owasp_result.result_data
            owasp_vulns = len(owasp_data.get('vulnerabilities', []))
            print(f"✅ OWASP scan completed: {owasp_vulns} OWASP vulnerabilities found")
        
        # Compliance check
        print(f"\n📋 Running compliance check...")
        compliance_result = agent.execute_task('demo_compliance_check', {
            'scan_type': 'compliance',
            'target_path': demo_dir,
            'standards': [ComplianceStandard.OWASP_TOP_10]
        })
        
        if compliance_result.status.value == 'completed':
            compliance_data = compliance_result.result_data
            if 'owasp_top_10' in compliance_data:
                owasp_compliance = compliance_data['owasp_top_10']
                score = owasp_compliance.get('overall_score', 0)
                print(f"✅ OWASP Top 10 Compliance Score: {score:.0%}")
                print(f"   ✅ Passed checks: {owasp_compliance.get('passed_checks', 0)}")
                print(f"   ❌ Failed checks: {owasp_compliance.get('failed_checks', 0)}")
        
        # Show agent metrics
        print(f"\n" + "="*60)
        print("📊 Agent Performance Metrics")
        print("="*60)
        
        status = agent.get_status()
        metrics = status.get('metrics', {})
        
        print(f"🎯 Tasks completed: {metrics.get('tasks_completed', 0)}")
        print(f"❌ Tasks failed: {metrics.get('tasks_failed', 0)}")
        print(f"📈 Success rate: {metrics.get('success_rate', 0):.0%}")
        print(f"⏱️  Total execution time: {metrics.get('total_execution_time', 0):.2f} seconds")
        
        # Show recent task history
        history = agent.get_task_history(limit=5)
        if history:
            print(f"\n📝 Recent Task History:")
            for task in history:
                status_emoji = {"completed": "✅", "failed": "❌", "timeout": "⏰"}.get(task['status'], "❓")
                print(f"   {status_emoji} {task['task_id']}: {task['status']} ({task['duration_seconds']:.2f}s)")
        
        print(f"\n" + "="*60)
        print("🎉 Security Specialist Agent Demo Complete!")
        print("="*60)
        print(f"\n📁 Demo files created in: {demo_dir}")
        print(f"🔍 The agent successfully identified multiple security vulnerabilities")
        print(f"💡 Review the recommendations to improve security posture")
        print(f"\n🧹 Cleaning up demo files...")
        
    finally:
        # Cleanup
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print(f"✅ Demo files cleaned up")

def main():
    """Main demo function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Security Specialist Agent Demo")
            print("Usage: python3 demo_security_specialist.py [--help]")
            print("\nThis demo creates a vulnerable application and runs")
            print("comprehensive security scans to demonstrate the agent's capabilities.")
            return
    
    try:
        run_security_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()