# Security Specialist Agent - Issue #74

## Overview

The Security Specialist Agent is a specialized domain agent designed for comprehensive security analysis of codebases. It provides automated vulnerability detection, compliance checking, and security best practices validation.

## Features

### Core Security Capabilities

- **OWASP Top 10 Vulnerability Detection**
  - SQL Injection
  - Cross-Site Scripting (XSS)
  - Broken Authentication
  - Sensitive Data Exposure
  - Security Misconfiguration
  - Cryptographic Failures

- **Secrets Detection**
  - API keys and tokens
  - Hardcoded passwords
  - Database credentials
  - Cloud service keys (AWS, OpenAI, GitHub, etc.)

- **Dependency Analysis**
  - Python package vulnerabilities
  - Node.js package vulnerabilities  
  - Known CVE detection

- **Authentication/Authorization Analysis**
  - Missing authentication checks
  - Weak password policies
  - JWT security issues

- **Compliance Checking**
  - OWASP Top 10 compliance
  - PCI DSS (basic checks)
  - Extensible framework for other standards

## Architecture

### Class Hierarchy

```
DomainAgent (Base Class)
    └── SecuritySpecialistAgent
        ├── VulnerabilityType (Enum)
        ├── SeverityLevel (Enum)
        ├── ComplianceStandard (Enum)
        ├── SecurityVulnerability (DataClass)
        ├── SecurityScanResult (DataClass)
        └── ComplianceResult (DataClass)
```

### Key Components

1. **Pattern Matching Engine**: Uses regex patterns to detect security vulnerabilities
2. **Vulnerability Prioritization**: Ranks findings by severity and confidence
3. **Compliance Framework**: Extensible system for checking against security standards
4. **Result Aggregation**: Comprehensive reporting with recommendations

## Usage

### Basic Security Scan

```python
from claude.commands.security_specialist_agent import SecuritySpecialistAgent

# Initialize agent
agent = SecuritySpecialistAgent()

# Comprehensive scan
task_data = {
    'scan_type': 'comprehensive',
    'target_path': './my-project'
}

result = agent.execute_task('security_scan', task_data)
print(f"Found {len(result.result_data['vulnerabilities'])} vulnerabilities")
```

### Specific Scan Types

```python
# Secrets-only scan
secrets_task = {
    'scan_type': 'secrets',
    'target_path': './src'
}
result = agent.execute_task('secrets_scan', secrets_task)

# OWASP Top 10 scan
owasp_task = {
    'scan_type': 'owasp',
    'target_path': './webapp'
}
result = agent.execute_task('owasp_scan', owasp_task)

# Compliance check
compliance_task = {
    'scan_type': 'compliance',
    'target_path': './project',
    'standards': ['owasp_top_10', 'pci_dss']
}
result = agent.execute_task('compliance_check', compliance_task)
```

### Command Line Usage

```bash
# Direct execution for testing
python3 claude/commands/security_specialist_agent.py ./project-path

# Via RIF orchestration
gh issue create --title "Security scan needed" --body "scan_type: comprehensive"
```

## Vulnerability Types Detected

### Injection Vulnerabilities
- SQL injection via string concatenation
- Command injection
- LDAP injection
- XPath injection

**Example Detection:**
```python
# Will be flagged as SQL injection
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
```

### Cross-Site Scripting (XSS)
- DOM-based XSS
- Unsafe innerHTML usage
- Dynamic script execution

**Example Detection:**
```javascript
// Will be flagged as XSS vulnerability
element.innerHTML = userInput;
document.write(content);
```

### Secrets Exposure
- API keys
- Database passwords
- Cloud service credentials
- Authentication tokens

**Example Detection:**
```python
# Will be flagged as secret exposure
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
DATABASE_URL = "postgresql://user:password@host/db"
```

### Authentication Issues
- Missing authentication checks
- Weak password policies
- Insecure JWT handling

**Example Detection:**
```python
# Will be flagged as missing authentication
@app.route('/admin')
def admin_panel():
    return render_template('admin.html')  # No auth check
```

### Cryptographic Failures
- Weak hashing algorithms (MD5, SHA1)
- Insecure encryption
- Hardcoded cryptographic keys

**Example Detection:**
```python
# Will be flagged as weak crypto
import md5
hash_value = md5.md5(data).hexdigest()
```

## Configuration

### Agent Configuration

```python
from claude.commands.domain_agent_base import AgentConfiguration

config = AgentConfiguration(
    timeout_seconds=600,        # 10 minute timeout
    max_retries=3,             # Retry failed scans
    memory_limit_mb=1024,      # Memory limit
    enable_logging=True,       # Enable detailed logging
    output_directory="./security_reports"
)

agent = SecuritySpecialistAgent(config)
```

### Custom Security Patterns

The agent can be extended with custom patterns:

```python
# Add custom SQL injection patterns
agent.sql_injection_patterns.append(r'my_custom_pattern')

# Add custom secret patterns  
agent.secrets_patterns.append(r'MY_API_KEY\s*=\s*["\'][^"\']{16,}["\']')
```

## Output Format

### Comprehensive Scan Result

```json
{
  "scan_id": "scan_1692808800",
  "vulnerabilities": [
    {
      "vuln_id": "inj_a1b2c3d4",
      "vuln_type": "injection",
      "severity": "HIGH",
      "title": "Potential SQL Injection",
      "description": "Dynamic SQL query construction detected...",
      "file_path": "src/database.py",
      "line_number": 45,
      "code_snippet": "query = \"SELECT * FROM users WHERE id = \" + user_id",
      "remediation": "Use parameterized queries...",
      "references": ["https://owasp.org/Top10/A03_2021-Injection/"],
      "cwe_id": "CWE-89",
      "confidence": 0.9
    }
  ],
  "summary": {
    "critical": 0,
    "high": 1,
    "medium": 3,
    "low": 0,
    "info": 0
  },
  "files_scanned": 25,
  "lines_scanned": 3420,
  "recommendations": [
    "Implement parameterized queries to prevent SQL injection",
    "Add input validation and sanitization",
    "Conduct security code reviews"
  ],
  "scan_duration_seconds": 2.3
}
```

### Compliance Check Result

```json
{
  "owasp_top_10": {
    "overall_score": 0.8,
    "passed_checks": 8,
    "failed_checks": 2,
    "total_checks": 10,
    "findings": [
      {
        "category": "A03:2021 – Injection",
        "severity": "HIGH",
        "file": "src/database.py",
        "line": 45,
        "description": "SQL injection vulnerability detected"
      }
    ],
    "recommendations": [
      "Address identified OWASP Top 10 vulnerabilities",
      "Implement secure coding training"
    ]
  }
}
```

## Integration with RIF Orchestration

The Security Specialist Agent integrates seamlessly with the RIF orchestration system:

### Automatic Activation

The agent activates automatically when:
- GitHub issues contain security-related keywords
- Code changes affect security-sensitive files
- Compliance scans are scheduled
- Vulnerability reports are requested

### Workflow Integration

```yaml
# Example RIF workflow trigger
- name: "Security Review"
  trigger:
    - file_patterns: ["*.py", "*.js", "*.java"]
    - keywords: ["security", "vulnerability", "compliance"]
  agents:
    - type: "security-specialist"
      priority: "high"
      scan_types: ["comprehensive", "compliance"]
```

## Best Practices

### For Development Teams

1. **Regular Scans**: Run comprehensive scans before major releases
2. **CI/CD Integration**: Include security scans in build pipelines
3. **False Positive Management**: Review and tune patterns based on findings
4. **Remediation Tracking**: Address high-severity findings promptly

### For Security Teams

1. **Custom Patterns**: Extend detection rules for organization-specific vulnerabilities
2. **Compliance Mapping**: Configure compliance checks for relevant standards
3. **Metrics Tracking**: Monitor vulnerability trends over time
4. **Training Integration**: Use findings to guide security training

### For Automated Scanning

1. **Incremental Scans**: Focus on changed files for faster feedback
2. **Prioritized Remediation**: Address critical and high-severity issues first
3. **Exception Management**: Document and track accepted risks
4. **Reporting Integration**: Export results to security dashboards

## Extensibility

### Adding New Vulnerability Types

```python
class CustomVulnerabilityType(Enum):
    BUSINESS_LOGIC = "business_logic"
    PRIVACY_LEAK = "privacy_leak"

# Extend the agent
class ExtendedSecurityAgent(SecuritySpecialistAgent):
    def __init__(self):
        super().__init__()
        self.custom_patterns = {
            'business_logic': [r'pattern1', r'pattern2'],
            'privacy_leak': [r'pattern3', r'pattern4']
        }
    
    def check_custom_vulnerabilities(self, code_base):
        # Implementation for custom checks
        pass
```

### Custom Compliance Standards

```python
def check_custom_compliance(self, code_base):
    """Check against custom compliance standard"""
    findings = []
    
    # Custom compliance logic
    custom_checks = [
        ("requirement_1", self._check_requirement_1(code_base)),
        ("requirement_2", self._check_requirement_2(code_base)),
    ]
    
    passed_checks = sum(1 for _, passed in custom_checks if passed)
    total_checks = len(custom_checks)
    
    return ComplianceResult(
        standard=ComplianceStandard.CUSTOM,
        overall_score=passed_checks / total_checks,
        passed_checks=passed_checks,
        failed_checks=total_checks - passed_checks,
        total_checks=total_checks,
        findings=findings,
        recommendations=self._generate_custom_recommendations(custom_checks)
    )
```

## Performance Considerations

### Resource Usage
- Memory usage scales with codebase size
- CPU usage peaks during pattern matching
- Disk I/O for reading source files

### Optimization Strategies
- **File Filtering**: Skip non-source files and build artifacts
- **Parallel Processing**: Scan multiple files concurrently
- **Pattern Optimization**: Use efficient regex patterns
- **Incremental Scanning**: Focus on changed files only

### Typical Performance
- **Small Projects** (< 1000 files): 1-5 seconds
- **Medium Projects** (1000-10000 files): 5-30 seconds  
- **Large Projects** (> 10000 files): 30+ seconds

## Troubleshooting

### Common Issues

**Import Errors**
```bash
ModuleNotFoundError: No module named 'domain_agent_base'
```
*Solution*: Ensure proper Python path configuration

**Permission Errors**
```bash
PermissionError: [Errno 13] Permission denied
```
*Solution*: Check file/directory permissions

**Memory Issues**
```bash
MemoryError: Unable to allocate array
```
*Solution*: Increase memory limits or process files in batches

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = SecuritySpecialistAgent()
# Run scans with detailed output
```

Check agent status:

```python
status = agent.get_status()
print(f"Agent status: {status}")

history = agent.get_task_history()
print(f"Recent tasks: {history}")
```

## Future Enhancements

### Planned Features
- Machine learning-based vulnerability detection
- Integration with external vulnerability databases
- Real-time monitoring and alerting
- Advanced compliance reporting
- Custom vulnerability scoring

### Extensibility Roadmap
- Plugin architecture for custom scanners
- REST API for external integrations
- Web dashboard for results visualization
- Advanced remediation recommendations
- Integration with security tools (SAST/DAST)

## Contributing

To contribute to the Security Specialist Agent:

1. **Add new vulnerability patterns** to existing detection methods
2. **Implement new compliance standards** following the existing framework
3. **Extend test coverage** for new functionality
4. **Improve performance** through optimization
5. **Add documentation** for new features

See the test suite in `tests/test_security_specialist_agent.py` for examples of how to test new functionality.

## License

This Security Specialist Agent is part of the RIF (Reactive Intelligence Framework) and follows the same licensing terms as the main project.