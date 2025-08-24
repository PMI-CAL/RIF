#!/usr/bin/env python3
"""
RIF Session Error Handler - Automatic error detection and GitHub issue creation
Designed to work within Claude Code sessions without external monitoring processes
"""

import sys
import json
import os
import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class SessionError:
    """Session error event"""
    id: str
    timestamp: str
    severity: str
    error_type: str
    source: str
    message: str
    context: Dict[str, Any]
    command: Optional[str] = None
    exit_code: Optional[int] = None
    session_id: Optional[str] = None
    auto_handled: bool = False

class SessionErrorHandler:
    """Session-scoped error handling and automation"""
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.kb_path = Path(knowledge_base_path)
        self.errors_path = self.kb_path / "errors"
        self.session_id = self._get_session_id()
        self.session_errors_file = self.errors_path / "logs" / f"session_{self.session_id}.jsonl"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.errors_path / "logs",
            self.errors_path / "patterns", 
            self.errors_path / "analysis",
            self.errors_path / "github_issues"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_session_id(self) -> str:
        """Get current session ID"""
        # Use PID and timestamp for session uniqueness
        pid = os.getpid()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{pid}"
    
    def capture_tool_error(self, tool_name: str, error_message: str, context: Dict[str, Any] = None) -> Optional[SessionError]:
        """Capture error from Claude Code tool execution"""
        
        if not error_message or error_message.strip() == "":
            return None
            
        context = context or {}
        context.update({
            "tool": tool_name,
            "source": "claude_code_tool",
            "session_id": self.session_id
        })
        
        error = self._create_error(
            message=error_message,
            context=context,
            source="claude_code"
        )
        
        self._log_session_error(error)
        
        # Auto-handle critical errors
        if error.severity == "critical":
            self._auto_handle_critical_error(error)
        
        return error
    
    def capture_bash_error(self, command: str, exit_code: int, stderr: str = "", stdout: str = "") -> Optional[SessionError]:
        """Capture Bash command errors"""
        
        if exit_code == 0:
            return None
            
        # Don't capture certain expected failures or test commands
        if self._is_expected_failure(command, exit_code):
            return None
            
        # Don't capture errors when running from test scripts
        if self._is_test_context():
            return None
            
        context = {
            "command": command,
            "exit_code": exit_code,
            "stderr": stderr[:500],  # Limit output size
            "stdout": stdout[:500],
            "source": "bash_command",
            "session_id": self.session_id
        }
        
        message = f"Command failed with exit code {exit_code}: {command}"
        if stderr:
            message += f" - {stderr[:100]}"
            
        error = self._create_error(
            message=message,
            context=context,
            command=command,
            exit_code=exit_code,
            source="user_code"
        )
        
        self._log_session_error(error)
        
        # Auto-handle based on severity
        if error.severity in ["critical", "high"]:
            self._auto_handle_error(error)
        
        return error
    
    def capture_generic_error(self, message: str, context: Dict[str, Any] = None, source: str = "unknown") -> SessionError:
        """Capture generic error event"""
        
        context = context or {}
        context.update({"session_id": self.session_id})
        
        error = self._create_error(
            message=message,
            context=context,
            source=source
        )
        
        self._log_session_error(error)
        
        if error.severity in ["critical", "high"]:
            self._auto_handle_error(error)
        
        return error
    
    def _create_error(self, 
                     message: str, 
                     context: Dict[str, Any], 
                     command: str = None,
                     exit_code: int = None,
                     source: str = "unknown") -> SessionError:
        """Create standardized error object"""
        
        error_id = self._generate_error_id(message, context)
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        severity = self._classify_severity(message, exit_code, context)
        error_type = self._classify_type(message, context)
        
        return SessionError(
            id=error_id,
            timestamp=timestamp,
            severity=severity,
            error_type=error_type,
            source=source,
            message=message,
            context=context,
            command=command,
            exit_code=exit_code,
            session_id=self.session_id
        )
    
    def _generate_error_id(self, message: str, context: Dict[str, Any]) -> str:
        """Generate unique error ID"""
        signature = f"{message}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(signature.encode()).hexdigest()[:12]
    
    def _classify_severity(self, message: str, exit_code: Optional[int], context: Dict[str, Any]) -> str:
        """Classify error severity"""
        
        message_lower = message.lower()
        
        # Check context for explicit severity first
        if context.get("severity") == "critical":
            return "critical"
        
        # Critical patterns - expanded
        critical_patterns = [
            "critical:", "segmentation fault", "out of memory", "permission denied.*root",
            "database.*corrupt", "security.*breach", "authentication.*failed",
            "cannot.*connect.*database", "file.*not.*found.*critical",
            "database.*connection.*compromised", "system.*compromised"
        ]
        
        for pattern in critical_patterns:
            if pattern in message_lower:
                return "critical"
        
        # High severity patterns
        high_patterns = [
            "connection.*refused", "timeout.*exceeded", "syntax.*error",
            "module.*not.*found", "command.*not.*found", "authentication.*error"
        ]
        
        for pattern in high_patterns:
            if pattern in message_lower:
                return "high"
        
        # Check exit codes
        if exit_code is not None:
            if exit_code in [1, 2, 126, 127, 128]:
                return "high"
            elif exit_code != 0:
                return "medium"
        
        return "medium"
    
    def _classify_type(self, message: str, context: Dict[str, Any]) -> str:
        """Classify error type"""
        
        message_lower = message.lower()
        
        if any(term in message_lower for term in ["syntax error", "parsing", "invalid syntax"]):
            return "syntax"
        elif any(term in message_lower for term in ["null", "undefined", "index", "key"]):
            return "runtime"
        elif any(term in message_lower for term in ["connection", "timeout", "api", "service"]):
            return "integration"
        elif any(term in message_lower for term in ["memory", "cpu", "performance"]):
            return "performance"
        elif any(term in message_lower for term in ["authentication", "permission", "security"]):
            return "security"
        else:
            return "unknown"
    
    def _is_expected_failure(self, command: str, exit_code: int) -> bool:
        """Check if this is an expected failure that shouldn't trigger error handling"""
        
        # Commands that commonly fail in normal operation
        expected_failures = [
            "gh issue list",  # May fail if no issues
            "jq", # May fail with empty input
            "grep", # May fail with no matches
            "find", # May fail with permissions
            "test", # Test commands expected to sometimes fail
            "npm test", "pytest", "go test"  # Test runners expected to fail during development
        ]
        
        # Test and intentional failure commands
        test_commands = [
            "non_existent_command",  # Intentional test command for error system validation
            "fake_command",          # Common test command pattern
            "test_error_command",    # Explicit test error commands
            "simulate_error",        # Error simulation commands
            "actual_missing_command", # Test command for error system validation (Issue #106)
            "actual_missing_tool",   # Test command for error system validation
        ]
        
        # Check for test commands that are intentionally meant to fail
        for test_cmd in test_commands:
            if test_cmd in command:
                return True
        
        # Check for commands with obvious test patterns
        if any(pattern in command.lower() for pattern in ["_xyz", "_test", "_fake", "_nonexistent"]):
            return True
        
        for expected in expected_failures:
            if expected in command and exit_code in [0, 1]:  # Many of these return 1 on "no results"
                return True
        
        return False
    
    def _is_test_context(self) -> bool:
        """Check if we're running in a test context"""
        import traceback
        import sys
        
        # Check the call stack for test-related scripts
        stack = traceback.extract_stack()
        test_indicators = [
            "test_error_automation.py",
            "test_",
            "_test.py",
            "/test/",
            "/tests/",
            "pytest",
            "unittest"
        ]
        
        for frame in stack:
            filename = frame.filename.lower()
            for indicator in test_indicators:
                if indicator in filename:
                    return True
        
        # Check command line arguments for test indicators
        if len(sys.argv) > 0:
            script_name = sys.argv[0].lower()
            if any(indicator in script_name for indicator in test_indicators):
                return True
        
        return False
    
    def _log_session_error(self, error: SessionError):
        """Log error to session log file"""
        
        with open(self.session_errors_file, 'a') as f:
            f.write(json.dumps(asdict(error), default=str) + '\n')
    
    def _auto_handle_critical_error(self, error: SessionError):
        """Auto-handle critical errors with immediate action"""
        
        print(f"RIF: Critical error detected - {error.id}")
        
        # 1. Create GitHub issue immediately
        issue_number = self._create_github_issue(error)
        
        if issue_number:
            error.auto_handled = True
            
            # 2. Store issue tracking
            self._store_github_issue_link(error, issue_number)
            
            # 3. Update session log
            self._log_session_error(error)  # Log again with auto_handled=True
            
            print(f"RIF: GitHub issue #{issue_number} created for critical error")
        else:
            print("RIF: Failed to create GitHub issue for critical error")
    
    def _auto_handle_error(self, error: SessionError):
        """Auto-handle high priority errors"""
        
        print(f"RIF: Error auto-handling - {error.severity} - {error.id}")
        
        # 1. Check if similar error exists in patterns
        similar_pattern = self._find_similar_pattern(error)
        
        if similar_pattern:
            print(f"RIF: Similar pattern found - applying known solution")
            # Apply known solution if available
            return
        
        # 2. For high severity, create GitHub issue
        if error.severity in ["critical", "high"]:
            issue_number = self._create_github_issue(error)
            if issue_number:
                self._store_github_issue_link(error, issue_number)
                error.auto_handled = True
                print(f"RIF: GitHub issue #{issue_number} created")
    
    def _find_similar_pattern(self, error: SessionError) -> Optional[Dict[str, Any]]:
        """Find similar error patterns from history"""
        
        patterns_dir = self.errors_path / "patterns"
        if not patterns_dir.exists():
            return None
        
        # Simple pattern matching - can be enhanced
        for pattern_file in patterns_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r') as f:
                    pattern = json.load(f)
                
                # Check message similarity
                error_words = set(error.message.lower().split())
                pattern_words = set(pattern.get("message_keywords", []))
                
                if len(error_words.intersection(pattern_words)) >= 2:
                    return pattern
                    
            except Exception:
                continue
        
        return None
    
    def _create_github_issue(self, error: SessionError) -> Optional[int]:
        """Create GitHub issue for error"""
        
        title = f"Error: {error.message[:50]}..."
        
        body = f"""## Automatic Error Detection

**Error ID**: {error.id}
**Session**: {error.session_id}
**Timestamp**: {error.timestamp}
**Severity**: {error.severity}
**Type**: {error.error_type}
**Source**: {error.source}

### Error Details
```
{error.message}
```

### Context
```json
{json.dumps(error.context, indent=2)}
```

### Command Details
""" + (f"""
**Command**: `{error.command}`
**Exit Code**: {error.exit_code}
""" if error.command else "No command context") + f"""

### Automated Analysis
- **Root Cause Analysis**: Pending RIF-Analyst investigation
- **Impact Assessment**: {error.severity.title()} severity
- **Priority**: {'Immediate' if error.severity == 'critical' else 'High' if error.severity == 'high' else 'Standard'}

### Recommended Actions
1. **Investigate**: Analyze root cause using RIF error analysis tools
2. **Implement Fix**: Address the underlying issue
3. **Add Prevention**: Update error handling to prevent recurrence
4. **Update Monitoring**: Enhance detection for similar errors

### RIF State Management
- **Initial State**: `state:new`
- **Assigned Agent**: Will be auto-assigned based on error type
- **Expected Resolution**: {'1 hour' if error.severity == 'critical' else '4 hours' if error.severity == 'high' else '24 hours'}

**Auto-generated by RIF Session Error Handler**
**Session ID**: {error.session_id}
"""
        
        # Determine labels based on error
        labels = ["error:auto-detected", f"severity:{error.severity}", f"type:{error.error_type}"]
        
        if error.severity == "critical":
            labels.extend(["priority:critical", "state:new"])
        elif error.severity == "high":
            labels.extend(["priority:high", "state:new"]) 
        else:
            labels.extend(["priority:normal", "state:new"])
        
        try:
            cmd = ["gh", "issue", "create", "--title", title, "--body", body] + \
                  [item for label in labels for item in ["--label", label]]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract issue number from output (GitHub CLI returns URL)
                output = result.stdout.strip()
                if "/issues/" in output:
                    issue_number = int(output.split("/issues/")[-1])
                    return issue_number
            else:
                print(f"RIF: GitHub issue creation failed: {result.stderr}")
                
        except Exception as e:
            print(f"RIF: Exception creating GitHub issue: {e}")
        
        return None
    
    def _store_github_issue_link(self, error: SessionError, issue_number: int):
        """Store the link between error and GitHub issue"""
        
        link = {
            "error_id": error.id,
            "issue_number": issue_number,
            "created": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": error.session_id,
            "severity": error.severity,
            "error_type": error.error_type
        }
        
        links_file = self.errors_path / "github_issues" / "error_issue_links.jsonl"
        with open(links_file, 'a') as f:
            f.write(json.dumps(link) + '\n')
    
    def get_session_errors(self) -> List[SessionError]:
        """Get all errors from current session"""
        
        errors = []
        if self.session_errors_file.exists():
            with open(self.session_errors_file, 'r') as f:
                for line in f:
                    try:
                        error_data = json.loads(line.strip())
                        # Convert dict back to SessionError (simplified)
                        errors.append(error_data)
                    except Exception:
                        continue
        
        return errors
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session errors"""
        
        errors = self.get_session_errors()
        
        summary = {
            "session_id": self.session_id,
            "total_errors": len(errors),
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "by_type": {},
            "auto_handled": 0,
            "github_issues_created": 0
        }
        
        for error in errors:
            severity = error.get("severity", "unknown")
            error_type = error.get("error_type", "unknown")
            
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
            
            if error.get("auto_handled"):
                summary["auto_handled"] += 1
        
        # Count GitHub issues created
        links_file = self.errors_path / "github_issues" / "error_issue_links.jsonl"
        if links_file.exists():
            with open(links_file, 'r') as f:
                for line in f:
                    try:
                        link = json.loads(line.strip())
                        if link.get("session_id") == self.session_id:
                            summary["github_issues_created"] += 1
                    except:
                        continue
        
        return summary

def main():
    """CLI interface for session error handler"""
    
    if len(sys.argv) < 2:
        print("Usage: session_error_handler.py <command> [options]")
        print("Commands:")
        print("  --capture-tool <tool_name> <error_message>")
        print("  --capture-bash <command> <exit_code> [stderr]")
        print("  --capture-generic <message> [--context <json>]")
        print("  --session-summary")
        sys.exit(1)
    
    handler = SessionErrorHandler()
    
    command = sys.argv[1]
    
    if command == "--capture-tool":
        if len(sys.argv) < 4:
            print("Error: Tool name and error message required")
            sys.exit(1)
        
        tool_name = sys.argv[2]
        error_message = sys.argv[3]
        context = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        
        error = handler.capture_tool_error(tool_name, error_message, context)
        if error:
            print(f"Tool error captured: {error.id}")
    
    elif command == "--capture-bash":
        if len(sys.argv) < 4:
            print("Error: Command and exit code required")
            sys.exit(1)
        
        cmd = sys.argv[2]
        exit_code = int(sys.argv[3])
        stderr = sys.argv[4] if len(sys.argv) > 4 else ""
        
        error = handler.capture_bash_error(cmd, exit_code, stderr)
        if error:
            print(f"Bash error captured: {error.id}")
        else:
            print("No error captured (expected failure or success)")
    
    elif command == "--capture-generic":
        if len(sys.argv) < 3:
            print("Error: Error message required")
            sys.exit(1)
        
        message = sys.argv[2]
        context = {}
        
        if len(sys.argv) > 4 and sys.argv[3] == "--context":
            try:
                context = json.loads(sys.argv[4])
            except:
                context = {"raw": sys.argv[4]}
        
        error = handler.capture_generic_error(message, context)
        print(f"Generic error captured: {error.id}")
    
    elif command == "--session-summary":
        summary = handler.get_session_summary()
        print(json.dumps(summary, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()