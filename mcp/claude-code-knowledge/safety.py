"""
Safety and Error Handling Module for Claude Code Knowledge MCP Server.

Provides comprehensive error handling, input validation, output sanitization,
graceful degradation, and monitoring for safe operation of the MCP server.

Features:
- Input validation and sanitization
- Output filtering and safety checks
- Graceful degradation when knowledge graph unavailable
- Rate limiting and resource protection
- Comprehensive error logging and recovery
- Health monitoring and alerting
"""

import re
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import json


@dataclass
class ValidationError:
    """Validation error details."""
    field: str
    message: str
    severity: str = "medium"  # low, medium, high
    code: str = "VALIDATION_ERROR"


@dataclass
class SafetyMetrics:
    """Safety and performance metrics."""
    validation_errors: int = 0
    sanitization_actions: int = 0
    rate_limit_hits: int = 0
    graceful_degradations: int = 0
    error_recoveries: int = 0
    uptime_seconds: float = 0.0
    last_health_check: float = 0.0
    
    def __post_init__(self):
        self.start_time = time.time()
    
    def update_uptime(self):
        self.uptime_seconds = time.time() - self.start_time


class InputValidator:
    """Validates and sanitizes input parameters for MCP tools."""
    
    # Maximum input sizes to prevent resource exhaustion
    MAX_TEXT_LENGTH = 10000
    MAX_LIST_SIZE = 100
    MAX_QUERY_LENGTH = 1000
    
    # Allowed characters and patterns
    SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_.,;:!?\'"()\[\]{}@#$%^&*+=<>/\\|`~\n\r\t]*$')
    SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE),
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_errors = []
    
    def validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate parameters for specific MCP tool."""
        errors = []
        
        try:
            if tool_name == "check_compatibility":
                errors.extend(self._validate_compatibility_params(params))
            elif tool_name == "recommend_pattern":
                errors.extend(self._validate_pattern_params(params))
            elif tool_name == "find_alternatives":
                errors.extend(self._validate_alternatives_params(params))
            elif tool_name == "validate_architecture":
                errors.extend(self._validate_architecture_params(params))
            elif tool_name == "query_limitations":
                errors.extend(self._validate_limitations_params(params))
            else:
                errors.append(ValidationError(
                    field="tool_name",
                    message=f"Unknown tool: {tool_name}",
                    severity="high",
                    code="UNKNOWN_TOOL"
                ))
            
        except Exception as e:
            errors.append(ValidationError(
                field="validation",
                message=f"Validation failed: {str(e)}",
                severity="high",
                code="VALIDATION_EXCEPTION"
            ))
        
        return errors
    
    def _validate_compatibility_params(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate check_compatibility parameters."""
        errors = []
        
        # Required field: issue_description
        if 'issue_description' not in params:
            errors.append(ValidationError(
                field="issue_description",
                message="Required field missing",
                severity="high",
                code="MISSING_REQUIRED_FIELD"
            ))
        else:
            errors.extend(self._validate_text_field(
                "issue_description", params['issue_description']
            ))
        
        # Optional field: approach
        if 'approach' in params:
            errors.extend(self._validate_text_field(
                "approach", params['approach']
            ))
        
        return errors
    
    def _validate_pattern_params(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate recommend_pattern parameters."""
        errors = []
        
        # Required fields
        for field in ['technology', 'task_type']:
            if field not in params:
                errors.append(ValidationError(
                    field=field,
                    message="Required field missing",
                    severity="high",
                    code="MISSING_REQUIRED_FIELD"
                ))
            else:
                errors.extend(self._validate_identifier_field(field, params[field]))
        
        # Optional limit field
        if 'limit' in params:
            errors.extend(self._validate_integer_field(
                "limit", params['limit'], min_val=1, max_val=10
            ))
        
        return errors
    
    def _validate_alternatives_params(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate find_alternatives parameters."""
        errors = []
        
        # Required field: problematic_approach
        if 'problematic_approach' not in params:
            errors.append(ValidationError(
                field="problematic_approach",
                message="Required field missing",
                severity="high",
                code="MISSING_REQUIRED_FIELD"
            ))
        else:
            errors.extend(self._validate_text_field(
                "problematic_approach", params['problematic_approach']
            ))
        
        return errors
    
    def _validate_architecture_params(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate validate_architecture parameters."""
        errors = []
        
        # Required field: system_design
        if 'system_design' not in params:
            errors.append(ValidationError(
                field="system_design",
                message="Required field missing",
                severity="high",
                code="MISSING_REQUIRED_FIELD"
            ))
        else:
            errors.extend(self._validate_text_field(
                "system_design", params['system_design']
            ))
        
        return errors
    
    def _validate_limitations_params(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Validate query_limitations parameters."""
        errors = []
        
        # Required field: capability_area
        if 'capability_area' not in params:
            errors.append(ValidationError(
                field="capability_area",
                message="Required field missing",
                severity="high",
                code="MISSING_REQUIRED_FIELD"
            ))
        else:
            errors.extend(self._validate_identifier_field(
                "capability_area", params['capability_area']
            ))
        
        # Optional severity field
        if 'severity' in params:
            if params['severity'] not in ['low', 'medium', 'high']:
                errors.append(ValidationError(
                    field="severity",
                    message="Invalid severity level",
                    severity="medium",
                    code="INVALID_ENUM_VALUE"
                ))
        
        return errors
    
    def _validate_text_field(self, field_name: str, value: Any) -> List[ValidationError]:
        """Validate text field."""
        errors = []
        
        # Check type
        if not isinstance(value, str):
            errors.append(ValidationError(
                field=field_name,
                message=f"Expected string, got {type(value).__name__}",
                severity="high",
                code="INVALID_TYPE"
            ))
            return errors
        
        # Check length
        if len(value) > self.MAX_TEXT_LENGTH:
            errors.append(ValidationError(
                field=field_name,
                message=f"Text too long (max {self.MAX_TEXT_LENGTH} characters)",
                severity="high",
                code="TEXT_TOO_LONG"
            ))
        
        # Check for empty required fields
        if len(value.strip()) == 0:
            errors.append(ValidationError(
                field=field_name,
                message="Field cannot be empty",
                severity="high",
                code="EMPTY_FIELD"
            ))
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(value):
                errors.append(ValidationError(
                    field=field_name,
                    message="Potentially dangerous content detected",
                    severity="high",
                    code="DANGEROUS_CONTENT"
                ))
        
        # Check character set
        if not self.SAFE_TEXT_PATTERN.match(value):
            errors.append(ValidationError(
                field=field_name,
                message="Contains unsafe characters",
                severity="medium",
                code="UNSAFE_CHARACTERS"
            ))
        
        return errors
    
    def _validate_identifier_field(self, field_name: str, value: Any) -> List[ValidationError]:
        """Validate identifier field (more restrictive than text)."""
        errors = []
        
        # Check type
        if not isinstance(value, str):
            errors.append(ValidationError(
                field=field_name,
                message=f"Expected string, got {type(value).__name__}",
                severity="high",
                code="INVALID_TYPE"
            ))
            return errors
        
        # Check length
        if len(value) > 100:  # Identifiers should be shorter
            errors.append(ValidationError(
                field=field_name,
                message="Identifier too long (max 100 characters)",
                severity="high",
                code="IDENTIFIER_TOO_LONG"
            ))
        
        # Check pattern
        if not self.SAFE_IDENTIFIER_PATTERN.match(value):
            errors.append(ValidationError(
                field=field_name,
                message="Invalid identifier format",
                severity="high",
                code="INVALID_IDENTIFIER"
            ))
        
        return errors
    
    def _validate_integer_field(self, field_name: str, value: Any, 
                               min_val: Optional[int] = None, 
                               max_val: Optional[int] = None) -> List[ValidationError]:
        """Validate integer field."""
        errors = []
        
        # Check type
        if not isinstance(value, int):
            errors.append(ValidationError(
                field=field_name,
                message=f"Expected integer, got {type(value).__name__}",
                severity="high",
                code="INVALID_TYPE"
            ))
            return errors
        
        # Check range
        if min_val is not None and value < min_val:
            errors.append(ValidationError(
                field=field_name,
                message=f"Value too small (minimum {min_val})",
                severity="medium",
                code="VALUE_TOO_SMALL"
            ))
        
        if max_val is not None and value > max_val:
            errors.append(ValidationError(
                field=field_name,
                message=f"Value too large (maximum {max_val})",
                severity="medium",
                code="VALUE_TOO_LARGE"
            ))
        
        return errors
    
    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input parameters."""
        sanitized = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Remove dangerous patterns
                sanitized_value = value
                for pattern in self.DANGEROUS_PATTERNS:
                    sanitized_value = pattern.sub('', sanitized_value)
                
                # Truncate if too long
                if len(sanitized_value) > self.MAX_TEXT_LENGTH:
                    sanitized_value = sanitized_value[:self.MAX_TEXT_LENGTH]
                
                sanitized[key] = sanitized_value
            elif isinstance(value, list):
                # Limit list size
                if len(value) > self.MAX_LIST_SIZE:
                    sanitized[key] = value[:self.MAX_LIST_SIZE]
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        
        return sanitized


class OutputSanitizer:
    """Sanitizes output data to prevent information leakage."""
    
    # Sensitive patterns to filter out
    SENSITIVE_PATTERNS = [
        re.compile(r'password[\'\":\s]*[\'\"]\w+[\'\"]\s*', re.IGNORECASE),
        re.compile(r'api[_-]?key[\'\":\s]*[\'\"]\w+[\'\"]\s*', re.IGNORECASE),
        re.compile(r'secret[\'\":\s]*[\'\"]\w+[\'\"]\s*', re.IGNORECASE),
        re.compile(r'token[\'\":\s]*[\'\"]\w+[\'\"]\s*', re.IGNORECASE),
        re.compile(r'/[a-zA-Z]+/[a-zA-Z]+/[a-zA-Z0-9\-_]+\.git', re.IGNORECASE),  # git URLs
        re.compile(r'file://[^\s]+', re.IGNORECASE),  # file URLs
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def sanitize_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize response data."""
        try:
            return self._sanitize_recursive(response_data)
        except Exception as e:
            self.logger.error(f"Output sanitization failed: {e}")
            return {"error": "Response sanitization failed"}
    
    def _sanitize_recursive(self, data: Any) -> Any:
        """Recursively sanitize data structure."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Skip sensitive keys
                if key.lower() in ['password', 'secret', 'token', 'api_key']:
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_recursive(value)
            return sanitized
        
        elif isinstance(data, list):
            return [self._sanitize_recursive(item) for item in data]
        
        elif isinstance(data, str):
            return self._sanitize_string(data)
        
        else:
            return data
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string content."""
        sanitized = text
        
        # Remove sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized


class RateLimiter:
    """Rate limiting to prevent resource exhaustion."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        
        # Token bucket algorithm
        self.tokens = burst_size
        self.last_refill = time.time()
        
        # Request tracking
        self.request_times = []
        
        self.logger = logging.getLogger(__name__)
    
    def allow_request(self, client_id: str = "default") -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        # Refill tokens based on time elapsed
        time_elapsed = current_time - self.last_refill
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = current_time
        
        # Check if tokens available
        if self.tokens >= 1:
            self.tokens -= 1
            self.request_times.append(current_time)
            
            # Clean old request times
            cutoff_time = current_time - 60  # 1 minute
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            return True
        else:
            self.logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        recent_requests = len([t for t in self.request_times if t > current_time - 60])
        
        return {
            'requests_per_minute_limit': self.requests_per_minute,
            'burst_size': self.burst_size,
            'current_tokens': int(self.tokens),
            'requests_last_minute': recent_requests,
            'next_token_in_seconds': max(0, 60 / self.requests_per_minute - (current_time - self.last_refill))
        }


class GracefulDegradation:
    """Provides fallback responses when knowledge graph is unavailable."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_responses = self._load_fallback_responses()
    
    def _load_fallback_responses(self) -> Dict[str, Dict[str, Any]]:
        """Load static fallback responses."""
        return {
            "check_compatibility": {
                "compatible": False,
                "confidence": 0.1,
                "concepts_analyzed": 0,
                "issues": [{
                    "type": "system_unavailable",
                    "severity": "high",
                    "message": "Knowledge graph temporarily unavailable - cannot perform compatibility check"
                }],
                "recommendations": [
                    "Retry request after knowledge graph is restored",
                    "Use basic Claude Code patterns as fallback"
                ],
                "execution_time_ms": 0
            },
            
            "recommend_pattern": {
                "patterns": [{
                    "pattern_id": "fallback",
                    "name": "Basic Tool Usage Pattern",
                    "description": "Use built-in tools directly without orchestration",
                    "technology": "general",
                    "task_type": "general",
                    "code_example": "# Use Read(), Write(), Edit(), Bash() tools directly\nRead(file_path='/path/to/file')",
                    "confidence": 0.5,
                    "supporting_tools": ["Read", "Write", "Edit", "Bash"],
                    "usage_count": 0
                }],
                "search_query": "",
                "total_found": 1,
                "fallback_mode": True
            },
            
            "find_alternatives": {
                "alternatives": [{
                    "id": "fallback_alternative",
                    "name": "Direct Tool Usage",
                    "description": "Use Claude Code's built-in tools instead of complex orchestration",
                    "confidence": 0.6,
                    "technology": "general"
                }],
                "total_found": 1,
                "search_approach": "",
                "fallback_mode": True
            },
            
            "validate_architecture": {
                "valid": False,
                "confidence": 0.0,
                "components_analyzed": 0,
                "validation_results": [],
                "issues_found": [{
                    "issue": "Cannot validate architecture - knowledge graph unavailable",
                    "recommendation": "Ensure design uses direct tool calls and avoids orchestration patterns",
                    "severity": "high"
                }],
                "recommendations": [
                    "Use direct tool usage patterns",
                    "Avoid Task() orchestration",
                    "Leverage MCP servers for complex integrations"
                ],
                "fallback_mode": True
            },
            
            "query_limitations": {
                "limitations": [{
                    "limitation_id": "fallback_limitation",
                    "name": "Knowledge Graph Unavailable",
                    "category": "system",
                    "description": "Cannot access limitation database",
                    "severity": "high",
                    "impact": "Limited guidance available",
                    "workarounds": ["Use basic Claude Code patterns", "Consult documentation"],
                    "alternatives": [],
                    "documentation_link": ""
                }],
                "capability_area": "",
                "total_found": 1,
                "severity_filter": "all",
                "fallback_mode": True
            }
        }
    
    def get_fallback_response(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback response when knowledge graph unavailable."""
        self.logger.warning(f"Using fallback response for {tool_name}")
        
        if tool_name in self.fallback_responses:
            response = self.fallback_responses[tool_name].copy()
            
            # Customize response based on parameters
            if tool_name == "recommend_pattern":
                response["patterns"][0]["technology"] = params.get("technology", "general")
                response["patterns"][0]["task_type"] = params.get("task_type", "general")
            
            return response
        else:
            return {
                "error": f"Knowledge graph unavailable and no fallback for tool: {tool_name}",
                "fallback_mode": True
            }


def safety_wrapper(func: Callable) -> Callable:
    """Decorator to add safety features to MCP tool functions."""
    
    @wraps(func)
    async def wrapper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Input validation
            validator = InputValidator()
            validation_errors = validator.validate_tool_params(func.__name__.replace('_', ''), params)
            
            if validation_errors:
                high_severity_errors = [e for e in validation_errors if e.severity == "high"]
                if high_severity_errors:
                    return {
                        "error": "Input validation failed",
                        "validation_errors": [
                            {"field": e.field, "message": e.message, "code": e.code}
                            for e in high_severity_errors
                        ]
                    }
            
            # Sanitize input
            sanitized_params = validator.sanitize_params(params)
            
            # Call original function
            result = await func(self, sanitized_params)
            
            # Sanitize output
            sanitizer = OutputSanitizer()
            sanitized_result = sanitizer.sanitize_response(result)
            
            # Add execution metadata
            sanitized_result['execution_time_ms'] = (time.time() - start_time) * 1000
            sanitized_result['safe_execution'] = True
            
            return sanitized_result
            
        except Exception as e:
            # Log error
            logging.error(f"Safety wrapper caught error in {func.__name__}: {e}")
            logging.error(traceback.format_exc())
            
            # Return safe error response
            return {
                "error": "Internal processing error",
                "error_code": "INTERNAL_ERROR",
                "execution_time_ms": (time.time() - start_time) * 1000,
                "safe_execution": False
            }
    
    return wrapper


class HealthMonitor:
    """Monitors server health and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = SafetyMetrics()
        self.alert_thresholds = {
            'error_rate_threshold': 0.1,  # 10% error rate
            'response_time_threshold': 1000,  # 1 second
            'memory_threshold': 0.8,  # 80% memory usage
        }
    
    def record_request(self, tool_name: str, execution_time_ms: float, success: bool):
        """Record request metrics."""
        if not success:
            self.metrics.error_recoveries += 1
        
        self.metrics.last_health_check = time.time()
        self.metrics.update_uptime()
    
    def record_validation_error(self):
        """Record validation error."""
        self.metrics.validation_errors += 1
    
    def record_rate_limit_hit(self):
        """Record rate limit hit."""
        self.metrics.rate_limit_hits += 1
    
    def record_graceful_degradation(self):
        """Record graceful degradation event."""
        self.metrics.graceful_degradations += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        self.metrics.update_uptime()
        
        return {
            "status": "healthy",  # Could be "degraded" or "unhealthy"
            "uptime_seconds": self.metrics.uptime_seconds,
            "metrics": {
                "validation_errors": self.metrics.validation_errors,
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "graceful_degradations": self.metrics.graceful_degradations,
                "error_recoveries": self.metrics.error_recoveries
            },
            "last_check": self.metrics.last_health_check,
            "timestamp": time.time()
        }