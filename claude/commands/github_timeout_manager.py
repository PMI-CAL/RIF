#!/usr/bin/env python3
"""
GitHub Timeout Manager - Advanced API Timeout Handling
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Intelligent timeout management with adaptive configuration based on endpoint performance,
request context preservation, and integration with rate limiting systems.
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import statistics
from collections import deque, defaultdict

from systems.event_service_bus import CircuitBreaker

logger = logging.getLogger(__name__)

class GitHubEndpoint(Enum):
    """GitHub API endpoint types with different timeout characteristics"""
    ISSUE_LIST = "issue_list"
    ISSUE_VIEW = "issue_view" 
    ISSUE_CREATE = "issue_create"
    ISSUE_EDIT = "issue_edit"
    ISSUE_COMMENT = "issue_comment"
    PR_CREATE = "pr_create"
    PR_LIST = "pr_list"
    REPO_INFO = "repo_info"
    SEARCH = "search"
    BULK_OPERATIONS = "bulk_operations"

class TimeoutStrategy(Enum):
    """Timeout strategy types"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    PROGRESSIVE = "progressive"
    ENDPOINT_BASED = "endpoint_based"

@dataclass
class TimeoutConfig:
    """Configuration for timeout handling"""
    base_timeout: float = 30.0
    max_timeout: float = 300.0
    min_timeout: float = 5.0
    progressive_multiplier: float = 1.5
    adaptive_percentile: float = 95.0
    sample_window: int = 100
    strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE

@dataclass
class RequestMetrics:
    """Metrics for tracking request performance"""
    endpoint: GitHubEndpoint
    duration: float
    success: bool
    timestamp: datetime
    timeout_used: float
    retry_count: int = 0
    error_type: Optional[str] = None

@dataclass
class EndpointProfile:
    """Performance profile for a specific endpoint"""
    endpoint: GitHubEndpoint
    response_times: deque
    success_rate: float
    avg_response_time: float
    p95_response_time: float
    failure_count: int
    total_requests: int
    last_updated: datetime
    recommended_timeout: float

class GitHubTimeoutManager:
    """
    Advanced timeout management for GitHub API operations with:
    - Adaptive timeout configuration based on endpoint performance
    - Progressive timeout escalation for retries
    - Endpoint-specific timeout profiles
    - Integration with circuit breaker patterns
    """
    
    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()
        self.endpoint_profiles: Dict[GitHubEndpoint, EndpointProfile] = {}
        self.circuit_breakers: Dict[GitHubEndpoint, CircuitBreaker] = {}
        self.metrics_history: deque = deque(maxlen=10000)
        self.lock = threading.RLock()
        
        # Initialize endpoint profiles
        self._initialize_endpoint_profiles()
        
        # Metrics persistence
        self.metrics_file = Path("knowledge/metrics/github_timeout_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_historical_metrics()
        
        # Background metrics analyzer
        self._start_metrics_analyzer()
        
    def _initialize_endpoint_profiles(self):
        """Initialize default profiles for all endpoint types"""
        default_timeouts = {
            GitHubEndpoint.ISSUE_LIST: 45.0,
            GitHubEndpoint.ISSUE_VIEW: 20.0,
            GitHubEndpoint.ISSUE_CREATE: 60.0,
            GitHubEndpoint.ISSUE_EDIT: 30.0,
            GitHubEndpoint.ISSUE_COMMENT: 25.0,
            GitHubEndpoint.PR_CREATE: 90.0,
            GitHubEndpoint.PR_LIST: 45.0,
            GitHubEndpoint.REPO_INFO: 15.0,
            GitHubEndpoint.SEARCH: 60.0,
            GitHubEndpoint.BULK_OPERATIONS: 180.0
        }
        
        for endpoint, default_timeout in default_timeouts.items():
            self.endpoint_profiles[endpoint] = EndpointProfile(
                endpoint=endpoint,
                response_times=deque(maxlen=self.config.sample_window),
                success_rate=1.0,
                avg_response_time=default_timeout * 0.7,
                p95_response_time=default_timeout,
                failure_count=0,
                total_requests=0,
                last_updated=datetime.now(),
                recommended_timeout=default_timeout
            )
            
            # Initialize circuit breaker for each endpoint
            self.circuit_breakers[endpoint] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60
            )
            
        logger.info(f"Initialized {len(self.endpoint_profiles)} endpoint profiles")
    
    def get_timeout(self, endpoint: GitHubEndpoint, retry_count: int = 0) -> float:
        """
        Get the appropriate timeout for an endpoint and retry attempt.
        
        Args:
            endpoint: The GitHub endpoint being called
            retry_count: Number of retry attempts (0 for first attempt)
            
        Returns:
            Timeout value in seconds
        """
        with self.lock:
            profile = self.endpoint_profiles.get(endpoint)
            if not profile:
                logger.warning(f"No profile found for endpoint {endpoint}, using default")
                return self.config.base_timeout
            
            if self.config.strategy == TimeoutStrategy.FIXED:
                base_timeout = self.config.base_timeout
            elif self.config.strategy == TimeoutStrategy.ADAPTIVE:
                base_timeout = profile.recommended_timeout
            elif self.config.strategy == TimeoutStrategy.ENDPOINT_BASED:
                base_timeout = profile.p95_response_time * 1.2  # 20% buffer
            else:  # PROGRESSIVE
                base_timeout = profile.recommended_timeout
            
            # Apply progressive multiplier for retries (except FIXED strategy)
            if retry_count > 0 and self.config.strategy != TimeoutStrategy.FIXED:
                timeout = base_timeout * (self.config.progressive_multiplier ** retry_count)
            else:
                timeout = base_timeout
            
            # Clamp to configured limits
            timeout = max(self.config.min_timeout, min(timeout, self.config.max_timeout))
            
            logger.debug(f"Timeout for {endpoint.value} (retry {retry_count}): {timeout:.2f}s")
            return timeout
    
    def can_attempt_request(self, endpoint: GitHubEndpoint) -> Tuple[bool, str]:
        """
        Check if a request can be attempted based on circuit breaker state.
        
        Args:
            endpoint: The GitHub endpoint to check
            
        Returns:
            Tuple of (can_attempt, reason)
        """
        circuit_breaker = self.circuit_breakers.get(endpoint)
        if not circuit_breaker:
            return True, "No circuit breaker configured"
        
        if circuit_breaker.can_execute():
            return True, "Circuit breaker allows execution"
        else:
            return False, f"Circuit breaker is {circuit_breaker.state} for endpoint {endpoint.value}"
    
    def record_request_metrics(self, endpoint: GitHubEndpoint, duration: float, 
                             success: bool, timeout_used: float, retry_count: int = 0, 
                             error_type: Optional[str] = None):
        """
        Record metrics for a completed request.
        
        Args:
            endpoint: The GitHub endpoint that was called
            duration: Request duration in seconds
            success: Whether the request succeeded
            timeout_used: The timeout value that was used
            retry_count: Number of retry attempts
            error_type: Type of error if request failed
        """
        metrics = RequestMetrics(
            endpoint=endpoint,
            duration=duration,
            success=success,
            timestamp=datetime.now(),
            timeout_used=timeout_used,
            retry_count=retry_count,
            error_type=error_type
        )
        
        with self.lock:
            # Add to history
            self.metrics_history.append(metrics)
            
            # Update endpoint profile
            self._update_endpoint_profile(metrics)
            
            # Update circuit breaker
            circuit_breaker = self.circuit_breakers.get(endpoint)
            if circuit_breaker:
                if success:
                    circuit_breaker.record_success()
                else:
                    circuit_breaker.record_failure()
            
            # Persist metrics periodically
            if len(self.metrics_history) % 50 == 0:
                self._persist_metrics()
        
        logger.debug(f"Recorded metrics for {endpoint.value}: {duration:.2f}s, success={success}")
    
    def _update_endpoint_profile(self, metrics: RequestMetrics):
        """Update endpoint profile with new metrics"""
        profile = self.endpoint_profiles.get(metrics.endpoint)
        if not profile:
            return
        
        # Add response time to history
        profile.response_times.append(metrics.duration)
        profile.total_requests += 1
        profile.last_updated = datetime.now()
        
        if not metrics.success:
            profile.failure_count += 1
        
        # Recalculate statistics
        if profile.response_times:
            profile.avg_response_time = statistics.mean(profile.response_times)
            profile.p95_response_time = statistics.quantiles(
                profile.response_times, n=20
            )[-1] if len(profile.response_times) >= 10 else profile.avg_response_time * 1.2
            
            # Update success rate
            success_count = profile.total_requests - profile.failure_count
            profile.success_rate = success_count / profile.total_requests
            
            # Calculate recommended timeout based on strategy
            if self.config.strategy == TimeoutStrategy.ADAPTIVE:
                # Use P95 + 20% buffer, but consider recent failures
                base_timeout = profile.p95_response_time * 1.2
                
                # Increase timeout if success rate is low
                if profile.success_rate < 0.9:
                    failure_penalty = (1.0 - profile.success_rate) * 2.0
                    base_timeout *= (1.0 + failure_penalty)
                
                profile.recommended_timeout = min(base_timeout, self.config.max_timeout)
            else:
                profile.recommended_timeout = profile.p95_response_time * 1.2
    
    def get_endpoint_stats(self, endpoint: GitHubEndpoint) -> Dict[str, Any]:
        """Get detailed statistics for an endpoint"""
        with self.lock:
            profile = self.endpoint_profiles.get(endpoint)
            if not profile:
                return {"error": f"No profile found for {endpoint.value}"}
            
            circuit_breaker = self.circuit_breakers.get(endpoint)
            
            return {
                "endpoint": endpoint.value,
                "total_requests": profile.total_requests,
                "success_rate": profile.success_rate,
                "failure_count": profile.failure_count,
                "avg_response_time": profile.avg_response_time,
                "p95_response_time": profile.p95_response_time,
                "recommended_timeout": profile.recommended_timeout,
                "last_updated": profile.last_updated.isoformat(),
                "circuit_breaker_state": circuit_breaker.state if circuit_breaker else "none",
                "sample_count": len(profile.response_times)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all endpoints"""
        with self.lock:
            stats = {
                "total_requests": len(self.metrics_history),
                "configuration": asdict(self.config),
                "endpoints": {}
            }
            
            for endpoint in GitHubEndpoint:
                stats["endpoints"][endpoint.value] = self.get_endpoint_stats(endpoint)
            
            # Overall statistics
            if self.metrics_history:
                recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 requests
                success_count = sum(1 for m in recent_metrics if m.success)
                stats["overall_success_rate"] = success_count / len(recent_metrics)
                stats["avg_duration"] = statistics.mean(m.duration for m in recent_metrics)
            
            return stats
    
    def optimize_timeouts(self):
        """
        Analyze metrics and optimize timeout configurations.
        Called periodically by the metrics analyzer.
        """
        with self.lock:
            optimizations = []
            
            for endpoint, profile in self.endpoint_profiles.items():
                if profile.total_requests < 10:
                    continue  # Need more data
                
                old_timeout = profile.recommended_timeout
                
                # Calculate new timeout based on recent performance
                if profile.success_rate > 0.95:
                    # High success rate - can potentially reduce timeout
                    new_timeout = profile.p95_response_time * 1.1  # Reduce buffer
                elif profile.success_rate < 0.85:
                    # Low success rate - increase timeout
                    new_timeout = profile.p95_response_time * 1.5  # Increase buffer
                else:
                    # Moderate success rate - standard buffer
                    new_timeout = profile.p95_response_time * 1.2
                
                # Smart clamping: don't force good performing endpoints above their current timeout
                if profile.success_rate > 0.95 and new_timeout < self.config.min_timeout:
                    # For excellent performance, keep the better of calculated timeout or current recommended
                    new_timeout = min(profile.p95_response_time * 1.1, old_timeout)
                else:
                    # Standard clamping for other cases
                    new_timeout = max(self.config.min_timeout, min(new_timeout, self.config.max_timeout))
                
                # Different thresholds for different scenarios
                threshold = 0.1 if profile.success_rate > 0.95 else 1.0  # Lower threshold for good performance
                if abs(new_timeout - old_timeout) > threshold:  # Significant change
                    profile.recommended_timeout = new_timeout
                    optimizations.append({
                        "endpoint": endpoint.value,
                        "old_timeout": old_timeout,
                        "new_timeout": new_timeout,
                        "success_rate": profile.success_rate,
                        "sample_size": len(profile.response_times)
                    })
            
            if optimizations:
                logger.info(f"Optimized timeouts for {len(optimizations)} endpoints")
                for opt in optimizations:
                    logger.debug(f"Timeout optimization: {opt}")
            
            return optimizations
    
    def _persist_metrics(self):
        """Persist metrics to disk for analysis"""
        try:
            recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 requests
            serializable_metrics = []
            
            for m in recent_metrics:
                serializable_metrics.append({
                    "endpoint": m.endpoint.value,
                    "duration": m.duration,
                    "success": m.success,
                    "timestamp": m.timestamp.isoformat(),
                    "timeout_used": m.timeout_used,
                    "retry_count": m.retry_count,
                    "error_type": m.error_type
                })
            
            # Serialize config with enum values
            config_dict = asdict(self.config)
            config_dict["strategy"] = self.config.strategy.value  # Convert enum to value
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "config": config_dict,
                "metrics": serializable_metrics,
                "endpoint_stats": {
                    endpoint.value: self.get_endpoint_stats(endpoint)
                    for endpoint in GitHubEndpoint
                }
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def _load_historical_metrics(self):
        """Load historical metrics from disk"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Restore metrics
                for m_data in data.get("metrics", [])[-100:]:  # Last 100 for startup
                    try:
                        metrics = RequestMetrics(
                            endpoint=GitHubEndpoint(m_data["endpoint"]),
                            duration=m_data["duration"],
                            success=m_data["success"],
                            timestamp=datetime.fromisoformat(m_data["timestamp"]),
                            timeout_used=m_data["timeout_used"],
                            retry_count=m_data.get("retry_count", 0),
                            error_type=m_data.get("error_type")
                        )
                        self.metrics_history.append(metrics)
                        self._update_endpoint_profile(metrics)
                    except Exception as e:
                        logger.warning(f"Failed to restore metric: {e}")
                
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
                
        except Exception as e:
            logger.warning(f"Failed to load historical metrics: {e}")
    
    def _start_metrics_analyzer(self):
        """Start background thread for metrics analysis"""
        def analyzer_loop():
            while True:
                try:
                    time.sleep(300)  # Analyze every 5 minutes
                    self.optimize_timeouts()
                    if len(self.metrics_history) % 500 == 0:
                        self._persist_metrics()
                except Exception as e:
                    logger.error(f"Metrics analyzer error: {e}")
        
        thread = threading.Thread(target=analyzer_loop, daemon=True)
        thread.start()
        logger.info("Started background metrics analyzer")

def create_timeout_manager(config: Optional[TimeoutConfig] = None) -> GitHubTimeoutManager:
    """Factory function to create a timeout manager instance"""
    return GitHubTimeoutManager(config)

# Global timeout manager instance (singleton pattern)
_timeout_manager = None

def get_timeout_manager() -> GitHubTimeoutManager:
    """Get the global timeout manager instance"""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = create_timeout_manager()
    return _timeout_manager