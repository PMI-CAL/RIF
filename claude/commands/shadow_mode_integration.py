#!/usr/bin/env python3
"""
Shadow Mode Integration for RIF Monitoring
Provides monitoring hooks for shadow mode testing from issue #37
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import logging
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ShadowModeResult:
    """Results from a shadow mode comparison"""
    operation: str
    primary_result: Any
    shadow_result: Any
    primary_latency: float
    shadow_latency: float
    timestamp: datetime
    accuracy_score: float = 0.0
    differences: List[str] = None
    
    def __post_init__(self):
        if self.differences is None:
            self.differences = []

class ShadowModeMonitor:
    """Monitoring integration for shadow mode testing"""
    
    def __init__(self, system_monitor=None):
        self.system_monitor = system_monitor
        self.logger = self._setup_logging()
        self.results_buffer: List[ShadowModeResult] = []
        self.comparison_handlers: Dict[str, Callable] = {}
        
        # Statistics tracking
        self.total_comparisons = 0
        self.accuracy_sum = 0.0
        self.performance_deltas = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for shadow mode monitoring"""
        logger = logging.getLogger("shadow_mode_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - SHADOW - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def register_comparison_handler(self, operation_type: str, handler: Callable):
        """Register a custom comparison handler for specific operation types"""
        self.comparison_handlers[operation_type] = handler
        self.logger.info(f"Registered comparison handler for: {operation_type}")
        
    @contextmanager
    def shadow_comparison(self, operation_name: str, primary_func: Callable, shadow_func: Callable):
        """Context manager for running shadow mode comparisons"""
        
        start_time = time.time()
        
        try:
            # Execute primary system
            primary_start = time.time()
            primary_result = primary_func()
            primary_latency = (time.time() - primary_start) * 1000  # ms
            
            # Execute shadow system
            shadow_start = time.time()
            shadow_result = shadow_func()
            shadow_latency = (time.time() - shadow_start) * 1000  # ms
            
            # Compare results
            comparison_result = ShadowModeResult(
                operation=operation_name,
                primary_result=primary_result,
                shadow_result=shadow_result,
                primary_latency=primary_latency,
                shadow_latency=shadow_latency,
                timestamp=datetime.now()
            )
            
            # Calculate accuracy and differences
            self._analyze_comparison(comparison_result)
            
            # Store result
            self.results_buffer.append(comparison_result)
            
            # Report to system monitor if available
            if self.system_monitor:
                self._report_to_monitor(comparison_result)
                
            # Log the comparison
            self.logger.info(
                f"Shadow comparison: {operation_name} | "
                f"Accuracy: {comparison_result.accuracy_score:.1f}% | "
                f"Latency: Primary {primary_latency:.1f}ms vs Shadow {shadow_latency:.1f}ms"
            )
            
            yield comparison_result
            
        except Exception as e:
            self.logger.error(f"Shadow comparison failed for {operation_name}: {e}")
            raise
            
        finally:
            total_time = (time.time() - start_time) * 1000
            if self.system_monitor:
                self.system_monitor.collector.track_latency(
                    "shadow_mode_comparison", 
                    total_time, 
                    {"operation": operation_name}
                )
                
    def _analyze_comparison(self, result: ShadowModeResult):
        """Analyze the comparison between primary and shadow results"""
        
        operation_type = result.operation.split('_')[0] if '_' in result.operation else result.operation
        
        # Use custom handler if available
        if operation_type in self.comparison_handlers:
            try:
                handler_result = self.comparison_handlers[operation_type](
                    result.primary_result, 
                    result.shadow_result
                )
                result.accuracy_score = handler_result.get("accuracy", 0.0)
                result.differences = handler_result.get("differences", [])
                return
            except Exception as e:
                self.logger.warning(f"Custom comparison handler failed: {e}")
                
        # Default comparison logic
        if result.primary_result is None and result.shadow_result is None:
            result.accuracy_score = 100.0
        elif result.primary_result is None or result.shadow_result is None:
            result.accuracy_score = 0.0
            result.differences.append("One result is None")
        elif isinstance(result.primary_result, (str, int, float, bool)):
            # Simple value comparison
            if result.primary_result == result.shadow_result:
                result.accuracy_score = 100.0
            else:
                result.accuracy_score = 0.0
                result.differences.append(
                    f"Values differ: {result.primary_result} vs {result.shadow_result}"
                )
        elif isinstance(result.primary_result, (dict, list)):
            # Complex structure comparison
            result.accuracy_score = self._compare_structures(
                result.primary_result, 
                result.shadow_result, 
                result.differences
            )
        else:
            # String representation comparison as fallback
            primary_str = str(result.primary_result)
            shadow_str = str(result.shadow_result)
            
            if primary_str == shadow_str:
                result.accuracy_score = 100.0
            else:
                result.accuracy_score = self._calculate_string_similarity(primary_str, shadow_str)
                if result.accuracy_score < 100:
                    result.differences.append("String representations differ")
                    
    def _compare_structures(self, primary: Any, shadow: Any, differences: List[str]) -> float:
        """Compare complex data structures"""
        try:
            if type(primary) != type(shadow):
                differences.append(f"Type mismatch: {type(primary)} vs {type(shadow)}")
                return 0.0
                
            if isinstance(primary, dict):
                return self._compare_dicts(primary, shadow, differences)
            elif isinstance(primary, list):
                return self._compare_lists(primary, shadow, differences)
            else:
                return 100.0 if primary == shadow else 0.0
                
        except Exception as e:
            differences.append(f"Comparison error: {e}")
            return 0.0
            
    def _compare_dicts(self, dict1: dict, dict2: dict, differences: List[str]) -> float:
        """Compare two dictionaries"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        if not all_keys:
            return 100.0
            
        matching_keys = 0
        for key in all_keys:
            if key in dict1 and key in dict2:
                if dict1[key] == dict2[key]:
                    matching_keys += 1
                else:
                    differences.append(f"Key '{key}': {dict1[key]} vs {dict2[key]}")
            else:
                missing_from = "shadow" if key in dict1 else "primary"
                differences.append(f"Key '{key}' missing from {missing_from}")
                
        return (matching_keys / len(all_keys)) * 100
        
    def _compare_lists(self, list1: list, list2: list, differences: List[str]) -> float:
        """Compare two lists"""
        if len(list1) != len(list2):
            differences.append(f"Length mismatch: {len(list1)} vs {len(list2)}")
            return 0.0
            
        if not list1:
            return 100.0
            
        matching_items = sum(1 for i, (a, b) in enumerate(zip(list1, list2)) if a == b)
        return (matching_items / len(list1)) * 100
        
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity percentage"""
        if not str1 and not str2:
            return 100.0
        if not str1 or not str2:
            return 0.0
            
        # Simple character-level similarity
        max_len = max(len(str1), len(str2))
        matching_chars = sum(1 for i, (a, b) in enumerate(zip(str1, str2)) if a == b)
        
        return (matching_chars / max_len) * 100
        
    def _report_to_monitor(self, result: ShadowModeResult):
        """Report comparison results to the system monitor"""
        if not self.system_monitor:
            return
            
        # Update statistics
        self.total_comparisons += 1
        self.accuracy_sum += result.accuracy_score
        
        performance_delta = abs(result.shadow_latency - result.primary_latency)
        performance_delta_percent = (performance_delta / result.primary_latency) * 100 if result.primary_latency > 0 else 0
        self.performance_deltas.append(performance_delta_percent)
        
        # Report metrics
        self.system_monitor.collector.track_shadow_mode_comparison(
            result.accuracy_score, 
            performance_delta_percent
        )
        
        # Track individual operation latencies
        self.system_monitor.collector.track_latency(
            f"shadow_primary_{result.operation}", 
            result.primary_latency, 
            {"system": "primary", "operation": result.operation}
        )
        
        self.system_monitor.collector.track_latency(
            f"shadow_secondary_{result.operation}", 
            result.shadow_latency, 
            {"system": "shadow", "operation": result.operation}
        )
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for shadow mode testing"""
        if not self.results_buffer:
            return {"error": "No shadow mode results available"}
            
        avg_accuracy = self.accuracy_sum / self.total_comparisons if self.total_comparisons > 0 else 0
        avg_performance_delta = sum(self.performance_deltas) / len(self.performance_deltas) if self.performance_deltas else 0
        
        # Recent results analysis
        recent_results = self.results_buffer[-10:] if len(self.results_buffer) >= 10 else self.results_buffer
        recent_accuracy = sum(r.accuracy_score for r in recent_results) / len(recent_results)
        
        # Count issues
        low_accuracy_count = len([r for r in self.results_buffer if r.accuracy_score < 90])
        high_latency_diff_count = len([r for r in self.results_buffer 
                                     if abs(r.shadow_latency - r.primary_latency) > r.primary_latency * 0.5])
        
        return {
            "total_comparisons": self.total_comparisons,
            "average_accuracy": avg_accuracy,
            "recent_accuracy": recent_accuracy,
            "average_performance_delta": avg_performance_delta,
            "low_accuracy_issues": low_accuracy_count,
            "high_latency_issues": high_latency_diff_count,
            "last_comparison": self.results_buffer[-1].timestamp.isoformat() if self.results_buffer else None
        }
        
    def export_results(self, output_path: str):
        """Export shadow mode results to file"""
        results_data = []
        
        for result in self.results_buffer:
            results_data.append({
                "operation": result.operation,
                "timestamp": result.timestamp.isoformat(),
                "accuracy_score": result.accuracy_score,
                "primary_latency": result.primary_latency,
                "shadow_latency": result.shadow_latency,
                "differences": result.differences,
                "performance_delta": abs(result.shadow_latency - result.primary_latency)
            })
            
        export_data = {
            "summary": self.get_summary_stats(),
            "results": results_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Shadow mode results exported to: {output_path}")

# Integration with existing monitoring system
def create_shadow_mode_monitor(system_monitor=None):
    """Create a shadow mode monitor integrated with system monitoring"""
    from claude.commands.system_monitor import get_monitor
    
    if system_monitor is None:
        system_monitor = get_monitor()
        
    return ShadowModeMonitor(system_monitor)

# Decorator for easy shadow mode testing
def shadow_mode_test(operation_name: str, shadow_monitor: ShadowModeMonitor = None):
    """Decorator for shadow mode testing of functions"""
    def decorator(primary_func):
        def wrapper(*args, **kwargs):
            # Extract shadow function from kwargs if provided
            shadow_func = kwargs.pop('shadow_func', None)
            
            if shadow_func and shadow_monitor:
                with shadow_monitor.shadow_comparison(
                    operation_name, 
                    lambda: primary_func(*args, **kwargs),
                    lambda: shadow_func(*args, **kwargs)
                ) as result:
                    return result.primary_result
            else:
                # No shadow testing, just run primary
                return primary_func(*args, **kwargs)
                
        return wrapper
    return decorator

# Example usage and integration tests
if __name__ == "__main__":
    # Example of how to use shadow mode monitoring
    
    # Create monitor
    shadow_monitor = ShadowModeMonitor()
    
    # Example: Test knowledge query operations
    def old_knowledge_query(query: str) -> dict:
        """Simulate old knowledge system"""
        time.sleep(0.1)  # Simulate processing time
        return {"result": f"Old system result for: {query}", "confidence": 0.8}
        
    def new_knowledge_query(query: str) -> dict:
        """Simulate new knowledge system"""
        time.sleep(0.05)  # Simulate faster processing
        return {"result": f"New system result for: {query}", "confidence": 0.85}
        
    # Register custom comparison handler for knowledge queries
    def knowledge_comparison_handler(primary_result, shadow_result):
        """Custom handler for comparing knowledge query results"""
        if not isinstance(primary_result, dict) or not isinstance(shadow_result, dict):
            return {"accuracy": 0.0, "differences": ["Results not in expected format"]}
            
        differences = []
        accuracy = 100.0
        
        # Compare confidence scores
        primary_conf = primary_result.get("confidence", 0)
        shadow_conf = shadow_result.get("confidence", 0)
        
        if abs(primary_conf - shadow_conf) > 0.1:
            differences.append(f"Confidence differs significantly: {primary_conf} vs {shadow_conf}")
            accuracy -= 20
            
        # Compare result content (simplified)
        primary_text = primary_result.get("result", "")
        shadow_text = shadow_result.get("result", "")
        
        if "Old system" in primary_text and "New system" in shadow_text:
            # Expected difference in system identifier
            pass
        else:
            # Unexpected difference
            differences.append("Unexpected content difference")
            accuracy -= 30
            
        return {"accuracy": max(0, accuracy), "differences": differences}
        
    shadow_monitor.register_comparison_handler("knowledge", knowledge_comparison_handler)
    
    # Run some test comparisons
    test_queries = ["What is RIF?", "How does monitoring work?", "System status"]
    
    print("Running shadow mode comparisons...")
    
    for query in test_queries:
        with shadow_monitor.shadow_comparison(
            f"knowledge_query",
            lambda: old_knowledge_query(query),
            lambda: new_knowledge_query(query)
        ) as result:
            print(f"Query: '{query}' - Accuracy: {result.accuracy_score:.1f}%")
            
    # Print summary statistics
    stats = shadow_monitor.get_summary_stats()
    print(f"\nShadow Mode Summary:")
    print(f"Total Comparisons: {stats['total_comparisons']}")
    print(f"Average Accuracy: {stats['average_accuracy']:.1f}%")
    print(f"Performance Delta: {stats['average_performance_delta']:.1f}%")
    print(f"Issues Found: {stats['low_accuracy_issues']} accuracy, {stats['high_latency_issues']} performance")
    
    # Export results
    output_path = "/tmp/shadow_mode_results.json"
    shadow_monitor.export_results(output_path)
    print(f"Results exported to: {output_path}")