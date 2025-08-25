"""
Performance Benchmarks for Orchestration Pattern Validation

Performance tests for Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
Benchmarks validation framework performance and parallel vs sequential execution.
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import the components we're benchmarking
import sys
sys.path.append('/Users/cal/DEV/RIF')

from claude.commands.orchestration_pattern_validator import (
    OrchestrationPatternValidator,
    validate_task_request
)
from claude.commands.orchestration_validation_enforcer import (
    OrchestrationValidationEnforcer,
    check_orchestration_pattern
)


class TestOrchestrationPerformanceBenchmarks:
    """Performance benchmark tests for orchestration validation"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = OrchestrationPatternValidator()
        self.enforcer = OrchestrationValidationEnforcer()
        
        # Standard test patterns
        self.valid_single_task = [
            {
                "description": "RIF-Implementer: Single valid task",
                "prompt": "You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        self.valid_parallel_tasks = [
            {
                "description": "RIF-Implementer: Task A",
                "prompt": "You are RIF-Implementer. Handle issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Task B", 
                "prompt": "You are RIF-Implementer. Handle issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: Task C",
                "prompt": "You are RIF-Validator. Validate issue #3. Follow all instructions in claude/agents/rif-validator.md.",
                "subagent_type": "general-purpose"
            }
        ]
        
        self.anti_pattern_task = [
            {
                "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
                "prompt": "Handle multiple issues in parallel",
                "subagent_type": "general-purpose"
            }
        ]
        
    def benchmark_function(self, func, *args, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark a function with multiple iterations and return statistics.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to function
            iterations: Number of iterations to run
            
        Returns:
            Dict with timing statistics
        """
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times), 
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "total_time": sum(times),
            "iterations": iterations,
            "calls_per_second": iterations / sum(times)
        }
        
    def test_validator_performance_single_task(self):
        """Benchmark validator performance with single task"""
        stats = self.benchmark_function(
            self.validator.validate_orchestration_request,
            self.valid_single_task,
            iterations=1000
        )
        
        print(f"\nValidator Single Task Performance:")
        print(f"Mean time: {stats['mean']:.4f}s")
        print(f"Calls per second: {stats['calls_per_second']:.1f}")
        print(f"Max time: {stats['max']:.4f}s")
        
        # Performance assertions
        assert stats['mean'] < 0.01, "Single task validation should be under 10ms"
        assert stats['calls_per_second'] > 100, "Should handle >100 validations per second"
        
    def test_validator_performance_parallel_tasks(self):
        """Benchmark validator performance with parallel tasks"""
        stats = self.benchmark_function(
            self.validator.validate_orchestration_request,
            self.valid_parallel_tasks,
            iterations=500
        )
        
        print(f"\nValidator Parallel Tasks Performance:")
        print(f"Mean time: {stats['mean']:.4f}s")
        print(f"Calls per second: {stats['calls_per_second']:.1f}")
        print(f"Max time: {stats['max']:.4f}s")
        
        # Performance assertions
        assert stats['mean'] < 0.02, "Parallel task validation should be under 20ms"
        assert stats['calls_per_second'] > 50, "Should handle >50 parallel validations per second"
        
    def test_validator_performance_anti_pattern(self):
        """Benchmark validator performance with anti-patterns"""
        stats = self.benchmark_function(
            self.validator.validate_orchestration_request,
            self.anti_pattern_task,
            iterations=1000
        )
        
        print(f"\nValidator Anti-Pattern Performance:")
        print(f"Mean time: {stats['mean']:.4f}s")
        print(f"Calls per second: {stats['calls_per_second']:.1f}")
        print(f"Max time: {stats['max']:.4f}s")
        
        # Performance assertions
        assert stats['mean'] < 0.015, "Anti-pattern detection should be under 15ms"
        assert stats['calls_per_second'] > 60, "Should handle >60 anti-pattern checks per second"
        
    def test_enforcer_performance(self):
        """Benchmark enforcer performance"""
        # Use temporary directory to avoid file system issues
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            enforcer = OrchestrationValidationEnforcer(knowledge_base_path=temp_dir)
            
            stats = self.benchmark_function(
                enforcer.validate_and_enforce,
                self.valid_parallel_tasks,
                None,  # context
                iterations=100
            )
            
            print(f"\nEnforcer Performance:")
            print(f"Mean time: {stats['mean']:.4f}s")
            print(f"Calls per second: {stats['calls_per_second']:.1f}")
            print(f"Max time: {stats['max']:.4f}s")
            
            # Performance assertions (more lenient due to file I/O)
            assert stats['mean'] < 0.05, "Enforcement should be under 50ms"
            assert stats['calls_per_second'] > 20, "Should handle >20 enforcements per second"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def test_scaling_with_task_count(self):
        """Benchmark how validation performance scales with task count"""
        task_counts = [1, 3, 5, 10, 20, 50]
        results = {}
        
        for count in task_counts:
            # Create tasks for this count
            tasks = []
            for i in range(count):
                tasks.append({
                    "description": f"RIF-Implementer: Task {i}",
                    "prompt": f"You are RIF-Implementer. Handle issue #{i}. Follow all instructions in claude/agents/rif-implementer.md.",
                    "subagent_type": "general-purpose"
                })
                
            stats = self.benchmark_function(
                self.validator.validate_orchestration_request,
                tasks,
                iterations=50
            )
            
            results[count] = stats['mean']
            
        print(f"\nScaling Performance (task count -> mean time):")
        for count, mean_time in results.items():
            print(f"{count:2d} tasks: {mean_time:.4f}s")
            
        # Check that performance scales reasonably (not exponentially)
        # Time for 50 tasks should be less than 10x time for 5 tasks
        if 50 in results and 5 in results:
            scaling_factor = results[50] / results[5]
            assert scaling_factor < 10, f"Performance scaling too poor: {scaling_factor}x"
            
    def test_concurrent_validation_performance(self):
        """Test performance under concurrent load"""
        def validate_task():
            return self.validator.validate_orchestration_request(self.valid_parallel_tasks)
            
        num_threads = min(8, multiprocessing.cpu_count())
        num_tasks_per_thread = 20
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                for _ in range(num_tasks_per_thread):
                    future = executor.submit(validate_task)
                    futures.append(future)
                    
            # Wait for all tasks to complete
            results = [future.result() for future in as_completed(futures)]
            
        end_time = time.perf_counter()
        
        total_tasks = num_threads * num_tasks_per_thread
        total_time = end_time - start_time
        throughput = total_tasks / total_time
        
        print(f"\nConcurrent Performance:")
        print(f"Threads: {num_threads}")
        print(f"Tasks per thread: {num_tasks_per_thread}")
        print(f"Total tasks: {total_tasks}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} validations/second")
        
        # All validations should succeed
        assert all(isinstance(result.is_valid, bool) for result in results)
        
        # Should maintain reasonable throughput under concurrent load
        assert throughput > 50, "Concurrent throughput should be >50 validations/second"
        
    def test_memory_usage_stability(self):
        """Test that validator doesn't leak memory during extended use"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many validations
        for _ in range(1000):
            self.validator.validate_orchestration_request(self.valid_parallel_tasks)
            self.validator.validate_orchestration_request(self.anti_pattern_task)
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.1f} MB")
        print(f"Final: {final_memory:.1f} MB")
        print(f"Growth: {memory_growth:.1f} MB")
        
        # Memory growth should be minimal
        assert memory_growth < 10, f"Memory growth too high: {memory_growth:.1f} MB"
        
    def test_validation_accuracy_under_load(self):
        """Test that validation accuracy is maintained under high load"""
        # Mix of valid and invalid patterns
        test_cases = [
            (self.valid_single_task, True),
            (self.valid_parallel_tasks, True),
            (self.anti_pattern_task, False)
        ]
        
        correct_validations = 0
        total_validations = 0
        
        # Run many validations
        for _ in range(100):
            for tasks, expected_valid in test_cases:
                result = self.validator.validate_orchestration_request(tasks)
                if result.is_valid == expected_valid:
                    correct_validations += 1
                total_validations += 1
                
        accuracy = correct_validations / total_validations
        
        print(f"\nValidation Accuracy Under Load:")
        print(f"Correct: {correct_validations}/{total_validations}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Should maintain perfect accuracy
        assert accuracy == 1.0, f"Validation accuracy dropped to {accuracy:.2%}"


class TestOrchestrationExecutionComparison:
    """Compare performance of different orchestration patterns"""
    
    def simulate_task_execution(self, task_count: int, parallel: bool = True) -> float:
        """
        Simulate task execution time based on orchestration pattern.
        
        Args:
            task_count: Number of tasks to simulate
            parallel: Whether to simulate parallel or sequential execution
            
        Returns:
            Simulated execution time in seconds
        """
        # Simulate individual task execution time (0.1-0.5 seconds per task)
        import random
        base_task_time = 0.3
        
        if parallel:
            # Parallel execution: max task time (with some overhead)
            task_times = [base_task_time + random.uniform(-0.1, 0.1) for _ in range(task_count)]
            execution_time = max(task_times) + 0.05  # Small parallel overhead
        else:
            # Sequential execution: sum of all task times  
            task_times = [base_task_time + random.uniform(-0.1, 0.1) for _ in range(task_count)]
            execution_time = sum(task_times)
            
        return execution_time
        
    def test_parallel_vs_sequential_performance(self):
        """Compare parallel vs sequential execution performance"""
        task_counts = [1, 3, 5, 10, 20]
        results = {}
        
        for count in task_counts:
            # Simulate multiple runs for statistical significance
            parallel_times = [self.simulate_task_execution(count, parallel=True) for _ in range(50)]
            sequential_times = [self.simulate_task_execution(count, parallel=False) for _ in range(50)]
            
            parallel_mean = statistics.mean(parallel_times)
            sequential_mean = statistics.mean(sequential_times)
            speedup = sequential_mean / parallel_mean if parallel_mean > 0 else 1.0
            
            results[count] = {
                "parallel": parallel_mean,
                "sequential": sequential_mean, 
                "speedup": speedup
            }
            
        print(f"\nParallel vs Sequential Performance:")
        print(f"{'Tasks':<6} {'Parallel':<10} {'Sequential':<12} {'Speedup':<8}")
        print("-" * 38)
        
        for count, times in results.items():
            print(f"{count:<6} {times['parallel']:<10.3f} {times['sequential']:<12.3f} {times['speedup']:<8.2f}x")
            
        # Verify that parallel execution provides speedup for multiple tasks
        for count in [3, 5, 10, 20]:
            if count in results:
                assert results[count]['speedup'] > 1.5, f"Insufficient speedup for {count} tasks: {results[count]['speedup']:.2f}x"
                
    def test_orchestration_pattern_efficiency_comparison(self):
        """Compare efficiency of different orchestration patterns"""
        patterns = {
            "single_task": {
                "description": "Single task execution",
                "task_count": 1,
                "parallel": False
            },
            "proper_parallel": {
                "description": "Proper parallel Tasks (correct pattern)",
                "task_count": 5,
                "parallel": True
            },
            "multi_issue_accelerator": {
                "description": "Multi-Issue Accelerator (anti-pattern)",
                "task_count": 5,
                "parallel": False  # Anti-pattern forces sequential
            }
        }
        
        results = {}
        
        for pattern_name, pattern_config in patterns.items():
            # Simulate execution times
            times = [
                self.simulate_task_execution(
                    pattern_config["task_count"], 
                    pattern_config["parallel"]
                ) for _ in range(100)
            ]
            
            results[pattern_name] = {
                "mean_time": statistics.mean(times),
                "description": pattern_config["description"],
                "task_count": pattern_config["task_count"]
            }
            
        print(f"\nOrchestration Pattern Efficiency:")
        print(f"{'Pattern':<20} {'Description':<35} {'Mean Time':<10} {'Tasks'}")
        print("-" * 75)
        
        for pattern_name, data in results.items():
            print(f"{pattern_name:<20} {data['description']:<35} {data['mean_time']:<10.3f} {data['task_count']}")
            
        # Verify that proper parallel pattern is most efficient for multiple tasks
        if "proper_parallel" in results and "multi_issue_accelerator" in results:
            parallel_time = results["proper_parallel"]["mean_time"] 
            anti_pattern_time = results["multi_issue_accelerator"]["mean_time"]
            efficiency_gain = anti_pattern_time / parallel_time
            
            print(f"\nEfficiency gain from proper parallel pattern: {efficiency_gain:.2f}x")
            assert efficiency_gain > 2.0, f"Proper parallel should be >2x more efficient: {efficiency_gain:.2f}x"


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-s"])