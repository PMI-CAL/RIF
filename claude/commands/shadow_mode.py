#!/usr/bin/env python3
"""
Shadow Mode Implementation for RIF Knowledge System Testing
Enables parallel processing of both legacy and LightRAG knowledge systems for comparison.

This module provides the core infrastructure for running the new LightRAG system
in shadow mode alongside the existing knowledge system, allowing safe testing
and comparison without impacting agent operations.
"""

import os
import sys
import json
import yaml
import time
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path

# Add knowledge interface to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'knowledge'))

try:
    from knowledge import get_knowledge_system, LightRAGKnowledgeAdapter, LIGHTRAG_AVAILABLE
    KNOWLEDGE_INTERFACE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INTERFACE_AVAILABLE = False
    LIGHTRAG_AVAILABLE = False

# Import LightRAG core
try:
    # Add lightrag path to sys.path
    lightrag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'lightrag')
    if lightrag_path not in sys.path:
        sys.path.append(lightrag_path)
    
    from core.lightrag_core import LightRAGCore
    LIGHTRAG_CORE_AVAILABLE = True
except ImportError as e:
    LightRAGCore = None
    LIGHTRAG_CORE_AVAILABLE = False


@dataclass
class OperationResult:
    """Result of a knowledge operation."""
    success: bool
    data: Any = None
    error: str = None
    duration_ms: float = 0
    metadata: Dict[str, Any] = None


@dataclass
class ComparisonResult:
    """Result of comparing two operation results."""
    primary_result: OperationResult
    shadow_result: OperationResult
    similar: bool = False
    similarity_score: float = 0.0
    differences: List[str] = None
    comparison_metadata: Dict[str, Any] = None


class LegacyKnowledgeSystem:
    """Legacy file-based knowledge system adapter."""
    
    def __init__(self, knowledge_path: str):
        self.knowledge_path = Path(knowledge_path)
        self.logger = logging.getLogger(f"{__name__}.LegacyKnowledgeSystem")
    
    def store_knowledge(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> OperationResult:
        """Store knowledge in legacy system."""
        start_time = time.time()
        
        try:
            # Determine file path based on collection
            collection_map = {
                'patterns': 'patterns',
                'decisions': 'decisions', 
                'code_snippets': 'learning',
                'issue_resolutions': 'issues'
            }
            
            folder = collection_map.get(collection, collection)
            folder_path = self.knowledge_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if no doc_id provided
            if not doc_id:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                doc_id = f"{collection}_{timestamp}_{content_hash}"
            
            # Store as JSON file
            file_path = folder_path / f"{doc_id}.json"
            data = {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat(),
                "collection": collection
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return OperationResult(
                success=True,
                data=doc_id,
                duration_ms=duration_ms,
                metadata={"file_path": str(file_path)}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Legacy store failed: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
    
    def retrieve_knowledge(self, query: str, collection: str = None, n_results: int = 5) -> OperationResult:
        """Retrieve knowledge from legacy system."""
        start_time = time.time()
        
        try:
            results = []
            
            # Determine collections to search
            if collection:
                collections = [collection]
            else:
                collections = ['patterns', 'decisions', 'code_snippets', 'issue_resolutions']
            
            # Simple text search in JSON files
            for coll in collections:
                collection_map = {
                    'patterns': 'patterns',
                    'decisions': 'decisions',
                    'code_snippets': 'learning', 
                    'issue_resolutions': 'issues'
                }
                
                folder = collection_map.get(coll, coll)
                folder_path = self.knowledge_path / folder
                
                if not folder_path.exists():
                    continue
                
                for file_path in folder_path.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Simple keyword matching
                        content_lower = data.get('content', '').lower()
                        query_lower = query.lower()
                        
                        if query_lower in content_lower:
                            # Calculate simple similarity score
                            similarity = len(query_lower) / len(content_lower) if content_lower else 0
                            
                            result = {
                                "collection": coll,
                                "id": data.get('id', file_path.stem),
                                "content": data.get('content', ''),
                                "metadata": data.get('metadata', {}),
                                "distance": 1.0 - similarity  # Convert to distance
                            }
                            results.append(result)
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to read {file_path}: {e}")
            
            # Sort by relevance (lower distance = more relevant)
            results.sort(key=lambda x: x.get('distance', 1.0))
            results = results[:n_results]
            
            duration_ms = (time.time() - start_time) * 1000
            
            return OperationResult(
                success=True,
                data=results,
                duration_ms=duration_ms,
                metadata={"results_count": len(results)}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Legacy retrieve failed: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )


class ShadowModeProcessor:
    """Main shadow mode processing system."""
    
    def __init__(self, config_path: str = None):
        """Initialize shadow mode processor."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                'config', 'shadow-mode.yaml'
            )
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize systems
        self._init_systems()
        
        # Metrics tracking
        self.metrics = {
            'operations': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'comparisons': 0
            },
            'performance': {
                'primary_avg_latency': 0.0,
                'shadow_avg_latency': 0.0,
                'comparison_avg_latency': 0.0
            },
            'differences': {
                'content_diffs': 0,
                'metadata_diffs': 0,
                'performance_diffs': 0
            }
        }
        
        # Thread pool for parallel execution
        max_workers = self.config['parallel_processing']['max_concurrent_operations']
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.logger.info("Shadow mode processor initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load shadow mode configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            # Fallback minimal config
            return {
                'shadow_mode': {'enabled': False},
                'logging': {'enabled': True, 'log_level': 'INFO'},
                'parallel_processing': {'timeout_ms': 5000, 'max_concurrent_operations': 2}
            }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        
        if not log_config.get('enabled', True):
            return
        
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file', True):
            log_file = log_config.get('log_file', 'shadow-mode.log')
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def _init_systems(self):
        """Initialize knowledge systems."""
        systems_config = self.config.get('systems', {})
        
        # Initialize legacy system
        legacy_config = systems_config.get('legacy', {})
        knowledge_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'knowledge'
        )
        self.legacy_system = LegacyKnowledgeSystem(knowledge_path)
        
        # Initialize LightRAG system
        self.lightrag_system = None
        if LIGHTRAG_CORE_AVAILABLE and systems_config.get('lightrag', {}).get('enabled', True):
            try:
                lightrag_config = systems_config.get('lightrag', {}).get('config', {})
                knowledge_path = lightrag_config.get('knowledge_path', self.legacy_system.knowledge_path)
                self.lightrag_system = LightRAGCore(knowledge_path)
                self.logger.info("LightRAG system initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize LightRAG: {e}")
                self.lightrag_system = None
        
        # Determine primary and shadow systems
        pp_config = self.config.get('parallel_processing', {})
        primary_name = pp_config.get('primary_system', 'legacy')
        shadow_name = pp_config.get('shadow_system', 'lightrag')
        
        self.primary_system = self.legacy_system if primary_name == 'legacy' else self.lightrag_system
        self.shadow_system = self.lightrag_system if shadow_name == 'lightrag' else self.legacy_system
        
        self.logger.info(f"Primary system: {primary_name}, Shadow system: {shadow_name}")
    
    def is_enabled(self) -> bool:
        """Check if shadow mode is enabled."""
        return self.config.get('shadow_mode', {}).get('enabled', False)
    
    def store_knowledge(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> Any:
        """Store knowledge with shadow mode processing."""
        if not self.is_enabled() or not self._should_shadow_operation('store_knowledge'):
            # Fall back to primary system only
            result = self._execute_store(self.primary_system, collection, content, metadata, doc_id)
            return result.data if result.success else None
        
        # Execute in shadow mode
        comparison = self._execute_parallel_store(collection, content, metadata, doc_id)
        
        # Log comparison if enabled
        if self.config.get('logging', {}).get('log_comparisons', True):
            self._log_comparison('store_knowledge', comparison)
        
        # Return primary result
        return comparison.primary_result.data if comparison.primary_result.success else None
    
    def retrieve_knowledge(self, query: str, collection: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve knowledge with shadow mode processing."""
        if not self.is_enabled() or not self._should_shadow_operation('retrieve_knowledge'):
            # Fall back to primary system only
            result = self._execute_retrieve(self.primary_system, query, collection, n_results)
            return result.data if result.success else []
        
        # Execute in shadow mode
        comparison = self._execute_parallel_retrieve(query, collection, n_results)
        
        # Log comparison if enabled
        if self.config.get('logging', {}).get('log_comparisons', True):
            self._log_comparison('retrieve_knowledge', comparison)
        
        # Return primary result
        return comparison.primary_result.data if comparison.primary_result.success else []
    
    def _should_shadow_operation(self, operation: str) -> bool:
        """Check if operation should be shadow tested."""
        operations_config = self.config.get('operations', {})
        op_config = operations_config.get(operation, {})
        return op_config.get('enabled', True)
    
    def _execute_store(self, system, collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> OperationResult:
        """Execute store operation on a system."""
        if system == self.legacy_system:
            return system.store_knowledge(collection, content, metadata, doc_id)
        elif system == self.lightrag_system and self.lightrag_system:
            start_time = time.time()
            try:
                result_id = system.store_knowledge(collection, content, metadata, doc_id)
                duration_ms = (time.time() - start_time) * 1000
                
                return OperationResult(
                    success=True,
                    data=result_id,
                    duration_ms=duration_ms
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return OperationResult(
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )
        else:
            return OperationResult(success=False, error="System not available")
    
    def _execute_retrieve(self, system, query: str, collection: str = None, n_results: int = 5) -> OperationResult:
        """Execute retrieve operation on a system."""
        if system == self.legacy_system:
            return system.retrieve_knowledge(query, collection, n_results)
        elif system == self.lightrag_system and self.lightrag_system:
            start_time = time.time()
            try:
                results = system.retrieve_knowledge(query, collection, n_results)
                duration_ms = (time.time() - start_time) * 1000
                
                return OperationResult(
                    success=True,
                    data=results,
                    duration_ms=duration_ms,
                    metadata={"results_count": len(results)}
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return OperationResult(
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )
        else:
            return OperationResult(success=False, error="System not available")
    
    def _execute_parallel_store(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> ComparisonResult:
        """Execute store operation in parallel on both systems."""
        timeout_ms = self.config['parallel_processing']['timeout_ms']
        timeout_seconds = timeout_ms / 1000.0
        
        # Submit both operations
        primary_future = self.executor.submit(self._execute_store, self.primary_system, collection, content, metadata, doc_id)
        shadow_future = self.executor.submit(self._execute_store, self.shadow_system, collection, content, metadata, doc_id)
        
        # Wait for primary result (blocking)
        try:
            primary_result = primary_future.result(timeout=timeout_seconds)
        except TimeoutError:
            primary_result = OperationResult(success=False, error="Primary system timeout")
        
        # Wait for shadow result (with timeout)
        try:
            shadow_result = shadow_future.result(timeout=timeout_seconds)
        except TimeoutError:
            shadow_result = OperationResult(success=False, error="Shadow system timeout")
        
        # Compare results
        comparison = self._compare_store_results(primary_result, shadow_result)
        
        # Update metrics
        self._update_metrics('store_knowledge', comparison)
        
        return comparison
    
    def _execute_parallel_retrieve(self, query: str, collection: str = None, n_results: int = 5) -> ComparisonResult:
        """Execute retrieve operation in parallel on both systems."""
        timeout_ms = self.config['parallel_processing']['timeout_ms']
        timeout_seconds = timeout_ms / 1000.0
        
        # Submit both operations
        primary_future = self.executor.submit(self._execute_retrieve, self.primary_system, query, collection, n_results)
        shadow_future = self.executor.submit(self._execute_retrieve, self.shadow_system, query, collection, n_results)
        
        # Wait for primary result (blocking)
        try:
            primary_result = primary_future.result(timeout=timeout_seconds)
        except TimeoutError:
            primary_result = OperationResult(success=False, error="Primary system timeout")
        
        # Wait for shadow result (with timeout)
        try:
            shadow_result = shadow_future.result(timeout=timeout_seconds)
        except TimeoutError:
            shadow_result = OperationResult(success=False, error="Shadow system timeout")
        
        # Compare results
        comparison = self._compare_retrieve_results(primary_result, shadow_result)
        
        # Update metrics
        self._update_metrics('retrieve_knowledge', comparison)
        
        return comparison
    
    def _compare_store_results(self, primary: OperationResult, shadow: OperationResult) -> ComparisonResult:
        """Compare store operation results."""
        differences = []
        similar = True
        similarity_score = 1.0
        
        # Compare success status
        if primary.success != shadow.success:
            differences.append(f"Success status differs: primary={primary.success}, shadow={shadow.success}")
            similar = False
            similarity_score -= 0.5
        
        # Compare data (document IDs)
        if primary.success and shadow.success:
            if primary.data != shadow.data:
                differences.append(f"Document IDs differ: primary={primary.data}, shadow={shadow.data}")
                # This is expected for different systems, so don't mark as dissimilar
        
        # Compare performance
        if abs(primary.duration_ms - shadow.duration_ms) > self.config.get('differences', {}).get('performance_differences', {}).get('latency_threshold_ms', 100):
            differences.append(f"Performance differs significantly: primary={primary.duration_ms}ms, shadow={shadow.duration_ms}ms")
            similarity_score -= 0.2
        
        return ComparisonResult(
            primary_result=primary,
            shadow_result=shadow,
            similar=similar and len(differences) == 0,
            similarity_score=similarity_score,
            differences=differences,
            comparison_metadata={
                "operation": "store_knowledge",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _compare_retrieve_results(self, primary: OperationResult, shadow: OperationResult) -> ComparisonResult:
        """Compare retrieve operation results."""
        differences = []
        similar = True
        similarity_score = 1.0
        
        # Compare success status
        if primary.success != shadow.success:
            differences.append(f"Success status differs: primary={primary.success}, shadow={shadow.success}")
            similar = False
            similarity_score -= 0.5
        
        # Compare results if both successful
        if primary.success and shadow.success and primary.data and shadow.data:
            primary_results = primary.data
            shadow_results = shadow.data
            
            # Compare result counts
            if len(primary_results) != len(shadow_results):
                differences.append(f"Result count differs: primary={len(primary_results)}, shadow={len(shadow_results)}")
                similarity_score -= 0.3
            
            # Compare content similarity
            content_similarity = self._calculate_content_similarity(primary_results, shadow_results)
            if content_similarity < self.config.get('operations', {}).get('retrieve_knowledge', {}).get('similarity_threshold', 0.8):
                differences.append(f"Content similarity low: {content_similarity:.2f}")
                similarity_score = min(similarity_score, content_similarity)
                similar = False
        
        # Compare performance
        if abs(primary.duration_ms - shadow.duration_ms) > self.config.get('differences', {}).get('performance_differences', {}).get('latency_threshold_ms', 100):
            differences.append(f"Performance differs significantly: primary={primary.duration_ms}ms, shadow={shadow.duration_ms}ms")
            similarity_score -= 0.2
        
        return ComparisonResult(
            primary_result=primary,
            shadow_result=shadow,
            similar=similar,
            similarity_score=similarity_score,
            differences=differences,
            comparison_metadata={
                "operation": "retrieve_knowledge",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_content_similarity(self, primary_results: List[Dict], shadow_results: List[Dict]) -> float:
        """Calculate similarity score between result sets."""
        if not primary_results and not shadow_results:
            return 1.0
        
        if not primary_results or not shadow_results:
            return 0.0
        
        # Simple similarity based on content overlap
        primary_content = {result.get('id', str(i)): result.get('content', '') for i, result in enumerate(primary_results)}
        shadow_content = {result.get('id', str(i)): result.get('content', '') for i, result in enumerate(shadow_results)}
        
        # Calculate Jaccard similarity of content
        primary_words = set()
        shadow_words = set()
        
        for content in primary_content.values():
            primary_words.update(content.lower().split())
        
        for content in shadow_content.values():
            shadow_words.update(content.lower().split())
        
        if not primary_words and not shadow_words:
            return 1.0
        
        intersection = len(primary_words.intersection(shadow_words))
        union = len(primary_words.union(shadow_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _log_comparison(self, operation: str, comparison: ComparisonResult):
        """Log comparison results."""
        log_data = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "primary_success": comparison.primary_result.success,
            "shadow_success": comparison.shadow_result.success,
            "similar": comparison.similar,
            "similarity_score": comparison.similarity_score,
            "primary_duration_ms": comparison.primary_result.duration_ms,
            "shadow_duration_ms": comparison.shadow_result.duration_ms,
            "differences_count": len(comparison.differences) if comparison.differences else 0
        }
        
        if comparison.differences:
            log_data["differences"] = comparison.differences
        
        if self.config.get('logging', {}).get('json_structured', True):
            self.logger.info(f"SHADOW_COMPARISON: {json.dumps(log_data)}")
        else:
            self.logger.info(f"Shadow comparison - {operation}: similar={comparison.similar}, "
                           f"score={comparison.similarity_score:.2f}, "
                           f"diffs={len(comparison.differences) if comparison.differences else 0}")
    
    def _update_metrics(self, operation: str, comparison: ComparisonResult):
        """Update metrics with comparison results."""
        self.metrics['operations']['total'] += 1
        
        if comparison.primary_result.success:
            self.metrics['operations']['successful'] += 1
        else:
            self.metrics['operations']['failed'] += 1
        
        self.metrics['operations']['comparisons'] += 1
        
        # Update performance metrics
        primary_latency = comparison.primary_result.duration_ms
        shadow_latency = comparison.shadow_result.duration_ms
        
        # Simple running average
        total_ops = self.metrics['operations']['total']
        self.metrics['performance']['primary_avg_latency'] = (
            (self.metrics['performance']['primary_avg_latency'] * (total_ops - 1) + primary_latency) / total_ops
        )
        self.metrics['performance']['shadow_avg_latency'] = (
            (self.metrics['performance']['shadow_avg_latency'] * (total_ops - 1) + shadow_latency) / total_ops
        )
        
        # Update differences
        if comparison.differences:
            for diff in comparison.differences:
                if 'content' in diff.lower():
                    self.metrics['differences']['content_diffs'] += 1
                elif 'metadata' in diff.lower():
                    self.metrics['differences']['metadata_diffs'] += 1
                elif 'performance' in diff.lower():
                    self.metrics['differences']['performance_diffs'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get shadow mode status."""
        return {
            "enabled": self.is_enabled(),
            "primary_system": "legacy" if self.primary_system == self.legacy_system else "lightrag",
            "shadow_system": "lightrag" if self.shadow_system == self.lightrag_system else "legacy",
            "lightrag_available": self.lightrag_system is not None,
            "metrics": self.get_metrics(),
            "config_path": self.config_path
        }
    
    def shutdown(self):
        """Shutdown shadow mode processor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.logger.info("Shadow mode processor shutdown complete")


# Global shadow mode processor instance
_shadow_processor = None


def get_shadow_processor() -> Optional[ShadowModeProcessor]:
    """Get global shadow mode processor instance."""
    global _shadow_processor
    
    if _shadow_processor is None:
        try:
            _shadow_processor = ShadowModeProcessor()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize shadow processor: {e}")
            return None
    
    return _shadow_processor


def shadow_store_knowledge(collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> Any:
    """Store knowledge with shadow mode processing."""
    processor = get_shadow_processor()
    if processor:
        return processor.store_knowledge(collection, content, metadata, doc_id)
    
    # Fallback to legacy system
    knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'knowledge')
    legacy_system = LegacyKnowledgeSystem(knowledge_path)
    result = legacy_system.store_knowledge(collection, content, metadata, doc_id)
    return result.data if result.success else None


def shadow_retrieve_knowledge(query: str, collection: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve knowledge with shadow mode processing."""
    processor = get_shadow_processor()
    if processor:
        return processor.retrieve_knowledge(query, collection, n_results)
    
    # Fallback to legacy system
    knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'knowledge')
    legacy_system = LegacyKnowledgeSystem(knowledge_path)
    result = legacy_system.retrieve_knowledge(query, collection, n_results)
    return result.data if result.success else []


if __name__ == "__main__":
    # Test shadow mode functionality
    processor = ShadowModeProcessor()
    
    print("Shadow Mode Status:")
    status = processor.get_status()
    print(json.dumps(status, indent=2))
    
    if processor.is_enabled():
        # Test store operation
        print("\nTesting store operation...")
        result = processor.store_knowledge(
            "patterns",
            "Test pattern content",
            {"test": True, "complexity": "low"}
        )
        print(f"Store result: {result}")
        
        # Test retrieve operation
        print("\nTesting retrieve operation...")
        results = processor.retrieve_knowledge("test pattern")
        print(f"Retrieve results: {len(results)} found")
        
        # Show metrics
        print("\nMetrics:")
        metrics = processor.get_metrics()
        print(json.dumps(metrics, indent=2))
    
    processor.shutdown()