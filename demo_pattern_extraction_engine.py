#!/usr/bin/env python3
"""
Demo script for Pattern Extraction Engine - Issue #75

This script demonstrates the complete pattern extraction system including:
- Multi-method pattern extraction from completed issues
- Success metrics calculation and analysis  
- Performance optimization with caching
- Comprehensive pattern analysis and reporting

Usage:
    python3 demo_pattern_extraction_engine.py
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add knowledge directory to path
sys.path.insert(0, str(Path(__file__).parent / 'knowledge'))

from knowledge.pattern_extraction import (
    PatternDiscoveryEngine,
    CodePatternExtractor, 
    WorkflowPatternExtractor,
    DecisionPatternExtractor,
    SuccessMetricsCalculator,
    PatternExtractionCache,
    get_global_cache
)


class MockKnowledgeSystem:
    """Mock knowledge system for demo purposes."""
    
    def __init__(self):
        self.stored_patterns = []
    
    def store_knowledge(self, collection, content, metadata=None, doc_id=None):
        """Mock storage method."""
        pattern_id = doc_id or f"pattern_{len(self.stored_patterns)}"
        self.stored_patterns.append({
            'id': pattern_id,
            'collection': collection,
            'content': content,
            'metadata': metadata or {}
        })
        return pattern_id
    
    def retrieve_knowledge(self, query, collection=None, n_results=5, filters=None):
        """Mock retrieval method."""
        # Return some sample patterns for similarity search
        return [
            {
                'id': 'sample_pattern_1',
                'content': {'title': 'Similar Pattern', 'description': 'A similar pattern'},
                'metadata': {'type': 'extracted_pattern'},
                'distance': 0.3
            }
        ]


def create_sample_completed_issue():
    """Create a comprehensive sample completed issue for demonstration."""
    return {
        'issue_number': 75,
        'title': 'Build pattern extraction engine',
        'body': '''
        ## Objective
        Extract successful patterns from completed work, categorize by type, and calculate success rates.
        
        ## Technical Requirements
        - Pattern identification logic
        - Pattern categorization with ML
        - Success rate calculation with statistics
        - Pattern storage in knowledge base
        
        ## Implementation Approach
        We decided to use a multi-method extraction approach combining AST analysis,
        workflow mining, and decision pattern extraction for comprehensive coverage.
        ''',
        'complexity': 'high',
        'state': 'closed',
        
        # Code changes simulation
        'code_changes': {
            'knowledge/pattern_extraction/discovery_engine.py': {
                'added_lines': '''
class PatternDiscoveryEngine:
    def __init__(self, knowledge_system=None):
        self.knowledge = knowledge_system or get_knowledge_system()
        self.extractors = {}
        self.pattern_signatures = set()
    
    def discover_patterns(self, completed_issue):
        patterns = []
        for pattern_type, extractor in self.extractors.items():
            try:
                extracted = extractor.extract_patterns(completed_issue)
                patterns.extend(extracted)
            except Exception as e:
                self.logger.error(f"Error in {pattern_type} extraction: {e}")
        
        unique_patterns = self._deduplicate_patterns(patterns)
        return unique_patterns
                '''
            },
            'knowledge/pattern_extraction/success_metrics.py': {
                'added_lines': '''
class SuccessMetricsCalculator:
    def calculate_pattern_metrics(self, pattern, application_data=None):
        success_rate, confidence_interval = self._calculate_success_rate(application_data)
        applicability_score = self._calculate_applicability_score(pattern, application_data)
        return SuccessMetrics(
            pattern_id=pattern.signature.combined_hash,
            success_rate=success_rate,
            confidence_interval=confidence_interval,
            applicability_score=applicability_score,
            sample_size=len(application_data or []),
            calculation_date=datetime.now()
        )
                '''
            }
        },
        
        # Workflow history simulation
        'history': [
            {
                'timestamp': '2023-01-01T10:00:00Z',
                'label_added': {'name': 'state:new'},
                'agent': 'rif-analyst'
            },
            {
                'timestamp': '2023-01-01T11:00:00Z',
                'label_added': {'name': 'state:analyzing'},
                'agent': 'rif-analyst'
            },
            {
                'timestamp': '2023-01-01T12:00:00Z',
                'label_added': {'name': 'state:planning'},
                'agent': 'rif-planner'
            },
            {
                'timestamp': '2023-01-01T13:00:00Z',
                'label_added': {'name': 'state:architecting'},
                'agent': 'rif-architect'
            },
            {
                'timestamp': '2023-01-01T14:00:00Z',
                'label_added': {'name': 'state:implementing'},
                'agent': 'rif-implementer'
            },
            {
                'timestamp': '2023-01-01T16:00:00Z',
                'label_added': {'name': 'state:validating'},
                'agent': 'rif-validator'
            },
            {
                'timestamp': '2023-01-01T17:00:00Z',
                'label_added': {'name': 'state:complete'},
                'agent': 'rif-learner'
            }
        ],
        
        # Architectural decisions simulation
        'decisions': [
            {
                'title': 'Multi-method pattern extraction approach',
                'context': 'Need comprehensive pattern coverage from different data sources',
                'decision': 'Implement separate extractors for code, workflow, and decision patterns',
                'alternatives': [
                    {
                        'name': 'Single unified extractor',
                        'pros': ['Simpler architecture', 'Less code complexity'],
                        'cons': ['Limited pattern coverage', 'Reduced accuracy']
                    },
                    {
                        'name': 'Multi-method extraction',
                        'pros': ['Comprehensive coverage', 'Specialized analysis'],
                        'cons': ['Higher complexity', 'More integration work']
                    }
                ],
                'rationale': 'Specialized extractors provide better accuracy and coverage',
                'consequences': ['Higher initial complexity', 'Better long-term maintainability'],
                'impact': 'high',
                'status': 'accepted'
            },
            {
                'title': 'Statistical success metrics framework',
                'context': 'Need reliable measurement of pattern effectiveness',
                'decision': 'Use Wilson confidence intervals and multi-dimensional scoring',
                'rationale': 'Provides statistical rigor and comprehensive assessment',
                'impact': 'medium',
                'status': 'accepted'
            }
        ],
        
        # Agent interactions simulation
        'agent_interactions': [
            {
                'timestamp': '2023-01-01T11:00:00Z',
                'agent': 'rif-analyst',
                'type': 'analysis',
                'activity': 'Requirements analysis and complexity assessment'
            },
            {
                'timestamp': '2023-01-01T12:00:00Z', 
                'agent': 'rif-planner',
                'type': 'planning',
                'activity': 'Implementation strategy and resource allocation'
            },
            {
                'timestamp': '2023-01-01T13:00:00Z',
                'agent': 'rif-architect', 
                'type': 'design',
                'activity': 'System architecture and component design'
            },
            {
                'timestamp': '2023-01-01T14:00:00Z',
                'agent': 'rif-implementer',
                'type': 'implementation', 
                'activity': 'Core pattern extraction implementation'
            },
            {
                'timestamp': '2023-01-01T16:00:00Z',
                'agent': 'rif-validator',
                'type': 'validation',
                'activity': 'Testing and quality assurance'
            }
        ],
        
        # Issue comments simulation
        'comments': [
            {
                'author': 'architect',
                'body': 'We decided to implement a multi-layered extraction approach for maximum pattern coverage'
            },
            {
                'author': 'developer',
                'body': 'The AST-based code analysis is working well for detecting design patterns'
            },
            {
                'author': 'qa-engineer',
                'body': 'Test coverage is at 86% with comprehensive integration tests'
            }
        ],
        
        # Files created simulation
        'files_created': [
            'knowledge/pattern_extraction/discovery_engine.py',
            'knowledge/pattern_extraction/code_extractor.py',
            'knowledge/pattern_extraction/workflow_extractor.py',
            'knowledge/pattern_extraction/decision_extractor.py',
            'knowledge/pattern_extraction/success_metrics.py',
            'knowledge/pattern_extraction/cache.py',
            'tests/test_pattern_extraction.py'
        ],
        
        # Performance metrics simulation
        'performance_metrics': {
            'implementation_time_hours': 6,
            'lines_of_code': 2500,
            'test_coverage': 0.86,
            'performance_benchmark_seconds': 1.2
        }
    }


def demonstrate_pattern_extraction():
    """Main demonstration function."""
    print("🔍 RIF Pattern Extraction Engine Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Pattern Extraction System...")
    
    # Mock knowledge system
    knowledge_system = MockKnowledgeSystem()
    
    # Initialize discovery engine
    discovery_engine = PatternDiscoveryEngine(knowledge_system, enable_cache=True)
    
    # Initialize extractors
    code_extractor = CodePatternExtractor()
    workflow_extractor = WorkflowPatternExtractor()
    decision_extractor = DecisionPatternExtractor()
    
    # Register extractors
    discovery_engine.register_extractor('code', code_extractor)
    discovery_engine.register_extractor('workflow', workflow_extractor)
    discovery_engine.register_extractor('decision', decision_extractor)
    
    # Initialize success metrics calculator
    metrics_calculator = SuccessMetricsCalculator()
    
    print("✅ System initialized with all extractors registered")
    
    # Create sample issue
    print("\n2. Creating Sample Completed Issue...")
    completed_issue = create_sample_completed_issue()
    print(f"✅ Created issue #{completed_issue['issue_number']}: {completed_issue['title']}")
    print(f"   - Complexity: {completed_issue['complexity']}")
    print(f"   - Code files: {len(completed_issue['code_changes'])}")
    print(f"   - Decisions: {len(completed_issue['decisions'])}")
    print(f"   - History events: {len(completed_issue['history'])}")
    
    # Extract patterns
    print("\n3. Extracting Patterns...")
    start_time = time.time()
    
    patterns = discovery_engine.discover_patterns(completed_issue)
    
    extraction_time = time.time() - start_time
    print(f"✅ Extracted {len(patterns)} patterns in {extraction_time:.2f} seconds")
    
    # Display extracted patterns
    print("\n4. Pattern Analysis Results:")
    print("-" * 40)
    
    pattern_types = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type
        if pattern_type not in pattern_types:
            pattern_types[pattern_type] = []
        pattern_types[pattern_type].append(pattern)
    
    for pattern_type, type_patterns in pattern_types.items():
        print(f"\n📊 {pattern_type.upper()} PATTERNS ({len(type_patterns)} found):")
        for i, pattern in enumerate(type_patterns, 1):
            print(f"   {i}. {pattern.title}")
            print(f"      Method: {pattern.extraction_method}")
            print(f"      Confidence: {pattern.confidence:.2f}")
            print(f"      Source: {pattern.source}")
    
    # Calculate success metrics
    print("\n5. Success Metrics Analysis...")
    
    if patterns:
        print("-" * 40)
        for i, pattern in enumerate(patterns[:3], 1):  # Analyze first 3 patterns
            print(f"\n📈 METRICS FOR PATTERN {i}: {pattern.title}")
            
            start_time = time.time()
            metrics = metrics_calculator.calculate_pattern_metrics(pattern)
            metrics_time = time.time() - start_time
            
            print(f"   Success Rate: {metrics.success_rate:.3f}")
            print(f"   Confidence Interval: ({metrics.confidence_interval[0]:.3f}, {metrics.confidence_interval[1]:.3f})")
            print(f"   Applicability Score: {metrics.applicability_score:.3f}")
            print(f"   Reusability Index: {metrics.reusability_index:.3f}")
            print(f"   Reliability Score: {metrics.reliability_score:.3f}")
            print(f"   Sample Size: {metrics.sample_size}")
            print(f"   Calculation Time: {metrics_time:.3f}s")
    
    # Pattern ranking
    print("\n6. Pattern Quality Ranking...")
    print("-" * 40)
    
    if patterns:
        rankings = metrics_calculator.get_pattern_ranking(patterns)
        
        print("\n🏆 TOP PATTERNS BY QUALITY:")
        for i, (pattern, score) in enumerate(rankings, 1):
            print(f"   {i}. {pattern.title}")
            print(f"      Quality Score: {score:.3f}")
            print(f"      Pattern Type: {pattern.pattern_type}")
            print(f"      Extraction Method: {pattern.extraction_method}")
            if i >= 5:  # Show top 5
                break
    
    # Cache performance analysis
    print("\n7. Performance Analysis...")
    print("-" * 40)
    
    cache = get_global_cache()
    cache_stats = cache.get_cache_stats()
    
    print(f"\n⚡ CACHE PERFORMANCE:")
    print(f"   Memory Cache Hit Rate: {cache_stats['memory_cache']['hit_rate']:.1%}")
    print(f"   Memory Cache Entries: {cache_stats['memory_cache']['size']}")
    print(f"   Total Cache Size: {cache_stats['total_cache_size_mb']:.1f} MB")
    print(f"   Disk Cache Files: {cache_stats['disk_cache']['files']}")
    
    # Extraction statistics
    stats = discovery_engine.get_extraction_statistics()
    print(f"\n📊 EXTRACTION STATISTICS:")
    print(f"   Total Patterns Discovered: {stats['patterns_discovered']}")
    print(f"   Duplicates Filtered: {stats['duplicates_filtered']}")
    print(f"   Errors Encountered: {stats['errors_encountered']}")
    print(f"   Extraction Methods Used: {len(stats['extraction_methods_used'])}")
    print(f"   Unique Patterns Cached: {stats['unique_patterns_cached']}")
    
    # Storage analysis
    print(f"\n💾 KNOWLEDGE BASE STORAGE:")
    print(f"   Patterns Stored: {len(knowledge_system.stored_patterns)}")
    
    stored_by_collection = {}
    for stored_pattern in knowledge_system.stored_patterns:
        collection = stored_pattern['collection']
        stored_by_collection[collection] = stored_by_collection.get(collection, 0) + 1
    
    for collection, count in stored_by_collection.items():
        print(f"   {collection}: {count} entries")
    
    # Export demonstration
    print("\n8. Export Capabilities...")
    print("-" * 40)
    
    # Export patterns
    export_file = "/tmp/extracted_patterns_demo.json"
    export_success = discovery_engine.export_patterns(export_file)
    
    if export_success:
        print(f"✅ Patterns exported to: {export_file}")
        
        # Check export file size
        export_path = Path(export_file)
        if export_path.exists():
            file_size = export_path.stat().st_size
            print(f"   Export file size: {file_size} bytes")
    else:
        print("❌ Pattern export failed")
    
    # Performance benchmarks
    print("\n9. Performance Benchmarks...")
    print("-" * 40)
    
    # Simulate processing multiple issues
    benchmark_start = time.time()
    
    for i in range(5):
        # Create slight variations of the issue
        benchmark_issue = completed_issue.copy()
        benchmark_issue['issue_number'] = 75 + i
        benchmark_issue['title'] = f"Pattern extraction benchmark {i+1}"
        
        benchmark_patterns = discovery_engine.discover_patterns(benchmark_issue)
    
    benchmark_time = time.time() - benchmark_start
    
    print(f"⏱️  BENCHMARK RESULTS:")
    print(f"   Processed 5 issues in {benchmark_time:.2f} seconds")
    print(f"   Average processing time: {benchmark_time/5:.2f} seconds per issue")
    print(f"   Processing rate: {5/benchmark_time:.1f} issues per second")
    
    # Success summary
    print("\n" + "=" * 60)
    print("🎉 PATTERN EXTRACTION ENGINE DEMO COMPLETE")
    print("=" * 60)
    
    print(f"\n✅ DEMONSTRATION SUMMARY:")
    print(f"   - Successfully extracted {len(patterns)} patterns")
    print(f"   - Used {len(stats['extraction_methods_used'])} extraction methods")
    print(f"   - Achieved {extraction_time:.2f}s extraction time")
    print(f"   - Maintained {cache_stats['memory_cache']['hit_rate']:.1%} cache hit rate")
    print(f"   - Processed patterns with statistical confidence")
    print(f"   - Demonstrated comprehensive pattern analysis capabilities")
    
    print(f"\n🚀 The Pattern Extraction Engine is ready for production use!")
    
    return {
        'patterns_extracted': len(patterns),
        'extraction_time': extraction_time,
        'cache_hit_rate': cache_stats['memory_cache']['hit_rate'],
        'patterns_stored': len(knowledge_system.stored_patterns),
        'benchmark_rate': 5/benchmark_time
    }


if __name__ == "__main__":
    try:
        results = demonstrate_pattern_extraction()
        print(f"\n✅ Demo completed successfully with results: {results}")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)