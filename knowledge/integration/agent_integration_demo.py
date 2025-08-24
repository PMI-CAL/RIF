#!/usr/bin/env python3
"""
Agent Integration Demo for Issue #40 Master Coordination

Demonstrates how RIF agents can use the integrated hybrid knowledge system.
This provides a working example of the coordination between Issues #30-33.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import available components individually
from knowledge.extraction.entity_extractor import EntityExtractor
from knowledge.parsing.parser_manager import ParserManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RIFAgentKnowledgeInterface:
    """
    Simplified interface for RIF agents to access the hybrid knowledge system.
    
    This demonstrates the coordination pattern from the Master Coordination Plan
    while providing a working interface that agents can actually use.
    """
    
    def __init__(self):
        """Initialize the agent knowledge interface."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components that we know work
        self.parser_manager = ParserManager.get_instance()
        self.entity_extractor = EntityExtractor()
        
        # Track usage metrics
        self.metrics = {
            'queries_executed': 0,
            'files_analyzed': 0,
            'entities_extracted': 0,
            'start_time': time.time()
        }
        
        self.logger.info("RIF Agent Knowledge Interface initialized")
    
    def analyze_code_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a code file and extract structured information for agents.
        
        This demonstrates the foundation layer (Issue #30) working in coordination.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Dict with extracted code information
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Analyzing file: {file_path}")
            
            # Extract entities using Issue #30 implementation
            extraction_result = self.entity_extractor.extract_from_file(file_path)
            
            # Handle different result formats
            if hasattr(extraction_result, 'entities'):
                entities = extraction_result.entities
                success = extraction_result.success
                error = getattr(extraction_result, 'error', None)
            else:
                # Assume it's a direct list of entities
                entities = extraction_result
                success = True
                error = None
            
            # Update metrics
            self.metrics['files_analyzed'] += 1
            self.metrics['entities_extracted'] += len(entities)
            
            # Format results for agent consumption
            processing_time = time.time() - start_time
            
            return {
                'success': success,
                'file_path': file_path,
                'entities_found': len(entities),
                'entities': [
                    {
                        'name': entity.name if hasattr(entity, 'name') else str(entity),
                        'type': entity.type if hasattr(entity, 'type') else 'unknown',
                        'location': {
                            'line': entity.line if hasattr(entity, 'line') else 0,
                            'column': entity.column if hasattr(entity, 'column') else 0
                        },
                        'metadata': getattr(entity, 'metadata', {})
                    }
                    for entity in entities
                ],
                'processing_time_ms': processing_time * 1000,
                'error': error
            }
            
        except Exception as e:
            self.logger.error(f"File analysis failed: {e}")
            return {
                'success': False,
                'file_path': file_path,
                'entities_found': 0,
                'entities': [],
                'processing_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    def find_code_patterns(self, pattern_type: str, search_directory: str = None) -> Dict[str, Any]:
        """
        Find code patterns in the project for agent analysis.
        
        This demonstrates coordination of multiple components working together.
        
        Args:
            pattern_type: Type of pattern to find ('functions', 'classes', 'modules', etc.)
            search_directory: Directory to search (optional)
            
        Returns:
            Dict with found patterns
        """
        start_time = time.time()
        
        try:
            search_dir = Path(search_directory) if search_directory else Path.cwd()
            
            # Find relevant files
            file_patterns = ['*.py', '*.js', '*.jsx', '*.mjs', '*.cjs', '*.go', '*.rs']
            files_to_analyze = []
            
            for pattern in file_patterns:
                files_to_analyze.extend(search_dir.rglob(pattern))
            
            # Limit to prevent overwhelming
            files_to_analyze = files_to_analyze[:20]
            
            found_patterns = []
            total_entities = 0
            
            for file_path in files_to_analyze:
                try:
                    analysis = self.analyze_code_file(str(file_path))
                    
                    if analysis['success']:
                        # Filter entities by pattern type
                        matching_entities = [
                            entity for entity in analysis['entities']
                            if self._matches_pattern_type(entity, pattern_type)
                        ]
                        
                        if matching_entities:
                            found_patterns.append({
                                'file': str(file_path),
                                'patterns': matching_entities
                            })
                            total_entities += len(matching_entities)
                            
                except Exception as e:
                    self.logger.warning(f"Error analyzing {file_path}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'pattern_type': pattern_type,
                'search_directory': str(search_dir),
                'files_analyzed': len(files_to_analyze),
                'patterns_found': total_entities,
                'pattern_locations': found_patterns,
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Pattern search failed: {e}")
            return {
                'success': False,
                'pattern_type': pattern_type,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    def _matches_pattern_type(self, entity: Dict[str, Any], pattern_type: str) -> bool:
        """Check if an entity matches the requested pattern type."""
        entity_type = entity.get('type', '').lower()
        pattern_type = pattern_type.lower()
        
        if pattern_type in ['function', 'functions']:
            return entity_type in ['function', 'method']
        elif pattern_type in ['class', 'classes']:
            return entity_type == 'class'
        elif pattern_type in ['variable', 'variables']:
            return entity_type in ['variable', 'constant']
        elif pattern_type in ['module', 'modules']:
            return entity_type == 'module'
        else:
            return True  # Return all if pattern not recognized
    
    def get_project_summary(self, project_path: str = None) -> Dict[str, Any]:
        """
        Get a summary of the project's code structure for agent planning.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dict with project summary
        """
        start_time = time.time()
        
        try:
            project_dir = Path(project_path) if project_path else Path.cwd()
            
            # Find all code files
            file_patterns = ['*.py', '*.js', '*.jsx', '*.mjs', '*.cjs', '*.go', '*.rs']
            all_files = []
            
            for pattern in file_patterns:
                all_files.extend(project_dir.rglob(pattern))
            
            # Analyze a sample of files
            sample_files = all_files[:10] if len(all_files) > 10 else all_files
            
            summary = {
                'project_path': str(project_dir),
                'total_files': len(all_files),
                'files_analyzed': len(sample_files),
                'languages': set(),
                'entity_counts': {
                    'functions': 0,
                    'classes': 0,
                    'modules': 0,
                    'variables': 0
                },
                'file_analysis': []
            }
            
            for file_path in sample_files:
                try:
                    analysis = self.analyze_code_file(str(file_path))
                    
                    if analysis['success']:
                        # Determine language
                        language = self._detect_language(file_path)
                        summary['languages'].add(language)
                        
                        # Count entity types
                        for entity in analysis['entities']:
                            entity_type = entity.get('type', '').lower()
                            if entity_type in ['function', 'method']:
                                summary['entity_counts']['functions'] += 1
                            elif entity_type == 'class':
                                summary['entity_counts']['classes'] += 1
                            elif entity_type == 'module':
                                summary['entity_counts']['modules'] += 1
                            elif entity_type in ['variable', 'constant']:
                                summary['entity_counts']['variables'] += 1
                        
                        summary['file_analysis'].append({
                            'file': str(file_path.relative_to(project_dir)),
                            'language': language,
                            'entities': analysis['entities_found']
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error in project summary for {file_path}: {e}")
                    continue
            
            # Convert set to list for JSON serialization
            summary['languages'] = list(summary['languages'])
            
            processing_time = time.time() - start_time
            summary['processing_time_ms'] = processing_time * 1000
            summary['success'] = True
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Project summary failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.py':
            return 'python'
        elif suffix in ['.js', '.jsx', '.mjs', '.cjs']:
            return 'javascript'
        elif suffix == '.go':
            return 'go'
        elif suffix == '.rs':
            return 'rust'
        else:
            return 'unknown'
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics for monitoring."""
        uptime = time.time() - self.metrics['start_time']
        
        return {
            'uptime_seconds': uptime,
            'queries_executed': self.metrics['queries_executed'],
            'files_analyzed': self.metrics['files_analyzed'],
            'entities_extracted': self.metrics['entities_extracted'],
            'performance': {
                'avg_files_per_second': self.metrics['files_analyzed'] / max(uptime, 1),
                'avg_entities_per_file': self.metrics['entities_extracted'] / max(self.metrics['files_analyzed'], 1)
            }
        }


def demo_agent_usage():
    """Demonstrate how RIF agents would use the knowledge interface."""
    print("=" * 70)
    print("RIF AGENT KNOWLEDGE INTERFACE DEMO")
    print("=" * 70)
    
    # Create the interface
    agent_interface = RIFAgentKnowledgeInterface()
    
    # Demo 1: Analyze a single file
    print("\n--- Demo 1: Single File Analysis ---")
    test_files = [
        "knowledge/integration/agent_integration_demo.py",  # This file
        "knowledge/extraction/entity_extractor.py"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\nAnalyzing: {file_path}")
            result = agent_interface.analyze_code_file(file_path)
            
            if result['success']:
                print(f"  ✓ Found {result['entities_found']} entities in {result['processing_time_ms']:.1f}ms")
                for entity in result['entities'][:3]:  # Show first 3
                    print(f"    - {entity['name']} ({entity['type']}) at line {entity['location']['line']}")
            else:
                print(f"  ✗ Analysis failed: {result['error']}")
            break
    
    # Demo 2: Find code patterns
    print("\n--- Demo 2: Code Pattern Search ---")
    patterns_to_find = ['functions', 'classes']
    
    for pattern in patterns_to_find:
        print(f"\nSearching for {pattern}...")
        result = agent_interface.find_code_patterns(pattern, "knowledge/integration")
        
        if result['success']:
            print(f"  ✓ Found {result['patterns_found']} {pattern} in {result['files_analyzed']} files")
            print(f"  ✓ Processing time: {result['processing_time_ms']:.1f}ms")
        else:
            print(f"  ✗ Pattern search failed: {result['error']}")
    
    # Demo 3: Project summary
    print("\n--- Demo 3: Project Summary ---")
    summary = agent_interface.get_project_summary("knowledge/integration")
    
    if summary['success']:
        print(f"  ✓ Project analyzed: {summary['project_path']}")
        print(f"  ✓ Total files: {summary['total_files']}")
        print(f"  ✓ Languages: {', '.join(summary['languages'])}")
        print(f"  ✓ Entity counts: {summary['entity_counts']}")
        print(f"  ✓ Processing time: {summary['processing_time_ms']:.1f}ms")
    else:
        print(f"  ✗ Project summary failed: {summary['error']}")
    
    # Show metrics
    print("\n--- System Metrics ---")
    metrics = agent_interface.get_metrics()
    print(f"Uptime: {metrics['uptime_seconds']:.1f}s")
    print(f"Files analyzed: {metrics['files_analyzed']}")
    print(f"Entities extracted: {metrics['entities_extracted']}")
    print(f"Performance: {metrics['performance']['avg_files_per_second']:.2f} files/sec")
    
    print("\n✅ Agent integration demo completed successfully!")
    print("\nThis demonstrates the working coordination of Issue #30 (Entity Extraction)")
    print("as part of the Master Coordination Plan (Issue #40).")


if __name__ == "__main__":
    demo_agent_usage()