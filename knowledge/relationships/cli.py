"""
Command-line interface for relationship detection system.

This CLI provides easy access to relationship detection, storage, and analysis
capabilities from the command line.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .relationship_detector import RelationshipDetector
from .storage_integration import RelationshipAnalysisPipeline, RelationshipStorage
from .relationship_types import RelationshipQuery, RelationshipType
from ..parsing.parser_manager import ParserManager


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Detect and analyze code relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Detect relationships in a single file
  python -m knowledge.relationships.cli detect-file src/main.py
  
  # Detect relationships in entire directory
  python -m knowledge.relationships.cli detect-dir src/ --extensions .py .js
  
  # Query relationships for specific entity
  python -m knowledge.relationships.cli query --source-entity 123e4567-e89b-12d3-a456-426614174000
  
  # Get relationship statistics
  python -m knowledge.relationships.cli stats
  
  # Find relationship patterns
  python -m knowledge.relationships.cli patterns --min-frequency 5
'''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Global options
    parser.add_argument('--db-path', default='knowledge/chromadb/entities.duckdb',
                       help='Path to DuckDB database (default: knowledge/chromadb/entities.duckdb)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--output', '-o', choices=['json', 'text'], default='text',
                       help='Output format (default: text)')
    
    # Detect relationships in a file
    detect_file = subparsers.add_parser('detect-file', help='Detect relationships in a single file')
    detect_file.add_argument('file_path', help='Path to source file to analyze')
    detect_file.add_argument('--update-mode', choices=['insert', 'upsert', 'replace'], 
                           default='upsert', help='Database update mode (default: upsert)')
    detect_file.add_argument('--store', action='store_true', default=True,
                           help='Store relationships in database (default: True)')
    
    # Detect relationships in directory
    detect_dir = subparsers.add_parser('detect-dir', help='Detect relationships in directory')
    detect_dir.add_argument('directory_path', help='Path to directory to analyze')
    detect_dir.add_argument('--extensions', nargs='+', 
                          default=['.py', '.js', '.jsx', '.go', '.rs'],
                          help='File extensions to process (default: .py .js .jsx .go .rs)')
    detect_dir.add_argument('--recursive', action='store_true', default=True,
                          help='Search subdirectories recursively (default: True)')
    detect_dir.add_argument('--exclude', nargs='+', 
                          default=['node_modules', '__pycache__', '.git', 'target', 'build'],
                          help='Patterns to exclude (default: node_modules __pycache__ .git target build)')
    detect_dir.add_argument('--update-mode', choices=['insert', 'upsert', 'replace'], 
                          default='upsert', help='Database update mode (default: upsert)')
    detect_dir.add_argument('--max-concurrent', type=int, default=4,
                          help='Maximum concurrent file processing (default: 4)')
    
    # Query relationships
    query = subparsers.add_parser('query', help='Query stored relationships')
    query.add_argument('--source-entity', help='Source entity UUID')
    query.add_argument('--target-entity', help='Target entity UUID')
    query.add_argument('--relationship-type', choices=[rt.value for rt in RelationshipType],
                      help='Relationship type to filter by')
    query.add_argument('--min-confidence', type=float, default=0.0,
                      help='Minimum confidence score (default: 0.0)')
    query.add_argument('--cross-file-only', action='store_true',
                      help='Only show cross-file relationships')
    query.add_argument('--limit', type=int, default=100,
                      help='Maximum number of results (default: 100)')
    
    # Get statistics
    stats = subparsers.add_parser('stats', help='Show relationship statistics')
    stats.add_argument('--detailed', action='store_true',
                      help='Show detailed breakdown by type and confidence')
    
    # Find patterns
    patterns = subparsers.add_parser('patterns', help='Find relationship patterns')
    patterns.add_argument('--min-frequency', type=int, default=3,
                        help='Minimum pattern frequency (default: 3)')
    patterns.add_argument('--pattern-types', nargs='+',
                        choices=['common_import', 'function_hotspot', 'inheritance_base'],
                        help='Pattern types to find (default: all)')
    
    # Validate relationships
    validate = subparsers.add_parser('validate', help='Validate stored relationships')
    validate.add_argument('--fix-errors', action='store_true',
                        help='Attempt to fix validation errors')
    
    # Cleanup relationships
    cleanup = subparsers.add_parser('cleanup', help='Clean up stale relationships')
    cleanup.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    
    return parser


def format_output(data, format_type: str) -> str:
    """Format output data according to specified format."""
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    else:
        return format_text_output(data)


def format_text_output(data) -> str:
    """Format data as human-readable text."""
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"{key}: {len(value)} items")
            else:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)
    elif isinstance(data, list):
        return f"{len(data)} items"
    else:
        return str(data)


def detect_file_relationships(args, parser_manager: ParserManager) -> dict:
    """Detect relationships in a single file."""
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        return {'error': f'File not found: {file_path}'}
    
    if args.store:
        # Use full pipeline with storage
        pipeline = RelationshipAnalysisPipeline(parser_manager, args.db_path)
        try:
            # We need entities - for CLI, we'll extract them on-demand
            from ..extraction.entity_extractor import EntityExtractor
            entity_extractor = EntityExtractor()
            extraction_result = entity_extractor.extract_from_file(str(file_path))
            
            if not extraction_result.success:
                return {
                    'error': f'Failed to extract entities: {extraction_result.error_message}',
                    'file_path': str(file_path)
                }
            
            result = pipeline.process_file(str(file_path), extraction_result.entities, args.update_mode)
            pipeline.close()
            return result
        except Exception as e:
            pipeline.close()
            return {'error': f'Pipeline error: {e}', 'file_path': str(file_path)}
    else:
        # Detection only, no storage
        detector = RelationshipDetector(parser_manager)
        try:
            # Extract entities first
            from ..extraction.entity_extractor import EntityExtractor
            entity_extractor = EntityExtractor()
            extraction_result = entity_extractor.extract_from_file(str(file_path))
            
            if not extraction_result.success:
                return {
                    'error': f'Failed to extract entities: {extraction_result.error_message}',
                    'file_path': str(file_path)
                }
            
            result = detector.detect_relationships_from_file(str(file_path), extraction_result.entities)
            
            return {
                'file_path': str(file_path),
                'success': result.success,
                'language': result.language,
                'relationships_found': len(result.relationships),
                'detection_time': result.detection_time,
                'error_message': result.error_message,
                'relationship_types': result.get_relationship_counts()
            }
        except Exception as e:
            return {'error': f'Detection error: {e}', 'file_path': str(file_path)}


def detect_directory_relationships(args, parser_manager: ParserManager) -> dict:
    """Detect relationships in a directory."""
    directory_path = Path(args.directory_path)
    
    if not directory_path.exists() or not directory_path.is_dir():
        return {'error': f'Directory not found: {directory_path}'}
    
    pipeline = RelationshipAnalysisPipeline(parser_manager, args.db_path)
    pipeline.detector.max_concurrent_files = args.max_concurrent
    
    try:
        result = pipeline.process_directory(
            str(directory_path),
            extensions=args.extensions,
            recursive=args.recursive,
            exclude_patterns=args.exclude,
            update_mode=args.update_mode
        )
        
        # Add metrics
        result['metrics'] = pipeline.get_pipeline_metrics()
        
        pipeline.close()
        return result
    except Exception as e:
        pipeline.close()
        return {'error': f'Pipeline error: {e}', 'directory': str(directory_path)}


def query_relationships(args) -> dict:
    """Query stored relationships."""
    storage = RelationshipStorage(args.db_path)
    
    try:
        # Build query
        query = RelationshipQuery()
        
        if args.source_entity:
            from uuid import UUID
            query.from_source(UUID(args.source_entity))
        
        if args.target_entity:
            from uuid import UUID
            query.to_target(UUID(args.target_entity))
        
        if args.relationship_type:
            query.of_type(RelationshipType(args.relationship_type))
        
        if args.min_confidence > 0:
            query.with_min_confidence(args.min_confidence)
        
        # Execute query
        relationships = storage.query_relationships(query)
        
        # Limit results
        if len(relationships) > args.limit:
            relationships = relationships[:args.limit]
        
        # Format results
        results = []
        for rel in relationships:
            result = {
                'id': str(rel.id),
                'source_id': str(rel.source_id),
                'target_id': str(rel.target_id),
                'type': rel.relationship_type.value,
                'confidence': rel.confidence,
                'metadata': rel.metadata
            }
            
            if rel.context:
                result['context'] = {
                    'line_number': rel.context.line_number,
                    'source_code': rel.context.source_code
                }
            
            results.append(result)
        
        storage.close()
        return {
            'total_results': len(results),
            'relationships': results,
            'query_parameters': {
                'source_entity': args.source_entity,
                'target_entity': args.target_entity,
                'relationship_type': args.relationship_type,
                'min_confidence': args.min_confidence,
                'limit': args.limit
            }
        }
    
    except Exception as e:
        storage.close()
        return {'error': f'Query error: {e}'}


def get_statistics(args) -> dict:
    """Get relationship statistics."""
    storage = RelationshipStorage(args.db_path)
    
    try:
        stats = storage.get_relationship_statistics()
        
        if args.detailed:
            # Add pattern analysis
            patterns = storage.find_relationship_patterns(min_frequency=2)
            stats['patterns'] = [p.to_dict() for p in patterns[:10]]  # Top 10 patterns
        
        storage.close()
        return stats
    
    except Exception as e:
        storage.close()
        return {'error': f'Statistics error: {e}'}


def find_patterns(args) -> dict:
    """Find relationship patterns."""
    storage = RelationshipStorage(args.db_path)
    
    try:
        patterns = storage.find_relationship_patterns(min_frequency=args.min_frequency)
        
        # Filter by pattern type if specified
        if args.pattern_types:
            patterns = [p for p in patterns if p.pattern_type in args.pattern_types]
        
        storage.close()
        return {
            'total_patterns': len(patterns),
            'patterns': [p.to_dict() for p in patterns]
        }
    
    except Exception as e:
        storage.close()
        return {'error': f'Pattern analysis error: {e}'}


def validate_relationships(args) -> dict:
    """Validate stored relationships."""
    storage = RelationshipStorage(args.db_path)
    
    try:
        # Get all relationships for validation
        query = RelationshipQuery()
        relationships = storage.query_relationships(query)
        
        # Use detector's validation method
        parser_manager = ParserManager()
        detector = RelationshipDetector(parser_manager)
        validation_results = detector.validate_relationships(relationships)
        
        if args.fix_errors and validation_results['errors']:
            # Attempt to fix errors (placeholder - would need specific fixing logic)
            fixed_count = 0
            validation_results['fixed_errors'] = fixed_count
        
        storage.close()
        return validation_results
    
    except Exception as e:
        storage.close()
        return {'error': f'Validation error: {e}'}


def cleanup_relationships(args) -> dict:
    """Clean up stale relationships."""
    storage = RelationshipStorage(args.db_path)
    
    try:
        if args.dry_run:
            # Count what would be deleted
            stats = storage.get_relationship_statistics()
            return {
                'dry_run': True,
                'current_relationships': stats['total_relationships'],
                'note': 'Use without --dry-run to perform actual cleanup'
            }
        else:
            # Perform cleanup
            deleted_count = storage.cleanup_stale_relationships()
            
            storage.close()
            return {
                'cleanup_performed': True,
                'relationships_deleted': deleted_count
            }
    
    except Exception as e:
        storage.close()
        return {'error': f'Cleanup error: {e}'}


def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging level
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    # Initialize parser manager
    parser_manager = ParserManager()
    
    # Execute command
    start_time = time.time()
    
    try:
        if args.command == 'detect-file':
            result = detect_file_relationships(args, parser_manager)
        elif args.command == 'detect-dir':
            result = detect_directory_relationships(args, parser_manager)
        elif args.command == 'query':
            result = query_relationships(args)
        elif args.command == 'stats':
            result = get_statistics(args)
        elif args.command == 'patterns':
            result = find_patterns(args)
        elif args.command == 'validate':
            result = validate_relationships(args)
        elif args.command == 'cleanup':
            result = cleanup_relationships(args)
        else:
            result = {'error': f'Unknown command: {args.command}'}
        
        execution_time = time.time() - start_time
        result['execution_time'] = f"{execution_time:.2f}s"
        
        # Output result
        print(format_output(result, args.output))
        
        # Return appropriate exit code
        return 0 if 'error' not in result else 1
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())