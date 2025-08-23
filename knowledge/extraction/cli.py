"""
Command-line interface for entity extraction system.
"""

import argparse
import logging
import json
import sys
from pathlib import Path

from .entity_extractor import EntityExtractor
from .storage_integration import EntityExtractionPipeline


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_command(args):
    """Handle the extract command."""
    print(f"Extracting entities from: {args.input}")
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    if Path(args.input).is_file():
        # Single file
        result = extractor.extract_from_file(args.input)
        
        if result.success:
            print(f"✅ Successfully extracted {len(result.entities)} entities")
            print(f"   Extraction time: {result.extraction_time:.3f}s")
            
            if args.output:
                # Save results to file
                output_data = {
                    'file_path': result.file_path,
                    'language': result.language,
                    'entities': [entity.to_db_dict() for entity in result.entities],
                    'extraction_time': result.extraction_time
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(f"   Results saved to: {args.output}")
            else:
                # Print entity summary
                entity_counts = result.get_entity_counts()
                print("   Entity counts:")
                for entity_type, count in entity_counts.items():
                    print(f"     {entity_type}: {count}")
        else:
            print(f"❌ Extraction failed: {result.error_message}")
            return 1
    
    else:
        # Directory
        results = extractor.extract_from_directory(
            args.input,
            recursive=not args.no_recursive,
            exclude_patterns=args.exclude.split(',') if args.exclude else None
        )
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ Processed {len(results)} files")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            total_entities = sum(len(r.entities) for r in successful)
            avg_time = sum(r.extraction_time for r in successful) / len(successful)
            print(f"   Total entities: {total_entities}")
            print(f"   Average time per file: {avg_time:.3f}s")
        
        if failed and args.verbose:
            print("\nFailed files:")
            for result in failed:
                print(f"  {result.file_path}: {result.error_message}")
        
        if args.output:
            # Save summary results
            summary_data = {
                'total_files': len(results),
                'successful_files': len(successful),
                'failed_files': len(failed),
                'total_entities': sum(len(r.entities) for r in successful),
                'results': [
                    {
                        'file_path': r.file_path,
                        'success': r.success,
                        'language': r.language,
                        'entity_count': len(r.entities),
                        'extraction_time': r.extraction_time,
                        'error': r.error_message
                    }
                    for r in results
                ]
            }
            
            with open(args.output, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"   Summary saved to: {args.output}")
    
    return 0


def pipeline_command(args):
    """Handle the pipeline command (extract + store)."""
    print(f"Running extraction pipeline on: {args.input}")
    
    try:
        # Initialize pipeline
        pipeline = EntityExtractionPipeline(args.database)
        
        if Path(args.input).is_file():
            # Single file
            result = pipeline.process_file(args.input, args.update_mode)
            
            if result['extraction_success']:
                print(f"✅ Successfully processed file")
                print(f"   Entities extracted: {result['entities_found']}")
                print(f"   Entities stored: {result['entities_stored']}")
                print(f"   Extraction time: {result['extraction_time']:.3f}s")
                print(f"   Storage: {result['storage_result']}")
            else:
                print(f"❌ Processing failed: {result['error_message']}")
                return 1
        
        else:
            # Directory
            result = pipeline.process_directory(
                args.input,
                recursive=not args.no_recursive,
                exclude_patterns=args.exclude.split(',') if args.exclude else None,
                update_mode=args.update_mode
            )
            
            print(f"✅ Pipeline completed")
            print(f"   Files processed: {result['total_files']}")
            print(f"   Successful extractions: {result['successful_extractions']}")
            print(f"   Failed extractions: {result['failed_extractions']}")
            print(f"   Entities extracted: {result['total_entities_extracted']}")
            print(f"   Entities stored: {result['total_entities_stored']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Storage stats: {result['storage_stats']}")
        
        # Show final metrics
        if args.verbose:
            metrics = pipeline.get_pipeline_metrics()
            print("\nPipeline Metrics:")
            print(json.dumps(metrics, indent=2, default=str))
        
        pipeline.close()
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def metrics_command(args):
    """Handle the metrics command."""
    try:
        if args.extraction_only:
            extractor = EntityExtractor()
            metrics = extractor.get_extraction_metrics()
        else:
            pipeline = EntityExtractionPipeline(args.database)
            metrics = pipeline.get_pipeline_metrics()
            pipeline.close()
        
        print(json.dumps(metrics, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RIF Entity Extraction System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract entities from a single file
  python -m knowledge.extraction.cli extract /path/to/file.py
  
  # Extract entities from a directory
  python -m knowledge.extraction.cli extract /path/to/project --output results.json
  
  # Run full pipeline (extract + store)
  python -m knowledge.extraction.cli pipeline /path/to/project
  
  # Get pipeline metrics
  python -m knowledge.extraction.cli metrics
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract entities from files')
    extract_parser.add_argument('input', help='File or directory path to process')
    extract_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    extract_parser.add_argument('--no-recursive', action='store_true',
                               help='Don\'t process subdirectories')
    extract_parser.add_argument('--exclude', help='Comma-separated patterns to exclude')
    extract_parser.set_defaults(func=extract_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', 
                                          help='Run complete extraction pipeline')
    pipeline_parser.add_argument('input', help='File or directory path to process')
    pipeline_parser.add_argument('--database', '-db', 
                                default='knowledge/chromadb/entities.duckdb',
                                help='DuckDB database path')
    pipeline_parser.add_argument('--update-mode', choices=['insert', 'upsert', 'replace'],
                                default='upsert',
                                help='How to handle existing entities')
    pipeline_parser.add_argument('--no-recursive', action='store_true',
                                help='Don\'t process subdirectories')
    pipeline_parser.add_argument('--exclude', help='Comma-separated patterns to exclude')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show system metrics')
    metrics_parser.add_argument('--database', '-db',
                               default='knowledge/chromadb/entities.duckdb',
                               help='DuckDB database path')
    metrics_parser.add_argument('--extraction-only', action='store_true',
                               help='Show only extraction metrics (no storage)')
    metrics_parser.set_defaults(func=metrics_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())