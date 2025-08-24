#!/usr/bin/env python3
"""
Pattern Export/Import CLI - Issue #80

Command-line interface for pattern export/import functionality.
Provides easy-to-use commands for backing up, sharing, and migrating patterns.

Usage:
    python pattern_export_import_cli.py export [options]
    python pattern_export_import_cli.py import <file> [options] 
    python pattern_export_import_cli.py list [options]
    python pattern_export_import_cli.py stats
    python pattern_export_import_cli.py validate <file>
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from claude.commands.pattern_portability import (
    PatternPortability, MergeStrategy, ConflictResolution
)


class PatternCLI:
    """Command-line interface for pattern export/import operations."""
    
    def __init__(self):
        self.setup_logging()
        self.portability = PatternPortability()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def export_patterns(self, args) -> int:
        """
        Export patterns to JSON file.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Parse pattern IDs if provided
            pattern_ids = None
            if args.patterns:
                pattern_ids = [p.strip() for p in args.patterns.split(',')]
                print(f"Exporting specific patterns: {pattern_ids}")
            else:
                print("Exporting all patterns...")
            
            # Perform export
            json_data = self.portability.export_patterns(
                pattern_ids=pattern_ids,
                output_file=args.output
            )
            
            # Parse export data for summary
            export_data = json.loads(json_data)
            pattern_count = export_data['metadata']['pattern_count']
            success_rate = export_data['metadata']['success_rate_avg']
            
            if args.output:
                print(f"✅ Successfully exported {pattern_count} patterns to {args.output}")
            else:
                print(f"✅ Successfully exported {pattern_count} patterns")
            
            print(f"📊 Average success rate: {success_rate:.2f}")
            
            # Show complexity breakdown if verbose
            if args.verbose:
                complexity_breakdown = export_data['metadata']['complexity_breakdown']
                print("\n📈 Complexity breakdown:")
                for complexity, count in complexity_breakdown.items():
                    print(f"  {complexity}: {count}")
                
                domain_breakdown = export_data['metadata']['domain_breakdown']
                print("\n🏷️ Domain breakdown:")
                for domain, count in domain_breakdown.items():
                    print(f"  {domain}: {count}")
            
            if not args.output:
                print("\n" + "="*50)
                print("JSON Export Data:")
                print("="*50)
                if args.pretty:
                    print(json.dumps(export_data, indent=2))
                else:
                    print(json_data)
            
            return 0
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            if args.verbose:
                self.logger.exception("Export error details")
            return 1
    
    def import_patterns(self, args) -> int:
        """
        Import patterns from JSON file.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Read import file
            import_file = Path(args.file)
            if not import_file.exists():
                print(f"❌ Import file not found: {args.file}")
                return 1
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            print(f"Importing patterns from {args.file}...")
            print(f"Source version: {import_data.get('version', 'unknown')}")
            print(f"Patterns to import: {len(import_data.get('patterns', []))}")
            
            # Parse merge strategy
            strategy = MergeStrategy(args.strategy)
            print(f"Merge strategy: {strategy.value}")
            
            # Validate before import if requested
            if args.validate_only:
                patterns = import_data.get('patterns', [])
                validation_results = self.portability.validate_patterns(patterns)
                
                valid_count = sum(1 for r in validation_results if r['valid'])
                invalid_count = len(validation_results) - valid_count
                
                print(f"\n📋 Validation Results:")
                print(f"Valid patterns: {valid_count}")
                print(f"Invalid patterns: {invalid_count}")
                
                if args.verbose and invalid_count > 0:
                    print("\n❌ Invalid patterns:")
                    for result in validation_results:
                        if not result['valid']:
                            print(f"  Pattern {result['pattern_id']}:")
                            for error in result['errors']:
                                print(f"    - {error}")
                
                return 0 if invalid_count == 0 else 1
            
            # Perform import
            result = self.portability.import_patterns(import_data, strategy)
            
            # Display results
            print(f"\n📊 Import Results:")
            print(f"✅ Imported: {result.imported_count}")
            print(f"⏭️  Skipped: {result.skipped_count}")
            print(f"❌ Errors: {result.error_count}")
            print(f"⏱️  Duration: {result.import_duration:.2f}s")
            
            # Show conflicts if any
            if result.conflicts:
                print(f"\n⚠️  Conflicts resolved: {len(result.conflicts)}")
                if args.verbose:
                    for conflict in result.conflicts:
                        print(f"  {conflict.pattern_id}: {conflict.resolution.value} - {conflict.details}")
            
            # Show errors if any
            if result.errors:
                print(f"\n❌ Errors encountered:")
                for error in result.errors:
                    print(f"  - {error}")
            
            # Show imported patterns
            if result.imported_patterns and args.verbose:
                print(f"\n📥 Imported patterns:")
                for pattern_id in result.imported_patterns:
                    print(f"  - {pattern_id}")
            
            return 0 if result.error_count == 0 else 1
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            if args.verbose:
                self.logger.exception("Import error details")
            return 1
    
    def list_patterns(self, args) -> int:
        """
        List available patterns.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            patterns = self.portability.get_all_patterns()
            
            if not patterns:
                print("No patterns found.")
                return 0
            
            print(f"Found {len(patterns)} patterns:\n")
            
            # Filter by domain if specified
            if args.domain:
                patterns = [p for p in patterns if p.domain == args.domain]
                print(f"Filtered to {len(patterns)} patterns in domain '{args.domain}':\n")
            
            # Filter by complexity if specified
            if args.complexity:
                patterns = [p for p in patterns if p.complexity == args.complexity]
                print(f"Filtered to {len(patterns)} patterns with complexity '{args.complexity}':\n")
            
            # Sort patterns
            if args.sort == 'name':
                patterns.sort(key=lambda p: p.name.lower())
            elif args.sort == 'success_rate':
                patterns.sort(key=lambda p: p.success_rate, reverse=True)
            elif args.sort == 'usage_count':
                patterns.sort(key=lambda p: p.usage_count, reverse=True)
            elif args.sort == 'complexity':
                complexity_order = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}
                patterns.sort(key=lambda p: complexity_order.get(p.complexity, 2))
            
            # Display patterns
            for i, pattern in enumerate(patterns, 1):
                if args.verbose:
                    print(f"{i}. {pattern.name} ({pattern.pattern_id})")
                    print(f"   Domain: {pattern.domain}")
                    print(f"   Complexity: {pattern.complexity}")
                    print(f"   Success Rate: {pattern.success_rate:.2f}")
                    print(f"   Usage Count: {pattern.usage_count}")
                    print(f"   Tags: {', '.join(pattern.tags)}")
                    if pattern.description:
                        print(f"   Description: {pattern.description[:100]}{'...' if len(pattern.description) > 100 else ''}")
                    print()
                else:
                    success_indicator = "✅" if pattern.success_rate >= 0.8 else "⚠️" if pattern.success_rate >= 0.6 else "❌"
                    print(f"{i:3}. {success_indicator} {pattern.name:<30} [{pattern.complexity:<10}] ({pattern.success_rate:.2f})")
            
            return 0
            
        except Exception as e:
            print(f"❌ List failed: {e}")
            if args.verbose:
                self.logger.exception("List error details")
            return 1
    
    def show_stats(self, args) -> int:
        """
        Show pattern statistics.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            stats = self.portability.get_export_stats()
            
            print("📊 Pattern Statistics:")
            print("="*50)
            print(f"Total patterns: {stats['total_patterns']}")
            print(f"Average success rate: {stats['avg_success_rate']:.2f}")
            print(f"Most successful domain: {stats['most_successful_domain']}")
            print(f"Patterns with examples: {stats['patterns_with_examples']}")
            
            print(f"\n📈 Complexity Breakdown:")
            for complexity, count in stats['complexity_breakdown'].items():
                percentage = (count / stats['total_patterns']) * 100 if stats['total_patterns'] > 0 else 0
                print(f"  {complexity:<12}: {count:3} ({percentage:5.1f}%)")
            
            print(f"\n🏷️ Domain Breakdown:")
            for domain, count in stats['domain_breakdown'].items():
                percentage = (count / stats['total_patterns']) * 100 if stats['total_patterns'] > 0 else 0
                print(f"  {domain:<12}: {count:3} ({percentage:5.1f}%)")
            
            return 0
            
        except Exception as e:
            print(f"❌ Stats failed: {e}")
            if args.verbose:
                self.logger.exception("Stats error details")
            return 1
    
    def validate_file(self, args) -> int:
        """
        Validate an export file without importing.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Read and parse file
            import_file = Path(args.file)
            if not import_file.exists():
                print(f"❌ File not found: {args.file}")
                return 1
            
            with open(import_file, 'r') as f:
                data = json.load(f)
            
            print(f"Validating export file: {args.file}")
            print("="*50)
            
            # Check version
            version = data.get('version', 'missing')
            is_valid_version = self.portability.validate_version(version)
            print(f"Version: {version} {'✅' if is_valid_version else '❌'}")
            
            # Check basic structure
            has_patterns = 'patterns' in data and isinstance(data['patterns'], list)
            pattern_count = len(data.get('patterns', []))
            print(f"Patterns array: {'✅' if has_patterns else '❌'}")
            print(f"Pattern count: {pattern_count}")
            
            if not has_patterns:
                return 1
            
            # Validate individual patterns
            patterns = data['patterns']
            validation_results = self.portability.validate_patterns(patterns)
            
            valid_count = sum(1 for r in validation_results if r['valid'])
            invalid_count = len(validation_results) - valid_count
            
            print(f"\n📋 Pattern Validation:")
            print(f"Valid patterns: {valid_count} ✅")
            print(f"Invalid patterns: {invalid_count} {'❌' if invalid_count > 0 else ''}")
            
            # Show invalid patterns
            if invalid_count > 0:
                print(f"\n❌ Invalid patterns:")
                for result in validation_results:
                    if not result['valid']:
                        pattern_id = result.get('pattern_id', f"index-{result['index']}")
                        print(f"\n  Pattern: {pattern_id}")
                        for error in result['errors']:
                            print(f"    ❌ {error}")
                        for warning in result['warnings']:
                            print(f"    ⚠️  {warning}")
            
            # Check metadata
            if 'metadata' in data:
                metadata = data['metadata']
                print(f"\n📊 Export Metadata:")
                print(f"Source project: {metadata.get('source_project', 'unknown')}")
                print(f"Export timestamp: {data.get('exported_at', 'unknown')}")
                print(f"Success rate avg: {metadata.get('success_rate_avg', 0):.2f}")
            
            overall_valid = is_valid_version and has_patterns and invalid_count == 0
            print(f"\n{'✅ File is valid for import' if overall_valid else '❌ File has validation errors'}")
            
            return 0 if overall_valid else 1
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return 1
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            if args.verbose:
                self.logger.exception("Validation error details")
            return 1
    
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Pattern Export/Import CLI - Issue #80",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Export all patterns:
    python pattern_export_import_cli.py export -o backup.json
    
  Export specific patterns:
    python pattern_export_import_cli.py export -p "pattern-001,pattern-002" -o selected.json
    
  Import with conservative strategy:
    python pattern_export_import_cli.py import backup.json -s conservative
    
  Import with overwrite strategy:
    python pattern_export_import_cli.py import backup.json -s overwrite -v
    
  List patterns by domain:
    python pattern_export_import_cli.py list -d web -v
    
  Validate export file:
    python pattern_export_import_cli.py validate backup.json
"""
        )
        
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose output')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export patterns to JSON')
        export_parser.add_argument('-o', '--output', help='Output file path')
        export_parser.add_argument('-p', '--patterns', 
                                 help='Comma-separated list of pattern IDs to export')
        export_parser.add_argument('--pretty', action='store_true',
                                 help='Pretty-print JSON output when printing to console')
        export_parser.add_argument('-v', '--verbose', action='store_true',
                                 help='Enable verbose output')
        
        # Import command
        import_parser = subparsers.add_parser('import', help='Import patterns from JSON')
        import_parser.add_argument('file', help='JSON file to import')
        import_parser.add_argument('-s', '--strategy', 
                                 choices=['conservative', 'overwrite', 'merge', 'versioned'],
                                 default='conservative',
                                 help='Merge strategy for conflicts (default: conservative)')
        import_parser.add_argument('--validate-only', action='store_true',
                                 help='Only validate patterns without importing')
        import_parser.add_argument('-v', '--verbose', action='store_true',
                                 help='Enable verbose output')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List available patterns')
        list_parser.add_argument('-d', '--domain', help='Filter by domain')
        list_parser.add_argument('-c', '--complexity', 
                               choices=['low', 'medium', 'high', 'very-high'],
                               help='Filter by complexity')
        list_parser.add_argument('--sort', 
                               choices=['name', 'success_rate', 'usage_count', 'complexity'],
                               default='name',
                               help='Sort patterns by field (default: name)')
        list_parser.add_argument('-v', '--verbose', action='store_true',
                               help='Enable verbose output')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show pattern statistics')
        stats_parser.add_argument('-v', '--verbose', action='store_true',
                                help='Enable verbose output')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate export file')
        validate_parser.add_argument('file', help='JSON file to validate')
        validate_parser.add_argument('-v', '--verbose', action='store_true',
                                   help='Enable verbose output')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 1
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Execute command
        if args.command == 'export':
            return self.export_patterns(args)
        elif args.command == 'import':
            return self.import_patterns(args)
        elif args.command == 'list':
            return self.list_patterns(args)
        elif args.command == 'stats':
            return self.show_stats(args)
        elif args.command == 'validate':
            return self.validate_file(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1


if __name__ == "__main__":
    cli = PatternCLI()
    exit_code = cli.main()
    sys.exit(exit_code)