#!/usr/bin/env python3
"""
Knowledge Base Cleaner for RIF Deployment

This script cleans the RIF knowledge base by:
1. Creating backups before cleanup
2. Removing development-specific artifacts
3. Preserving valuable patterns and learnings
4. Compressing and organizing knowledge files
5. Providing rollback capability

Usage:
    python scripts/clean_knowledge_for_deploy.py [--dry-run] [--backup-dir PATH]
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import tarfile
import gzip
import logging
from typing import Dict, List, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeCleaner:
    """Handles cleaning and organizing the RIF knowledge base for deployment"""
    
    def __init__(self, knowledge_dir: str, backup_dir: Optional[str] = None):
        self.knowledge_dir = Path(knowledge_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else Path("knowledge_backup")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Directories to completely remove (RIF development specific)
        self.remove_dirs = {
            'audits',           # 4.2MB - RIF development audit logs
            'checkpoints',      # 1.8MB - RIF issue checkpoints  
            'issues',          # 580KB - RIF-specific issue resolutions
            'metrics',         # 376KB - RIF development metrics
            'enforcement_logs', # 40KB - Orchestration enforcement logs
            'evidence_collection', # 84KB - Evidence packages
            'migrated_backup', # 24KB - Migration artifacts
            'coordination',    # 12KB - RIF coordination logs
            'false_positive_detection', # 12KB - Testing artifacts
            'demo_monitoring', # 20KB - Demo files
            'state_cleanup',   # 44KB - State cleanup logs
            'state_transitions', # 12KB - Transition logs
            'recovery',        # 8KB - Recovery logs
            'reports'          # 4KB - Report files
        }
        
        # Directories to selectively filter
        self.filter_dirs = {
            'patterns',        # Keep reusable patterns, remove issue-specific
            'decisions',       # Keep framework decisions, remove RIF-specific  
            'learning',        # Keep methodology, remove RIF issue learnings
            'analysis',        # Keep reusable analysis patterns
            'validation'       # Keep validation frameworks
        }
        
        # Directories to preserve as-is (core framework components)
        self.preserve_dirs = {
            'arbitration',     # 208KB - Arbitration system code
            'consensus',       # 172KB - Consensus system code  
            'conversations',   # 272KB - Conversation capture system
            'context',         # 96KB - Context optimization
            'database',        # 192KB - Database integration
            'embeddings',      # 104KB - Embedding system
            'extraction',      # 196KB - Entity extraction
            'indexing',        # 152KB - Indexing system
            'integration',     # 160KB - System integration
            'monitoring',      # 132KB - Monitoring framework
            'parsing',         # 124KB - Code parsing
            'pattern_application', # 2.0MB - Pattern application engine
            'pattern_extraction',  # 220KB - Pattern extraction
            'pattern_matching',    # 160KB - Pattern matching
            'query',           # 148KB - Query system
            'research',        # 160KB - Research framework
            'schema',          # 148KB - Schema definitions
            'agents',          # 8KB - Agent registry
            'capabilities',    # 20KB - Capability matrix
            'lightrag_archive', # 4KB - Archive metadata
            'quality_metrics', # 24KB - Quality metrics framework
            'plans'            # 24KB - Planning framework
        }

    def analyze_size(self) -> Dict:
        """Analyze current knowledge base size and composition"""
        logger.info("Analyzing knowledge base size and composition...")
        
        stats = {
            'total_size_mb': 0,
            'total_files': 0,
            'subdirs': {}
        }
        
        for subdir in self.knowledge_dir.iterdir():
            if subdir.is_dir():
                size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                count = len(list(subdir.rglob('*'))) - len(list(subdir.rglob('*')))  # Files only
                file_count = len([f for f in subdir.rglob('*') if f.is_file()])
                
                stats['subdirs'][subdir.name] = {
                    'size_mb': round(size / (1024 * 1024), 2),
                    'file_count': file_count,
                    'action': self._get_action_for_dir(subdir.name)
                }
                stats['total_size_mb'] += size
                stats['total_files'] += file_count
        
        # Include individual files
        individual_files = [f for f in self.knowledge_dir.iterdir() if f.is_file()]
        for file in individual_files:
            size = file.stat().st_size
            stats['total_size_mb'] += size
            stats['total_files'] += 1
        
        stats['total_size_mb'] = round(stats['total_size_mb'] / (1024 * 1024), 2)
        
        return stats

    def _get_action_for_dir(self, dirname: str) -> str:
        """Determine what action will be taken for a directory"""
        if dirname in self.remove_dirs:
            return "REMOVE"
        elif dirname in self.filter_dirs:
            return "FILTER"
        elif dirname in self.preserve_dirs:
            return "PRESERVE"
        elif dirname == 'chromadb':
            return "RESET"
        else:
            return "REVIEW"

    def create_backup(self) -> Path:
        """Create compressed backup of current knowledge base"""
        logger.info(f"Creating backup in {self.backup_dir}...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = self.backup_dir / f"knowledge_backup_{self.timestamp}.tar.gz"
        
        with tarfile.open(backup_file, "w:gz") as tar:
            tar.add(self.knowledge_dir, arcname="knowledge")
        
        backup_size_mb = round(backup_file.stat().st_size / (1024 * 1024), 2)
        logger.info(f"Backup created: {backup_file} ({backup_size_mb}MB)")
        
        return backup_file

    def export_valuable_knowledge(self) -> Dict:
        """Export valuable patterns and learnings for preservation"""
        logger.info("Exporting valuable knowledge patterns...")
        
        export = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'source': 'RIF Knowledge Base',
                'version': '1.0'
            },
            'patterns': [],
            'decisions': [],
            'learnings': []
        }
        
        # Export reusable patterns
        patterns_dir = self.knowledge_dir / 'patterns'
        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob('*.json'):
                if self._is_reusable_pattern(pattern_file):
                    try:
                        with open(pattern_file, 'r') as f:
                            pattern_data = json.load(f)
                            pattern_data['source_file'] = pattern_file.name
                            export['patterns'].append(pattern_data)
                    except Exception as e:
                        logger.warning(f"Could not export pattern {pattern_file}: {e}")
        
        # Export framework decisions
        decisions_dir = self.knowledge_dir / 'decisions'
        if decisions_dir.exists():
            for decision_file in decisions_dir.glob('*.json'):
                if self._is_framework_decision(decision_file):
                    try:
                        with open(decision_file, 'r') as f:
                            decision_data = json.load(f)
                            decision_data['source_file'] = decision_file.name
                            export['decisions'].append(decision_data)
                    except Exception as e:
                        logger.warning(f"Could not export decision {decision_file}: {e}")
        
        # Export methodology learnings
        learning_dir = self.knowledge_dir / 'learning'
        if learning_dir.exists():
            for learning_file in learning_dir.glob('*.json'):
                if self._is_methodology_learning(learning_file):
                    try:
                        with open(learning_file, 'r') as f:
                            learning_data = json.load(f)
                            learning_data['source_file'] = learning_file.name
                            export['learnings'].append(learning_data)
                    except Exception as e:
                        logger.warning(f"Could not export learning {learning_file}: {e}")
        
        return export

    def _is_reusable_pattern(self, pattern_file: Path) -> bool:
        """Determine if a pattern is reusable across projects"""
        filename = pattern_file.name.lower()
        
        # Keep patterns with these prefixes (reusable)
        keep_prefixes = [
            'database-resilience-',
            'api-resilience-',
            'context-optimization-',
            'error-handling-',
            'testing-strategy-',
            'documentation-',
            'quality-gate-',
            'monitoring-',
            'consensus-',
            'arbitration-',
            'pattern-application-',
            'enterprise-'
        ]
        
        # Remove patterns with these prefixes (RIF-specific)
        remove_prefixes = [
            'issue-',
            'rif-',
            'claude-code-',
            'comprehensive-implementation-learnings-'
        ]
        
        # First check if it should be removed
        if any(filename.startswith(prefix) for prefix in remove_prefixes):
            return False
            
        # Then check if it should be kept
        if any(filename.startswith(prefix) for prefix in keep_prefixes):
            return True
            
        # For remaining patterns, check content
        try:
            with open(pattern_file, 'r') as f:
                content = f.read().lower()
                # Keep if it contains reusable concepts
                reusable_indicators = [
                    'database', 'api', 'resilience', 'pattern', 'framework',
                    'architecture', 'strategy', 'methodology', 'best practice'
                ]
                return any(indicator in content for indicator in reusable_indicators)
        except:
            return False

    def _is_framework_decision(self, decision_file: Path) -> bool:
        """Determine if a decision is about framework architecture"""
        filename = decision_file.name.lower()
        
        # Keep architectural decisions
        if any(term in filename for term in [
            'architecture', 'framework', 'system', 'integration',
            'consensus', 'arbitration', 'pattern', 'quality-gate'
        ]):
            return True
        
        # Remove issue-specific decisions
        if filename.startswith('issue-') and any(term in filename for term in [
            'implementation', 'planning', 'validation'
        ]):
            return False
            
        return True

    def _is_methodology_learning(self, learning_file: Path) -> bool:
        """Determine if a learning contains reusable methodology"""
        filename = learning_file.name.lower()
        
        # Remove issue-specific learnings
        if filename.startswith('issue-') or 'rif-learner' in filename:
            return False
        
        # Keep methodology learnings
        methodology_indicators = [
            'methodology', 'framework', 'architecture', 'system',
            'comprehensive', 'pattern', 'strategy', 'approach'
        ]
        
        return any(indicator in filename for indicator in methodology_indicators)

    def clean_for_deployment(self, dry_run: bool = False) -> Dict:
        """Clean knowledge base for deployment"""
        logger.info(f"Starting deployment cleanup {'(DRY RUN)' if dry_run else ''}...")
        
        cleanup_stats = {
            'removed_dirs': [],
            'filtered_dirs': [],
            'preserved_dirs': [],
            'files_removed': 0,
            'files_kept': 0,
            'size_before_mb': 0,
            'size_after_mb': 0
        }
        
        # Get initial size
        initial_stats = self.analyze_size()
        cleanup_stats['size_before_mb'] = initial_stats['total_size_mb']
        
        # Remove development-specific directories
        for dir_name in self.remove_dirs:
            dir_path = self.knowledge_dir / dir_name
            if dir_path.exists():
                if not dry_run:
                    shutil.rmtree(dir_path)
                cleanup_stats['removed_dirs'].append(dir_name)
                logger.info(f"{'Would remove' if dry_run else 'Removed'} {dir_name}/")
        
        # Filter selective directories
        for dir_name in self.filter_dirs:
            dir_path = self.knowledge_dir / dir_name
            if dir_path.exists():
                filtered_count = self._filter_directory(dir_path, dry_run)
                cleanup_stats['filtered_dirs'].append(f"{dir_name} ({filtered_count} files)")
                logger.info(f"{'Would filter' if dry_run else 'Filtered'} {dir_name}/ - {filtered_count} files processed")
        
        # Log preserved directories
        for dir_name in self.preserve_dirs:
            dir_path = self.knowledge_dir / dir_name
            if dir_path.exists():
                cleanup_stats['preserved_dirs'].append(dir_name)
        
        # Reset chromadb
        chromadb_path = self.knowledge_dir / 'chromadb'
        if chromadb_path.exists():
            if not dry_run:
                shutil.rmtree(chromadb_path)
                chromadb_path.mkdir()
                logger.info("Reset chromadb/ directory")
            else:
                logger.info("Would reset chromadb/ directory")
        
        # Reset conversations database
        conv_db = self.knowledge_dir / 'conversations.duckdb'
        if conv_db.exists():
            if not dry_run:
                conv_db.unlink()
                logger.info("Removed conversations.duckdb")
            else:
                logger.info("Would remove conversations.duckdb")
        
        # Clean individual files
        self._clean_individual_files(dry_run)
        
        # Get final size
        if not dry_run:
            final_stats = self.analyze_size()
            cleanup_stats['size_after_mb'] = final_stats['total_size_mb']
        
        return cleanup_stats

    def _filter_directory(self, dir_path: Path, dry_run: bool) -> int:
        """Filter files in a directory based on content"""
        filtered_count = 0
        
        for file_path in dir_path.rglob('*.json'):
            should_remove = False
            
            if dir_path.name == 'patterns':
                should_remove = not self._is_reusable_pattern(file_path)
            elif dir_path.name == 'decisions':
                should_remove = not self._is_framework_decision(file_path)
            elif dir_path.name == 'learning':
                should_remove = not self._is_methodology_learning(file_path)
            elif dir_path.name == 'analysis':
                # Keep only reusable analysis patterns
                should_remove = file_path.name.startswith('issue-')
            
            if should_remove:
                if not dry_run:
                    file_path.unlink()
                filtered_count += 1
        
        return filtered_count

    def _clean_individual_files(self, dry_run: bool):
        """Clean individual files in knowledge root"""
        files_to_remove = [
            'events.jsonl',
            'issue_closure_prevention_log.json',
            'pending_user_validations.json',
            'user_validation_log.json',
            'cutover_config.json',
            'migration_state.json',
            'migration_final_report.json',
            'demo_health_registry.json',
            'demo_performance_metrics.db',
            'shadow-mode.log',
            'orchestration.duckdb',
            'orchestration.duckdb.wal'
        ]
        
        for filename in files_to_remove:
            file_path = self.knowledge_dir / filename
            if file_path.exists():
                if not dry_run:
                    file_path.unlink()
                logger.info(f"{'Would remove' if dry_run else 'Removed'} {filename}")

    def generate_cleanup_report(self, cleanup_stats: Dict, export_data: Dict) -> str:
        """Generate cleanup report"""
        report = f"""
# Knowledge Base Cleanup Report
Generated: {datetime.now().isoformat()}

## Summary
- **Size Before**: {cleanup_stats['size_before_mb']:.2f}MB
- **Size After**: {cleanup_stats['size_after_mb']:.2f}MB  
- **Space Saved**: {cleanup_stats['size_before_mb'] - cleanup_stats['size_after_mb']:.2f}MB ({((cleanup_stats['size_before_mb'] - cleanup_stats['size_after_mb']) / cleanup_stats['size_before_mb'] * 100):.1f}%)

## Actions Taken

### Removed Directories ({len(cleanup_stats['removed_dirs'])})
{chr(10).join(f'- {d}' for d in cleanup_stats['removed_dirs'])}

### Filtered Directories ({len(cleanup_stats['filtered_dirs'])})
{chr(10).join(f'- {d}' for d in cleanup_stats['filtered_dirs'])}

### Preserved Directories ({len(cleanup_stats['preserved_dirs'])})  
{chr(10).join(f'- {d}' for d in cleanup_stats['preserved_dirs'])}

## Exported Knowledge
- **Patterns**: {len(export_data['patterns'])} reusable patterns
- **Decisions**: {len(export_data['decisions'])} framework decisions  
- **Learnings**: {len(export_data['learnings'])} methodology learnings

## Next Steps
1. Test the cleaned knowledge base
2. Deploy using `scripts/init_project_knowledge.py`
3. Keep backup for rollback if needed
"""
        return report

    def create_rollback_script(self, backup_file: Path) -> Path:
        """Create rollback script to restore from backup"""
        rollback_script = self.backup_dir / f"rollback_{self.timestamp}.sh"
        
        script_content = f"""#!/bin/bash
# Rollback script for knowledge base cleanup
# Generated: {datetime.now().isoformat()}

set -e

echo "Rolling back knowledge base from backup..."
echo "Backup file: {backup_file}"

# Remove current knowledge directory
if [ -d "{self.knowledge_dir}" ]; then
    echo "Removing current knowledge directory..."
    rm -rf "{self.knowledge_dir}"
fi

# Extract backup
echo "Restoring from backup..."
tar -xzf "{backup_file}" -C "{self.knowledge_dir.parent}"

echo "Rollback complete!"
echo "Knowledge base restored from {backup_file}"
"""
        
        with open(rollback_script, 'w') as f:
            f.write(script_content)
        
        rollback_script.chmod(0o755)
        logger.info(f"Created rollback script: {rollback_script}")
        
        return rollback_script


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Clean RIF knowledge base for deployment")
    parser.add_argument(
        "--knowledge-dir", 
        default="knowledge",
        help="Path to knowledge directory (default: knowledge)"
    )
    parser.add_argument(
        "--backup-dir",
        default="knowledge_backup", 
        help="Path to backup directory (default: knowledge_backup)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without making changes"
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true", 
        help="Skip backup creation (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = KnowledgeCleaner(args.knowledge_dir, args.backup_dir)
    
    # Analyze current state
    logger.info("Analyzing current knowledge base...")
    initial_stats = cleaner.analyze_size()
    logger.info(f"Current size: {initial_stats['total_size_mb']:.2f}MB ({initial_stats['total_files']} files)")
    
    # Create backup (unless skipped)
    backup_file = None
    if not args.skip_backup and not args.dry_run:
        backup_file = cleaner.create_backup()
    
    # Export valuable knowledge
    logger.info("Exporting valuable knowledge...")
    export_data = cleaner.export_valuable_knowledge()
    
    # Save exported knowledge
    if not args.dry_run:
        export_file = Path(args.backup_dir) / f"valuable_knowledge_{cleaner.timestamp}.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Valuable knowledge exported to: {export_file}")
    
    # Perform cleanup
    cleanup_stats = cleaner.clean_for_deployment(dry_run=args.dry_run)
    
    # Generate report
    report = cleaner.generate_cleanup_report(cleanup_stats, export_data)
    
    # Save report
    if not args.dry_run:
        report_file = Path(args.backup_dir) / f"cleanup_report_{cleaner.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Cleanup report saved to: {report_file}")
    
    # Create rollback script
    if backup_file and not args.dry_run:
        rollback_script = cleaner.create_rollback_script(backup_file)
    
    # Print summary
    print("\n" + "="*60)
    print("KNOWLEDGE BASE CLEANUP SUMMARY")  
    print("="*60)
    print(report)
    
    if args.dry_run:
        print("This was a DRY RUN - no changes were made.")
        print("Remove --dry-run to perform actual cleanup.")


if __name__ == "__main__":
    main()