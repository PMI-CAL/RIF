#!/usr/bin/env python3
"""
Path Migration Script for RIF
Automatically updates hard-coded paths to use PathResolver
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'claude' / 'commands'))

try:
    from path_resolver import PathResolver
except ImportError:
    print("❌ PathResolver not found. Please ensure path_resolver.py is in claude/commands/")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathMigrator:
    """
    Migrates hard-coded paths to use PathResolver
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.resolver = PathResolver()
        
        # Common hard-coded path patterns to replace
        self.path_patterns = [
            # Direct hard-coded paths
            (r'/Users/cal/DEV/RIF', '${PROJECT_ROOT}'),
            (r'/Users/[^/]+/[^/]+/RIF', '${PROJECT_ROOT}'),
            
            # Knowledge base paths
            (r'/Users/cal/DEV/RIF/knowledge', '${PROJECT_ROOT}/.rif/knowledge'),
            (r'/Users/[^/]+/[^/]+/RIF/knowledge', '${PROJECT_ROOT}/.rif/knowledge'),
            
            # Config paths  
            (r'/Users/cal/DEV/RIF/config', '${PROJECT_ROOT}/config'),
            (r'/Users/[^/]+/[^/]+/RIF/config', '${PROJECT_ROOT}/config'),
            
            # Claude commands paths
            (r'/Users/cal/DEV/RIF/claude/commands', '${PROJECT_ROOT}/.rif/commands'),
            (r'/Users/[^/]+/[^/]+/RIF/claude/commands', '${PROJECT_ROOT}/.rif/commands'),
        ]
        
        # Python code patterns for PathResolver integration
        self.python_patterns = [
            # Replace path concatenation with resolver
            (
                r'(\w+_path\s*=\s*)(["\'])([^"\']*)/Users/[^/]+/[^/]+/RIF([^"\']*)\2',
                r'\1resolver.resolve("rif_home") / "\4"'
            ),
            
            # Replace hardcoded knowledge paths
            (
                r'knowledge_path\s*=\s*["\'][^"\']*knowledge["\']',
                'knowledge_path = resolver.resolve("knowledge_base")'
            ),
            
            # Replace hardcoded config paths
            (
                r'config_path\s*=\s*["\'][^"\']*config["\']',
                'config_path = resolver.resolve("config")'
            ),
        ]
        
        self.files_modified = []
        self.backup_dir = self.project_root / '.migration_backups'
    
    def migrate_project(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """
        Migrate all Python files in the project
        
        Args:
            dry_run: If True, only show what would be changed
            
        Returns:
            Dictionary with migration results
        """
        results = {
            'files_checked': [],
            'files_modified': [],
            'patterns_found': [],
            'errors': []
        }
        
        logger.info(f"Starting path migration ({'DRY RUN' if dry_run else 'LIVE MODE'})")
        
        # Find Python files
        python_files = list(self.project_root.rglob('*.py'))
        
        for py_file in python_files:
            # Skip backup directory
            if '.migration_backups' in str(py_file):
                continue
                
            results['files_checked'].append(str(py_file))
            
            try:
                modified, patterns = self.migrate_file(py_file, dry_run)
                if modified:
                    results['files_modified'].append(str(py_file))
                    results['patterns_found'].extend(patterns)
            except Exception as e:
                logger.error(f"Error migrating {py_file}: {e}")
                results['errors'].append(f"{py_file}: {e}")
        
        return results
    
    def migrate_file(self, file_path: Path, dry_run: bool = True) -> Tuple[bool, List[str]]:
        """
        Migrate a single file
        
        Args:
            file_path: Path to the file to migrate
            dry_run: If True, only show what would be changed
            
        Returns:
            Tuple of (was_modified, patterns_found)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"Skipping {file_path}: {e}")
            return False, []
        
        original_content = content
        patterns_found = []
        
        # Apply string replacement patterns
        for pattern, replacement in self.path_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                patterns_found.append(f"String pattern: {match.group()}")
                if not dry_run:
                    content = re.sub(pattern, replacement, content)
        
        # Apply Python-specific patterns
        for pattern, replacement in self.python_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                patterns_found.append(f"Python pattern: {match.group()}")
                if not dry_run:
                    content = re.sub(pattern, replacement, content)
        
        # Check if we need to add PathResolver import
        needs_import = False
        if patterns_found and 'from path_resolver import PathResolver' not in content:
            needs_import = True
            patterns_found.append("Added PathResolver import")
        
        if patterns_found:
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Found patterns in {file_path}:")
            for pattern in patterns_found:
                logger.info(f"  - {pattern}")
            
            if not dry_run:
                # Create backup
                self.create_backup(file_path)
                
                # Add import if needed
                if needs_import:
                    content = self.add_path_resolver_import(content)
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"✅ Migrated {file_path}")
                self.files_modified.append(str(file_path))
        
        return len(patterns_found) > 0, patterns_found
    
    def add_path_resolver_import(self, content: str) -> str:
        """
        Add PathResolver import to Python file content
        
        Args:
            content: File content
            
        Returns:
            Modified content with import added
        """
        # Find the best place to add the import
        lines = content.split('\n')
        
        import_line = "from claude.commands.path_resolver import PathResolver"
        resolver_init = "resolver = PathResolver()"
        
        # Find last import line
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                last_import_idx = i
        
        # Insert after imports
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_line)
            # Add resolver initialization after imports
            lines.insert(last_import_idx + 2, resolver_init)
            lines.insert(last_import_idx + 3, "")
        else:
            # Add at the top after any shebang/docstring
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('#!') or '"""' in line or "'''" in line:
                    insert_idx = i + 1
                else:
                    break
            
            lines.insert(insert_idx, import_line)
            lines.insert(insert_idx + 1, resolver_init)
            lines.insert(insert_idx + 2, "")
        
        return '\n'.join(lines)
    
    def create_backup(self, file_path: Path):
        """
        Create backup of file before modification
        
        Args:
            file_path: Path to file to backup
        """
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
        
        # Create relative backup path
        rel_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / rel_path
        
        # Create parent directories for backup
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file to backup
        import shutil
        shutil.copy2(file_path, backup_path)
    
    def restore_backups(self):
        """
        Restore all files from backups
        """
        if not self.backup_dir.exists():
            logger.warning("No backup directory found")
            return
        
        logger.info("Restoring files from backups...")
        
        for backup_file in self.backup_dir.rglob('*'):
            if backup_file.is_file():
                # Get relative path and restore to original location
                rel_path = backup_file.relative_to(self.backup_dir)
                original_path = self.project_root / rel_path
                
                import shutil
                shutil.copy2(backup_file, original_path)
                logger.info(f"Restored {original_path}")
        
        # Remove backup directory
        import shutil
        shutil.rmtree(self.backup_dir)
        logger.info("Backup directory removed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate hard-coded paths to use PathResolver')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Show what would be changed without modifying files (default)')
    parser.add_argument('--migrate', action='store_true',
                        help='Actually perform the migration')
    parser.add_argument('--restore', action='store_true',
                        help='Restore files from backups')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT,
                        help='Project root directory')
    
    args = parser.parse_args()
    
    # Ensure we don't run in both dry-run and migrate mode
    if args.migrate:
        dry_run = False
    else:
        dry_run = True
    
    migrator = PathMigrator(args.project_root)
    
    if args.restore:
        migrator.restore_backups()
        return
    
    # Run migration
    results = migrator.migrate_project(dry_run=dry_run)
    
    # Print summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print(f"Files checked: {len(results['files_checked'])}")
    print(f"Files with patterns found: {len(results['files_modified'])}")
    print(f"Total patterns found: {len(results['patterns_found'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['files_modified']:
        print("\nFiles that would be modified:" if dry_run else "\nFiles modified:")
        for file_path in results['files_modified']:
            print(f"  📄 {file_path}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    if dry_run and results['files_modified']:
        print(f"\n💡 To actually perform the migration, run:")
        print(f"   python3 {__file__} --migrate")
        print(f"\n💡 To restore backups after migration, run:")
        print(f"   python3 {__file__} --restore")
    
    print("="*60)

if __name__ == "__main__":
    main()