#!/usr/bin/env python3
"""
GitignoreManager - Manage gitignore files for RIF deployment and development modes

This script provides functionality to switch between RIF development mode
(tracks all files for RIF framework development) and deployment mode
(excludes RIF framework files for clean project deployment).
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class GitignoreManager:
    """Manages gitignore files for different RIF modes."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize GitignoreManager with project root path."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.rif_gitignore = self.project_root / '.gitignore.rif'
        self.project_gitignore = self.project_root / '.gitignore'
        self.deployment_gitignore = self.project_root / '.gitignore.deployment'
        self.backup_dir = self.project_root / '.gitignore-backups'
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_current_gitignore(self) -> Path:
        """Backup current .gitignore file with timestamp."""
        if not self.project_gitignore.exists():
            print("No .gitignore file exists to backup")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f'.gitignore.backup.{timestamp}'
        
        shutil.copy(self.project_gitignore, backup_path)
        print(f"✅ Backed up current .gitignore to: {backup_path}")
        return backup_path
    
    def setup_for_deployment(self) -> bool:
        """Setup gitignore for deployed project (exclude RIF framework files)."""
        try:
            if not self.deployment_gitignore.exists():
                print("❌ Error: .gitignore.deployment template not found")
                print(f"   Expected location: {self.deployment_gitignore}")
                return False
            
            # Backup existing gitignore
            self.backup_current_gitignore()
            
            # Copy deployment gitignore as main gitignore
            shutil.copy(self.deployment_gitignore, self.project_gitignore)
            print("✅ Switched to deployment mode gitignore")
            
            # Add project-specific patterns if needed
            self.add_project_patterns()
            
            print("🚀 Deployment mode activated!")
            print("   - RIF framework files will be excluded from git")
            print("   - Project development files will be tracked")
            print("   - Use 'switch_to_development_mode()' to reverse")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up deployment mode: {e}")
            return False
    
    def switch_to_development_mode(self) -> bool:
        """Switch back to RIF development mode (track all files)."""
        try:
            # Backup current gitignore
            self.backup_current_gitignore()
            
            # If RIF development gitignore exists, use it
            if self.rif_gitignore.exists():
                shutil.copy(self.rif_gitignore, self.project_gitignore)
                print("✅ Switched to RIF development mode gitignore")
            else:
                # Create minimal RIF development gitignore
                self.create_rif_development_gitignore()
                print("✅ Created RIF development mode gitignore")
            
            print("🛠️  Development mode activated!")
            print("   - All RIF framework files will be tracked")
            print("   - Use 'setup_for_deployment()' to switch back")
            
            return True
            
        except Exception as e:
            print(f"❌ Error switching to development mode: {e}")
            return False
    
    def create_rif_development_gitignore(self):
        """Create minimal gitignore for RIF development."""
        development_content = """# RIF Development Mode Gitignore
# This mode tracks RIF framework files for development

# Standard development ignores
__pycache__/
*.py[cod]
*$py.class
*.so
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# IDE
.vscode/
.idea/
*.sublime-*

# Environment
.env
.env.local
.env.*.local

# Logs  
*.log
logs/

# Testing
coverage/
.coverage
htmlcov/
.pytest_cache/

# Build outputs
dist/
build/

# Temporary files
*.tmp
*.temp
*.bak
*.backup

# OS generated
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
desktop.ini

# Add project-specific development ignores below
"""
        with open(self.project_gitignore, 'w') as f:
            f.write(development_content)
    
    def add_project_patterns(self):
        """Add project-specific ignore patterns to deployment gitignore."""
        project_patterns = self.detect_project_patterns()
        
        if project_patterns:
            with open(self.project_gitignore, 'a') as f:
                f.write("\n# ==================================================\n")
                f.write("# Auto-detected Project-Specific Ignores\n")
                f.write("# ==================================================\n")
                
                for pattern in project_patterns:
                    f.write(f"{pattern}\n")
                    
            print(f"✅ Added {len(project_patterns)} auto-detected project patterns")
    
    def detect_project_patterns(self) -> List[str]:
        """Detect project-specific patterns based on project structure."""
        patterns = []
        
        # Check for common project indicators
        if (self.project_root / 'package.json').exists():
            patterns.extend([
                "# Node.js project detected",
                "node_modules/",
                ".npm/",
                "npm-debug.log*"
            ])
        
        if (self.project_root / 'requirements.txt').exists():
            patterns.extend([
                "# Python project detected", 
                "__pycache__/",
                "*.pyc",
                ".pytest_cache/"
            ])
        
        if (self.project_root / 'Cargo.toml').exists():
            patterns.extend([
                "# Rust project detected",
                "target/",
                "**/*.rs.bk"
            ])
        
        if (self.project_root / 'go.mod').exists():
            patterns.extend([
                "# Go project detected",
                "vendor/"
            ])
        
        if (self.project_root / 'pom.xml').exists():
            patterns.extend([
                "# Java Maven project detected",
                "target/",
                ".m2/"
            ])
            
        return patterns
    
    def get_mode_status(self) -> dict:
        """Get current gitignore mode status."""
        status = {
            "current_mode": "unknown",
            "gitignore_exists": self.project_gitignore.exists(),
            "deployment_template_exists": self.deployment_gitignore.exists(),
            "rif_template_exists": self.rif_gitignore.exists(),
            "backup_count": len(list(self.backup_dir.glob("*.backup.*"))) if self.backup_dir.exists() else 0
        }
        
        if not self.project_gitignore.exists():
            status["current_mode"] = "none"
        else:
            # Try to detect mode by checking content
            with open(self.project_gitignore, 'r') as f:
                content = f.read()
                
            if "RIF Framework Files (DO NOT COMMIT)" in content:
                status["current_mode"] = "deployment"
            elif "RIF Development Mode" in content:
                status["current_mode"] = "development"
            else:
                status["current_mode"] = "custom"
        
        return status
    
    def print_status(self):
        """Print current gitignore configuration status."""
        status = self.get_mode_status()
        
        print("\n" + "="*50)
        print("GitignoreManager Status")
        print("="*50)
        print(f"Project Root: {self.project_root}")
        print(f"Current Mode: {status['current_mode']}")
        print(f"Gitignore Exists: {'✅' if status['gitignore_exists'] else '❌'}")
        print(f"Deployment Template: {'✅' if status['deployment_template_exists'] else '❌'}")
        print(f"RIF Dev Template: {'✅' if status['rif_template_exists'] else '❌'}")
        print(f"Backups Available: {status['backup_count']}")
        print("="*50)
    
    def restore_from_backup(self, backup_name: Optional[str] = None) -> bool:
        """Restore gitignore from backup."""
        try:
            if not self.backup_dir.exists():
                print("❌ No backup directory found")
                return False
            
            backups = sorted(list(self.backup_dir.glob("*.backup.*")))
            
            if not backups:
                print("❌ No backups found")
                return False
            
            if backup_name:
                backup_path = self.backup_dir / backup_name
                if not backup_path.exists():
                    print(f"❌ Backup {backup_name} not found")
                    return False
            else:
                # Use most recent backup
                backup_path = backups[-1]
            
            # Backup current before restore
            self.backup_current_gitignore()
            
            # Restore from backup
            shutil.copy(backup_path, self.project_gitignore)
            print(f"✅ Restored gitignore from: {backup_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            return False
    
    def list_backups(self):
        """List available gitignore backups."""
        if not self.backup_dir.exists():
            print("No backup directory found")
            return
        
        backups = sorted(list(self.backup_dir.glob("*.backup.*")))
        
        if not backups:
            print("No backups found")
            return
        
        print("\nAvailable Gitignore Backups:")
        print("-" * 30)
        
        for backup in backups:
            stat = backup.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            print(f"{backup.name:<25} {timestamp.strftime('%Y-%m-%d %H:%M')} ({size} bytes)")


def main():
    """Command line interface for GitignoreManager."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage gitignore files for RIF deployment and development modes"
    )
    parser.add_argument(
        "--project-root", 
        help="Path to project root (default: current directory)",
        default="."
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deployment mode command
    deploy_parser = subparsers.add_parser(
        'deploy', 
        help='Switch to deployment mode (exclude RIF framework files)'
    )
    
    # Development mode command
    dev_parser = subparsers.add_parser(
        'develop', 
        help='Switch to development mode (track all files)'
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status', 
        help='Show current gitignore mode status'
    )
    
    # Backup commands
    backup_parser = subparsers.add_parser(
        'backup', 
        help='Backup current gitignore'
    )
    
    restore_parser = subparsers.add_parser(
        'restore', 
        help='Restore gitignore from backup'
    )
    restore_parser.add_argument(
        '--backup-name', 
        help='Specific backup to restore (default: most recent)'
    )
    
    list_parser = subparsers.add_parser(
        'list-backups', 
        help='List available backups'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = GitignoreManager(args.project_root)
    
    if args.command == 'deploy':
        success = manager.setup_for_deployment()
        sys.exit(0 if success else 1)
        
    elif args.command == 'develop':
        success = manager.switch_to_development_mode()
        sys.exit(0 if success else 1)
        
    elif args.command == 'status':
        manager.print_status()
        
    elif args.command == 'backup':
        backup_path = manager.backup_current_gitignore()
        sys.exit(0 if backup_path else 1)
        
    elif args.command == 'restore':
        success = manager.restore_from_backup(args.backup_name)
        sys.exit(0 if success else 1)
        
    elif args.command == 'list-backups':
        manager.list_backups()


if __name__ == '__main__':
    main()