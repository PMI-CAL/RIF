#!/usr/bin/env python3
"""
RIF Branch Cleanup Automation
Cleans up merged branches following retention policies
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to path to import BranchManager
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.branch_manager import BranchManager


def main():
    parser = argparse.ArgumentParser(
        description="RIF Branch Cleanup Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python branch-cleanup.py --dry-run       # Show what would be cleaned
  python branch-cleanup.py --cleanup       # Actually perform cleanup
  python branch-cleanup.py --force         # Force cleanup (ignore age limits)
  python branch-cleanup.py --exclude "release-*" "hotfix-*"  # Exclude patterns
        """
    )
    
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be cleaned without actually doing it")
    parser.add_argument("--cleanup", action="store_true",
                       help="Actually perform the cleanup")
    parser.add_argument("--force", action="store_true",
                       help="Force cleanup regardless of age limits")
    parser.add_argument("--exclude", nargs="*", default=[],
                       help="Additional patterns to exclude from cleanup")
    parser.add_argument("--repo-path", default=".",
                       help="Path to git repository (default: current directory)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dry_run and not args.cleanup:
        print("Error: Must specify either --dry-run or --cleanup")
        sys.exit(1)
    
    if args.dry_run and args.cleanup:
        print("Error: Cannot specify both --dry-run and --cleanup")
        sys.exit(1)
    
    try:
        # Initialize branch manager
        branch_manager = BranchManager(args.repo_path)
        
        # Default exclusion patterns
        default_exclude = ["main", "master", "develop", "staging", "production"]
        exclude_patterns = default_exclude + args.exclude
        
        if args.dry_run:
            print("🔍 DRY RUN: Analyzing branches for cleanup...")
            print(f"📋 Excluding patterns: {exclude_patterns}")
            
            # Get merged branches (simulate cleanup without doing it)
            from claude.commands.branch_manager import subprocess
            import re
            from datetime import datetime
            
            try:
                # Get merged branches
                result = branch_manager._run_git(["branch", "--merged", "main"])
                merged_branches = [line.strip().lstrip('* ') for line in result.stdout.split('\n') if line.strip()]
                merged_branches = [b for b in merged_branches if b != "main"]
                
                print(f"📊 Found {len(merged_branches)} merged branches")
                
                cleanup_candidates = []
                for branch in merged_branches:
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if re.search(pattern, branch):
                            should_exclude = True
                            print(f"⏭️  Skipping {branch} (matches pattern: {pattern})")
                            break
                    
                    if not should_exclude:
                        try:
                            last_commit_date = branch_manager._get_branch_last_commit_date(branch)
                            age_days = (datetime.now() - last_commit_date).days
                            
                            if age_days > 7 or args.force:
                                cleanup_candidates.append((branch, age_days))
                                print(f"🗑️  Would clean: {branch} (age: {age_days} days)")
                            else:
                                print(f"⏰ Too recent: {branch} (age: {age_days} days)")
                        except Exception as e:
                            print(f"⚠️  Error checking {branch}: {str(e)}")
                
                print(f"\n📈 Summary:")
                print(f"   Total merged branches: {len(merged_branches)}")
                print(f"   Cleanup candidates: {len(cleanup_candidates)}")
                print(f"   Would be cleaned: {len(cleanup_candidates)}")
                
                if cleanup_candidates:
                    print(f"\n🚀 To perform cleanup, run:")
                    print(f"   python {__file__} --cleanup")
                
            except Exception as e:
                print(f"❌ Error during dry run: {str(e)}")
                sys.exit(1)
        
        elif args.cleanup:
            print("🧹 CLEANUP: Performing branch cleanup...")
            print(f"📋 Excluding patterns: {exclude_patterns}")
            
            result = branch_manager.cleanup_merged_branches(exclude_patterns)
            
            if result["status"] == "completed":
                print(f"✅ Cleanup completed successfully")
                print(f"📊 Branches checked: {result['candidates_checked']}")
                print(f"🗑️  Branches cleaned: {result['branches_cleaned']}")
                
                if result["cleaned_branches"]:
                    print("\n📋 Cleaned branches:")
                    for branch in result["cleaned_branches"]:
                        print(f"   • {branch}")
                
                if result["errors"]:
                    print("\n⚠️  Errors encountered:")
                    for error in result["errors"]:
                        print(f"   • {error}")
                
                print(f"\n🎯 Branch cleanup completed with {len(result['errors'])} errors")
            else:
                print(f"❌ Cleanup failed: {result.get('message', 'Unknown error')}")
                if result.get("errors"):
                    print("Errors:")
                    for error in result["errors"]:
                        print(f"   • {error}")
                sys.exit(1)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()