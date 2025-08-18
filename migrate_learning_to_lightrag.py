#!/usr/bin/env python3
"""
Migration script to move existing learning content from .md and JSON files to LightRAG vector database.
This script is part of Phase 2 implementation for GitHub issue #10.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Add the lightrag module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lightrag'))

try:
    from lightrag.core.lightrag_core import LightRAGCore
    print("✅ LightRAG core imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LightRAG core: {e}")
    sys.exit(1)


class LearningMigrator:
    """Migrate existing learning content to LightRAG vector database."""
    
    def __init__(self, knowledge_path: str = None):
        """Initialize the migrator with LightRAG core."""
        self.knowledge_path = knowledge_path or "knowledge"
        print(f"🔄 Initializing LightRAG with knowledge path: {self.knowledge_path}")
        
        try:
            self.rag = LightRAGCore(knowledge_path=self.knowledge_path)
            print("✅ LightRAG initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LightRAG: {e}")
            raise
        
        self.migration_stats = {
            "md_files_processed": 0,
            "json_files_processed": 0,
            "patterns_stored": 0,
            "decisions_stored": 0,
            "resolutions_stored": 0,
            "templates_stored": 0,
            "errors": []
        }
    
    def migrate_orchestration_report(self, file_path: str) -> bool:
        """Migrate orchestration report to issue_resolutions collection."""
        print(f"📄 Migrating orchestration report: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract metadata from the markdown content
            metadata = {
                "type": "orchestration_report",
                "source": "orchestration-report-2025-08-18.md",
                "date": "2025-08-18",
                "issues_processed": 2,
                "complexity": "medium",
                "tags": "orchestration,workflow,agent-delegation,state-management"
            }
            
            # Store in issue_resolutions collection
            doc_id = self.rag.store_knowledge(
                collection_name="issue_resolutions",
                content=content,
                metadata=metadata
            )
            
            print(f"✅ Stored orchestration report as {doc_id}")
            self.migration_stats["md_files_processed"] += 1
            self.migration_stats["resolutions_stored"] += 1
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate orchestration report: {e}"
            print(f"❌ {error_msg}")
            self.migration_stats["errors"].append(error_msg)
            return False
    
    def migrate_error_analysis_template(self, file_path: str) -> bool:
        """Migrate error analysis template to patterns collection."""
        print(f"📄 Migrating error analysis template: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract template data for pattern storage
            pattern_data = {
                "title": "Error Analysis Implementation Template",
                "description": "Proven framework for implementing comprehensive error analysis systems",
                "type": "implementation_template",
                "complexity": "very-high",
                "implementation_time": "15-20 hours",
                "success_rate": "94%",
                "reusability_score": 9.2,
                "phases": [
                    "Foundation Infrastructure (6 hours)",
                    "Analysis Engine Development (5 hours)", 
                    "System Integration (3 hours)",
                    "Advanced Features & Testing (4-6 hours)"
                ],
                "key_components": [
                    "Error detection and capture",
                    "Root cause analysis framework",
                    "Adversarial security analysis",
                    "Continuous improvement engine",
                    "Specialized agent architecture"
                ],
                "source": "error-analysis-implementation-template.md",
                "tags": ["template", "error-analysis", "implementation", "framework", "security"]
            }
            
            # Store the template content as a pattern
            metadata = {
                "type": "pattern",
                "subtype": "implementation_template",
                "complexity": "very-high",
                "source": "error-analysis-implementation-template.md",
                "reusability": "very-high",
                "tags": ",".join(pattern_data["tags"])
            }
            
            doc_id = self.rag.store_knowledge(
                collection_name="patterns",
                content=json.dumps(pattern_data, indent=2),
                metadata=metadata
            )
            
            print(f"✅ Stored error analysis template as pattern {doc_id}")
            self.migration_stats["md_files_processed"] += 1
            self.migration_stats["patterns_stored"] += 1
            self.migration_stats["templates_stored"] += 1
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate error analysis template: {e}"
            print(f"❌ {error_msg}")
            self.migration_stats["errors"].append(error_msg)
            return False
    
    def migrate_comprehensive_learnings(self, file_path: str) -> bool:
        """Migrate comprehensive learnings JSON to appropriate collections."""
        print(f"📄 Migrating comprehensive learnings: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                learning_data = json.load(f)
            
            # Store as issue resolution
            resolution_metadata = {
                "type": "issue_resolution",
                "issue_number": 6,
                "issue_title": learning_data.get("issue_title", "Error analysis system implementation"),
                "complexity": learning_data.get("complexity_handled", "very-high"),
                "final_state": learning_data.get("final_state", "complete"),
                "source": "issue-6-comprehensive-learnings.json",
                "tags": "comprehensive-learnings,error-analysis,implementation,patterns"
            }
            
            resolution_doc_id = self.rag.store_knowledge(
                collection_name="issue_resolutions",
                content=json.dumps(learning_data, indent=2),
                metadata=resolution_metadata
            )
            
            print(f"✅ Stored comprehensive learnings as resolution {resolution_doc_id}")
            
            # Extract and store individual patterns
            patterns_stored = 0
            if "technical_learnings" in learning_data and "architecture_insights" in learning_data["technical_learnings"]:
                for insight in learning_data["technical_learnings"]["architecture_insights"]:
                    pattern_data = {
                        "title": f"Architecture Pattern: {insight['insight']}",
                        "description": insight["insight"],
                        "evidence": insight["evidence"],
                        "application": insight["application"],
                        "reusability": insight["reusability"],
                        "source": "issue-6-comprehensive-learnings.json",
                        "category": "architecture_insight"
                    }
                    
                    pattern_metadata = {
                        "type": "pattern",
                        "subtype": "architecture_insight", 
                        "reusability": insight["reusability"].lower().replace(" ", "_"),
                        "source": "issue-6-comprehensive-learnings.json",
                        "tags": "architecture,insight,pattern,learning"
                    }
                    
                    pattern_doc_id = self.rag.store_knowledge(
                        collection_name="patterns",
                        content=json.dumps(pattern_data, indent=2),
                        metadata=pattern_metadata
                    )
                    patterns_stored += 1
            
            print(f"✅ Extracted and stored {patterns_stored} architecture insight patterns")
            
            # Extract and store implementation insights as patterns
            if "technical_learnings" in learning_data and "implementation_insights" in learning_data["technical_learnings"]:
                for insight in learning_data["technical_learnings"]["implementation_insights"]:
                    pattern_data = {
                        "title": f"Implementation Pattern: {insight['insight']}",
                        "description": insight["insight"],
                        "evidence": insight["evidence"],
                        "application": insight["application"],
                        "reusability": insight["reusability"],
                        "source": "issue-6-comprehensive-learnings.json",
                        "category": "implementation_insight"
                    }
                    
                    pattern_metadata = {
                        "type": "pattern",
                        "subtype": "implementation_insight",
                        "reusability": insight["reusability"].lower().replace(" ", "_"),
                        "source": "issue-6-comprehensive-learnings.json", 
                        "tags": "implementation,insight,pattern,learning"
                    }
                    
                    pattern_doc_id = self.rag.store_knowledge(
                        collection_name="patterns",
                        content=json.dumps(pattern_data, indent=2),
                        metadata=pattern_metadata
                    )
                    patterns_stored += 1
            
            print(f"✅ Total patterns extracted: {patterns_stored}")
            
            self.migration_stats["json_files_processed"] += 1
            self.migration_stats["resolutions_stored"] += 1
            self.migration_stats["patterns_stored"] += patterns_stored
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate comprehensive learnings: {e}"
            print(f"❌ {error_msg}")
            self.migration_stats["errors"].append(error_msg)
            return False
    
    def migrate_reusable_components(self, file_path: str) -> bool:
        """Migrate reusable components catalog to patterns collection."""
        print(f"📄 Migrating reusable components catalog: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                components_data = json.load(f)
            
            # Store the entire catalog
            catalog_metadata = {
                "type": "pattern",
                "subtype": "component_catalog",
                "source": "reusable-components-catalog.json",
                "complexity": "medium",
                "tags": "components,catalog,reusable,patterns"
            }
            
            catalog_doc_id = self.rag.store_knowledge(
                collection_name="patterns",
                content=json.dumps(components_data, indent=2),
                metadata=catalog_metadata
            )
            
            print(f"✅ Stored reusable components catalog as {catalog_doc_id}")
            
            self.migration_stats["json_files_processed"] += 1
            self.migration_stats["patterns_stored"] += 1
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate reusable components: {e}"
            print(f"❌ {error_msg}")
            self.migration_stats["errors"].append(error_msg)
            return False
    
    def run_migration(self) -> bool:
        """Run the complete migration process."""
        print("🚀 Starting learning content migration to LightRAG")
        print("=" * 60)
        
        # Migration tasks
        tasks = [
            {
                "file": os.path.join(self.knowledge_path, "orchestration-report-2025-08-18.md"),
                "method": self.migrate_orchestration_report,
                "description": "Orchestration report to issue_resolutions"
            },
            {
                "file": os.path.join(self.knowledge_path, "learning", "error-analysis-implementation-template.md"),
                "method": self.migrate_error_analysis_template,
                "description": "Error analysis template to patterns"
            },
            {
                "file": os.path.join(self.knowledge_path, "learning", "issue-6-comprehensive-learnings.json"),
                "method": self.migrate_comprehensive_learnings,
                "description": "Comprehensive learnings to patterns and resolutions"
            },
            {
                "file": os.path.join(self.knowledge_path, "learning", "reusable-components-catalog.json"),
                "method": self.migrate_reusable_components,
                "description": "Reusable components to patterns"
            }
        ]
        
        # Execute migration tasks
        success_count = 0
        for task in tasks:
            print(f"\n📋 Task: {task['description']}")
            if os.path.exists(task["file"]):
                if task["method"](task["file"]):
                    success_count += 1
                    print(f"✅ Success: {task['description']}")
                else:
                    print(f"❌ Failed: {task['description']}")
            else:
                print(f"⚠️  File not found: {task['file']}")
        
        # Print migration summary
        print("\n" + "=" * 60)
        print("📊 MIGRATION SUMMARY")
        print("=" * 60)
        print(f"✅ Tasks completed successfully: {success_count}/{len(tasks)}")
        print(f"📄 Markdown files processed: {self.migration_stats['md_files_processed']}")
        print(f"📄 JSON files processed: {self.migration_stats['json_files_processed']}")
        print(f"🔍 Patterns stored: {self.migration_stats['patterns_stored']}")
        print(f"📋 Resolutions stored: {self.migration_stats['resolutions_stored']}")
        print(f"📝 Templates stored: {self.migration_stats['templates_stored']}")
        
        if self.migration_stats["errors"]:
            print(f"\n❌ Errors encountered: {len(self.migration_stats['errors'])}")
            for error in self.migration_stats["errors"]:
                print(f"   • {error}")
        
        # Verify migration with collection stats
        print("\n📈 LIGHTRAG COLLECTION STATUS")
        print("=" * 60)
        try:
            stats = self.rag.get_collection_stats()
            for collection, info in stats.items():
                if "count" in info:
                    print(f"📚 {collection}: {info['count']} documents")
                else:
                    print(f"❌ {collection}: {info.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
        
        # Test semantic search
        print("\n🔍 TESTING SEMANTIC SEARCH")
        print("=" * 60)
        try:
            test_queries = [
                "error analysis patterns",
                "implementation template",
                "orchestration learnings"
            ]
            
            for query in test_queries:
                results = self.rag.retrieve_knowledge(query, n_results=3)
                print(f"🔍 Query: '{query}' → {len(results)} results found")
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")
        
        migration_success = success_count == len(tasks) and len(self.migration_stats["errors"]) == 0
        
        if migration_success:
            print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
            print("All learning content has been migrated to LightRAG vector database.")
        else:
            print("\n⚠️  MIGRATION COMPLETED WITH ISSUES")
            print("Some content may not have been migrated successfully.")
        
        return migration_success
    
    def cleanup_old_files(self, backup_dir: str = None) -> bool:
        """Move migrated files to backup directory."""
        if backup_dir is None:
            backup_dir = os.path.join(self.knowledge_path, "migrated_backup")
        
        os.makedirs(backup_dir, exist_ok=True)
        
        files_to_backup = [
            "orchestration-report-2025-08-18.md",
            "learning/error-analysis-implementation-template.md"
        ]
        
        print(f"\n🗂️  Moving migrated files to backup: {backup_dir}")
        
        for file_path in files_to_backup:
            source = os.path.join(self.knowledge_path, file_path)
            if os.path.exists(source):
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                try:
                    os.rename(source, backup_path)
                    print(f"✅ Moved {file_path} → {backup_path}")
                except Exception as e:
                    print(f"❌ Failed to move {file_path}: {e}")
                    return False
        
        return True


def main():
    """Main migration function."""
    print("🔄 RIF Learning Content Migration to LightRAG")
    print("This script migrates existing .md and JSON learning files to the LightRAG vector database.")
    print("=" * 80)
    
    # Initialize migrator
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge")
    migrator = LearningMigrator(knowledge_path)
    
    # Run migration
    success = migrator.run_migration()
    
    if success:
        # Ask about cleanup
        response = input("\n🗂️  Move migrated .md files to backup directory? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            migrator.cleanup_old_files()
        
        print("\n✅ Migration process completed successfully!")
        print("Learning content is now stored in LightRAG and available for semantic search.")
        return 0
    else:
        print("\n❌ Migration process completed with errors.")
        print("Please review the errors above and re-run if necessary.")
        return 1


if __name__ == "__main__":
    sys.exit(main())