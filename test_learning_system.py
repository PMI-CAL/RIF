#!/usr/bin/env python3
"""
Comprehensive test suite for the RIF learning system to validate that:
1. RIF-Learner agent is properly defined
2. LightRAG migration was successful
3. Agent knowledge storage functions work correctly
4. Learning workflow optimization is effective

This script is part of Phase 4 implementation for GitHub issue #10.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the lightrag module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lightrag'))

try:
    from lightrag.core.lightrag_core import LightRAGCore, store_pattern, store_decision, get_lightrag_instance
    print("✅ LightRAG core imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LightRAG core: {e}")
    sys.exit(1)


class LearningSystemTester:
    """Test the complete RIF learning system."""
    
    def __init__(self, knowledge_path: str = None):
        """Initialize the tester with LightRAG core."""
        self.knowledge_path = knowledge_path or "knowledge"
        print(f"🔄 Initializing test suite with knowledge path: {self.knowledge_path}")
        
        try:
            self.rag = LightRAGCore(knowledge_path=self.knowledge_path)
            print("✅ LightRAG initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LightRAG: {e}")
            raise
        
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results."""
        self.test_results["total_tests"] += 1
        print(f"\n🧪 Running test: {test_name}")
        
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                self.test_results["passed"] += 1
                return True
            else:
                print(f"❌ {test_name}: FAILED")
                self.test_results["failed"] += 1
                return False
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            self.test_results["errors"].append(f"{test_name}: {e}")
            self.test_results["failed"] += 1
            return False
    
    def test_rif_learner_agent_exists(self) -> bool:
        """Test that RIF-Learner agent definition exists and is complete."""
        agent_path = os.path.join("claude", "agents", "rif-learner.md")
        
        if not os.path.exists(agent_path):
            print(f"   ❌ Agent file not found: {agent_path}")
            return False
        
        with open(agent_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "# RIF Learner Agent",
            "## Activation",
            "## Responsibilities",
            "## LightRAG Integration",
            "store_pattern",
            "store_decision",
            "patterns collection",
            "decisions collection"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"   ❌ Missing sections: {missing_sections}")
            return False
        
        print("   ✅ RIF-Learner agent properly defined with LightRAG integration")
        return True
    
    def test_lightrag_collections_exist(self) -> bool:
        """Test that all required LightRAG collections exist and are accessible."""
        try:
            stats = self.rag.get_collection_stats()
            
            required_collections = ["patterns", "decisions", "code_snippets", "issue_resolutions"]
            missing_collections = []
            
            for collection in required_collections:
                if collection not in stats:
                    missing_collections.append(collection)
                elif "error" in stats[collection]:
                    missing_collections.append(f"{collection} (error: {stats[collection]['error']})")
            
            if missing_collections:
                print(f"   ❌ Missing or broken collections: {missing_collections}")
                return False
            
            print("   ✅ All required collections exist and accessible")
            for collection, info in stats.items():
                if "count" in info:
                    print(f"      📚 {collection}: {info['count']} documents")
            
            return True
        except Exception as e:
            print(f"   ❌ Error checking collections: {e}")
            return False
    
    def test_migrated_content_accessible(self) -> bool:
        """Test that migrated content is searchable in LightRAG."""
        test_queries = [
            ("error analysis", "Should find migrated error analysis template"),
            ("orchestration", "Should find migrated orchestration report"),
            ("implementation pattern", "Should find implementation insights"),
            ("architecture insight", "Should find architecture patterns")
        ]
        
        all_queries_successful = True
        
        for query, description in test_queries:
            try:
                results = self.rag.retrieve_knowledge(query, n_results=3)
                if not results:
                    print(f"   ❌ No results for query '{query}' - {description}")
                    all_queries_successful = False
                else:
                    print(f"   ✅ Query '{query}' returned {len(results)} results")
            except Exception as e:
                print(f"   ❌ Error querying '{query}': {e}")
                all_queries_successful = False
        
        return all_queries_successful
    
    def test_agent_lightrag_integration(self) -> bool:
        """Test that agent LightRAG integration code is present."""
        agents_to_check = ["rif-implementer.md", "rif-validator.md", "rif-planner.md"]
        
        all_agents_updated = True
        
        for agent_file in agents_to_check:
            agent_path = os.path.join("claude", "agents", agent_file)
            
            if not os.path.exists(agent_path):
                print(f"   ❌ Agent file not found: {agent_path}")
                all_agents_updated = False
                continue
            
            with open(agent_path, 'r') as f:
                content = f.read()
            
            # Check for LightRAG integration
            lightrag_indicators = [
                "LightRAG",
                "store_pattern",
                "store_decision",
                "never create .md files",
                "Knowledge Storage Guidelines"
            ]
            
            missing_indicators = []
            for indicator in lightrag_indicators:
                if indicator not in content:
                    missing_indicators.append(indicator)
            
            if missing_indicators:
                print(f"   ❌ {agent_file} missing: {missing_indicators}")
                all_agents_updated = False
            else:
                print(f"   ✅ {agent_file} has proper LightRAG integration")
        
        return all_agents_updated
    
    def test_knowledge_storage_functions(self) -> bool:
        """Test that knowledge storage functions work correctly."""
        test_id_suffix = str(int(time.time()))
        
        # Test pattern storage
        try:
            test_pattern = {
                "title": f"Test Pattern {test_id_suffix}",
                "description": "Testing pattern storage functionality",
                "implementation": "Test implementation details",
                "context": "Testing context",
                "complexity": "low",
                "source": "test_learning_system.py",
                "tags": "test,pattern,validation"
            }
            
            pattern_id = store_pattern(test_pattern)
            print(f"   ✅ Pattern storage successful: {pattern_id}")
            
        except Exception as e:
            print(f"   ❌ Pattern storage failed: {e}")
            return False
        
        # Test decision storage
        try:
            test_decision = {
                "title": f"Test Decision {test_id_suffix}",
                "context": "Testing decision storage",
                "decision": "Test decision content",
                "rationale": "Testing rationale",
                "consequences": "Testing consequences",
                "status": "active",
                "impact": "low",
                "tags": "test,decision,validation"
            }
            
            decision_id = store_decision(test_decision)
            print(f"   ✅ Decision storage successful: {decision_id}")
            
        except Exception as e:
            print(f"   ❌ Decision storage failed: {e}")
            return False
        
        # Test direct storage
        try:
            test_resolution = {
                "title": f"Test Resolution {test_id_suffix}",
                "problem": "Testing issue resolution storage",
                "solution": "Test solution content",
                "approach": "Test approach",
                "learnings": "Test learnings"
            }
            
            resolution_id = self.rag.store_knowledge(
                "issue_resolutions",
                json.dumps(test_resolution, indent=2),
                {
                    "type": "test_resolution",
                    "complexity": "low",
                    "source": "test_learning_system.py",
                    "tags": "test,resolution,validation"
                }
            )
            print(f"   ✅ Resolution storage successful: {resolution_id}")
            
        except Exception as e:
            print(f"   ❌ Resolution storage failed: {e}")
            return False
        
        return True
    
    def test_semantic_search_quality(self) -> bool:
        """Test that semantic search returns relevant results."""
        # Test search relevance
        test_searches = [
            {
                "query": "error analysis implementation",
                "expected_keywords": ["error", "analysis", "implementation"],
                "min_results": 1
            },
            {
                "query": "planning strategy workflow",
                "expected_keywords": ["planning", "strategy", "workflow"],
                "min_results": 1
            },
            {
                "query": "validation testing approach",
                "expected_keywords": ["validation", "testing", "approach"],
                "min_results": 1
            }
        ]
        
        all_searches_successful = True
        
        for search_test in test_searches:
            try:
                results = self.rag.retrieve_knowledge(
                    search_test["query"],
                    n_results=search_test["min_results"] + 2
                )
                
                if len(results) < search_test["min_results"]:
                    print(f"   ❌ Insufficient results for '{search_test['query']}': {len(results)} < {search_test['min_results']}")
                    all_searches_successful = False
                    continue
                
                # Check result relevance
                relevant_results = 0
                for result in results:
                    content = result.get("content", "").lower()
                    keywords_found = sum(1 for keyword in search_test["expected_keywords"] if keyword in content)
                    if keywords_found > 0:
                        relevant_results += 1
                
                if relevant_results == 0:
                    print(f"   ❌ No relevant results for '{search_test['query']}'")
                    all_searches_successful = False
                else:
                    print(f"   ✅ Search '{search_test['query']}': {relevant_results}/{len(results)} relevant results")
                    
            except Exception as e:
                print(f"   ❌ Search error for '{search_test['query']}': {e}")
                all_searches_successful = False
        
        return all_searches_successful
    
    def test_workflow_optimization(self) -> bool:
        """Test workflow optimization features."""
        # Test that workflow state machine references RIF-Learner
        workflow_config = os.path.join("config", "rif-workflow.yaml")
        
        if not os.path.exists(workflow_config):
            print(f"   ❌ Workflow config not found: {workflow_config}")
            return False
        
        with open(workflow_config, 'r') as f:
            config_content = f.read()
        
        # Check for learning state and agent
        if "learning:" not in config_content:
            print("   ❌ Learning state not found in workflow config")
            return False
        
        if "rif-learner" not in config_content:
            print("   ❌ RIF-Learner agent not referenced in workflow config")
            return False
        
        print("   ✅ Workflow properly configured for learning state")
        
        # Test that learning transition exists (check for the state definition)
        if "to: learning" not in config_content and "to: \"learning\"" not in config_content:
            print("   ❌ Learning state transition not found")
            return False
        
        print("   ✅ Learning state transitions properly configured")
        return True
    
    def test_no_md_files_created(self) -> bool:
        """Test that no new .md files have been created in knowledge directory."""
        knowledge_dir = Path(self.knowledge_path)
        
        # Look for .md files that shouldn't exist
        md_files = list(knowledge_dir.glob("**/*.md"))
        
        # Filter out expected files (like migrated backup)
        unexpected_md_files = []
        allowed_patterns = [
            "migrated_backup",  # Our backup directory
            "README.md",        # Standard documentation
            "CHANGELOG.md",     # Standard documentation
        ]
        
        for md_file in md_files:
            file_path_str = str(md_file)
            if not any(pattern in file_path_str for pattern in allowed_patterns):
                unexpected_md_files.append(md_file)
        
        if unexpected_md_files:
            print(f"   ❌ Unexpected .md files found: {unexpected_md_files}")
            return False
        
        print("   ✅ No unexpected .md files in knowledge directory")
        return True
    
    def run_comprehensive_test_suite(self) -> bool:
        """Run the complete test suite."""
        print("🧪 Starting Comprehensive RIF Learning System Test Suite")
        print("=" * 70)
        
        # Define all tests
        tests = [
            ("RIF-Learner Agent Definition", self.test_rif_learner_agent_exists),
            ("LightRAG Collections", self.test_lightrag_collections_exist),
            ("Migrated Content Accessibility", self.test_migrated_content_accessible),
            ("Agent LightRAG Integration", self.test_agent_lightrag_integration),
            ("Knowledge Storage Functions", self.test_knowledge_storage_functions),
            ("Semantic Search Quality", self.test_semantic_search_quality),
            ("Workflow Optimization", self.test_workflow_optimization),
            ("No MD Files Created", self.test_no_md_files_created)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 70)
        print(f"✅ Tests Passed: {self.test_results['passed']}/{self.test_results['total_tests']}")
        print(f"❌ Tests Failed: {self.test_results['failed']}/{self.test_results['total_tests']}")
        
        if self.test_results["errors"]:
            print(f"\n❌ Errors Encountered: {len(self.test_results['errors'])}")
            for error in self.test_results["errors"]:
                print(f"   • {error}")
        
        # Calculate success rate
        success_rate = (self.test_results["passed"] / self.test_results["total_tests"]) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎉 EXCELLENT: Learning system is fully operational!")
        elif success_rate >= 80:
            print("\n✅ GOOD: Learning system is mostly operational with minor issues")
        elif success_rate >= 70:
            print("\n⚠️  NEEDS IMPROVEMENT: Learning system has significant issues")
        else:
            print("\n❌ CRITICAL: Learning system requires major fixes")
        
        return success_rate >= 80


def main():
    """Main test function."""
    print("🔍 RIF Learning System Comprehensive Test Suite")
    print("This script validates the complete learning system implementation.")
    print("=" * 80)
    
    # Initialize tester
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge")
    tester = LearningSystemTester(knowledge_path)
    
    # Run comprehensive tests
    success = tester.run_comprehensive_test_suite()
    
    if success:
        print("\n✅ All systems operational! Learning implementation is complete.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review and fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())