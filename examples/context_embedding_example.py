#!/usr/bin/env python3
"""
Context Embedding Engine Example

This example demonstrates how to use the Context Embedding Engine to generate
complete GitHub issues with embedded context from project documentation.

Usage:
    python3 examples/context_embedding_example.py
"""

import json
import tempfile
import sys
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.context_embedder import ContextEmbedder, TokenManager, ContextTemplates


def create_example_documentation():
    """Create example project documentation XML"""
    return '''<?xml version="1.0"?>
<project_documentation version="1.0" timestamp="2025-01-25T10:00:00">
  <metadata>
    <project_name>TaskManager</project_name>
    <generated_at>2025-01-25T10:00:00</generated_at>
    <source_directory>docs/</source_directory>
  </metadata>
  <documents>
    <document type="prd" name="task-management" path="docs/task-management.prd.md">
      <section level="1" title="Task Management System">
        <content>
The task management system allows users to create, organize, and track tasks.
Users can create tasks with titles, descriptions, due dates, and priority levels.
Tasks can be organized into projects and assigned to team members.
The system supports task filtering, sorting, and search functionality.
        </content>
      </section>
      <section level="2" title="Task Operations">
        <content>
Users can perform the following operations on tasks:
- Create new tasks with required and optional fields
- Edit existing tasks including title, description, due date, and priority
- Delete tasks with confirmation dialog
- Mark tasks as complete or incomplete
- Assign tasks to team members
- Add comments and attachments to tasks
        </content>
      </section>
    </document>
    
    <document type="architecture" name="system-architecture" path="docs/architecture.md">
      <section level="1" title="Task Service Architecture">
        <content>
The TaskService component handles all task-related operations.
It integrates with the TaskRepository for database operations.
The NotificationService sends alerts for due dates and assignments.
The SearchEngine provides full-text search capabilities across tasks.
The TaskValidator ensures data integrity and business rule compliance.
        </content>
      </section>
      <section level="2" title="Database Schema">
        <content>
Tasks table contains: id, title, description, due_date, priority, status, created_at, updated_at
Projects table contains: id, name, description, owner_id, created_at
Users table contains: id, email, name, role, created_at
Task_assignments table contains: task_id, user_id, assigned_at
        </content>
      </section>
      <section level="3" title="API Endpoints">
        <content>
GET /api/tasks - List tasks with filtering and pagination
POST /api/tasks - Create new task
GET /api/tasks/{id} - Get specific task details
PUT /api/tasks/{id} - Update existing task
DELETE /api/tasks/{id} - Delete task
POST /api/tasks/{id}/assign - Assign task to user
POST /api/tasks/{id}/complete - Mark task as complete
        </content>
      </section>
    </document>
    
    <document type="design" name="task-ui" path="docs/task-design.md">
      <section level="1" title="Task Interface Design">
        <content>
The TaskListComponent displays tasks in a paginated table view.
The TaskFormComponent provides create/edit functionality with validation.
The TaskCardComponent shows task summary in card format.
The FilterPanel allows users to filter tasks by status, priority, and assignee.
The SearchBar enables full-text search across task titles and descriptions.
        </content>
      </section>
      <section level="2" title="Mobile Responsiveness">
        <content>
The interface adapts to mobile devices with:
- Collapsible sidebar navigation
- Touch-friendly button sizes
- Responsive card layout for task display
- Swipe gestures for task actions
- Mobile-optimized forms with appropriate input types
        </content>
      </section>
    </document>
  </documents>
</project_documentation>'''


def create_example_requirements():
    """Create example requirements extracted from documentation"""
    return {
        "requirements": [
            {
                "id": "FR-001",
                "type": "functional",
                "title": "Task Creation",
                "description": "The system shall allow users to create new tasks with title, description, due date, and priority level",
                "source_document": "task-management.prd.md",
                "source_section": "Task Operations",
                "priority": "high",
                "complexity": "medium",
                "dependencies": ["FR-002", "FR-003"],
                "acceptance_criteria": [
                    "User can enter task title (required)",
                    "User can enter task description (optional)",
                    "User can set due date using date picker",
                    "User can select priority level (low, medium, high, critical)",
                    "Task is saved to database with validation",
                    "User receives confirmation of successful creation"
                ],
                "tags": ["task", "crud", "ui", "backend"]
            },
            {
                "id": "FR-002", 
                "type": "functional",
                "title": "Task Validation",
                "description": "The system shall validate task data to ensure integrity and business rule compliance",
                "source_document": "task-management.prd.md",
                "source_section": "Task Operations",
                "priority": "high",
                "complexity": "low",
                "dependencies": [],
                "acceptance_criteria": [
                    "Task title is required and limited to 200 characters",
                    "Due date cannot be in the past",
                    "Priority must be one of allowed values",
                    "Appropriate error messages for validation failures"
                ],
                "tags": ["validation", "backend", "data-integrity"]
            }
        ],
        "stories": [
            {
                "id": "US-001",
                "type": "user_story", 
                "title": "Create New Task",
                "actor": "project manager",
                "action": "create a new task with all necessary details",
                "benefit": "I can assign work to team members and track progress",
                "description": "As a project manager, I want to create a new task with all necessary details so that I can assign work to team members and track progress",
                "source_document": "task-management.prd.md",
                "source_section": "Task Operations",
                "acceptance_criteria": [
                    "Task creation form is accessible from main dashboard",
                    "Form includes fields for title, description, due date, priority",
                    "Form validates required fields before submission",
                    "Successful creation shows confirmation and redirects to task list",
                    "Created task appears in task list immediately"
                ],
                "estimated_points": 8,
                "complexity": "medium",
                "priority": "high",
                "tags": ["ui", "form", "crud", "task-management"]
            },
            {
                "id": "US-002",
                "type": "user_story",
                "title": "Search Tasks",
                "actor": "team member", 
                "action": "search for specific tasks using keywords",
                "benefit": "I can quickly find relevant tasks without scrolling through long lists",
                "description": "As a team member, I want to search for specific tasks using keywords so that I can quickly find relevant tasks without scrolling through long lists",
                "source_document": "task-management.prd.md",
                "source_section": "Task Management System",
                "acceptance_criteria": [
                    "Search bar is prominently displayed on task list page",
                    "Search works on task titles and descriptions",
                    "Results are filtered in real-time as user types",
                    "Search highlights matching keywords in results",
                    "Empty state shown when no results found"
                ],
                "estimated_points": 5,
                "complexity": "medium", 
                "priority": "medium",
                "tags": ["search", "ui", "performance"]
            }
        ],
        "epics": [
            {
                "id": "EPIC-001",
                "type": "epic",
                "title": "Core Task Management",
                "description": "Complete task management functionality including creation, editing, deletion, and organization of tasks",
                "source_document": "task-management.prd.md", 
                "source_section": "Task Management System",
                "stories": ["US-001", "US-003", "US-004", "US-005"],
                "estimated_effort": "3-4 weeks",
                "complexity": "high",
                "tags": ["core-functionality", "crud", "task-management"]
            }
        ],
        "statistics": {
            "total_requirements": 2,
            "total_stories": 2,
            "total_epics": 1
        }
    }


def demonstrate_context_embedding():
    """Demonstrate the context embedding functionality"""
    print("🚀 Context Embedding Engine Demonstration")
    print("=" * 50)
    
    # Create temporary files with example data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as xml_file:
        xml_file.write(create_example_documentation())
        xml_file_path = xml_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as req_file:
        json.dump(create_example_requirements(), req_file, indent=2)
        req_file_path = req_file.name
    
    try:
        # Initialize the context embedder
        print("\n📚 Initializing Context Embedder...")
        embedder = ContextEmbedder(
            xml_file_path=xml_file_path,
            requirements_file_path=req_file_path
        )
        
        print(f"   - Loaded {len(embedder.documents)} documentation files")
        print(f"   - Loaded {len(embedder.requirements.get('requirements', []))} requirements")
        print(f"   - Loaded {len(embedder.requirements.get('stories', []))} user stories")
        print(f"   - Loaded {len(embedder.requirements.get('epics', []))} epics")
        
        # Demonstrate token management
        print("\n🔢 Token Management Example:")
        tm = TokenManager()
        sample_text = "This is an example of how the token manager counts and manages tokens in text content."
        token_count = tm.count_tokens(sample_text)
        print(f"   Text: '{sample_text}'")
        print(f"   Token count: {token_count}")
        
        long_text = sample_text * 10
        trimmed = tm.trim_to_tokens(long_text, 50)
        print(f"   Original text length: {len(long_text)} characters")
        print(f"   Trimmed to 50 tokens: {len(trimmed)} characters")
        
        # Demonstrate context embedding for different requirement types
        requirement_types = [
            ("FR-001", "Functional Requirement"),
            ("US-001", "User Story"),
            ("EPIC-001", "Epic")
        ]
        
        for req_id, req_type_name in requirement_types:
            print(f"\n📝 {req_type_name} Context Embedding ({req_id}):")
            
            # Generate context
            context = embedder.embed_context_for_requirement(req_id)
            requirement = embedder._get_requirement(req_id)
            
            if requirement:
                print(f"   - Primary requirement tokens: {tm.count_tokens(context.primary_requirement)}")
                print(f"   - Architecture context tokens: {tm.count_tokens(context.architecture_context)}")
                print(f"   - Implementation hints tokens: {tm.count_tokens(context.implementation_hints)}")
                print(f"   - Design context tokens: {tm.count_tokens(context.design_context)}")
                print(f"   - Total estimated tokens: {context.estimated_tokens}")
                print(f"   - Overflow strategy: {context.overflow_strategy}")
                print(f"   - Dependencies found: {len(context.dependencies)}")
                print(f"   - Related requirements: {len(context.related_requirements)}")
                
                # Generate complete issue
                if req_id.startswith("US-"):
                    issue_content = ContextTemplates.user_story_with_context(requirement, context)
                elif req_id.startswith("EPIC-"):
                    issue_content = ContextTemplates.epic_with_context(requirement, context)
                else:
                    issue_content = ContextTemplates.issue_with_full_context(requirement, context)
                
                issue_tokens = tm.count_tokens(issue_content)
                print(f"   - Complete issue tokens: {issue_tokens}")
                
                if issue_tokens <= 2500:
                    print("   ✅ Issue fits within optimal token limit (≤2500)")
                elif issue_tokens <= 3000:
                    print("   ⚠️  Issue slightly exceeds optimal limit but acceptable")
                else:
                    print("   ❌ Issue exceeds recommended token limit")
            else:
                print(f"   ❌ Requirement {req_id} not found")
        
        # Show a complete generated issue example
        print("\n📄 Complete Generated Issue Example (User Story US-001):")
        print("=" * 60)
        
        context = embedder.embed_context_for_requirement("US-001")
        story = embedder._get_requirement("US-001")
        complete_issue = ContextTemplates.user_story_with_context(story, context)
        
        # Show first 1000 characters
        preview = complete_issue[:1000] + "...\n[Content truncated for example]" if len(complete_issue) > 1000 else complete_issue
        print(preview)
        print("=" * 60)
        
        # Summary statistics
        print(f"\n📊 Summary Statistics:")
        all_requirements = ["FR-001", "FR-002", "US-001", "US-002", "EPIC-001"]
        token_counts = []
        
        for req_id in all_requirements:
            try:
                context = embedder.embed_context_for_requirement(req_id)
                req = embedder._get_requirement(req_id)
                if req:
                    if req_id.startswith("US-"):
                        issue = ContextTemplates.user_story_with_context(req, context)
                    elif req_id.startswith("EPIC-"):
                        issue = ContextTemplates.epic_with_context(req, context)
                    else:
                        issue = ContextTemplates.issue_with_full_context(req, context)
                    
                    tokens = tm.count_tokens(issue)
                    token_counts.append(tokens)
            except Exception as e:
                print(f"   Warning: Could not process {req_id}: {e}")
        
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            min_tokens = min(token_counts)
            
            print(f"   - Average issue tokens: {avg_tokens:.0f}")
            print(f"   - Maximum issue tokens: {max_tokens}")
            print(f"   - Minimum issue tokens: {min_tokens}")
            print(f"   - Issues within optimal limit (≤2500): {sum(1 for t in token_counts if t <= 2500)}/{len(token_counts)}")
        
        print("\n✅ Context Embedding Engine demonstration completed!")
        print("\nKey Features Demonstrated:")
        print("   ✓ Token counting and management")
        print("   ✓ Context extraction from documentation")
        print("   ✓ Architecture and design context embedding")
        print("   ✓ Implementation guidance generation")
        print("   ✓ Dependency and relationship analysis")
        print("   ✓ Overflow handling strategies")
        print("   ✓ Template-based issue generation")
        print("   ✓ Support for functional requirements, user stories, and epics")
        
    finally:
        # Clean up temporary files
        Path(xml_file_path).unlink()
        Path(req_file_path).unlink()


if __name__ == "__main__":
    demonstrate_context_embedding()