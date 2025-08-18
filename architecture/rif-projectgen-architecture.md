# RIF-ProjectGen System Architecture

## System Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        CMD[/newproject Command]
        CLI[Claude Code CLI]
    end
    
    subgraph "Orchestration Layer"
        PG[RIF-ProjectGen Agent]
        WF[Workflow Engine]
        SM[State Machine]
    end
    
    subgraph "Agent Layer"
        BA[Business Analyst]
        PM[Project Manager]
        UX[UX/UI Designer]
        ARCH[Architect]
    end
    
    subgraph "Generation Layer"
        DG[Discovery Engine]
        PRD[PRD Generator]
        UIG[UI Generator]
        AG[Architecture Gen]
    end
    
    subgraph "Automation Layer"
        RC[Repository Creator]
        GH[GitHub Integration]
        IG[Issue Generator]
        TM[Template Manager]
    end
    
    subgraph "Storage Layer"
        KB[Knowledge Base]
        TP[Templates]
        CFG[Configurations]
    end
    
    CMD --> CLI
    CLI --> PG
    PG --> WF
    WF --> SM
    
    SM --> BA
    SM --> PM
    SM --> UX
    SM --> ARCH
    
    BA --> DG
    PM --> PRD
    UX --> UIG
    ARCH --> AG
    
    PRD --> RC
    RC --> GH
    PRD --> IG
    IG --> GH
    
    TM --> TP
    KB --> PG
    CFG --> WF
```

## Component Architecture

### 1. Command Interface
```yaml
Component: /newproject Command Handler
Location: /scripts/newproject.sh
Responsibilities:
  - Parse command invocation
  - Initialize workflow context
  - Handle user interrupts
  - Manage session state
Interfaces:
  - Input: Claude Code command
  - Output: Workflow trigger
Dependencies:
  - Claude Code hooks system
  - Bash environment
```

### 2. RIF-ProjectGen Agent
```yaml
Component: Master Orchestrator
Location: /claude/agents/rif-projectgen.md
Responsibilities:
  - Coordinate sub-agents
  - Manage workflow phases
  - Handle state transitions
  - Aggregate outputs
Interfaces:
  - Input: Project requirements
  - Output: Complete project
State Management:
  phases:
    - discovery
    - documentation
    - setup
    - kickoff
```

### 3. Discovery Engine
```yaml
Component: Interactive Discovery System
Location: /engines/discovery.sh
Responsibilities:
  - Guide user through questions
  - Collect project requirements
  - Analyze constraints
  - Generate project brief
Data Flow:
  Input: User responses
  Processing: Requirement analysis
  Output: project-brief.json
```

### 4. Document Generators
```yaml
PRD Generator:
  Location: /generators/prd-generator.sh
  Input: project-brief.json
  Output: PRD.md
  
UI Specification Generator:
  Location: /generators/ui-generator.sh
  Input: PRD.md
  Output: ui-spec.md, wireframes/
  
Architecture Generator:
  Location: /generators/arch-generator.sh
  Input: PRD.md
  Output: architecture.md, diagrams/
```

### 5. Repository Automation
```yaml
Component: Repository Setup System
Location: /automation/repo-setup.sh
Functions:
  - Clone RIF framework
  - Initialize git repository
  - Create GitHub remote
  - Configure hooks
  - Set up CI/CD
Workflow:
  1. cp -r /RIF-template/ /new-project/
  2. git init
  3. gh repo create
  4. git remote add origin
  5. git push -u origin main
```

### 6. Issue Generation System
```yaml
Component: PRD to Issue Converter
Location: /automation/issue-generator.sh
Algorithm:
  1. Parse PRD sections
  2. Extract epics and stories
  3. Generate issue templates
  4. Apply labels and milestones
  5. Create dependencies
  6. Batch create via GitHub API
```

## Data Flow Architecture

### Phase 1: Discovery Flow
```
User Input → Interactive Prompts → Response Collection → 
Analysis → Project Brief → Knowledge Base Update
```

### Phase 2: Documentation Flow
```
Project Brief → Agent Activation → Document Generation →
Review & Refinement → Final Documents → Storage
```

### Phase 3: Setup Flow
```
Documents → Template Selection → Repository Creation →
Configuration → GitHub Sync → Hook Setup
```

### Phase 4: Kickoff Flow
```
PRD → Issue Parsing → Issue Creation → Label Application →
RIF Activation → Development Start
```

## File System Structure

```
/Users/cal/DEV/RIF/
├── scripts/
│   └── newproject.sh              # Main command handler
├── engines/
│   ├── discovery.sh               # Discovery engine
│   └── workflow.sh                # Workflow orchestrator
├── generators/
│   ├── prd-generator.sh          # PRD generation
│   ├── ui-generator.sh           # UI specification
│   └── arch-generator.sh         # Architecture docs
├── automation/
│   ├── repo-setup.sh             # Repository automation
│   ├── issue-generator.sh        # Issue creation
│   └── github-integration.sh     # GitHub API wrapper
├── templates/
│   └── projects/
│       ├── web-app/              # Web application template
│       ├── api-service/          # API service template
│       ├── cli-tool/             # CLI tool template
│       ├── library/              # Library template
│       └── custom/               # Custom template
├── config/
│   ├── projectgen-config.yaml    # Main configuration
│   ├── prompts.yaml              # Discovery prompts
│   └── templates.yaml            # Template registry
└── claude/
    └── agents/
        └── rif-projectgen.md     # Agent definition
```

## Integration Points

### Claude Code Integration
```json
{
  "commands": {
    "/newproject": {
      "script": "${RIF_HOME}/scripts/newproject.sh",
      "description": "Create new RIF-enabled project",
      "interactive": true
    }
  }
}
```

### GitHub API Integration
```yaml
Endpoints:
  - POST /repos              # Create repository
  - POST /repos/{}/issues    # Create issues
  - POST /repos/{}/labels    # Create labels
  - POST /repos/{}/milestones # Create milestones
Rate Limiting:
  - Implement exponential backoff
  - Cache responses
  - Batch operations
```

### Knowledge Base Integration
```yaml
Patterns Storage:
  Location: /knowledge/patterns/projects/
  Format: JSON
  Schema:
    - project_type
    - technology_stack
    - success_metrics
    - common_issues
    
Learning Integration:
  - Store successful configurations
  - Track generation metrics
  - Update templates based on feedback
```

## Security Considerations

1. **Input Validation**: Sanitize all user inputs
2. **API Key Management**: Secure storage of GitHub tokens
3. **Template Injection**: Prevent code injection in templates
4. **Rate Limiting**: Respect GitHub API limits
5. **Error Handling**: Graceful failure with recovery

## Performance Optimization

1. **Parallel Processing**: Run independent generators concurrently
2. **Caching**: Cache template renders and API responses
3. **Lazy Loading**: Load templates on demand
4. **Batch Operations**: Group GitHub API calls
5. **Progressive Enhancement**: Start with MVP, add features

## Monitoring & Observability

```yaml
Metrics:
  - Project generation time
  - Success/failure rates
  - User satisfaction scores
  - Template usage statistics
  
Logging:
  - Workflow state transitions
  - Agent activations
  - API calls and responses
  - Error conditions
  
Telemetry:
  Location: /knowledge/metrics/projectgen/
  Format: JSONL event stream
```

## Error Recovery

### Checkpoint System
```yaml
Checkpoints:
  - After discovery phase
  - After each document generation
  - Before repository creation
  - After GitHub sync
  
Recovery:
  - Restore from last checkpoint
  - Retry failed operations
  - Provide manual intervention options
```

## Scalability Considerations

1. **Template Extensibility**: Plugin architecture for new templates
2. **Agent Modularity**: Easy addition of new specialized agents
3. **Workflow Flexibility**: Configurable phase ordering
4. **Storage Abstraction**: Support for different backends
5. **Multi-language Support**: Internationalization ready

## Testing Strategy

### Unit Tests
- Individual generator functions
- Template rendering
- API integration

### Integration Tests
- End-to-end workflow
- Agent communication
- GitHub operations

### User Acceptance Tests
- Project generation scenarios
- Error recovery paths
- Performance benchmarks