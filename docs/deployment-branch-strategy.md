# RIF Deployment Branch Strategy

## Overview

The RIF deployment branch strategy implements a dual-branch architecture that separates development activities from production-ready deployments, ensuring clean, reliable project initialization for end users.

## Architecture

### Branch Structure

```
main (development)  →  deploy (production-ready)
     │                      │
     ├── Full RIF development  ├── Clean template
     ├── Test artifacts       ├── Essential framework only
     ├── Development logs     ├── Deployment-optimized
     ├── Issue workflows      ├── User-ready
     └── Self-development     └── No development artifacts
```

### Branch Purposes

#### Main Branch
- **Purpose**: Full RIF development environment
- **Contents**: Complete development history, test artifacts, development issues, audit logs
- **Audience**: RIF framework contributors and maintainers
- **Workflow**: All development, testing, and framework enhancement

#### Deploy Branch
- **Purpose**: Clean, production-ready template for project initialization
- **Contents**: Essential framework components, production configurations, user documentation
- **Audience**: End users who want to use RIF in their projects
- **Workflow**: Automated sync from main with development artifacts removed

## Automated Synchronization

### Sync Triggers
- Push to main branch
- Manual workflow dispatch
- Scheduled sync (weekly)

### Sync Process
1. **Checkout main branch** with full history
2. **Create/update deploy branch** from main
3. **Remove development artifacts** using exclusion patterns
4. **Clean knowledge base** for production use
5. **Update configuration** for deployment mode
6. **Commit and push** changes to deploy branch

### Exclusion Patterns

#### Development Files
```
knowledge/audits/**
knowledge/enforcement_logs/**
knowledge/evidence_collection/**
knowledge/false_positive_detection/**
tests/test_issue_*
validation/**
incidents/**
*.test.*
*issue_[0-9]*
audit_*.py
validate_*.py
```

#### Temporary Files
```
*.log
*.tmp
.coverage
htmlcov/**
test_output/**
migration.log
.env.*
```

#### Self-Development Artifacts
```
CHAT_ERROR_CAPTURE_ANALYSIS_SUMMARY.md
DPIBS_RESEARCH_PHASE1_IMPLEMENTATION_COMPLETE.md
IMPLEMENTATION_COMPLETE_SUMMARY.md
ISSUE_*_IMPLEMENTATION_COMPLETE.md
RIF_LEARNING_REPORT_*.md
VALIDATION_REPORT_*.md
```

## Branch Protection Rules

### Deploy Branch Protection
- **Direct pushes**: Disabled (automated sync only)
- **Force pushes**: Disabled
- **Required status checks**: None (trust main branch testing)
- **Required reviews**: None (automated sync)
- **Restrict pushes**: GitHub Actions only

### Main Branch Protection
- **Direct pushes**: Enabled for maintainers
- **Force pushes**: Disabled
- **Required status checks**: All quality gates must pass
- **Required reviews**: 1+ for external contributors

## Configuration Management

### Deploy Configuration Override
```json
{
  "version": "1.0.0",
  "deployment_mode": "template",
  "features": {
    "self_development_checks": false,
    "audit_logging": false,
    "development_telemetry": false,
    "shadow_mode": false,
    "quality_gates": true,
    "pattern_learning": true
  },
  "knowledge": {
    "clean_on_init": true,
    "preserve_patterns": true,
    "preserve_decisions": false
  }
}
```

### Environment Variables
```bash
# Main branch (development)
RIF_MODE=development
RIF_SELF_DEVELOPMENT=true
RIF_AUDIT_LOGGING=true

# Deploy branch (production)
RIF_MODE=production
RIF_SELF_DEVELOPMENT=false
RIF_AUDIT_LOGGING=false
```

## Knowledge Base Management

### Development Knowledge (main)
- Full issue history and resolutions
- Development patterns and learnings
- Self-improvement metrics
- Quality enforcement logs
- Audit trails and evidence

### Deployment Knowledge (deploy)
- Essential patterns for common use cases
- Production-tested architectural decisions
- Core templates and configurations
- User-focused documentation
- Clean initialization data

## User Experience

### Cloning for Development
```bash
# For RIF framework development
git clone https://github.com/user/rif.git
cd rif
./rif-init.sh --mode development
```

### Cloning for Project Use
```bash
# For using RIF in your project
git clone -b deploy https://github.com/user/rif.git my-project
cd my-project
./rif-init.sh --mode production
```

## Maintenance Procedures

### Manual Sync Trigger
```bash
# Trigger manual sync
gh workflow run deploy-sync.yml
```

### Emergency Rollback
```bash
# Rollback deploy branch to previous state
git checkout deploy
git reset --hard HEAD~1
git push --force-with-lease origin deploy
```

### Branch Health Check
```bash
# Check deploy branch status
git checkout deploy
git log --oneline -10
git diff main...deploy --stat
```

## Quality Assurance

### Pre-Sync Validation
- Main branch tests must pass
- No security vulnerabilities in sync scope
- Configuration validation
- Essential file integrity check

### Post-Sync Validation
- Deploy branch integrity check
- Configuration file validation
- Essential functionality verification
- User initialization test

### Monitoring
- Sync success/failure alerts
- Branch divergence monitoring
- User feedback tracking
- Performance impact assessment

## Troubleshooting

### Common Issues

#### Sync Failures
1. **Permission denied**: Check GitHub Actions token permissions
2. **Merge conflicts**: Manual resolution required, check exclusion patterns
3. **Missing files**: Verify exclusion patterns didn't remove essential files

#### Deploy Branch Issues
1. **Incomplete functionality**: Review exclusion patterns for over-removal
2. **Configuration errors**: Validate deploy.config.json syntax
3. **Initialization failures**: Test user onboarding flow

### Recovery Procedures

#### Rebuild Deploy Branch
```bash
# Full rebuild from main
git checkout main
git pull origin main
git branch -D deploy
git checkout -b deploy
# Run cleaning script
./scripts/clean-for-deploy.sh
git push --force-with-lease origin deploy
```

## Benefits

### For Framework Development
- Clean separation of concerns
- Full development history preserved
- Unrestricted experimentation space
- Complete audit trail

### For End Users
- Clean, professional experience
- Fast initialization
- No development clutter
- Production-ready defaults

### For Maintenance
- Automated synchronization
- Reduced manual effort
- Consistent deployments
- Clear release process

## Implementation Notes

- Sync process is idempotent and safe to re-run
- Deploy branch is fully automated - no manual commits
- Knowledge base cleaning preserves user-valuable patterns
- Configuration override ensures production-appropriate behavior