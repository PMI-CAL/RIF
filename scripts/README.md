# RIF Knowledge Base Cleanup and Deployment Scripts

This directory contains scripts for cleaning the RIF knowledge base and preparing it for deployment. The cleanup system removes development-specific artifacts while preserving valuable patterns and framework components.

## Overview

The RIF knowledge base accumulates significant development data (~90MB+) during its own development. For deployment to new projects, this needs to be cleaned to:

- Remove RIF-specific development artifacts
- Preserve reusable patterns and framework components
- Reduce size for efficient deployment (~5-10MB target)
- Initialize clean databases for new projects

## Scripts

### 1. `deploy_knowledge.sh` - Master Deployment Script

**Primary script that orchestrates the complete cleanup process.**

```bash
# Complete deployment preparation
./scripts/deploy_knowledge.sh

# Test what would happen without making changes
./scripts/deploy_knowledge.sh --dry-run

# Quick deployment without backup (not recommended)
./scripts/deploy_knowledge.sh --skip-backup
```

**What it does:**
1. Analyzes current knowledge base size and composition
2. Creates compressed backup with rollback capability
3. Performs intelligent cleanup of development artifacts
4. Validates cleaning results for deployment readiness
5. Creates deployment package with initialization scripts
6. Generates deployment instructions

### 2. `clean_knowledge_for_deploy.py` - Core Cleanup Engine

**Intelligent cleanup script that safely removes development artifacts.**

```bash
# Perform cleanup with backup
python3 scripts/clean_knowledge_for_deploy.py

# Test cleanup without making changes
python3 scripts/clean_knowledge_for_deploy.py --dry-run

# Custom backup location
python3 scripts/clean_knowledge_for_deploy.py --backup-dir /custom/backup/path
```

**Cleanup Strategy:**

| Category | Action | Rationale |
|----------|--------|-----------|
| **Remove Completely** | `audits/`, `checkpoints/`, `issues/`, `metrics/`, `enforcement_logs/` | RIF development artifacts |
| **Selective Filter** | `patterns/`, `decisions/`, `learning/` | Mix of reusable and RIF-specific content |
| **Preserve As-Is** | `conversations/`, `context/`, `embeddings/`, `parsing/` | Core framework components |
| **Reset/Initialize** | `chromadb/`, `*.duckdb` | Clear databases for new project |

### 3. `analyze_knowledge_size.py` - Size Analysis Tool

**Detailed analysis of knowledge base size and composition.**

```bash
# Basic analysis
python3 scripts/analyze_knowledge_size.py

# Detailed analysis with file listings
python3 scripts/analyze_knowledge_size.py --detailed

# Export analysis to CSV
python3 scripts/analyze_knowledge_size.py --export-csv analysis.csv
```

**Provides:**
- Size breakdown by directory
- File type analysis
- Large file identification
- Cleanup recommendations
- Deployment action mapping

### 4. `test_cleaned_knowledge.py` - Validation Suite

**Comprehensive validation of cleaned knowledge base for deployment readiness.**

```bash
# Validate cleaned knowledge base
python3 scripts/test_cleaned_knowledge.py

# Save detailed validation report
python3 scripts/test_cleaned_knowledge.py --output-report validation.md

# Export results as JSON
python3 scripts/test_cleaned_knowledge.py --json-output results.json
```

**Validation Checks:**
- ✅ **Structure Check**: Required directories present, forbidden removed
- ✅ **Content Check**: Reusable patterns preserved, development artifacts removed
- ✅ **Database Check**: Databases properly reset/initialized
- ✅ **Cleanup Check**: No RIF-specific artifacts remaining
- ✅ **Size Check**: Size appropriate for deployment (<20MB)

### 5. `init_project_knowledge.py` - Project Initialization

**Initializes clean knowledge base for new projects.**

```bash
# Initialize for new project
python3 scripts/init_project_knowledge.py \
  --project-name "My Web App" \
  --project-type "web-app" \
  --preserved-knowledge valuable_knowledge_20250825_123456.json
```

**Project Types:**
- `web-app`, `mobile-app`, `desktop-app`
- `library`, `framework`, `api`, `microservices`
- `enterprise`, `prototype`, `poc`, `experimental`

**Creates:**
- Project metadata configuration
- Quality thresholds based on project type
- Minimal directory structure
- Empty databases
- Migration guide

## Usage Workflow

### Complete Deployment (Recommended)

```bash
# 1. Run master deployment script
./scripts/deploy_knowledge.sh

# 2. Use deployment package in new project
tar -xzf knowledge_backup/rif_knowledge_deployment_*.tar.gz -C /target/project

# 3. Initialize for specific project
cd /target/project
python3 scripts/init_project_knowledge.py \
  --project-name "New Project" \
  --project-type "web-app"

# 4. Complete RIF setup
./rif-init.sh
```

### Custom Cleanup Process

```bash
# 1. Analyze current state
python3 scripts/analyze_knowledge_size.py --detailed

# 2. Test cleanup (dry run)
python3 scripts/clean_knowledge_for_deploy.py --dry-run

# 3. Perform actual cleanup
python3 scripts/clean_knowledge_for_deploy.py

# 4. Validate results
python3 scripts/test_cleaned_knowledge.py

# 5. Initialize for new project
python3 scripts/init_project_knowledge.py \
  --project-name "My Project" \
  --project-type "web-app"
```

### Development/Testing Workflow

```bash
# Quick size check
python3 scripts/analyze_knowledge_size.py | head -20

# Test cleanup without changes
python3 scripts/clean_knowledge_for_deploy.py --dry-run

# Validate current state
python3 scripts/test_cleaned_knowledge.py
```

## Output Locations

All outputs are saved to `knowledge_backup/` directory:

```
knowledge_backup/
├── knowledge_backup_YYYYMMDD_HHMMSS.tar.gz    # Full backup
├── valuable_knowledge_YYYYMMDD_HHMMSS.json    # Exported patterns/decisions
├── cleanup_report_YYYYMMDD_HHMMSS.md          # Cleanup summary
├── validation_report_YYYYMMDD_HHMMSS.md       # Validation results
├── rif_knowledge_deployment_YYYYMMDD_HHMMSS.tar.gz  # Deployment package
├── rollback_YYYYMMDD_HHMMSS.sh                # Rollback script
└── DEPLOYMENT_INSTRUCTIONS.md                 # Deployment guide
```

## Size Reduction Results

**Typical Results:**
- **Before Cleanup**: ~90MB (1000+ files)
- **After Cleanup**: ~5-10MB (200-300 files)
- **Space Saved**: ~85MB (95% reduction)

**What's Removed:**
- 4.2MB audits/ (99 RIF audit files)
- 1.8MB checkpoints/ (238 issue checkpoints)
- 580KB issues/ (52 RIF issue resolutions)
- 376KB metrics/ (32 development metrics)
- 62MB chromadb/ (reset for new project)
- Development logs, evidence packages, enforcement logs

**What's Preserved:**
- Reusable patterns (API resilience, database patterns, etc.)
- Framework architecture decisions
- Core system components (conversations, context, parsing, etc.)
- Quality metrics framework
- Pattern application engine

## Safety Features

### Backup and Rollback
- **Automatic backup** before any cleanup
- **Compressed storage** with timestamps
- **Rollback scripts** generated automatically
- **Validation** before deployment

### Intelligent Filtering
- **Pattern recognition** for reusable vs. RIF-specific content
- **Safe removal** of development artifacts only
- **Preservation** of framework components
- **Content validation** to prevent data loss

### Dry Run Mode
- **Test all operations** without making changes
- **Preview cleanup results** before committing
- **Size estimation** and impact analysis
- **Validation simulation**

## Error Handling

### Common Issues and Solutions

**Issue**: Validation fails with structure errors
```bash
# Check what's missing
python3 scripts/test_cleaned_knowledge.py --json-output results.json
# Re-run cleanup if needed
python3 scripts/clean_knowledge_for_deploy.py
```

**Issue**: Size still too large after cleanup
```bash
# Analyze remaining large files
python3 scripts/analyze_knowledge_size.py --detailed
# Check for missed development artifacts
```

**Issue**: Need to restore from backup
```bash
# Use generated rollback script
./knowledge_backup/rollback_YYYYMMDD_HHMMSS.sh
```

**Issue**: Deployment package creation fails
```bash
# Check validation passed first
python3 scripts/test_cleaned_knowledge.py
# Ensure sufficient disk space
```

### Troubleshooting

**Permissions Issues:**
```bash
# Ensure scripts are executable
chmod +x scripts/*.sh
# Check knowledge directory permissions
ls -la knowledge/
```

**Python Dependencies:**
```bash
# Most scripts use only standard library
# No additional dependencies required
python3 --version  # Requires Python 3.6+
```

**Disk Space:**
```bash
# Cleanup requires temporary space for backup
df -h .
# Deployment package needs ~2x knowledge size
```

## Integration with RIF

These scripts are designed to work with the complete RIF ecosystem:

1. **Development Phase**: Knowledge base accumulates learnings
2. **Cleanup Phase**: Use these scripts to prepare for deployment
3. **Deployment Phase**: Deploy clean knowledge base to new projects
4. **Production Phase**: New project starts with clean slate + preserved patterns

## Best Practices

### Before Cleanup
- ✅ Commit any pending changes to git
- ✅ Review current knowledge base contents
- ✅ Test with `--dry-run` first
- ✅ Ensure sufficient disk space for backup

### During Cleanup
- ✅ Use default backup settings (don't skip)
- ✅ Monitor cleanup logs for any warnings
- ✅ Validate results before deploying
- ✅ Keep rollback script accessible

### After Cleanup
- ✅ Test deployment package before distribution
- ✅ Initialize project with appropriate type
- ✅ Validate RIF functionality in new project
- ✅ Archive original backup safely

## Contributing

When modifying these scripts:

1. **Test thoroughly** with dry run mode
2. **Preserve backup functionality** for safety
3. **Update validation checks** for new requirements  
4. **Document any new project types** or cleanup rules
5. **Maintain backward compatibility** with existing backups

## Support

For issues with the cleanup system:

1. Check validation reports for specific errors
2. Review cleanup logs for warnings
3. Test with dry run mode to debug issues
4. Use rollback scripts if restoration needed
5. Refer to deployment instructions for new project setup