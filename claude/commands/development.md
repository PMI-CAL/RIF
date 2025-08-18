# Development Commands Reference

## Technology Stack Setup

```bash
# Install dependencies (technology-specific)
{{INSTALL_DEPENDENCIES_COMMAND}}

# Install additional packages if needed
{{INSTALL_ADDITIONAL_PACKAGES_COMMAND}}

# Copy configuration template
{{COPY_CONFIG_COMMAND}}

# Test installation
{{TEST_INSTALLATION_COMMAND}}
```

## Application Launchers

```bash
# Main application launcher
{{MAIN_LAUNCHER_COMMAND}}

# Development server (if applicable)
{{DEV_SERVER_COMMAND}}

# Build application
{{BUILD_COMMAND}}

# Start application
{{START_COMMAND}}

# Status check
{{STATUS_COMMAND}}
```

## Testing and Quality

```bash
# Run all tests
{{TEST_COMMAND}}

# Run tests with coverage
{{TEST_WITH_COVERAGE_COMMAND}}

# Test specific module/file
{{TEST_SPECIFIC_COMMAND}}

# Code formatting
{{FORMAT_COMMAND}}

# Linting
{{LINT_COMMAND}}

# Type checking (if applicable)
{{TYPE_CHECK_COMMAND}}

# Security scanning
{{SECURITY_SCAN_COMMAND}}
```

## Build and Deployment

```bash
# Build for production
{{BUILD_PRODUCTION_COMMAND}}

# Package application
{{PACKAGE_COMMAND}}

# Deploy to staging
{{DEPLOY_STAGING_COMMAND}}

# Deploy to production
{{DEPLOY_PRODUCTION_COMMAND}}
```

## Database Operations (if applicable)

```bash
# Run database migrations
{{DB_MIGRATE_COMMAND}}

# Create database backup
{{DB_BACKUP_COMMAND}}

# Seed database with test data
{{DB_SEED_COMMAND}}

# Reset database
{{DB_RESET_COMMAND}}
```

## Development Utilities

```bash
# Start development environment
{{DEV_ENV_START_COMMAND}}

# Stop development environment
{{DEV_ENV_STOP_COMMAND}}

# View logs
{{LOG_VIEW_COMMAND}}

# Clear cache
{{CACHE_CLEAR_COMMAND}}

# Run performance benchmarks
{{BENCHMARK_COMMAND}}
```

## Code Quality Automation

```bash
# Run all quality checks
{{QUALITY_CHECK_COMMAND}}

# Fix automatic issues
{{AUTO_FIX_COMMAND}}

# Generate documentation
{{DOCS_GENERATE_COMMAND}}

# Update dependencies
{{DEPS_UPDATE_COMMAND}}
```

## Git and Version Control

```bash
# Create feature branch
git checkout -b feature/{{ISSUE_NUMBER}}-{{FEATURE_DESCRIPTION}}

# Commit with issue reference
git commit -m "{{COMMIT_TYPE}}(#{{ISSUE_NUMBER}}): {{COMMIT_DESCRIPTION}}"

# Push feature branch
git push -u origin feature/{{ISSUE_NUMBER}}-{{FEATURE_DESCRIPTION}}

# Create pull request
gh pr create --title "{{COMMIT_TYPE}}: [Issue #{{ISSUE_NUMBER}}] {{FEATURE_DESCRIPTION}}" --body "Closes #{{ISSUE_NUMBER}}"
```

## GitHub Integration

```bash
# Check current issues
gh issue list --state open

# View specific issue
gh issue view {{ISSUE_NUMBER}}

# Comment on issue
gh issue comment {{ISSUE_NUMBER}} --body "{{COMMENT_BODY}}"

# Add/remove labels
gh issue edit {{ISSUE_NUMBER}} --add-label "{{LABEL_NAME}}"
gh issue edit {{ISSUE_NUMBER}} --remove-label "{{LABEL_NAME}}"

# Close issue
gh issue close {{ISSUE_NUMBER}} --comment "{{CLOSE_COMMENT}}"
```

## Technology-Specific Workflows

### For JavaScript/Node.js Projects
```bash
# Package manager commands
npm install
npm run dev
npm run build
npm test
npm run lint

# Yarn alternative
yarn install
yarn dev
yarn build
yarn test
```

### For Python Projects  
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt
python -m pytest
black src/
flake8 src/
```

### For Java Projects
```bash
# Maven
mvn clean install
mvn test
mvn package

# Gradle
./gradlew build
./gradlew test
./gradlew run
```

### For Go Projects
```bash
# Go commands
go mod download
go build
go test ./...
go run main.go
```

### For Rust Projects
```bash
# Cargo commands
cargo build
cargo test
cargo run
cargo check
```

## Environment Variables

```bash
# Set development environment
export NODE_ENV=development
export {{PROJECT_NAME}}_ENV=development
export {{PROJECT_NAME}}_DEBUG=true

# API keys and secrets
export {{API_KEY_NAME}}="your-api-key"
export {{SECRET_NAME}}="your-secret"

# Database configuration
export DATABASE_URL="{{DATABASE_CONNECTION_STRING}}"
```

## Docker Operations (if applicable)

```bash
# Build Docker image
docker build -t {{PROJECT_NAME}} .

# Run container
docker run -p {{PORT}}:{{PORT}} {{PROJECT_NAME}}

# Docker Compose
docker-compose up -d
docker-compose down
docker-compose logs -f
```

## Monitoring and Debugging

```bash
# Check application health
{{HEALTH_CHECK_COMMAND}}

# View application metrics
{{METRICS_COMMAND}}

# Debug application
{{DEBUG_COMMAND}}

# Profile performance
{{PROFILE_COMMAND}}
```

## Troubleshooting Commands

```bash
# Clear all caches
{{CLEAR_ALL_CACHE_COMMAND}}

# Reinstall dependencies
{{REINSTALL_DEPS_COMMAND}}

# Reset development environment
{{RESET_DEV_ENV_COMMAND}}

# Check system requirements
{{CHECK_REQUIREMENTS_COMMAND}}
```

## Quick Reference

### Essential Commands by Technology

| Technology | Install | Test | Build | Run |
|------------|---------|------|-------|-----|
| Node.js | `npm install` | `npm test` | `npm run build` | `npm start` |
| Python | `pip install -r requirements.txt` | `pytest` | `python setup.py build` | `python main.py` |
| Java | `mvn install` | `mvn test` | `mvn package` | `java -jar target/*.jar` |
| Go | `go mod download` | `go test ./...` | `go build` | `go run main.go` |
| Rust | `cargo build` | `cargo test` | `cargo build --release` | `cargo run` |

### Common Issue Labels for GitHub
- `workflow-state:planning` - Ready for project management
- `workflow-state:implementing` - Ready for development
- `workflow-state:testing` - Ready for quality assurance
- `workflow-state:reviewing` - Ready for code review
- `workflow-state:context-discovery` - Trigger context server discovery

### Development Workflow
1. Check issues: `gh issue list --state open`
2. Create branch: `git checkout -b feature/issue-{{ISSUE_NUMBER}}`
3. Implement changes following project standards
4. Test: `{{TEST_COMMAND}}`
5. Quality check: `{{QUALITY_CHECK_COMMAND}}`
6. Commit: `git commit -m "feat(#{{ISSUE_NUMBER}}): description"`
7. Push: `git push -u origin feature/issue-{{ISSUE_NUMBER}}`
8. Create PR: `gh pr create --title "..." --body "Closes #{{ISSUE_NUMBER}}"`

Remember: All commands are technology-specific and should be replaced with the appropriate commands for your project's technology stack during framework setup.