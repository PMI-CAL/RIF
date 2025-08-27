# PR Automation Operational Runbook

**Issue**: #283 - Evolution of RIF PR Automation  
**Phase**: 4 - Optimization and Documentation  
**System**: GitHub-Native Parallel Processing  
**Version**: 1.0.0  
**Last Updated**: 2025-08-27

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Standard Operating Procedures](#standard-operating-procedures)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Emergency Procedures](#emergency-procedures)
7. [Performance Tuning](#performance-tuning)
8. [Common Issues and Solutions](#common-issues-and-solutions)
9. [Escalation Procedures](#escalation-procedures)

---

## System Overview

The PR Automation System is a GitHub-native parallel processing pipeline that intelligently routes pull requests through different automation levels based on complexity scoring. It replaces the previous blocking PR priority system with a weighted, parallel execution model.

### Key Features
- **Progressive Automation**: 3 levels from full automation to RIF agent integration
- **Parallel Processing**: Non-blocking execution of PR validation and development work
- **Intelligent Routing**: Complexity-based PR classification and handling
- **Quality Gate Enforcement**: Automated quality checks with auto-fix capabilities
- **Auto-Merge System**: Safe automatic merging based on configurable rules

### System Architecture

```
GitHub PR → Complexity Assessment → Automation Router
                                          ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    │  Level 1        │  Level 2        │  Level 3        │
    │  GitHub Only    │  Copilot        │  RIF            │
    │  (Trivial)      │  Assisted       │  Integration    │
    │                 │  (Simple/Med)   │  (Complex)      │
    └─────────────────┴─────────────────┴─────────────────┘
              ↓                 ↓                 ↓
         Auto-Merge      Copilot Review    RIF Agent Review
```

---

## Architecture Components

### 1. GitHub Actions Workflows

#### Primary Workflows
- **`rif-pr-automation.yml`**: Main orchestration workflow
- **`rif-pr-quality-gates.yml`**: Quality validation pipeline
- **`intelligent-pr-automation.yml`**: Complexity assessment and routing

#### Workflow Triggers
- `pull_request`: [opened, synchronize, reopened, ready_for_review]
- `check_run`: [completed] 
- `workflow_run`: [completed]

### 2. Quality Gates

#### Gate 1: Code Quality Analysis
- ESLint/TypeScript checking
- Python linting (flake8, black, isort)
- Quality score calculation (target: >80%)

#### Gate 2: Security Scanning  
- CodeQL analysis
- Dependency vulnerability scanning (npm audit, Snyk, safety)
- Critical/high vulnerability count (target: 0)

#### Gate 3: Test Coverage
- JavaScript/TypeScript test execution
- Python test execution with coverage
- Coverage percentage tracking (target: >80%)

#### Gate 4: Performance Testing
- Performance benchmark execution
- Performance score calculation (target: >90%)

#### Gate 5: RIF Validation
- Branch protection validation
- RIF workflow compliance
- Agent instruction validation
- Code quality standards verification

### 3. Auto-Fix System

#### Enabled Fixes
- Linting errors (ESLint --fix)
- Code formatting (black, isort)
- Security updates (npm audit fix)
- Dependency updates

#### Fix Process
1. Quality gate failure detected
2. Auto-fix workflow triggered
3. Fixes applied and committed
4. Re-run validation
5. Continue to merge if successful

### 4. Complexity Scoring

#### Classification Levels
- **Trivial**: <10 lines, dependency updates, documentation
- **Simple**: <50 lines, <5 files, no architecture changes
- **Medium**: <200 lines, <15 files, test changes required
- **Complex**: Large changes, architecture impact, security sensitive

#### Routing Logic
```yaml
Trivial → GitHub Only (Full Automation)
Simple/Medium → Copilot Assisted (AI Review)
Complex → RIF Integration (Agent Review)
```

---

## Standard Operating Procedures

### 1. Daily Operations

#### Morning Checklist
1. **Check GitHub Actions Dashboard**
   ```bash
   # Navigate to: https://github.com/{org}/{repo}/actions
   # Verify: No failed workflows in last 24h
   ```

2. **Review Automation Metrics**
   ```bash
   # Check monitoring dashboard
   cat monitoring/pr-automation-metrics.json
   # Verify targets are being met
   ```

3. **Validate Auto-Merge Queue**
   ```bash
   gh pr list --state open --label "ready-for-merge"
   # Ensure PRs are processing through merge queue
   ```

#### Weekly Maintenance
1. **Performance Analysis**
   - Review PR processing times
   - Analyze automation success rates
   - Check resource utilization

2. **Configuration Tuning**
   - Adjust complexity thresholds if needed
   - Update quality gate settings
   - Optimize timeout values

3. **System Health Check**
   - Validate all workflows are operational
   - Check branch protection rules
   - Verify webhook configurations

### 2. PR Processing Workflow

#### Automatic PR Processing
1. **PR Created** → Complexity assessment runs
2. **Classification** → Route to appropriate automation level
3. **Quality Gates** → Run in parallel
4. **Auto-Fix** → Apply fixes if gates fail
5. **Review** → Copilot or RIF agent review
6. **Auto-Merge** → Merge if all conditions met

#### Manual Intervention Points
- Complex PRs always require human approval
- Security-sensitive changes need security team review
- Architecture changes need architect approval
- Failed auto-fixes need developer attention

### 3. Configuration Management

#### Updating Automation Rules
1. **Edit configuration file**
   ```bash
   vim config/pr-automation-tuning.yaml
   ```

2. **Validate configuration**
   ```bash
   python scripts/validate_configuration.py
   ```

3. **Deploy changes**
   ```bash
   scripts/deploy-pr-automation.sh
   ```

4. **Monitor impact**
   ```bash
   # Watch for 1-2 hours after changes
   gh workflow list --all
   ```

---

## Monitoring and Alerting

### 1. Key Metrics

#### Performance Metrics
- **PR Processing Time**: Target <35 minutes
- **Automation Rate**: Target >85%
- **Quality Gate Pass Rate**: Target >95%
- **Parallel Execution**: Target 4-6 streams

#### Health Metrics
- **Workflow Success Rate**: Target >98%
- **Auto-Fix Success Rate**: Target >70%
- **False Positive Rate**: Target <5%
- **System Availability**: Target >99.9%

### 2. Monitoring Dashboard

#### GitHub Actions Dashboard
```
URL: https://github.com/{org}/{repo}/actions
Key Views:
- Workflow runs (last 24h)
- Success/failure rates
- Execution times
- Resource usage
```

#### Custom Metrics Dashboard
```bash
# View current metrics
cat monitoring/pr-automation-metrics.json

# Key metrics to monitor:
# - pr_processing_time.current_avg
# - automation_rate.current
# - quality_gate_pass_rate.current
# - parallel_execution_capacity.current
```

### 3. Alert Conditions

#### Critical Alerts (Immediate Response)
- Security vulnerabilities detected in main branch
- System-wide workflow failures (>80% failure rate)
- Auto-merge system disabled
- Branch protection rules compromised

#### Warning Alerts (1-hour Response)
- PR processing time >60 minutes
- Quality gate pass rate <90%
- Automation rate <75%
- High auto-fix failure rate

#### Info Alerts (Daily Review)
- Performance degradation trends
- Unusual complexity distribution
- Resource usage increases

---

## Troubleshooting Guide

### 1. Common Workflow Issues

#### Problem: Workflow Not Triggering
**Symptoms**: PR created but no automation workflows start

**Diagnosis**:
```bash
# Check workflow files exist
ls -la .github/workflows/

# Verify branch protection
gh api repos/{org}/{repo}/branches/main/protection

# Check webhook delivery
gh api repos/{org}/{repo}/hooks
```

**Solution**:
```bash
# Re-deploy workflows
scripts/deploy-pr-automation.sh

# Manually trigger workflow
gh workflow run "RIF PR Automation" --ref main
```

#### Problem: Quality Gates Failing
**Symptoms**: All PRs failing quality checks

**Diagnosis**:
```bash
# Check recent workflow runs
gh run list --workflow="RIF PR Quality Gates" --limit=10

# View specific failure
gh run view {run-id}
```

**Solution**:
```bash
# Lower quality thresholds temporarily
vim config/pr-automation-tuning.yaml
# Update: quality_gates.code_quality.minimum_score: 70

# Re-deploy configuration
scripts/deploy-pr-automation.sh --validate-only
```

#### Problem: Auto-Merge Not Working  
**Symptoms**: PRs passing all checks but not auto-merging

**Diagnosis**:
```bash
# Check auto-merge setting
gh api repos/{org}/{repo} --jq '.allow_auto_merge'

# Verify PR labels
gh pr view {pr-number} --json labels
```

**Solution**:
```bash
# Enable auto-merge
gh api repos/{org}/{repo} --method PATCH -f allow_auto_merge=true

# Manually enable for PR
gh pr merge {pr-number} --auto --squash
```

### 2. Performance Issues

#### Problem: Slow PR Processing
**Symptoms**: PRs taking >60 minutes to process

**Diagnosis**:
```bash
# Check workflow execution times
gh run list --limit=20 --json conclusion,createdAt,updatedAt

# Analyze bottlenecks
gh run view {run-id} --log
```

**Solution**:
```bash
# Increase parallel execution
vim config/pr-automation-tuning.yaml
# Update: performance.parallel_execution.max_concurrent_workflows: 6

# Enable caching optimization
# Update: performance.caching.dependency_cache_ttl: 7200
```

#### Problem: Resource Exhaustion
**Symptoms**: Workflows queued, "waiting for runner" messages

**Diagnosis**:
```bash
# Check GitHub Actions usage
gh api repos/{org}/{repo}/actions/cache/usage

# Monitor concurrent workflows
gh run list --status in_progress
```

**Solution**:
```bash
# Reduce parallel execution temporarily
vim config/pr-automation-tuning.yaml
# Update: performance.parallel_execution.max_concurrent_workflows: 2

# Optimize resource usage
# Update: performance.parallel_execution.resource_limits.memory_per_workflow: "2GB"
```

### 3. Auto-Fix Issues

#### Problem: Auto-Fix Loops
**Symptoms**: PRs getting repeated auto-fix commits

**Diagnosis**:
```bash
# Check commit history
git log --oneline --grep="Auto-fix" --since="1 day ago"

# View auto-fix configuration
grep -A 10 "auto_fix:" config/pr-automation-tuning.yaml
```

**Solution**:
```bash
# Limit auto-fix attempts
vim config/pr-automation-tuning.yaml
# Update: timeouts.retry_policies.auto_fix_attempts.max_retries: 1

# Add cooldown period
# Update: timeouts.retry_policies.auto_fix_attempts.retry_delay: 300
```

---

## Emergency Procedures

### 1. System-Wide Issues

#### Emergency Shutdown
If the automation system is causing widespread issues:

```bash
# 1. Disable auto-merge immediately
gh api repos/{org}/{repo} --method PATCH -f allow_auto_merge=false

# 2. Disable branch protection (temporary)
gh api repos/{org}/{repo}/branches/main/protection --method DELETE

# 3. Pause all workflows
# Navigate to GitHub Actions settings and disable workflows

# 4. Notify team
echo "PR automation system disabled due to emergency" | \
  gh issue create --title "EMERGENCY: PR Automation Disabled" --body-file -
```

#### Emergency Rollback
```bash
# 1. Switch to emergency branch
git checkout emergency/rollback-pr-automation

# 2. Revert to previous stable configuration
git revert HEAD~1

# 3. Re-deploy immediately
scripts/deploy-pr-automation.sh

# 4. Monitor for 30 minutes
watch -n 30 'gh workflow list --all'
```

### 2. Security Incidents

#### Vulnerable Code Merged
```bash
# 1. Immediately revert merge
gh pr view {pr-number} --json mergeCommit
git revert {merge-commit-sha}

# 2. Force push to main
git push origin main --force-with-lease

# 3. Create security incident
gh issue create --title "SECURITY: Vulnerable code merged" \
  --label "security,critical" \
  --body "Vulnerable code from PR #{pr-number} has been reverted"

# 4. Review security scanning configuration
vim config/pr-automation-tuning.yaml
# Check: quality_gates.security.max_critical_vulnerabilities: 0
```

#### Compromised Auto-Merge
```bash
# 1. Disable auto-merge immediately
gh api repos/{org}/{repo} --method PATCH -f allow_auto_merge=false

# 2. Review all recent merges
gh pr list --state merged --limit 20

# 3. Audit automation rules
git log --oneline --grep="Auto-merge" --since="24 hours ago"

# 4. Implement additional safeguards
vim config/pr-automation-tuning.yaml
# Update: auto_merge_delay to higher value
# Update: required_checks to include manual approval
```

### 3. Data Loss Prevention

#### Backup Critical Configuration
```bash
# Daily backup (automated in cron)
#!/bin/bash
cp -r .github/workflows/ backup/workflows-$(date +%Y%m%d)/
cp config/pr-automation-tuning.yaml backup/config-$(date +%Y%m%d).yaml
cp monitoring/pr-automation-metrics.json backup/metrics-$(date +%Y%m%d).json
```

#### Recovery Procedures
```bash
# Restore from backup
cp backup/workflows-{date}/* .github/workflows/
cp backup/config-{date}.yaml config/pr-automation-tuning.yaml

# Validate and deploy
scripts/deploy-pr-automation.sh --validate-only
scripts/deploy-pr-automation.sh
```

---

## Performance Tuning

### 1. Optimization Strategies

#### Workflow Performance
```yaml
# config/pr-automation-tuning.yaml
performance:
  parallel_execution:
    max_concurrent_workflows: 4  # Adjust based on runner availability
    max_parallel_quality_gates: 6  # Balance speed vs resource usage
    
  caching:
    dependency_cache_ttl: 3600  # Increase for stable dependencies
    build_cache_ttl: 86400      # Long-lived for unchanged builds
    
  optimization_strategies:
    incremental_builds: true    # Only rebuild changed components
    selective_testing: true     # Run only affected tests
    parallel_linting: true      # Parallel lint execution
```

#### Quality Gate Optimization
```yaml
quality_gates:
  code_quality:
    minimum_score: 75          # Lower for faster feedback, raise gradually
    eslint_max_warnings: 10    # Allow more warnings initially
    
  test_coverage:
    minimum_coverage: 75       # Start lower, increase over time
    new_code_coverage_minimum: 85  # Higher bar for new code
```

### 2. Monitoring-Based Tuning

#### Performance Metrics Analysis
```bash
# Weekly performance review
python scripts/analyze-pr-metrics.py --period=7days

# Key metrics to track:
# - Average processing time per complexity level
# - Quality gate pass rates by type
# - Auto-fix success rates
# - Resource utilization patterns
```

#### Threshold Adjustment Process
1. **Collect baseline metrics** (2 weeks)
2. **Identify bottlenecks** using performance data
3. **Adjust thresholds** incrementally (10-20% changes)
4. **Monitor impact** for 1 week
5. **Iterate** until targets are met

### 3. Capacity Planning

#### Scaling Guidelines
```yaml
# Current capacity: ~20 PRs/hour
# Target capacity: ~50 PRs/hour

# Scaling factors:
concurrent_workflows: 4 → 8    # 2x throughput
parallel_quality_gates: 6 → 12 # 2x validation speed
resource_limits:
  cpu_per_workflow: "2 cores" → "4 cores"  # Faster execution
  memory_per_workflow: "4GB" → "8GB"       # Handle larger PRs
```

---

## Common Issues and Solutions

### 1. Configuration Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Invalid YAML** | Workflows not loading | Use `yaml-lint` to validate syntax |
| **Missing secrets** | Authentication failures | Check repository secrets in Settings |
| **Wrong branch protection** | Auto-merge blocked | Update protection rules via GitHub API |
| **Outdated workflow syntax** | Deprecation warnings | Update to latest Actions syntax |

### 2. Integration Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Webhook failures** | Events not triggering | Check webhook delivery logs |
| **API rate limiting** | Intermittent failures | Increase rate limit buffer |
| **Token permissions** | Permission denied errors | Update token scopes |
| **RIF agent conflicts** | Duplicate work | Coordinate through state management |

### 3. Quality Gate Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **False positive security alerts** | Clean code flagged | Adjust security tool configuration |
| **Flaky tests** | Intermittent test failures | Implement test retry logic |
| **Coverage calculation errors** | Incorrect coverage reports | Fix coverage tool configuration |
| **Slow performance tests** | Timeout issues | Optimize test execution |

### 4. Auto-Fix Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Fix conflicts** | Merge conflicts in fixes | Improve fix conflict resolution |
| **Incomplete fixes** | Issues not fully resolved | Enhance fix patterns |
| **Fix loops** | Repeated fix attempts | Add fix attempt limits |
| **Large fix commits** | Overwhelming changes | Limit changes per fix |

---

## Escalation Procedures

### 1. Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| **Critical** | 15 minutes | On-call → Engineering Lead → Director |
| **High** | 1 hour | Team Lead → Engineering Lead |
| **Medium** | 4 hours | Developer → Team Lead |
| **Low** | 1 business day | Developer handles |

### 2. Escalation Triggers

#### Critical (15-minute response)
- Main branch protection compromised
- Security vulnerabilities in production
- System-wide automation failure
- Data loss or corruption

#### High (1-hour response)  
- Quality gates completely failing
- Auto-merge system malfunction
- Performance degradation >50%
- Multiple workflow failures

#### Medium (4-hour response)
- Single workflow consistently failing
- Performance degradation 20-50%
- Configuration issues affecting subset
- Integration problems

#### Low (1-day response)
- Individual PR processing issues
- Minor performance optimization
- Documentation updates
- Monitoring improvements

### 3. Communication Channels

#### Internal Communication
- **Slack**: #pr-automation-alerts (real-time)
- **Email**: engineering-leads@company.com (formal)
- **GitHub Issues**: Label with "pr-automation" + severity

#### External Communication
- **Status Page**: Update if user-impacting
- **Team Notifications**: Use appropriate channels
- **Documentation**: Update runbook with lessons learned

### 4. Post-Incident Process

#### Immediate (0-24 hours)
1. **Resolve immediate issue**
2. **Document timeline and actions**
3. **Implement temporary monitoring**
4. **Communicate resolution**

#### Short-term (1-7 days)
1. **Root cause analysis**
2. **Implement permanent fix**
3. **Update monitoring/alerting**
4. **Update documentation**

#### Long-term (1-4 weeks)
1. **Review system design**
2. **Implement preventive measures**
3. **Update procedures**
4. **Team training/knowledge sharing**

---

## Appendix

### A. Configuration File References
- **Main Config**: `config/pr-automation-tuning.yaml`
- **Workflows**: `.github/workflows/`
- **Scripts**: `scripts/deploy-pr-automation.sh`
- **Monitoring**: `monitoring/pr-automation-metrics.json`

### B. Useful Commands
```bash
# Quick health check
gh workflow list --all | grep -E "(RIF|PR)"

# View recent PR processing
gh pr list --state all --limit 10 --json number,title,state,createdAt

# Check automation metrics
cat monitoring/pr-automation-metrics.json | jq '.metrics'

# Validate configuration
python scripts/validate_configuration.py

# Emergency disable
gh api repos/{org}/{repo} --method PATCH -f allow_auto_merge=false
```

### C. Contact Information
- **Primary**: Development Team Lead
- **Secondary**: Engineering Manager  
- **Emergency**: On-call rotation (see PagerDuty)
- **Vendor Support**: GitHub Enterprise Support

### D. External Documentation Links
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests)
- [Auto-merge Documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request)

---

*This runbook is a living document. Update it as the system evolves and new issues are discovered.*