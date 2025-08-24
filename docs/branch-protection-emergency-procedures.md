# GitHub Branch Protection Emergency Procedures
**Issue #203 - Phase 1: Configure GitHub branch protection rules for main branch**

## Overview

This document outlines emergency procedures for overriding or disabling GitHub branch protection rules when critical situations require immediate access to the main branch.

## Emergency Scenarios

### When to Use Emergency Override

Use emergency override procedures only in the following situations:

1. **Critical Security Patch**
   - Zero-day vulnerability requires immediate patching
   - Security incident response needs urgent code deployment
   - Compliance violation requires immediate correction

2. **Emergency Hotfix**
   - Production system outage caused by recent deployment
   - Data corruption or loss prevention measures
   - Critical business function failure

3. **System Outage Recovery**
   - CI/CD pipeline failure preventing normal workflow
   - GitHub Actions outage blocking quality gates
   - Infrastructure issues preventing proper PR process

4. **Deployment Blocker**
   - Release deadline with quality gate false positives
   - External dependency failures blocking legitimate changes
   - Critical business milestone requiring immediate delivery

### When NOT to Use Emergency Override

Do **NOT** use emergency procedures for:
- Convenience or time pressure
- Avoiding code review processes
- Working around test failures that indicate real issues
- Personal preference or workflow shortcuts
- Non-critical bug fixes or features

## Emergency Override Procedures

### Step 1: Assessment and Authorization

1. **Verify Emergency Status**
   ```bash
   # Document the emergency situation
   EMERGENCY_REASON="[critical_security_patch|emergency_hotfix|system_outage_recovery|deployment_blocker]"
   IMPACT_ASSESSMENT="Description of impact if not resolved immediately"
   ROLLBACK_PLAN="Plan for restoring normal branch protection after emergency"
   ```

2. **Get Authorization**
   - Emergency contact: **PMI-CAL** (repository owner)
   - Required role: **admin** or **owner**
   - Document approval: timestamp, approver, reason

### Step 2: Emergency Rollback Execution

#### Option A: Using Rollback Script (Recommended)

```bash
# Navigate to repository root
cd /path/to/RIF

# Execute rollback with audit reason
python3 scripts/rollback_branch_protection.py \
  --branch main \
  --reason "critical_security_patch" \
  --emergency-contact "PMI-CAL" \
  --approval-timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

#### Option B: Manual GitHub CLI

```bash
# Remove branch protection directly
gh api \
  -X DELETE \
  /repos/{owner}/{repo}/branches/main/protection \
  --silent

# Verify removal
gh api /repos/{owner}/{repo}/branches/main/protection || echo "Protection removed"
```

#### Option C: GitHub Web Interface

1. Navigate to: `Settings > Branches > main > Edit`
2. Uncheck all protection rules
3. Save changes
4. Document action in emergency log

### Step 3: Emergency Operations

Once protection is removed:

1. **Perform Emergency Work**
   ```bash
   # Create emergency branch from main
   git checkout main
   git pull origin main
   git checkout -b emergency-fix-$(date +%Y%m%d-%H%M%S)
   
   # Make required changes
   # ... emergency fixes ...
   
   # Commit and push directly to main (emergency only)
   git add .
   git commit -m "EMERGENCY: [description] - Bypass protection due to [reason]"
   git push origin main
   ```

2. **Document Emergency Actions**
   ```bash
   # Log all actions taken
   echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): Emergency push completed" >> .github/branch-protection-emergency-log.json
   ```

### Step 4: Restoration

**Critical**: Protection must be restored immediately after emergency work:

1. **Restore Branch Protection**
   ```bash
   # Restore protection using setup script
   python3 scripts/setup_branch_protection.py \
     --branch main \
     --start-step 5  # Start from final step for full protection
   ```

2. **Verify Restoration**
   ```bash
   # Check protection is active
   gh api /repos/{owner}/{repo}/branches/main/protection
   
   # Verify all rules are in place
   python3 scripts/setup_branch_protection.py --validate-only
   ```

3. **Create Post-Emergency PR**
   ```bash
   # Create proper PR for emergency changes (if needed)
   git checkout -b post-emergency-review
   # ... document emergency changes ...
   git commit -m "Post-emergency documentation and review"
   gh pr create --title "Post-Emergency Review: [Emergency Description]" \
                --body "## Emergency Override Summary
   
   **Emergency Type**: $EMERGENCY_REASON
   **Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
   **Changes Made**: [Describe emergency changes]
   **Authorization**: [Approver name and timestamp]
   **Protection Restored**: [Timestamp]
   
   ## Review Required
   - [ ] Validate emergency changes were appropriate
   - [ ] Confirm no additional issues introduced
   - [ ] Verify branch protection is fully restored
   - [ ] Update incident documentation if needed"
   ```

## Rollback Script Usage

### Basic Usage

```bash
# Standard rollback
python3 scripts/rollback_branch_protection.py

# With custom reason
python3 scripts/rollback_branch_protection.py --reason "critical_security_patch"

# Emergency rollback with full logging
python3 scripts/rollback_branch_protection.py \
  --branch main \
  --reason "emergency_hotfix" \
  --emergency-contact "PMI-CAL" \
  --approval-timestamp "2025-08-24T14:30:00Z"
```

### Script Options

- `--branch`: Branch to remove protection from (default: main)
- `--reason`: Reason for removal (required for audit)
- `--emergency-contact`: Emergency contact who authorized removal
- `--approval-timestamp`: Timestamp of authorization
- `--confirm`: Skip confirmation prompt (for automated use)
- `--audit-log`: Custom path for audit log

## Monitoring and Alerting

### Automatic Alerts

Branch protection overrides trigger automatic alerts:

1. **Slack/Email Notifications**
   - Sent to: repository administrators, security team
   - Contains: timestamp, reason, who performed override
   - Requires: acknowledgment within 1 hour

2. **GitHub Issue Creation**
   - Automatic issue created for each emergency override
   - Assigned to: repository owner
   - Labels: `emergency`, `security`, `branch-protection`
   - Auto-closes: when protection is restored

3. **Audit Log Updates**
   - All actions logged to: `.github/branch-protection-emergency-log.json`
   - Includes: timestamps, reasons, actors, restoration status
   - Retained: indefinitely for compliance

### Manual Monitoring

Check emergency override status:

```bash
# View recent emergency actions
cat .github/branch-protection-emergency-log.json | tail -10

# Check current protection status
gh api /repos/{owner}/{repo}/branches/main/protection

# View audit trail
python3 scripts/setup_branch_protection.py --show-audit-log
```

## Recovery Procedures

### If Rollback Script Fails

1. **Manual API Rollback**
   ```bash
   # Direct API call
   gh api -X DELETE /repos/{owner}/{repo}/branches/main/protection
   ```

2. **GitHub Web Interface**
   - Access repository settings directly
   - Manually disable all branch protection rules
   - Document manual override in emergency log

3. **Contact GitHub Support**
   - If API and web interface fail
   - Provide: repository details, emergency reason, timeline
   - Request: manual branch protection removal

### If Protection Restoration Fails

1. **Retry with Different Step**
   ```bash
   # Try restoring from step 1
   python3 scripts/setup_branch_protection.py --start-step 1
   ```

2. **Manual Configuration**
   - Use GitHub web interface to manually configure protection
   - Follow configuration in `.github/branch-protection.json`
   - Verify each setting matches the desired configuration

3. **Reset and Reconfigure**
   ```bash
   # Remove any partial configuration
   python3 scripts/rollback_branch_protection.py --reason "failed_restoration"
   
   # Clean restore from beginning
   python3 scripts/setup_branch_protection.py
   ```

## Compliance and Documentation

### Required Documentation

For each emergency override, document:

1. **Emergency Details**
   - Date/time of override
   - Reason and justification  
   - Impact if not performed immediately
   - Authorization details

2. **Actions Taken**
   - Commands executed
   - Files modified
   - Systems affected
   - Timeline of actions

3. **Resolution**
   - Protection restoration timestamp
   - Verification of normal operations
   - Post-emergency review completion
   - Lessons learned

### Audit Requirements

- **Immediate**: Log entry in emergency audit log
- **Within 24 hours**: Post-emergency review PR created
- **Within 1 week**: Incident report completed
- **Monthly**: Review all emergency overrides for patterns

### Review Process

1. **Immediate Review** (within 1 hour)
   - Verify emergency was legitimate
   - Confirm protection has been restored
   - Check for any additional security concerns

2. **Post-Emergency Review** (within 24 hours)
   - Evaluate whether override was necessary
   - Assess if process could be improved
   - Document lessons learned

3. **Monthly Audit** (first week of month)
   - Review all emergency overrides from previous month
   - Identify patterns or recurring issues
   - Update procedures if needed

## Contact Information

### Emergency Contacts

- **Primary**: PMI-CAL (Repository Owner)
- **Secondary**: Repository Administrators
- **Escalation**: GitHub Support

### Emergency Communication

- **Slack**: `#rif-emergency` channel
- **Email**: [emergency contact email]
- **GitHub**: Create issue with `emergency` label

## Testing and Validation

### Quarterly Tests

Test emergency procedures quarterly:

1. **Test Rollback Script**
   ```bash
   # Test on non-production branch
   git checkout -b test-emergency-procedures
   python3 scripts/rollback_branch_protection.py --branch test-emergency-procedures --reason "quarterly_test"
   ```

2. **Test Restoration**
   ```bash
   # Verify restoration works
   python3 scripts/setup_branch_protection.py --branch test-emergency-procedures
   ```

3. **Validate Audit Logging**
   ```bash
   # Check all actions were logged
   cat .github/branch-protection-emergency-log.json
   ```

### Success Criteria

Emergency procedures pass testing if:
- Rollback completes within 2 minutes
- Emergency work can be performed immediately
- Protection restoration completes successfully
- All actions are properly logged
- No data or configuration loss occurs

---

**Last Updated**: 2025-08-24  
**Document Version**: 1.0  
**Owner**: RIF-Implementer  
**Approval**: PMI-CAL