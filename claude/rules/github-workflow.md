# GitHub Workflow and Issue Integration Rules

## Critical Rules for GitHub Integration

**IMPORTANT**: These rules MUST be followed for all GitHub-related work.

### When Asked About Next Steps or Tasks
1. **ALWAYS** run `gh issue list --state open` first
2. **NEVER** create generic todos without checking existing issues
3. **ALWAYS** reference specific issue numbers in TodoWrite items

### TodoWrite and GitHub Issue Integration

**When using TodoWrite tool, you must:**
- Reference the GitHub issue number in every todo item
- Follow the format: "[Issue #XX] Task description"
- Check existing GitHub issues before creating any todos
- Ensure todos align with issue acceptance criteria

**Critical Rules:**
- Never close issues until ALL acceptance criteria are met
- Always test fixes thoroughly before marking complete
- Document complete solutions, not just progress
- Follow proper development cycle for all issues

### GitHub Issue Workflow Enforcement

**Before closing any GitHub issue, you MUST:**
1. Complete ALL acceptance criteria listed in the issue
2. Test the fix thoroughly to ensure it works
3. Verify no related problems remain
4. Document the complete solution (not just progress)
5. Confirm the original problem is fully resolved

**Issue Status Rules:**
- Only mark "in progress" when actively working
- Only close when problem is completely solved
- Reopen if issues persist after "fixing"

### Proper Development Workflow

1. **Start Work**: Comment on the issue that work is beginning
2. **Work**: Implement the actual solution 
3. **Test**: Verify the fix works completely
4. **Verify**: Confirm all acceptance criteria are met
5. **Document**: Provide complete solution documentation
6. **Close**: Only when everything above is complete

### GitHub Commands to Use

```bash
# Always check issues first
gh issue list --state open
gh issue view <number>

# Work with issues
gh issue comment <number> --body "Starting work on this issue"
gh issue edit <number> --add-label "in-progress"

# Create PRs linked to issues
gh pr create --title "Fix: [Issue #XX] Description" --body "Closes #XX"
```

### Issue Label Context Loading

When working on issues with these labels, include specific context:
- **development**: Development best practices, coding standards, Git workflow
- **code-quality**: Sandi Metz principles, refactoring patterns, quality metrics
- **architecture**: Design patterns, architectural decision records, system design
- **testing**: Testing strategies, framework documentation, quality assurance
- **security**: Security guidelines, vulnerability assessment, compliance requirements
- **performance**: Performance optimization, profiling techniques, scalability patterns

### Agent Workflow Triggers

The framework includes specialized agents that are automatically triggered by GitHub issue labels:

#### Workflow State Labels
- **workflow-state:planning** → Project Manager Agent
- **workflow-state:implementing** → Developer Agent  
- **workflow-state:testing** → Quality Assurance Agent
- **workflow-state:reviewing** → System Architect Agent
- **workflow-state:documenting** → Business Analyst Agent
- **workflow-state:context-discovery** → Context Server Discovery Agent

#### Agent Assignment Labels
- **workflow-agent:project-manager** → Assign to Project Manager Agent
- **workflow-agent:architect** → Assign to System Architect Agent
- **workflow-agent:developer** → Assign to Developer Agent
- **workflow-agent:quality-assurance** → Assign to Quality Assurance Agent
- **workflow-agent:business-analyst** → Assign to Business Analyst Agent
- **workflow-agent:scrum-master** → Assign to Scrum Master Agent

#### Context Server Discovery Integration
The Context Server Discovery Agent automatically activates when:
- Issue receives label: `workflow-state:context-discovery`
- New project is initialized with the framework
- Configuration files are updated indicating new technology stack
- Comment contains: "**Request**: Context server discovery"

### Common Mistakes to Avoid

1. **Creating todos without checking GitHub issues**
2. **Closing issues prematurely**
3. **Not referencing issue numbers in commits/PRs**
4. **Skipping the test/verify steps**
5. **Marking issues complete with only partial fixes**

### Integration with Project Workflow

- All work should be traceable to a GitHub issue
- Use issue numbers in branch names: `feature/issue-{{ISSUE_NUMBER}}-{{FEATURE_DESCRIPTION}}`
- Reference issues in all commits: `git commit -m "fix(#{{ISSUE_NUMBER}}): {{COMMIT_DESCRIPTION}}"`
- Link PRs to issues for automatic closure

Remember: GitHub issues are the source of truth for project work. TodoWrite is a tool to help track progress on those issues, not a replacement for them.