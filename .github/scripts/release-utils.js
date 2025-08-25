/**
 * RIF Release Management Utilities
 * Supporting functions for the GitHub release automation workflow
 */

const fs = require('fs');
const path = require('path');

class ReleaseUtils {
  constructor(github, context) {
    this.github = github;
    this.context = context;
  }

  /**
   * Analyze commit messages to determine semantic version type
   * @param {string} fromTag - Starting tag for comparison
   * @param {string} toRef - Ending reference (usually HEAD)
   * @returns {Object} Analysis results
   */
  async analyzeCommitsForVersionType(fromTag, toRef = 'HEAD') {
    try {
      const commits = await this.github.rest.repos.compareCommits({
        owner: this.context.repo.owner,
        repo: this.context.repo.repo,
        base: fromTag,
        head: toRef
      });

      const commitMessages = commits.data.commits.map(commit => commit.commit.message);
      
      let versionType = 'patch';
      let features = [];
      let fixes = [];
      let breakingChanges = [];
      let improvements = [];

      for (const message of commitMessages) {
        const lowerMessage = message.toLowerCase();
        
        // Breaking changes (major version)
        if (this.isBreakingChange(message)) {
          breakingChanges.push(message);
          versionType = 'major';
        }
        // Features (minor version)
        else if (this.isFeature(message)) {
          features.push(message);
          if (versionType !== 'major') {
            versionType = 'minor';
          }
        }
        // Bug fixes (patch version)
        else if (this.isBugFix(message)) {
          fixes.push(message);
        }
        // Improvements/refactoring
        else if (this.isImprovement(message)) {
          improvements.push(message);
        }
      }

      return {
        versionType,
        totalCommits: commitMessages.length,
        features,
        fixes,
        breakingChanges,
        improvements,
        commitMessages
      };
    } catch (error) {
      console.error('Error analyzing commits:', error);
      return {
        versionType: 'patch',
        totalCommits: 0,
        features: [],
        fixes: [],
        breakingChanges: [],
        improvements: [],
        commitMessages: []
      };
    }
  }

  /**
   * Check if commit message indicates a breaking change
   */
  isBreakingChange(message) {
    const patterns = [
      /BREAKING[\s:]*/i,
      /breaking.*change/i,
      /💔/,
      /major[\s:]*/i,
      /!:/
    ];
    return patterns.some(pattern => pattern.test(message));
  }

  /**
   * Check if commit message indicates a new feature
   */
  isFeature(message) {
    const patterns = [
      /^feat[\s(:]*/i,
      /^feature[\s(:]*/i,
      /🚀/,
      /add.*feature/i,
      /implement.*feature/i,
      /new.*feature/i,
      /^add[\s:]*/i
    ];
    return patterns.some(pattern => pattern.test(message));
  }

  /**
   * Check if commit message indicates a bug fix
   */
  isBugFix(message) {
    const patterns = [
      /^fix[\s(:]*/i,
      /^bug[\s(:]*/i,
      /🐛/,
      /hotfix/i,
      /patch/i,
      /resolve.*bug/i,
      /fix.*issue/i,
      /correct.*error/i
    ];
    return patterns.some(pattern => pattern.test(message));
  }

  /**
   * Check if commit message indicates an improvement
   */
  isImprovement(message) {
    const patterns = [
      /^refactor[\s(:]*/i,
      /^perf[\s(:]*/i,
      /^improve[\s(:]*/i,
      /🔧/,
      /optimize/i,
      /enhance/i,
      /update/i,
      /upgrade/i
    ];
    return patterns.some(pattern => pattern.test(message));
  }

  /**
   * Generate categorized changelog from commit analysis
   * @param {Object} analysis - Results from analyzeCommitsForVersionType
   * @param {string} version - Release version
   * @param {string} previousTag - Previous release tag
   * @returns {string} Formatted changelog
   */
  generateChangelog(analysis, version, previousTag = null) {
    let changelog = `# Release Notes for ${version}\n\n`;
    
    if (previousTag) {
      changelog += `## Changes since ${previousTag}\n\n`;
      changelog += `**Total commits**: ${analysis.totalCommits}\n`;
      changelog += `**Version type**: ${analysis.versionType}\n\n`;
    }

    // Breaking Changes (highest priority)
    if (analysis.breakingChanges.length > 0) {
      changelog += `### 💔 Breaking Changes\n\n`;
      analysis.breakingChanges.forEach(change => {
        changelog += `- ${this.formatCommitMessage(change)}\n`;
      });
      changelog += '\n';
    }

    // Features
    if (analysis.features.length > 0) {
      changelog += `### 🚀 Features\n\n`;
      analysis.features.forEach(feature => {
        changelog += `- ${this.formatCommitMessage(feature)}\n`;
      });
      changelog += '\n';
    }

    // Bug Fixes
    if (analysis.fixes.length > 0) {
      changelog += `### 🐛 Bug Fixes\n\n`;
      analysis.fixes.forEach(fix => {
        changelog += `- ${this.formatCommitMessage(fix)}\n`;
      });
      changelog += '\n';
    }

    // Improvements
    if (analysis.improvements.length > 0) {
      changelog += `### 🔧 Improvements\n\n`;
      analysis.improvements.forEach(improvement => {
        changelog += `- ${this.formatCommitMessage(improvement)}\n`;
      });
      changelog += '\n';
    }

    // Add comparison link if previous tag exists
    if (previousTag) {
      changelog += `**Full Changelog**: https://github.com/${this.context.repo.owner}/${this.context.repo.repo}/compare/${previousTag}...${version}\n\n`;
    }

    // Installation instructions
    changelog += this.getInstallationInstructions(version);

    return changelog;
  }

  /**
   * Format commit message for changelog display
   */
  formatCommitMessage(message) {
    // Remove conventional commit prefixes for cleaner display
    return message
      .replace(/^(feat|fix|docs|style|refactor|perf|test|chore|build|ci)(\(.+\))?:\s*/i, '')
      .replace(/^(🚀|🐛|🔧|📚|🎨|♻️|⚡|🧪|👷|🔨)?\s*/, '')
      .trim();
  }

  /**
   * Get standardized installation instructions
   */
  getInstallationInstructions(version) {
    return `## Installation\n\n` +
           `### Quick Install\n` +
           `\`\`\`bash\n` +
           `git clone https://github.com/${this.context.repo.owner}/${this.context.repo.repo}.git\n` +
           `cd ${this.context.repo.repo}\n` +
           `git checkout ${version}\n` +
           `./rif-init.sh\n` +
           `\`\`\`\n\n` +
           `### Documentation\n` +
           `- 📖 [Setup Guide](README.md)\n` +
           `- 🔧 [Configuration](CLAUDE.md)\n` +
           `- 🚀 [Quick Start](rif-init.sh)\n` +
           `- 📋 [Workflows](.github/workflows/)\n\n`;
  }

  /**
   * Calculate next semantic version based on analysis
   * @param {string} currentVersion - Current version (e.g., "v1.2.3")
   * @param {string} versionType - Type of version bump ("major", "minor", "patch")
   * @returns {string} Next version
   */
  calculateNextVersion(currentVersion, versionType) {
    // Remove 'v' prefix if present
    const version = currentVersion.replace(/^v/, '');
    const parts = version.split('.').map(Number);
    
    // Ensure we have 3 parts (major.minor.patch)
    while (parts.length < 3) {
      parts.push(0);
    }

    let [major, minor, patch] = parts;

    switch (versionType) {
      case 'major':
        major += 1;
        minor = 0;
        patch = 0;
        break;
      case 'minor':
        minor += 1;
        patch = 0;
        break;
      case 'patch':
      default:
        patch += 1;
        break;
    }

    return `v${major}.${minor}.${patch}`;
  }

  /**
   * Validate version format
   * @param {string} version - Version to validate
   * @returns {boolean} Whether version is valid
   */
  isValidVersion(version) {
    const semverPattern = /^v?(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/;
    return semverPattern.test(version);
  }

  /**
   * Check if version is a prerelease
   * @param {string} version - Version to check
   * @returns {boolean} Whether version is prerelease
   */
  isPrerelease(version) {
    return /-(alpha|beta|rc|pre|preview|dev)/i.test(version);
  }

  /**
   * Find related issues for this release
   * @param {string} version - Release version
   * @returns {Array} Array of related issues
   */
  async findRelatedIssues(version) {
    try {
      const { data: issues } = await this.github.rest.issues.listForRepo({
        owner: this.context.repo.owner,
        repo: this.context.repo.repo,
        state: 'all',
        per_page: 100
      });

      const releaseIssues = issues.filter(issue => {
        const titleMatch = issue.title.toLowerCase().includes('release');
        const versionMatch = issue.title.includes(version.replace('v', ''));
        const labelMatch = issue.labels.some(label => 
          label.name.includes('release') || 
          label.name.includes('milestone')
        );
        
        return titleMatch || versionMatch || labelMatch;
      });

      return releaseIssues;
    } catch (error) {
      console.error('Error finding related issues:', error);
      return [];
    }
  }

  /**
   * Create comprehensive release assets
   * @param {string} version - Release version
   * @returns {Object} Asset creation results
   */
  async createReleaseAssets(version) {
    const assetsDir = 'release-assets';
    
    if (!fs.existsSync(assetsDir)) {
      fs.mkdirSync(assetsDir, { recursive: true });
    }

    const results = {
      created: [],
      errors: []
    };

    try {
      // 1. Source archives
      const sourceAssets = await this.createSourceArchives(version, assetsDir);
      results.created.push(...sourceAssets);

      // 2. Documentation bundle
      const docAssets = await this.createDocumentationBundle(version, assetsDir);
      results.created.push(...docAssets);

      // 3. Configuration templates
      const configAssets = await this.createConfigurationBundle(version, assetsDir);
      results.created.push(...configAssets);

      // 4. Generate checksums
      await this.generateChecksums(assetsDir);
      results.created.push('checksums.sha256', 'checksums.md5');

    } catch (error) {
      results.errors.push(error.message);
    }

    return results;
  }

  /**
   * Generate checksums for all assets
   */
  async generateChecksums(assetsDir) {
    const crypto = require('crypto');
    
    const files = fs.readdirSync(assetsDir)
      .filter(file => file.endsWith('.tar.gz') || file.endsWith('.zip'))
      .filter(file => fs.statSync(path.join(assetsDir, file)).isFile());

    let sha256Content = '';
    let md5Content = '';

    for (const file of files) {
      const filePath = path.join(assetsDir, file);
      const fileBuffer = fs.readFileSync(filePath);
      
      const sha256Hash = crypto.createHash('sha256').update(fileBuffer).digest('hex');
      const md5Hash = crypto.createHash('md5').update(fileBuffer).digest('hex');
      
      sha256Content += `${sha256Hash}  ${file}\n`;
      md5Content += `${md5Hash}  ${file}\n`;
    }

    fs.writeFileSync(path.join(assetsDir, 'checksums.sha256'), sha256Content);
    fs.writeFileSync(path.join(assetsDir, 'checksums.md5'), md5Content);
  }
}

module.exports = ReleaseUtils;