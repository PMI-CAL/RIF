#!/usr/bin/env node

/**
 * PR Complexity Calculator
 * Part of GitHub-First Automation System (Issue #283)
 *
 * Analyzes PR complexity and determines appropriate automation level:
 * - trivial: Fully automated GitHub processing
 * - simple: Copilot-assisted automation
 * - medium: RIF integration with standard validation
 * - complex: RIF integration with comprehensive validation
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Color codes for console output
const colors = {
  reset: '\033[0m',
  bright: '\033[1m',
  red: '\033[31m',
  green: '\033[32m',
  yellow: '\033[33m',
  blue: '\033[34m',
  magenta: '\033[35m',
  cyan: '\033[36m',
};

function log(message, color = 'reset') {
  const timestamp = new Date().toISOString();
  console.log(`${colors[color]}[${timestamp}] ${message}${colors.reset}`);
}

/**
 * Configuration for complexity assessment
 */
const COMPLEXITY_CONFIG = {
  thresholds: {
    trivial: {
      maxLines: 10,
      maxFiles: 1,
      requiresDependencyUpdate: true,
      allowedFileTypes: ['.json', '.md', '.txt', '.yml', '.yaml'],
    },
    simple: {
      maxLines: 50,
      maxFiles: 5,
      excludeArchitectureChanges: true,
      minTestCoverage: 80,
      allowedComplexity: 'low',
    },
    medium: {
      maxLines: 200,
      maxFiles: 10,
      allowArchitectureChanges: false,
      allowedComplexity: 'medium',
    },
    complex: {
      minLines: 200,
      significantArchitectureChanges: true,
      allowedComplexity: 'high',
    },
  },
  fileTypeWeights: {
    // Architecture and configuration files (high impact)
    '.yml': 3,
    '.yaml': 3,
    '.json': 2,
    '.config': 3,
    '.env': 3,
    Dockerfile: 4,
    'docker-compose.yml': 4,
    '.github/workflows': 4,

    // Source code files (medium impact)
    '.js': 2,
    '.jsx': 2,
    '.ts': 2,
    '.tsx': 2,
    '.py': 2,
    '.java': 2,
    '.cpp': 2,
    '.c': 2,
    '.go': 2,
    '.rs': 2,
    '.php': 2,
    '.rb': 2,

    // Database and migration files (high impact)
    '.sql': 3,
    '.migration': 4,

    // Documentation and tests (low impact)
    '.md': 1,
    '.txt': 1,
    '.test.js': 1,
    '.spec.js': 1,
    '.test.py': 1,
    '_test.go': 1,

    // Build and tooling (medium impact)
    'package.json': 2,
    'requirements.txt': 2,
    Makefile: 2,
    'setup.py': 2,
    'Cargo.toml': 2,
    'pom.xml': 2,
    'build.gradle': 2,
  },
  criticalPaths: [
    // System critical paths
    '/src/core/',
    '/lib/',
    '/api/',
    '/database/',
    '/migrations/',
    '/config/',
    '/security/',
    '/auth/',

    // Infrastructure paths
    '/.github/workflows/',
    '/docker/',
    '/kubernetes/',
    '/terraform/',
    '/ansible/',

    // Frontend critical paths
    '/src/components/core/',
    '/src/store/',
    '/src/services/',
    '/src/utils/',

    // Backend critical paths
    '/models/',
    '/controllers/',
    '/middleware/',
    '/routes/',
    '/services/',
  ],
};

/**
 * Analyzes PR data to determine complexity
 */
class PRComplexityCalculator {
  constructor(prData) {
    this.prData = prData;
    this.analysis = {
      linesChanged: 0,
      filesChanged: 0,
      fileTypes: new Set(),
      criticalPathsAffected: [],
      isDependencyUpdate: false,
      hasArchitectureChanges: false,
      hasSecurityImpact: false,
      hasPerformanceImpact: false,
      weightedComplexityScore: 0,
      riskFactors: [],
    };
  }

  /**
   * Main analysis method
   */
  async analyze() {
    log('🔍 Starting PR complexity analysis...', 'blue');

    try {
      await this.gatherBasicMetrics();
      await this.analyzeFileTypes();
      await this.assessCriticalPathImpact();
      await this.detectSpecialChangeTypes();
      await this.calculateWeightedScore();
      await this.identifyRiskFactors();

      const complexity = this.determineComplexity();
      const automationLevel = this.determineAutomationLevel(complexity);

      const result = {
        complexity,
        automationLevel,
        analysis: this.analysis,
        recommendations: this.generateRecommendations(),
        timestamp: new Date().toISOString(),
      };

      log(
        `✅ Analysis complete - Complexity: ${complexity}, Automation: ${automationLevel}`,
        'green'
      );
      return result;
    } catch (error) {
      log(`❌ Analysis failed: ${error.message}`, 'red');
      throw error;
    }
  }

  /**
   * Gather basic PR metrics
   */
  async gatherBasicMetrics() {
    log('📊 Gathering basic metrics...', 'cyan');

    if (this.prData.additions && this.prData.deletions) {
      this.analysis.linesChanged = this.prData.additions + this.prData.deletions;
    } else {
      // Fallback to git analysis
      try {
        const gitStats = execSync('git diff --stat HEAD~1 HEAD', {
          encoding: 'utf8',
        });
        const matches = gitStats.match(
          /(\d+) files? changed(?:, (\d+) insertions?)?(?:, (\d+) deletions?)?/
        );
        if (matches) {
          this.analysis.filesChanged = parseInt(matches[1]);
          const insertions = parseInt(matches[2]) || 0;
          const deletions = parseInt(matches[3]) || 0;
          this.analysis.linesChanged = insertions + deletions;
        }
      } catch (error) {
        log('⚠️ Could not determine git stats, using PR data', 'yellow');
        this.analysis.linesChanged = 100; // Conservative estimate
      }
    }

    this.analysis.filesChanged = this.prData.changed_files || this.analysis.filesChanged;

    log(
      `📈 Lines changed: ${this.analysis.linesChanged}, Files changed: ${this.analysis.filesChanged}`,
      'cyan'
    );
  }

  /**
   * Analyze file types and their impact
   */
  async analyzeFileTypes() {
    log('🗂️ Analyzing file types...', 'cyan');

    try {
      const changedFiles = execSync('git diff --name-only HEAD~1 HEAD', {
        encoding: 'utf8',
      })
        .split('\n')
        .filter((file) => file.trim());

      for (const file of changedFiles) {
        const ext = path.extname(file);
        const basename = path.basename(file);
        this.analysis.fileTypes.add(ext || basename);

        // Check for architecture files
        if (this.isArchitectureFile(file)) {
          this.analysis.hasArchitectureChanges = true;
        }

        // Check for security-related files
        if (this.isSecurityRelated(file)) {
          this.analysis.hasSecurityImpact = true;
        }

        // Check for performance-critical files
        if (this.isPerformanceCritical(file)) {
          this.analysis.hasPerformanceImpact = true;
        }
      }

      log(`📁 File types detected: ${Array.from(this.analysis.fileTypes).join(', ')}`, 'cyan');
    } catch (error) {
      log('⚠️ Could not analyze changed files', 'yellow');
    }
  }

  /**
   * Assess impact on critical system paths
   */
  async assessCriticalPathImpact() {
    log('🎯 Assessing critical path impact...', 'cyan');

    try {
      const changedFiles = execSync('git diff --name-only HEAD~1 HEAD', {
        encoding: 'utf8',
      })
        .split('\n')
        .filter((file) => file.trim());

      for (const file of changedFiles) {
        for (const criticalPath of COMPLEXITY_CONFIG.criticalPaths) {
          if (file.startsWith(criticalPath.substring(1))) {
            // Remove leading slash
            this.analysis.criticalPathsAffected.push({
              path: criticalPath,
              file: file,
            });
          }
        }
      }

      if (this.analysis.criticalPathsAffected.length > 0) {
        log(`⚠️ Critical paths affected: ${this.analysis.criticalPathsAffected.length}`, 'yellow');
      }
    } catch (error) {
      log('⚠️ Could not assess critical path impact', 'yellow');
    }
  }

  /**
   * Detect special change types (dependency updates, etc.)
   */
  async detectSpecialChangeTypes() {
    log('🔍 Detecting special change types...', 'cyan');

    const title = this.prData.title || '';
    const body = this.prData.body || '';

    // Detect dependency updates
    const depUpdatePatterns = [
      /^(bump|update|upgrade|deps|dependencies)/i,
      /dependabot/i,
      /renovate/i,
      /package.*update/i,
      /security.*update/i,
    ];

    this.analysis.isDependencyUpdate = depUpdatePatterns.some(
      (pattern) => pattern.test(title) || pattern.test(body)
    );

    if (this.analysis.isDependencyUpdate) {
      log('📦 Detected dependency update', 'cyan');
    }
  }

  /**
   * Calculate weighted complexity score
   */
  async calculateWeightedScore() {
    log('⚖️ Calculating weighted complexity score...', 'cyan');

    let score = 0;

    // Base score from lines and files
    score += Math.min(this.analysis.linesChanged * 0.1, 50);
    score += Math.min(this.analysis.filesChanged * 2, 30);

    // File type weights
    for (const fileType of this.analysis.fileTypes) {
      const weight = COMPLEXITY_CONFIG.fileTypeWeights[fileType] || 1;
      score += weight * 2;
    }

    // Critical path impact
    score += this.analysis.criticalPathsAffected.length * 10;

    // Special change type adjustments
    if (this.analysis.isDependencyUpdate && this.analysis.linesChanged < 20) {
      score *= 0.3; // Reduce score for small dependency updates
    }

    if (this.analysis.hasArchitectureChanges) {
      score += 25;
    }

    if (this.analysis.hasSecurityImpact) {
      score += 20;
    }

    if (this.analysis.hasPerformanceImpact) {
      score += 15;
    }

    this.analysis.weightedComplexityScore = Math.round(score);

    log(`📊 Weighted complexity score: ${this.analysis.weightedComplexityScore}`, 'cyan');
  }

  /**
   * Identify risk factors
   */
  async identifyRiskFactors() {
    log('⚠️ Identifying risk factors...', 'yellow');

    const risks = [];

    if (this.analysis.linesChanged > 500) {
      risks.push('Large number of lines changed (>500)');
    }

    if (this.analysis.filesChanged > 20) {
      risks.push('Many files changed (>20)');
    }

    if (this.analysis.criticalPathsAffected.length > 0) {
      risks.push(`Critical system paths affected (${this.analysis.criticalPathsAffected.length})`);
    }

    if (this.analysis.hasSecurityImpact) {
      risks.push('Security-related changes detected');
    }

    if (this.analysis.hasArchitectureChanges) {
      risks.push('Architecture changes detected');
    }

    if (this.analysis.fileTypes.has('.sql') || this.analysis.fileTypes.has('.migration')) {
      risks.push('Database changes detected');
    }

    this.analysis.riskFactors = risks;

    if (risks.length > 0) {
      log(`⚠️ Risk factors identified: ${risks.length}`, 'yellow');
      risks.forEach((risk) => log(`   - ${risk}`, 'yellow'));
    }
  }

  /**
   * Determine overall complexity level
   */
  determineComplexity() {
    log('🎯 Determining complexity level...', 'magenta');

    const config = COMPLEXITY_CONFIG.thresholds;

    // Check for trivial
    if (
      this.analysis.isDependencyUpdate &&
      this.analysis.linesChanged <= config.trivial.maxLines &&
      this.analysis.filesChanged <= config.trivial.maxFiles &&
      !this.analysis.hasArchitectureChanges &&
      !this.analysis.hasSecurityImpact
    ) {
      return 'trivial';
    }

    // Check for simple
    if (
      this.analysis.linesChanged <= config.simple.maxLines &&
      this.analysis.filesChanged <= config.simple.maxFiles &&
      !this.analysis.hasArchitectureChanges &&
      this.analysis.criticalPathsAffected.length === 0 &&
      !this.analysis.hasSecurityImpact
    ) {
      return 'simple';
    }

    // Check for medium
    if (
      this.analysis.linesChanged <= config.medium.maxLines &&
      this.analysis.filesChanged <= config.medium.maxFiles &&
      this.analysis.criticalPathsAffected.length <= 2 &&
      this.analysis.weightedComplexityScore < 80
    ) {
      return 'medium';
    }

    // Everything else is complex
    return 'complex';
  }

  /**
   * Determine appropriate automation level
   */
  determineAutomationLevel(complexity) {
    log('🤖 Determining automation level...', 'magenta');

    switch (complexity) {
      case 'trivial':
        return 'github_only';
      case 'simple':
        return 'copilot_assisted';
      case 'medium':
      case 'complex':
      default:
        return 'rif_integration';
    }
  }

  /**
   * Generate recommendations
   */
  generateRecommendations() {
    const recommendations = [];

    if (this.analysis.riskFactors.length > 0) {
      recommendations.push('Consider breaking this PR into smaller, focused changes');
    }

    if (this.analysis.hasArchitectureChanges) {
      recommendations.push('Ensure architecture review is completed before merge');
    }

    if (this.analysis.hasSecurityImpact) {
      recommendations.push('Security review required before merge');
    }

    if (this.analysis.criticalPathsAffected.length > 0) {
      recommendations.push('Extra testing recommended due to critical path changes');
    }

    if (this.analysis.linesChanged > 200 && !this.analysis.isDependencyUpdate) {
      recommendations.push('Consider adding detailed testing plan');
    }

    return recommendations;
  }

  // Helper methods
  isArchitectureFile(file) {
    const architecturePatterns = [
      /\.config$/,
      /\.env$/,
      /^config\//,
      /^architecture\//,
      /docker/i,
      /kubernetes/i,
      /terraform/i,
      /^\.github\/workflows\//,
    ];

    return architecturePatterns.some((pattern) => pattern.test(file));
  }

  isSecurityRelated(file) {
    const securityPatterns = [
      /security/i,
      /auth/i,
      /password/i,
      /token/i,
      /secret/i,
      /certificate/i,
      /\.key$/,
      /\.pem$/,
      /\.crt$/,
    ];

    return securityPatterns.some((pattern) => pattern.test(file));
  }

  isPerformanceCritical(file) {
    const performancePatterns = [
      /database/i,
      /query/i,
      /cache/i,
      /performance/i,
      /optimization/i,
      /\.sql$/,
      /migration/i,
    ];

    return performancePatterns.some((pattern) => pattern.test(file));
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    log('🚀 PR Complexity Calculator starting...', 'bright');

    // Get PR data from environment variables (GitHub Actions context)
    const prData = {
      title: process.env.PR_TITLE || process.env.GITHUB_EVENT_PULL_REQUEST_TITLE,
      body: process.env.PR_BODY || process.env.GITHUB_EVENT_PULL_REQUEST_BODY,
      additions:
        parseInt(process.env.PR_ADDITIONS || process.env.GITHUB_EVENT_PULL_REQUEST_ADDITIONS) || 0,
      deletions:
        parseInt(process.env.PR_DELETIONS || process.env.GITHUB_EVENT_PULL_REQUEST_DELETIONS) || 0,
      changed_files:
        parseInt(
          process.env.PR_CHANGED_FILES || process.env.GITHUB_EVENT_PULL_REQUEST_CHANGED_FILES
        ) || 0,
    };

    log('📋 PR Data received:', 'blue');
    log(`   Title: ${prData.title || 'N/A'}`, 'blue');
    log(`   Additions: ${prData.additions}`, 'blue');
    log(`   Deletions: ${prData.deletions}`, 'blue');
    log(`   Changed Files: ${prData.changed_files}`, 'blue');

    const calculator = new PRComplexityCalculator(prData);
    const result = await calculator.analyze();

    // Output results
    console.log('\n' + '='.repeat(60));
    console.log('🎯 PR COMPLEXITY ANALYSIS RESULTS');
    console.log('='.repeat(60));
    console.log(`Complexity Level: ${result.complexity.toUpperCase()}`);
    console.log(`Automation Level: ${result.automationLevel}`);
    console.log(`Weighted Score: ${result.analysis.weightedComplexityScore}`);
    console.log(`Lines Changed: ${result.analysis.linesChanged}`);
    console.log(`Files Changed: ${result.analysis.filesChanged}`);
    console.log(`Critical Paths Affected: ${result.analysis.criticalPathsAffected.length}`);
    console.log(`Risk Factors: ${result.analysis.riskFactors.length}`);

    if (result.recommendations.length > 0) {
      console.log('\n📋 RECOMMENDATIONS:');
      result.recommendations.forEach((rec, i) => {
        console.log(`${i + 1}. ${rec}`);
      });
    }

    // Save detailed results to file
    const outputFile = 'pr-complexity-analysis.json';
    fs.writeFileSync(outputFile, JSON.stringify(result, null, 2));
    log(`💾 Detailed results saved to ${outputFile}`, 'green');

    // Set GitHub Actions outputs
    if (process.env.GITHUB_OUTPUT) {
      const outputs = [
        `complexity=${result.complexity}`,
        `automation_level=${result.automationLevel}`,
        `weighted_score=${result.analysis.weightedComplexityScore}`,
        `lines_changed=${result.analysis.linesChanged}`,
        `files_changed=${result.analysis.filesChanged}`,
        `risk_factors=${result.analysis.riskFactors.length}`,
        `has_architecture_changes=${result.analysis.hasArchitectureChanges}`,
        `has_security_impact=${result.analysis.hasSecurityImpact}`,
        `is_dependency_update=${result.analysis.isDependencyUpdate}`,
      ];

      fs.appendFileSync(process.env.GITHUB_OUTPUT, outputs.join('\n') + '\n');
      log('✅ GitHub Actions outputs set', 'green');
    }

    log('🎉 Analysis completed successfully!', 'bright');
  } catch (error) {
    log(`❌ Fatal error: ${error.message}`, 'red');
    console.error(error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { PRComplexityCalculator, COMPLEXITY_CONFIG };
