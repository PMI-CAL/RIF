#!/usr/bin/env python3
"""
Pre-Flight Blocking Validator - Issue #228 Implementation
RIF-Implementer: Critical orchestration failure prevention

This module provides standalone pre-flight validation for blocking issues
that can be used before any orchestration decisions are made.

ISSUE #228: Prevents the critical orchestration failure where Issue #225
declared "THIS ISSUE BLOCKS ALL OTHERS" but was ignored by the orchestrator.

Usage:
    python pre_flight_blocking_validator.py --issues 225,226,227
    python pre_flight_blocking_validator.py --validate-single 225
"""

import json
import subprocess
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreFlightBlockingValidator:
    """
    Standalone validator for blocking issue declarations.
    Prevents orchestration failures by detecting blocking statements early.
    """
    
    def __init__(self):
        self.blocking_phrases = [
            "this issue blocks all others",
            "this issue blocks all other work", 
            "blocks all other work",
            "blocks all others",
            "stop all work",
            "must complete before all",
            "must complete before all other work",
            "must complete before all others"
        ]
    
    def validate_issues_for_blocking(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Validate multiple issues for blocking declarations.
        
        Args:
            issue_numbers: List of GitHub issue numbers to validate
            
        Returns:
            Dict with comprehensive validation results
        """
        logger.info(f"Performing pre-flight blocking validation for issues: {issue_numbers}")
        
        validation_result = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_issues_checked': len(issue_numbers),
            'blocking_issues': [],
            'non_blocking_issues': [],
            'validation_errors': [],
            'has_blocking_issues': False,
            'enforcement_recommendation': 'ALLOW_ORCHESTRATION',
            'detailed_analysis': {}
        }
        
        for issue_num in issue_numbers:
            try:
                issue_result = self._validate_single_issue(issue_num)
                validation_result['detailed_analysis'][str(issue_num)] = issue_result
                
                if issue_result['is_blocking']:
                    validation_result['blocking_issues'].append(issue_num)
                    logger.warning(f"BLOCKING ISSUE DETECTED: #{issue_num}")
                else:
                    validation_result['non_blocking_issues'].append(issue_num)
                    
            except Exception as e:
                error_msg = f"Error validating issue {issue_num}: {e}"
                logger.error(error_msg)
                validation_result['validation_errors'].append({
                    'issue': issue_num,
                    'error': str(e)
                })
                # Assume non-blocking on error to avoid false positives
                validation_result['non_blocking_issues'].append(issue_num)
        
        # Set final validation status
        validation_result['has_blocking_issues'] = len(validation_result['blocking_issues']) > 0
        
        if validation_result['has_blocking_issues']:
            validation_result['enforcement_recommendation'] = 'HALT_ALL_ORCHESTRATION'
            logger.critical(f"ORCHESTRATION SHOULD BE HALTED - {len(validation_result['blocking_issues'])} blocking issues detected")
        else:
            validation_result['enforcement_recommendation'] = 'ALLOW_ORCHESTRATION'
            logger.info("Pre-flight validation passed - no blocking issues detected")
        
        return validation_result
    
    def _validate_single_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Validate a single issue for blocking declarations.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Dict with single issue validation results
        """
        # Get issue data with comments
        issue_data = self._fetch_issue_data(issue_number)
        
        if not issue_data:
            return {
                'is_blocking': False,
                'validation_error': 'Could not fetch issue data',
                'blocking_sources': [],
                'confidence': 0.0
            }
        
        blocking_sources = []
        
        # Check issue body
        body_text = (issue_data.get('body') or '').lower()
        body_blocking_phrases = [phrase for phrase in self.blocking_phrases if phrase in body_text]
        
        if body_blocking_phrases:
            blocking_sources.append({
                'source_type': 'issue_body',
                'detected_phrases': body_blocking_phrases,
                'evidence': self._extract_evidence(body_text, body_blocking_phrases[0]),
                'confidence': 1.0  # High confidence for body declarations
            })
        
        # Check comments
        comments = issue_data.get('comments', [])
        for i, comment in enumerate(comments):
            comment_body = (comment.get('body') or '').lower()
            comment_blocking_phrases = [phrase for phrase in self.blocking_phrases if phrase in comment_body]
            
            if comment_blocking_phrases:
                author = comment.get('author', {}).get('login', 'unknown')
                blocking_sources.append({
                    'source_type': 'comment',
                    'comment_index': i,
                    'author': author,
                    'detected_phrases': comment_blocking_phrases,
                    'evidence': self._extract_evidence(comment_body, comment_blocking_phrases[0]),
                    'confidence': 0.9  # Slightly lower confidence for comments
                })
        
        is_blocking = len(blocking_sources) > 0
        overall_confidence = max([source['confidence'] for source in blocking_sources], default=0.0)
        
        return {
            'is_blocking': is_blocking,
            'blocking_sources': blocking_sources,
            'blocking_phrase_count': sum(len(source['detected_phrases']) for source in blocking_sources),
            'confidence': overall_confidence,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _fetch_issue_data(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Fetch issue data including comments from GitHub"""
        try:
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number),
                '--json', 'number,title,body,state,labels,comments'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"Failed to fetch issue {issue_number}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout fetching issue {issue_number}")
            return None
        except Exception as e:
            logger.error(f"Error fetching issue {issue_number}: {e}")
            return None
    
    def _extract_evidence(self, text: str, phrase: str, context_chars: int = 100) -> str:
        """Extract context around blocking phrase for evidence"""
        phrase_pos = text.find(phrase)
        if phrase_pos == -1:
            return phrase
            
        start = max(0, phrase_pos - context_chars // 2)
        end = min(len(text), phrase_pos + len(phrase) + context_chars // 2)
        
        context = text[start:end].strip()
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def generate_validation_report(self, validation_result: Dict[str, Any]) -> str:
        """Generate a formatted validation report"""
        report = []
        report.append("=" * 80)
        report.append("PRE-FLIGHT BLOCKING VALIDATION REPORT - Issue #228")
        report.append("=" * 80)
        report.append(f"Validation Time: {validation_result['validation_timestamp']}")
        report.append(f"Issues Checked: {validation_result['total_issues_checked']}")
        report.append(f"Blocking Issues: {len(validation_result['blocking_issues'])}")
        report.append(f"Non-Blocking Issues: {len(validation_result['non_blocking_issues'])}")
        report.append(f"Validation Errors: {len(validation_result['validation_errors'])}")
        report.append("")
        
        # Enforcement recommendation
        recommendation = validation_result['enforcement_recommendation']
        if recommendation == 'HALT_ALL_ORCHESTRATION':
            report.append("🚨 ENFORCEMENT RECOMMENDATION: HALT ALL ORCHESTRATION")
            report.append("   Blocking issues detected - complete them before proceeding")
        else:
            report.append("✅ ENFORCEMENT RECOMMENDATION: ALLOW ORCHESTRATION")
            report.append("   No blocking issues detected - proceed with normal orchestration")
        
        report.append("")
        
        # Detailed analysis
        if validation_result['blocking_issues']:
            report.append("🚫 BLOCKING ISSUES DETECTED:")
            for issue_num in validation_result['blocking_issues']:
                analysis = validation_result['detailed_analysis'][str(issue_num)]
                report.append(f"   Issue #{issue_num}:")
                report.append(f"     Confidence: {analysis['confidence']:.1f}")
                report.append(f"     Blocking Sources: {len(analysis['blocking_sources'])}")
                
                for source in analysis['blocking_sources']:
                    report.append(f"       - {source['source_type']}: {source['detected_phrases']}")
                    if source.get('author'):
                        report.append(f"         Author: {source['author']}")
                    report.append(f"         Evidence: {source['evidence'][:100]}...")
                report.append("")
        
        if validation_result['non_blocking_issues']:
            report.append("✅ NON-BLOCKING ISSUES:")
            report.append(f"   Issues: {validation_result['non_blocking_issues']}")
            report.append("")
        
        if validation_result['validation_errors']:
            report.append("⚠️ VALIDATION ERRORS:")
            for error in validation_result['validation_errors']:
                report.append(f"   Issue #{error['issue']}: {error['error']}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Command-line interface for pre-flight blocking validation"""
    parser = argparse.ArgumentParser(
        description="Pre-Flight Blocking Validator for Issue #228",
        epilog="Examples:\n"
               "  python pre_flight_blocking_validator.py --issues 225,226,227\n"
               "  python pre_flight_blocking_validator.py --validate-single 225\n"
               "  python pre_flight_blocking_validator.py --issues 225,226,227 --output report.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--issues', type=str,
                       help='Comma-separated list of issue numbers to validate')
    parser.add_argument('--validate-single', type=int,
                       help='Validate a single issue number')
    parser.add_argument('--output', type=str,
                       help='Output file for detailed JSON results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output except for final recommendation')
    
    args = parser.parse_args()
    
    if not args.issues and not args.validate_single:
        parser.error("Must specify either --issues or --validate-single")
    
    # Configure logging based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    validator = PreFlightBlockingValidator()
    
    # Determine issue numbers to validate
    if args.validate_single:
        issue_numbers = [args.validate_single]
    else:
        issue_numbers = [int(num.strip()) for num in args.issues.split(',')]
    
    # Perform validation
    validation_result = validator.validate_issues_for_blocking(issue_numbers)
    
    # Generate and display report
    if not args.quiet:
        report = validator.generate_validation_report(validation_result)
        print(report)
    
    # Output final recommendation
    recommendation = validation_result['enforcement_recommendation']
    if recommendation == 'HALT_ALL_ORCHESTRATION':
        print(f"\n🚨 RESULT: HALT ORCHESTRATION - {len(validation_result['blocking_issues'])} blocking issues detected")
        exit_code = 1
    else:
        print(f"\n✅ RESULT: ALLOW ORCHESTRATION - No blocking issues detected")
        exit_code = 0
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(validation_result, f, indent=2)
        print(f"\n💾 Detailed results saved to: {args.output}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()