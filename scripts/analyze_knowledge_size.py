#!/usr/bin/env python3
"""
Knowledge Base Size Analyzer

This script analyzes the RIF knowledge base to provide detailed size breakdown,
helping with deployment planning and cleanup decisions.

Usage:
    python scripts/analyze_knowledge_size.py [--detailed] [--export-csv]
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeSizeAnalyzer:
    """Analyzes knowledge base size and composition"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
    
    def analyze_full_breakdown(self) -> Dict:
        """Perform comprehensive size analysis"""
        logger.info("Performing comprehensive knowledge base analysis...")
        
        analysis = {
            'overview': {
                'total_size_mb': 0,
                'total_files': 0,
                'total_directories': 0,
                'analysis_date': Path(__file__).stat().st_mtime
            },
            'directories': {},
            'large_files': [],
            'file_types': {},
            'recommendations': []
        }
        
        # Analyze each directory
        for item in self.knowledge_dir.iterdir():
            if item.is_dir():
                dir_analysis = self._analyze_directory(item)
                analysis['directories'][item.name] = dir_analysis
                analysis['overview']['total_size_mb'] += dir_analysis['size_mb']
                analysis['overview']['total_files'] += dir_analysis['file_count']
                analysis['overview']['total_directories'] += 1
            elif item.is_file():
                file_size = item.stat().st_size
                analysis['overview']['total_size_mb'] += file_size / (1024 * 1024)
                analysis['overview']['total_files'] += 1
                
                # Track large individual files
                if file_size > 1024 * 1024:  # > 1MB
                    analysis['large_files'].append({
                        'name': item.name,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'type': item.suffix or 'no-extension'
                    })
                
                # Track file types
                file_type = item.suffix or 'no-extension'
                if file_type not in analysis['file_types']:
                    analysis['file_types'][file_type] = {'count': 0, 'size_mb': 0}
                analysis['file_types'][file_type]['count'] += 1
                analysis['file_types'][file_type]['size_mb'] += file_size / (1024 * 1024)
        
        # Round total size
        analysis['overview']['total_size_mb'] = round(analysis['overview']['total_size_mb'], 2)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_directory(self, dir_path: Path) -> Dict:
        """Analyze a single directory"""
        dir_info = {
            'size_mb': 0,
            'file_count': 0,
            'subdirectories': [],
            'largest_files': [],
            'file_types': {},
            'deployment_action': self._get_deployment_action(dir_path.name),
            'notes': self._get_directory_notes(dir_path.name)
        }
        
        try:
            # Get all files in directory
            all_files = [f for f in dir_path.rglob('*') if f.is_file()]
            dir_info['file_count'] = len(all_files)
            
            file_sizes = []
            for file_path in all_files:
                file_size = file_path.stat().st_size
                dir_info['size_mb'] += file_size / (1024 * 1024)
                
                # Track largest files in directory
                file_sizes.append((file_path.name, file_size))
                
                # Track file types
                file_type = file_path.suffix or 'no-extension'
                if file_type not in dir_info['file_types']:
                    dir_info['file_types'][file_type] = {'count': 0, 'size_mb': 0}
                dir_info['file_types'][file_type]['count'] += 1
                dir_info['file_types'][file_type]['size_mb'] += file_size / (1024 * 1024)
            
            # Get top 5 largest files
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            dir_info['largest_files'] = [
                {'name': name, 'size_mb': round(size / (1024 * 1024), 2)}
                for name, size in file_sizes[:5]
            ]
            
            # Get subdirectories
            dir_info['subdirectories'] = [d.name for d in dir_path.iterdir() if d.is_dir()]
            
        except PermissionError:
            logger.warning(f"Permission denied accessing {dir_path}")
        
        dir_info['size_mb'] = round(dir_info['size_mb'], 2)
        return dir_info
    
    def _get_deployment_action(self, dirname: str) -> str:
        """Get recommended deployment action for directory"""
        remove_dirs = {
            'audits', 'checkpoints', 'issues', 'metrics', 'enforcement_logs',
            'evidence_collection', 'migrated_backup', 'coordination',
            'false_positive_detection', 'demo_monitoring', 'state_cleanup',
            'state_transitions', 'recovery', 'reports'
        }
        
        filter_dirs = {
            'patterns', 'decisions', 'learning', 'analysis', 'validation'
        }
        
        if dirname in remove_dirs:
            return "REMOVE"
        elif dirname in filter_dirs:
            return "FILTER"
        elif dirname == 'chromadb':
            return "RESET"
        else:
            return "KEEP"
    
    def _get_directory_notes(self, dirname: str) -> str:
        """Get notes about directory purpose and cleanup strategy"""
        notes_map = {
            'audits': 'RIF development audit logs - safe to remove for deployment',
            'checkpoints': 'Issue-specific checkpoints - safe to remove for deployment',
            'issues': 'RIF-specific issue resolutions - safe to remove for deployment', 
            'metrics': 'RIF development metrics - safe to remove for deployment',
            'chromadb': 'Vector database - reset for new project',
            'patterns': 'Mix of reusable and RIF-specific patterns - filter needed',
            'decisions': 'Mix of framework and issue-specific decisions - filter needed',
            'learning': 'Mix of methodology and RIF-specific learnings - filter needed',
            'conversations': 'Framework component - keep as-is',
            'context': 'Framework component - keep as-is',
            'embeddings': 'Framework component - keep as-is',
            'parsing': 'Framework component - keep as-is'
        }
        return notes_map.get(dirname, 'Review required for deployment action')
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate cleanup and deployment recommendations"""
        recommendations = []
        
        # Size-based recommendations
        total_size = analysis['overview']['total_size_mb']
        if total_size > 50:
            recommendations.append(f"Knowledge base is {total_size:.1f}MB - cleanup recommended for deployment")
        
        # Directory-specific recommendations
        for dirname, info in analysis['directories'].items():
            if info['size_mb'] > 10 and info['deployment_action'] == 'REMOVE':
                recommendations.append(f"Remove {dirname}/ ({info['size_mb']:.1f}MB) - development artifacts")
            elif info['size_mb'] > 5 and info['deployment_action'] == 'FILTER':
                recommendations.append(f"Filter {dirname}/ ({info['size_mb']:.1f}MB) - contains mixed content")
        
        # File type recommendations
        for file_type, info in analysis['file_types'].items():
            if file_type in ['.log', '.tmp', '.cache'] and info['size_mb'] > 1:
                recommendations.append(f"Clean {file_type} files ({info['size_mb']:.1f}MB) - temporary files")
        
        # Large file recommendations
        if analysis['large_files']:
            recommendations.append(f"Review {len(analysis['large_files'])} large files for optimization")
        
        # Final size estimate
        removable_size = sum(
            info['size_mb'] for dirname, info in analysis['directories'].items()
            if info['deployment_action'] == 'REMOVE'
        )
        
        if removable_size > 0:
            final_size = total_size - removable_size
            recommendations.append(f"After cleanup: ~{final_size:.1f}MB (saved {removable_size:.1f}MB)")
        
        return recommendations
    
    def generate_report(self, analysis: Dict, detailed: bool = False) -> str:
        """Generate analysis report"""
        report = f"""# Knowledge Base Size Analysis Report

## Overview
- **Total Size**: {analysis['overview']['total_size_mb']:.2f}MB
- **Total Files**: {analysis['overview']['total_files']:,}
- **Total Directories**: {analysis['overview']['total_directories']}

## Directory Breakdown (by size)
"""
        
        # Sort directories by size
        sorted_dirs = sorted(
            analysis['directories'].items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )
        
        for dirname, info in sorted_dirs:
            action_icon = {
                'REMOVE': '🗑️',
                'FILTER': '🔍',
                'RESET': '🔄',
                'KEEP': '✅'
            }.get(info['deployment_action'], '❓')
            
            report += f"\n### {action_icon} {dirname}/ - {info['size_mb']:.2f}MB ({info['file_count']} files)\n"
            report += f"**Action**: {info['deployment_action']}\n"
            if info['notes']:
                report += f"**Notes**: {info['notes']}\n"
            
            if detailed and info['largest_files']:
                report += "**Largest files**:\n"
                for file_info in info['largest_files'][:3]:
                    report += f"- {file_info['name']} ({file_info['size_mb']:.2f}MB)\n"
        
        # File type analysis
        report += f"\n## File Type Analysis\n"
        sorted_types = sorted(
            analysis['file_types'].items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )
        
        for file_type, info in sorted_types[:10]:  # Top 10 types
            report += f"- **{file_type}**: {info['count']} files, {info['size_mb']:.2f}MB\n"
        
        # Large files
        if analysis['large_files']:
            report += f"\n## Large Individual Files (>1MB)\n"
            for file_info in analysis['large_files']:
                report += f"- {file_info['name']} ({file_info['size_mb']:.2f}MB)\n"
        
        # Recommendations
        if analysis['recommendations']:
            report += f"\n## Recommendations\n"
            for rec in analysis['recommendations']:
                report += f"- {rec}\n"
        
        return report
    
    def export_csv(self, analysis: Dict, output_file: str):
        """Export analysis to CSV format"""
        logger.info(f"Exporting analysis to {output_file}")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'Directory', 'Size_MB', 'File_Count', 'Deployment_Action', 'Notes'
            ])
            
            # Data rows
            for dirname, info in analysis['directories'].items():
                writer.writerow([
                    dirname,
                    info['size_mb'],
                    info['file_count'],
                    info['deployment_action'],
                    info['notes']
                ])


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Analyze RIF knowledge base size")
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Path to knowledge directory (default: knowledge)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed file listings"
    )
    parser.add_argument(
        "--export-csv",
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--output-report",
        help="Save report to file"
    )
    
    args = parser.parse_args()
    
    # Perform analysis
    analyzer = KnowledgeSizeAnalyzer(args.knowledge_dir)
    analysis = analyzer.analyze_full_breakdown()
    
    # Generate report
    report = analyzer.generate_report(analysis, detailed=args.detailed)
    
    # Export CSV if requested
    if args.export_csv:
        analyzer.export_csv(analysis, args.export_csv)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_report}")
    
    # Print report
    print(report)


if __name__ == "__main__":
    main()