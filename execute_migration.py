#!/usr/bin/env python3
"""
RIF Knowledge System Migration Execution Script
Issue #39: Execute 4-week phased migration from LightRAG to hybrid system

This script executes the migration phases according to the PRD:
- Phase 1: Parallel Installation (Week 1)
- Phase 2: Read Migration (Week 2) 
- Phase 3: Write Migration (Week 3)
- Phase 4: Cutover (Week 4)

Usage:
    python execute_migration.py --phase 1 --execute
    python execute_migration.py --status
    python execute_migration.py --rollback pre_migration
"""

import argparse
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add knowledge directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge'))

from knowledge.migration_coordinator import MigrationCoordinator, MigrationPhase, MigrationError


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('migration.log')
        ]
    )
    return logging.getLogger(__name__)


def print_migration_status(coordinator: MigrationCoordinator):
    """Print formatted migration status."""
    status = coordinator.get_migration_status()
    
    print("\n" + "="*60)
    print("RIF KNOWLEDGE SYSTEM MIGRATION STATUS")
    print("="*60)
    
    print(f"Current Phase: {status['current_phase']}")
    
    if status['migration_start_time']:
        print(f"Migration Started: {status['migration_start_time']}")
    
    if status['phase_deadline']:
        print(f"Phase Deadline: {status['phase_deadline']}")
    
    print(f"Rollback Points: {', '.join(status['rollback_points'])}")
    
    print("\nSystems Status:")
    systems = status['systems_status']
    print(f"  LightRAG Available: {'✅' if systems['lightrag_available'] else '❌'}")
    print(f"  Hybrid System Available: {'✅' if systems['hybrid_available'] else '❌'}")
    
    print("\nPerformance Metrics:")
    metrics = status['performance_metrics']
    if metrics.get('status') == 'no_data':
        print("  No performance data available yet")
    else:
        print(f"  Total Operations: {metrics.get('total_operations', 0)}")
        print(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
        if 'avg_response_time' in metrics:
            print(f"  Average Response Time: {metrics['avg_response_time']:.3f}s")
    
    print("="*60 + "\n")


def execute_phase_1(coordinator: MigrationCoordinator, logger: logging.Logger) -> bool:
    """Execute Phase 1: Parallel Installation."""
    logger.info("Starting Phase 1 execution: Parallel Installation")
    
    print("\n🚀 PHASE 1: PARALLEL INSTALLATION")
    print("="*50)
    print("Setting up hybrid system alongside LightRAG...")
    print("• Shadow mode indexing")
    print("• Performance validation")
    print("• No agent behavior changes")
    print("="*50)
    
    try:
        success = coordinator.start_migration()
        
        if success:
            print("✅ Phase 1 completed successfully!")
            print("• Hybrid system is running in parallel")
            print("• Existing knowledge migrated")
            print("• Shadow indexing active")
            print("• Performance validated")
            
            logger.info("Phase 1 completed successfully")
            return True
        else:
            print("❌ Phase 1 failed!")
            print("Check logs for details.")
            logger.error("Phase 1 execution failed")
            return False
            
    except MigrationError as e:
        print(f"❌ Migration error: {e}")
        logger.error(f"Migration error in Phase 1: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in Phase 1: {e}")
        return False


def execute_phase_2(coordinator: MigrationCoordinator, logger: logging.Logger) -> bool:
    """Execute Phase 2: Read Migration."""
    logger.info("Starting Phase 2 execution: Read Migration")
    
    print("\n🚀 PHASE 2: READ MIGRATION")
    print("="*50)
    print("Routing read queries to hybrid system...")
    print("• A/B testing query results")
    print("• Performance monitoring")
    print("• Keep writes on LightRAG")
    print("="*50)
    
    try:
        success = coordinator.progress_to_next_phase()
        
        if success:
            print("✅ Phase 2 completed successfully!")
            print("• Read routing to hybrid system active")
            print("• A/B testing validated results")
            print("• Performance meets requirements")
            print("• Writes still on LightRAG")
            
            logger.info("Phase 2 completed successfully")
            return True
        else:
            print("❌ Phase 2 failed!")
            print("Check logs for details.")
            logger.error("Phase 2 execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in Phase 2: {e}")
        return False


def execute_phase_3(coordinator: MigrationCoordinator, logger: logging.Logger) -> bool:
    """Execute Phase 3: Write Migration."""
    logger.info("Starting Phase 3 execution: Write Migration")
    
    print("\n🚀 PHASE 3: WRITE MIGRATION")
    print("="*50)
    print("Enabling dual-write to both systems...")
    print("• Writes go to both LightRAG and hybrid")
    print("• Data consistency verification")
    print("• Performance monitoring")
    print("="*50)
    
    try:
        success = coordinator.progress_to_next_phase()
        
        if success:
            print("✅ Phase 3 completed successfully!")
            print("• Dual-write mode active")
            print("• Data consistency verified")
            print("• Write performance acceptable")
            print("• Ready for final cutover")
            
            logger.info("Phase 3 completed successfully")
            return True
        else:
            print("❌ Phase 3 failed!")
            print("Check logs for details.")
            logger.error("Phase 3 execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in Phase 3: {e}")
        return False


def execute_phase_4(coordinator: MigrationCoordinator, logger: logging.Logger) -> bool:
    """Execute Phase 4: Cutover."""
    logger.info("Starting Phase 4 execution: Cutover")
    
    print("\n🚀 PHASE 4: CUTOVER")
    print("="*50)
    print("Completing migration to hybrid system...")
    print("• Final system validation")
    print("• Disable LightRAG system")
    print("• Archive LightRAG data")
    print("• Migration cleanup")
    print("="*50)
    
    try:
        success = coordinator.progress_to_next_phase()
        
        if success:
            print("✅ Phase 4 completed successfully!")
            print("• Migration fully complete")
            print("• Hybrid system is primary")
            print("• LightRAG archived safely")
            print("• System optimized")
            
            logger.info("Phase 4 completed successfully")
            return True
        else:
            print("❌ Phase 4 failed!")
            print("Check logs for details.")
            logger.error("Phase 4 execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in Phase 4: {e}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='RIF Knowledge System Migration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python execute_migration.py --phase 1 --execute    # Execute Phase 1
  python execute_migration.py --status               # Show status
  python execute_migration.py --rollback pre_migration # Rollback to point
        """
    )
    
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Migration phase to execute (1-4)')
    parser.add_argument('--execute', action='store_true',
                        help='Execute the specified phase')
    parser.add_argument('--status', action='store_true',
                        help='Show migration status')
    parser.add_argument('--rollback', type=str,
                        help='Rollback to specified point')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--knowledge-path', type=str,
                        help='Path to knowledge directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize coordinator
        coordinator = MigrationCoordinator(
            knowledge_path=args.knowledge_path,
        )
        
        # Handle status request
        if args.status:
            print_migration_status(coordinator)
            return 0
        
        # Handle rollback request
        if args.rollback:
            print(f"🔄 Rollback to '{args.rollback}' not implemented yet")
            print("This would restore system state to the specified rollback point")
            return 0
        
        # Handle phase execution
        if args.execute and args.phase:
            if args.phase == 1:
                success = execute_phase_1(coordinator, logger)
                if success:
                    print("\n📋 NEXT STEPS:")
                    print("1. Monitor system performance for 24-48 hours")
                    print("2. Verify shadow indexing is working correctly")
                    print("3. When ready, proceed to Phase 2:")
                    print("   python execute_migration.py --phase 2 --execute")
                    return 0
                else:
                    print("\n🛠️  TROUBLESHOOTING:")
                    print("1. Check migration.log for detailed error information")
                    print("2. Verify both LightRAG and hybrid systems are accessible")
                    print("3. Ensure sufficient disk space and memory available")
                    print("4. Consider rollback if issues persist:")
                    print("   python execute_migration.py --rollback pre_migration")
                    return 1
            
            elif args.phase == 2:
                success = execute_phase_2(coordinator, logger)
                if success:
                    print("\n📋 NEXT STEPS:")
                    print("1. Monitor read routing for 24-48 hours")
                    print("2. Verify A/B test results are satisfactory")
                    print("3. When ready, proceed to Phase 3:")
                    print("   python execute_migration.py --phase 3 --execute")
                    return 0
                else:
                    print("\n🛠️  TROUBLESHOOTING:")
                    print("1. Check A/B test similarity scores")
                    print("2. Verify read routing configuration")
                    print("3. Consider rollback if performance issues persist")
                    return 1
                
            elif args.phase == 3:
                success = execute_phase_3(coordinator, logger)
                if success:
                    print("\n📋 NEXT STEPS:")
                    print("1. Monitor dual-write consistency for 24-48 hours")
                    print("2. Verify data consistency between systems")
                    print("3. When ready, proceed to final cutover:")
                    print("   python execute_migration.py --phase 4 --execute")
                    return 0
                else:
                    print("\n🛠️  TROUBLESHOOTING:")
                    print("1. Check data consistency verification")
                    print("2. Monitor write performance metrics")
                    print("3. Consider rollback if consistency issues persist")
                    return 1
                
            elif args.phase == 4:
                success = execute_phase_4(coordinator, logger)
                if success:
                    print("\n🎉 MIGRATION COMPLETE!")
                    print("✅ Successfully migrated to hybrid knowledge system")
                    print("📊 LightRAG archived, hybrid system active")
                    print("📈 Check final report: knowledge/migration_final_report.json")
                    return 0
                else:
                    print("\n🛠️  TROUBLESHOOTING:")
                    print("1. Check final validation results")
                    print("2. Verify hybrid system health")
                    print("3. Consider rollback if critical issues found")
                    return 1
        
        # No specific action requested
        parser.print_help()
        print("\n💡 TIP: Start with --status to see current migration state")
        return 0
        
    except Exception as e:
        logger.error(f"Migration tool error: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())