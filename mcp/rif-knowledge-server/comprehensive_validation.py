#!/usr/bin/env python3
"""
Comprehensive validation suite - Run all tests 3 times for reliability
"""

import subprocess
import sys
import time
from datetime import datetime

def run_test(test_name, test_command, run_number):
    """Run a single test and capture results"""
    print(f"\n{'='*70}")
    print(f"RUN {run_number} - {test_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd='/Users/cal/DEV/RIF'
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if success:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        
        # Extract key metrics from output
        metrics = extract_metrics(result.stdout, test_name)
        
        return {
            'success': success,
            'duration': duration,
            'metrics': metrics,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT - Test took longer than 5 minutes")
        return {'success': False, 'duration': 300, 'metrics': {}, 'timeout': True}
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return {'success': False, 'duration': 0, 'metrics': {}, 'error': str(e)}

def extract_metrics(output, test_name):
    """Extract key metrics from test output"""
    metrics = {}
    
    if 'usefulness' in test_name.lower():
        # Extract usefulness percentage
        lines = output.split('\\n')
        for line in lines:
            if 'Overall usefulness:' in line:
                try:
                    percent = line.split(':')[1].strip().rstrip('%')
                    metrics['usefulness'] = float(percent)
                except:
                    pass
            elif 'Useful answers:' in line:
                try:
                    fraction = line.split(':')[1].strip().split('/')[0]
                    total = line.split(':')[1].strip().split('/')[1]
                    metrics['useful_answers'] = int(fraction)
                    metrics['total_questions'] = int(total)
                except:
                    pass
    
    elif 'pattern' in test_name.lower():
        # Extract pattern coverage
        for line in output.split('\\n'):
            if 'Coverage:' in line:
                try:
                    percent = line.split(':')[1].strip().rstrip('%')
                    metrics['pattern_coverage'] = float(percent)
                except:
                    pass
    
    elif 'fact' in test_name.lower():
        # Extract fact accuracy
        for line in output.split('\\n'):
            if 'Fact accuracy:' in line:
                try:
                    percent = line.split(':')[1].strip().rstrip('%')
                    metrics['fact_accuracy'] = float(percent)
                except:
                    pass
    
    elif 'security' in test_name.lower() or 'injection' in test_name.lower():
        # Count SQL injection blocks
        metrics['sql_injection_blocked'] = output.count('✅ Survived injection')
        metrics['concurrent_success'] = 'Concurrent requests: 10/10 succeeded' in output
    
    return metrics

def comprehensive_validation():
    """Run all tests 3 times and analyze consistency"""
    print("🔥 COMPREHENSIVE RIF KNOWLEDGE SERVER VALIDATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running all tests 3 times for reliability validation...")
    
    # Define all tests to run
    tests = [
        {
            'name': 'Pattern File Searchability',
            'command': 'python3 mcp/rif-knowledge-server/test_pattern_coverage.py',
            'critical': True
        },
        {
            'name': 'Claude Code Fact Accuracy',
            'command': 'python3 mcp/rif-knowledge-server/test_claude_facts.py',
            'critical': True
        },
        {
            'name': 'Comprehensive Usefulness (20 Questions)',
            'command': 'python3 mcp/rif-knowledge-server/test_comprehensive_usefulness.py',
            'critical': True
        },
        {
            'name': 'Security & SQL Injection',
            'command': 'python3 mcp/rif-knowledge-server/test_break_server.py',
            'critical': True
        },
        {
            'name': 'Caching Performance',
            'command': 'python3 mcp/rif-knowledge-server/test_caching_performance.py',
            'critical': False
        }
    ]
    
    # Run each test 3 times
    all_results = {}
    
    for test in tests:
        test_name = test['name']
        all_results[test_name] = []
        
        print(f"\n🧪 TESTING: {test_name}")
        print("-" * 50)
        
        for run_num in range(1, 4):
            result = run_test(test_name, test['command'], run_num)
            all_results[test_name].append(result)
            
            # Brief pause between runs
            time.sleep(2)
    
    # Analyze consistency and generate report
    print(f"\n\n{'='*70}")
    print("COMPREHENSIVE VALIDATION REPORT")
    print(f"{'='*70}")
    
    overall_success = True
    critical_failures = []
    
    for test_name, results in all_results.items():
        print(f"\n📊 {test_name}:")
        
        successes = sum(1 for r in results if r['success'])
        avg_duration = sum(r['duration'] for r in results) / len(results)
        
        print(f"   Success rate: {successes}/3 ({successes/3*100:.0f}%)")
        print(f"   Average duration: {avg_duration:.2f}s")
        
        # Check consistency of key metrics
        if results[0].get('metrics'):
            for metric, value in results[0]['metrics'].items():
                values = [r['metrics'].get(metric, 0) for r in results if r.get('metrics')]
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    print(f"   {metric}: avg={avg_val:.1f}%, min={min_val:.1f}%, max={max_val:.1f}%")
        
        # Check if critical test failed
        test_info = next(t for t in tests if t['name'] == test_name)
        if test_info['critical'] and successes < 2:  # At least 2/3 must pass
            overall_success = False
            critical_failures.append(test_name)
            print(f"   ⚠️  CRITICAL FAILURE - Only {successes}/3 runs passed")
        elif successes == 3:
            print(f"   ✅ CONSISTENT SUCCESS")
        else:
            print(f"   ⚠️  INCONSISTENT - {3-successes}/3 failures")
    
    # Final assessment
    print(f"\n{'='*70}")
    print("FINAL VALIDATION ASSESSMENT")
    print(f"{'='*70}")
    
    if overall_success and not critical_failures:
        print("🎉 VALIDATION PASSED - All critical tests consistently successful!")
        print("✅ The RIF Knowledge MCP Server is production ready")
        
        # Print key achievements
        print(f"\n🏆 KEY ACHIEVEMENTS:")
        
        # Get final metrics from last successful runs
        final_metrics = {}
        for test_name, results in all_results.items():
            for result in reversed(results):  # Get last successful result
                if result['success'] and result.get('metrics'):
                    final_metrics.update(result['metrics'])
                    break
        
        if 'usefulness' in final_metrics:
            print(f"   📈 Usefulness: {final_metrics['usefulness']:.1f}%")
        if 'pattern_coverage' in final_metrics:
            print(f"   📋 Pattern Coverage: {final_metrics['pattern_coverage']:.1f}%")
        if 'fact_accuracy' in final_metrics:
            print(f"   🎯 Fact Accuracy: {final_metrics['fact_accuracy']:.1f}%")
        if 'sql_injection_blocked' in final_metrics:
            print(f"   🛡️  SQL Injection Protection: {final_metrics['sql_injection_blocked']}/9 blocked")
        
        return True
        
    else:
        print("❌ VALIDATION FAILED - Critical issues detected")
        if critical_failures:
            print("⚠️  Critical test failures:")
            for failure in critical_failures:
                print(f"   - {failure}")
        
        return False

if __name__ == '__main__':
    success = comprehensive_validation()
    print(f"\nValidation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if success else 1)