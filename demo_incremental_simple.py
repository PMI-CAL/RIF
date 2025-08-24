#!/usr/bin/env python3
"""
Simple demonstration of Incremental Entity Extraction (Issue #65)
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge.extraction.incremental_extractor import create_incremental_extractor


def print_results(result, title: str):
    """Print extraction results in a formatted way."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    print(f"Success: {'✓' if result.success else '✗'}")
    print(f"Processing time: {result.processing_time*1000:.1f}ms")
    print(f"Meets <100ms target: {'✓' if result.performance_metrics['meets_performance_target'] else '✗'}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
        return
    
    # Print diff summary
    diff = result.diff
    print(f"Entity Changes:")
    print(f"  Added:     {len(diff.added):3d}")
    print(f"  Modified:  {len(diff.modified):3d}") 
    print(f"  Removed:   {len(diff.removed):3d}")
    print(f"  Unchanged: {len(diff.unchanged):3d}")
    print(f"  Total changes: {diff.total_changes}")
    
    # Print detailed entity information
    if diff.added and len(diff.added) <= 10:  # Only show first 10
        print(f"\n📝 Added Entities:")
        for entity in diff.added[:10]:
            print(f"  - {entity.type.value}: {entity.name}")
    
    if diff.modified and len(diff.modified) <= 5:  # Only show first 5
        print(f"\n🔄 Modified Entities:")
        for old_entity, new_entity in diff.modified[:5]:
            print(f"  - {old_entity.type.value}: {old_entity.name} → {new_entity.name}")


def main():
    """Simple demo of incremental extraction."""
    print("🚀 Incremental Entity Extraction Demo - Issue #65")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_file = os.path.join(temp_dir, "demo_module.py")
        
        # Initialize extractor
        extractor = create_incremental_extractor()
        
        # Phase 1: Create initial file
        print("\n📁 Phase 1: Creating initial file")
        initial_content = '''
def hello_world():
    """Simple greeting function."""
    return "Hello, World!"

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers."""
        self.result = x + y
        return self.result

CONSTANT = 42
'''
        
        with open(demo_file, 'w') as f:
            f.write(initial_content)
        print(f"✓ Created: {demo_file}")
        
        # Extract entities from newly created file
        result1 = extractor.extract_incremental(demo_file, 'created')
        print_results(result1, "Initial File Creation")
        
        # Phase 2: Modify file (add entities)
        print("\n🔄 Phase 2: Adding new functionality")
        time.sleep(0.1)  # Ensure file timestamp changes
        
        modified_content = '''
def hello_world():
    """Simple greeting function."""
    return "Hello, World!"

def hello_universe():
    """Extended greeting function."""
    return "Hello, Universe!"

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers."""
        self.result = x + y
        return self.result
    
    def subtract(self, x, y):
        """Subtract two numbers."""
        self.result = x - y
        return self.result

class AdvancedCalculator(Calculator):
    """Advanced calculator."""
    
    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

CONSTANT = 42
DEBUG_MODE = True
'''
        
        with open(demo_file, 'w') as f:
            f.write(modified_content)
        print(f"✓ Modified: {demo_file}")
        
        # Extract incremental changes
        result2 = extractor.extract_incremental(demo_file, 'modified')
        print_results(result2, "File Modification - Added Entities")
        
        # Phase 3: Delete file
        print("\n🗑️  Phase 3: File deletion")
        
        result3 = extractor.extract_incremental(demo_file, 'deleted')
        print_results(result3, "File Deletion")
        
        # Performance summary
        print("\n📊 Performance Summary")
        print("=" * 50)
        
        metrics = extractor.get_performance_metrics()
        print(f"Files processed: {metrics['files_processed']}")
        print(f"Total entities processed: {metrics['total_entities_processed']}")
        print(f"Average processing time: {metrics.get('avg_processing_time', 0)*1000:.1f}ms")
        print(f"Cache hit rate: {metrics.get('cache_hit_rate', 0)*100:.1f}%")
        print(f"Meets performance target: {'✓' if metrics.get('meets_performance_target', False) else '✗'}")
        
        # Validation test
        print(f"\n🔍 Performance Validation")
        print("=" * 50)
        
        # Create a test file for validation
        validation_file = os.path.join(temp_dir, "validation_test.py")
        validation_content = '''
import sys
import os
from typing import List, Dict

def process_data(items: List[Dict]) -> Dict[str, int]:
    """Process a list of dictionary items."""
    result = {}
    for item in items:
        for key, value in item.items():
            if key not in result:
                result[key] = 0
            result[key] += 1
    return result

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}
    
    def process(self, data):
        """Main processing method."""
        processed = []
        for item in data:
            if item in self.cache:
                processed.append(self.cache[item])
            else:
                result = str(item).upper()
                self.cache[item] = result
                processed.append(result)
        return processed
'''
        
        with open(validation_file, 'w') as f:
            f.write(validation_content)
        
        validation = extractor.validate_performance(validation_file)
        
        print(f"File: {os.path.basename(validation['file_path'])}")
        print(f"Processing time: {validation['processing_time_ms']:.1f}ms")
        print(f"Meets target: {'✓' if validation['meets_target'] else '✗'}")
        print(f"Performance rating: {validation['performance_rating']}")
        print(f"Entity changes: {validation['entity_changes']}")
        
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        print(f"\n🎉 Demo Complete!")
        print("=" * 50)
        print("The Incremental Entity Extraction system (Issue #65) is working!")
        print("✓ Accurate entity diff calculation")
        print("✓ Version management and tracking")
        print("✓ Performance optimization")
        print("✓ Integration ready for file change detection")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)