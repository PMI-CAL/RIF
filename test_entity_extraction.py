#!/usr/bin/env python3
"""
Simple test to verify entity extraction system works.
"""

def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers."""  
    return a * b

class Calculator:
    """Simple calculator class."""
    
    def __init__(self, precision=2):
        self.precision = precision
    
    def add(self, x, y):
        return round(x + y, self.precision)
    
    def multiply(self, x, y):
        return round(x * y, self.precision)

# Module-level constants
PI = 3.14159
DEBUG_MODE = True