#! /usr/bin/env python3

from python_project.class_definition import Double
from python_project.multiply_by import MultiplyBy

if __name__ == '__main__':
    
    double = Double()
    
    print(f"doubling 1 yields {double.compute(1)}")
    print(f"doubling 1 twice yields {double.compute_twice(1)}")
    
    quintuple = MultiplyBy(5)
    
    print(f"quintupling 1 yields {quintuple.compute(1)}")
    print(f"quintupling 1 twice yields {quintuple.compute_twice(1)}")
