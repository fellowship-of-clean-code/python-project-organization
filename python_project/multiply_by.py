from .class_definition import Abstract

class MultiplyBy(Abstract):
    
    def __init__(self, multiplication_factor):
        self.multiplication_factor = multiplication_factor
    
    def compute(self, value):
        return self.multiplication_factor * value
