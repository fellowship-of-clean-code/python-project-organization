from abc import ABC, abstractmethod

class Abstract(ABC):
    
    @abstractmethod
    def compute(self, value):
        ...
    
    def compute_twice(self, value):
        return self.compute(self.compute(value))
    

class Double(Abstract):
    
    def compute(self, value):
        return 2*value
    
class Square(Abstract):
    
    def compute(self, value):
        return value**2
