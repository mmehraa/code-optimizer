from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod

@dataclass
class OptimizationResult:
    """Result of code optimization process"""
    original_code: str
    optimized_code: str
    optimizations_applied: List[str]
    performance_gain: str

class BaseOptimizer(ABC):
    """Base class for all code optimizers"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    @abstractmethod
    def optimize(self, code: str) -> OptimizationResult:
        """Main optimization entry point"""
        pass
    
    def reset_optimizations(self):
        """Reset the list of applied optimizations"""
        self.optimizations_applied = []