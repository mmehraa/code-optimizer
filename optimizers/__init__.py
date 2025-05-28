__version__ = "1.0.0"
__author__ = "Code Optimizer Team"

# Import main optimization classes
from .optimizers.c_optimizer import COptimizer
from .optimizers.python_optimizer import PythonOptimizer
from .utils.code_analyzer import CodeAnalyzer
from .utils.optimization_report import OptimizationReport

__all__ = [
    'COptimizer',
    'PythonOptimizer', 
    'CodeAnalyzer',
    'OptimizationReport'
]

# Package-level configuration
SUPPORTED_LANGUAGES = ['c', 'python']
DEFAULT_OPTIMIZATIONS = [
    'dead_code_removal',
    'loop_optimization', 
    'constant_folding',
    'strength_reduction',
    'function_inlining'
]

def get_optimizer(language):
    """Factory function to get appropriate optimizer"""
    if language.lower() == 'c':
        return COptimizer()
    elif language.lower() == 'python':
        return PythonOptimizer()
    else:
        raise ValueError(f"Unsupported language: {language}")

def get_supported_languages():
    """Return list of supported programming languages"""
    return SUPPORTED_LANGUAGES.copy()

def get_available_optimizations():
    """Return list of available optimization techniques"""
    return DEFAULT_OPTIMIZATIONS.copy()