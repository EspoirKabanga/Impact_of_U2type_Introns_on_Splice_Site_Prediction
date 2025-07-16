"""
U2 Intron Splice Site Prediction Package

A refactored and improved implementation of splice site prediction models
for U2-type introns in Arabidopsis thaliana.

Modules:
    config: Configuration management
    utils: Utility functions
    data_preprocessing: Data loading and preprocessing
    models: Neural network model definitions
    metrics: Metrics calculation and results management
    training: Training pipeline and cross-validation
    main: Main execution script

Usage:
    python -m src.main --help
    
Example:
    # Run quick test
    python -m src.main --quick-test
    
    # Run specific experiments
    python -m src.main --filetypes donor --models IntSplicer --data-categories short long
"""

__version__ = "2.0.0"
__author__ = "Refactored U2 Intron Splice Site Prediction"

# Import main classes for convenience
from .config import Config
from .models import ModelFactory
from .training import ExperimentRunner
from .metrics import ResultsManager

__all__ = [
    'Config',
    'ModelFactory', 
    'ExperimentRunner',
    'ResultsManager'
] 