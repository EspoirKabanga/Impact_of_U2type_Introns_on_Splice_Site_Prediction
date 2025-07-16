import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('splice_prediction.log')
        ]
    )
    return logging.getLogger(__name__)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is 0"""
    return numerator / denominator if denominator > 0 else default

def read_sequences_from_file(filepath: Path) -> List[str]:
    """
    Read DNA sequences from a file safely
    
    Args:
        filepath: Path to the file containing sequences
        
    Returns:
        List of cleaned DNA sequences
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(filepath, 'r') as f:
            sequences = [line.strip() for line in f.readlines() if line.strip()]
        return sequences
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}")

def create_directory_safe(directory: Path) -> None:
    """Create directory safely if it doesn't exist"""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Error creating directory {directory}: {e}")

def remove_directory_safe(directory: Path) -> None:
    """Remove directory safely if it exists"""
    import shutil
    try:
        if directory.exists():
            shutil.rmtree(directory)
    except OSError as e:
        logging.warning(f"Error removing directory {directory}: {e}")

def validate_sequence_length(sequences: List[str], expected_length: int) -> bool:
    """
    Validate that all sequences have the expected length
    
    Args:
        sequences: List of DNA sequences
        expected_length: Expected sequence length
        
    Returns:
        True if all sequences have correct length, False otherwise
    """
    if not sequences:
        return False
    
    invalid_sequences = [seq for seq in sequences if len(seq) != expected_length]
    
    if invalid_sequences:
        logging.warning(f"Found {len(invalid_sequences)} sequences with incorrect length")
        return False
    
    return True

def validate_dna_sequences(sequences: List[str]) -> bool:
    """
    Validate that sequences contain only valid DNA nucleotides
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        True if all sequences are valid, False otherwise
    """
    valid_nucleotides = set('ACGTN')
    
    for i, seq in enumerate(sequences):
        seq_set = set(seq.upper())
        if not seq_set.issubset(valid_nucleotides):
            invalid_chars = seq_set - valid_nucleotides
            logging.warning(f"Sequence {i} contains invalid characters: {invalid_chars}")
            return False
    
    return True

def get_model_project_name(model_name: str, data_category: str, 
                          filetype: str, batch_size: int, fold: int) -> str:
    """Generate standardized project name for model training"""
    return f"{model_name}_zzz_{data_category}_{filetype}_{batch_size}_fold_{fold}"

def format_metrics_for_display(metrics: Tuple[float, ...]) -> str:
    """Format metrics tuple for readable display"""
    metric_names = ['Recall', 'Precision', 'F1', 'FPR', 'FDR', 'Specificity', 'Sensitivity', 'MCC']
    formatted = [f"{name}: {value:.4f}" for name, value in zip(metric_names, metrics)]
    return " | ".join(formatted)

def calculate_dataset_stats(positive_sequences: List[str], 
                          negative_sequences: List[str]) -> dict:
    """Calculate and return dataset statistics"""
    return {
        'positive_count': len(positive_sequences),
        'negative_count': len(negative_sequences),
        'total_count': len(positive_sequences) + len(negative_sequences),
        'positive_ratio': len(positive_sequences) / (len(positive_sequences) + len(negative_sequences)),
        'negative_ratio': len(negative_sequences) / (len(positive_sequences) + len(negative_sequences))
    } 
