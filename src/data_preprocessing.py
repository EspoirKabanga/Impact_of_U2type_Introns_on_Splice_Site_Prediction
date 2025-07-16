import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
import random
import logging
from pathlib import Path

from .config import Config
from .utils import (
    read_sequences_from_file, 
    validate_sequence_length, 
    validate_dna_sequences,
    calculate_dataset_stats
)

logger = logging.getLogger(__name__)

class DNASequenceProcessor:
    """Class for processing DNA sequences for splice site prediction"""
    
    # One-hot encoding mapping for DNA nucleotides
    NUCLEOTIDE_MAPPING = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0], 
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # Unknown nucleotide
    }
    
    def __init__(self, config: Config = Config):
        self.config = config
        
    def one_hot_encode(self, sequences: List[str]) -> np.ndarray:
        """
        One-hot encode DNA sequences with validation
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            One-hot encoded sequences as numpy array
            
        Raises:
            ValueError: If sequences are invalid
        """
        if not sequences:
            raise ValueError("Empty sequences list provided")
            
        # Validate sequences
        if not validate_sequence_length(sequences, self.config.SEQUENCE_LENGTH):
            raise ValueError(f"All sequences must be {self.config.SEQUENCE_LENGTH} nucleotides long")
            
        if not validate_dna_sequences(sequences):
            raise ValueError("Sequences contain invalid nucleotides")
        
        try:
            encoded = []
            for seq in sequences:
                seq_encoded = [self.NUCLEOTIDE_MAPPING[nucleotide.upper()] 
                             for nucleotide in seq]
                encoded.append(seq_encoded)
            
            return np.array(encoded, dtype=np.float32)
            
        except KeyError as e:
            raise ValueError(f"Invalid nucleotide found: {e}")
        except Exception as e:
            raise ValueError(f"Error during one-hot encoding: {e}")
    
    def remove_duplicates(self, sequences: List[str]) -> List[str]:
        """Remove duplicate sequences while preserving order"""
        return list(OrderedDict.fromkeys(sequences))
    
    def load_and_preprocess_data(self, positive_file: Path, negative_file: Path,
                               sample_size: int = None, 
                               balance_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess positive and negative sequence data
        
        Args:
            positive_file: Path to positive sequences file
            negative_file: Path to negative sequences file  
            sample_size: Number of positive samples to use (None for all)
            balance_ratio: Negative to positive ratio (1.0 = balanced)
            
        Returns:
            Tuple of (one-hot encoded sequences, labels)
        """
        logger.info(f"Loading data from {positive_file} and {negative_file}")
        
        # Load sequences
        positive_sequences = read_sequences_from_file(positive_file)
        negative_sequences = read_sequences_from_file(negative_file)
        
        # Remove duplicates
        positive_sequences = self.remove_duplicates(positive_sequences)
        negative_sequences = self.remove_duplicates(negative_sequences)
        
        logger.info(f"Loaded {len(positive_sequences)} positive and {len(negative_sequences)} negative sequences")
        
        # Sample positive sequences if specified
        if sample_size and sample_size < len(positive_sequences):
            positive_sequences = random.sample(positive_sequences, sample_size)
            logger.info(f"Sampled {sample_size} positive sequences")
        
        # Sample negative sequences based on balance ratio
        neg_sample_size = int(len(positive_sequences) * balance_ratio)
        if neg_sample_size < len(negative_sequences):
            negative_sequences = random.sample(negative_sequences, neg_sample_size)
            logger.info(f"Sampled {neg_sample_size} negative sequences (ratio: {balance_ratio})")
        
        # Log dataset statistics
        stats = calculate_dataset_stats(positive_sequences, negative_sequences)
        logger.info(f"Final dataset: {stats}")
        
        # One-hot encode sequences
        try:
            positive_encoded = self.one_hot_encode(positive_sequences)
            negative_encoded = self.one_hot_encode(negative_sequences)
        except ValueError as e:
            logger.error(f"Error during encoding: {e}")
            raise
        
        # Combine data and create labels
        X = np.vstack([positive_encoded, negative_encoded])
        y = np.hstack([
            np.ones(len(positive_sequences), dtype=np.int32),
            np.zeros(len(negative_sequences), dtype=np.int32)
        ])
        
        logger.info(f"Created dataset with shape {X.shape} and {len(y)} labels")
        
        return X, y

class DataLoader:
    """Class for loading different types of datasets"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.processor = DNASequenceProcessor(config)
    
    def load_dataset(self, filetype: str, data_category: str, 
                    balance_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset for specified filetype and category
        
        Args:
            filetype: 'donor' or 'acceptor'
            data_category: Dataset category (e.g., 'short', 'long', etc.)
            balance_ratio: Negative to positive sample ratio
            
        Returns:
            Tuple of (features, labels)
        """
        if filetype not in self.config.FILETYPES:
            raise ValueError(f"Invalid filetype: {filetype}")
            
        if data_category not in self.config.DATASET_CATEGORIES:
            raise ValueError(f"Invalid data category: {data_category}")
        
        # Get file paths
        positive_file = self.config.get_data_file_path(filetype, data_category)
        negative_file = self.config.get_negative_file_path(filetype)
        
        # Get sample size
        sample_size = self.config.get_sample_size(filetype, data_category)
        
        # Load and preprocess data
        return self.processor.load_and_preprocess_data(
            positive_file, negative_file, sample_size, balance_ratio
        )
    
    def prepare_training_data(self, X: np.ndarray, y: np.ndarray, 
                            fold_indices: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare training data for a specific fold
        
        Args:
            X: Feature data
            y: Labels
            fold_indices: Tuple of (train_indices, validation_indices)
            
        Returns:
            Dictionary with prepared training data
        """
        train_idx, val_idx = fold_indices
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Reshape for CNN input (samples, sequence_length, channels)
        X_train = X_train.reshape(-1, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS)
        X_val = X_val.reshape(-1, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS)
        
        # Convert to categorical labels
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=self.config.N_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=self.config.N_CLASSES)
        
        # Ensure float32 dtype for GPU compatibility
        return {
            'X_train': X_train.astype(np.float32),
            'y_train': y_train_cat.astype(np.float32),
            'X_val': X_val.astype(np.float32),
            'y_val': y_val_cat.astype(np.float32)
        }
    
    def prepare_test_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Prepare test data
        
        Args:
            X: Feature data
            y: Labels
            
        Returns:
            Dictionary with prepared test data
        """
        # Reshape for CNN input
        X_test = X.reshape(-1, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS)
        
        # Convert to categorical labels
        from tensorflow.keras.utils import to_categorical
        y_test_cat = to_categorical(y, num_classes=self.config.N_CLASSES)
        
        return {
            'X_test': X_test.astype(np.float32),
            'y_test': y_test_cat.astype(np.float32)
        } 
