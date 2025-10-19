from pathlib import Path
from typing import List, Dict, Any
import os

class Config:
    """Central configuration class for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    SORTED_DIR = SRC_DIR / "sorted"
    RESULTS_DIR = PROJECT_ROOT / "results"
    SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
    WEIGHTS_DIR = PROJECT_ROOT / "best_weights"
    
    # Sequence parameters
    SEQUENCE_LENGTH = 402
    N_CLASSES = 2
    INPUT_CHANNELS = 4  # A, C, G, T
    
    # Training parameters
    N_FOLDS = 5
    TEST_SIZE = 0.2
    EPOCHS = 30
    BATCH_SIZES = [64]  # Can extend to [32, 64, 128, 256] for experiments
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    
    # Data loading parameters
    USE_ALL_NEGATIVES = True  # When True, do not sample negatives; use all available
    
    # Model parameters
    PATIENCE = 5  # Early stopping patience
    MONITOR_METRIC = 'val_loss'
    
    # Data categories and file types
    FILETYPES = ['donor', 'acceptor']
    DATASET_CATEGORIES = [
        'short', 'long', 'mix_long_short', 
        'multiple', 'single', 'mix_single_multiple'
    ]
    
    # Sample sizes for different categories
    SAMPLE_SIZES: Dict[str, Dict[str, int]] = {
        'donor': {
            'length_based': 18859,  # short, long, mix_long_short
            'count_based': 13987    # multiple, single, mix_single_multiple
        },
        'acceptor': {
            'length_based': 19322,  # short, long, mix_long_short
            'count_based': 14112    # multiple, single, mix_single_multiple
        }
    }
    
    # Intron length threshold
    INTRON_LENGTH_THRESHOLD = 90
    
    # TensorFlow configuration
    TF_CONFIG = {
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '3',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true'
    }
    
    # Metrics to calculate and save
    METRICS_COLUMNS = [
        'Fold', 'Sensitivity', 'Precision', 'F1-score', 
        'False Positive Rate', 'False Discovery Rate', 
        'Specificity', 'Recall', 'MCC', 'AUROC', 'AUPRC'
    ]
    
    @classmethod
    def setup_tensorflow(cls):
        """Configure TensorFlow environment variables"""
        for key, value in cls.TF_CONFIG.items():
            os.environ[key] = value
    
    @classmethod
    def get_sample_size(cls, filetype: str, data_category: str) -> int:
        """Get appropriate sample size for given filetype and data category"""
        if data_category in ['long', 'short', 'mix_long_short']:
            return cls.SAMPLE_SIZES[filetype]['length_based']
        else:
            return cls.SAMPLE_SIZES[filetype]['count_based']
    
    @classmethod
    def get_data_file_path(cls, filetype: str, data_category: str) -> Path:
        """Get path to positive data file"""
        return cls.SORTED_DIR / f"arabidopsis_{filetype}_{data_category}.txt"
    
    @classmethod
    def get_negative_file_path(cls, filetype: str) -> Path:
        """Get path to negative data file"""
        return cls.SRC_DIR / f"arabidopsis_{filetype}_negative.txt"
    
    @classmethod
    def get_results_file_path(cls, model_name: str, data_category: str, 
                             filetype: str, batch_size: int) -> Path:
        """Get path for results CSV file"""
        filename = f"yyy_{model_name}_{data_category}_{filetype}_{batch_size}.csv"
        return cls.RESULTS_DIR / filename
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.RESULTS_DIR, cls.SAVED_MODELS_DIR, cls.WEIGHTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True) 
