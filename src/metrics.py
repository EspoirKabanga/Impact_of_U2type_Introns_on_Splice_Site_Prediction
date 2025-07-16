"""
Metrics calculation module for U2 Intron Splice Site Prediction
"""
import numpy as np
from typing import Tuple, Dict, List
import csv
from pathlib import Path
import logging

from .config import Config
from .utils import safe_divide

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Class for calculating classification metrics"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def calculate_confusion_matrix(self, predictions: np.ndarray, 
                                 labels: np.ndarray) -> Dict[str, int]:
        """
        Calculate confusion matrix components
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels (one-hot encoded)
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        # Convert predictions and labels to binary
        pred_binary = (predictions[:, 1] >= self.threshold).astype(int)
        true_binary = labels[:, 1].astype(int)
        
        tp = np.sum((pred_binary == 1) & (true_binary == 1))
        tn = np.sum((pred_binary == 0) & (true_binary == 0))
        fp = np.sum((pred_binary == 1) & (true_binary == 0))
        fn = np.sum((pred_binary == 0) & (true_binary == 1))
        
        return {
            'tp': int(tp), 'tn': int(tn), 
            'fp': int(fp), 'fn': int(fn)
        }
    
    def calculate_metrics(self, predictions: np.ndarray, 
                         labels: np.ndarray) -> Tuple[float, ...]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels (one-hot encoded)
            
        Returns:
            Tuple of (recall, precision, f1, fpr, fdr, specificity, sensitivity, mcc)
        """
        cm = self.calculate_confusion_matrix(predictions, labels)
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
        
        # Basic metrics with safe division
        recall = safe_divide(tp, tp + fn)  # Sensitivity/Recall
        precision = safe_divide(tp, tp + fp)
        specificity = safe_divide(tn, tn + fp)
        
        # Derived metrics
        f1 = safe_divide(2 * recall * precision, recall + precision)
        fpr = safe_divide(fp, fp + tn)  # False Positive Rate
        fdr = safe_divide(fp, tp + fp)  # False Discovery Rate
        sensitivity = recall  # Same as recall
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = safe_divide((tp * tn) - (fp * fn), mcc_denominator)
        
        return (recall, precision, f1, fpr, fdr, specificity, sensitivity, mcc)
    
    def get_metrics_dict(self, predictions: np.ndarray, 
                        labels: np.ndarray) -> Dict[str, float]:
        """Get metrics as a dictionary"""
        metrics = self.calculate_metrics(predictions, labels)
        metric_names = ['recall', 'precision', 'f1', 'fpr', 'fdr', 
                       'specificity', 'sensitivity', 'mcc']
        return dict(zip(metric_names, metrics))

class ResultsManager:
    """Class for managing experiment results"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.metrics_calculator = MetricsCalculator()
    
    def save_fold_metrics_to_csv(self, model_name: str, batch_size: int,
                                fold_metrics_list: List[Tuple[float, ...]],
                                avg_metrics: np.ndarray, std_metrics: np.ndarray,
                                data_category: str, filetype: str) -> None:
        """
        Save metrics results to CSV file
        
        Args:
            model_name: Name of the model
            batch_size: Batch size used
            fold_metrics_list: List of metrics for each fold
            avg_metrics: Average metrics across folds
            std_metrics: Standard deviation of metrics across folds
            data_category: Data category (e.g., 'short', 'long')
            filetype: File type ('donor' or 'acceptor')
        """
        # Ensure results directory exists
        self.config.ensure_directories()
        
        # Get output file path
        output_file = self.config.get_results_file_path(
            model_name, data_category, filetype, batch_size
        )
        
        try:
            with open(output_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                csv_writer.writerow(self.config.METRICS_COLUMNS)
                
                # Write fold metrics
                for fold_idx, fold_metrics in enumerate(fold_metrics_list, start=1):
                    row = [f'fold {fold_idx}'] + [f'{metric:.4f}' for metric in fold_metrics]
                    csv_writer.writerow(row)
                
                # Write summary statistics
                avg_row = ['Average'] + [f'{metric:.4f}' for metric in avg_metrics]
                std_row = ['Std Deviation'] + [f'{metric:.4f}' for metric in std_metrics]
                
                csv_writer.writerow(avg_row)
                csv_writer.writerow(std_row)
                
            logger.info(f"Results saved to {output_file}")
            
        except IOError as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
    
    def calculate_fold_summary(self, fold_metrics_list: List[Tuple[float, ...]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate summary statistics across folds
        
        Args:
            fold_metrics_list: List of metrics tuples for each fold
            
        Returns:
            Tuple of (average_metrics, std_metrics)
        """
        if not fold_metrics_list:
            raise ValueError("No fold metrics provided")
        
        fold_metrics_array = np.array(fold_metrics_list)
        avg_metrics = np.mean(fold_metrics_array, axis=0)
        std_metrics = np.std(fold_metrics_array, axis=0)
        
        return avg_metrics, std_metrics
    
    def log_metrics_summary(self, model_name: str, data_category: str, 
                           avg_metrics: np.ndarray, std_metrics: np.ndarray) -> None:
        """Log metrics summary"""
        metric_names = ['Recall', 'Precision', 'F1', 'FPR', 'FDR', 'Specificity', 'Sensitivity', 'MCC']
        
        logger.info(f"\n=== {model_name} - {data_category} Summary ===")
        for name, avg, std in zip(metric_names, avg_metrics, std_metrics):
            logger.info(f"{name}: {avg:.4f} ± {std:.4f}")
    
    def save_experiment_summary(self, results: Dict[str, Dict[str, any]], 
                               output_file: Path) -> None:
        """
        Save overall experiment summary
        
        Args:
            results: Nested dictionary with experiment results
            output_file: Path to save summary
        """
        try:
            with open(output_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                header = ['Model', 'Data_Category', 'Batch_Size'] + [
                    f'{metric}_{stat}' 
                    for metric in ['Recall', 'Precision', 'F1', 'Specificity', 'MCC']
                    for stat in ['Mean', 'Std']
                ]
                csv_writer.writerow(header)
                
                # Write results
                for model_name, model_results in results.items():
                    for data_cat, metrics in model_results.items():
                        row = [model_name, data_cat, metrics.get('batch_size', 64)]
                        
                        # Add key metrics with mean and std
                        key_indices = [0, 1, 2, 5, 7]  # recall, precision, f1, specificity, mcc
                        for idx in key_indices:
                            row.extend([
                                f"{metrics['avg'][idx]:.4f}",
                                f"{metrics['std'][idx]:.4f}"
                            ])
                        
                        csv_writer.writerow(row)
                        
            logger.info(f"Experiment summary saved to {output_file}")
            
        except IOError as e:
            logger.error(f"Error saving experiment summary: {e}")
            raise

class PerformanceTracker:
    """Class for tracking model performance during training"""
    
    def __init__(self):
        self.fold_results = []
        self.current_fold = 0
    
    def start_fold(self, fold_number: int):
        """Start tracking a new fold"""
        self.current_fold = fold_number
        logger.info(f"Starting fold {fold_number}")
    
    def record_fold_metrics(self, metrics: Tuple[float, ...], 
                           model_name: str, data_category: str):
        """Record metrics for current fold"""
        metrics_dict = {
            'fold': self.current_fold,
            'model': model_name,
            'data_category': data_category,
            'metrics': metrics
        }
        self.fold_results.append(metrics_dict)
        
        # Log fold completion
        metric_names = ['Recall', 'Precision', 'F1', 'FPR', 'FDR', 'Specificity', 'Sensitivity', 'MCC']
        metrics_str = " | ".join([f"{name}: {value:.4f}" for name, value in zip(metric_names, metrics)])
        logger.info(f"Fold {self.current_fold} completed - {metrics_str}")
    
    def get_fold_metrics_list(self, model_name: str = None, 
                             data_category: str = None) -> List[Tuple[float, ...]]:
        """Get metrics list filtered by model and/or data category"""
        filtered_results = self.fold_results
        
        if model_name:
            filtered_results = [r for r in filtered_results if r['model'] == model_name]
        
        if data_category:
            filtered_results = [r for r in filtered_results if r['data_category'] == data_category]
        
        return [r['metrics'] for r in filtered_results]
    
    def reset(self):
        """Reset tracker for new experiment"""
        self.fold_results = []
        self.current_fold = 0 