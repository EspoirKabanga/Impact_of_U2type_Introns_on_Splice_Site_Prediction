"""
Training module for U2 Intron Splice Site Prediction
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import glob
import os
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .utils import (
    create_directory_safe, 
    remove_directory_safe, 
    get_model_project_name
)
from .metrics import MetricsCalculator, PerformanceTracker
from .models import BaseModel

logger = logging.getLogger(__name__)

class ModelCallbacks:
    """Class for managing training callbacks"""
    
    def __init__(self, config: Config = Config):
        self.config = config
    
    def create_callbacks(self, project_name: str, data_category: str, 
                        filetype: str) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks for model training
        
        Args:
            project_name: Unique project name for this training run
            data_category: Data category being used
            filetype: File type ('donor' or 'acceptor')
            
        Returns:
            List of configured callbacks
        """
        # Create weights directory
        weights_dir = self.config.WEIGHTS_DIR / f"{data_category}_{filetype}"
        create_directory_safe(weights_dir)
        
        # Model checkpoint callback
        checkpoint_filepath = weights_dir / f"{project_name}_weights.{{epoch:02d}}-{{val_loss:.2f}}.h5"
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            monitor=self.config.MONITOR_METRIC,
            mode='min',
            save_freq='epoch',
            save_best_only=True,
            verbose=1
        )
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.config.MONITOR_METRIC,
            verbose=1,
            patience=self.config.PATIENCE,
            restore_best_weights=True
        )
        
        # Learning rate reduction callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.MONITOR_METRIC,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        return [model_checkpoint, early_stopping, reduce_lr]

class ModelManager:
    """Class for managing model saving and loading"""
    
    def __init__(self, config: Config = Config):
        self.config = config
    
    def save_model_artifacts(self, model: keras.Model, history: keras.callbacks.History,
                           project_name: str, data_category: str) -> None:
        """
        Save model artifacts including weights, architecture, and training history
        
        Args:
            model: Trained Keras model
            history: Training history
            project_name: Project name for organizing files
            data_category: Data category for organizing files
        """
        # Create model directory
        model_dir = self.config.SAVED_MODELS_DIR / f"{project_name}_{data_category}"
        create_directory_safe(model_dir)
        
        try:
            # Save model architecture as JSON
            model_json = model.to_json()
            with open(model_dir / f"{project_name}.json", 'w') as json_file:
                json_file.write(model_json)
            
            # Save model weights
            model.save(model_dir / f"{project_name}.h5")
            
            # Save training history
            if history and hasattr(history, 'history'):
                import pandas as pd
                hist_df = pd.DataFrame(history.history)
                hist_csv_file = model_dir / f"{project_name}_history.csv"
                hist_df.to_csv(hist_csv_file, index=False)
            
            logger.info(f"Model artifacts saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    
    def load_best_weights(self, weights_dir: Path, model: keras.Model) -> None:
        """
        Load the best weights from the weights directory
        
        Args:
            weights_dir: Directory containing weight files
            model: Model to load weights into
        """
        try:
            # Find all weight files
            weight_files = list(weights_dir.glob("*.h5"))
            
            if not weight_files:
                logger.warning(f"No weight files found in {weights_dir}")
                return
            
            # Get the most recent weight file (best from training)
            latest_file = max(weight_files, key=lambda x: x.stat().st_ctime)
            
            # Load weights
            model.load_weights(str(latest_file))
            logger.info(f"Loaded weights from {latest_file}")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise

class CrossValidationTrainer:
    """Class for handling k-fold cross-validation training"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.callbacks_manager = ModelCallbacks(config)
        self.model_manager = ModelManager(config)
        self.metrics_calculator = MetricsCalculator()
        self.performance_tracker = PerformanceTracker()
    
    def create_kfold_splitter(self) -> StratifiedKFold:
        """Create stratified k-fold splitter"""
        return StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
    
    def train_single_fold(self, model: BaseModel, train_data: Dict[str, np.ndarray],
                         project_name: str, data_category: str, filetype: str) -> keras.callbacks.History:
        """
        Train model for a single fold
        
        Args:
            model: Model instance to train
            train_data: Dictionary with training and validation data
            project_name: Project name for organizing files
            data_category: Data category
            filetype: File type
            
        Returns:
            Training history
        """
        # Get fresh model instance
        keras_model = model.get_model()
        
        # Create callbacks
        callbacks = self.callbacks_manager.create_callbacks(
            project_name, data_category, filetype
        )
        
        # Train model
        logger.info(f"Training {project_name}")
        history = keras_model.fit(
            train_data['X_train'], train_data['y_train'],
            epochs=self.config.EPOCHS,
            validation_data=(train_data['X_val'], train_data['y_val']),
            batch_size=train_data.get('batch_size', 64),
            callbacks=callbacks,
            verbose=2
        )
        
        return history
    
    def evaluate_model(self, model: keras.Model, test_data: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate model on test data
        
        Args:
            model: Trained model
            test_data: Test dataset
            
        Returns:
            Tuple of (loss, accuracy, predictions)
        """
        loss, accuracy = model.evaluate(test_data['X_test'], test_data['y_test'], verbose=0)
        predictions = model.predict(test_data['X_test'], verbose=0)
        
        return loss, accuracy, predictions
    
    def train_cross_validation(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                              test_data: Dict[str, np.ndarray], data_category: str,
                              filetype: str, batch_size: int = 64) -> List[Tuple[float, ...]]:
        """
        Perform complete k-fold cross-validation training
        
        Args:
            model: Model to train
            X: Training features
            y: Training labels  
            test_data: Test dataset
            data_category: Data category
            filetype: File type
            batch_size: Batch size for training
            
        Returns:
            List of metrics for each fold
        """
        logger.info(f"Starting {self.config.N_FOLDS}-fold cross-validation for {model.get_model().name}")
        
        # Create k-fold splitter
        kf = self.create_kfold_splitter()
        
        fold_metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            logger.info(f"=== Fold {fold}/{self.config.N_FOLDS} ===")
            
            # Create weights directory for this fold
            weights_dir = self.config.WEIGHTS_DIR / f"{data_category}_{filetype}"
            create_directory_safe(weights_dir)
            
            try:
                # Generate project name
                project_name = get_model_project_name(
                    model.get_model().name, data_category, filetype, batch_size, fold
                )
                
                # Prepare fold data
                from .data_preprocessing import DataLoader
                data_loader = DataLoader(self.config)
                
                train_data = data_loader.prepare_training_data(X, y, (train_idx, val_idx))
                train_data['batch_size'] = batch_size
                
                logger.info(f"Train size: {len(train_data['X_train'])}, "
                           f"Validation size: {len(train_data['X_val'])}, "
                           f"Test size: {len(test_data['X_test'])}")
                
                # Train model
                history = self.train_single_fold(
                    model, train_data, project_name, data_category, filetype
                )
                
                # Load best weights
                self.model_manager.load_best_weights(weights_dir, model.get_model())
                
                # Evaluate on test set
                loss, accuracy, predictions = self.evaluate_model(model.get_model(), test_data)
                
                # Calculate metrics
                fold_metrics = self.metrics_calculator.calculate_metrics(
                    predictions, test_data['y_test']
                )
                fold_metrics_list.append(fold_metrics)
                
                # Track performance
                self.performance_tracker.start_fold(fold)
                self.performance_tracker.record_fold_metrics(
                    fold_metrics, model.get_model().name, data_category
                )
                
                # Save model artifacts
                self.model_manager.save_model_artifacts(
                    model.get_model(), history, project_name, data_category
                )
                
                logger.info(f"Fold {fold} completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                raise
                
            finally:
                # Cleanup
                keras.backend.clear_session()
                remove_directory_safe(weights_dir)
        
        logger.info(f"Cross-validation completed for {model.get_model().name}")
        return fold_metrics_list

class ExperimentRunner:
    """Class for running complete experiments"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CrossValidationTrainer(config)
    
    def run_single_experiment(self, model: BaseModel, data_category: str,
                             filetype: str, batch_size: int = 64) -> Dict[str, any]:
        """
        Run a single experiment (model + data category + filetype)
        
        Args:
            model: Model to train
            data_category: Data category
            filetype: File type
            batch_size: Batch size
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting experiment: {model.get_model().name} - {data_category} - {filetype}")
        logger.info(f"{'='*60}")
        
        # Load data
        from .data_preprocessing import DataLoader
        data_loader = DataLoader(self.config)
        
        X, y = data_loader.load_dataset(filetype, data_category)
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_SEED, stratify=y
        )
        
        # Prepare test data
        test_data = data_loader.prepare_test_data(X_test, y_test)
        
        # Run cross-validation
        fold_metrics_list = self.trainer.train_cross_validation(
            model, X_train, y_train, test_data, data_category, filetype, batch_size
        )
        
        # Calculate summary statistics
        from .metrics import ResultsManager
        results_manager = ResultsManager(self.config)
        avg_metrics, std_metrics = results_manager.calculate_fold_summary(fold_metrics_list)
        
        # Save results
        results_manager.save_fold_metrics_to_csv(
            model.get_model().name, batch_size, fold_metrics_list,
            avg_metrics, std_metrics, data_category, filetype
        )
        
        # Log summary
        results_manager.log_metrics_summary(
            model.get_model().name, data_category, avg_metrics, std_metrics
        )
        
        return {
            'model_name': model.get_model().name,
            'data_category': data_category,
            'filetype': filetype,
            'batch_size': batch_size,
            'fold_metrics': fold_metrics_list,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        } 