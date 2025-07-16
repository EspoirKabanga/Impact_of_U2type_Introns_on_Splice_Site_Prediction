import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from utils import setup_logging, set_random_seeds
from models import ModelFactory
from training import ExperimentRunner
from metrics import ResultsManager

logger = logging.getLogger(__name__)

def setup_environment(config: Config) -> None:
    """Setup the environment for training"""
    # Setup TensorFlow
    config.setup_tensorflow()
    
    # Setup random seeds
    set_random_seeds(config.RANDOM_SEED)
    
    # Ensure directories exist
    config.ensure_directories()
    
    logger.info("Environment setup completed")

def run_experiments(config: Config, filetypes: List[str] = None, 
                   data_categories: List[str] = None,
                   models: List[str] = None) -> Dict[str, Any]:
    """
    Run all experiments with specified parameters
    
    Args:
        config: Configuration object
        filetypes: List of filetypes to test (default: all)
        data_categories: List of data categories to test (default: all)
        models: List of models to test (default: all)
        
    Returns:
        Dictionary with all experiment results
    """
    # Use defaults if not specified
    filetypes = filetypes or config.FILETYPES
    data_categories = data_categories or config.DATASET_CATEGORIES
    model_names = models or ModelFactory.list_available_models()
    
    logger.info(f"Running experiments for:")
    logger.info(f"  Filetypes: {filetypes}")
    logger.info(f"  Data categories: {data_categories}")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Batch sizes: {config.BATCH_SIZES}")
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner(config)
    results_manager = ResultsManager(config)
    
    all_results = {}
    
    # Run experiments for each filetype
    for filetype in filetypes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting experiments for filetype: {filetype}")
        logger.info(f"{'='*80}")
        
        filetype_results = {}
        
        for data_category in data_categories:
            logger.info(f"\nProcessing data category: {data_category}")
            
            for batch_size in config.BATCH_SIZES:
                logger.info(f"Using batch size: {batch_size}")
                
                for model_name in model_names:
                    try:
                        # Create model instance
                        model = ModelFactory.create_model(model_name, config)
                        
                        # Run experiment
                        experiment_result = experiment_runner.run_single_experiment(
                            model, data_category, filetype, batch_size
                        )
                        
                        # Store results
                        key = f"{model_name}_{data_category}_{batch_size}"
                        filetype_results[key] = experiment_result
                        
                        logger.info(f"Completed: {model_name} - {data_category} - batch {batch_size}")
                        
                    except Exception as e:
                        logger.error(f"Error in experiment {model_name} - {data_category}: {e}")
                        continue
        
        all_results[filetype] = filetype_results
    
    # Save overall experiment summary
    summary_file = config.RESULTS_DIR / "experiment_summary.csv"
    try:
        # Flatten results for summary
        flattened_results = {}
        for filetype, filetype_results in all_results.items():
            for key, result in filetype_results.items():
                summary_key = f"{filetype}_{key}"
                flattened_results[summary_key] = {
                    'avg': result['avg_metrics'],
                    'std': result['std_metrics'],
                    'batch_size': result['batch_size']
                }
        
        results_manager.save_experiment_summary(flattened_results, summary_file)
        
    except Exception as e:
        logger.error(f"Error saving experiment summary: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("="*80)
    
    return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="U2 Intron Splice Site Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to custom configuration file'
    )
    
    # Data arguments
    parser.add_argument(
        '--filetypes', nargs='+', choices=['donor', 'acceptor'],
        help='Filetypes to process (default: both)'
    )
    
    parser.add_argument(
        '--data-categories', nargs='+',
        choices=['short', 'long', 'mix_long_short', 'multiple', 'single', 'mix_single_multiple'],
        help='Data categories to process (default: all)'
    )
    
    # Model arguments
    parser.add_argument(
        '--models', nargs='+',
        choices=['IntSplicer', 'SpliceRover', 'SpliceFinder', 'DeepSplicer'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--batch-sizes', nargs='+', type=int,
        help='Batch sizes to use (default: from config)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help='Number of cross-validation folds'
    )
    
    # Other arguments
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help='Logging level'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Run quick test with single model and data category'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting U2 Intron Splice Site Prediction")
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.n_folds:
        config.N_FOLDS = args.n_folds
    if args.batch_sizes:
        config.BATCH_SIZES = args.batch_sizes
    if args.output_dir:
        config.RESULTS_DIR = Path(args.output_dir)
        config.SAVED_MODELS_DIR = Path(args.output_dir) / "saved_models"
    
    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode")
        filetypes = ['donor']
        data_categories = ['short']
        models = ['SpliceFinder']
        config.EPOCHS = 2
        config.N_FOLDS = 2
    else:
        filetypes = args.filetypes
        data_categories = args.data_categories
        models = args.models
    
    try:
        # Setup environment
        setup_environment(config)
        
        # Run experiments
        results = run_experiments(
            config=config,
            filetypes=filetypes,
            data_categories=data_categories,
            models=models
        )
        
        logger.info("Experiment completed successfully!")
        
        # Print summary
        logger.info(f"\nResults saved to: {config.RESULTS_DIR}")
        logger.info(f"Models saved to: {config.SAVED_MODELS_DIR}")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
