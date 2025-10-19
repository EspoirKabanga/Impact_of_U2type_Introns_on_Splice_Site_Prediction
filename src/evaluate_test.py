import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reduce TensorFlow verbosity before import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from data_preprocessing import DataLoader
from metrics import MetricsCalculator


logger = logging.getLogger("evaluate_test")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


CATEGORY_CHOICES = [
    'short', 'long', 'mix_long_short', 'multiple', 'single', 'mix_single_multiple'
]

FILETYPE_CHOICES = ['donor', 'acceptor']


def parse_model_metadata(model_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort parse of filetype and data_category from filename.
    Returns (filetype, data_category) or (None, None) if not found.
    """
    name = model_path.stem.lower()
    filetype = None
    for ft in FILETYPE_CHOICES:
        if ft in name:
            filetype = ft
            break

    data_category = None
    for cat in CATEGORY_CHOICES:
        if cat in name:
            data_category = cat
            break

    return filetype, data_category


def evaluate_single_model(
    model_file: Path,
    filetype: str,
    data_category: str,
    config: Config,
) -> Dict[str, object]:
    """Load a saved Keras model and evaluate it on the test split.
    Returns a dict of metrics and identifiers suitable for CSV writing.
    """
    # Load dataset and split with seed 42
    loader = DataLoader(config)
    X, y = loader.load_dataset(filetype=filetype, data_category=data_category)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    # Prepare test data
    test_data = loader.prepare_test_data(X_test, y_test)

    # Load model
    try:
        model = tf.keras.models.load_model(str(model_file))
    except Exception as e:
        logger.error(f"Failed to load model {model_file}: {e}")
        raise

    # Predict
    predictions = model.predict(test_data['X_test'], verbose=0)

    # Metrics
    calculator = MetricsCalculator()
    metrics_tuple = calculator.calculate_metrics(predictions, test_data['y_test'])
    metric_names = ['recall', 'precision', 'f1', 'fpr', 'fdr', 'specificity', 'sensitivity', 'mcc', 'auroc', 'auprc']
    metrics = dict(zip(metric_names, [float(x) for x in metrics_tuple]))

    result: Dict[str, object] = {
        'model_file': model_file.name,
        'keras_model_name': getattr(model, 'name', ''),
        'filetype': filetype,
        'data_category': data_category,
        **metrics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved models on consistent 80:20 test split (seed=42)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--models-dir', type=str,
        default='/home/youngmin/seonil/src/saved_models',
        help='Directory containing saved Keras model files (*.h5 or SavedModel dirs)'
    )
    parser.add_argument(
        '--filetypes', nargs='+', choices=FILETYPE_CHOICES, default=FILETYPE_CHOICES,
        help='Filetypes to evaluate'
    )
    parser.add_argument(
        '--data-categories', nargs='+', choices=CATEGORY_CHOICES, default=['short', 'long', 'mix_long_short'],
        help='Data categories to evaluate'
    )
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = Config()
    config.setup_tensorflow()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return

    # Collect model files recursively (.h5/.keras files or SavedModel directories)
    model_paths: List[Path] = []
    savedmodel_dirs: set = set()
    for p in models_dir.rglob('*'):
        try:
            if p.is_file() and p.suffix.lower() in ['.h5', '.keras']:
                model_paths.append(p)
            elif p.is_file() and p.name == 'saved_model.pb':
                savedmodel_dirs.add(p.parent)
        except Exception:
            continue
    model_paths.extend(sorted(savedmodel_dirs))

    if not model_paths:
        logger.warning(f"No model files found in {models_dir}")
        return

    logger.info(f"Found {len(model_paths)} model(s) for evaluation")

    results: List[Dict[str, object]] = []

    for model_path in model_paths:
        inferred_filetype, inferred_category = parse_model_metadata(model_path)

        # Try to evaluate across requested combos; prefer inferred when available
        target_filetypes = [inferred_filetype] if inferred_filetype in args.filetypes else args.filetypes
        target_categories = [inferred_category] if inferred_category in args.data_categories else args.data_categories

        for ft in target_filetypes:
            for cat in target_categories:
                if ft is None or cat is None:
                    continue
                try:
                    logger.info(f"Evaluating {model_path.name} on {ft} / {cat}")
                    res = evaluate_single_model(model_path, ft, cat, config)
                    results.append(res)
                    logger.info(
                        f"{model_path.name} | {ft}/{cat} | F1-score={res['f1']:.4f}, MCC={res['mcc']:.4f}, AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_path} on {ft}/{cat}: {e}")
                    continue

    if results:
        out_dir = models_dir
        out_file = out_dir / 'evaluation_results.csv'
        df = pd.DataFrame(results)
        # Order columns
        cols = ['model_file', 'keras_model_name', 'filetype', 'data_category', 'recall', 'precision', 'f1', 'fpr', 'fdr', 'specificity', 'sensitivity', 'mcc', 'auroc', 'auprc']
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(out_file, index=False)
        logger.info(f"Saved evaluation results to {out_file}")
    else:
        logger.warning("No successful evaluations to record.")


if __name__ == '__main__':
    main()


