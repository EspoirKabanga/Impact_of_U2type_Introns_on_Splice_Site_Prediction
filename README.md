# Impact of U2-type Introns on Splice Site Prediction

## Project Overview

This project investigates **two key hypotheses about splice site prediction in Arabidopsis thaliana**:

1. **Short introns may enhance prediction effectiveness** due to reduced spatial complexity between donor and acceptor sites
2. **Sequences with multiple introns provide richer context** for accurate splicing event prediction

## Research Approach

### Data Categories
- **Intron Length**: Short (<90bp) vs Long (≥90bp) introns
- **Intron Count**: Single vs Multiple introns per sequence
- **Mixed Datasets**: Combined categories for comprehensive analysis

### Models Tested
Four state-of-the-art CNN architectures:
- **IntSplicer**: Multi-layer CNN with progressive filtering
- **SpliceRover**: Deep architecture with varied kernel sizes  
- **SpliceFinder**: Lightweight single-layer CNN
- **DeepSplicer**: Three-layer CNN with consistent filtering

### Evaluation
- **5-fold cross-validation** for robust performance assessment
- **Comprehensive metrics**: Sensitivity, Precision, F1-score, MCC, Specificity
- **Statistical analysis** with mean and standard deviation reporting

## Key Findings

- **Short introns improved acceptor site prediction** but not donor sites
- **Multiple introns enhanced prediction for both donor and acceptor sites**
- Results provide insights into the spatial and contextual factors affecting splice site recognition

## Code Structure

```
src/
├── __init__.py              # Package initialization
├── config.py               # Centralized configuration (paths, parameters, sample sizes)
├── utils.py                # Logging, random seeds, file operations
├── data_preprocessing.py   # DNA sequence encoding and data loading
├── models.py               # CNN model architectures (IntSplicer, SpliceRover, etc.)
├── metrics.py              # Performance metrics calculation and CSV export
├── training.py             # Cross-validation training pipeline
└── main.py                 # Command-line interface and experiment execution
```

## Installation & Usage

### Prerequisites
```bash
# Required Python packages
tensorflow>=2.8.0
numpy
scikit-learn
pandas
```

### Quick Start
```bash
# Quick test run (subset of data and models)
python -m src.main --quick-test

# Full experiments (all models and data categories)
python -m src.main

# Specific experiments
python -m src.main --filetypes donor --models IntSplicer SpliceRover
python -m src.main --data-categories short long --batch-sizes 64
```

### Command-Line Options
- `--quick-test`: Fast test with subset of data/models
- `--filetypes`: Choose 'donor', 'acceptor', or both
- `--models`: Select specific models to test
- `--data-categories`: Choose data categories to analyze
- `--batch-sizes`: Set batch sizes for training
- `--epochs`: Number of training epochs (default: 30)

## Configuration

Key parameters in `config.py`:
- **Sample sizes**: 18,859 (donor) / 19,322 (acceptor) for length-based categories
- **Sample sizes**: 13,987 (donor) / 14,112 (acceptor) for count-based categories
- **Sequence length**: 402bp DNA sequences
- **Encoding**: One-hot encoding (A, C, G, T)
- **Cross-validation**: 5-fold stratified split

## Output

Results are saved as CSV files with format:
```
results/yyy_{model}_{category}_{filetype}_{batch_size}.csv
```

Each CSV contains:
- Individual fold performance metrics
- Average performance across folds
- Standard deviation statistics

## Reproducibility

- Pinned environment (install exact versions):
  - tensorflow==2.10.1
  - numpy==1.23.5
  - scikit-learn==1.2.2
  - pandas==1.5.3
  - matplotlib==3.7.1
  - statsmodels==0.13.5
  - scipy==1.10.1

- Random seeds and determinism:
  - Seed constant: `Config.RANDOM_SEED = 42` (see `src/config.py`).
  - Seeds applied via `utils.set_random_seeds(42)` in `main.setup_environment`.
  - Cross-validation uses `StratifiedKFold(..., random_state=42)`; train/test split uses `train_test_split(..., random_state=42, stratify=y)`.

- Train/val/test split:
  - Single 80:20 stratified split with seed 42 (see `src/training.py`).
  - Validation is taken from training via `validation_split=0.1` where applicable.

- Pretrained weights / saved models:
  - Best weights: `best_weights/` (per fold, latest `.h5`).
  - Full models + history: `saved_models/`.
  - To evaluate external pretrained models on the exact 80:20 test split (seed=42), use:
    - `python src/evaluate_test.py --models-dir /path/to/saved_models`

## Data Requirements

Expected data files in project root:
```
sorted/arabidopsis_{filetype}_{category}.txt    # Positive sequences
arabidopsis_{filetype}_negative.txt             # Negative sequences
```

Where:
- `{filetype}`: 'donor' or 'acceptor'
- `{category}`: 'short', 'long', 'mix_long_short', 'single', 'multiple', 'mix_single_multiple' 