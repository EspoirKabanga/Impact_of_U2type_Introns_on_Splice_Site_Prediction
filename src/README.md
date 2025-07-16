# U2 Intron Splice Site Prediction - Refactored Code

This directory contains a refactored and improved version of the U2 Intron Splice Site Prediction codebase with better software engineering practices, modularity, and maintainability.

## 🚀 Key Improvements

### ✅ **Better Code Organization**
- **Modular design** with clear separation of concerns
- **Configuration management** centralized in `config.py`
- **Factory pattern** for model creation
- **Class-based architecture** for better encapsulation

### ✅ **Enhanced Error Handling**
- Comprehensive error handling with try/catch blocks
- Input validation for sequences and parameters
- Graceful failure with informative error messages

### ✅ **Improved Configuration**
- All hardcoded values moved to `config.py`
- Easy to modify parameters without changing code
- Path management with `pathlib`
- Environment variable configuration for TensorFlow

### ✅ **Better Logging & Monitoring**
- Comprehensive logging throughout the pipeline
- Progress tracking and performance monitoring
- Detailed metrics reporting

### ✅ **Enhanced Usability**
- Command-line interface with argument parsing
- Quick test mode for development
- Flexible experiment configuration

## 📁 File Structure

```
src/
├── __init__.py              # Package initialization
├── README.md               # This file
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── data_preprocessing.py   # Data loading and preprocessing
├── models.py               # Neural network model definitions
├── metrics.py              # Metrics calculation and results
├── training.py             # Training pipeline and cross-validation
└── main.py                 # Main execution script
```

## 📋 Module Overview

### `config.py`
- **Purpose**: Centralized configuration management
- **Features**: 
  - All parameters in one place
  - Path management
  - TensorFlow environment setup
  - Easy parameter modification

### `utils.py`
- **Purpose**: Common utility functions
- **Features**:
  - Logging setup
  - Random seed management
  - File operations with error handling
  - Data validation functions

### `data_preprocessing.py`
- **Purpose**: Data loading and preprocessing
- **Features**:
  - Robust data loading with validation
  - One-hot encoding with error checking
  - Dataset balancing and sampling
  - Duplicate removal

### `models.py`
- **Purpose**: Neural network model definitions
- **Features**:
  - Abstract base class for models
  - Factory pattern for model creation
  - Clean architecture definitions
  - Consistent compilation settings

### `metrics.py`
- **Purpose**: Metrics calculation and results management
- **Features**:
  - Comprehensive classification metrics
  - Safe division operations
  - CSV results export
  - Performance tracking

### `training.py`
- **Purpose**: Training pipeline and cross-validation
- **Features**:
  - Modular training components
  - K-fold cross-validation
  - Callback management
  - Model artifact saving

### `main.py`
- **Purpose**: Main execution script
- **Features**:
  - Command-line interface
  - Flexible experiment configuration
  - Quick test mode
  - Comprehensive logging

## 🚀 Usage Examples

### Basic Usage
```bash
# Run all experiments (default behavior)
python -m src.main

# Quick test with minimal configuration
python -m src.main --quick-test

# Run specific experiments
python -m src.main --filetypes donor --models IntSplicer SpliceFinder

# Run with custom parameters
python -m src.main --epochs 50 --batch-sizes 32 64 --n-folds 10
```

### Advanced Usage
```bash
# Run specific data categories
python -m src.main --data-categories short long --models DeepSplicer

# Custom output directory
python -m src.main --output-dir /path/to/results --log-level DEBUG

# Run only acceptor sites with specific models
python -m src.main --filetypes acceptor --models IntSplicer SpliceRover
```

### Command Line Options
```bash
python -m src.main --help
```

**Available Options:**
- `--filetypes`: Choose 'donor', 'acceptor', or both
- `--data-categories`: Select data categories to test
- `--models`: Choose specific models to train
- `--batch-sizes`: Specify batch sizes
- `--epochs`: Number of training epochs
- `--n-folds`: Number of cross-validation folds
- `--log-level`: Logging verbosity
- `--output-dir`: Custom output directory
- `--quick-test`: Run minimal test

## 🔧 Configuration

### Modifying Parameters
Edit `config.py` to change:
- **Sequence parameters**: Length, input channels
- **Training parameters**: Epochs, batch sizes, learning rate
- **Model parameters**: Architecture details
- **File paths**: Data locations, output directories

### Example Configuration Changes
```python
# In config.py
class Config:
    EPOCHS = 50          # Increase training epochs
    BATCH_SIZES = [32, 64, 128]  # Test multiple batch sizes
    LEARNING_RATE = 0.0005       # Lower learning rate
    N_FOLDS = 10         # More cross-validation folds
```

## 📊 Output Structure

```
results/
├── yyy_ModelName_DataCategory_FileType_BatchSize.csv  # Individual results
├── experiment_summary.csv                             # Overall summary
└── splice_prediction.log                             # Execution log

saved_models/
├── ModelName_zzz_DataCategory_FileType_BatchSize_fold_N_DataCategory/
│   ├── ModelName.json                # Model architecture
│   ├── ModelName.h5                  # Model weights
│   └── ModelName_history.csv         # Training history
```

## 🎯 Key Benefits

### **For Researchers**
- **Easy experimentation** with different configurations
- **Reproducible results** with proper seed management
- **Comprehensive metrics** for analysis
- **Flexible parameter tuning**

### **For Developers**
- **Clean, maintainable code** structure
- **Proper error handling** and logging
- **Extensible architecture** for new models
- **Good software engineering practices**

### **For Users**
- **Simple command-line interface**
- **Clear documentation** and examples
- **Flexible configuration** options
- **Robust execution** with error recovery

## 🔄 Migration from Original Code

To use the refactored code instead of the original:

1. **Replace the main execution**:
   ```bash
   # Instead of: python Main.py
   python -m src.main
   ```

2. **Update any custom scripts** to import from the new modules:
   ```python
   # Old
   import mymodels as mdl
   
   # New
   from src.models import ModelFactory
   model = ModelFactory.create_model('IntSplicer')
   ```

3. **Use the new configuration system**:
   ```python
   # Old
   filetype = 'donor'
   
   # New
   from src.config import Config
   config = Config()
   filetype = config.FILETYPES[0]
   ```

## 🧪 Testing

```bash
# Quick functionality test
python -m src.main --quick-test

# Test specific components
python -c "from src.models import ModelFactory; print('Models OK')"
python -c "from src.config import Config; print('Config OK')"
```

## 🔮 Future Enhancements

The refactored architecture makes it easy to add:
- **New model architectures** via the factory pattern
- **Different data preprocessing** methods
- **Advanced metrics** and visualizations
- **Hyperparameter optimization**
- **Distributed training** support

---

This refactored codebase maintains all the functionality of the original while providing a much cleaner, more maintainable, and extensible foundation for splice site prediction research. 