import tensorflow as tf
from typing import Dict, Any
from abc import ABC, abstractmethod

from config import Config

class BaseModel(ABC):
    """Abstract base class for splice site prediction models"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.model = None
        
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return the model architecture"""
        pass
    
    def compile_model(self, model: tf.keras.Model, 
                     learning_rate: float = None) -> tf.keras.Model:
        """Compile model with standard settings"""
        lr = learning_rate or self.config.LEARNING_RATE
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def get_model(self, learning_rate: float = None) -> tf.keras.Model:
        """Get compiled model"""
        if self.model is None:
            self.model = self.build_model()
            self.model = self.compile_model(self.model, learning_rate)
        return self.model

class IntSplicerModel(BaseModel):
    """IntSplicer model architecture - Deep CNN with progressive filter increase"""
    
    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name='IntSplicer')
        
        # First convolutional block
        model.add(tf.keras.layers.Conv1D(
            filters=64, kernel_size=10, strides=4, padding='same',
            batch_input_shape=(None, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS),
            activation='relu', name='conv1'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        
        # Second convolutional block
        model.add(tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, strides=1, padding='same',
            activation='relu', name='conv2'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Third convolutional block
        model.add(tf.keras.layers.Conv1D(
            filters=256, kernel_size=3, strides=1, padding='same',
            activation='relu', name='conv3'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        
        # Fourth convolutional block
        model.add(tf.keras.layers.Conv1D(
            filters=512, kernel_size=2, strides=1, padding='same',
            activation='relu', name='conv4'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.config.N_CLASSES, activation='softmax', name='output'))
        
        return model

class SpliceRoverModel(BaseModel):
    """SpliceRover model architecture - 5-layer CNN with consistent kernel sizes"""
    
    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name='SpliceRover')
        
        # First convolutional layer
        model.add(tf.keras.layers.Conv1D(
            filters=70, kernel_size=9, strides=4, padding='same',
            batch_input_shape=(None, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS),
            activation='relu', name='conv1'
        ))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Second convolutional layer
        model.add(tf.keras.layers.Conv1D(
            filters=100, kernel_size=7, strides=1, padding='same',
            activation='relu', name='conv2'
        ))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Third convolutional layer with pooling
        model.add(tf.keras.layers.Conv1D(
            filters=100, kernel_size=7, strides=1, padding='same',
            activation='relu', name='conv3'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=1))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Fourth convolutional layer with pooling
        model.add(tf.keras.layers.Conv1D(
            filters=200, kernel_size=7, strides=1, padding='same',
            activation='relu', name='conv4'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=1))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Fifth convolutional layer with pooling
        model.add(tf.keras.layers.Conv1D(
            filters=250, kernel_size=7, strides=1, padding='same',
            activation='relu', name='conv5'
        ))
        model.add(tf.keras.layers.MaxPool1D(pool_size=4, strides=1))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.config.N_CLASSES, activation='softmax', name='output'))
        
        return model

class SpliceFinderModel(BaseModel):
    """SpliceFinder model architecture - Simple 1-layer CNN"""
    
    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name='SpliceFinder')
        
        # Single convolutional layer
        model.add(tf.keras.layers.Conv1D(
            filters=50, kernel_size=9, strides=1, padding='same',
            batch_input_shape=(None, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS),
            activation='relu', name='conv1'
        ))
        
        # Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(self.config.N_CLASSES, activation='softmax', name='output'))
        
        return model

class DeepSplicerModel(BaseModel):
    """DeepSplicer model architecture - 3-layer CNN with consistent filters"""
    
    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name='DeepSplicer')
        
        # Three convolutional layers with same parameters
        model.add(tf.keras.layers.Conv1D(
            filters=50, kernel_size=9, strides=1, padding='same',
            batch_input_shape=(None, self.config.SEQUENCE_LENGTH, self.config.INPUT_CHANNELS),
            activation='relu', name='conv1'
        ))
        
        model.add(tf.keras.layers.Conv1D(
            filters=50, kernel_size=9, strides=1, padding='same',
            activation='relu', name='conv2'
        ))
        
        model.add(tf.keras.layers.Conv1D(
            filters=50, kernel_size=9, strides=1, padding='same',
            activation='relu', name='conv3'
        ))
        
        # Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(self.config.N_CLASSES, activation='softmax', name='output'))
        
        return model

class ModelFactory:
    """Factory class for creating model instances"""
    
    _models = {
        'IntSplicer': IntSplicerModel,
        'SpliceRover': SpliceRoverModel,
        'SpliceFinder': SpliceFinderModel,
        'DeepSplicer': DeepSplicerModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Config = Config) -> BaseModel:
        """Create model instance by name"""
        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        return cls._models[model_name](config)
    
    @classmethod
    def get_all_models(cls, config: Config = Config) -> Dict[str, BaseModel]:
        """Get all available models"""
        return {name: cls.create_model(name, config) for name in cls._models.keys()}
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model names"""
        return list(cls._models.keys())

# Convenience function for backward compatibility
def create_all_models(config: Config = Config) -> list:
    """Create all model instances for experiments"""
    factory = ModelFactory()
    models = []
    
    for model_name in factory.list_available_models():
        model_instance = factory.create_model(model_name, config)
        models.append(model_instance.get_model())
    
    return models 
