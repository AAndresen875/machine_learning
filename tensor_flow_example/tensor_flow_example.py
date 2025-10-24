"""
TensorFlow/Keras Deep Learning Pipeline
This module demonstrates building and training neural networks using TensorFlow
and Keras with best practices for production environments.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential, Model
import numpy as np
from typing import Tuple, Dict, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class NeuralNetworkPipeline:
    """
    A complete neural network pipeline using TensorFlow/Keras.
    
    Args:
        input_shape (Tuple): Shape of input data
        num_classes (int): Number of output classes
        model_type (str): Type of model architecture
    
    Attributes:
        model: Keras model
        history: Training history
    """
    
    def __init__(
        self,
        input_shape: Tuple,
        num_classes: int,
        model_type: str = 'mlp'
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
    
    def build_mlp(self, hidden_units: List[int] = [128, 64]) -> Model:
        """
        Build a Multi-Layer Perceptron model.
        
        Args:
            hidden_units: List of units in hidden layers
        
        Returns:
            Keras Model
        """
        model = Sequential([
            layers.Input(shape=self.input_shape),
            layers.Flatten(),
        ])
        
        # Add hidden layers with dropout
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_cnn(self) -> Model:
        """
        Build a Convolutional Neural Network model.
        
        Returns:
            Keras Model
        """
        model = Sequential([
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam'
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer to use
        """
        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported")
        
        # Select loss function
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC()]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'sparse_top_k_categorical_accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
    
    def get_callbacks(
        self,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ) -> List[callbacks.Callback]:
        """
        Create callbacks for training.
        
        Args:
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
        
        Returns:
            List of Keras callbacks
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(log_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        return callback_list
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> None:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info("Starting model training...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        logger.info("Training completed")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            logger.info(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
        
        Returns:
            Predicted class probabilities or labels
        """
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Model:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded Keras model
        """
        model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


def prepare_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare MNIST dataset.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    logger.info("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def main():
    """Main function demonstrating the neural network pipeline."""
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_mnist_data()
    
    # Split training data for validation
    val_split = int(0.1 * len(X_train))
    X_val = X_train[:val_split]
    y_val = y_train[:val_split]
    X_train = X_train[val_split:]
    y_train = y_train[val_split:]
    
    # Initialize pipeline
    pipeline = NeuralNetworkPipeline(
        input_shape=(28, 28, 1),
        num_classes=10,
        model_type='cnn'
    )
    
    # Build and compile model
    pipeline.model = pipeline.build_cnn()
    pipeline.compile_model(learning_rate=0.001, optimizer_type='adam')
    
    # Print model summary
    pipeline.model.summary()
    
    # Train model
    pipeline.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=128
    )
    
    # Evaluate on test set
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Save model
    pipeline.save_model('mnist_model.h5')
    
    # Make predictions on sample data
    sample_images = X_test[:5]
    predictions = pipeline.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    logger.info(f"Sample predictions: {predicted_classes}")
    logger.info(f"True labels: {y_test[:5]}")


if __name__ == "__main__":
    # Enable mixed precision for faster training on compatible GPUs
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    main()