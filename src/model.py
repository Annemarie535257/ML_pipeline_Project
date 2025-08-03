import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseClassifier:
    def __init__(self, num_classes=2, img_size=(128, 128), model_type='custom'):
        """
        Initialize the plant disease classifier
        
        Args:
            num_classes (int): Number of classes (Disease, Healthy)
            img_size (tuple): Input image size
            model_type (str): Type of base model ('mobilenet' or 'resnet')
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        self.history = None
        self.classes = ['Healthy', 'Disease']
        
    def create_model(self, learning_rate=0.001, dropout_rate=0.5):
        """
        Create the classification model
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Input layer
        if self.model_type == 'custom':
            model = models.Sequential()
            model.add(layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(*self.img_size, 3)))
            model.add(layers.Conv2D(32, 3, activation='relu'))
            model.add(layers.MaxPool2D(2, 2))
            model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
            model.add(layers.Conv2D(64, 3, activation='relu'))
            model.add(layers.MaxPool2D(2, 2))
            model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
            model.add(layers.Conv2D(128, 3, activation='relu'))
            model.add(layers.MaxPool2D(2, 2))
            model.add(layers.Conv2D(256, 3, padding='same', activation='relu'))
            model.add(layers.Conv2D(256, 3, activation='relu'))
            model.add(layers.MaxPool2D(2, 2))
            model.add(layers.Conv2D(512, 3, padding='same', activation='relu'))
            model.add(layers.Conv2D(512, 3, activation='relu'))
            model.add(layers.MaxPool2D(2, 2))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Flatten())
            model.add(layers.Dense(1500, activation='relu'))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.model = model
            logger.info("Custom CNN model created (not MobileNet/ResNet)")
            return self.model
        
        # Data augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Input layer
        inputs = layers.Input(shape=(*self.img_size, 3))
        x = data_augmentation(inputs)
        
        # Preprocessing layer
        x = layers.Rescaling(1./255)(x)
        
        # Base model
        if self.model_type == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_type == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model created with {self.model_type} base")
        return self.model
    
    def train(self, train_generator, val_generator, epochs=20, batch_size=32, 
              early_stopping=True, model_checkpoint=True, model_save_path='models/'):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            early_stopping (bool): Whether to use early stopping
            model_checkpoint (bool): Whether to save best model
            model_save_path (str): Path to save model
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.create_model()
        
        # Create callbacks
        callbacks_list = []
        
        if early_stopping:
            early_stopping_cb = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks_list.append(early_stopping_cb)
        
        if model_checkpoint:
            os.makedirs(model_save_path, exist_ok=True)
            checkpoint_cb = callbacks.ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
            callbacks_list.append(checkpoint_cb)
        
        # Add learning rate reduction
        lr_reduction = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
        callbacks_list.append(lr_reduction)
        
        # Train the model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save the final model
        final_model_path = os.path.join(model_save_path, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return self.history
    
    def evaluate(self, test_images, test_labels):
        """
        Evaluate the model on test data
        
        Args:
            test_images: Test images
            test_labels: Test labels (one-hot encoded)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert one-hot encoded labels back to class indices
        test_labels_idx = np.argmax(test_labels, axis=1)
        
        # Make predictions
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels_idx, predicted_classes)
        
        # Classification report
        report = classification_report(
            test_labels_idx, 
            predicted_classes, 
            target_names=self.classes,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels_idx, predicted_classes)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist(),
            'true_classes': test_labels_idx.tolist()
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics
    
    def retrain(self, new_data_dir, epochs=10, learning_rate=0.0001, 
                model_save_path='models/'):
        """
        Retrain the model with new data
        
        Args:
            new_data_dir (str): Directory containing new training data
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for retraining
            model_save_path (str): Path to save retrained model
            
        Returns:
            dict: Training history
        """
        from .preprocessing import ImagePreprocessor
        
        # Load new data
        preprocessor = ImagePreprocessor(data_dir=new_data_dir)
        new_images, new_labels = preprocessor.load_data()
        
        if len(new_images) == 0:
            raise ValueError("No new data found for retraining")
        
        # Convert labels to categorical
        new_labels_cat = tf.keras.utils.to_categorical(new_labels, num_classes=self.num_classes)
        
        # Split new data
        from sklearn.model_selection import train_test_split
        train_images, val_images, train_labels, val_labels = train_test_split(
            new_images, new_labels_cat, test_size=0.2, random_state=42, stratify=new_labels
        )
        
        # Create data generators
        train_generator, val_generator = preprocessor.create_data_generators(
            train_images, train_labels, val_images, val_labels, batch_size=16
        )
        
        # Unfreeze some layers for fine-tuning
        if self.model is not None:
            # Unfreeze the top layers of the base model
            base_model = self.model.layers[3]  # Assuming base model is at index 3
            base_model.trainable = True
            
            # Freeze the bottom layers
            for layer in base_model.layers[:-10]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        
        # Retrain
        logger.info("Starting model retraining...")
        retrain_history = self.train(
            train_generator, val_generator, 
            epochs=epochs, 
            model_save_path=model_save_path
        )
        
        # Save retrained model with timestamp
        retrained_model_path = os.path.join(
            model_save_path, 
            f'retrained_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        )
        self.model.save(retrained_model_path)
        logger.info(f"Retrained model saved to {retrained_model_path}")
        
        return retrain_history
    
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Make prediction
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'class': self.classes[predicted_class],
            'confidence': float(confidence),
            'probabilities': prediction[0].tolist()
        }
    
    def save_model(self, filepath):
        """
        Save the model and metadata
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save as .keras if specified, else .h5
        if filepath.endswith('.keras'):
            self.model.save(filepath)
        else:
            self.model.save(filepath, save_format='h5')
        
        # Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'model_type': self.model_type,
            'classes': self.classes,
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and metadata saved to {filepath}")
    
    @classmethod
    def load_model(cls, model_path, metadata_path=None):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
            metadata_path (str): Path to metadata file (optional)
            
        Returns:
            PlantDiseaseClassifier: Loaded model instance
        """
        # Load .keras or .h5
        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Unsupported model file extension")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.h5', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Default metadata - match the training notebook exactly
            metadata = {
                'num_classes': 2,
                'img_size': (128, 128),  # Match training notebook
                'model_type': 'custom',   # Match training notebook
                'classes': ['Healthy', 'Disease']  # Match training notebook exactly
            }
        
        # Create instance
        classifier = cls(
            num_classes=metadata['num_classes'],
            img_size=tuple(metadata['img_size']),
            model_type=metadata['model_type']
        )
        classifier.model = model
        classifier.classes = metadata['classes']
        
        logger.info(f"Model loaded from {model_path}")
        return classifier
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path (str): Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
