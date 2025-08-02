import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, data_dir="data/Dataset", img_size=(128, 128)):
        """
        Initialize the image preprocessor
        
        Args:
            data_dir (str): Path to the dataset directory
            img_size (tuple): Target image size (width, height)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['Healthy', 'Disease']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_data(self):
        """
        Load images and labels from the dataset directory
        
        Returns:
            tuple: (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        logger.info("Loading images from dataset...")
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory {class_dir} does not exist")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        logger.warning(f"Error loading {img_path}: {e}")
                        continue
        
        logger.info(f"Loaded {len(images)} images with {len(set(labels))} classes")
        return np.array(images), np.array(labels)
    
    def create_data_generators(self, train_images, train_labels, val_images, val_labels, 
                              batch_size=32, augmentation=True):
        """
        Create data generators for training and validation
        
        Args:
            train_images, train_labels: Training data
            val_images, val_labels: Validation data
            batch_size (int): Batch size for training
            augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        if augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
            
        val_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=len(self.classes))
        val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes=len(self.classes))
        
        train_generator = train_datagen.flow(
            train_images, train_labels_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            val_images, val_labels_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def prepare_single_image(self, image_path):
        """
        Prepare a single image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def prepare_bulk_data(self, data_dir):
        """
        Prepare bulk data for retraining
        
        Args:
            data_dir (str): Directory containing new data
            
        Returns:
            tuple: (images, labels) for new data
        """
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        logger.warning(f"Error loading {img_path}: {e}")
                        continue
        
        return np.array(images), np.array(labels)
    
    def save_preprocessor(self, filepath):
        """
        Save the preprocessor configuration
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        config = {
            'img_size': self.img_size,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """
        Load a saved preprocessor configuration
        
        Args:
            filepath (str): Path to the saved preprocessor
            
        Returns:
            ImagePreprocessor: Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.img_size = config['img_size']
        preprocessor.classes = config['classes']
        preprocessor.class_to_idx = config['class_to_idx']
        
        return preprocessor

def split_data(images, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        images, labels: Input data
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of remaining data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    # First split: separate test set
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation set from training set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, 
        test_size=val_size, random_state=random_state, stratify=train_val_labels
    )
    
    logger.info(f"Data split - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
