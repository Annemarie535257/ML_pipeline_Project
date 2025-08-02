import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize the prediction service
        
        Args:
            model_path (str): Path to the trained model
            preprocessor_path (str): Path to the preprocessor configuration
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.classes = ['Healthy', 'Disease']
        self.img_size = (128, 128)
        self.start_time = None
        
        if model_path:
            self.load_model(model_path, preprocessor_path)
    
    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Load the trained model and preprocessor
        
        Args:
            model_path (str): Path to the model file
            preprocessor_path (str): Path to the preprocessor configuration
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {model_path}")
            
            # Load preprocessor if provided
            if preprocessor_path and os.path.exists(preprocessor_path):
                from .preprocessing import ImagePreprocessor
                self.preprocessor = ImagePreprocessor.load_preprocessor(preprocessor_path)
                self.classes = self.preprocessor.classes
                self.img_size = self.preprocessor.img_size
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            
            # Load metadata if available
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.classes = metadata.get('classes', self.classes)
                    self.img_size = tuple(metadata.get('img_size', self.img_size))
                    logger.info(f"Model metadata loaded from {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_data) -> Optional[np.ndarray]:
        """
        Preprocess image for prediction
        
        Args:
            image_data: Image data (file, bytes, or path)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Handle different input types
            if isinstance(image_data, str):
                # File path
                if not os.path.exists(image_data):
                    raise FileNotFoundError(f"Image file not found: {image_data}")
                img = Image.open(image_data)
            elif hasattr(image_data, 'read'):
                # File-like object
                img = Image.open(image_data)
            elif isinstance(image_data, bytes):
                # Bytes data
                img = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("Unsupported image data type")
            
            # Convert to RGB
            img = img.convert('RGB')
            
            # Resize
            img = img.resize(self.img_size)
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_data) -> Dict:
        """
        Make prediction on an image
        
        Args:
            image_data: Image data (file, bytes, or path)
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Start timing
        self.start_time = time.time()
        
        # Preprocess image
        preprocessed_image = self.preprocess_image(image_data)
        if preprocessed_image is None:
            return {
                'error': 'Failed to preprocess image',
                'class': None,
                'confidence': 0.0,
                'probabilities': [],
                'processing_time': 0.0
            }
        
        try:
            # Make prediction
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            predicted_class = self.classes[predicted_class_idx]
            
            # Calculate processing time
            processing_time = time.time() - self.start_time
            
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': predictions[0].tolist(),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            logger.info(f"Prediction completed: {predicted_class} ({confidence:.3f}) in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'class': None,
                'confidence': 0.0,
                'probabilities': [],
                'processing_time': time.time() - self.start_time if self.start_time else 0.0
            }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            image_paths (List[str]): List of image file paths
            
        Returns:
            List[Dict]: List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': image_path,
                    'class': None,
                    'confidence': 0.0,
                    'probabilities': [],
                    'processing_time': 0.0
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        info = {
            'model_path': self.model_path,
            'classes': self.classes,
            'img_size': self.img_size,
            'model_summary': [],
            'total_params': int(self.model.count_params()),  # Convert to regular int
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),  # Convert to regular int
            'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]))  # Convert to regular int
        }
        
        # Get model summary
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        info['model_summary'] = stringlist
        
        return info
    
    def health_check(self) -> Dict:
        """
        Perform health check on the prediction service
        
        Returns:
            dict: Health check results
        """
        health_status = {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - (self.start_time or time.time())
        }
        
        if self.model is None:
            health_status['status'] = 'unhealthy'
            health_status['error'] = 'Model not loaded'
        
        return health_status
    
    def save_prediction_log(self, prediction_result: Dict, log_file: str = 'prediction_log.json'):
        """
        Save prediction result to log file
        
        Args:
            prediction_result (dict): Prediction result to log
            log_file (str): Path to log file
        """
        try:
            # Load existing log
            log_data = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            
            # Add new prediction
            log_data.append(prediction_result)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            logger.info(f"Prediction logged to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving prediction log: {e}")
    
    def get_prediction_stats(self, log_file: str = 'prediction_log.json') -> Dict:
        """
        Get statistics from prediction log
        
        Args:
            log_file (str): Path to prediction log file
            
        Returns:
            dict: Prediction statistics
        """
        try:
            if not os.path.exists(log_file):
                return {'error': 'No prediction log found'}
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            if not log_data:
                return {'error': 'Empty prediction log'}
            
            # Calculate statistics
            total_predictions = len(log_data)
            successful_predictions = len([p for p in log_data if 'error' not in p])
            avg_processing_time = np.mean([p.get('processing_time', 0) for p in log_data if 'error' not in p])
            
            # Class distribution
            class_counts = {}
            for pred in log_data:
                if 'error' not in pred and 'class' in pred:
                    class_name = pred['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Confidence statistics
            confidences = [p.get('confidence', 0) for p in log_data if 'error' not in p]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            stats = {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'failed_predictions': total_predictions - successful_predictions,
                'success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0,
                'avg_processing_time': avg_processing_time,
                'avg_confidence': avg_confidence,
                'class_distribution': class_counts,
                'last_prediction': log_data[-1] if log_data else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating prediction stats: {e}")
            return {'error': str(e)}

def create_prediction_service(model_path: str = None, preprocessor_path: str = None) -> PredictionService:
    """
    Factory function to create a prediction service
    
    Args:
        model_path (str): Path to the model file
        preprocessor_path (str): Path to the preprocessor configuration
        
    Returns:
        PredictionService: Initialized prediction service
    """
    return PredictionService(model_path, preprocessor_path)
