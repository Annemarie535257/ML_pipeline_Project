from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
import threading
import time
from datetime import datetime
import tempfile
import shutil
from werkzeug.utils import secure_filename
import zipfile
import tensorflow as tf

# Import our modules
from src.prediction import PredictionService
from src.model import PlantDiseaseClassifier
from src.preprocessing import ImagePreprocessor, split_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
prediction_service = None
model_classifier = None
is_training = False
training_status = {'status': 'idle', 'progress': 0, 'message': ''}

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/best_model.h5'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('temp_data', exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the prediction service with the best available model"""
    global prediction_service
    
    # Find the best model
    if os.path.exists(MODEL_PATH):
        model_to_load = MODEL_PATH
    else:
        # Look for any .h5 file in models directory
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        if model_files:
            model_to_load = os.path.join('models', model_files[0])
        else:
            logger.warning("No model found. Please train a model first.")
            return
    
    try:
        prediction_service = PredictionService(model_to_load, PREPROCESSOR_PATH)
        logger.info(f"Model initialized from {model_to_load}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if prediction_service:
        health_status = prediction_service.health_check()
    else:
        health_status = {
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'No model loaded'
        }
    
    return jsonify(health_status)

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if prediction_service:
        return jsonify(prediction_service.get_model_info())
    else:
        return jsonify({'error': 'No model loaded'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    if not prediction_service:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(temp_path)
        
        # Make prediction
        result = prediction_service.predict(temp_path)
        
        # Log prediction
        prediction_service.save_prediction_log(result)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict endpoint for multiple images"""
    if not prediction_service:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    temp_paths = []
    try:
        # Save files temporarily
        for file in files:
            if file.filename == '':
                continue
            if not allowed_file(file.filename):
                continue
            
            temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(temp_path)
            temp_paths.append(temp_path)
        
        if not temp_paths:
            return jsonify({'error': 'No valid files provided'}), 400
        
        # Make predictions
        results = prediction_service.predict_batch(temp_paths)
        
        # Log predictions
        for result in results:
            prediction_service.save_prediction_log(result)
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics"""
    if prediction_service:
        stats = prediction_service.get_prediction_stats()
        return jsonify(stats)
    else:
        return jsonify({'error': 'No model loaded'}), 404

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload new data for retraining"""
    if 'data' not in request.files:
        return jsonify({'error': 'No data file provided'}), 400
    
    file = request.files['data']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a ZIP file'}), 400
    
    try:
        # Save uploaded file
        zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(zip_path)
        
        # Extract to temp directory
        temp_dir = os.path.join('temp_data', f'new_data_{int(time.time())}')
        os.makedirs(temp_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Validate structure
        expected_classes = ['Disease', 'Healthy']
        for class_name in expected_classes:
            class_dir = os.path.join(temp_dir, class_name)
            if not os.path.exists(class_dir):
                return jsonify({'error': f'Missing {class_name} directory in ZIP'}), 400
        
        # Clean up zip file
        os.remove(zip_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Data uploaded successfully',
            'data_path': temp_dir
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new data"""
    global is_training, training_status, prediction_service
    
    if is_training:
        return jsonify({'error': 'Training already in progress'}), 409
    
    # Get data path from request
    data = request.get_json()
    data_path = data.get('data_path') if data else None
    
    if not data_path or not os.path.exists(data_path):
        return jsonify({'error': 'Invalid data path'}), 400
    
    def train_model():
        global is_training, training_status, prediction_service
        
        try:
            is_training = True
            training_status = {'status': 'training', 'progress': 0, 'message': 'Initializing...'}
            
            # Initialize preprocessor
            preprocessor = ImagePreprocessor(data_dir=data_path)
            training_status['progress'] = 10
            training_status['message'] = 'Loading data...'
            
            # Load data
            images, labels = preprocessor.load_data()
            if len(images) == 0:
                raise ValueError("No valid images found in data")
            
            training_status['progress'] = 30
            training_status['message'] = 'Splitting data...'
            
            # Split data
            train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(
                images, labels, test_size=0.2, val_size=0.2
            )
            
            training_status['progress'] = 50
            training_status['message'] = 'Creating data generators...'
            
            # Create data generators
            train_generator, val_generator = preprocessor.create_data_generators(
                train_images, train_labels, val_images, val_labels, batch_size=16
            )
            
            training_status['progress'] = 60
            training_status['message'] = 'Training model...'
            
            # Initialize and train model
            model_classifier = PlantDiseaseClassifier()
            model_classifier.create_model()
            
            # Train model
            history = model_classifier.train(
                train_generator, val_generator, 
                epochs=15, 
                model_save_path='models/'
            )
            
            training_status['progress'] = 90
            training_status['message'] = 'Evaluating model...'
            
            # Evaluate model
            test_labels_cat = tf.keras.utils.to_categorical(test_labels, num_classes=2)
            metrics = model_classifier.evaluate(test_images, test_labels_cat)
            
            # Save preprocessor
            preprocessor.save_preprocessor(PREPROCESSOR_PATH)
            
            training_status['progress'] = 100
            training_status['message'] = 'Training completed successfully'
            
            # Reload prediction service with new model
            initialize_model()
            
            # Clean up temp data
            shutil.rmtree(data_path, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            training_status = {'status': 'error', 'progress': 0, 'message': str(e)}
        finally:
            is_training = False
    
    # Start training in background thread
    thread = threading.Thread(target=train_model)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Training started in background'
    })

@app.route('/training/status', methods=['GET'])
def training_status_endpoint():
    """Get training status"""
    return jsonify(training_status)

@app.route('/refresh', methods=['POST'])
def refresh_model():
    """Refresh/restart the model loading"""
    global prediction_service
    try:
        initialize_model()
        if prediction_service:
            return jsonify({'status': 'success', 'message': 'Model refreshed successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'No model found to load'}), 404
    except Exception as e:
        logger.error(f"Error refreshing model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/model', methods=['GET'])
def download_model():
    """Download the current model"""
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'No model available'}), 404
    
    return send_file(MODEL_PATH, as_attachment=True, download_name='plant_disease_model.h5')

@app.route('/download/logs', methods=['GET'])
def download_logs():
    """Download prediction logs"""
    log_file = 'prediction_log.json'
    if not os.path.exists(log_file):
        return jsonify({'error': 'No logs available'}), 404
    
    return send_file(log_file, as_attachment=True, download_name='prediction_logs.json')

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 