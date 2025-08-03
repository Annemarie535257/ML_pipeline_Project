import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
import zipfile
import tempfile
import shutil
from PIL import Image
import io

# Auto-start API server functionality
def start_api_server():
    """Start the API server in background"""
    try:
        # Get the path to the parent directory (where api_server.py is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        api_server_path = os.path.join(parent_dir, 'api_server.py')
        
        # Check if API server is already running
        try:
            response = requests.get("http://127.0.0.1:5000/health", timeout=2)
            if response.status_code == 200:
                return True  # Server is already running
        except:
            pass
        
        # Start API server in background
        if os.path.exists(api_server_path):
            # Use subprocess to start the API server
            subprocess.Popen([sys.executable, api_server_path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait a bit for the server to start
            time.sleep(3)
            
            # Check if server started successfully
            try:
                response = requests.get("http://127.0.0.1:5000/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
        
        return False
    except Exception as e:
        st.error(f"Failed to start API server: {e}")
        return False

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://127.0.0.1:5000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get('status') == 'healthy' or health_data.get('model_loaded', False)
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
        st.error(f"API health check error: {e}")
        return False

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_prediction_stats():
    """Get prediction statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_image(image_file):
    """Make prediction on uploaded image"""
    try:
        files = {'image': image_file}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            # Log the error response
            print(f"API Error: {response.status_code} - {response.text}")
            return {'error': f'API returned status {response.status_code}'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection to API server failed'}
    except requests.exceptions.Timeout:
        return {'error': 'Request timed out'}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'error': str(e)}

def upload_data_for_retraining(zip_file):
    """Upload data for retraining"""
    try:
        files = {'data': zip_file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def trigger_retraining(data_path):
    """Trigger model retraining"""
    try:
        data = {'data_path': data_path}
        response = requests.post(f"{API_BASE_URL}/retrain", json=data)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_training_status():
    """Get current training status"""
    try:
        response = requests.get(f"{API_BASE_URL}/training/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def refresh_model():
    """Refresh model via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/refresh")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Main app
def main():
    st.title("🌱 Plant Disease Classification System")
    st.markdown("---")
    
    # Auto-start API server with better feedback
    with st.spinner("Starting API server..."):
        if start_api_server():
            st.success("✅ API server started successfully!")
        else:
            st.warning("⚠️ API server is not running. Please ensure `api_server.py` is in the parent directory.")
    
    # Check API health with debugging
    api_healthy = check_api_health()
    
    # Add debugging information
    if not api_healthy:
        st.error("⚠️ API server is not running. Please start the API server first.")
        st.info("To start the API server, run: `python api_server.py`")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"**API URL:** {API_BASE_URL}")
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                st.write(f"**Response Status:** {response.status_code}")
                st.write(f"**Response Content:** {response.text}")
            except Exception as e:
                st.write(f"**Connection Error:** {e}")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "📁 Data Upload", "🔄 Retraining", "📈 Monitoring"]
    )
    
    # Dashboard page
    if page == "🏠 Dashboard":
        show_dashboard()
    
    # Prediction page
    elif page == "🔮 Prediction":
        show_prediction()
    
    # Analytics page
    elif page == "📊 Analytics":
        show_analytics()
    
    # Data Upload page
    elif page == "📁 Data Upload":
        show_data_upload()
    
    # Retraining page
    elif page == "🔄 Retraining":
        show_retraining()
    
    # Monitoring page
    elif page == "📈 Monitoring":
        show_monitoring()

def show_dashboard():
    """Show the main dashboard"""
    st.header("Dashboard")
    
    # Get model info and stats
    model_info = get_model_info()
    stats = get_prediction_stats()
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model Status")
        if model_info and 'error' not in model_info:
            st.success("✅ Model Loaded")
            st.write(f"**Classes:** {', '.join(model_info.get('classes', []))}")
            st.write(f"**Image Size:** {model_info.get('img_size', 'N/A')}")
        else:
            st.error("❌ Model Not Loaded")
    
    with col2:
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API Running")
        else:
            st.error("❌ API Down")
    
    with col3:
        st.subheader("Recent Activity")
        if stats and 'error' not in stats:
            st.write(f"**Total Predictions:** {stats.get('total_predictions', 0)}")
            st.write(f"**Success Rate:** {stats.get('success_rate', 0):.1%}")
            st.write(f"**Avg Response Time:** {stats.get('avg_processing_time', 0):.3f}s")
        else:
            st.write("No data available")
    
    # Add refresh button
    if st.button("🔄 Refresh Model Status"):
        with st.spinner("Refreshing model..."):
            result = refresh_model()
        
        if result and result.get('status') == 'success':
            st.success("✅ Model refreshed successfully!")
            st.rerun()  # Refresh the page to show updated status
        else:
            st.error("❌ Failed to refresh model")
    
    # Quick prediction section
    st.subheader("Quick Prediction")
    uploaded_file = st.file_uploader("Upload an image for quick prediction", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔮 Predict"):
                with st.spinner("Making prediction..."):
                    result = predict_image(uploaded_file)
                
                if result and 'error' not in result:
                    st.success(f"**Prediction:** {result['class']}")
                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                    st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
                    
                    # Confidence bar
                    fig = go.Figure(go.Bar(
                        x=[result['class']],
                        y=[result['confidence']],
                        marker_color='green' if result['class'] == 'Healthy' else 'red'
                    ))
                    fig.update_layout(title="Prediction Confidence", yaxis_title="Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error') if result else 'No response from API'}")

def show_prediction():
    """Show the prediction page"""
    st.header("🔮 Image Prediction")
    
    # Single image prediction
    st.subheader("Single Image Prediction")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔮 Predict"):
                with st.spinner("Making prediction..."):
                    result = predict_image(uploaded_file)
                
                if result and 'error' not in result:
                    st.success(f"**Prediction:** {result['class']}")
                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                    st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
                    
                    # Detailed probabilities
                    if 'probabilities' in result:
                        prob_data = pd.DataFrame({
                            'Class': ['Healthy', 'Disease'],  # Fixed: match model class ordering
                            'Probability': result['probabilities']
                        })
                        
                        fig = px.bar(prob_data, x='Class', y='Probability', 
                                   color='Class', color_discrete_map={'Disease': 'red', 'Healthy': 'green'})
                        fig.update_layout(title="Class Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error') if result else 'No response from API'}")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    uploaded_files = st.file_uploader("Choose multiple images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🔮 Predict All"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                result = predict_image(file)
                if result:
                    result['filename'] = file.name
                    results.append(result)
                else:
                    # Add error result for failed predictions
                    results.append({
                        'filename': file.name,
                        'error': 'No response from API',
                        'class': None,
                        'confidence': 0.0,
                        'processing_time': 0.0
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if results:
                # Display results in a table
                df = pd.DataFrame(results)
                # Show all columns including errors
                display_columns = ['filename', 'class', 'confidence', 'processing_time']
                if 'error' in df.columns:
                    display_columns.append('error')
                st.dataframe(df[display_columns])
                
                # Summary statistics
                st.subheader("Batch Prediction Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Images", len(results))
                
                with col2:
                    success_count = len([r for r in results if 'error' not in r])
                    st.metric("Successful", success_count)
                
                with col3:
                    avg_confidence = np.mean([r.get('confidence', 0) for r in results if 'error' not in r])
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

def show_analytics():
    """Show analytics and visualizations"""
    st.header("📊 Analytics & Visualizations")
    
    stats = get_prediction_stats()
    
    if not stats or 'error' in stats:
        st.warning("No prediction data available for analytics")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Class Distribution", "⏱️ Performance", "📊 Feature Analysis"])
    
    with tab1:
        st.subheader("Prediction Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
        
        with col3:
            st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.3f}s")
        
        with col4:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        
        # Time series of predictions (if available)
        st.subheader("Prediction Timeline")
        # This would require timestamp data in the logs
        
    with tab2:
        st.subheader("Class Distribution")
        
        class_dist = stats.get('class_distribution', {})
        if class_dist:
            fig = px.pie(values=list(class_dist.values()), names=list(class_dist.keys()),
                        title="Prediction Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart
            fig2 = px.bar(x=list(class_dist.keys()), y=list(class_dist.values()),
                         title="Class Distribution (Bar Chart)")
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Metrics")
        
        # Processing time distribution (simulated)
        if stats['total_predictions'] > 0:
            # Simulate processing time distribution
            times = np.random.normal(stats['avg_processing_time'], 0.1, 100)
            times = np.maximum(times, 0)  # Ensure non-negative
            
            fig = px.histogram(x=times, nbins=20, title="Processing Time Distribution")
            fig.update_xaxes(title="Processing Time (seconds)")
            fig.update_yaxes(title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Analysis")
        
        # Simulate feature importance (this would come from model analysis)
        features = ['Color Distribution', 'Texture Features', 'Shape Analysis', 'Edge Detection']
        importance = [0.35, 0.28, 0.22, 0.15]
        
        fig = px.bar(x=features, y=importance, title="Feature Importance Analysis")
        fig.update_yaxes(title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap (simulated)
        st.subheader("Feature Correlation Matrix")
        # This would show correlations between different image features

def show_data_upload():
    """Show data upload page"""
    st.header("📁 Data Upload")
    
    st.info("""
    **Instructions for uploading data:**
    1. Create a ZIP file containing your images
    2. Organize images in folders: `Disease/` and `Healthy/`
    3. Upload the ZIP file below
    4. The system will extract and validate your data
    """)
    
    uploaded_file = st.file_uploader("Upload ZIP file with images", type=['zip'])
    
    if uploaded_file:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        if st.button("📤 Upload Data"):
            with st.spinner("Uploading and validating data..."):
                result = upload_data_for_retraining(uploaded_file)
            
            if result and 'error' not in result:
                st.success("✅ Data uploaded successfully!")
                st.write(f"**Data Path:** {result['data_path']}")
                
                # Store data path in session state for retraining
                st.session_state['uploaded_data_path'] = result['data_path']
                
                st.info("Data is ready for retraining. Go to the Retraining page to start training.")
            else:
                st.error("❌ Upload failed")
                if result:
                    st.write(f"Error: {result.get('error', 'Unknown error')}")

def show_retraining():
    """Show retraining page"""
    st.header("🔄 Model Retraining")
    
    # Check if data is uploaded
    data_path = st.session_state.get('uploaded_data_path')
    
    if not data_path:
        st.warning("⚠️ No data uploaded. Please upload data first in the Data Upload page.")
        return
    
    st.success(f"✅ Data ready for retraining: {data_path}")
    
    # Retraining configuration
    st.subheader("Retraining Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 5, 50, 15)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
    
    with col2:
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001], index=1)
        model_type = st.selectbox("Model Type", ["mobilenet", "resnet"])
    
    if st.button("🚀 Start Retraining"):
        with st.spinner("Starting retraining..."):
            result = trigger_retraining(data_path)
        
        if result and 'error' not in result:
            st.success("✅ Retraining started!")
            st.info("Training is running in the background. Check the status below.")
        else:
            st.error("❌ Failed to start retraining")
    
    # Training status monitoring
    st.subheader("Training Status")
    
    if st.button("🔄 Refresh Status"):
        status = get_training_status()
        
        if status:
            if status['status'] == 'training':
                st.info(f"🔄 Training in progress... {status['progress']}%")
                st.write(f"**Message:** {status['message']}")
                
                # Progress bar
                st.progress(status['progress'] / 100)
                
            elif status['status'] == 'completed':
                st.success("✅ Training completed successfully!")
                st.write(f"**Message:** {status['message']}")
                
            elif status['status'] == 'error':
                st.error("❌ Training failed")
                st.write(f"**Error:** {status['message']}")
                
            else:
                st.info("💤 No training in progress")
        else:
            st.warning("Unable to get training status")

def show_monitoring():
    """Show monitoring page"""
    st.header("📈 System Monitoring")
    
    # Real-time metrics
    st.subheader("Real-time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current stats
    stats = get_prediction_stats()
    
    with col1:
        if stats and 'error' not in stats:
            st.metric("Total Predictions", stats['total_predictions'])
        else:
            st.metric("Total Predictions", 0)
    
    with col2:
        if stats and 'error' not in stats:
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
        else:
            st.metric("Success Rate", "0%")
    
    with col3:
        if stats and 'error' not in stats:
            st.metric("Avg Response Time", f"{stats['avg_processing_time']:.3f}s")
        else:
            st.metric("Avg Response Time", "0s")
    
    with col4:
        # API health
        api_healthy = check_api_health()
        if api_healthy:
            st.metric("API Status", "🟢 Online")
        else:
            st.metric("API Status", "🔴 Offline")
    
    # System uptime and performance
    st.subheader("System Performance")
    
    # Simulate system metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU usage simulation
        cpu_usage = np.random.normal(45, 10)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%")
        
        # Memory usage simulation
        memory_usage = np.random.normal(60, 15)
        st.metric("Memory Usage", f"{memory_usage:.1f}%")
    
    with col2:
        # Model accuracy over time (simulated)
        st.subheader("Model Accuracy Trend")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        accuracy = np.random.normal(0.85, 0.05, 30)
        accuracy = np.clip(accuracy, 0.7, 0.95)
        
        df = pd.DataFrame({'Date': dates, 'Accuracy': accuracy})
        fig = px.line(df, x='Date', y='Accuracy', title="Model Accuracy Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions log
    st.subheader("Recent Predictions")
    
    if stats and 'error' not in stats and 'last_prediction' in stats:
        last_pred = stats['last_prediction']
        if last_pred:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Last Prediction:** {last_pred.get('class', 'N/A')}")
            
            with col2:
                st.write(f"**Confidence:** {last_pred.get('confidence', 0):.1%}")
            
            with col3:
                st.write(f"**Time:** {last_pred.get('timestamp', 'N/A')}")
    else:
        st.info("No recent predictions available")

if __name__ == "__main__":
    main()
